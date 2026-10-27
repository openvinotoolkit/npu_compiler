//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <string>

/*
 * YUV to RGB Color Conversion Pattern Fusion
 * ==========================================
 *
 * This pass matches and fuses the following pattern:
 *
 * Pattern to match:
 *   Y Input ──► AffineReshape ──┐
 *                               ├─► Concat ──► Convolution ──► Add ──► Clamp/FQ──► RGB
 *        UV Input ─► Transpose ──► Interpolate ──┘    (YUV→RGB)   (bias)
 *               (NHWC→NCHW)    (2x upscale)
 *
 * After fusion:
 *   Y Input ──┐
 *             ├─► IE.YuvToRgb ──► IE.Transpose ──► [IE.Multiply] ──► [Clamp/FQ] ──► RGB Output
 *   UV Input ─┘    (NV12→RGB)     (NHWC→NCHW)      (scale factor)
 *
 * Note: FakeQuantize and Convert operations are transparently handled.
 *       IE.Multiply is conditionally added when scale factor != 1.0 and enableYuvToRgbShaveScale is false.
 *       Clamp/FQ is also conditionally kept when scale > 1 to avoid out-of-range values after scaling and keep IR
 *       consistent.
 */

namespace vpux::IE {
#define GEN_PASS_DECL_FUSECOLORCONVERSION
#define GEN_PASS_DEF_FUSECOLORCONVERSION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// Helper functions
//

bool isValidColorOutputRange(float minVal, float maxVal) {
    // Check for [0, 255] range
    if (isFloatEqual(minVal, 0.0f) && isFloatEqual(maxVal, 255.0f)) {
        return true;
    }
    // Check for [0, 1] range
    if (isFloatEqual(minVal, 0.0f) && isFloatEqual(maxVal, 1.0f)) {
        return true;
    }
    return false;
}

bool isValidClamp(IE::ClampOp clampOp) {
    auto minAttr = clampOp.getMinAttr();
    auto maxAttr = clampOp.getMaxAttr();

    if (minAttr == nullptr || maxAttr == nullptr) {
        return false;
    }

    float minVal = static_cast<float>(minAttr.getValueAsDouble());
    float maxVal = static_cast<float>(maxAttr.getValueAsDouble());
    return isValidColorOutputRange(minVal, maxVal);
}

bool isValidFQ(IE::FakeQuantizeOp fqOp) {
    auto outputLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outputHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

    if (!outputLowConst || !outputHighConst) {
        return false;
    }

    auto minVal = Const::getSplatValue<float>(outputLowConst);
    auto maxVal = Const::getSplatValue<float>(outputHighConst);

    if (mlir::failed(minVal) || mlir::failed(maxVal)) {
        return false;
    }

    return isValidColorOutputRange(minVal.value(), maxVal.value());
}

// Skip Convert operation if present
mlir::Operation* skipConvertIfPresent(mlir::Operation* op) {
    if (auto convertOp = mlir::dyn_cast_or_null<IE::ConvertOp>(op)) {
        return convertOp.getInput().getDefiningOp();
    }
    return op;
}

// Skip FakeQuantize or DynamicDequantize operation if present
mlir::Operation* skipDequantOpIfPresent(mlir::Operation* op) {
    if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op)) {
        return fqOp.getInput().getDefiningOp();
    }
    if (auto dynDeqOp = mlir::dyn_cast_or_null<IE::DynamicDequantizeOp>(op)) {
        return dynDeqOp.getInput().getDefiningOp();
    }
    return op;
}

// Skip both Convert and FakeQuantize operations if present
mlir::Operation* skipConvertAndDequantOpIfPresent(mlir::Operation* op) {
    op = skipConvertIfPresent(op);
    op = skipDequantOpIfPresent(op);
    return op;
}

// Skip AffineReshape in static case and DynamicReshape in the dynamic shapes case
std::optional<mlir::Value> skipReshape(mlir::Operation* yPath) {
    if (auto affineReshapeOp = mlir::dyn_cast_or_null<IE::AffineReshapeOp>(yPath)) {
        auto yConvertInput = skipConvertIfPresent(affineReshapeOp.getInput().getDefiningOp());
        if (yConvertInput == nullptr) {
            return affineReshapeOp.getInput();
        } else {
            return yConvertInput->getOperand(0);
        }
    } else if (auto dynamicReshapeOp = mlir::dyn_cast_or_null<IE::DynamicReshapeOp>(yPath)) {
        auto yConvertInput = skipConvertIfPresent(dynamicReshapeOp.getInput().getDefiningOp());
        if (yConvertInput == nullptr) {
            return dynamicReshapeOp.getInput();
        } else {
            return yConvertInput->getOperand(0);
        }
    }

    return std::nullopt;
}

// Skip element-wise arithmetic ops on constants (dequantization chains like
// Const::DeclareOp [CastElemType] → IE.Subtract → IE.Multiply) to find the underlying Const::DeclareOp.
Const::DeclareOp findUnderlyingConstant(mlir::Operation* op) {
    constexpr int maxDepth = 5;
    for (int i = 0; i < maxDepth && op != nullptr; ++i) {
        if (auto constOp = mlir::dyn_cast<Const::DeclareOp>(op)) {
            return constOp;
        }
        if (mlir::isa<IE::MultiplyOp, IE::SubtractOp, IE::AddOp, IE::ConvertOp>(op)) {
            op = op->getOperand(0).getDefiningOp();
            continue;
        }
        break;
    }
    return nullptr;
}

// Detect RGB or BGR format based on convolution weights
std::optional<IE::ColorFmt> detectOutputColorFormat(mlir::Value convWeights) {
    mlir::Value weightsValue = convWeights;
    if (auto dynDeqOp = convWeights.getDefiningOp<IE::DynamicDequantizeOp>()) {
        weightsValue = dynDeqOp.getInput();
    }

    auto weightsConst = findUnderlyingConstant(weightsValue.getDefiningOp());
    if (!weightsConst) {
        return std::nullopt;
    }

    auto weightsContent = weightsConst.getContent();

    // Weights are 9 values arranged as: [Ch0_Y, Ch0_U, Ch0_V, Ch1_Y, Ch1_U, Ch1_V, Ch2_Y, Ch2_U, Ch2_V]
    // RGB has Red first (large V coef at index 2), BGR has Blue first (small V coef at index 2)
    const auto colorFormat = weightsContent.read([](auto weightsValues) -> std::optional<IE::ColorFmt> {
        if (weightsValues.size() != 9) {
            return std::nullopt;
        }

        float firstChannelVComponent = checked_cast<float>(weightsValues[2]);
        float thirdChannelVComponent = checked_cast<float>(weightsValues[8]);
        if (std::abs(thirdChannelVComponent) > std::abs(firstChannelVComponent)) {
            return IE::ColorFmt::BGR;
        }
        return IE::ColorFmt::RGB;
    });

    return colorFormat;
}

//
// Pattern matching for color conversion
//

bool matchYuvToRgbPattern(IE::AddOp addOp, mlir::Value& yInput, mlir::Value& uvInput, mlir::Value& convWeights,
                          mlir::Value& convBias) {
    auto convOp = addOp.getInput1().getDefiningOp<IE::ConvolutionOp>();
    if (convOp == nullptr) {
        return false;
    }

    convWeights = convOp.getFilter();
    convBias = addOp.getInput2();

    auto convInput = skipDequantOpIfPresent(convOp.getInput().getDefiningOp());

    auto concatOp = mlir::dyn_cast_or_null<IE::ConcatOp>(convInput);
    if (concatOp == nullptr || concatOp.getInputs().size() != 2) {
        return false;
    }

    auto yPath = skipConvertAndDequantOpIfPresent(concatOp.getInputs()[0].getDefiningOp());

    auto optionalYInput = skipReshape(yPath);
    if (!optionalYInput.has_value()) {
        return false;
    }
    yInput = optionalYInput.value();

    auto uvPath = skipDequantOpIfPresent(concatOp.getInputs()[1].getDefiningOp());
    auto interpolateOp = mlir::dyn_cast_or_null<IE::InterpolateOp>(uvPath);
    if (interpolateOp == nullptr) {
        return false;
    }

    // Check if interpolation has 2x scale
    if (auto scalesAttr = interpolateOp.getScalesAttr()) {
        auto scales = parseFPArrayAttr<double>(scalesAttr.value());
        if (scales.size() >= 2) {
            auto scaleH = scales[scales.size() - 2];
            auto scaleW = scales[scales.size() - 1];
            if (!isDoubleEqual(scaleH, 2.0) || !isDoubleEqual(scaleW, 2.0)) {
                return false;
            }
        }
    }

    auto transposeOp = interpolateOp.getInput().getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr) {
        return false;
    }

    auto uvConvertInput = skipConvertIfPresent(transposeOp.getInput().getDefiningOp());
    if (uvConvertInput == nullptr) {
        uvInput = transposeOp.getInput();
    } else {
        uvInput = uvConvertInput->getOperand(0);
    }
    return true;
}

mlir::LogicalResult fuseColorConversionPattern(mlir::Operation* op, mlir::PatternRewriter& rewriter,
                                               bool enableYuvToRgbShaveScale, Logger log) {
    mlir::Value inputValue;

    // Check range for Clamp or FakeQuantize operations
    if (auto clampOp = mlir::dyn_cast<IE::ClampOp>(op)) {
        if (!isValidClamp(clampOp)) {
            return mlir::failure();
        }
        inputValue = clampOp.getInput();
    } else if (auto fqOp = mlir::dyn_cast<IE::FakeQuantizeOp>(op)) {
        if (!isValidFQ(fqOp)) {
            return mlir::failure();
        }
        inputValue = fqOp.getInput();
    } else {
        return mlir::failure();
    }

    auto addOp = inputValue.getDefiningOp<IE::AddOp>();
    if (addOp == nullptr) {
        return mlir::failure();
    }

    mlir::Value yInput, uvInput, convWeights, convBias;

    if (!matchYuvToRgbPattern(addOp, yInput, uvInput, convWeights, convBias)) {
        return mlir::failure();
    }

    auto outputColorFmt = detectOutputColorFormat(convWeights);
    if (!outputColorFmt.has_value()) {
        return mlir::failure();
    }

    log.trace("Color conversion pattern matched for operation {0} at {1}", op->getName(), op->getLoc());
    auto builder = mlir::OpBuilder(addOp);

    // Calculate scale factor from bias values
    double scaleFactor = 1.0;  // Default value

    if (auto biasConst = convBias.getDefiningOp<Const::DeclareOp>()) {
        auto biasContent = biasConst.getContent();
        auto biasValues = to_small_vector(biasContent.getValues<float>());

        // Y range [16-235] and UV centered at 128
        // R = 1.164(-16) + 1.596(-128) = -222.912
        // G = 1.164(-16) - 0.391(-128) - 0.813(-128) = 135.488
        // B = 1.164(-16) + 2.018(-128) = -276.928
        std::vector<float> standardBiasValues;
        if (outputColorFmt.value() == IE::ColorFmt::RGB) {
            standardBiasValues = {-222.912f, 135.488f, -276.928f};
        } else {
            standardBiasValues = {-276.928f, 135.488f, -222.912f};
        }

        if (biasValues.size() == standardBiasValues.size()) {
            double totalRatio = 0.0;
            for (size_t i = 0; i < biasValues.size(); ++i) {
                totalRatio += biasValues[i] / standardBiasValues[i];
            }
            scaleFactor = totalRatio / biasValues.size();
        }
    }

    // Use SHAVE-side scale only when enabled; otherwise keep YuvToRgb scale neutral and
    // preserve behavior via the explicit IE.Multiply below.
    const float yuvToRgbScale = enableYuvToRgbShaveScale ? static_cast<float>(scaleFactor) : 1.0f;

    // Create YuvToRgb operation with detected color format
    auto yuvToRgbOp = IE::YuvToRgbOp::create(builder, appendLoc(addOp->getLoc(), "nv12_to_rgb"), yInput, uvInput,
                                             nullptr,                 // input3 (optional V channel)
                                             IE::ColorFmt::NV12,      // inFmt
                                             outputColorFmt.value(),  // outFmt (RGB or BGR)
                                             builder.getF32FloatAttr(yuvToRgbScale));

    auto outputType = mlir::cast<vpux::NDTypeInterface>(addOp.getType());
    auto yuvOutputType = mlir::cast<vpux::NDTypeInterface>(yuvToRgbOp.getType());

    mlir::Value finalOutput = yuvToRgbOp.getOutput();

    const auto outputShape = outputType.getShape();
    const auto yuvOutputShape = yuvOutputType.getShape();

    if (outputShape != yuvOutputShape) {
        // Check if we need transpose from NHWC to NCHW
        // YuvToRgb output: [N, H, W, C], Expected output: [N, C, H, W]
        SmallVector<uint32_t> permuteOrder = {0, 3, 1, 2};

        const auto orderAttr =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permuteOrder, builder.getContext()));

        auto transposeOp = IE::TransposeOp::create(builder, appendLoc(addOp->getLoc(), "transpose"), finalOutput,
                                                   nullptr, orderAttr);

        finalOutput = transposeOp.getOutput();
    }

    if (!isDoubleEqual(scaleFactor, 1.0)) {
        // Default path (enableYuvToRgbShaveScale=false): keep YuvToRgb scale at 1.0 and
        // apply scaling as a separate Multiply.
        if (!enableYuvToRgbShaveScale) {
            // Create constant for scaleFactor
            const auto scaleFactorType = mlir::RankedTensorType::get({1}, builder.getF32Type());
            const auto scaleFactorConst = Const::createFloatConst(builder, appendLoc(addOp->getLoc(), "scale_factor"),
                                                                  scaleFactorType, static_cast<float>(scaleFactor));

            // Create Multiply operation
            auto multiplyOp =
                    IE::MultiplyOp::create(builder, appendLoc(addOp->getLoc(), "scale_multiply"), finalOutput.getType(),
                                           finalOutput, scaleFactorConst, IE::AutoBroadcastType::NUMPY,
                                           /*post_op=*/nullptr,
                                           /*clamp=*/nullptr,
                                           /*output_channels=*/nullptr,
                                           /*input_channels=*/nullptr);
            finalOutput = multiplyOp.getOutput();
        }

        // Keep the original tail semantics only for scale > 1.0, where values can exceed
        // the original output domain.
        if (scaleFactor > 1.0) {
            if (auto clampOp = mlir::dyn_cast<IE::ClampOp>(op)) {
                auto restoredClampOp = IE::ClampOp::create(builder, appendLoc(addOp->getLoc(), "scale_clamp"),
                                                           finalOutput, clampOp.getMinAttr(), clampOp.getMaxAttr());
                finalOutput = restoredClampOp.getOutput();
            } else if (auto fqOp = mlir::dyn_cast<IE::FakeQuantizeOp>(op)) {
                auto restoredFqOp = IE::FakeQuantizeOp::create(
                        builder, appendLoc(addOp->getLoc(), "scale_fq"), finalOutput.getType(), finalOutput,
                        fqOp.getInputLow(), fqOp.getInputHigh(), fqOp.getOutputLow(), fqOp.getOutputHigh(),
                        fqOp.getLevelsAttr(), fqOp.getLowFpTypeAttr(), fqOp.getAutoBroadcastAttr());
                finalOutput = restoredFqOp.getOutput();
            }
        }
    }

    rewriter.replaceOp(op, finalOutput);
    return mlir::success();
}

//
// FuseColorConversionClampPattern
//

class FuseColorConversionClampPattern final : public mlir::OpRewritePattern<IE::ClampOp> {
public:
    FuseColorConversionClampPattern(mlir::MLIRContext* ctx, bool enableYuvToRgbShaveScale, Logger log)
            : mlir::OpRewritePattern<IE::ClampOp>(ctx), _enableYuvToRgbShaveScale(enableYuvToRgbShaveScale), _log(log) {
        setDebugName("FuseColorConversionClampPattern");
    }

    mlir::LogicalResult matchAndRewrite(IE::ClampOp clampOp, mlir::PatternRewriter& rewriter) const final {
        return fuseColorConversionPattern(clampOp.getOperation(), rewriter, _enableYuvToRgbShaveScale, _log);
    }

private:
    bool _enableYuvToRgbShaveScale;
    Logger _log;
};

//
// FuseColorConversionFakeQuantizePattern
//

class FuseColorConversionFakeQuantizePattern final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    FuseColorConversionFakeQuantizePattern(mlir::MLIRContext* ctx, bool enableYuvToRgbShaveScale, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx),
              _enableYuvToRgbShaveScale(enableYuvToRgbShaveScale),
              _log(log) {
        setDebugName("FuseColorConversionFakeQuantizePattern");
    }

    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const final {
        return fuseColorConversionPattern(fqOp.getOperation(), rewriter, _enableYuvToRgbShaveScale, _log);
    }

private:
    bool _enableYuvToRgbShaveScale;
    Logger _log;
};

//
// FuseColorConversionPass
//

class FuseColorConversionPass final : public IE::impl::FuseColorConversionBase<FuseColorConversionPass> {
public:
    explicit FuseColorConversionPass(bool enableYuvToRgbShaveScale, Logger log) {
        this->enableYuvToRgbShaveScale = enableYuvToRgbShaveScale;
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseColorConversionPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseColorConversionClampPattern>(&ctx, enableYuvToRgbShaveScale, _log);
    patterns.add<FuseColorConversionFakeQuantizePattern>(&ctx, enableYuvToRgbShaveScale, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createFuseColorConversionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseColorConversionPass(const bool enableYuvToRgbShaveScale, Logger log) {
    return std::make_unique<FuseColorConversionPass>(enableYuvToRgbShaveScale, log);
}
