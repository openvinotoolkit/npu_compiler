//
// Copyright (C) 2025 Intel Corporation.
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
 *             ├─► IE.YuvToRgb ──► IE.Transpose ──► [IE.Multiply] ──► RGB Output
 *   UV Input ─┘    (NV12→RGB)     (NHWC→NCHW)      (scale factor)
 *
 * Note: FakeQuantize and Convert operations are transparently handled.
 *       IE.Multiply is conditionally added when scale factor != 1.0.
 */

namespace vpux::IE {
#define GEN_PASS_DECL_FUSECOLORCONVERSION
#define GEN_PASS_DEF_FUSECOLORCONVERSION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseColorConversionPass
//

class FuseColorConversionPass final : public IE::impl::FuseColorConversionBase<FuseColorConversionPass> {
public:
    explicit FuseColorConversionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

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

    if (!minAttr || !maxAttr) {
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

    auto lowContent = outputLowConst.getContent();
    auto highContent = outputHighConst.getContent();
    auto lowValues = to_small_vector(lowContent.getValues<float>());
    auto highValues = to_small_vector(highContent.getValues<float>());

    if (lowValues.empty() || highValues.empty()) {
        return false;
    }

    float minVal = lowValues[0];
    float maxVal = highValues[0];
    return isValidColorOutputRange(minVal, maxVal);
}

// Skip Convert operation if present
mlir::Operation* skipConvertIfPresent(mlir::Operation* op) {
    if (auto convertOp = mlir::dyn_cast_or_null<IE::ConvertOp>(op)) {
        return convertOp.getInput().getDefiningOp();
    }
    return op;
}

// Skip FakeQuantize operation if present
mlir::Operation* skipFakeQuantizeIfPresent(mlir::Operation* op) {
    if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op)) {
        return fqOp.getInput().getDefiningOp();
    }
    return op;
}

// Skip both Convert and FakeQuantize operations if present
mlir::Operation* skipConvertAndFakeQuantizeIfPresent(mlir::Operation* op) {
    op = skipConvertIfPresent(op);
    op = skipFakeQuantizeIfPresent(op);
    return op;
}

//
// Pattern matching for color conversion
//

bool matchYuvToRgbPattern(IE::AddOp addOp, mlir::Value& yInput, mlir::Value& uvInput, mlir::Value& convWeights,
                          mlir::Value& convBias) {
    auto convOp = addOp.getInput1().getDefiningOp<IE::ConvolutionOp>();
    if (!convOp) {
        return false;
    }

    convWeights = convOp.getFilter();
    convBias = addOp.getInput2();

    auto convInput = skipFakeQuantizeIfPresent(convOp.getInput().getDefiningOp());

    auto concatOp = mlir::dyn_cast_or_null<IE::ConcatOp>(convInput);
    if (!concatOp || concatOp.getInputs().size() != 2) {
        return false;
    }

    auto yPath = skipConvertAndFakeQuantizeIfPresent(concatOp.getInputs()[0].getDefiningOp());
    auto affineReshapeOp = mlir::dyn_cast_or_null<IE::AffineReshapeOp>(yPath);
    if (!affineReshapeOp) {
        return false;
    }

    auto yConvertInput = skipConvertIfPresent(affineReshapeOp.getInput().getDefiningOp());
    if (!yConvertInput) {
        yInput = affineReshapeOp.getInput();
    } else {
        yInput = yConvertInput->getOperand(0);
    }

    auto uvPath = skipFakeQuantizeIfPresent(concatOp.getInputs()[1].getDefiningOp());
    auto interpolateOp = mlir::dyn_cast_or_null<IE::InterpolateOp>(uvPath);
    if (!interpolateOp) {
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
    if (!transposeOp) {
        return false;
    }

    auto uvConvertInput = skipConvertIfPresent(transposeOp.getInput().getDefiningOp());
    if (!uvConvertInput) {
        uvInput = transposeOp.getInput();
    } else {
        uvInput = uvConvertInput->getOperand(0);
    }
    return true;
}

//
// safeRunOnFunc
//

void FuseColorConversionPass::safeRunOnFunc() {
    auto func = getOperation();

    auto processOperation = [&](mlir::Operation* op) {
        mlir::Value inputValue;

        // Check range for Clamp or FakeQuantize operations
        if (auto clampOp = mlir::dyn_cast<IE::ClampOp>(op)) {
            if (!isValidClamp(clampOp)) {
                return;
            }
            inputValue = clampOp.getInput();
        } else if (auto fqOp = mlir::dyn_cast<IE::FakeQuantizeOp>(op)) {
            if (!isValidFQ(fqOp)) {
                return;
            }
            inputValue = fqOp.getInput();
        } else {
            return;
        }

        auto addOp = inputValue.getDefiningOp<IE::AddOp>();
        if (!addOp) {
            return;
        }

        mlir::Value yInput, uvInput, convWeights, convBias;

        if (!matchYuvToRgbPattern(addOp, yInput, uvInput, convWeights, convBias)) {
            return;
        }

        _log.trace("Color conversion pattern matched for operation {0} at {1}", op->getName(), op->getLoc());
        auto builder = mlir::OpBuilder(addOp);

        // Determine if output is RGB or BGR based on convolution weights
        IE::ColorFmt outputColorFmt = IE::ColorFmt::RGB;  // Default to RGB

        if (auto weightsConst = convWeights.getDefiningOp<Const::DeclareOp>()) {
            auto weightsContent = weightsConst.getContent();
            auto weightsValues = to_small_vector(weightsContent.getValues<float>());

            if (weightsValues.size() >= 9) {  // At least 3x3 matrix

                float firstChannelVComponent = weightsValues[2];
                float thirdChannelVComponent = weightsValues[8];

                if (std::abs(thirdChannelVComponent) > std::abs(firstChannelVComponent)) {
                    outputColorFmt = IE::ColorFmt::BGR;
                    _log.trace("Detected BGR output format based on convolution weights");
                } else {
                    _log.trace("Detected RGB output format based on convolution weights");
                }
            }
        }

        // Create YuvToRgb operation with detected color format
        auto yuvToRgbOp = builder.create<IE::YuvToRgbOp>(appendLoc(addOp->getLoc(), "_nv12_to_rgb"), yInput, uvInput,
                                                         nullptr,             // input3 (optional V channel)
                                                         IE::ColorFmt::NV12,  // inFmt
                                                         outputColorFmt       // outFmt (RGB or BGR)
        );

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

            auto transposeOp = builder.create<IE::TransposeOp>(appendLoc(addOp->getLoc(), "_transpose"), finalOutput,
                                                               nullptr, orderAttr);

            finalOutput = transposeOp.getOutput();
        }

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
            if (outputColorFmt == IE::ColorFmt::RGB) {
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

        // Add Multiply operation with scaleFactor constant if scaleFactor != 1.0
        if (!isDoubleEqual(scaleFactor, 1.0)) {
            // Create constant for scaleFactor
            const auto scaleFactorType = mlir::RankedTensorType::get({1}, builder.getF32Type());
            const auto scaleFactorConst = Const::createFloatConst(builder, appendLoc(addOp->getLoc(), "_scale_factor"),
                                                                  scaleFactorType, static_cast<float>(scaleFactor));

            // Create Multiply operation
            auto multiplyOp =
                    builder.create<IE::MultiplyOp>(appendLoc(addOp->getLoc(), "_scale_multiply"), finalOutput.getType(),
                                                   finalOutput, scaleFactorConst, IE::AutoBroadcastType::NUMPY,
                                                   /*post_op=*/nullptr,
                                                   /*clamp=*/nullptr,
                                                   /*output_channels=*/nullptr,
                                                   /*input_channels=*/nullptr);
            finalOutput = multiplyOp.getOutput();
        }
        op->replaceAllUsesWith(finalOutput.getDefiningOp());
    };

    func->walk([&](IE::ClampOp clampOp) {
        processOperation(clampOp.getOperation());
    });

    func->walk([&](IE::FakeQuantizeOp fqOp) {
        processOperation(fqOp.getOperation());
    });
}

}  // namespace

//
// createFuseColorConversionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseColorConversionPass(Logger log) {
    return std::make_unique<FuseColorConversionPass>(log);
}
