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
 *                               ├─► Concat ──► Convolution ──► Add ──► RGB
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
    func->walk([&](IE::AddOp addOp) {
        mlir::Value yInput, uvInput, convWeights, convBias;

        if (!matchYuvToRgbPattern(addOp, yInput, uvInput, convWeights, convBias)) {
            return;
        }

        _log.trace("Color conversion pattern matched for operation {0} at {1}", addOp->getName(), addOp->getLoc());
        auto builder = mlir::OpBuilder(addOp);

        // Create YuvToRgb operation
        auto yuvToRgbOp = builder.create<IE::YuvToRgbOp>(appendLoc(addOp->getLoc(), "_nv12_to_rgb"), yInput, uvInput,
                                                         nullptr,             // input3 (optional V channel)
                                                         IE::ColorFmt::NV12,  // inFmt
                                                         IE::ColorFmt::RGB    // outFmt
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

            // Standard bias values for NV12 to RGB conversion
            std::vector<float> standardBiasValues = {-276.928f, 135.488f, -222.912f};

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

        addOp->replaceAllUsesWith(finalOutput.getDefiningOp());
    });
}

}  // namespace

//
// createFuseColorConversionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseColorConversionPass(Logger log) {
    return std::make_unique<FuseColorConversionPass>(log);
}
