//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/convert_to_mixed_precision.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/transforms/factories/convert_to_mixed_precision_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/Value.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTTOMIXEDPRECISION
#define GEN_PASS_DEF_CONVERTTOMIXEDPRECISION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;
using namespace IE;

mlir::LogicalResult FloatOutConvRewriter::matchAndRewrite(IE::ConvolutionOp convolutionOp,
                                                          mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(convolutionOp) || !_isMixPrecisionSupported(convolutionOp, false, _log)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(convolutionOp))) {
        return mlir::failure();
    }

    auto dequantizeInput = IE::findQuantizedInput(convolutionOp.getInput(), false);
    auto filterDequantizeInput = IE::findQuantizedInput(convolutionOp.getFilter(), true);

    if (dequantizeInput == nullptr || filterDequantizeInput == nullptr) {
        return mlir::failure();
    }

    auto newConv = rewriter.create<IE::ConvolutionOp>(
            convolutionOp->getLoc(), convolutionOp.getType(), dequantizeInput, filterDequantizeInput,
            convolutionOp.getBias(), convolutionOp.getStrides(), convolutionOp.getPadsBegin(),
            convolutionOp.getPadsEnd(), convolutionOp.getDilations(), convolutionOp.getPostOpAttr(),
            convolutionOp.getClampAttr(), convolutionOp.getStaticScaleAttr(), convolutionOp.getOutputPaddingAttr(),
            convolutionOp.getInputPaddingAttr());
    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newConv)) {
        rewriter.eraseOp(newConv);
        return mlir::failure();
    }

    rewriter.replaceOp(convolutionOp, newConv.getOutput());

    return mlir::success();
}

mlir::LogicalResult FloatOutGroupConvRewriter::matchAndRewrite(IE::GroupConvolutionOp groupConvolutionOp,
                                                               mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(groupConvolutionOp) || !_isMixPrecisionSupported(groupConvolutionOp, false, _log)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(groupConvolutionOp))) {
        return mlir::failure();
    }

    auto dequantizeType = IE::findQuantizedInput(groupConvolutionOp.getInput(), true);
    auto filterDequantizeType = IE::findQuantizedInput(groupConvolutionOp.getFilter(), true);

    if (dequantizeType == nullptr || filterDequantizeType == nullptr) {
        return mlir::failure();
    }

    auto newGroupConv = rewriter.create<IE::GroupConvolutionOp>(
            groupConvolutionOp->getLoc(), groupConvolutionOp.getType(), dequantizeType, filterDequantizeType,
            groupConvolutionOp.getBias(), groupConvolutionOp.getStrides(), groupConvolutionOp.getPadsBegin(),
            groupConvolutionOp.getPadsEnd(), groupConvolutionOp.getDilations(), groupConvolutionOp.getGroupsAttr(),
            groupConvolutionOp.getPostOpAttr(), groupConvolutionOp.getClampAttr(),
            groupConvolutionOp.getOutputPaddingAttr(), groupConvolutionOp.getInputPaddingAttr());

    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newGroupConv)) {
        rewriter.eraseOp(newGroupConv);
        return mlir::failure();
    }

    rewriter.replaceOp(groupConvolutionOp, newGroupConv.getOutput());

    return mlir::success();
}

mlir::LogicalResult FloatOutTransposedConvRewriter::matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(origOp) || !_isMixPrecisionSupported(origOp, false, _log)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(origOp))) {
        return mlir::failure();
    }

    auto dequantizeInput = IE::findQuantizedInput(origOp.getInput(), false);
    auto filterDequantizeInput = IE::findQuantizedInput(origOp.getFilter(), true);

    if (dequantizeInput == nullptr || filterDequantizeInput == nullptr) {
        return mlir::failure();
    }

    auto newTransposedConv = rewriter.create<IE::TransposedConvolutionOp>(
            origOp->getLoc(), origOp.getType(), dequantizeInput, filterDequantizeInput, origOp.getOutputShape(),
            origOp.getBias(), origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(),
            origOp.getSpatialOutputPaddingAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newTransposedConv)) {
        rewriter.eraseOp(newTransposedConv);
        return mlir::failure();
    }

    rewriter.replaceOp(origOp, newTransposedConv.getOutput());

    return mlir::success();
}

mlir::LogicalResult FloatOutMatMulRewriter::matchAndRewrite(IE::MatMulOp matmulOp,
                                                            mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(matmulOp) || !_isMixPrecisionSupported(matmulOp, false, _log)) {
        return mlir::failure();
    }

    auto dequantizeInput = IE::findQuantizedInput(matmulOp.getInput1(), false);
    auto filterDequantizeInput = IE::findQuantizedInput(matmulOp.getInput2(), true);

    if (dequantizeInput == nullptr || filterDequantizeInput == nullptr) {
        return mlir::failure();
    }

    auto newMatmulOp = rewriter.create<IE::MatMulOp>(matmulOp->getLoc(), matmulOp.getType(), dequantizeInput,
                                                     filterDequantizeInput, matmulOp.getTransposeA(),
                                                     matmulOp.getTransposeB(), matmulOp.getPostOpAttr());
    // E#157376: Following check is always true for IE::Matmuls, but should be updated to do similar checks with
    // Convolutions
    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newMatmulOp)) {
        rewriter.eraseOp(newMatmulOp);
        return mlir::failure();
    }

    rewriter.replaceOp(matmulOp, newMatmulOp.getOutput());

    return mlir::success();
}

mlir::LogicalResult FloatOutAvgPoolRewriter::matchAndRewrite(IE::AvgPoolOp avgPoolOp,
                                                             mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(avgPoolOp) || !IE::arch37xx::isMixPrecisionSupported(avgPoolOp, false, _log)) {
        return mlir::failure();
    }
    // Although the operation could support per channel quant params because is depthwise,
    // it does not have access to weights table, which is where per channel quant params
    // are placed. Only global, per tensor quantization is supported by AVG Pool.
    auto dequantizeType = IE::findQuantizedInput(avgPoolOp.getInput(), false);
    if (dequantizeType == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AvgPoolOp>(
            avgPoolOp, avgPoolOp.getType(), dequantizeType, avgPoolOp.getKernelSize(), avgPoolOp.getStrides(),
            avgPoolOp.getPadsBegin(), avgPoolOp.getPadsEnd(), avgPoolOp.getRoundingTypeAttr(),
            avgPoolOp.getExcludePadsAttr(), avgPoolOp.getPostOpAttr(), avgPoolOp.getClampAttr(),
            avgPoolOp.getStaticScaleAttr(), avgPoolOp.getOutputPaddingAttr(), avgPoolOp.getInputPaddingAttr());

    return mlir::success();
}

mlir::LogicalResult FloatOutAddRewriter::matchAndRewrite(IE::AddOp addOp, mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(addOp) || !_isMixPrecisionSupported(addOp, false, _log)) {
        return mlir::failure();
    }
    // This transformation assumes that each input has IE::DequantizeOp producer
    auto lhsDequant = IE::findQuantizedInput(addOp.getInput1(), false);
    if (lhsDequant == nullptr) {
        return mlir::failure();
    }
    auto rhsDequant = IE::findQuantizedInput(addOp.getInput2(), false);
    if (rhsDequant == nullptr) {
        return mlir::failure();
    }

    auto lhsElemType = mlir::cast<vpux::NDTypeInterface>(lhsDequant.getType()).getElementType();
    auto rhsElemType = mlir::cast<vpux::NDTypeInterface>(rhsDequant.getType()).getElementType();

    if (!isSupportedEltwiseQuantization(lhsElemType, rhsElemType, _allowDifferentScales, /*allowDifferentZp=*/true,
                                        VPU::EltwiseType::ADD)) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AddOp>(addOp, addOp.getType(), lhsDequant, rhsDequant, addOp.getAutoBroadcast(),
                                           addOp.getPostOpAttr(), addOp.getClampAttr(), addOp.getOutputPaddingAttr(),
                                           addOp.getInputPaddingAttr());

    return mlir::success();
}

mlir::LogicalResult QuantizeWithNCERewriter::matchAndRewrite(IE::QuantizeOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    const auto isOutputPerAxisQuant = isPerAxisQuant(origOp.getOutput());
    const auto maybeNCETask = origOp.getInput().getDefiningOp();
    if (maybeNCETask == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Producer is a block argument");
    }
    if (!maybeNCETask->getResult(0).hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "NCE task has more than one consumer");
    }
    if (mlir::isa<IE::MaxPoolOp>(maybeNCETask)) {
        return matchFailed(_log, rewriter, origOp,
                           "{0} is a quantization-agnostic operation, mixed precision is not supported",
                           maybeNCETask->getName());
    }
    if (!_isPerAxesSupported && isOutputPerAxisQuant &&
        mlir::isa<IE::AddOp, IE::SubtractOp, IE::MultiplyOp, IE::AvgPoolOp>(maybeNCETask)) {
        return matchFailed(_log, rewriter, origOp,
                           "IE.AvgPool and Eltwise do not support per-channel quantized output");
    }

    auto layerWithPostOp = mlir::dyn_cast_or_null<IE::LayerWithPostOpInterface>(maybeNCETask);
    if (layerWithPostOp != nullptr && layerWithPostOp.getPostOp() != nullptr &&
        !_checkPostOp(layerWithPostOp, isOutputPerAxisQuant, /*isFloatInput=*/true)) {
        return matchFailed(_log, rewriter, origOp, "Layer with PostOp not supported");
    }

    // NCE tasks with float input and quant output support LeakyReLU only per-tensor quantize output.
    // One would expect that with ops ran sequential: BIAS->SCALE->PRELU, we could easily support prelu and per axis
    // quant params. But actually in HW, depending on the sign of the FP BIAS result, you either execute SCALE or PRELU.
    // So for the negative values we'd have to combine the prelu alpha parameter and the requant scale into the per
    // tensor param for prelu scale. This explains why we can't have prelu with per axis quant in fp mode
    if (!_isMixPrecisionSupported(maybeNCETask, !isOutputPerAxisQuant, _log)) {
        return matchFailed(_log, rewriter, origOp, "Producer {0} is not supported", maybeNCETask->getName());
    }

    auto* newNCETask = rewriter.clone(*maybeNCETask);
    newNCETask->getResult(0).setType(origOp.getOutput().getType());
    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newNCETask)) {
        rewriter.eraseOp(newNCETask);
        return mlir::failure();
    }
    rewriter.replaceOp(origOp, newNCETask->getResult(0));
    rewriter.eraseOp(maybeNCETask);

    return mlir::success();
}

namespace {

//
// ConvertToMixedPrecisionPass
//

class ConvertToMixedPrecisionPass final : public IE::impl::ConvertToMixedPrecisionBase<ConvertToMixedPrecisionPass> {
public:
    ConvertToMixedPrecisionPass(bool enableFloatInQuantWeightsMixedMode, Logger log) {
        this->enableFloatInQuantWeightsMixedMode = enableFloatInQuantWeightsMixedMode;
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() override;
};

void ConvertToMixedPrecisionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto strategy = IE::createConvertToMixedPrecisionStrategy(func, enableFloatInQuantWeightsMixedMode);

    mlir::RewritePatternSet patterns(&ctx);
    strategy->addPatterns(patterns, _log);

    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToMixedPrecision
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToMixedPrecision(bool enableFloatInQuantWeightsMixedMode,
                                                                    Logger log) {
    return std::make_unique<ConvertToMixedPrecisionPass>(enableFloatInQuantWeightsMixedMode, log);
}
