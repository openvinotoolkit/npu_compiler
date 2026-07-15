//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/convert_to_mixed_precision.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <numeric>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTTOMIXEDPRECISION
#define GEN_PASS_DEF_CONVERTTOMIXEDPRECISION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;
using namespace IE;

mlir::LogicalResult FloatOutConvRewriter::matchAndRewrite(IE::ConvolutionOp convolutionOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(convolutionOp.getOperation());
    if (!quantizedLayerOp || !quantizedLayerOp.isMixPrecisionSupported(false)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(convolutionOp))) {
        return mlir::failure();
    }

    auto dequantizeInput = IE::findQuantizedInput(convolutionOp.getInput(), false);
    auto filterDequantizeInput = IE::findQuantizedInput(convolutionOp.getFilter(), true);

    if (!IE::isInputQuantizationSupported(dequantizeInput, filterDequantizeInput)) {
        return mlir::failure();
    }

    auto newConv = cloneConvolutionOp(rewriter, convolutionOp, convolutionOp.getType(), dequantizeInput,
                                      filterDequantizeInput);

    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newConv)) {
        rewriter.eraseOp(newConv);
        return mlir::failure();
    }

    rewriter.replaceOp(convolutionOp, newConv.getOutput());

    return mlir::success();
}

mlir::LogicalResult FloatOutGroupConvRewriter::matchAndRewrite(IE::GroupConvolutionOp groupConvolutionOp,
                                                               mlir::PatternRewriter& rewriter) const {
    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(groupConvolutionOp.getOperation());
    if (IE::areAnyUserQuantizeOps(groupConvolutionOp) || !quantizedLayerOp ||
        !quantizedLayerOp.isMixPrecisionSupported(false)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(groupConvolutionOp))) {
        return mlir::failure();
    }

    auto dequantizeType = IE::findQuantizedInput(groupConvolutionOp.getInput(), true);
    auto filterDequantizeType = IE::findQuantizedInput(groupConvolutionOp.getFilter(), true);

    if (!IE::isInputQuantizationSupported(dequantizeType, filterDequantizeType)) {
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
    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(origOp.getOperation());
    if (IE::areAnyUserQuantizeOps(origOp) || !quantizedLayerOp || !quantizedLayerOp.isMixPrecisionSupported(false)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(origOp))) {
        return mlir::failure();
    }

    auto dequantizeInput = IE::findQuantizedInput(origOp.getInput(), false);
    auto filterDequantizeInput = IE::findQuantizedInput(origOp.getFilter(), true);

    if (!IE::isInputQuantizationSupported(dequantizeInput, filterDequantizeInput)) {
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
    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(matmulOp.getOperation());
    if (IE::areAnyUserQuantizeOps(matmulOp) || !quantizedLayerOp || !quantizedLayerOp.isMixPrecisionSupported(false)) {
        return mlir::failure();
    }

    auto dequantizeInput = IE::findQuantizedInput(matmulOp.getInput1(), false);
    auto filterDequantizeInput = IE::findQuantizedInput(matmulOp.getInput2(), true);

    if (!IE::isInputQuantizationSupported(dequantizeInput, filterDequantizeInput)) {
        return mlir::failure();
    }

    auto newMatmulOp = cloneMatMulOp(rewriter, matmulOp, matmulOp.getType(), dequantizeInput, filterDequantizeInput);
    // E#157376: Following check is always true for IE::Matmuls, but should be updated to do similar checks with
    // Convolutions
    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newMatmulOp)) {
        rewriter.eraseOp(newMatmulOp);
        return mlir::failure();
    }

    rewriter.replaceOp(matmulOp, newMatmulOp);

    return mlir::success();
}

mlir::LogicalResult FloatOutAvgPoolRewriter::matchAndRewrite(IE::AvgPoolOp avgPoolOp,
                                                             mlir::PatternRewriter& rewriter) const {
    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(avgPoolOp.getOperation());
    if (IE::areAnyUserQuantizeOps(avgPoolOp) || !quantizedLayerOp || !quantizedLayerOp.isMixPrecisionSupported(false)) {
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
    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(addOp.getOperation());
    if (IE::areAnyUserQuantizeOps(addOp) || !quantizedLayerOp || !quantizedLayerOp.isMixPrecisionSupported(false)) {
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

    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(maybeNCETask);
    if (!quantizedLayerOp) {
        return matchFailed(_log, rewriter, origOp, "Producer {0} does not support QuantizedLayerOpInterface",
                           maybeNCETask->getName());
    }

    if (!quantizedLayerOp.isOutputQuantizationFusable(isOutputPerAxisQuant, /*isFloatInput=*/true)) {
        return matchFailed(_log, rewriter, origOp,
                           "Producer {0} does not support output quantization fusion for float-input mixed "
                           "precision: per-channel quantized output or required PPE/post-op is unsupported",
                           maybeNCETask->getName());
    }

    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    auto qElemType = mlir::cast<mlir::quant::QuantizedType>(outType.getElementType());
    auto isQuantWidthSupported = qElemType.getStorageTypeIntegralWidth() != 16;

    if (!isQuantWidthSupported) {
        return mlir::failure();
    }

    // NCE tasks with float input and quant output support LeakyReLU only per-tensor quantize output.
    // One would expect that with ops ran sequential: BIAS->SCALE->PRELU, we could easily support prelu and per axis
    // quant params. But actually in HW, depending on the sign of the FP BIAS result, you either execute SCALE or PRELU.
    // So for the negative values we'd have to combine the prelu alpha parameter and the requant scale into the per
    // tensor param for prelu scale. This explains why we can't have prelu with per axis quant in fp mode
    if (!quantizedLayerOp.isMixPrecisionSupported(!isOutputPerAxisQuant)) {
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

template <typename ConcreteOp>
mlir::LogicalResult MixedFloatInQuantWeightsWithDynamicDequantizeRewriter<ConcreteOp>::matchAndRewrite(
        ConcreteOp convOp, mlir::PatternRewriter& rewriter) const {
    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(convOp.getOperation());
    if (!quantizedLayerOp || !quantizedLayerOp.isMixPrecisionSupported(true)) {
        return mlir::failure();
    }

    auto ctx = convOp->getContext();
    auto op = convOp.getOperation();

    const auto dequantizeType = IE::findQuantizedInput(op->getOperand(0), false);
    auto dynamicDequantOp =
            mlir::dyn_cast_if_present<IE::DynamicDequantizeOp>(IE::findDynDequantized(op->getOperand(1), true));

    auto isFP8 = false;

    if (dequantizeType != nullptr) {
        const auto inType = mlir::cast<vpux::NDTypeInterface>(dequantizeType.getType());
        const auto elemType = inType.getElementType();
        // Only FP8 dequantized activations are supported by this rewriter; other dequantized activations are handled
        // elsewhere.
        isFP8 = vpux::isFloat8(elemType) || vpux::isFloat8Quantized(elemType);
        if (!isFP8) {
            return mlir::failure();
        }
    }

    if (dynamicDequantOp == nullptr) {
        return mlir::failure();
    }
    const auto filterDequantizeInput = dynamicDequantOp.getInput();
    const auto filterDequantizeScale = dynamicDequantOp.getScale();
    const auto filterDequantizeZP = dynamicDequantOp.getZp();

    const auto quantFilterDequantizeType = mlir::dyn_cast<mlir::quant::QuantizedType>(
            mlir::cast<vpux::NDTypeInterface>(filterDequantizeInput.getType()).getElementType());
    if (quantFilterDequantizeType == nullptr) {
        return mlir::failure();
    }

    const auto isSignedQuantizedType = [](mlir::quant::QuantizedType quantType) {
        if (mlir::isa<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(quantType)) {
            const auto quantized = mlir::cast<mlir::quant::QuantizedType>(quantType);
            if (const auto quantileFloatStorage =
                        mlir::dyn_cast<vpux::type::QuantileType>(quantized.getStorageType())) {
                mlir::Type quantileType = quantileFloatStorage.getQuantileType();
                auto intType = mlir::dyn_cast<mlir::IntegerType>(quantileType);
                return intType == nullptr || intType.isSigned();
            }
        }
        return quantType.isSigned();
    };

    const auto perChannelQuantType =
            mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(quantFilterDequantizeType);
    const auto perTensorQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(quantFilterDequantizeType);
    const auto isSymmetricQuant = IE::isSymmetricQuantType(quantFilterDequantizeType);
    auto moduleOp = getModuleOp(convOp.getOperation());
    const auto isAsymmetricPerChannelSupported = config::asymmetricPerChannelZeroPointSupported(moduleOp);
    const auto isAsymmetricPerTensorSupported = config::asymmetricPerTensorZeroPointSupported(moduleOp);

    // Only signed quant is supported for input + wt mixed precision
    if (!isSignedQuantizedType(quantFilterDequantizeType) ||
        (perChannelQuantType && !isAsymmetricPerChannelSupported && !isSymmetricQuant) ||
        (perTensorQuantType && !isAsymmetricPerTensorSupported && !isSymmetricQuant)) {
        return mlir::failure();
    }

    if (isFP8) {
        const auto storageType = quantFilterDequantizeType.getStorageType();
        if (mlir::isa<mlir::IntegerType>(storageType) && storageType.getIntOrFloatBitWidth() == 8) {
            // Activation FP8 quantization with uint8/int8 weights is not supported in mixed precision mode.
            return mlir::failure();
        }
    }

    const auto hasLeakyReLUConsumer = llvm::any_of(convOp->getUsers(), [](mlir::Operation* op) {
        return mlir::isa<IE::LeakyReluOp>(op);
    });

    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(quantFilterDequantizeType) &&
        (hasLeakyReLUConsumer || IE::hasLeakyReLUPostOp(convOp))) {
        return mlir::failure();
    }

    auto scale = [&]() -> mlir::Value {
        if (filterDequantizeScale == nullptr) {
            return nullptr;
        }

        const auto scaleType = mlir::cast<vpux::NDTypeInterface>(filterDequantizeScale.getType());
        if (scaleType.getElementType().isF32()) {
            return filterDequantizeScale;
        }

        // Use a dummy identity MaxPool (1x1 kernel, stride 1, no padding) to perform
        // the f16 -> f32 type conversion instead of a plain ConvertOp.
        // Reshape the scale tensor so channels <= 256; remaining elements are
        // distributed as equally as possible across H and W.
        const auto origShape = scaleType.getShape();
        const int64_t totalElems =
                std::accumulate(origShape.begin(), origShape.end(), int64_t(1), std::multiplies<int64_t>());

        // Find the largest supported channel count that divides totalElems.
        const auto& strategyFactory = VPU::getVPUStrategyFactory(ctx);
        SmallVector<int64_t> supportedChannels(strategyFactory->getSupportedChannelsDW());
        int64_t C = 1;
        for (const int64_t candidate : supportedChannels) {
            if (candidate <= totalElems && totalElems % candidate == 0) {
                C = candidate;
                break;
            }
        }

        // Split the remaining elements into H and W (roughly equal).
        const auto rem = totalElems / C;
        auto H = static_cast<int64_t>(std::sqrt(static_cast<double>(rem)));
        while (H > 1 && rem % H != 0) {
            --H;
        }
        const auto W = rem / H;

        const SmallVector<int64_t> poolShape = {1, C, H, W};
        const auto poolShapeAttr = getIntArrayAttr(ctx, poolShape);

        // Reshape scale into [1, C, H, W] for the MaxPool.
        auto reshapedScale = rewriter.createOrFold<IE::ReshapeOp>(
                appendLoc(filterDequantizeScale.getLoc(), "scale_reshape_in"), filterDequantizeScale, poolShapeAttr);

        // Identity MaxPool outputs fp32.
        const auto reshapedType = mlir::cast<vpux::NDTypeInterface>(reshapedScale.getType());
        const auto fp32PoolType = reshapedType.changeElemType(mlir::Float32Type::get(ctx));
        auto maxPoolOut = IE::createIdentityMaxPool(reshapedScale, fp32PoolType, rewriter)->getResult(0);

        // Reshape back to the original logical shape with fp32 element type.
        const auto origShapeAttr = getIntArrayAttr(ctx, SmallVector<int64_t>(origShape.begin(), origShape.end()));
        return rewriter.createOrFold<IE::ReshapeOp>(appendLoc(filterDequantizeScale.getLoc(), "scale_reshape_out"),
                                                    maxPoolOut, origShapeAttr);
    }();

    auto newInput = dequantizeType != nullptr ? dequantizeType : convOp.getInput();

    if (auto origOp = mlir::dyn_cast_if_present<IE::ConvolutionOp>(op)) {
        rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
                origOp, origOp.getOutput().getType(), newInput, filterDequantizeInput, origOp.getBias(), scale,
                filterDequantizeZP, origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(),
                origOp.getDilationsAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getStaticScaleAttr(),
                origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
    }

    return mlir::success();
}

template class vpux::IE::MixedFloatInQuantWeightsWithDynamicDequantizeRewriter<vpux::IE::ConvolutionOp>;

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

    const auto& strategyFactory = IE::getIEStrategyFactory(&ctx);
    auto strategy = strategyFactory->getConvertToMixedPrecisionStrategy(enableFloatInQuantWeightsMixedMode);

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
