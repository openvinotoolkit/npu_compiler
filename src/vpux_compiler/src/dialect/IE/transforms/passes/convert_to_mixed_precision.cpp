//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/convert_to_mixed_precision.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/sdpa_heuristics.hpp"

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

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

    auto dequantizeInput = IE::findQuantizedInput(convolutionOp.getInput(), {});
    auto filterDequantizeInput =
            IE::findQuantizedInput(convolutionOp.getFilter(), IE::getLegalWeightsQuantAxes(convolutionOp));

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

    auto dequantizeType =
            IE::findQuantizedInput(groupConvolutionOp.getInput(), IE::getLegalActivationQuantAxes(groupConvolutionOp));
    auto filterDequantizeType =
            IE::findQuantizedInput(groupConvolutionOp.getFilter(), IE::getLegalWeightsQuantAxes(groupConvolutionOp));

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

    auto dequantizeInput = IE::findQuantizedInput(origOp.getInput(), {});
    auto filterDequantizeInput = IE::findQuantizedInput(origOp.getFilter(), IE::getLegalWeightsQuantAxes(origOp));

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

    auto dequantizeInput = IE::findQuantizedInput(matmulOp.getInput1(), {});
    auto filterDequantizeInput = IE::findQuantizedInput(matmulOp.getInput2(), IE::getLegalWeightsQuantAxes(matmulOp));

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
    auto dequantizeType = IE::findQuantizedInput(avgPoolOp.getInput(), {});
    if (dequantizeType == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AvgPoolOp>(
            avgPoolOp, avgPoolOp.getType(), dequantizeType, avgPoolOp.getScale(), avgPoolOp.getKernelSize(),
            avgPoolOp.getStrides(), avgPoolOp.getPadsBegin(), avgPoolOp.getPadsEnd(), avgPoolOp.getRoundingTypeAttr(),
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
    auto lhsDequant = IE::findQuantizedInput(addOp.getInput1(), {});
    if (lhsDequant == nullptr) {
        return mlir::failure();
    }
    auto rhsDequant = IE::findQuantizedInput(addOp.getInput2(), {});
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

mlir::LogicalResult QuantizeWithMultiplyRewriter::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    if (isPerAxisQuant(quantizeOp.getOutput())) {
        return matchFailed(_log, rewriter, quantizeOp, "Per-axis output quantization is not supported for Multiply");
    }

    const auto outType = mlir::cast<vpux::NDTypeInterface>(quantizeOp.getOutput().getType());
    const auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outType.getElementType());
    // Reject non-integer 8-bit storage types
    if (qType == nullptr || !qType.getStorageType().isInteger(8)) {
        return matchFailed(_log, rewriter, quantizeOp,
                           "Only per-tensor i8/u8 output quantization is supported for Multiply");
    }

    auto multiplyOp = quantizeOp.getInput().getDefiningOp<IE::MultiplyOp>();
    if (multiplyOp == nullptr) {
        return matchFailed(_log, rewriter, quantizeOp, "Producer is not a MultiplyOp");
    }

    if (multiplyOp.getScale() != nullptr) {
        return matchFailed(_log, rewriter, quantizeOp, "MultiplyOp with scales is not supported");
    }

    const auto in1ElemType = mlir::cast<vpux::NDTypeInterface>(multiplyOp.getInput1().getType()).getElementType();
    const auto in2ElemType = mlir::cast<vpux::NDTypeInterface>(multiplyOp.getInput2().getType()).getElementType();
    if (!mlir::isa<mlir::Float16Type>(in1ElemType) || !mlir::isa<mlir::Float16Type>(in2ElemType)) {
        return matchFailed(_log, rewriter, quantizeOp, "Only f16 inputs are supported for quantized-output Multiply");
    }

    const auto in1Shape = mlir::cast<vpux::NDTypeInterface>(multiplyOp.getInput1().getType()).getShape();
    const auto in2Shape = mlir::cast<vpux::NDTypeInterface>(multiplyOp.getInput2().getType()).getShape();
    if (in1Shape != in2Shape) {
        return matchFailed(_log, rewriter, quantizeOp,
                           "Broadcast Multiply (differing input shapes) is not supported for quantized-output fusion");
    }

    if (!multiplyOp->getResult(0).hasOneUse()) {
        return matchFailed(_log, rewriter, quantizeOp, "MultiplyOp has more than one consumer");
    }

    // Skip fusion when Q->Deq or Q->QuantizeCast->Deq immediately follows: downstream passes fold
    // these chains away, so fusing would add an extra Shave dequantize kernel.
    if (quantizeOp.getOutput().hasOneUse() &&
        mlir::isa<IE::DequantizeOp, IE::QuantizeCastOp>(*quantizeOp.getOutput().getUsers().begin())) {
        return matchFailed(
                _log, rewriter, quantizeOp,
                "Quantize output feeds a Deq/QuantizeCast chain; fusion would add an extra dequantize kernel");
    }

    // Skip fusion if the Quantize output reaches a multi-input op whose other operands stay
    // f16, causing a type mismatch after fusion. Past a Dequantize the type resets to f16,
    // but IE::AddOp is still checked because FloatOutAddRewriter can cascade on it.

    mlir::Value current = quantizeOp.getOutput();
    bool hasMultiInputConsumer = false;
    bool passedThroughDequantize = false;
    while (current.hasOneUse()) {
        auto* user = *current.getUsers().begin();
        if (mlir::isa<IE::DequantizeOp>(user)) {
            if (user->getNumResults() != 1) {
                break;
            }
            passedThroughDequantize = true;
            current = user->getResult(0);
            continue;
        }
        // Pure view ops are transparent regardless of operand count (e.g. Squeeze with
        // axes-as-tensor has >1 tensor operand but is still layout-only).
        if (IE::isPureViewOp(user) && user->getNumResults() == 1) {
            current = user->getResult(0);
            continue;
        }
        const auto isTensor = [](mlir::Type t) {
            return mlir::isa<mlir::TensorType>(t);
        };
        if (llvm::count_if(user->getOperandTypes(), isTensor) > 1) {
            // Post-Dequantize, only IE::AddOp needs checking (FloatOutAddRewriter risk).
            if (!passedThroughDequantize || mlir::isa<IE::AddOp>(user)) {
                hasMultiInputConsumer = true;
            }
        }
        break;
    }

    if (hasMultiInputConsumer) {
        return matchFailed(_log, rewriter, quantizeOp,
                           "Quantize output feeds a multi-input op; fusion would cause buffer conflicts");
    }

    rewriter.replaceOpWithNewOp<IE::MultiplyOp>(quantizeOp, quantizeOp.getOutput().getType(), multiplyOp.getInput1(),
                                                multiplyOp.getInput2(), multiplyOp.getAutoBroadcastAttr(),
                                                multiplyOp.getPostOpAttr(), multiplyOp.getClampAttr(),
                                                multiplyOp.getOutputPaddingAttr(), multiplyOp.getInputPaddingAttr());
    rewriter.eraseOp(multiplyOp);

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

    if (shouldRejectQuantizeFusionForSdpaSoftmax(maybeNCETask, qElemType)) {
        return matchFailed(_log, rewriter, origOp,
                           "SDPA output with non-zero zero-point quantization must stay unfused");
    }

    // When the NCE op has a PPE post-op (ReLU, PReLU, Clamp), hardware executes SCALE before the
    // post-op. For a negative output scale (e.g. quant.uniform<u8:f16, -0.0039:255>), SCALE maps
    // positive accumulators to negative integers; the subsequent ReLU/Clamp then clips them to the
    // wrong range, producing incorrect quantized values.
    auto postOpIfc = mlir::dyn_cast<IE::LayerWithPostOpInterface>(maybeNCETask);
    if (IE::hasNegativeScales(qElemType) && postOpIfc != nullptr && postOpIfc.hasPPE()) {
        return matchFailed(_log, rewriter, origOp,
                           "Output quantization has negative scale(s) combined with PPE post-op; fuse rejected");
    }

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

mlir::Value QuantizeWithNCERewriter::peelLeadingPureViewOps(mlir::Value value) const {
    while (auto* defOp = value.getDefiningOp()) {
        if (!IE::isPureViewOp(defOp)) {
            break;
        }
        value = defOp->getOperand(0);
    }
    return value;
}

mlir::Value QuantizeWithNCERewriter::peelTrailingPureViewOps(mlir::Value value) const {
    while (value.hasOneUse()) {
        auto* user = *value.getUsers().begin();
        if (!IE::isPureViewOp(user)) {
            break;
        }
        value = user->getResult(0);
    }
    return value;
}

bool QuantizeWithNCERewriter::isSdpaSoftmaxConsumerOperand(mlir::Value operand) const {
    operand = peelLeadingPureViewOps(operand);
    auto softmaxOp = mlir::dyn_cast_if_present<IE::SoftMaxOp>(operand.getDefiningOp());
    if (softmaxOp == nullptr) {
        return false;
    }

    const auto softmaxConsumerOperand = peelTrailingPureViewOps(softmaxOp.getOutput());
    if (!softmaxConsumerOperand.hasOneUse() ||
        !mlir::isa<IE::ConvolutionOp>(*softmaxConsumerOperand.getUsers().begin())) {
        return false;
    }

    auto softmaxInput = peelLeadingPureViewOps(softmaxOp.getInput());
    return mlir::isa<IE::ConvolutionOp>(softmaxInput.getDefiningOp());
}

// Looks for conv-lowered SDPA shape:
// Convolution -> Softmax -> Convolution -> Quantize
// Intentionally excludes MatMul-form SDPA:
// MatMul -> Softmax -> MatMul -> Quantize
// because this variant currently does not benefit from DecomposeSoftmaxInSdpa.
bool QuantizeWithNCERewriter::shouldRejectQuantizeFusionForSdpaSoftmax(mlir::Operation* producerOp,
                                                                       mlir::quant::QuantizedType qElemType) const {
    if (!mlir::isa<IE::ConvolutionOp>(producerOp) || IE::areAllQuantTypeZeroPointsEqualToZero(qElemType)) {
        return false;
    }

    const auto producerOutputShape = getShape(producerOp->getResult(0));
    return llvm::any_of(producerOp->getOperands(), [this, producerOutputShape](mlir::Value operand) {
        operand = peelLeadingPureViewOps(operand);
        auto softmaxOp = mlir::dyn_cast_if_present<IE::SoftMaxOp>(operand.getDefiningOp());
        if (softmaxOp == nullptr || !isSdpaSoftmaxConsumerOperand(operand)) {
            return false;
        }

        const auto softmaxOutputShape = getShape(softmaxOp.getOutput());
        return isSdpaSoftmaxDecompositionBeneficialByShape(softmaxOutputShape, producerOutputShape);
    });
}

template <typename ConcreteOp>
mlir::LogicalResult MixedFloatInQuantWeightsWithDynamicDequantizeRewriter<ConcreteOp>::matchAndRewrite(
        ConcreteOp convOp, mlir::PatternRewriter& rewriter) const {
    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(convOp.getOperation());
    if (!quantizedLayerOp || !quantizedLayerOp.isMixPrecisionSupported(true)) {
        return mlir::failure();
    }

    auto op = convOp.getOperation();
    auto* ctx = convOp.getContext();

    const auto dequantizeType = IE::findQuantizedInput(op->getOperand(0), {});
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
    auto hasAlreadyScale = convOp.getScale() != nullptr;

    mlir::Value newScale;
    if (hasAlreadyScale) {
        mlir::Value scale = convOp.getScale();
        auto initialShape = getShape(scale);
        while (mlir::isa_and_present<ViewLikeOpInterface>(scale.getDefiningOp())) {
            scale = scale.getDefiningOp()->getOperand(0);
        }
        auto maxpoolOp = mlir::dyn_cast<IE::MaxPoolOp>(scale.getDefiningOp());
        if (maxpoolOp == nullptr) {
            return mlir::failure();
        }
        auto output = maxpoolOp.getOutput();
        auto input1 = maxpoolOp.getInput();
        auto input2 =
                rewriter.createOrFold<IE::ReshapeOp>(appendLoc(scale.getLoc(), "scale_reshape_in_2"),
                                                     filterDequantizeScale, getIntArrayAttr(ctx, getShape(input1)));

        const auto nhwcOrderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
        auto layoutCast1 = rewriter.create<IE::LayoutCastOp>(appendLoc(scale.getLoc(), "scale_layout_cast_in_1"),
                                                             input1, nhwcOrderAttr);
        auto layoutCast2 = rewriter.create<IE::LayoutCastOp>(appendLoc(scale.getLoc(), "scale_layout_cast_in_2"),
                                                             input2, nhwcOrderAttr);
        auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
        auto newOutput = outputType.changeDimsOrder(DimsOrder::NHWC);
        auto multiply = rewriter.create<IE::MultiplyOp>(appendLoc(scale.getLoc(), "scale_multiply"), newOutput,
                                                        layoutCast1, layoutCast2, IE::AutoBroadcastType::NUMPY, nullptr,
                                                        nullptr, nullptr, nullptr);
        auto layoutCastOut =
                rewriter.create<IE::LayoutCastOp>(appendLoc(scale.getLoc(), "scale_layout_cast_out"), multiply,
                                                  mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(ctx)));

        newScale = rewriter.createOrFold<IE::ReshapeOp>(appendLoc(scale.getLoc(), "scale_reshape_out"), layoutCastOut,
                                                        getIntArrayAttr(ctx, initialShape));
    } else {
        newScale = IE::createConvertPoolingForScaleTable(filterDequantizeScale, rewriter);
    }

    auto newInput = dequantizeType != nullptr ? dequantizeType : convOp.getInput();

    if (auto origOp = mlir::dyn_cast_if_present<IE::ConvolutionOp>(op)) {
        rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
                origOp, origOp.getOutput().getType(), newInput, filterDequantizeInput, origOp.getBias(), newScale,
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
