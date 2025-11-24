//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/convert_quantize_ops_to_nce_ops.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/convert_quantize_ops_to_nce_ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/conv_utils.hpp"

namespace vpux::IE {

//
// QuantizeToDwRewriterImpl
//

mlir::LogicalResult QuantizeToDwRewriter::matchAndRewrite(IE::QuantizeOp originOp,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto origType = mlir::cast<vpux::NDTypeInterface>(originOp.getInput().getType());
    const auto origShape = origType.getShape();
    const auto OC = origShape[Dims4D::Act::C];
    auto weights = vpux::buildDwWeights(originOp->getLoc(), OC, origType.getElementType(), rewriter);

    const auto ctx = rewriter.getContext();
    const auto attrStrides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto attrPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto attrPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto dilationsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            originOp, originOp.getOutput().getType(), originOp.getInput(), weights,
            /*bias=*/nullptr, attrStrides, attrPadsBegin, attrPadsEnd, dilationsAttr, getIntAttr(ctx, OC),
            /*post_opAttr=*/nullptr, /*clampAttr*/ nullptr, /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);

    return mlir::success();
}

//
// DequantizeToDwRewriterImpl
//

mlir::LogicalResult DequantizeToDwRewriter::matchAndRewrite(IE::DequantizeOp originOp,
                                                            mlir::PatternRewriter& rewriter) const {
    const auto origType = mlir::cast<vpux::NDTypeInterface>(originOp.getInput().getType());
    auto intType = mlir::cast<mlir::quant::QuantizedType>(origType.getElementType());
    if (intType.getStorageTypeIntegralWidth() > 8) {
        _log.trace("Invalid storage bit width, expected 8, but got {0}", intType.getStorageTypeIntegralWidth());
        return mlir::failure();
    }
    const auto origShape = origType.getShape();
    const auto OC = origShape[Dims4D::Act::C];
    const auto ctx = rewriter.getContext();

    const auto quantizeType = mlir::quant::UniformQuantizedType::get(
            /*flags=*/0, /*storageType=*/getUInt8Type(ctx), /*expressedType=*/mlir::Float16Type::get(ctx),
            /*scale=*/1.0, /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);
    auto quantWeightsOp = vpux::buildDwWeights(originOp->getLoc(), OC, quantizeType, rewriter);

    const auto attrStrides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto attrPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto attrPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto dilationsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            originOp, originOp.getOutput().getType(), originOp.getInput(), quantWeightsOp,
            /*bias=*/nullptr, attrStrides, attrPadsBegin, attrPadsEnd, dilationsAttr, getIntAttr(ctx, OC),
            /*post_opAttr=*/nullptr, /*clampAttr*/ nullptr, /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);

    return mlir::success();
}

//
// DequantizeToAddRewriterImpl
//

mlir::LogicalResult DequantizeToAddRewriter::matchAndRewrite(IE::DequantizeOp originOp,
                                                             mlir::PatternRewriter& rewriter) const {
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);

    auto inElemType = mlir::cast<vpux::NDTypeInterface>(originOp.getInput().getType()).getElementType();
    auto uniformQInElemType = mlir::cast<mlir::quant::UniformQuantizedType>(inElemType);
    const auto scale = uniformQInElemType.getScale();
    // originQElemType = <u8:fp32, scale>
    // newQElemType = <u8:fp32, scale / 2>
    // Op -> originQElemType -> QuantizeCastOp -> newQElemType -> AddOp(output x2) -> result
    const auto newScale = static_cast<double>(scale / 2.0);
    const auto zeroPoint = uniformQInElemType.getZeroPoint();

    auto qType = mlir::cast<mlir::quant::QuantizedType>(inElemType);
    auto outQuantizeElemType = mlir::quant::UniformQuantizedType::get(
            qType.getFlags(), qType.getStorageType(), qType.getExpressedType(), newScale, zeroPoint,
            qType.getStorageTypeMin(), qType.getStorageTypeMax());

    auto quantizeCastOp =
            rewriter.create<IE::QuantizeCastOp>(originOp.getLoc(), originOp.getInput(), outQuantizeElemType);

    rewriter.replaceOpWithNewOp<IE::AddOp>(originOp, originOp.getType(), quantizeCastOp.getResult(),
                                           quantizeCastOp.getResult(), broadcastType, nullptr, nullptr, nullptr,
                                           nullptr);

    return mlir::success();
}

//
// QuantizeToAddRewriterImpl
//

mlir::LogicalResult QuantizeToAddRewriter::matchAndRewrite(IE::QuantizeOp originOp,
                                                           mlir::PatternRewriter& rewriter) const {
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);

    auto outElemType = mlir::cast<vpux::NDTypeInterface>(originOp.getOutput().getType()).getElementType();
    auto uniformQOutElemType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outElemType);
    const auto scale = uniformQOutElemType.getScale();
    // originQElemType = <u8:fp32, scale>
    // newQElemType = <u8:fp32, scale * 2>
    // Op -> AddOp(output x2) -> newQElemType -> QuantizeCastOp -> originQElemType -> result
    const auto newScale = static_cast<double>(scale * 2.0);
    const auto zeroPoint = uniformQOutElemType.getZeroPoint();

    auto qType = mlir::cast<mlir::quant::QuantizedType>(outElemType);
    auto quantizeElemType = mlir::quant::UniformQuantizedType::get(
            qType.getFlags(), qType.getStorageType(), qType.getExpressedType(), newScale, zeroPoint,
            qType.getStorageTypeMin(), qType.getStorageTypeMax());
    auto newAddOutType = mlir::cast<NDTypeInterface>(originOp.getType()).changeElemType(quantizeElemType);

    auto addOp = rewriter.create<IE::AddOp>(originOp.getLoc(), newAddOutType, originOp.getInput(), originOp.getInput(),
                                            broadcastType, nullptr, nullptr, nullptr, nullptr);

    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(originOp, addOp.getResult(), outElemType);

    return mlir::success();
}

//
// DequantizeToConvImpl
//

mlir::LogicalResult DequantizeToConvRewriter::matchAndRewrite(IE::DequantizeOp originOp,
                                                              mlir::PatternRewriter& rewriter) const {
    if (IE::isQuantizedPerAxis(originOp.getInput())) {
        _log.trace("Activations only support per tensor quantization");
        return mlir::failure();
    }

    auto inNDType = mlir::cast<vpux::NDTypeInterface>(originOp.getInput().getType());
    auto inputShape = inNDType.getShape();
    auto inChannels = inputShape[Dims4D::Act::C];
    auto outChannels = inChannels;

    // Create an identity filter
    SmallVector<float> filterValues(outChannels * inChannels, 0.0f);
    for (int64_t i = 0; i < inChannels; ++i) {
        filterValues[i * inChannels + i] = 1.0f;
    }
    auto filterType = mlir::RankedTensorType::get(
            {static_cast<int64_t>(outChannels), static_cast<int64_t>(inChannels), 1, 1}, inNDType.getElementType());
    auto filter = Const::buildWeightsConst(rewriter, originOp.getLoc(), filterType, filterValues);
    auto strides = rewriter.getI64ArrayAttr({1, 1});
    auto pads_begin = rewriter.getI64ArrayAttr({0, 0});
    auto pads_end = rewriter.getI64ArrayAttr({0, 0});
    auto dilations = rewriter.getI64ArrayAttr({1, 1});
    auto convOp = rewriter.create<IE::ConvolutionOp>(originOp.getLoc(), originOp.getType(), originOp.getInput(), filter,
                                                     /*bias=*/nullptr, strides, pads_begin, pads_end, dilations,
                                                     nullptr, nullptr, nullptr, nullptr, nullptr);
    rewriter.replaceOp(originOp, convOp.getResult());
    return mlir::success();
}
}  // namespace vpux::IE
