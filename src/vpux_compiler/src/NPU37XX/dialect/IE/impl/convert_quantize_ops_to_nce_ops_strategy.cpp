//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/convert_quantize_ops_to_nce_ops_strategy.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/convert_quantize_ops_to_nce_ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

namespace {
// cst -> dequant case.
// When a dequantize operation has a constant producer, execute dequantization on activation shaves.
// ConvertQuantizeOpsToNCEOps must skip such dequantize operations.
// Mark them legal.
bool hasConstProducer(IE::DequantizeOp dequantOp) {
    return mlir::isa_and_nonnull<Const::DeclareOp>(dequantOp.getInput().getDefiningOp());
}
};  // namespace

namespace vpux::IE::arch37xx {

mlir::Value buildDwWeights(const mlir::Location& loc, const int64_t OC, const mlir::Type& elementType,
                           mlir::PatternRewriter& rewriter) {
    const auto ctx = rewriter.getContext();
    if (auto quantizeType = elementType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        const auto baseType = mlir::RankedTensorType::get({OC, 1, 1, 1}, mlir::Float16Type::get(ctx));
        const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vpux::type::float16(1.f)));
        const auto contentAttr = Const::ContentAttr::get(baseAttr);
        auto quantWeightsConstAttr = contentAttr.transform().castElemType(quantizeType).get();
        const auto weightsType = contentAttr.getType().cast<vpux::NDTypeInterface>().changeElemType(quantizeType);
        return rewriter.create<Const::DeclareOp>(loc, weightsType, std::move(quantWeightsConstAttr));
    }
    if (elementType.isF16()) {
        const auto baseType = mlir::RankedTensorType::get({OC, 1, 1, 1}, mlir::Float16Type::get(ctx));
        return Const::createConst(rewriter, loc, baseType, ArrayRef(vpux::type::float16(1.f)));
    }
    if (elementType.isF32()) {
        const auto baseType = mlir::RankedTensorType::get({OC, 1, 1, 1}, mlir::Float32Type::get(ctx));
        return Const::createConst(rewriter, loc, baseType, ArrayRef(1.f));
    }
    VPUX_THROW("buildDwWeights: other types are not supported");
}

//
// QuantizeDequantizeToAvgPool
//

template <class ConcreteOp>
class QuantizeDequantizeToAvgPool final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    QuantizeDequantizeToAvgPool(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefitLow), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult QuantizeDequantizeToAvgPool<ConcreteOp>::matchAndRewrite(ConcreteOp originOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    auto newPooling = IE::createIdentityAvgPool(originOp.getInput(), originOp.getType(), rewriter, originOp->getLoc());
    rewriter.replaceOp(originOp, newPooling->getResult(0));
    return mlir::success();
}

//
// DequantizeToAddRewriter
//

class DequantizeToAddRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeToAddRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DequantizeToAddRewriter::matchAndRewrite(IE::DequantizeOp originOp,
                                                             mlir::PatternRewriter& rewriter) const {
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);

    auto inElemType = originOp.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
    auto uniformQInElemType = inElemType.dyn_cast<mlir::quant::UniformQuantizedType>();
    const auto scale = uniformQInElemType.getScale();
    // originQElemType = <u8:fp32, scale>
    // newQElemType = <u8:fp32, scale / 2>
    // Op -> originQElemType -> QuantizeCastOp -> newQElemType -> AddOp(output x2) -> result
    const auto newScale = static_cast<double>(scale / 2.0);
    const auto zeroPoint = uniformQInElemType.getZeroPoint();

    auto qType = inElemType.dyn_cast<mlir::quant::QuantizedType>();
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
// DequantizeToDwRewriter
//

class DequantizeToDwRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeToDwRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp DequantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DequantizeToDwRewriter::matchAndRewrite(IE::DequantizeOp originOp,
                                                            mlir::PatternRewriter& rewriter) const {
    const auto origType = originOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto origShape = origType.getShape();
    const auto OC = origShape[Dims4D::Act::C];
    const auto ctx = rewriter.getContext();

    const auto quantizeType = mlir::quant::UniformQuantizedType::get(
            /*flags=*/0, /*storageType=*/getUInt8Type(ctx), /*expressedType=*/mlir::Float16Type::get(ctx),
            /*scale=*/1.0, /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);
    auto quantWeightsOp = buildDwWeights(originOp->getLoc(), OC, quantizeType, rewriter);

    const auto attrStrides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto attrPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto attrPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto dilationsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            originOp, originOp.getOutput().getType(), originOp.getInput(), quantWeightsOp,
            /*bias=*/nullptr, attrStrides, attrPadsBegin, attrPadsEnd, dilationsAttr, getIntAttr(ctx, OC),
            /*post_opAttr=*/nullptr, /*clampAttr*/ nullptr, /*output_channels=*/nullptr, /*input_channels=*/nullptr);

    return mlir::success();
}

//
// QuantizeToAddRewriter
//

class QuantizeToAddRewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeToAddRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult QuantizeToAddRewriter::matchAndRewrite(IE::QuantizeOp originOp,
                                                           mlir::PatternRewriter& rewriter) const {
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);

    auto outElemType = originOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();
    auto uniformQOutElemType = outElemType.dyn_cast<mlir::quant::UniformQuantizedType>();
    const auto scale = uniformQOutElemType.getScale();
    // originQElemType = <u8:fp32, scale>
    // newQElemType = <u8:fp32, scale * 2>
    // Op -> AddOp(output x2) -> newQElemType -> QuantizeCastOp -> originQElemType -> result
    const auto newScale = static_cast<double>(scale * 2.0);
    const auto zeroPoint = uniformQOutElemType.getZeroPoint();

    auto qType = outElemType.dyn_cast<mlir::quant::QuantizedType>();
    auto quantizeElemType = mlir::quant::UniformQuantizedType::get(
            qType.getFlags(), qType.getStorageType(), qType.getExpressedType(), newScale, zeroPoint,
            qType.getStorageTypeMin(), qType.getStorageTypeMax());
    auto newAddOutType = mlir::RankedTensorType::get(originOp.getType().getShape(), quantizeElemType);

    auto addOp = rewriter.create<IE::AddOp>(originOp.getLoc(), newAddOutType, originOp.getInput(), originOp.getInput(),
                                            broadcastType, nullptr, nullptr, nullptr, nullptr);

    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(originOp, addOp.getResult(), outElemType);

    return mlir::success();
}

//
// QuantizeToDwRewriter
//

class QuantizeToDwRewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeToDwRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult QuantizeToDwRewriter::matchAndRewrite(IE::QuantizeOp originOp,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto origType = originOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto origShape = origType.getShape();
    const auto OC = origShape[Dims4D::Act::C];
    auto weights = buildDwWeights(originOp->getLoc(), OC, origType.getElementType(), rewriter);

    const auto ctx = rewriter.getContext();
    const auto attrStrides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto attrPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto attrPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto dilationsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            originOp, originOp.getOutput().getType(), originOp.getInput(), weights,
            /*bias=*/nullptr, attrStrides, attrPadsBegin, attrPadsEnd, dilationsAttr, getIntAttr(ctx, OC),
            /*post_opAttr=*/nullptr, /*clampAttr*/ nullptr, /*outputChannels=*/nullptr, /*input_channels=*/nullptr);

    return mlir::success();
}

//
// ConvertQuantizeOpsToNceOpsStrategy
//

bool isPerChannelQuantizedType(mlir::Value val) {
    auto elemType = val.getType().cast<vpux::NDTypeInterface>().getElementType();
    auto perAxisType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();
    if (perAxisType == nullptr) {
        return false;
    }

    auto rank = val.getType().cast<vpux::NDTypeInterface>().getRank();
    if (rank != 4) {
        return false;
    }

    auto quantizeDim = perAxisType.getQuantizedDimension();
    return quantizeDim == Dims4D::Act::C.ind();
}

void ConvertQuantizeOpsToNceOpsStrategy::prepareAvgPool(mlir::ConversionTarget& toAvgPoolTarget,
                                                        mlir::RewritePatternSet& toAvgPoolPatterns,
                                                        mlir::MLIRContext& ctx, Logger& log) {
    // HW Eltwise and AvgPool supports only per-tensor bias/scale parameters
    // perTensor quantize/dequantize convert to avgpool
    // E#98802 avgpool is faster than add for big input size.
    // avgpool support rank >= 3, and currently convert shape to 4D does not support avgpool, so here limit to rank = 4

    // Dummy AvgPool's numerical stability is preferred for low-precision floating-point quantization (FP8),
    // eltwise Add with self could shift values further away from 0.0 causing precision loss.

    toAvgPoolTarget.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
        auto inType = quantizeOp.getInput().getType().cast<vpux::NDTypeInterface>();
        auto inRank = inType.getRank();

        return isLegalQuantizeOp(quantizeOp, _canUseCMajor) || inRank != 4 ||
               (inType.getTotalAllocSize() <= vpux::VPU::getTotalCMXSize(quantizeOp) &&
                !vpux::isFloat8Quantized(quantizeOp.getDstElemType()));
    });
    toAvgPoolTarget.addDynamicallyLegalOp<IE::DequantizeOp>([&](IE::DequantizeOp dequantizeOp) {
        auto inType = dequantizeOp.getInput().getType().cast<vpux::NDTypeInterface>();
        auto inRank = inType.getRank();

        return isLegalDequantizeOp(dequantizeOp) || inRank != 4 ||
               (inType.getTotalAllocSize() <= vpux::VPU::getTotalCMXSize(dequantizeOp) &&
                !vpux::isFloat8Quantized(inType.getElementType())) ||
               hasConstProducer(dequantizeOp);
    });
    toAvgPoolTarget.addLegalOp<IE::AvgPoolOp>();

    toAvgPoolPatterns.add<IE::arch37xx::QuantizeDequantizeToAvgPool<IE::QuantizeOp>>(&ctx, log);
    toAvgPoolPatterns.add<IE::arch37xx::QuantizeDequantizeToAvgPool<IE::DequantizeOp>>(&ctx, log);
}

void ConvertQuantizeOpsToNceOpsStrategy::prepareEltwise(mlir::ConversionTarget& toEltwiseTarget,
                                                        mlir::RewritePatternSet& toEltwisePatterns,
                                                        mlir::MLIRContext& ctx, Logger& log) {
    toEltwiseTarget.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
        return IE::isLegalQuantizeOp(quantizeOp, _canUseCMajor);
    });
    toEltwiseTarget.addDynamicallyLegalOp<IE::DequantizeOp>([&](IE::DequantizeOp dequantizeOp) {
        return IE::isLegalDequantizeOp(dequantizeOp) || hasConstProducer(dequantizeOp);
    });

    toEltwiseTarget.addLegalOp<IE::AddOp>();
    toEltwiseTarget.addLegalOp<IE::QuantizeCastOp>();

    toEltwisePatterns.add<IE::arch37xx::DequantizeToAddRewriter>(&ctx, log);
    toEltwisePatterns.add<IE::arch37xx::QuantizeToAddRewriter>(&ctx, log);
}

void ConvertQuantizeOpsToNceOpsStrategy::prepareQuantToDw(mlir::ConversionTarget& quantToDwTarget,
                                                          mlir::RewritePatternSet& quantToDwPatterns,
                                                          mlir::MLIRContext& ctx, Logger& log) {
    quantToDwTarget.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
        const auto isPerChannelQuantized = isPerChannelQuantizedType(quantizeOp.getOutput());
        auto outputLayerUsers = quantizeOp.getOutput().getUsers();
        auto anyUserIsConv = !outputLayerUsers.empty() && ::llvm::any_of(outputLayerUsers, [](auto user) {
            return mlir::isa<IE::ConvolutionOp>(user);
        });

        return (anyUserIsConv && _canUseCMajor) || !isPerChannelQuantized;
    });

    quantToDwTarget.addDynamicallyLegalOp<IE::DequantizeOp>([&](IE::DequantizeOp dequantizeOp) {
        const auto isPerChannelQuantized = isPerChannelQuantizedType(dequantizeOp.getInput());
        auto outElemmentType = dequantizeOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();
        return !isPerChannelQuantized || !outElemmentType.isF16() || hasConstProducer(dequantizeOp);
    });

    quantToDwTarget.addLegalOp<Const::DeclareOp>();
    quantToDwTarget.addLegalOp<IE::GroupConvolutionOp>();

    quantToDwPatterns.add<QuantizeToDwRewriter>(&ctx, log);
    quantToDwPatterns.add<DequantizeToDwRewriter>(&ctx, log);
}

}  // namespace vpux::IE::arch37xx
