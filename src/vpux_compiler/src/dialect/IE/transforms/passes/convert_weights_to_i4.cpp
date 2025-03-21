//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTWEIGHTSTOI4
#define GEN_PASS_DEF_CONVERTWEIGHTSTOI4
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// LayerRewriter
//

class LayerRewriter final : public mlir::OpInterfaceConversionPattern<IE::LayerOpInterface> {
public:
    LayerRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceConversionPattern<IE::LayerOpInterface>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayerOpInterface origOp, ArrayRef<mlir::Value> newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LayerRewriter::matchAndRewrite(IE::LayerOpInterface origOp, ArrayRef<mlir::Value> newOperands,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());

    const auto* typeConverter = this->getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter was not set");

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == newOperands.size(), "Wrong operands size : {0}", newOperands.size());

    if (mlir::isa<IE::QuantizeOp, IE::QuantizeCastOp>(origOp.getOperation())) {
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    mapper.map(origOperands, newOperands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto result : newOp->getResults()) {
        result.setType(typeConverter->convertType(result.getType()));
    }

    rewriter.replaceOp(origOp, newOp->getResults());
    return mlir::success();
}

//
// QuantizeLikeOpRewriter
//

template <class QuantizeLikeOp>
class QuantizeLikeOpRewriter final : public mlir::OpConversionPattern<QuantizeLikeOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<QuantizeLikeOp>::OpAdaptor;

public:
    QuantizeLikeOpRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<QuantizeLikeOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(QuantizeLikeOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class QuantizeLikeOp>
mlir::LogicalResult QuantizeLikeOpRewriter<QuantizeLikeOp>::matchAndRewrite(
        QuantizeLikeOp origOp, OpAdaptor newArgs, mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());

    auto* typeConverter = this->getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter was not set");

    auto resultType = origOp->getResult(0).getType();
    const auto dstElemType =
            typeConverter->convertType(resultType).template cast<vpux::NDTypeInterface>().getElementType();
    rewriter.replaceOpWithNewOp<QuantizeLikeOp>(origOp, newArgs.getInput(), dstElemType);
    return mlir::success();
}

//
// ConstRewriter
//

class ConstRewriter final : public mlir::OpConversionPattern<Const::DeclareOp> {
public:
    ConstRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<Const::DeclareOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConstRewriter::matchAndRewrite(Const::DeclareOp origOp, OpAdaptor,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());

    const auto* typeConverter = this->getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter was not set");

    const auto outputType = origOp.getType();
    const auto origQuantType =
            outputType.cast<vpux::NDTypeInterface>().getElementType().dyn_cast<mlir::quant::QuantizedType>();
    if (origQuantType == nullptr) {
        _log.trace("Unsupported element type");
        return mlir::failure();
    }

    const auto newType = typeConverter->convertType(outputType).cast<vpux::NDTypeInterface>();
    const auto newQuantType = newType.getElementType().cast<mlir::quant::QuantizedType>();

    _log.nest().trace("Convert content from '{0}' to '{1}'", origQuantType, newQuantType);

    auto newContentAttr = origOp.getContentAttr().transform().convertElemType(newQuantType).get();

    rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, newType, std::move(newContentAttr));
    return mlir::success();
}

//
// changeStorageTypeToI4
//

// change storage type to I4 and shift zp, min, max attributes by the value of storage type max
mlir::quant::QuantizedType changeStorageTypeToI4(mlir::quant::QuantizedType originQType) {
    if (const auto uniformType = originQType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        const auto high = uniformType.getStorageTypeMax();
        const auto offset = checked_cast<uint64_t>((high + 1) / 2);

        return mlir::quant::UniformQuantizedType::get(
                mlir::quant::QuantizationFlags::Signed, getSInt4Type(uniformType.getContext()),
                uniformType.getExpressedType(), uniformType.getScale(), uniformType.getZeroPoint() - offset,
                uniformType.getStorageTypeMin() - offset, uniformType.getStorageTypeMax() - offset);
    } else if (const auto perAxisType = originQType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto high = perAxisType.getStorageTypeMax();
        const auto offset = checked_cast<uint64_t>((high + 1) / 2);
        const auto zeroPoints = perAxisType.getZeroPoints();

        SmallVector<int64_t> newZeroPoints(zeroPoints.size());
        std::transform(zeroPoints.begin(), zeroPoints.end(), newZeroPoints.begin(), [offset](int64_t zp) {
            return zp - offset;
        });

        return mlir::quant::UniformQuantizedPerAxisType::get(
                mlir::quant::QuantizationFlags::Signed, getSInt4Type(perAxisType.getContext()),
                perAxisType.getExpressedType(), perAxisType.getScales(), newZeroPoints,
                perAxisType.getQuantizedDimension(), perAxisType.getStorageTypeMin() - offset,
                perAxisType.getStorageTypeMax() - offset);
    }

    VPUX_THROW("Unsupported Quantized Type '{0}'", originQType);
}

//
// ConvertWeightsToI4Pass
//

class ConvertWeightsToI4Pass final : public IE::impl::ConvertWeightsToI4Base<ConvertWeightsToI4Pass> {
public:
    explicit ConvertWeightsToI4Pass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertWeightsToI4Pass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](vpux::NDTypeInterface tensor) {
        // Handle U4 only storage type with zero point of 8
        const auto elementType = tensor.getElementType();
        if (mlir::isa_and_nonnull<mlir::quant::QuantileQuantizedType, mlir::quant::QuantileQuantizedPerAxisType>(
                    elementType)) {
            // in QuantileQuantizedType the u4 storage type remains unchanged while it's the type in the lut
            // (quantileType) that should be updated, but u4 is not actually supported as a palletization table type, so
            // nothing to be done
            return tensor;
        } else if (const auto uniformType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(elementType)) {
            const uint64_t zeroPoint = uniformType.getZeroPoint();
            if (!uniformType.isSigned() && uniformType.getStorageTypeIntegralWidth() == 4 && zeroPoint == 8) {
                const auto newElemType = changeStorageTypeToI4(uniformType);
                return tensor.changeElemType(newElemType);
            }
        } else if (const auto perAxisType =
                           mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(elementType)) {
            const auto zeroPoints = perAxisType.getZeroPoints();
            bool isAllEight = std::all_of(zeroPoints.begin(), zeroPoints.end(), [](int n) {
                return n == 8;
            });
            if (!perAxisType.isSigned() && perAxisType.getStorageTypeIntegralWidth() == 4 && isAllEight) {
                const auto newElemType = changeStorageTypeToI4(perAxisType);
                return tensor.changeElemType(newElemType);
            }
        }

        return tensor;
    });
    typeConverter.addSourceMaterialization(dummyConverter<mlir::RankedTensorType>);
    typeConverter.addTargetMaterialization(dummyConverter<mlir::RankedTensorType>);
    typeConverter.addArgumentMaterialization(dummyConverter<mlir::RankedTensorType>);

    const auto isLegalConstDeclareOp = [&](Const::DeclareOp constOp) {
        // only handle U4 type with zero point of 8
        const auto constTensor = constOp.getResult();
        const auto elementType = constTensor.getType().cast<vpux::NDTypeInterface>().getElementType();
        if (mlir::isa_and_nonnull<mlir::quant::QuantileQuantizedType, mlir::quant::QuantileQuantizedPerAxisType>(
                    elementType)) {
            // in QuantileQuantizedType the u4 storage type remains unchanged while it's the type in the lut
            // (quantileType) that should be updated, but u4 is not actually supported as a palletization table type, so
            // nothing to be done (declare as legal)
            return true;
        } else if (const auto uniformType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(elementType)) {
            const uint64_t zeroPoint = uniformType.getZeroPoint();
            if (!uniformType.isSigned() && uniformType.getStorageTypeIntegralWidth() == 4 && zeroPoint == 8) {
                return false;
            }
        } else if (const auto perAxisType =
                           mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(elementType)) {
            const auto zeroPoints = perAxisType.getZeroPoints();
            bool isAllEight = std::all_of(zeroPoints.begin(), zeroPoints.end(), [](int n) {
                return n == 8;
            });
            if (!perAxisType.isSigned() && perAxisType.getStorageTypeIntegralWidth() == 4 && isAllEight) {
                return false;
            }
        }
        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<Const::DeclareOp>(isLegalConstDeclareOp);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (mlir::isa<IE::LayerOpInterface>(op)) {
            return typeConverter.isLegal(op);
        }
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConstRewriter>(typeConverter, &ctx, _log);
    patterns.add<QuantizeLikeOpRewriter<IE::QuantizeOp>>(typeConverter, &ctx, _log);
    patterns.add<QuantizeLikeOpRewriter<IE::QuantizeCastOp>>(typeConverter, &ctx, _log);
    patterns.add<LayerRewriter>(typeConverter, &ctx, _log);

    if (mlir::failed(applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createConvertWeightsToI4Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertWeightsToI4Pass(Logger log) {
    return std::make_unique<ConvertWeightsToI4Pass>(log);
}
