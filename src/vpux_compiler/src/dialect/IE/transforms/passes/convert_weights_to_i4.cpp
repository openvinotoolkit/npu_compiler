//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

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
    const auto dstElemType = mlir::cast<vpux::NDTypeInterface>(typeConverter->convertType(resultType)).getElementType();
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
            mlir::dyn_cast<mlir::quant::QuantizedType>(mlir::cast<vpux::NDTypeInterface>(outputType).getElementType());
    if (origQuantType == nullptr) {
        _log.trace("Unsupported element type");
        return mlir::failure();
    }

    const auto newType = mlir::cast<vpux::NDTypeInterface>(typeConverter->convertType(outputType));
    const auto newQuantType = mlir::cast<mlir::quant::QuantizedType>(newType.getElementType());

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
    if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(originQType)) {
        const auto high = uniformType.getStorageTypeMax();
        const auto offset = checked_cast<uint64_t>((high + 1) / 2);

        return mlir::quant::UniformQuantizedType::get(
                mlir::quant::QuantizationFlags::Signed, getSInt4Type(uniformType.getContext()),
                uniformType.getExpressedType(), uniformType.getScale(), uniformType.getZeroPoint() - offset,
                uniformType.getStorageTypeMin() - offset, uniformType.getStorageTypeMax() - offset);
    } else if (const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(originQType)) {
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

    const auto isQuantileStorage = [](mlir::quant::QuantizedType quantType) -> bool {
        return mlir::isa<vpux::type::QuantileType>(quantType.getStorageType());
    };

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([isQuantileStorage](vpux::NDTypeInterface tensor) {
        // Handle U4 only storage type with zero point of 8
        const auto elementType = tensor.getElementType();
        if (const auto uniformType = mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedType>(elementType)) {
            if (isQuantileStorage(uniformType)) {
                return tensor;
            }
            const uint64_t zeroPoint = uniformType.getZeroPoint();
            if (!uniformType.isSigned() && uniformType.getStorageTypeIntegralWidth() == 4 && zeroPoint == 8) {
                const auto newElemType = changeStorageTypeToI4(uniformType);
                return tensor.changeElemType(newElemType);
            }
        } else if (const auto perAxisType =
                           mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedPerAxisType>(elementType)) {
            if (isQuantileStorage(perAxisType)) {
                return tensor;
            }
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

    // Check if a Conv-like op has subbyte-capable activations (unsigned 8/16-bit quantized input 0)
    const auto inConvLikeOp = [](mlir::Operation* op) -> bool {
        if (!mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::TransposedConvolutionOp,
                       IE::GroupTransposedConvolutionOp, IE::MatMulOp>(op)) {
            return false;
        }
        auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
        const auto quantType = mlir::dyn_cast_if_present<mlir::quant::QuantizedType>(inputType.getElementType());
        if (quantType == nullptr) {
            return false;
        }
        const auto actBits = quantType.getStorageTypeIntegralWidth();
        return !quantType.isSigned() && (actBits == 8 || actBits == 16);
    };

    // Determine if a u4:zp=8 const should stay u4 (legal) instead of being converted to i4.
    // Const(quant<fp16:u4:scale,8>)
    //            |
    //   IE::ElemTypeInfoOpInterface* — if multi-input/output, keep u4 conservatively
    //            |
    //   Conv-like op && input 0 is unsigned quantized (subbyte-capable) → keep u4
    //   Otherwise → convert to i4
    const auto shouldKeepU4 = [&](Const::DeclareOp constOp) -> bool {
        for (auto user : constOp.getResult().getUsers()) {
            auto* op = user;
            while (mlir::isa<IE::ElemTypeInfoOpInterface>(op)) {
                if (op->getNumOperands() != 1 || op->getNumResults() != 1 || !op->getResult(0).hasOneUse()) {
                    return true;
                }
                op = *op->getResult(0).getUsers().begin();
            }
            if (!inConvLikeOp(op)) {
                return false;
            }
        }
        return true;
    };

    const auto isLegalConstDeclareOp = [&](Const::DeclareOp constOp) {
        const auto constTensor = constOp.getResult();
        const auto elementType = mlir::cast<vpux::NDTypeInterface>(constTensor.getType()).getElementType();
        if (const auto uniformType = mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedType>(elementType)) {
            if (isQuantileStorage(uniformType)) {
                return true;
            }
            const uint64_t zeroPoint = uniformType.getZeroPoint();
            if (!uniformType.isSigned() && uniformType.getStorageTypeIntegralWidth() == 4 && zeroPoint == 8) {
                return shouldKeepU4(constOp);
            }
        } else if (const auto perAxisType =
                           mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedPerAxisType>(elementType)) {
            if (isQuantileStorage(perAxisType)) {
                return true;
            }
            const auto zeroPoints = perAxisType.getZeroPoints();
            bool isAllEight = std::all_of(zeroPoints.begin(), zeroPoints.end(), [](int n) {
                return n == 8;
            });
            if (!perAxisType.isSigned() && perAxisType.getStorageTypeIntegralWidth() == 4 && isAllEight) {
                return shouldKeepU4(constOp);
            }
        }
        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<Const::DeclareOp>(isLegalConstDeclareOp);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (mlir::isa<IE::LayerOpInterface>(op)) {
            if (typeConverter.isLegal(op)) {
                return true;
            }
            // Allow Conv-like ops with subbyte-capable activations to keep u4 weights
            return inConvLikeOp(op);
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
