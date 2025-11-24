//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <algorithm>
#include <cstdint>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTWEIGHTSTOI8
#define GEN_PASS_DEF_CONVERTWEIGHTSTOI8
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool isLegalTensor(vpux::NDTypeInterface tensorType, mlir::ModuleOp moduleOp, int64_t symmetricalZeroPoint = 128) {
    const auto isAsymmetricPerChannelZeroPointSupported = config::asymmetricPerChannelZeroPointSupported(moduleOp);
    const auto isAsymmetricPerTensorZeroPointSupported = config::asymmetricPerTensorZeroPointSupported(moduleOp);
    const auto isSymmetricalZeroPoint = [](mlir::Type quantizedType, const int64_t symmetricalZeroPoint) -> bool {
        if (const auto uniformQuantizedType =
                    mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(quantizedType)) {
            const int64_t zeroPoint = uniformQuantizedType.getZeroPoint();
            return zeroPoint == symmetricalZeroPoint;
        } else if (const auto perAxisQuantizedType =
                           mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(quantizedType)) {
            const auto zeroPoints = perAxisQuantizedType.getZeroPoints();
            return std::all_of(zeroPoints.begin(), zeroPoints.end(), [&](int n) -> bool {
                return n == symmetricalZeroPoint;
            });
        }
        return false;
    };

    const auto isQuantileQuantizedTypeLegal = [isSymmetricalZeroPoint](auto quantizedType,
                                                                       const int64_t symmetricalZeroPoint) -> bool {
        auto quantileTypeInt = mlir::dyn_cast<mlir::IntegerType>(quantizedType.getQuantileType());
        const bool isConversionRequired = quantileTypeInt && quantileTypeInt.isUnsigned() &&
                                          quantileTypeInt.getWidth() == 8 &&
                                          isSymmetricalZeroPoint(quantizedType, symmetricalZeroPoint);
        // mark tensor as illegal when conversion has to happen
        return !isConversionRequired;
    };

    const auto isUniformQuantizedTypeLegal = [isSymmetricalZeroPoint](auto quantizedType,
                                                                      const int64_t symmetricalZeroPoint,
                                                                      bool isAsymmetricZeroPointSupported) -> bool {
        const bool isConversionRequired =
                !quantizedType.isSigned() && quantizedType.getStorageTypeIntegralWidth() == 8 &&
                (isAsymmetricZeroPointSupported || isSymmetricalZeroPoint(quantizedType, symmetricalZeroPoint));
        return !isConversionRequired;
    };

    // only handle U8 type with zero point of 128
    const auto elementType = tensorType.getElementType();
    if (const auto uniformQuantileType = mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedType>(elementType)) {
        return isQuantileQuantizedTypeLegal(uniformQuantileType, symmetricalZeroPoint);

    } else if (const auto perAxisQuantileType =
                       mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedPerAxisType>(elementType)) {
        return isQuantileQuantizedTypeLegal(perAxisQuantileType, symmetricalZeroPoint);

    } else if (const auto uniformType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(elementType)) {
        return isUniformQuantizedTypeLegal(uniformType, symmetricalZeroPoint, isAsymmetricPerTensorZeroPointSupported);

    } else if (const auto perAxisType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(elementType)) {
        return isUniformQuantizedTypeLegal(perAxisType, symmetricalZeroPoint, isAsymmetricPerChannelZeroPointSupported);
    }
    return true;
};

//
// ConvolutionRewriter
//

class ConvolutionRewriter final : public mlir::OpConversionPattern<IE::ConvolutionOp> {
public:
    ConvolutionRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ConvolutionOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;
    Const::DeclareOp replaceConstDeclare(Const::DeclareOp origOp, mlir::ConversionPatternRewriter& rewriter) const;

private:
    Logger _log;
};

Const::DeclareOp ConvolutionRewriter::replaceConstDeclare(Const::DeclareOp origOp,
                                                          mlir::ConversionPatternRewriter& rewriter) const {
    const auto outputType = origOp.getType();
    const auto origQuantType =
            mlir::dyn_cast<mlir::quant::QuantizedType>(mlir::cast<vpux::NDTypeInterface>(outputType).getElementType());

    const auto newType = mlir::cast<vpux::NDTypeInterface>(typeConverter->convertType(outputType));

    const auto newQuantType = mlir::cast<mlir::quant::QuantizedType>(newType.getElementType());
    _log.nest().trace("Convert content from '{0}' to '{1}'", origQuantType, newQuantType);

    auto newContentAttr = origOp.getContentAttr().transform().convertElemType(newQuantType).get();
    auto newConstantOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), newType, std::move(newContentAttr));
    return newConstantOp;
}

mlir::LogicalResult ConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp, OpAdaptor,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());

    auto* typeConverter = this->getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter was not set");

    // Prior legality checks ensures all ops are defined and not null
    auto filterOp = origOp.getFilter().getDefiningOp<IE::DequantizeOp>();
    auto weightDeclareOp = filterOp.getInput().getDefiningOp<Const::DeclareOp>();
    auto newCstDeclareOp = replaceConstDeclare(weightDeclareOp, rewriter);

    auto newDequantizeOp = rewriter.create<IE::DequantizeOp>(origOp->getLoc(), newCstDeclareOp.getOutput(),
                                                             filterOp.getDstElemTypeAttr());
    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            origOp, origOp.getInput(), newDequantizeOp, origOp.getBias(), origOp.getStrides(), origOp.getPadsBegin(),
            origOp.getPadsEnd(), origOp.getDilations(), origOp.getPostOpAttr(), origOp.getClampAttr(),
            origOp.getStaticScaleAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
    return mlir::success();
}

//
// changeStorageTypeToI8
//

// change storage type to I8 and shift zp, min, max attributes by the value of storage type max
mlir::quant::QuantizedType changeStorageTypeToI8(mlir::quant::QuantizedType originQType) {
    const auto GetNewQuantileQuantizedType = [](auto quantileQuantizedType) -> mlir::quant::QuantizedType {
        // in QuantileQuantizedType we modify the type of the elements in the LUT from u8 to i8 while the
        // storageType and its limits remain the same
        constexpr unsigned u8BitWidth = 8;
        const unsigned quantileTypeMax = llvm::maxUIntN(u8BitWidth);
        auto offset = (quantileTypeMax + 1) / 2;
        SmallVector<double> quantileLUT(quantileQuantizedType.getQuantiles());
        for (auto& q : quantileLUT) {
            q -= offset;
        }

        if (const auto uniformQuantileType =
                    mlir::dyn_cast<mlir::quant::QuantileQuantizedType>(quantileQuantizedType)) {
            return mlir::quant::QuantileQuantizedType::get(
                    uniformQuantileType.getFlags(), uniformQuantileType.getStorageType(),
                    getSInt8Type(uniformQuantileType.getContext()), uniformQuantileType.getExpressedType(), quantileLUT,
                    uniformQuantileType.getScale(), /*zp=*/0, uniformQuantileType.getStorageTypeMin(),
                    uniformQuantileType.getStorageTypeMax());
        }

        const auto perAxisQuantileType =
                mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(quantileQuantizedType);
        VPUX_THROW_UNLESS(perAxisQuantileType != nullptr,
                          "quantizedType should be either a QuantileQuantizedType or a QuantileQuantizedPerAxisType!");

        const auto zeroPoints = perAxisQuantileType.getZeroPoints();
        const SmallVector<int64_t> newZeroPoints(zeroPoints.size(), 0);
        return mlir::quant::QuantileQuantizedPerAxisType::get(
                perAxisQuantileType.getFlags(), perAxisQuantileType.getStorageType(),
                getSInt8Type(perAxisQuantileType.getContext()), perAxisQuantileType.getExpressedType(), quantileLUT,
                perAxisQuantileType.getScales(), newZeroPoints, perAxisQuantileType.getQuantizedDimension(),
                perAxisQuantileType.getStorageTypeMin(), perAxisQuantileType.getStorageTypeMax());
    };

    if (const auto uniformQuantileType = mlir::dyn_cast<mlir::quant::QuantileQuantizedType>(originQType)) {
        return GetNewQuantileQuantizedType(uniformQuantileType);

    } else if (const auto perAxisQuantileType =
                       mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(originQType)) {
        return GetNewQuantileQuantizedType(perAxisQuantileType);

    } else if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(originQType)) {
        auto offset = (uniformType.getStorageTypeMax() + 1) / 2;
        auto zeroPoint = uniformType.getZeroPoint();
        auto newZeroPoint = zeroPoint - offset;
        return mlir::quant::UniformQuantizedType::get(
                mlir::quant::QuantizationFlags::Signed, getSInt8Type(uniformType.getContext()),
                uniformType.getExpressedType(), uniformType.getScale(),
                /*zp=*/newZeroPoint, /*min=*/uniformType.getStorageTypeMin() - offset,
                /*max=*/uniformType.getStorageTypeMax() - offset);

    } else if (const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(originQType)) {
        const auto zeroPoints = perAxisType.getZeroPoints();

        SmallVector<int64_t> newZeroPoints(zeroPoints.size(), 0);
        auto offset = (perAxisType.getStorageTypeMax() + 1) / 2;
        std::transform(zeroPoints.begin(), zeroPoints.end(), newZeroPoints.begin(), [&](int64_t oldZP) {
            return oldZP - offset;
        });
        return mlir::quant::UniformQuantizedPerAxisType::get(mlir::quant::QuantizationFlags::Signed,
                                                             getSInt8Type(perAxisType.getContext()),
                                                             perAxisType.getExpressedType(), perAxisType.getScales(),
                                                             newZeroPoints, perAxisType.getQuantizedDimension(),
                                                             /*min=*/perAxisType.getStorageTypeMin() - offset,
                                                             /*max=*/perAxisType.getStorageTypeMax() - offset);
    }

    VPUX_THROW("Unsupported Quantized Type '{0}'", originQType);
}  // namespace

//
// ConvertWeightsToI8Pass
//

class ConvertWeightsToI8Pass final : public IE::impl::ConvertWeightsToI8Base<ConvertWeightsToI8Pass> {
public:
    explicit ConvertWeightsToI8Pass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertWeightsToI8Pass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::TypeConverter typeConverter;
    auto moduleOp = getModuleOp(func);
    typeConverter.addConversion([&](vpux::NDTypeInterface tensor) {
        auto quantType = mlir::dyn_cast_or_null<mlir::quant::QuantizedType>(tensor.getElementType());
        if (!isLegalTensor(tensor, moduleOp)) {
            const auto newElemType = changeStorageTypeToI8(quantType);
            return tensor.changeElemType(newElemType);
        }

        return tensor;
    });
    typeConverter.addSourceMaterialization(dummyConverter<mlir::RankedTensorType>);
    typeConverter.addTargetMaterialization(dummyConverter<mlir::RankedTensorType>);
    typeConverter.addArgumentMaterialization(dummyConverter<mlir::RankedTensorType>);

    mlir::ConversionTarget target(ctx);

    // We can't convert any operations that have operands with symmetric and asymmetric zero points, i.e.: IE::Add
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) {
        auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getElementType();
        // Input should be F16 and should not be FQ
        if (!inputType.isF16()) {
            return true;
        }
        auto inputOp = op.getInput().getDefiningOp();
        if (inputOp != nullptr && mlir::isa<IE::FakeQuantizeOp, IE::DequantizeOp>(inputOp)) {
            return true;
        }
        auto filterOp = op.getFilter().getDefiningOp<IE::DequantizeOp>();
        if (filterOp == nullptr) {
            return true;
        }
        auto weightDeclareOp = filterOp.getInput().getDefiningOp<Const::DeclareOp>();
        if (weightDeclareOp == nullptr) {
            return true;
        }

        return isLegalTensor(mlir::cast<vpux::NDTypeInterface>(weightDeclareOp.getOutput().getType()), moduleOp);
    });
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::DequantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvolutionRewriter>(typeConverter, &ctx, _log);

    if (mlir::failed(applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createConvertWeightsToI8Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertWeightsToI8Pass(Logger log) {
    return std::make_unique<ConvertWeightsToI8Pass>(log);
}
