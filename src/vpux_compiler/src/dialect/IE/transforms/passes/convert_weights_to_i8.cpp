//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>

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

    const auto isQuantizedTypeLegal = [isSymmetricalZeroPoint](auto quantizedType, const int64_t symmetricalZeroPoint,
                                                               bool isAsymmetricZeroPointSupported) -> bool {
        bool isConversionRequired = false;

        if (const auto quantileStorage = mlir::dyn_cast<vpux::type::QuantileType>(quantizedType.getStorageType())) {
            const auto quantileTypeInt = mlir::dyn_cast<mlir::IntegerType>(quantileStorage.getQuantileType());
            isConversionRequired = quantileTypeInt && quantileTypeInt.isUnsigned() && quantileTypeInt.getWidth() == 8 &&
                                   isSymmetricalZeroPoint(quantizedType, symmetricalZeroPoint);
        } else {
            isConversionRequired =
                    !quantizedType.isSigned() && quantizedType.getStorageTypeIntegralWidth() == 8 &&
                    (isAsymmetricZeroPointSupported || isSymmetricalZeroPoint(quantizedType, symmetricalZeroPoint));
        }
        return !isConversionRequired;
    };

    // only handle U8 type with zero point of 128
    const auto elementType = tensorType.getElementType();
    if (const auto uniformType = mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedType>(elementType)) {
        return isQuantizedTypeLegal(uniformType, symmetricalZeroPoint, isAsymmetricPerTensorZeroPointSupported);
    }
    if (const auto perAxisType = mlir::dyn_cast_if_present<mlir::quant::UniformQuantizedPerAxisType>(elementType)) {
        return isQuantizedTypeLegal(perAxisType, symmetricalZeroPoint, isAsymmetricPerChannelZeroPointSupported);
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
    auto newConstantOp =
            rewriter.create<Const::DeclareOp>(takeOpLoc(origOp, "as_i8"), newType, std::move(newContentAttr));
    return newConstantOp;
}

mlir::LogicalResult ConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp, OpAdaptor,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());

    auto* typeConverter = this->getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter was not set");

    // Prior legality checks ensures all ops are defined and not null
    auto dequantizeOp = origOp.getFilter().getDefiningOp<IE::DequantizeOp>();
    IE::ConvolutionOp newConvOp = nullptr;
    if (dequantizeOp != nullptr) {
        auto weightDeclareOp = dequantizeOp.getInput().getDefiningOp<Const::DeclareOp>();
        auto newCstDeclareOp = replaceConstDeclare(weightDeclareOp, rewriter);
        auto newDequantizeOp = rewriter.create<IE::DequantizeOp>(
                takeOpLoc(origOp, "dequantize_filter"), newCstDeclareOp.getOutput(), dequantizeOp.getDstElemTypeAttr());
        newConvOp = cloneConvolutionOp(rewriter, origOp, origOp.getInput(), newDequantizeOp);
    } else {
        auto weightDeclareOp = origOp.getFilter().getDefiningOp<Const::DeclareOp>();
        auto newCstDeclareOp = replaceConstDeclare(weightDeclareOp, rewriter);
        newConvOp = cloneConvolutionOp(rewriter, origOp, origOp.getInput(), newCstDeclareOp.getOutput());
    }
    auto newConvOutType = mlir::cast<NDTypeInterface>(newConvOp.getOutput().getType());
    auto newConvOutElemType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType()).getElementType();
    newConvOutType = newConvOutType.changeElemType(newConvOutElemType);
    newConvOp->getResult(0).setType(newConvOutType);
    rewriter.replaceOp(origOp, newConvOp);

    return mlir::success();
}

//
// changeStorageTypeToI8
//

// change storage type to I8 and shift zp, min, max attributes by the value of storage type max
mlir::quant::QuantizedType changeStorageTypeToI8(mlir::quant::QuantizedType originQType) {
    const auto changeQuantileStorageToI8 = [](auto uniformType) -> mlir::quant::QuantizedType {
        // in QuantileType we modify the type of the elements in the LUT from u8 to i8 while the
        // storageType and its limits remain the same
        const auto quantileStorage = mlir::dyn_cast<vpux::type::QuantileType>(uniformType.getStorageType());

        constexpr unsigned u8BitWidth = 8;
        const unsigned quantileTypeMax = llvm::maxUIntN(u8BitWidth);
        auto offset = (quantileTypeMax + 1) / 2;
        SmallVector<double> quantileLUT(quantileStorage.getQuantiles());
        for (auto& q : quantileLUT) {
            q -= offset;
        }

        const auto newQuantileStorageType =
                vpux::type::QuantileType::get(quantileStorage.getContext(), quantileStorage.getStorageType(),
                                              getSInt8Type(quantileStorage.getContext()), quantileLUT);

        if (const auto uniformQuantileType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(uniformType)) {
            return mlir::quant::UniformQuantizedType::get(
                    uniformType.getFlags(), newQuantileStorageType, uniformQuantileType.getExpressedType(),
                    uniformQuantileType.getScale(), /*zp=*/0, uniformQuantileType.getStorageTypeMin(),
                    uniformQuantileType.getStorageTypeMax());
        } else if (const auto perAxisQuantileType =
                           mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(uniformType)) {
            const auto zeroPoints = perAxisQuantileType.getZeroPoints();
            const SmallVector<int64_t> newZeroPoints(zeroPoints.size(), 0);
            return mlir::quant::UniformQuantizedPerAxisType::get(
                    perAxisQuantileType.getFlags(), newQuantileStorageType, perAxisQuantileType.getExpressedType(),
                    perAxisQuantileType.getScales(), newZeroPoints, perAxisQuantileType.getQuantizedDimension(),
                    perAxisQuantileType.getStorageTypeMin(), perAxisQuantileType.getStorageTypeMax());
        }
        VPUX_THROW("Unsupported Quantized Type '{0}'", uniformType);
    };

    if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(originQType)) {
        if (mlir::isa<vpux::type::QuantileType>(uniformType.getStorageType())) {
            return changeQuantileStorageToI8(uniformType);
        }
        const auto offset = (uniformType.getStorageTypeMax() + 1) / 2;
        return mlir::quant::UniformQuantizedType::get(
                mlir::quant::QuantizationFlags::Signed, getSInt8Type(uniformType.getContext()),
                uniformType.getExpressedType(), uniformType.getScale(), uniformType.getZeroPoint() - offset,
                uniformType.getStorageTypeMin() - offset, uniformType.getStorageTypeMax() - offset);
    }

    if (const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(originQType)) {
        if (mlir::isa<vpux::type::QuantileType>(perAxisType.getStorageType())) {
            return changeQuantileStorageToI8(perAxisType);
        }
        const auto offset = (perAxisType.getStorageTypeMax() + 1) / 2;
        SmallVector<int64_t> newZeroPoints;
        llvm::transform(perAxisType.getZeroPoints(), std::back_inserter(newZeroPoints), [offset](int64_t zp) {
            return zp - offset;
        });
        return mlir::quant::UniformQuantizedPerAxisType::get(
                mlir::quant::QuantizationFlags::Signed, getSInt8Type(perAxisType.getContext()),
                perAxisType.getExpressedType(), perAxisType.getScales(), newZeroPoints,
                perAxisType.getQuantizedDimension(), perAxisType.getStorageTypeMin() - offset,
                perAxisType.getStorageTypeMax() - offset);
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

    mlir::ConversionTarget target(ctx);

    // We can't convert any operations that have operands with symmetric and asymmetric zero points, i.e.: IE::Add
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) {
        if (!op.getInput() || !op.getFilter()) {
            return true;
        }

        auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getElementType();
        // Input should be F16 and should not be FQ
        if (!inputType.isF16()) {
            return true;
        }

        // Input cannot be FakeQuantize or Dequantize
        if (mlir::matchPattern(op.getInput(), mlir::m_Op<IE::FakeQuantizeOp>()) ||
            mlir::matchPattern(op.getInput(), mlir::m_Op<IE::DequantizeOp>())) {
            return true;
        }

        auto filterOp = op.getFilter();
        Const::DeclareOp weightDeclareOp = nullptr;

        auto isEligibleQuantType = [](Const::DeclareOp constantOp) -> bool {
            if (constantOp == nullptr) {
                return false;
            }
            const auto outputType = mlir::cast<vpux::NDTypeInterface>(constantOp.getOutput().getType());
            const auto outputElemType = outputType.getElementType();
            if (auto outputQType = mlir::dyn_cast<mlir::quant::QuantizedType>(outputElemType)) {
                if (const auto quantileStorage =
                            mlir::dyn_cast<vpux::type::QuantileType>(outputQType.getStorageType())) {
                    auto outputQuantileType = quantileStorage.getQuantileType();
                    if (!outputQuantileType.isUnsignedInteger(8) && !outputQuantileType.isSignlessInteger(8)) {
                        return false;
                    }
                } else {
                    auto outputStorageType = outputQType.getStorageType();
                    if (!outputStorageType.isUnsignedInteger(8) && !outputStorageType.isSignlessInteger(8)) {
                        return false;
                    }
                }
            }
            return true;
        };

        if (mlir::matchPattern(filterOp, mlir::m_Op<Const::DeclareOp>())) {
            auto constantOp = filterOp.getDefiningOp<Const::DeclareOp>();
            if (!isEligibleQuantType(constantOp)) {
                return true;
            }
            weightDeclareOp = constantOp;
        } else if (mlir::matchPattern(filterOp, mlir::m_Op<IE::DequantizeOp>(mlir::m_Op<Const::DeclareOp>()))) {
            auto dequantOp = filterOp.getDefiningOp<IE::DequantizeOp>();
            if (dequantOp == nullptr) {
                return true;
            }

            auto innerConstOp = dequantOp.getInput().getDefiningOp<Const::DeclareOp>();
            if (!isEligibleQuantType(innerConstOp)) {
                return true;
            }
            weightDeclareOp = innerConstOp;
        } else {
            // Fallback case
            return true;
        }

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
