//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/IE/impl/convert_to_palletization_lut_strategy.hpp"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/convert_to_palletization_lut.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux::IE::arch40xx {

// Returns false when type conversion is required.
bool isLegalTensorElem40XX(mlir::Type elementType) {
    return IE::isLegalTensorElemForPalletization(elementType, /*convertOnlyAsymmetric=*/false,
                                                 /*allowPerChannelZp=*/false);
}

// change quantized type to a quantile quantized type, with float/integer quantile type and zp fully subtracted
mlir::quant::QuantizedType changeWeightTypeToLUT(mlir::quant::QuantizedType originWeightType, mlir::Type actType) {
    if (originWeightType == nullptr) {
        return nullptr;
    }

    const bool isActFp16 = actType.isF16();
    const bool isActInt8 = isInt8Quantized(actType);
    if (!(isActFp16 || isActInt8)) {
        return nullptr;
    }

    const auto storageTypeInt = mlir::cast<mlir::IntegerType>(originWeightType.getStorageType());
    const auto bitWidth = storageTypeInt.getWidth();
    VPUX_THROW_UNLESS(bitWidth > 0 && bitWidth <= 4, "Unsupported bitWidth '{0}'", bitWidth);

    const auto getShiftedLUTUniformType = [bitWidth](mlir::quant::QuantizedType wgtType) -> SmallVector<double> {
        // The legality checks on the weight type ensure that in case of PerAxis quantized types, all the zp have the
        // same value, so we can just take the head zero point
        auto res = getSingleZeroPointOrFail(wgtType);
        VPUX_THROW_WHEN(mlir::failed(res), "Failed to get zero point for quantized type '{0}'", wgtType);

        const auto originalStorageTypeMin =
                mlir::quant::QuantizedType::getDefaultMinimumForInteger(wgtType.isSigned(), bitWidth);
        const auto originalStorageTypeMax =
                mlir::quant::QuantizedType::getDefaultMaximumForInteger(wgtType.isSigned(), bitWidth);
        const auto zeroPoint = res.value();
        const int lutElemNum = 1 << bitWidth;
        VPUX_THROW_WHEN(lutElemNum != (originalStorageTypeMax - originalStorageTypeMin + 1),
                        "Unexpected storage type range '{0}'", originalStorageTypeMax - originalStorageTypeMin + 1);
        SmallVector<double> quantileLUT(lutElemNum);
        const int64_t bitMask = (1LL << bitWidth) - 1LL;

        for (int64_t wgtVal = originalStorageTypeMin; wgtVal <= originalStorageTypeMax; ++wgtVal) {
            // Interpreting integer weights with their unsigned encodings (for instance -7 i4 is 4'b1001,
            // which can be interpreted as 9 as u4 and used as an unsigned index to the LUT)
            const unsigned unsignedTableIdx = static_cast<unsigned>(wgtVal & bitMask);
            quantileLUT[unsignedTableIdx] = static_cast<double>(wgtVal - zeroPoint);
        }

        return quantileLUT;
    };

    auto ctx = originWeightType.getContext();
    const auto newStorageIntegerType = mlir::IntegerType::get(ctx, bitWidth, mlir::IntegerType::Unsigned);
    constexpr unsigned newStorageIntegerTypeMin = 0;
    const unsigned newStorageIntegerTypeMax = (1 << bitWidth) - 1;
    mlir::Type newQuantileType;
    if (isActFp16) {
        newQuantileType = mlir::Float16Type::get(ctx);
    } else if (isActInt8) {
        newQuantileType = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
    } else {
        VPUX_THROW("Unsupported activation type '{0}'", actType);
    }

    if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(originWeightType)) {
        const auto newStorageQuantileType = mlir::quant::QuantileType::get(ctx, newStorageIntegerType, newQuantileType,
                                                                           getShiftedLUTUniformType(uniformType));

        return mlir::quant::UniformQuantizedType::get(0, newStorageQuantileType, uniformType.getExpressedType(),
                                                      uniformType.getScale(), 0, newStorageIntegerTypeMin,
                                                      newStorageIntegerTypeMax);

    } else if (const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(originWeightType)) {
        const auto newStorageQuantileType = mlir::quant::QuantileType::get(ctx, newStorageIntegerType, newQuantileType,
                                                                           getShiftedLUTUniformType(perAxisType));
        const SmallVector<int64_t> newZeroPoints(perAxisType.getZeroPoints().size(), 0);
        return mlir::quant::UniformQuantizedPerAxisType::get(
                0, newStorageQuantileType, perAxisType.getExpressedType(), perAxisType.getScales(), newZeroPoints,
                perAxisType.getQuantizedDimension(), newStorageIntegerTypeMin, newStorageIntegerTypeMax);
    }

    VPUX_THROW("Unsupported Quantized Type '{0}'", originWeightType);
}

//
// ConvertToPalletizationLUTStrategy
//

void ConvertToPalletizationLUTStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();
    patterns.add<vpux::IE::ConvolutionLUTRewriter<IE::ConvolutionOp>>(ctx, isLegalTensorElem40XX, changeWeightTypeToLUT,
                                                                      log);
    patterns.add<vpux::IE::ConvolutionLUTRewriter<IE::GroupConvolutionOp>>(ctx, isLegalTensorElem40XX,
                                                                           changeWeightTypeToLUT, log);
    patterns.add<vpux::IE::ConvolutionLUTRewriter<IE::MatMulOp>>(ctx, isLegalTensorElem40XX, changeWeightTypeToLUT,
                                                                 log);
}

bool isLegalOp(mlir::Operation* op) {
    // only fp16 and quant.uniform<i8:..., ...> are supported as activations types for this pass
    auto inputDequantOp = mlir::dyn_cast_or_null<IE::DequantizeOp>(op->getOperand(0).getDefiningOp());
    const auto actType =
            inputDequantOp == nullptr
                    ? mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType()
                    : mlir::cast<vpux::NDTypeInterface>(inputDequantOp.getInput().getType()).getElementType();

    const bool isActFp16 = actType.isF16();
    const bool isActInt8 = isInt8Quantized(actType);

    if (!(isActFp16 || isActInt8)) {
        return true;
    }

    // For fp16 activations, FakeQuantize/Dequantize on the input path indicates the activation will be quantized
    // by a later pass; only int8 activations may legitimately appear behind a Dequantize at this stage.
    if (isActFp16) {
        auto inputOp = op->getOperand(0).getDefiningOp();
        if (mlir::isa_and_nonnull<IE::FakeQuantizeOp, IE::DequantizeOp>(inputOp)) {
            return true;
        }
    }

    auto filterDequantOp = op->getOperand(1).getDefiningOp<IE::DequantizeOp>();
    const auto weightType = mlir::cast<vpux::NDTypeInterface>(
            filterDequantOp != nullptr ? filterDequantOp.getInput().getType() : op->getOperand(1).getType());
    const auto weightElemType = weightType.getElementType();

    if (mlir::isa<mlir::quant::QuantizedType>(weightElemType)) {
        return isLegalTensorElem40XX(weightElemType);
    }

    return true;
}

void ConvertToPalletizationLUTStrategy::markOpLegality(mlir::ConversionTarget& target, Logger&) const {
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::MatMulOp>(isLegalOp);
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::DequantizeOp>();
    target.addLegalOp<IE::QuantizeCastOp>();
}

}  // namespace vpux::IE::arch40xx
