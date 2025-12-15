//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/impl/convert_to_palletization_lut_strategy.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/convert_to_palletization_lut.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux::IE::arch50xx {

// Returns false when type conversion is required;
bool isLegalTensorElem50XX(mlir::Type elementType) {
    return IE::isLegalTensorElemForPalletization(elementType, /*convertOnlyAsymmetric=*/false,
                                                 /*allowPerChannelZp=*/false);
}

// change quantized type to a quantile quantized type, with float quantile type and zp fully subtracted
mlir::quant::QuantizedType changeWeightTypeToLUT(mlir::quant::QuantizedType originWeightType, mlir::Type actType) {
    if (originWeightType == nullptr) {
        return nullptr;
    }

    const bool isActFp16 = actType.isF16();
    const bool isActFp8 = isFloat8Quantized(actType);
    if (!(isActFp16 || isActFp8)) {
        return nullptr;
    }

    const auto storageTypeInt = mlir::cast<mlir::IntegerType>(originWeightType.getStorageType());
    const auto bitWidth = storageTypeInt.getWidth();
    VPUX_THROW_UNLESS(bitWidth > 0 && bitWidth <= 4, "Unsupported bitWidth '{0}'", bitWidth);

    const auto getShiftedLUTUniformType = [isActFp16,
                                           bitWidth](mlir::quant::QuantizedType wgtType) -> SmallVector<double> {
        // The legality checks on the weight type ensure that in case of PerAxis quantized types, all the zp have
        // the same value, so we can just take the head zero point
        auto res = getSingleZeroPointOrFail(wgtType);
        VPUX_THROW_WHEN(mlir::failed(res), "Failed to get zero point for quantized type '{0}'", wgtType);

        const auto originalWeightStorageTypeMin =
                mlir::quant::QuantizedType::getDefaultMinimumForInteger(wgtType.isSigned(), bitWidth);
        const auto originalWeightStorageTypeMax =
                mlir::quant::QuantizedType::getDefaultMaximumForInteger(wgtType.isSigned(), bitWidth);
        const auto zeroPoint = res.value();
        const int lutElemNum = 1 << bitWidth;
        VPUX_THROW_WHEN(lutElemNum != (originalWeightStorageTypeMax - originalWeightStorageTypeMin + 1),
                        "Unexpected storage type range '{0}'",
                        originalWeightStorageTypeMax - originalWeightStorageTypeMin + 1);
        SmallVector<double> quantileLUT(lutElemNum);
        const int64_t bitMask = (1LL << bitWidth) - 1LL;

        // Interpreting integer weights with their unsigned encodings (for instance -7 i4 is 4'b1001,
        // which can be interpreted as 9 as u4 and used as an unsigned index to the LUT)
        if (isActFp16) {
            for (int64_t wgtVal = originalWeightStorageTypeMin; wgtVal <= originalWeightStorageTypeMax; ++wgtVal) {
                const unsigned unsignedTableIdx = static_cast<unsigned>(wgtVal & bitMask);
                quantileLUT[unsignedTableIdx] = static_cast<double>(wgtVal - zeroPoint);
            }
        } else {
            // fp8 activations and symmetric quantized weights
            // always use F8E4M3FN for the quantile type, regardless of fp8 activations type
            for (int64_t wgtVal = originalWeightStorageTypeMin; wgtVal <= originalWeightStorageTypeMax; ++wgtVal) {
                float lutElem = static_cast<float>(wgtVal - zeroPoint);
                const unsigned unsignedTableIdx = static_cast<unsigned>(wgtVal & bitMask);
                quantileLUT[unsignedTableIdx] = static_cast<float>(vpux::type::float8_e4m3(lutElem));
            }
        }

        return quantileLUT;
    };

    auto ctx = originWeightType.getContext();
    const auto newStorageIntegerType = mlir::IntegerType::get(ctx, bitWidth, mlir::IntegerType::Unsigned);
    constexpr unsigned newStorageIntegerTypeMin = 0;
    const unsigned newStorageIntegerTypeMax = (1 << bitWidth) - 1;
    // in case of fp8 activations default to use F8E4M3FN because it allows for better precision in the
    // representation of values around 0 (and all integers between -16 and +16 can be represented)
    // In both u4 and i4 wgt storageType, the range of possible values when subtracting (wgt - zp) goes from
    // -15 to +16 which is fully representable in F8E4M3FN format
    mlir::FloatType newQuantileType;
    if (isActFp16) {
        newQuantileType = mlir::Float16Type::get(ctx);
    } else {
        newQuantileType = mlir::Float8E4M3FNType::get(ctx);
    }

    if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(originWeightType)) {
        return mlir::quant::QuantileQuantizedType::get(0, newStorageIntegerType, newQuantileType,
                                                       uniformType.getExpressedType(),
                                                       getShiftedLUTUniformType(uniformType), uniformType.getScale(),
                                                       /*zp=*/0, newStorageIntegerTypeMin, newStorageIntegerTypeMax);

    } else if (const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(originWeightType)) {
        const SmallVector<int64_t> newZeroPoints(perAxisType.getZeroPoints().size(), 0);
        return mlir::quant::QuantileQuantizedPerAxisType::get(
                0, newStorageIntegerType, newQuantileType, perAxisType.getExpressedType(),
                getShiftedLUTUniformType(perAxisType), perAxisType.getScales(), newZeroPoints,
                perAxisType.getQuantizedDimension(), newStorageIntegerTypeMin, newStorageIntegerTypeMax);
    }

    VPUX_THROW("Unsupported Quantized Type '{0}'", originWeightType);
}

//
// ConvertToPalletizationLUTStrategy
//

void ConvertToPalletizationLUTStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();
    patterns.add<vpux::IE::ConvolutionLUTRewriter<IE::ConvolutionOp>>(ctx, isLegalTensorElem50XX, changeWeightTypeToLUT,
                                                                      log);
    patterns.add<vpux::IE::ConvolutionLUTRewriter<IE::GroupConvolutionOp>>(ctx, isLegalTensorElem50XX,
                                                                           changeWeightTypeToLUT, log);
}

bool isLegalOp(mlir::Operation* convOp) {
    // only fp16 and quant.uniform<f8E4M3FN:..., ...> or quant.uniform<f8E5M2:..., ...> are supported as activations
    // types for this pass
    auto inputDequantOp = mlir::dyn_cast_or_null<IE::DequantizeOp>(convOp->getOperand(0).getDefiningOp());
    const auto actType =
            inputDequantOp == nullptr
                    ? mlir::cast<vpux::NDTypeInterface>(convOp->getOperand(0).getType()).getElementType()
                    : mlir::cast<vpux::NDTypeInterface>(inputDequantOp.getInput().getType()).getElementType();

    const bool isActFp16 = actType.isF16();
    const bool isActFp8 = isFloat8Quantized(actType);

    if (!(isActFp16 || isActFp8)) {
        return true;
    }
    auto filterOp = convOp->getOperand(1).getDefiningOp<IE::DequantizeOp>();
    if (filterOp == nullptr) {
        return true;
    }

    return isLegalTensorElem50XX(mlir::cast<vpux::NDTypeInterface>(filterOp.getInput().getType()).getElementType());
}

void ConvertToPalletizationLUTStrategy::markOpLegality(mlir::ConversionTarget& target, Logger&) const {
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>(isLegalOp);
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::DequantizeOp>();
    target.addLegalOp<IE::QuantizeCastOp>();
}

}  // namespace vpux::IE::arch50xx
