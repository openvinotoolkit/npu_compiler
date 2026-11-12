//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/IE/impl/convert_to_palletization_lut_strategy.hpp"
#include <mlir/Support/LLVM.h>
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/convert_to_palletization_lut.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/core/type/float4_e2m1.hpp"

namespace vpux::IE::arch60xx {

// Returns false when type conversion is required;
bool isLegalTensorElem60XX(mlir::Type elementType) {
    // FP4 storage requires palletization to a float LUT.
    if (const auto quantized = mlir::dyn_cast<mlir::quant::QuantizedType>(elementType)) {
        if (mlir::isa<mlir::Float4E2M1FNType>(quantized.getStorageType())) {
            return false;
        }
    }
    return IE::isLegalTensorElemForPalletization(elementType, /*convertOnlyAsymmetric=*/false,
                                                 /*allowPerChannelZp=*/true);
}

// Converts a quantized weight type to a QuantileType backed by a pre-computed LUT.
// Supported quantile types: int8, f16, F8E4M3FN.
// Zero points are baked into LUT entries except in the fp16 per-axis case, where the IDU
// subtracts the zero point before the upscaling function.
mlir::quant::QuantizedType changeWeightTypeToLUT(mlir::quant::QuantizedType originWeightType, mlir::Type actType) {
    if (originWeightType == nullptr) {
        return nullptr;
    }

    const bool isActFp16 = actType.isF16();
    const bool isActFp8 = isFloat8Quantized(actType);
    const bool isActInt8 = isInt8Quantized(actType);
    if (!(isActFp16 || isActFp8 || isActInt8)) {
        return nullptr;
    }

    auto ctx = originWeightType.getContext();

    const bool isFp4Storage = mlir::isa<mlir::Float4E2M1FNType>(originWeightType.getStorageType());
    // FP4 weights are palletized through an FP8 LUT; the LUT entries are encoded as F8E4M3 regardless of
    // whether the activation path is FP16 or FP8. Other activation types are not supported for FP4 palletization.
    // f4E2M1FN is a symmetric float format; non-zero zero points have no quantization semantics.
    if (isFp4Storage) {
        if (!(isActFp8 || isActFp16)) {
            return nullptr;
        }
        const auto zpOrFail = getSingleZeroPointOrFail(originWeightType);
        VPUX_THROW_WHEN(mlir::failed(zpOrFail),
                        "FP4 palletization requires equal zero points across all axes, "
                        "got type '{0}'",
                        originWeightType);
        VPUX_THROW_WHEN(zpOrFail.value() != 0,
                        "Asymmetric FP4 quantization (non-zero zero point '{0}') is not supported for "
                        "palletization LUT conversion",
                        zpOrFail.value());
    }

    // For FP4 storage the bit width is 4; for integer storage extract it from the type.
    const auto bitWidth =
            isFp4Storage ? 4u : mlir::cast<mlir::IntegerType>(originWeightType.getStorageType()).getWidth();
    const bool isPerAxis = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(originWeightType);
    VPUX_THROW_UNLESS(bitWidth > 0 && bitWidth <= 4, "Unsupported bitWidth '{0}'", bitWidth);

    const auto getShiftedLUTUniformType = [isFp4Storage, isActFp16, isActInt8, bitWidth,
                                           isPerAxis](mlir::quant::QuantizedType wgtType) -> SmallVector<double> {
        // FP4 storage: enumerate all 16 bit-patterns, decode each using the FP4 encoding, and round the
        // result to F8E4M3. FP4 weights always use an FP8 LUT entry format, regardless of whether the
        // activation path is FP16 or FP8.
        if (isFp4Storage) {
            constexpr unsigned kFp4LutSize = 16;
            SmallVector<double> fp4Lut(kFp4LutSize);
            for (unsigned i = 0; i < kFp4LutSize; ++i) {
                const float decodedVal =
                        static_cast<float>(vpux::type::float4_e2m1::from_bits(static_cast<uint8_t>(i)));
                fp4Lut[i] = static_cast<double>(static_cast<float>(vpux::type::float8_e4m3(decodedVal)));
            }
            return fp4Lut;
        }

        // Integer storage: map each signed/unsigned bit-pattern to its dequantized value.
        const auto originalWeightStorageTypeMin =
                mlir::quant::QuantizedType::getDefaultMinimumForInteger(wgtType.isSigned(), bitWidth);
        const auto originalWeightStorageTypeMax =
                mlir::quant::QuantizedType::getDefaultMaximumForInteger(wgtType.isSigned(), bitWidth);
        const int lutElemNum = 1 << bitWidth;
        VPUX_THROW_WHEN(lutElemNum != (originalWeightStorageTypeMax - originalWeightStorageTypeMin + 1),
                        "Unexpected storage type range '{0}'",
                        originalWeightStorageTypeMax - originalWeightStorageTypeMin + 1);
        SmallVector<double> quantileLUT(lutElemNum);
        const int64_t bitMask = (1LL << bitWidth) - 1LL;

        if (isPerAxis && isActFp16) {
            // In case of fp16 activations and per-axis quantized weights the LUT uses an 8-bit integer quantile type
            // (ui8 for unsigned weights, si8 for signed). The ZP is not subtracted in the LUT entries but is instead
            // subtracted by HW itself before the fp16 upscaling function in the IDU. The quantile type signedness
            // determines WMODE, which controls how HW extends the 4-bit ZP: zero-pad for U8, sign-extend for I8.
            // With the identity LUT mapping and matching signedness, the HW subtraction is correct for both cases.
            for (int64_t wgtVal = originalWeightStorageTypeMin; wgtVal <= originalWeightStorageTypeMax; ++wgtVal) {
                const unsigned unsignedTableIdx = static_cast<unsigned>(wgtVal & bitMask);
                quantileLUT[unsignedTableIdx] = static_cast<double>(wgtVal);
            }
        } else {
            auto res = getSingleZeroPointOrFail(wgtType);
            VPUX_THROW_WHEN(mlir::failed(res), "Failed to get zero point for quantized type '{0}'", wgtType);
            const auto zeroPoint = res.value();

            // Interpreting integer weights with their unsigned encodings (for instance -7 i4 is 4'b1001,
            // which can be interpreted as 9 as u4 and used as an unsigned index to the LUT)
            if (isActFp16 || isActInt8) {
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
        }

        return quantileLUT;
    };

    const auto newStorageIntegerType = mlir::IntegerType::get(ctx, bitWidth, mlir::IntegerType::Unsigned);
    constexpr unsigned newStorageIntegerTypeMin = 0;
    const unsigned newStorageIntegerTypeMax = (1 << bitWidth) - 1;
    const auto newQuantileType = [&]() -> mlir::Type {
        // FP4 weights are always palletized into an FP8 lookup table, even for FP16 activations
        // (including the per-axis case). The LUT values are already rounded to float8_e4m3 in
        // getShiftedLUTUniformType.
        if (isFp4Storage) {
            return mlir::Float8E4M3FNType::get(ctx);
        }
        if (isPerAxis && isActFp16) {
            // For 4-bit unsigned weights use ui8 (WMODE=U8 → ZP zero-padded) so that the IDU correctly
            // zero-extends ZP values ≥ 8. For signed 4-bit and all 2-bit weights, si8 (WMODE=I8) is correct
            // because ZP values fit without sign-extension issues.
            const auto signedness = (!originWeightType.isSigned() && bitWidth == 4) ? mlir::IntegerType::Unsigned
                                                                                    : mlir::IntegerType::Signed;
            return mlir::IntegerType::get(ctx, 8, signedness);
        }
        // For fp8 activations with integer weights, use F8E4M3FN: it provides better precision around 0
        // and can exactly represent all integers in [-16, +16], covering the full dequantized range.
        if (isActFp16) {
            return mlir::Float16Type::get(ctx);
        } else if (isActInt8) {
            return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
        } else {
            return mlir::Float8E4M3FNType::get(ctx);
        }
    }();

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
                0, newStorageQuantileType, perAxisType.getExpressedType(), perAxisType.getScales(),
                isActFp16 ? perAxisType.getZeroPoints() : newZeroPoints, perAxisType.getQuantizedDimension(),
                newStorageIntegerTypeMin, newStorageIntegerTypeMax);
    }

    VPUX_THROW("Unsupported Quantized Type '{0}'", originWeightType);
}

//
// ConvertToPalletizationLUTStrategy
//

void ConvertToPalletizationLUTStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();
    patterns.add<vpux::IE::ConvolutionLUTRewriter<IE::ConvolutionOp>>(ctx, isLegalTensorElem60XX, changeWeightTypeToLUT,
                                                                      log);
    patterns.add<vpux::IE::ConvolutionLUTRewriter<IE::GroupConvolutionOp>>(ctx, isLegalTensorElem60XX,
                                                                           changeWeightTypeToLUT, log);
    patterns.add<vpux::IE::ConvolutionLUTRewriter<IE::MatMulOp>>(ctx, isLegalTensorElem60XX, changeWeightTypeToLUT,
                                                                 log);
}

bool isLegalOp(mlir::Operation* op) {
    // only fp16, quant.uniform<f8E4M3FN:..., ...>, quant.uniform<f8E5M2:..., ...>, or quant.uniform<i8:..., ...>
    // are supported as activations types for this pass
    auto inputDequantOp = mlir::dyn_cast_or_null<IE::DequantizeOp>(op->getOperand(0).getDefiningOp());
    const auto actType =
            inputDequantOp == nullptr
                    ? mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType()
                    : mlir::cast<vpux::NDTypeInterface>(inputDequantOp.getInput().getType()).getElementType();

    const bool isActFp16 = actType.isF16();
    const bool isActFp8 = isFloat8Quantized(actType);
    const bool isActInt8 = isInt8Quantized(actType);

    if (!(isActFp16 || isActFp8 || isActInt8)) {
        return true;
    }

    // In NPU6 it's possible to have fp16 activations and int4 weights, converted to a palletized LUT with int8
    // quantileType and channel-wise zero points. FP4 weights are also supported by palletizing the LUT into FP8,
    // with the stored LUT entries encoded as F8E4M3 regardless of whether the activation path is FP16 or FP8.
    auto maybeDequantFilterOp = op->getOperand(1).getDefiningOp();
    const auto weightElemType =
            mlir::isa_and_present<IE::DequantizeOp, IE::DynamicDequantizeOp>(maybeDequantFilterOp)
                    ? mlir::cast<vpux::NDTypeInterface>(maybeDequantFilterOp->getOperand(0).getType()).getElementType()
                    : mlir::cast<vpux::NDTypeInterface>(op->getOperand(1).getType()).getElementType();

    if (const auto quantized = mlir::dyn_cast<mlir::quant::QuantizedType>(weightElemType)) {
        if (mlir::isa<mlir::Float4E2M1FNType>(quantized.getStorageType())) {
            const auto zpOrFail = getSingleZeroPointOrFail(quantized);
            if (mlir::failed(zpOrFail) || zpOrFail.value() != 0) {
                return true;
            }

            // FP4 weights need to be converted to a palletized LUT only for FP16 or FP8 activations;
            // any other activation type is left legal/unconverted since that combination is not
            // supported by changeWeightTypeToLUT.
            return !(isActFp16 || isActFp8);
        }
        return IE::isLegalTensorElemForPalletization(weightElemType, /*convertOnlyAsymmetric=*/false,
                                                     /*allowPerChannelZp=*/isActFp16);
    }

    return true;
}

void ConvertToPalletizationLUTStrategy::markOpLegality(mlir::ConversionTarget& target, Logger&) const {
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::MatMulOp>(isLegalOp);
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::DequantizeOp>();
    target.addLegalOp<IE::DynamicDequantizeOp>();
    target.addLegalOp<IE::QuantizeCastOp>();
}

}  // namespace vpux::IE::arch60xx
