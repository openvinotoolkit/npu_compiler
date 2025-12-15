//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"

#include <llvm/ADT/bit.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

VPU::NCESparsity::IntOrFloatType VPU::arch37xx::getScale(uint8_t shift, int16_t mult, double rescale,
                                                         mlir::Type inputType) {
    // VPUX37XX expects scale in IEEE754 format in NCE_DPU_PPE_FP_SCALE register in case input has FP16/BF16 type
    auto inStorageType = mlir::quant::QuantizedType::castToStorageType(inputType);
    if (mlir::isa<mlir::FloatType>(inputType) ||
        mlir::isa<mlir::Float8E5M2Type, mlir::Float8E4M3FNType>(inStorageType)) {
        return VPU::NCESparsity::toHex(rescale);
    }

    int32_t PPE_SHIFT_OFFSET = 8;
    int32_t PPE_SHIFT_VALUE = shift;

    int32_t PPE_MULT_OFFSET = 16;
    // FIXME: PPE multiplier has sign, which may affect lower bits
    int32_t PPE_MULT_VALUE = mult;

    return (PPE_SHIFT_VALUE << PPE_SHIFT_OFFSET) | (PPE_MULT_VALUE << PPE_MULT_OFFSET);
}

double VPU::arch37xx::retrieveScaleFromTable(VPU::NCESparsity::IntOrFloatType val, mlir::Type inputType) {
    // VPUX37XX expects scale in IEEE754 format in NCE_DPU_PPE_FP_SCALE register in case input has FP16/BF16 type
    const auto realVal = std::get<int32_t>(val);
    auto inStorageType = mlir::quant::QuantizedType::castToStorageType(inputType);
    if (mlir::isa<mlir::FloatType>(inputType) ||
        mlir::isa<mlir::Float8E5M2Type, mlir::Float8E4M3FNType>(inStorageType)) {
        return static_cast<double>(llvm::bit_cast<float>(realVal));
    }

    constexpr int32_t PPE_SHIFT_OFFSET = 8;
    constexpr int32_t PPE_SHIFT_MASK = 0xFF;
    const int32_t shift = (realVal >> PPE_SHIFT_OFFSET) & PPE_SHIFT_MASK;

    constexpr int32_t PPE_MULT_OFFSET = 16;
    constexpr int32_t PPE_MULT_MASK = 0xFFFF;
    const auto mult = static_cast<int16_t>((realVal >> PPE_MULT_OFFSET) & PPE_MULT_MASK);

    // Applying reverse transformation done by QuantizationApproximation
    constexpr uint8_t bits = 15;
    const auto doubleMult = static_cast<double>(mult) / static_cast<double>(1 << bits);
    const auto exponent = bits - shift;

    return ldexp(doubleMult, exponent);
}

VPU::NCESparsity::IntOrFloatType VPU::arch37xx::getBias(double realVal, mlir::Type inputType) {
    // On NPU 37xx and 4000, the PPE has a FP and an INT PPE pipeline. Both pipelines have the possibility to apply
    // per-output-channel bias stored in weights table. Depending on input data type bias is aplied as follow: for
    // I/U8/4 input data type, bias is applied with INT pipeline and bias should have int32_t values while for FP input
    // data type bias is applied with FP pipeline and should have float32_t values
    if (mlir::isa<mlir::quant::QuantizedType>(inputType)) {
        return checked_cast<int32_t>(std::round(realVal));
    }
    return VPU::NCESparsity::toHex(realVal);
}
