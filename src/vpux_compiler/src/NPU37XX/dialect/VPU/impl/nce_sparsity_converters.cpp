//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
    if (inputType.isa<mlir::FloatType>() || inStorageType.isFloat8E5M2() || inStorageType.isFloat8E4M3FN()) {
        return VPU::NCESparsity::toHex(rescale);
    }

    int32_t PPE_SHIFT_OFFSET = 8;
    int32_t PPE_SHIFT_VALUE = shift;

    int32_t PPE_MULT_OFFSET = 16;
    // FIXME: PPE multiplier has sign, which may affect lower bits
    int32_t PPE_MULT_VALUE = mult;

    return (PPE_SHIFT_VALUE << PPE_SHIFT_OFFSET) | (PPE_MULT_VALUE << PPE_MULT_OFFSET);
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
