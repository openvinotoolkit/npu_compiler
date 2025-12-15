//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPU/impl/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"

#include <llvm/ADT/bit.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

VPU::NCESparsity::IntOrFloatType VPU::arch50xx::getScale(uint8_t, int16_t, double rescale, mlir::Type) {
    return VPU::NCESparsity::toHex(rescale);
}

double VPU::arch50xx::retrieveScaleFromTable(VPU::NCESparsity::IntOrFloatType val, mlir::Type) {
    const auto realVal = std::get<int32_t>(val);
    return static_cast<double>(llvm::bit_cast<float>(realVal));
}

VPU::NCESparsity::IntOrFloatType VPU::arch50xx::getBias(double realVal, mlir::Type) {
    return VPU::NCESparsity::toHex(realVal);
}
