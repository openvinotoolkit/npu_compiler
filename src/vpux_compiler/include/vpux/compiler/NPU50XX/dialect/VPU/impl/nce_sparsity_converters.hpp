//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"

namespace vpux::VPU::arch50xx {

VPU::NCESparsity::IntOrFloatType getScale(uint8_t shift, int16_t mult, double rescale, mlir::Type inputType);
VPU::NCESparsity::IntOrFloatType getBias(double realVal, mlir::Type);
double retrieveScaleFromTable(VPU::NCESparsity::IntOrFloatType val, mlir::Type);

}  // namespace vpux::VPU::arch50xx
