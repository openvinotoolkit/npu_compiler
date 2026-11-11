//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"

#include <mlir/IR/Types.h>

namespace vpux::VPU::arch60xx {

NCESparsity::IntOrFloatType getScale(uint8_t shift, int16_t mult, double rescale, mlir::Type scalesType);
NCESparsity::IntOrFloatType getBias(double realVal, mlir::Type);
double retrieveScaleFromTable(VPU::NCESparsity::IntOrFloatType val, mlir::Type);

}  // namespace vpux::VPU::arch60xx
