//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPU/impl/nce_sparsity_converters.hpp"

#include <llvm/ADT/bit.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

VPU::NCESparsity::IntOrFloatType VPU::arch60xx::getScale(uint8_t, int16_t, double rescale, mlir::Type scalesType) {
    if (mlir::isa<mlir::Float8E4M3FNType>(scalesType)) {
        return static_cast<vpux::type::float8_e4m3>(rescale);
    } else if (mlir::isa<mlir::Float8E5M2Type>(scalesType)) {
        return static_cast<vpux::type::float8_e5m2>(rescale);
    } else if (mlir::isa<mlir::Float8E8M0FNUType>(scalesType)) {
        return static_cast<vpux::type::float8_e8m0>(rescale);
    } else {
        return static_cast<float>(rescale);
    }
}

double VPU::arch60xx::retrieveScaleFromTable(VPU::NCESparsity::IntOrFloatType val, mlir::Type scalesType) {
    if (mlir::isa<mlir::Float8E4M3FNType>(scalesType)) {
        return static_cast<double>(std::get<vpux::type::float8_e4m3>(val));
    } else if (mlir::isa<mlir::Float8E5M2Type>(scalesType)) {
        return static_cast<double>(std::get<vpux::type::float8_e5m2>(val));
    } else if (mlir::isa<mlir::Float8E8M0FNUType>(scalesType)) {
        return static_cast<double>(std::get<vpux::type::float8_e8m0>(val));
    }

    return static_cast<double>(std::get<float>(val));
}

VPU::NCESparsity::IntOrFloatType VPU::arch60xx::getBias(double realVal, mlir::Type) {
    return static_cast<float>(realVal);
}
