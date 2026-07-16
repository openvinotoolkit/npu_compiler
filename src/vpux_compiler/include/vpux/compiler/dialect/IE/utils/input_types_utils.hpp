//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

namespace vpux {
namespace IE {

// Arch-specific constraints for activation and filter storage types.
struct InputTypeConstraints {
    bool (*isInputStorageAllowed)(mlir::Type) = nullptr;
    bool (*isFilterStorageAllowed)(mlir::Type) = nullptr;
    bool allowPerAxisInput = false;
};

// Check whether the quantized inputs have allowed storage types.
inline bool areInputTypesSupported(mlir::Value inputValue, mlir::Value filterValue,
                                   const InputTypeConstraints& constraints) {
    // Activation: uniform quantization required; per-axis allowed when constraints permit
    auto inputElemType = mlir::cast<vpux::NDTypeInterface>(inputValue.getType()).getElementType();
    auto inputQType = mlir::dyn_cast<mlir::quant::QuantizedType>(inputElemType);
    if (constraints.allowPerAxisInput) {
        if (!mlir::isa_and_nonnull<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(
                    inputQType)) {
            return false;
        }
    } else {
        if (!mlir::isa_and_nonnull<mlir::quant::UniformQuantizedType>(inputQType)) {
            return false;
        }
    }

    if (constraints.isInputStorageAllowed != nullptr &&
        !constraints.isInputStorageAllowed(inputQType.getStorageType())) {
        return false;
    }

    // Filter: per-tensor or per-axis uniform quantization
    auto filterElemType = mlir::cast<vpux::NDTypeInterface>(filterValue.getType()).getElementType();
    auto filterQType = mlir::dyn_cast<mlir::quant::QuantizedType>(filterElemType);
    if (!mlir::isa_and_nonnull<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(
                filterQType)) {
        return false;
    }

    // Input and filter storage types must share the same type category (both float or both integer)
    if (mlir::isa<mlir::FloatType>(inputQType.getStorageType()) !=
        mlir::isa<mlir::FloatType>(filterQType.getStorageType())) {
        return false;
    }

    if (constraints.isFilterStorageAllowed != nullptr &&
        !constraints.isFilterStorageAllowed(filterQType.getStorageType())) {
        return false;
    }

    return true;
}

}  // namespace IE

}  // namespace vpux
