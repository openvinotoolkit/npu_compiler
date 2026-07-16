//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

namespace vpux::IE::utils {

// This check is mode-agnostic for NCE-oriented rewrites.
// It returns true for F32/F64 element types.
// Returns false if the value type does not implement NDTypeInterface.
inline bool hasF32OrF64Precision(mlir::Value value) {
    if (!value) {
        return false;
    }

    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(value.getType());
    if (!ndType) {
        return false;
    }

    const auto elemType = ndType.getElementType();
    return elemType.isF32() || elemType.isF64();
}

}  // namespace vpux::IE::utils
