//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

namespace vpux {

// Rejects signless integers
inline mlir::LogicalResult isTypeSignedOrUnsigned(mlir::Operation* op, mlir::Value type) {
    if (type == nullptr) {
        return errorAt(op, "Type value is null");
    }

    const auto elemType = mlir::cast<vpux::NDTypeInterface>(type.getType()).getElementType();
    if (const auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        if (intType.isSignless()) {
            return errorAt(op, "Type element must not be a signless integer; use signed (si) or unsigned (u).");
        }
    }

    return mlir::success();
}

// Rejects integers with width >= 32b and signless integers
inline mlir::LogicalResult isZeroPointsTypeValidForQuantization(mlir::Operation* op, mlir::Value zeroPoints) {
    if (zeroPoints == nullptr) {
        return errorAt(op, "Value for zeroPoints is null");
    }

    const auto elemType = mlir::cast<vpux::NDTypeInterface>(zeroPoints.getType()).getElementType();
    if (const auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        if (intType.getWidth() >= 32) {
            return errorAt(
                    op, "ZeroPoints element type must be narrower than 32 bits; si32/u32 and wider are not supported.");
        }
    }

    return isTypeSignedOrUnsigned(op, zeroPoints);
}

}  // namespace vpux
