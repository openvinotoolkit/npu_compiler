//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

//
// Generated
//

namespace vpux::VPU {
/** @brief A version of the vpux::areTypesCompatible() specific to VPU::CopyOp.

    This version allows certain type differences that a normal type comparison
    prohibits at the moment. For instance, VPU::CopyOp allows a copy between
    bounded and tensors with dynamic dims mask (including distributed tensors)
    as both types are equivalent but encode the same information differently.

    @note checkInferredDimsOrder and checkInferredMemSpace parameters of
    vpux::areTypesCompatible() are assumed to be set to `true`.
 */
bool areTypesCompatibleForCopy(mlir::TypeRange lhs, mlir::TypeRange rhs);
}  // namespace vpux::VPU

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/data_movement.hpp.inc>
