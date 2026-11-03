//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

// Forward declarations for IE ops referenced in VPU ops
namespace vpux {
namespace IE {
class AttentionOp;
}  // namespace IE
}  // namespace vpux

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/specialized.hpp.inc>
