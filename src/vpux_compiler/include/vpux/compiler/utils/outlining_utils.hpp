//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

bool isConstOperandOp(mlir::Operation* op);

}  // namespace VPU
}  // namespace vpux
