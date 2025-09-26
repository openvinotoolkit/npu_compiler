//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPU {

void reorderOperations(mlir::ArrayRef<mlir::Operation*> operations);

}  // namespace VPU
}  // namespace vpux
