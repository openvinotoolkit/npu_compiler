//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPU {

void reorderOperations(mlir::ArrayRef<mlir::Operation*> operations, Logger log = Logger::global());

}  // namespace VPU
}  // namespace vpux
