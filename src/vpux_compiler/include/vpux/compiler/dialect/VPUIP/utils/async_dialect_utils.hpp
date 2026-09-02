//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"

#include <mlir/Dialect/Async/IR/Async.h>

namespace vpux::VPUIP {

// Get the executor type from the async execute operation.
config::ExecutorKind getExecutorType(mlir::async::ExecuteOp execOp);
config::ExecutorKind getExecutorType(size_t opIdx, AsyncDepsInfo& depsInfo);

// Get the DMA type operation from the async execute operation.
VPUIP::DMATypeOpInterface getDmaTypeOp(mlir::async::ExecuteOp execOp);
// Get DMA direction relative to CMX
bool isDmaDDR2CMX(mlir::async::ExecuteOp execOp);
bool isDmaCMX2DDR(mlir::async::ExecuteOp execOp);
bool isDmaDDR2DDR(mlir::async::ExecuteOp execOp);
bool isDmaCMX2CMX(mlir::async::ExecuteOp execOp);

}  // namespace vpux::VPUIP
