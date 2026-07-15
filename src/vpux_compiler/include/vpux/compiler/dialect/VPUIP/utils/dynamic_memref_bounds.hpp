//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/CallInterfaces.h>

// Public VPUIP interface for call-boundary dynamic memref allocation. It declares the helpers
// that other passes use when they need to allocate input or output buffers for nested calls
// while preserving dynamic memref types in IR.

namespace vpux::VPUIP {

// Returns per-dimension upper bounds for a specific input or output buffer of the call's callee.
// Resolves the callee module via the call's nested symbol reference and looks up the
// net::NetworkInfoOp declared there. Returns an empty vector when the callee module cannot be
// found, when no NetworkInfoOp exists, or when the DataInfo entry carries no BoundedTensorType.
// Only Core.NestedCall-style callees (@Module::@func) have a submodule with a separate NetworkInfo;
// flat func.call targets (e.g. @main) return empty.
SmallVector<int64_t> getBoundsForCalleeBuffer(mlir::CallOpInterface callOp, size_t bufferIdx, bool isInput, Logger& log,
                                              StringRef sourceTag);

// Allocates a memref buffer at a call boundary:
// - For static memrefs the buffer is allocated directly.
// - For dynamic memrefs the per-dimension upper bounds are fetched from the callee module's
//   net::NetworkInfoOp and materialised as arith.constant operands to memref.alloc, so the
//   physical buffer is large enough for any runtime shape up to the declared bound. The IR type
//   (e.g. memref<?x64xf16>) is preserved in the returned Value so consumers retain dynamic
//   semantics.
mlir::Value allocateCallBoundaryMemref(mlir::CallOpInterface callOp, size_t bufferIdx, bool isInput,
                                       mlir::MemRefType memrefType, mlir::OpBuilder& builder, Logger& log,
                                       StringRef sourceTag);

}  // namespace vpux::VPUIP
