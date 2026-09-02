//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::VPUIP {

//! @brief Allocates buffers for the results of the funcOps.
//! Only pass in the funcOp and callOp that need buffer allocation.
void allocateBuffersForNetResults(const mlir::DenseSet<mlir::CallOpInterface>& callOps,
                                  const mlir::DenseSet<mlir::func::FuncOp>& funcOps, Logger& log);

//! @brief Extends the signatures of the given funcOps with output-buffer arguments and rewrites their
//! return ops to write results into those arguments (destination-passing ABI).
template <typename CopyOp = VPUIP::CopyOp>
void updateFuncBoundariesForNetResults(const mlir::DenseSet<mlir::func::FuncOp>& funcOps, Logger& log);

//! @brief Rewrites the given call ops to pass caller-supplied output buffers, reusing a dominating
//! destination buffer when one is available. Run after updateFuncBoundariesForNetResults has extended
//! all relevant callee and caller function boundaries.
void updateCallsForNetResults(const mlir::DenseSet<mlir::CallOpInterface>& callOps, Logger& log);

}  // namespace vpux::VPUIP
