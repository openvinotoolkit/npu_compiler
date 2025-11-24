//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux {

//! @brief Allocates buffers for the results of the funcOps.
//! Only pass in the funcOp and callOp that need buffer allocation.
template <typename CopyOp = VPUIP::CopyOp>
void allocateBuffersForNetResults(ArrayRef<mlir::CallOpInterface> callOps, ArrayRef<mlir::func::FuncOp> funcOps,
                                  Logger& log);

}  // namespace vpux
