//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>

namespace vpux::IE::Gather {

/** @brief Returns an axis of an IE::GatherOp.
 */
llvm::FailureOr<int64_t> parseAxis(mlir::Location loc, mlir::ValueRange opInputs);

}  // namespace vpux::IE::Gather
