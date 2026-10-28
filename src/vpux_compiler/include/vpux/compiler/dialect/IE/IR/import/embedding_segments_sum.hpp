//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>

namespace vpux::IE::EmbeddingSegmentsSum {

/** @brief Returns num segments of an IE::EmbeddingSegmentsSumOp.
 */
llvm::FailureOr<int64_t> parseNumSegments(mlir::Location loc, mlir::ValueRange opInputs);

/** @brief Returns default index of an IE::EmbeddingSegmentsSumOp. If index is
           not provided via operation inputs, -1 is returned as a default value.
 */
int64_t parseDefaultIndex(mlir::ValueRange opInputs);

}  // namespace vpux::IE::EmbeddingSegmentsSum
