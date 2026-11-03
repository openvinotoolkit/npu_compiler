//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>

namespace vpux::IE::EmbeddingBagOffsetsSum {

/** @brief Returns default index of an IE::EmbeddingBagOffsetsSumOp. If index is
           not provided via operation inputs, -1 is returned as a default value.
 */
int64_t parseDefaultIndex(mlir::ValueRange opInputs);

}  // namespace vpux::IE::EmbeddingBagOffsetsSum
