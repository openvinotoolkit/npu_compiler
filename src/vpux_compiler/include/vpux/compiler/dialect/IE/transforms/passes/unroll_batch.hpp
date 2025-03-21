//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {

bool doesOpNeedToUnroll(mlir::Operation* op);
bool doesEltwiseNeedToUnroll(mlir::Operation* op);
bool doesMemPermuteNeedToUnroll(IE::MemPermuteOp permuteOp);

}  // namespace IE
}  // namespace vpux
