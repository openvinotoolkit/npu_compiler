//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/control_flow.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;
using namespace mlir;

//
// RegionBranchTerminatorOpInterface
//

mlir::MutableOperandRange vpux::IE::YieldOp::getMutableSuccessorOperands(::mlir::RegionSuccessor) {
    return getOperandsMutable();
}

//
// verify
//

mlir::LogicalResult vpux::IE::YieldOp::verify() {
    const auto op = getOperation();
    if (op->getNumOperands() < 1) {
        return errorAt(op->getLoc(), "Operation must have at least one operand");
    }

    return mlir::success();
}
