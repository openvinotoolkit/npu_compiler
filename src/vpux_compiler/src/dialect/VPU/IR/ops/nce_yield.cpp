//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/control_flow.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// RegionBranchTerminatorOpInterface
//

mlir::MutableOperandRange vpux::VPU::YieldOp::getMutableSuccessorOperands(::mlir::RegionBranchPoint) {
    return getOperandsMutable();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::YieldOp::verify() {
    const auto op = getOperation();
    if (op->getNumOperands() < 1) {
        return errorAt(op->getLoc(), "Operation must have at least one operand");
    }

    return mlir::success();
}
