//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/outlining_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

using namespace vpux;

bool VPU::isConstOperandOp(mlir::Operation* op) {
    if (mlir::isa<VPU::StorageElementTableOp, Const::DeclareOp>(op)) {
        return true;
    }

    if (mlir::isa<VPU::GroupedViewLikeOpInterface>(op)) {
        return llvm::all_of(op->getOperands(), [&](mlir::Value v) {
            if (mlir::isa<mlir::BlockArgument>(v)) {
                return true;
            }
            auto parentOp = v.getDefiningOp();
            return isConstOperandOp(parentOp);
        });
    }

    return false;
}
