//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edit_plan.hpp"

using namespace vpux;
using namespace vpux::VPUIP::DmaOptimization;

void EditPlan::replaceAllUsesWith(mlir::Value lhs, mlir::Value rhs) {
    if (lhs == nullptr || rhs == nullptr || lhs == rhs || _replaceAllUsesSources.contains(lhs)) {
        return;
    }

    EditOp edit;
    edit.kind = EditKind::ReplaceAllUsesWith;
    edit.lhsValue = lhs;
    edit.rhsValue = rhs;
    _ops.push_back(edit);
    _replaceAllUsesSources.insert(lhs);
}

void EditPlan::eraseOp(mlir::Operation* op) {
    if (op == nullptr || !_erasedSet.insert(op).second) {
        return;
    }

    EditOp edit;
    edit.kind = EditKind::EraseOp;
    edit.op = op;
    _ops.push_back(edit);
}

mlir::LogicalResult EditPlan::verify(Logger log) const {
    llvm::DenseSet<mlir::Value> replaceAllUsesSources;
    llvm::DenseSet<mlir::Operation*> erasedOps;

    for (const auto& edit : _ops) {
        switch (edit.kind) {
        case EditKind::ReplaceAllUsesWith:
            if (edit.lhsValue == nullptr || edit.rhsValue == nullptr || edit.lhsValue == edit.rhsValue ||
                !replaceAllUsesSources.insert(edit.lhsValue).second) {
                log.error("[EditPlan] invalid or duplicate replace-all-uses edit");
                return mlir::failure();
            }
            break;
        case EditKind::EraseOp:
            if (edit.op == nullptr || !erasedOps.insert(edit.op).second) {
                log.error("[EditPlan] invalid or duplicate erase");
                return mlir::failure();
            }
            break;
        }
    }

    return mlir::success();
}
