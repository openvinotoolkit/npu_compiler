//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edit_plan.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;
using namespace vpux::VPUIP::DmaOptimization;

mlir::LogicalResult EditPlanApplier::run(mlir::IRRewriter& rewriter, const EditPlan& plan) {
    if (mlir::failed(plan.verify(_log))) {
        return mlir::failure();
    }

    for (const auto& edit : plan.ops()) {
        if (edit.kind == EditKind::ReplaceAllUsesWith) {
            auto lhsValue = edit.lhsValue;
            lhsValue.replaceAllUsesWith(edit.rhsValue);
        }
    }

    for (const auto& edit : plan.ops()) {
        if (edit.kind != EditKind::EraseOp) {
            continue;
        }
        VPUX_THROW_UNLESS(edit.op->use_empty(),
                          "[EditPlanApplier] cannot erase '{0}' at '{1}' because it still has uses", edit.op->getName(),
                          edit.op->getLoc());
        rewriter.eraseOp(edit.op);
    }

    return mlir::success();
}
