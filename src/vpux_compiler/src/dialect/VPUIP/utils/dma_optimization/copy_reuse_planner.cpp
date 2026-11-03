//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/copy_reuse_planner.hpp"

#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/copy_reuse_approval_window.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/copy_reuse_candidate_collection.hpp"

#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edit_plan.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>

using namespace vpux;

namespace vpux::VPUIP::DmaOptimization {

namespace {
// Remove dead input views after copy reuse approval
SmallVector<mlir::Operation*, 8> collectDeadInputViews(llvm::ArrayRef<ApprovedCopyReuse> approvedReuses) {
    llvm::SmallPtrSet<mlir::Operation*, 8> removedCopies;
    llvm::SmallPtrSet<mlir::Operation*, 8> candidateViews;
    for (const auto& approval : approvedReuses) {
        auto removedCopy = approval.removedCopy;
        removedCopies.insert(removedCopy.getOperation());
        candidateViews.insert_range(approval.removedCopyInputViews);
    }

    llvm::SmallPtrSet<mlir::Operation*, 8> deadViews;
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto* view : candidateViews) {
            if (deadViews.contains(view) || view->getNumResults() != 1) {
                continue;
            }
            if (llvm::all_of(view->getResult(0).getUses(), [&](mlir::OpOperand& use) {
                    return removedCopies.contains(use.getOwner()) || deadViews.contains(use.getOwner());
                })) {
                deadViews.insert(view);
                changed = true;
            }
        }
    }

    SmallVector<mlir::Operation*, 8> eraseOrder;
    while (!deadViews.empty()) {
        auto leaf = llvm::find_if(deadViews, [&](mlir::Operation* view) {
            return llvm::none_of(view->getResult(0).getUsers(), [&](mlir::Operation* user) {
                return deadViews.contains(user);
            });
        });
        if (leaf == deadViews.end()) {
            return {};
        }
        eraseOrder.push_back(*leaf);
        deadViews.erase(*leaf);
    }
    return eraseOrder;
}

void emitApprovedReuses(llvm::ArrayRef<ApprovedCopyReuse> approvedReuses, EditPlan& out) {
    for (const auto& approval : approvedReuses) {
        auto removedCopy = approval.removedCopy;
        out.replaceAllUsesWith(approval.oldResult, approval.newResult);
        out.eraseOp(removedCopy.getOperation());
        if (approval.deadOutputBufferOwner != nullptr) {
            out.eraseOp(approval.deadOutputBufferOwner);
        }
    }
    for (auto* deadView : collectDeadInputViews(approvedReuses)) {
        out.eraseOp(deadView);
    }
}

}  // namespace

size_t CopyReusePlanner::planCopyReuse(const EdgeCollection& edges, const AliasesInfo& aliasesInfo,
                                       EditPlan& out) const {
    // Step 1: Collect copy reuse candidates.
    const auto candidates = collectCopyReuseCandidates(edges);
    if (candidates.size() < 2) {
        return 0;
    }
    // Step 2: Approve the collected copy reuse candidates based on buffer roles and loop region requirements.
    CopyReuseApprovalWindow approvalWindow(aliasesInfo, _log);
    _log.trace("[CopyReuse] EdgeCollector collected {0} copy reuse candidates", candidates.size());
    const auto approvedReuses = approvalWindow.approve(candidates);
    if (approvedReuses.empty()) {
        return 0;
    }
    // Step 3: Emit the approved copy reuses into the edit plan
    emitApprovedReuses(approvedReuses, out);
    _log.trace("[CopyReuse] EditPlan scheduled {0} copy reuses", approvedReuses.size());
    return approvedReuses.size();
}

}  // namespace vpux::VPUIP::DmaOptimization
