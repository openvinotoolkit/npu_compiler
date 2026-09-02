//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_tiling_policy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <algorithm>

namespace vpux::VPU::VF::v2::details {

SmallVector<OpWithViewInputs> getParentOpWithViewInputs(mlir::Operation* op) {
    SmallVector<OpWithViewInputs> parentOps;
    for (auto& item : op->getOpOperands()) {
        auto operand = item.get();
        auto idx = item.getOperandNumber();

        auto parent = operand.getDefiningOp();
        SmallVector<mlir::Operation*> viewOps;
        while (parent != nullptr && isPureViewOp(parent)) {
            viewOps.push_back(parent);
            parent = parent->getOperand(0).getDefiningOp();
        }
        // Reverse: the loop above collects view ops from consumer toward
        // producer, but inferDistributedTypeThroughViewOps iterates forward
        // from the producer distribution, so producer-to-consumer order is required.
        std::reverse(viewOps.begin(), viewOps.end());
        if (auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(findParent(operand))) {
            parentOps.push_back({idx, clusteredOp, std::move(viewOps)});
        }
    }
    return parentOps;
}

// Collect all non-viewlike uses of the VF block argument by traversing through chains of
// TilingViewLikeOpInterface ops
SmallVector<mlir::OpOperand*> getComputeUses(mlir::BlockArgument currInputArg) {
    SmallVector<mlir::OpOperand*> uses;
    SmallVector<mlir::OpOperand*> worklist;
    for (auto& use : currInputArg.getUses()) {
        worklist.push_back(&use);
    }
    while (!worklist.empty()) {
        auto* use = worklist.pop_back_val();
        auto userOp = use->getOwner();
        if (mlir::isa<VPU::TilingViewLikeOpInterface>(userOp)) {
            // Skip view-like op, but add its users to the worklist
            for (auto& user : userOp->getUses()) {
                worklist.push_back(&user);
            }
        } else {
            uses.push_back(use);
        }
    }
    return uses;
}

mlir::Operation* getMappedOpInMergedVF(mlir::Operation* op, VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currOp,
                                       VPU::VerticalFusionOp mergedOp) {
    auto vfOp = op->getParentOfType<VPU::VerticalFusionOp>();
    VPUX_THROW_UNLESS(vfOp == prevOp || vfOp == currOp,
                      "Operation {0} does not belong to previous VF {1} or current VF {2}", op->getLoc(),
                      prevOp->getLoc(), currOp->getLoc());
    auto opIdx = std::distance(vfOp.getBody()->getOperations().begin(), mlir::Block::iterator(op));
    if (vfOp == prevOp) {
        return &*std::next(mergedOp.getBody()->getOperations().begin(), opIdx);
    }

    auto prevOutputOp = prevOp.getBody()->getTerminator()->getOperands().back().getDefiningOp();
    auto prevOutputOpIdx =
            std::distance(prevOp.getBody()->getOperations().begin(), mlir::Block::iterator(prevOutputOp));
    return &*std::next(mergedOp.getBody()->getOperations().begin(), opIdx + prevOutputOpIdx + 1);
}

VPU::MultiClusterStrategy getMultiClusterStrategy(
        VPU::ClusteredOpInterface clusteredOp,
        const DenseMap<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy>& rollbackStrategy) {
    auto iter = rollbackStrategy.find(clusteredOp);
    if (iter != rollbackStrategy.end()) {
        return iter->second;
    }
    return clusteredOp.getMultiClusterStrategy().value();
}

}  // namespace vpux::VPU::VF::v2::details
