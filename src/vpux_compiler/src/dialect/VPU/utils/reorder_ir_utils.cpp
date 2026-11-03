//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/reorder_ir_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include "vpux/utils/core/range.hpp"

namespace vpux::VPU {

// for passed operations move it close to their users or parents:
// for two-inputs operations place it right after the last parent in order not to
// execute it later than both inputs are ready. Usually misalignment happens
// when one of the input is executed too early
//               CONV0
//         |             |
//       CONV1         CONV2
//                       |
//                     CONV3
//         |             |
//               ...
//             ELTWISE
// so, it would have been more beneficial to execute eltwise earlier just after convolutions
// for other operations, place them close to users in order to execute them just before
// they are needed
void reorderOperations(mlir::ArrayRef<mlir::Operation*> operations, Logger log) {
    log.trace("Reordering {0} operation(s)", operations.size());
    for (auto origOp : operations | reversed) {
        if (origOp->hasTrait<VPU::EltwiseOp>() && origOp->getNumOperands() > 1) {
            SmallVector<mlir::Operation*> parents;
            for (auto operand : origOp->getOperands()) {
                if (auto parentOp = operand.getDefiningOp()) {
                    parents.push_back(parentOp);
                }
            }
            if (!parents.empty()) {
                llvm::sort(parents, [](auto* lhs, auto* rhs) {
                    return lhs->isBeforeInBlock(rhs);
                });
                log.nest().trace("Eltwise op '{0}' at '{1}' has multiple inputs; moving after last parent '{2}'",
                                 origOp->getName(), origOp->getLoc(), parents.back()->getLoc());
                origOp->moveAfter(parents.back());
            }

            continue;
        }

        // Find the earliest first-user across all results to preserve dominance
        // for multi-result ops (e.g. NCE ops with a fused reduce output).
        mlir::Operation* firstUser = nullptr;
        for (auto result : origOp->getResults()) {
            auto* candidate = getFirstUser(result);
            if (candidate == nullptr) {
                continue;
            }
            if (firstUser == nullptr || candidate->isBeforeInBlock(firstUser)) {
                firstUser = candidate;
            }
        }

        if (firstUser != nullptr) {
            log.nest().trace("Op '{0}' at '{1}' is moved before its first user '{2}'", origOp->getName(),
                             origOp->getLoc(), firstUser->getLoc());
            origOp->moveBefore(firstUser);

            // Moving NCE ops closer to their users may end up leaving their SW op producers for weights (Dequant in
            // this case) behind. This increases the chance of a spill happening in between.
            if (mlir::isa<VPU::NCEOpInterface>(origOp) && origOp->getNumOperands() > 1) {
                auto weightsParent = origOp->getOperand(1).getDefiningOp<VPU::DequantizeOp>();
                if (weightsParent == nullptr || weightsParent->getBlock() != origOp->getBlock()) {
                    continue;
                }

                // Parent op might be produced by another op, in which case moving it may prove detrimental
                if (!mlir::isa<mlir::BlockArgument>(weightsParent.getInput()) &&
                    weightsParent.getInput().getDefiningOp<Const::DeclareOp>() == nullptr) {
                    continue;
                }

                // Make sure the op dominance isn't broken by moving the Dequant near the NCEOp.
                // A use may live inside a nested region captured directly from this block (e.g. an
                // scf.for/scf.if body, which is not IsolatedFromAbove), so isBeforeInBlock cannot be
                // called on it directly; resolve it to its ancestor op within origOp's block first.
                if (!llvm::all_of(weightsParent->getUses(), [&](mlir::OpOperand& use) {
                        if (use.getOwner() == origOp) {
                            return true;
                        }

                        auto* ancestorInBlock = origOp->getBlock()->findAncestorOpInBlock(*use.getOwner());
                        if (ancestorInBlock == nullptr) {
                            return false;
                        }

                        return origOp->isBeforeInBlock(ancestorInBlock);
                    })) {
                    continue;
                }

                weightsParent->moveBefore(origOp);
                log.nest().trace("Op '{0}' at '{1}' has Dequant producer for weights; move Dequant before it",
                                 origOp->getName(), origOp->getLoc());
            }
        }
    }
}
}  // namespace vpux::VPU
