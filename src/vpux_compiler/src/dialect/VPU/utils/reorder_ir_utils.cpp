//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/reorder_ir_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
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
void reorderOperations(mlir::ArrayRef<mlir::Operation*> operations) {
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
                origOp->moveAfter(parents.back());
            }

            continue;
        }

        auto* firstUser = getFirstUser(origOp->getResult(0));

        if (firstUser != nullptr) {
            origOp->moveBefore(firstUser);
        }
    }
}
}  // namespace vpux::VPU
