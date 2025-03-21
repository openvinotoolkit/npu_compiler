//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/STLExtras.h>

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_SPLITENQUEUEOPS
#define GEN_PASS_DEF_SPLITENQUEUEOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

// enqueue = [var1, var2(breakingPoint), var3, var4(breakingPoint), var5]
// we are going to replace it with 3 enqueues
// enqueue1 = [var1, var2]
// enqueue2 = [var3, var4]
// enqueue3 = [var5]

class SplitEnqueueOpsPass : public VPUMI40XX::impl::SplitEnqueueOpsBase<SplitEnqueueOpsPass> {
public:
    explicit SplitEnqueueOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SplitEnqueueOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto firstEnqu = mpi.getWorkItemTasks();
    if (firstEnqu == nullptr) {
        return;
    }

    auto firstEnquOp = mlir::cast<VPURegMapped::EnqueueOp>(firstEnqu.getDefiningOp());

    for (auto enqueueOp : llvm::make_early_inc_range(netFunc.getOps<VPURegMapped::EnqueueOp>())) {
        auto start = mlir::cast<VPURegMapped::TaskOpInterface>(enqueueOp.getStart().getDefiningOp());
        auto end = mlir::cast<VPURegMapped::TaskOpInterface>(enqueueOp.getEnd().getDefiningOp());
        if (enqueueOp.getStart() == enqueueOp.getEnd()) {
            continue;
        }

        mlir::OpBuilder builder(enqueueOp);
        builder.setInsertionPoint(enqueueOp.getOperation());
        auto prevEnqueue = enqueueOp.getPreviousTaskIdx();

        bool breakPointDetected = false;
        auto taskOp = start;
        // loop over all tasks covered by enqueue except last
        // last task shouldn't be checked as even if it has lastSecondaryTaskInExecutionGroup
        // attribute we don't need to split
        do {
            if (taskOp->hasAttr(VPUMI40XX::lastSecondaryTaskInExecutionGroup)) {
                breakPointDetected = true;
                auto breakingPoint = taskOp;

                auto newEnque = builder.create<VPURegMapped::EnqueueOp>(
                        start.getLoc(), enqueueOp.getType(), prevEnqueue, enqueueOp.getBarrier(),
                        /*previousTaskIdxOnSameBarrier*/ nullptr, enqueueOp.getTaskType(), start.getResult(),
                        breakingPoint.getResult());
                start = mlir::cast<VPURegMapped::TaskOpInterface>(breakingPoint).getNextTask();
                prevEnqueue = newEnque;

                if (enqueueOp == firstEnquOp) {
                    enqueueOp.getResult().replaceUsesWithIf(newEnque, [](mlir::OpOperand& operand) {
                        return mlir::isa<VPUMI40XX::MappedInferenceOp>(operand.getOwner());
                    });
                    firstEnquOp = newEnque;
                }
            }
            taskOp = taskOp.getNextTask();
        } while (taskOp != end);

        // If breakpoint was detected at least once. Old enqueue is no longer valid and replace it with
        // one that will properly take int oaccount already created ones
        if (breakPointDetected) {
            auto lastNewEnq = builder.create<VPURegMapped::EnqueueOp>(
                    start.getLoc(), enqueueOp.getType(), prevEnqueue, enqueueOp.getBarrier(),
                    /*previousTaskIdxOnSameBarrier*/ nullptr, enqueueOp.getTaskType(), start.getResult(),
                    end.getResult());

            enqueueOp.getResult().replaceUsesWithIf(lastNewEnq, [](mlir::OpOperand& operand) {
                return !mlir::isa<VPUMI40XX::MappedInferenceOp>(operand.getOwner());
            });

            enqueueOp->erase();
        }
    }
    firstEnqu = mpi.getWorkItemTasks();
    auto newCount = VPUMI40XX::reindexEnqueueList(mlir::cast<VPURegMapped::EnqueueOp>(firstEnqu.getDefiningOp()));
    mpi.setWorkItemCount(newCount);
}

}  // namespace

//
// createSplitEnqueueOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createSplitEnqueueOpsPass(Logger log) {
    return std::make_unique<SplitEnqueueOpsPass>(log);
}
