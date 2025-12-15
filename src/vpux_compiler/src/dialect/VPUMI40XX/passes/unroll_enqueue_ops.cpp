//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_UNROLLENQUEUEOPS
#define GEN_PASS_DEF_UNROLLENQUEUEOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {
class UnrollEnqueuePattern final : public mlir::OpRewritePattern<VPURegMapped::EnqueueOp> {
public:
    UnrollEnqueuePattern(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURegMapped::EnqueueOp>(ctx), _log(log) {
        setDebugName("UnrollEnqueuePatternRewriter");
    }

    mlir::LogicalResult match(VPURegMapped::EnqueueOp enqueueOp) const final;
    void rewrite(VPURegMapped::EnqueueOp enqueueOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UnrollEnqueuePattern::match(VPURegMapped::EnqueueOp enqueueOp) const {
    // only care to rewrite IF and only IF we have more than one task to enqueue
    if (enqueueOp.getStart() != enqueueOp.getEnd()) {
        return mlir::success();
    } else {
        return mlir::failure();
    }
}

void UnrollEnqueuePattern::rewrite(VPURegMapped::EnqueueOp enqueueOp, mlir::PatternRewriter& rewriter) const {
    auto start = mlir::cast<VPURegMapped::TaskOpInterface>(enqueueOp.getStart().getDefiningOp());
    auto end = mlir::cast<VPURegMapped::TaskOpInterface>(enqueueOp.getEnd().getDefiningOp());

    // enqueue ops start-end encapsulate a range, and have to iterate over the end included
    auto iterEnd = VPUMI40XX::getNextOp(end);
    mlir::Value last = enqueueOp.getPreviousTaskIdx();
    mlir::Value first;
    for (auto taskOp = start; taskOp != iterEnd; taskOp = VPUMI40XX::getNextOp(taskOp)) {
        auto newEnqueue = rewriter.create<VPURegMapped::EnqueueOp>(
                taskOp.getLoc(), enqueueOp.getType(), last, enqueueOp.getBarrier(),
                /*previousTaskIdxOnSameBarrier*/ nullptr, enqueueOp.getTaskType(), taskOp.getResult(),
                taskOp.getResult());
        last = newEnqueue.getResult();

        if (!first) {
            first = last;
        }
    }

    enqueueOp.getResult().replaceUsesWithIf(first, [](mlir::OpOperand& operand) {
        return mlir::isa<VPUMI40XX::MappedInferenceOp>(operand.getOwner());
    });

    enqueueOp.getResult().replaceUsesWithIf(last, [](mlir::OpOperand& operand) {
        return !mlir::isa<VPUMI40XX::MappedInferenceOp>(operand.getOwner());
    });

    rewriter.eraseOp(enqueueOp.getOperation());
}

class UnrollEnqueueOpsPass : public VPUMI40XX::impl::UnrollEnqueueOpsBase<UnrollEnqueueOpsPass> {
public:
    explicit UnrollEnqueueOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollEnqueueOpsPass::safeRunOnFunc() {
    auto ctx = &getContext();
    auto netFunc = getOperation();

    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto firstEnqu = mpi.getWorkItemTasks();
    // if we don't have workItems skip an iteration over the whole IR
    if (!firstEnqu) {
        return;
    }

    mlir::RewritePatternSet patterns(ctx);
    patterns.add<UnrollEnqueuePattern>(ctx, _log);

    collectOpsAndApplyPatterns(netFunc, std::move(patterns));

    // pattern rewriter may have changed the firstEnqu
    firstEnqu = mpi.getWorkItemTasks();
    auto newCount = VPUMI40XX::reindexEnqueueList(mlir::cast<VPURegMapped::EnqueueOp>(firstEnqu.getDefiningOp()));
    mpi.setWorkItemCount(newCount);
}

}  // namespace

//
// createUnrollEnqueueOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createUnrollEnqueueOpsPass(Logger log) {
    return std::make_unique<UnrollEnqueueOpsPass>(log);
}
