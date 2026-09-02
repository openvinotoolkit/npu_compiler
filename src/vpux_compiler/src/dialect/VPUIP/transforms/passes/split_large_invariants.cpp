//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <algorithm>
#include <climits>
#include <map>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_SPLITLARGEINVARIANTS
#define GEN_PASS_DEF_SPLITLARGEINVARIANTS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

namespace vpux::VPUIP {
namespace {

//
// SplitLargeInvariantsPass
//
// This pass mitigates runtime problem of handling preemption in case descriptor group contains only single invariant.
// It splits VPURT.Task-wrapped VPUIP.NCEClusterTask operations
// when their number of DPU variants exceeds the architecture-specific threshold
// derived from NPU constraints.
// New VPURT.Tasks are created such that each doesn't exceed the threshold anymore and overall workload stays the same:
//  - all the outputs and inputs of the new NCEClusterTasks refer to the original ones
//  - all VPURT.Tasks are chained by barriers to preserve the order
//  - the first task waits on the original wait barriers and the last task updates the original update barriers
//
// EXAMPLE: 8 variants with threshold=4
//
//                 BEFORE                                            AFTER
// [AwaitBarrier0] [...] [AwaitBarrierX]             [AwaitBarrier0] [...] [AwaitBarrierX]
//          |        |         |                              |        |         |
// ┌─────────────────────────────────────┐          ┌─────────────────────────────────────┐
// │  VPURT.Task                         │          │  VPURT.Task #0                      │
// │ ┌─────────────────────────────────┐ │          │ ┌─────────────────────────────────┐ │
// │ │  [Input0]   [...]    [InputN]   │ │          │ │  [Input0]   [...]    [InputN]   │ │
// │ │       │       │        │        │ │          │ │       │       │        │        │ │
// │ │   ┌─────────────────────────┐   │ │          │ │   ┌─────────────────────────┐   │ │
// │ │   │ NCEClusterTask          │   │ │          │ │   │ NCEClusterTask #0       │   │ │
// │ │   │─────────────────────────│   │ │          │ │   │─────────────────────────│   │ │
// │ │   │ 8 DPU variants (V0-V7)  │   │ │          │ │   │ 4 DPU variants (V0-V3)  │   │ │
// │ │   │ V0 V1 V2 V3 V4 V5 V6 V7 │   │ │          │ │   │      V0 V1 V2 V3        │   │ │
// │ │   └─────────────────────────┘   │ │          │ │   └─────────────────────────┘   │ │
// │ │      │        │        │        │ │          │ │      │        │        │        │ │
// │ │  [Output0]  [...]  [OutputM]    │ │          │ │  [Output0]  [...]  [OutputM]    │ │
// │ └─────────────────────────────────┘ │          │ └─────────────────────────────────┘ │
// └─────────────────────────────────────┘          └─────────────────┼───────────────────┘
//         |          |          |                                    |
// [UpdateBarrier0] [...] [UpdateBarrierY]                            |
//                                                         [VPURT.VirtualBarrier]
//                                                 (updated by Task #0, awaited by Task #1)
//                                                                    |
//                                                                    v
//                                                  ┌─────────────────┼───────────────────┐
//                                                  │  VPURT.Task #1                      │
//                                                  │ ┌─────────────────────────────────┐ │
//                                                  │ │  [Input0]   [...]    [InputN]   │ │
//                                                  │ │       │       │        │        │ │
//                                                  │ │   ┌─────────────────────────┐   │ │
//                                                  │ │   │ NCEClusterTask #1       │   │ │
//                                                  │ │   │─────────────────────────│   │ │
//                                                  │ │   │ 4 DPU variants (V4-V7)  │   │ │
//                                                  │ │   │       V4 V5 V6 V7       │   │ │
//                                                  │ │   └─────────────────────────┘   │ │
//                                                  │ │      │        │        │        │ │
//                                                  │ │  [Output0]  [...]  [OutputM]    │ │
//                                                  │ └─────────────────────────────────┘ │
//                                                  └─────────────────┼───────────────────┘
//                                                         |          |          |
//                                                    [UpdateBarrier0] [...] [UpdateBarrierY]

class SplitLargeInvariantsPass final : public impl::SplitLargeInvariantsBase<SplitLargeInvariantsPass> {
public:
    explicit SplitLargeInvariantsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void safeRunOnFunc() final;

private:
    static void keepOnlyVariantChunk(VPUIP::NCEClusterTaskOp nceTask, int64_t begin, int64_t end);
    static void splitNceTaskByVariants(VPURT::TaskOp taskOp, unsigned variantsPerChunk);
};

void SplitLargeInvariantsPass::keepOnlyVariantChunk(VPUIP::NCEClusterTaskOp nceTask, int64_t begin, int64_t end) {
    SmallVector<VPUIP::DPUTaskOp> variants = llvm::to_vector(nceTask.getVariants().getOps<VPUIP::DPUTaskOp>());

    for (int64_t idx = 0; idx < static_cast<int64_t>(variants.size()); ++idx) {
        if (idx >= begin && idx < end) {
            continue;
        }
        variants[idx].erase();
    }
}

void SplitLargeInvariantsPass::splitNceTaskByVariants(VPURT::TaskOp taskOp, unsigned variantsPerChunk) {
    auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp());
    VPUX_THROW_WHEN(nceTask == nullptr, "Expected VPURT task to wrap VPUIP.NCEClusterTask");

    const auto numVariants = nceTask.getNumVariants();
    const auto numChunks = (numVariants + variantsPerChunk - 1) / variantsPerChunk;
    if (numChunks <= 1) {
        return;
    }

    SmallVector<mlir::Value> originalWaitBarriers(taskOp.getWaitBarriers().begin(), taskOp.getWaitBarriers().end());
    SmallVector<mlir::Value> originalUpdateBarriers(taskOp.getUpdateBarriers().begin(),
                                                    taskOp.getUpdateBarriers().end());
    mlir::OpBuilder builder(taskOp);

    SmallVector<mlir::Value> chainBarriers;
    chainBarriers.reserve(static_cast<size_t>(numChunks - 1));
    for (int64_t idx = 0; idx < numChunks - 1; ++idx) {
        auto chainBarrierOp = builder.create<VPURT::DeclareVirtualBarrierOp>(taskOp->getLoc());
        chainBarriers.push_back(chainBarrierOp.getResult());
    }

    for (int64_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
        auto clonedTaskOp = mlir::cast<VPURT::TaskOp>(builder.clone(*taskOp.getOperation()));
        auto clonedNceTask = mlir::cast<VPUIP::NCEClusterTaskOp>(clonedTaskOp.getInnerTaskOp());
        extendOpLoc(clonedNceTask, "part_{0}", chunkIdx);
        if (chunkIdx != 0) {
            clonedNceTask.removeProfilingMetadataAttr();
        }

        const auto begin = chunkIdx * variantsPerChunk;
        const auto end = std::min(begin + variantsPerChunk, numVariants);
        keepOnlyVariantChunk(clonedNceTask, begin, end);

        SmallVector<mlir::Value> waitBarriers;
        SmallVector<mlir::Value> updateBarriers;

        if (chunkIdx == 0) {
            waitBarriers = originalWaitBarriers;
        } else {
            waitBarriers.push_back(chainBarriers[static_cast<size_t>(chunkIdx - 1)]);
        }

        if (chunkIdx == numChunks - 1) {
            updateBarriers = originalUpdateBarriers;
        } else {
            updateBarriers.push_back(chainBarriers[static_cast<size_t>(chunkIdx)]);
        }

        clonedTaskOp.getWaitBarriersMutable().assign(waitBarriers);
        clonedTaskOp.getUpdateBarriersMutable().assign(updateBarriers);

        // Keep cloned tasks in chunk order.
        builder.setInsertionPointAfter(clonedTaskOp);
    }

    taskOp.erase();
}

void SplitLargeInvariantsPass::safeRunOnFunc() {
    auto func = getOperation();
    const auto useVariantCount = maxVariantCount.hasValue();
    const auto variantsPerNceClusterTask = useVariantCount ? checked_cast<size_t>(maxVariantCount.getValue()) / 2
                                                           : getMaxNumberOfDpuVariantsPerInvariant(func) / 2;
    VPUX_THROW_UNLESS(variantsPerNceClusterTask >= 1,
                      "Invalid NCEClusterTask split threshold: computed variants-per-task threshold must be >= 1, "
                      "but got {0}",
                      variantsPerNceClusterTask);

    SmallVector<VPURT::TaskOp> tasksToSplit;

    func.walk([&](VPURT::TaskOp taskOp) {
        auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp());
        if (nceTask == nullptr) {
            return;
        }

        if (nceTask.getNumVariants() > (int64_t)variantsPerNceClusterTask) {
            tasksToSplit.push_back(taskOp);
        }
    });

    if (tasksToSplit.empty()) {
        _log.trace("No NCEClusterTaskOps with variants count exceeding the threshold of {0} were found",
                   variantsPerNceClusterTask);
        return;
    }

    _log.trace("Found {0} NCEClusterTaskOps with variants count exceeding the threshold of {1}, splitting them",
               tasksToSplit.size(), variantsPerNceClusterTask);

    // Stamp each task with a unique ID before splitting so all resulting chunks
    // can be correlated back to the original invariant after the pass.
    for (auto&& [idx, taskOp] : llvm::enumerate(tasksToSplit)) {
        auto nceTask = mlir::cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp());
        nceTask.setSplitIdAttr(vpux::getIntAttr(nceTask.getContext(), static_cast<int64_t>(idx)));
    }

    auto nestedLog = _log.nest();
    for (auto taskOp : tasksToSplit) {
        auto nceTask = mlir::cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp());
        nestedLog.trace("Splitting NCEClusterTaskOp at '{0}' with '{1}' variants into '{2}' NCEClusterTaskOps",
                        nceTask.getLoc(), nceTask.getNumVariants(),
                        (nceTask.getNumVariants() + variantsPerNceClusterTask - 1) / variantsPerNceClusterTask);
        splitNceTaskByVariants(taskOp, variantsPerNceClusterTask);
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> createSplitLargeInvariantsPass(Logger log) {
    return std::make_unique<SplitLargeInvariantsPass>(log);
}

}  // namespace vpux::VPUIP
