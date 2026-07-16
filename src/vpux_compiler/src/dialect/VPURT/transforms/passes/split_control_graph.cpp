//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_SPLITCONTROLGRAPH
#define GEN_PASS_DEF_SPLITCONTROLGRAPH
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

using TaskRange = std::pair<size_t, size_t>;
using ExcludedTaskRanges = SmallVector<TaskRange>;
std::optional<int64_t> getLogicalTaskIdxForShvDmaTask(VPURT::TaskOp taskOp) {
    if (taskOp.getExecutorKind() != config::ExecutorKind::SHAVE_ACT) {
        return std::nullopt;
    }
    auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp());
    if (swKernelOp == nullptr || !VPUIP::isIoDmaSwKernel(swKernelOp)) {
        return std::nullopt;
    }
    auto logicalTaskAttr = swKernelOp->getAttrOfType<mlir::IntegerAttr>(VPUIP::LOGICAL_TASK_INDEX_ATTR_NAME);
    if (logicalTaskAttr == nullptr) {
        return std::nullopt;
    }
    return logicalTaskAttr.getInt();
}

// Returns the initial task indices for sync points based on the block size.
SmallVector<size_t> getInitialSyncTaskIndices(const size_t tasksSize, const size_t blockSize) {
    SmallVector<size_t> syncTaskIndices;
    if (tasksSize < 2 || blockSize == 0) {
        return syncTaskIndices;
    }

    syncTaskIndices.reserve(tasksSize / blockSize);
    for (size_t taskIdx = blockSize - 1; taskIdx + 1 < tasksSize; taskIdx += blockSize) {
        syncTaskIndices.push_back(taskIdx);
    }

    return syncTaskIndices;
}

//
// Exclude range: the range of tasks we don't want to be chosen as sync-task
//
ExcludedTaskRanges getExcludedRange(mlir::ArrayRef<VPURT::TaskOp> allTasks) {
    ExcludedTaskRanges excludedRanges;
    std::optional<int64_t> currentLogicalTaskIdx;
    std::optional<TaskRange> currentRange;

    for (const auto& [taskIdx, taskOp] : llvm::enumerate(allTasks)) {
        auto logicalTaskIdx = getLogicalTaskIdxForShvDmaTask(taskOp);
        if (!logicalTaskIdx) {
            continue;
        }

        if (!currentLogicalTaskIdx.has_value() || currentLogicalTaskIdx.value() != logicalTaskIdx.value()) {
            if (currentRange.has_value()) {
                excludedRanges.push_back(currentRange.value());
            }
            currentLogicalTaskIdx = logicalTaskIdx;
            currentRange = TaskRange{taskIdx, taskIdx};
            continue;
        }

        currentRange->second = taskIdx;
    }

    if (currentRange.has_value()) {
        excludedRanges.push_back(currentRange.value());
    }

    return excludedRanges;
}

// Adjust sync task indices to ensure they are outside the excluded ranges and within valid bounds
// If the sync-task chosen: Y , which is in excluded range [X, Z], then we will adjust the next sync task to be Z+1
void adjustSyncTaskIndices(SmallVector<size_t>& syncTaskIndices, const ExcludedTaskRanges& excludedRange,
                           const size_t tasksSize, Logger log) {
    auto adjustSyncPointOutsideExcludedRanges = [&](size_t syncIdx, size_t minSyncIdx) {
        if (tasksSize < 2) {
            return tasksSize;
        }

        // Keep sync points monotonic across blocks by honoring the minimum allowed index.
        size_t adjustedSyncIdx = std::max(syncIdx, minSyncIdx);
        while (adjustedSyncIdx < tasksSize - 1) {
            // Re-check after each shift: moving to rangeEnd + 1 can land in another excluded range.
            // Assume SHV submit DMA tasks for one logical task form a single contiguous range.
            // Under that assumption, one task index can belong to at most one excluded range.
            std::optional<size_t> rangeEnd;
            // Excluded ranges are ordered by start index, so once rangeBegin exceeds the candidate index
            // no later range can contain it.
            for (const auto& [rangeBegin, currentRangeEnd] : excludedRange) {
                if (rangeBegin > adjustedSyncIdx) {
                    break;
                }
                if (adjustedSyncIdx >= rangeBegin && adjustedSyncIdx <= currentRangeEnd) {
                    rangeEnd = currentRangeEnd;
                    log.trace("Sync task '{0}' is inside excluded range [{1}, {2}]", adjustedSyncIdx, rangeBegin,
                              currentRangeEnd);
                    break;
                }
            }

            if (!rangeEnd.has_value()) {
                return adjustedSyncIdx;
            }

            const auto nextSyncIdx = rangeEnd.value() + 1;
            log.trace("Adjust sync task index from '{0}' to '{1}'", adjustedSyncIdx, nextSyncIdx);
            adjustedSyncIdx = nextSyncIdx;
        }

        return tasksSize;
    };

    // Collect the final sync-point list after shifting each candidate out of excluded ranges.
    SmallVector<size_t> adjustedSyncTaskIndices;
    adjustedSyncTaskIndices.reserve(syncTaskIndices.size());
    for (const auto syncIdx : syncTaskIndices) {
        // Enforce strictly increasing sync points after any prior adjustment.
        const auto minSyncIdx = adjustedSyncTaskIndices.empty() ? 0 : (adjustedSyncTaskIndices.back() + 1);
        const auto adjustedSyncIdx = adjustSyncPointOutsideExcludedRanges(syncIdx, minSyncIdx);
        // The last task cannot be used as a sync point because there is no following block.
        if (adjustedSyncIdx >= tasksSize - 1) {
            log.trace("Skipping sync task index '{0}' after adjustment because it is out of valid range", syncIdx);
            continue;
        }
        adjustedSyncTaskIndices.push_back(adjustedSyncIdx);
    }

    syncTaskIndices = std::move(adjustedSyncTaskIndices);
}

SmallVector<size_t> getSyncTaskIndices(mlir::ArrayRef<VPURT::TaskOp> allTasks, const size_t blockSize,
                                       config::ArchKind arch, Logger log) {
    // For all platforms we always compute the initial split based on block size.
    auto syncTaskIndices = getInitialSyncTaskIndices(allTasks.size(), blockSize);

    // Platform-specific behavior:
    // - NPU40XX/NPU50XX: compute excluded ranges and adjust sync points so that
    //   selected sync tasks do not fall into SHV IO-DMA logical-task ranges.
    // - Other platforms: keep the initial split as-is (no excluded-range step).
    const bool needsExcludedRangeAdjust = (arch == config::ArchKind::NPU40XX) || (arch == config::ArchKind::NPU50XX);
    if (!needsExcludedRangeAdjust) {
        return syncTaskIndices;
    }

    auto excludedRange = getExcludedRange(allTasks);
    if (!excludedRange.empty()) {
        adjustSyncTaskIndices(syncTaskIndices, excludedRange, allTasks.size(), log);
    }
    return syncTaskIndices;
}

class SplitControlGraphPass final : public VPURT::impl::SplitControlGraphBase<SplitControlGraphPass> {
public:
    explicit SplitControlGraphPass(const int controlGraphSplitBlockSize, Logger log)
            : _controlGraphSplitBlockSize(static_cast<size_t>(controlGraphSplitBlockSize)) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    size_t _controlGraphSplitBlockSize;
};

void SplitControlGraphPass::safeRunOnFunc() {
    auto func = getOperation();
    auto arch = config::getArch(func);
    if (blockSize.hasValue()) {
        _controlGraphSplitBlockSize = blockSize.getValue();
    }

    auto allTasks = to_small_vector(func.getOps<VPURT::TaskOp>());
    _log.trace("Requested block size - '{0}', number of tasks in the model - '{1}'", _controlGraphSplitBlockSize,
               allTasks.size());
    if (allTasks.size() <= _controlGraphSplitBlockSize) {
        _log.trace("Split not needed");
        return;
    }

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    _log.trace("Original number of barriers {0}", barrierInfo.getNumOfBarrierOps());
    auto syncPointTaskIndices = getSyncTaskIndices(allTasks, _controlGraphSplitBlockSize, arch, _log);
    if (syncPointTaskIndices.empty()) {
        _log.trace("No valid sync points computed, skipping split");
        return;
    }

    barrierInfo.splitControlGraphToBlocks(syncPointTaskIndices);

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");

    _log.trace("New number of barriers {0}", barrierInfo.getNumOfBarrierOps());
    barrierInfo.clearAttributes();

    VPURT::postProcessBarrierOps(func);
}

}  // namespace

//
// createSplitControlGraphPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createSplitControlGraphPass(const int controlGraphSplitBlockSize, Logger log) {
    return std::make_unique<SplitControlGraphPass>(controlGraphSplitBlockSize, log);
}
