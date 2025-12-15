//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_LINKENQUEUETARGETS
#define GEN_PASS_DEF_LINKENQUEUETARGETS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class LinkEnqueueTargetsPass : public VPUMI40XX::impl::LinkEnqueueTargetsBase<LinkEnqueueTargetsPass> {
public:
    explicit LinkEnqueueTargetsPass(const WorkloadManagementMode workloadManagementMode, Logger log)
            : _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    void processEnqueueDmaOps(mlir::func::FuncOp netFunc);

    void processEnqueueOps(mlir::func::FuncOp netFunc);

    WorkloadManagementMode _workloadManagementMode;
};

// Identify enqueue DMAs ops and process all tasks. If multiple tasks of the same type are enqueued
// by the same enqueue DMA and those tasks support task linking then enable the link. Later pass when
// updating enqueue DMA will only enqueue the head of the linked list of tasks.
void LinkEnqueueTargetsPass::processEnqueueDmaOps(mlir::func::FuncOp netFunc) {
    auto mpi = VPUMI40XX::getMPI(netFunc);

    auto dmaTile0List0Head = mpi.getListHead(VPURegMapped::TaskType::DMA, 0, 0);
    if (!dmaTile0List0Head) {
        return;
    }

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = config::getTileExecutor(parentModule).getCount();
    const auto shavesCountPerTile = config::getAvailableExecutor(parentModule, VPU::ExecutorKind::SHAVE_ACT).getCount();

    auto firstDmaTile0List0Op = dmaTile0List0Head.getDefiningOp<VPUMI40XX::NNDMAOp>();
    auto enqueueDmasPerHwQueue = VPUMI40XX::getEnqueueDmaData(firstDmaTile0List0Op, _log);

    if (enqueueDmasPerHwQueue.empty()) {
        _log.trace("No Enqueue DMAs available for task linking");
        return;
    }

    const mlir::DenseSet<std::pair<VPURegMapped::TaskType, uint32_t>> taskTypesWithListCountPerTile = {
            {{VPURegMapped::TaskType::DPUVariant, 1},
             {VPURegMapped::TaskType::ActKernelInvocation, shavesCountPerTile}}};

    // Iterate over DPU/SHV tasks on each tile and list and check if multiple tasks are enqueued by the
    // same enqueue DMA. If yes, check if those tasks support task linking and if so, link them.
    for (uint32_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (const auto& [taskType, listCount] : taskTypesWithListCountPerTile) {
            for (uint32_t listIdx = 0; listIdx < listCount; listIdx++) {
                auto listHead = mpi.getListHead(taskType, tileIdx, listIdx);
                if (!listHead) {
                    continue;
                }

                _log.trace("Check task type {0} on tile {1}, list {2} if task linking is possible", taskType, tileIdx,
                           listIdx);
                auto taskOp = mlir::cast<VPURegMapped::TaskOpInterface>(listHead.getDefiningOp());

                auto hwQueue = VPUMI40XX::HwQueueType{taskType, tileIdx, listIdx};

                VPUX_THROW_WHEN(enqueueDmasPerHwQueue.find(hwQueue) == enqueueDmasPerHwQueue.end(),
                                "No Enqueue DMAs available for task type {0} on tile {1}, list {2}", taskType, tileIdx,
                                listIdx);
                // Initial tasks must be enqueued by first enqueue DMA for given HW queue type
                size_t curEnqueueIndex = 0;
                // Get start and end task range enqueued by enqueue DMA to understand what range of tasks
                // are processed by a single enqueue DMA
                auto [enqueueDmaStartIdx, enqueueDmaEndIdx, _] = enqueueDmasPerHwQueue[hwQueue][curEnqueueIndex];
                _log.trace("Enqueue DMA task range: {0} - {1}", enqueueDmaStartIdx, enqueueDmaEndIdx);

                // Iterate over tasks in the list and check if enabling link to previous is possible
                do {
                    auto taskInd = mlir::cast<VPURegMapped::IndexType>(taskOp.getResult().getType()).getValue();

                    // If current task index is greater than end index of current enqueue DMA it means that
                    // this task is enqueued by next enqueue DMA -> switch to next enqueue DMA
                    if (taskInd > enqueueDmaEndIdx) {
                        _log.trace("Task {0} is after end task {1} of current enqueue DMA. Move to next enqueue DMA op",
                                   taskInd, enqueueDmaEndIdx);
                        // Move to next enqueue DMA and get start and end indexes
                        curEnqueueIndex++;
                        VPUX_THROW_UNLESS(
                                curEnqueueIndex < enqueueDmasPerHwQueue[hwQueue].size(),
                                "No enqueue DMAs available for task type {0} on tile {1}, list {2} at index {3}",
                                taskType, tileIdx, listIdx, curEnqueueIndex);
                        enqueueDmaStartIdx = enqueueDmasPerHwQueue[hwQueue][curEnqueueIndex].startTaskIdx;
                        enqueueDmaEndIdx = enqueueDmasPerHwQueue[hwQueue][curEnqueueIndex].endTaskIdx;

                        _log.trace("Enqueue DMA task range: {0} - {1}", enqueueDmaStartIdx, enqueueDmaEndIdx);
                    }

                    // If task index is larger than first task enqueued by a DMA than link it to previous task
                    if (taskInd > enqueueDmaStartIdx) {
                        _log.trace("Link task {0} to previous", taskInd);
                        taskOp.linkToPreviousTask();
                    }

                    taskOp = taskOp.getNextTask();
                } while (taskOp);
            }
        }
    }
}

// Iterate over all enqueue ops. For each op that enqueues a range of tasks of the given type
// if the tasks support linking, enable task linking and update the op to enqueue only the head
// of the linked task chain
void LinkEnqueueTargetsPass::processEnqueueOps(mlir::func::FuncOp netFunc) {
    bool fifoPerShaveEngineEnabled = config::isFifoPerShaveEngineEnabled(netFunc);

    for (auto enqueue : netFunc.getOps<VPURegMapped::EnqueueOp>()) {
        if (enqueue.getStart() == enqueue.getEnd()) {
            continue;
        }

        auto start = mlir::cast<VPURegMapped::TaskOpInterface>(enqueue.getStart().getDefiningOp());
        auto end = mlir::cast<VPURegMapped::TaskOpInterface>(enqueue.getEnd().getDefiningOp());

        if (!end.supportsTaskLink()) {
            continue;
        }

        if (enqueue.getTaskType() != VPURegMapped::TaskType::ActKernelInvocation || fifoPerShaveEngineEnabled) {
            while (end != start) {
                end.linkToPreviousTask();
                end = end.getPreviousTask();
            }

            // if we've hard-linked all n+1th tasks, then we only have to enqueue the first task
            enqueue.getEndMutable().assign(start.getResult());
        } else {
            // shave kernels are special in a way we have 2 link lists per enqueue instead of 1

            const auto firstInvocationIndex = start.getIndexType().getValue();
            const auto lastInvocationIndex = end.getIndexType().getValue();
            const auto invocationsCount = lastInvocationIndex - firstInvocationIndex + 1;

            if (invocationsCount >= 3) {
                // if you have enough invocations link them in round-robin fashion
                // invo0, invo1, invo2(prev:invo0), invo3(prev:invo1), invo4(prev:invo2), ...
                auto head0 = start;
                auto head1 = start.getNextTask();

                // we still need to enqueue both heads
                enqueue.getEndMutable().assign(head1.getResult());

                // minimize amount of getNextTask call as it may be expensive
                for (auto i : irange((invocationsCount - 1) / 2)) {
                    const auto next0Idx = 2 * (i + 1);
                    assert(next0Idx < invocationsCount);

                    auto next0 = head1.getNextTask();
                    next0.linkToTask(VPURegMapped::IndexTypeAttr::get(netFunc.getContext(), head0.getIndexType()));

                    const auto next1Idx = next0Idx + 1;
                    assert(next1Idx <= invocationsCount);

                    if (next1Idx == invocationsCount) {
                        continue;
                    }

                    auto next1 = next0.getNextTask();
                    next1.linkToTask(VPURegMapped::IndexTypeAttr::get(netFunc.getContext(), head1.getIndexType()));

                    head0 = next0;
                    head1 = next1;
                }
            }
        }
    }
}

void LinkEnqueueTargetsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    if (workloadManagementModeOpt.hasValue()) {
        _workloadManagementMode = workloadManagementModeOpt.getValue();
    }

    if (_workloadManagementMode == WorkloadManagementMode::FWLM_V1_PAGES) {
        processEnqueueDmaOps(netFunc);
    }

    processEnqueueOps(netFunc);
}
}  // namespace

//
// createLinkEnqueueTargetsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createLinkEnqueueTargetsPass(WorkloadManagementMode workloadManagementMode,
                                                                          Logger log) {
    return std::make_unique<LinkEnqueueTargetsPass>(workloadManagementMode, log);
}
