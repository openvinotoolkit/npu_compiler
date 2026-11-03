//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"

namespace vpux {
namespace VPUMI40XX {

//
// AddEnqueue Utils
//

bool contains(const llvm::SmallVector<mlir::Value>& vec, const mlir::Value& element) {
    return std::find(vec.begin(), vec.end(), element) != vec.end();
};

VPUMI40XX::ConfigureBarrierOp getBarrierOp(mlir::Operation* op) {
    if (op == nullptr) {
        return nullptr;
    }

    auto maybeBarrier = mlir::dyn_cast_or_null<VPUMI40XX::ConfigureBarrierOp>(op);
    return maybeBarrier;
}

size_t getBarrierIndex(mlir::Operation* op) {
    auto barrierOp = getBarrierOp(op);
    VPUX_THROW_WHEN(barrierOp == nullptr, "Expected barrier: got {0}", op);
    return barrierOp.getType().getValue();
};

bool taskOpComparator(mlir::Operation* lhs, mlir::Operation* rhs) {
    auto lhsTask = mlir::cast<VPURegMapped::TaskOpInterface>(lhs);
    auto rhsTask = mlir::cast<VPURegMapped::TaskOpInterface>(rhs);
    return lhsTask.getIndexType().getValue() < rhsTask.getIndexType().getValue();
}

void reindexEnqueueOps(llvm::SmallVector<VPURegMapped::EnqueueOp> enquOps) {
    if (enquOps.size() == 0) {
        return;
    }

    auto ctx = enquOps[0].getContext();
    auto index = [ctx](auto taskIdx) {
        return VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(taskIdx));
    };

    enquOps[0].getResult().setType(index(0));
    enquOps[0].getPreviousTaskIdxMutable().clear();

    for (size_t i = 1; i < enquOps.size(); i++) {
        auto enqu = enquOps[i];
        enqu.getResult().setType(index(i));
        enqu.getPreviousTaskIdxMutable().assign(enquOps[i - 1]);
    }

    return;
}

void dfs(mlir::Value val, llvm::SetVector<mlir::Value>& visited, size_t indexMax) {
    visited.insert(val);
    for (auto user : val.getUsers()) {
        auto barOp = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(user);
        if (!barOp) {
            continue;
        }

        auto bar = barOp.getResult();

        auto barIndex = mlir::cast<vpux::VPURegMapped::IndexType>(bar.getType()).getValue();
        if (barIndex > indexMax) {
            continue;
        }

        if (!visited.contains(bar)) {
            dfs(bar, visited, indexMax);
        }
    }
}

VPURegMapped::TaskOpInterface getNextOp(VPURegMapped::TaskOpInterface op) {
    auto users = op.getResult().getUsers();
    auto nexOpIt = llvm::find_if(users, [&op](mlir::Operation* user) {
        auto nextTask = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(user);
        return nextTask && (nextTask.getTaskType() == op.getTaskType()) && (nextTask.getPreviousTask() == op);
    });

    op = nexOpIt != users.end() ? mlir::cast<VPURegMapped::TaskOpInterface>(*nexOpIt) : nullptr;
    return op;
}

llvm::SmallVector<mlir::Value> getPreviousUsages(mlir::ValueRange barrs) {
    llvm::SmallDenseSet<mlir::Value> seenBarriers;
    llvm::SmallVector<mlir::Value> previousUsages;

    for (auto barr : barrs) {
        auto barrOp = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(barr.getDefiningOp());
        if (barrOp == nullptr) {
            continue;
        }

        const auto previousSameId = barrOp.getPreviousSameId();
        if (previousSameId != nullptr && !seenBarriers.contains(previousSameId)) {
            seenBarriers.insert(previousSameId);
            previousUsages.push_back(previousSameId);
        }
    }

    return previousUsages;
}

// TODO E#132327: ned to figure out a clean way to get barriers purely from taskOpInterface
VPUMI40XX::ExecutableTaskOpInterface getBarrieredOp(VPURegMapped::TaskOpInterface primary,
                                                    VPURegMapped::TaskOpInterface secondary) {
    if (primary.getTaskType() == VPURegMapped::TaskType::DPUInvariant) {
        return mlir::cast<VPUMI40XX::ExecutableTaskOpInterface>(primary.getOperation());
    } else if (primary.getTaskType() == VPURegMapped::TaskType::ActKernelRange) {
        return mlir::cast<VPUMI40XX::ExecutableTaskOpInterface>(secondary.getOperation());
    } else {
        VPUX_THROW("Unknown TaskType for pair {0} {1}", primary.getResult(), secondary.getResult());
        return nullptr;
    }

    return nullptr;
}

//
// ConfigureBarrier Utils
//

// Set next_same_id attribute and previousSameId operand for each ConfigureBarrier operation,
// and here we don't need to verify barrier if it has same previousSameId with other same physical id barrier,
// because the previousSameId operand is continuously increasing
void setBarrierIDs(mlir::MLIRContext* ctx, mlir::func::FuncOp funcOp) {
    auto MAX_PID = VPUIP::getNumAvailableBarriers(funcOp);
    std::vector<VPUMI40XX::ConfigureBarrierOp> lastAssignedBarrier(MAX_PID);

    for (auto op : funcOp.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        auto vid = mlir::cast<vpux::VPURegMapped::IndexType>(op.getOperation()->getResult(0).getType()).getValue();
        auto pid = op.getId();

        auto& lastBarrier = lastAssignedBarrier[pid];
        if (lastBarrier != nullptr) {
            op.getPreviousSameIdMutable().assign(lastBarrier.getOperation()->getResult(0));
            lastBarrier.setNextSameIdAttr(
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed), vid));
        }

        lastBarrier = op;
    }
}

//
// Log Fetch Tasks
//

VPUMI40XX::NNDMAOp getPreviousDMAWithBarriers(VPURegMapped::TaskOpInterface taskOpInterface) {
    while (taskOpInterface != nullptr) {
        if (mlir::isa<VPURegMapped::FetchTaskOp>(taskOpInterface.getOperation())) {
            taskOpInterface = taskOpInterface.getPreviousTask();
            continue;
        }

        if (auto previousDma = mlir::dyn_cast<VPUMI40XX::NNDMAOp>(taskOpInterface.getOperation())) {
            return previousDma;
        }
        taskOpInterface = taskOpInterface.getPreviousTask();
    }
    return nullptr;
}

void logFetchOpsDetails(mlir::func::FuncOp netFunc, Logger log) {
    auto fetchTaskOps = netFunc.getOps<VPURegMapped::FetchTaskOp>();
    SmallVector<FetchTaskDetails> groupedTasks;
    std::map<std::string, std::map<size_t, SmallVector<FetchTaskDetails>>> groupedTasksMap;

    for (auto fetchTaskOp : fetchTaskOps) {
        auto primaryTaskOpStart =
                mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getPrimaryStart().getDefiningOp());

        auto primaryTaskOpEnd = mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getPrimaryEnd().getDefiningOp());
        auto secondaryTaskOpStart =
                mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getSecondaryStart().getDefiningOp());
        auto secondaryTaskOpEnd =
                mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getSecondaryEnd().getDefiningOp());

        size_t executionGroupIndex = 0;
        if (fetchTaskOp.getAssociatedExecutionGroupIndex().has_value()) {
            executionGroupIndex = fetchTaskOp.getAssociatedExecutionGroupIndex().value();
        }
        size_t tileIndex = 0;
        if (fetchTaskOp.getAssociatedTileIndex().has_value()) {
            tileIndex = fetchTaskOp.getAssociatedTileIndex().value();
        }

        std::string taskType = "N/A";
        if (fetchTaskOp.getAssociatedTaskType().has_value()) {
            taskType = (fetchTaskOp.getAssociatedTaskType().value() == VPURegMapped::TaskType::ActKernelInvocation ||
                        fetchTaskOp.getAssociatedTaskType().value() == VPURegMapped::TaskType::ActKernelRange)
                               ? "SHV"
                               : "DPU";
        }

        size_t taskIndex = fetchTaskOp.getIndexType().getValue();
        size_t dmaWithBarriers = SIZE_MAX;
        size_t barrierIdx = 0;

        if (fetchTaskOp.getPreviousTask() != nullptr) {
            auto taskOpInterface =
                    mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getPreviousTask().getDefiningOp());
            if (auto prevDmaWithBarriers = getPreviousDMAWithBarriers(taskOpInterface)) {
                dmaWithBarriers = prevDmaWithBarriers.getIndexType().getValue();

                for (auto updateBarr : prevDmaWithBarriers.getWaitBarriers()) {
                    auto barrOp = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(updateBarr.getDefiningOp());
                    barrierIdx = std::max(barrierIdx, static_cast<size_t>(barrOp.getType().getValue()));
                }
            }
        }

        FetchTaskDetails taskDetails = {tileIndex,
                                        taskIndex,
                                        dmaWithBarriers,
                                        barrierIdx,
                                        primaryTaskOpStart.getIndexType().getValue(),
                                        primaryTaskOpEnd.getIndexType().getValue(),
                                        secondaryTaskOpStart.getIndexType().getValue(),
                                        secondaryTaskOpEnd.getIndexType().getValue(),
                                        taskType,
                                        executionGroupIndex};

        groupedTasksMap[taskType][executionGroupIndex].push_back(taskDetails);
    }

    // Iterate and print in the required format
    for (const auto& taskTypePair : groupedTasksMap) {
        const std::string& taskType = taskTypePair.first;
        log.trace("{0}", taskType);

        for (const auto& execGroupPair : taskTypePair.second) {
            size_t execGroup = execGroupPair.first;
            log.nest().trace("Exec Group {0} - {1}", execGroup, (execGroup % 2 == 0 ? "BUF A" : "BUF B"));

            const auto& tasksInGroup = execGroupPair.second;

            for (const auto& task : tasksInGroup) {
                if (taskType == "DPU") {
                    log.nest(2).trace(
                            "Tile {0} - FetchIndex: {1}, Prev DMA: {2}, BarrierBlockingFetch: {3}, Invariant: "
                            "{4}-{5}, Variant: {6}-{7}",
                            task.tileIndex, task.taskIndex,
                            (task.dmaWithBarriers == SIZE_MAX ? "N/A" : std::to_string(task.dmaWithBarriers)),
                            (task.barrierIdx == SIZE_MAX ? "N/A" : std::to_string(task.barrierIdx)), task.primaryStart,
                            task.primaryEnd, task.secondaryStart, task.secondaryEnd);
                } else {
                    log.nest(2).trace(
                            "Tile {0} - FetchIndex: {1}, Prev DMA: {2}, BarrierBlockingFetch: {3}, Invocation: "
                            "{4}-{5}, Range: {6}-{7}",
                            task.tileIndex, task.taskIndex,
                            (task.dmaWithBarriers == SIZE_MAX ? "N/A" : std::to_string(task.dmaWithBarriers)),
                            (task.barrierIdx == SIZE_MAX ? "N/A" : std::to_string(task.barrierIdx)), task.primaryStart,
                            task.primaryEnd, task.secondaryStart, task.secondaryEnd);
                }
            }
        }
    }
}

// Check if there are any Enqueue DMAs present in the schedule and gather data
// about them
// Map:
// - key - HwQueueType - {taskType, tileIdx, listIdx}
// - value - vector of {startTaskIndex, endTaskIndex, dmaOp}
mlir::DenseMap<VPUMI40XX::HwQueueType, SmallVector<EnqDmaInfo>> getEnqueueDmaData(
        VPUMI40XX::NNDMAOp firstDmaTile0List0Op, Logger log) {
    mlir::DenseMap<VPUMI40XX::HwQueueType, SmallVector<EnqDmaInfo>> enqueueDmasPerHwQueue;

    // Iterate over all DMAs on tile0 list 0 (Port 0, Channel DDR) and check for EnqueueDma attribute
    auto dmaTile0List0Task = firstDmaTile0List0Op;
    do {
        auto enqueueDmaAttr = dmaTile0List0Task.getEnqueueDmaAttr();
        if (enqueueDmaAttr.has_value()) {
            auto taskType = VPUMI40XX::convertExecutorKindToExecutableTaskType(
                    enqueueDmaAttr.value().getTargetExecutorKindAttr().getValue());
            auto tileIdx = static_cast<uint32_t>(enqueueDmaAttr.value().getTileIdx().getValue().getSExtValue());
            auto listIdx = static_cast<uint32_t>(enqueueDmaAttr.value().getListIdx().getValue().getSExtValue());
            auto hwQueue = VPUMI40XX::HwQueueType{taskType, tileIdx, listIdx};

            auto startTaskIdx = enqueueDmaAttr.value().getStartTaskIdx().getValue().getSExtValue();
            auto endTaskIdx = enqueueDmaAttr.value().getEndTaskIdx().getValue().getSExtValue();
            enqueueDmasPerHwQueue[hwQueue].push_back(EnqDmaInfo{startTaskIdx, endTaskIdx, dmaTile0List0Task});
            log.trace("Found Enqueue DMA for task type {0} on tile {1}, list {2} with task index range {3} - {4}",
                      taskType, tileIdx, listIdx, startTaskIdx, endTaskIdx);
        }
        dmaTile0List0Task = VPUMI40XX::getNextOp(dmaTile0List0Task);
    } while (dmaTile0List0Task);

    return enqueueDmasPerHwQueue;
}

}  // namespace VPUMI40XX
}  // namespace vpux
