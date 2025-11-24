//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/utils/wlm_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

#include <mlir/Pass/AnalysisManager.h>

namespace vpux {

void updateBarriersForDma(const SmallVector<size_t>& consumes, const SmallVector<size_t>& producesIn,
                          VPURT::TaskOp dmaOp, BarrierInfo& barrierInfo) {
    auto dmaIdx = barrierInfo.getIndex(dmaOp);
    for (auto pIn : producesIn) {
        barrierInfo.addProducer(pIn, dmaIdx);
    }
    for (auto consume : consumes) {
        barrierInfo.addConsumer(consume, dmaIdx);
    }
}

void updateBarriersForDma(const SmallVector<mlir::Value>& consumes, const SmallVector<mlir::Value>& producesIn,
                          VPURT::TaskOp dmaOp, BarrierInfo& barrierInfo) {
    auto dmaIdx = barrierInfo.getIndex(dmaOp);
    for (auto pIn : producesIn) {
        auto barrOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(pIn.getDefiningOp());
        barrierInfo.addProducer(barrOp, dmaIdx);
    }
    for (auto consume : consumes) {
        auto barrOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(consume.getDefiningOp());
        barrierInfo.addConsumer(barrOp, dmaIdx);
    }
}

// Fetch tasks are only attached to DMAs on port 0 and list 0 in later dialect
// In this context supportedDMA is a DMA which has channel DDR and port 0
bool isDMAOnSupportedPortAndChannel(VPURT::TaskOp dmaTaskOp) {
    if (auto dma = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(dmaTaskOp.getInnerTaskOp())) {
        // Check if this is DMA on Port 0 Channel DDR
        if (vpux::getDMAQueueIdEncoding(0, VPUIP::DmaChannelType::DDR) ==
            vpux::getDMAQueueIdEncoding(dma.getPortVal().value_or(0), dma.getChannelType())) {
            return true;
        }
    }
    return false;
}

// Function to find min or max position in a vector of TaskOps
VPURT::TaskOp findMinMaxPosition(const SmallVector<size_t>& dmas, BarrierInfo& barrierInfo, MinMaxOption option) {
    if (dmas.empty()) {
        return nullptr;
    }

    auto comparePositions = [](size_t lhs, size_t rhs) {
        return lhs < rhs;
    };

    if (option == MinMaxOption::Min) {
        auto minPosIt = std::min_element(dmas.begin(), dmas.end(), comparePositions);
        return barrierInfo.getTaskOpAtIndex(*minPosIt);
    }

    // Execution falls here is the if condition is false
    auto maxPosIt = std::max_element(dmas.begin(), dmas.end(), comparePositions);
    return barrierInfo.getTaskOpAtIndex(*maxPosIt);
}

// Check if two list of barriers have any barrier in common and return earliest of them
std::optional<size_t> getEarliestCommonBarrier(const BarrierInfo::TaskSet& taskSetOne,
                                               const BarrierInfo::TaskSet& taskSetTwo, BarrierInfo& barrierInfo) {
    size_t earliestBarrier = SIZE_MAX;  // Initialize to a very large value

    // Iterate through taskSetTwo and check if the task exists in taskSetOne
    for (size_t task : taskSetTwo) {
        if (taskSetOne.contains(task)) {
            // Update earliestBarrier if the current task is earlier
            if (compareVPURTOpPosition(task, earliestBarrier, barrierInfo, true)) {
                earliestBarrier = task;
            }
        }
    }

    if (earliestBarrier == SIZE_MAX) {
        return std::nullopt;  // No common barriers found
    }

    return earliestBarrier;
}

// Return last task that updates series of barriers
// As we check for the last DMA, sort the vector and use the last one
mlir::Operation* findLastTaskToUpdate(const BarrierInfo::TaskSet& barriers, BarrierInfo& barrierInfo) {
    // Collect all tasks that update any barrier in barrierVector
    SmallVector<VPURT::TaskOp> allUpdatingTasks;
    for (auto barrierIdx : barriers) {
        // Lambda to check if a task updates the current barrierIdx
        auto validUser = [&barrierIdx, &barrierInfo](VPURT::TaskOp op) -> bool {
            auto opIdx = barrierInfo.getIndex(op);

            auto updateBarrierList = barrierInfo.getUpdateBarriers(opIdx);
            return getEarliestCommonBarrier(
                           updateBarrierList,
                           [&]() {
                               BarrierInfo::TaskSet set;
                               set.insert(barrierIdx);
                               return set;
                           }(),
                           barrierInfo)
                    .has_value();
        };

        // Get all users of the current barrierIdx and filter based on the validUser condition
        auto bOp = barrierInfo.getBarrierOpAtIndex(barrierIdx).getBarrier();
        for (auto usr : bOp.getUsers()) {
            auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(usr);
            if (taskOp && validUser(taskOp)) {
                allUpdatingTasks.push_back(taskOp);
            }
        }
    }

    // Sort all updating tasks collected based on position to find the last one
    if (!allUpdatingTasks.empty()) {
        llvm::sort(allUpdatingTasks, [&](const auto& lhs, const auto& rhs) {
            return compareVPURTOpPosition(lhs, rhs, barrierInfo, true);
        });
        return allUpdatingTasks[allUpdatingTasks.size() - 1];
    }

    return nullptr;
}

// Create new barrier and add producer and consumer
VPURT::DeclareVirtualBarrierOp createNewBarrier(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                                mlir::Operation* insertionPoint, VPURT::TaskOp producer,
                                                VPURT::TaskOp consumer) {
    if (insertionPoint != nullptr) {
        builder.setInsertionPointAfter(insertionPoint);
    }
    auto newBarrierOp = builder.create<VPURT::DeclareVirtualBarrierOp>(mlir::UnknownLoc::get(builder.getContext()));
    barrierInfo.addNewBarrier(newBarrierOp);

    if (producer != nullptr) {
        barrierInfo.addProducer(newBarrierOp, barrierInfo.getIndex(producer));
    }

    if (consumer != nullptr) {
        barrierInfo.addConsumer(newBarrierOp, barrierInfo.getIndex(consumer));
    }

    return newBarrierOp;
}

/// Utils for inserting dependencies

void addElementsToSet(BarrierInfo::TaskSet& targetSet, const BarrierInfo::TaskSet& sourceSet) {
    for (const auto& element : sourceSet) {
        targetSet.insert(element);
    }
}

bool lastTaskInGroupHasMandatoryUpdateBarrier(const ExecutionGroup& executionGroup, BarrierInfo& barrierInfo) {
    auto lastTaskIdx = executionGroup.back();
    auto lastTaskUpdateBarriers = barrierInfo.getUpdateBarriers(lastTaskIdx);
    if (lastTaskUpdateBarriers.empty()) {
        return false;
    }
    return true;
}

bool inSameTaskBlock(size_t task1, size_t task2, const BlockRange& blockRange) {
    return std::any_of(blockRange.begin(), blockRange.end(), [&](const std::pair<size_t, size_t>& range) {
        return (task1 >= range.first && task1 <= range.second) && (task2 >= range.first && task2 <= range.second);
    });
}

/// Utils for adding placeholder fetch DMAs

// Function returns index of a task
size_t getIndexOfTask(IndexType indexType, ArrayRef<VPURT::TaskOp> dummyDMAs, BarrierInfo& barrierInfo) {
    if (indexType.second == Type::Dummy) {
        return barrierInfo.getIndex(dummyDMAs[indexType.first]);
    }
    return indexType.first;
}

// Function returns index of a barrier
size_t getIndexOfBarrier(IndexType indexType, ArrayRef<VPURT::DeclareVirtualBarrierOp> dummyBarriers,
                         BarrierInfo& barrierInfo) {
    if (indexType.second == Type::Dummy) {
        return barrierInfo.getIndex(dummyBarriers[indexType.first]);
    }
    return indexType.first;
}

VPURT::TaskOp createFetchDMA(mlir::OpBuilder& builder, mlir::Value input, mlir::Value output, int port,
                             mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers,
                             VPUIP::FetchDMAAttr fetchDMAAttr, llvm::StringLiteral opName) {
    auto* ctx = builder.getContext();
    auto syncDmaLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, opName));
    auto portAttr = vpux::getIntAttr(ctx, port);

    auto fetchDMAOp = VPURT::wrapIntoTaskOp<VPUIP::FetchDMAOp>(
            builder, waitBarriers, updateBarriers, syncDmaLoc, input, output, portAttr,
            /*isOutOfOrder*/ nullptr, /*isCritical*/ nullptr, /*dmaHwpId*/ nullptr,
            /*dmaProfilingMetaData*/ nullptr, fetchDMAAttr);

    return fetchDMAOp->getParentOfType<VPURT::TaskOp>();
}

VPUIP::FetchDMAAttr getFetchDMAAttr(int64_t groupIdx, BarrierInfo& barrierInfo, size_t taskIndex) {
    auto taskOp = barrierInfo.getTaskOpAtIndex(taskIndex);
    const auto ctx = taskOp->getContext();
    auto taskQueueType = barrierInfo.getTaskQueueType(taskIndex);
    auto executorKindAttr = VPU::ExecutorKindAttr::get(ctx, taskQueueType.type);
    auto tileIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), VPURT::getTileIndexForDpuOrShv(taskOp, taskQueueType));
    auto listIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), VPURT::getListIndexForDpuOrShv(taskOp));
    auto groupIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), groupIdx);
    return VPUIP::FetchDMAAttr::get(ctx, executorKindAttr, tileIdxAttr, listIdxAttr, groupIdxAttr);
}

void legalizeScheduleForNonWlm(mlir::func::FuncOp netFunc, BarrierInfo& barrierInfo, Logger log) {
    log.info("Legalize schedule for non-WLM");

    // Build task queue type map for all queues in order to test paths between tasks on different FIFOs.
    barrierInfo.clearTaskQueueTypeMap();
    barrierInfo.initializeTaskQueueTypeMap(
            {VPU::ExecutorKind::DMA_NN, VPU::ExecutorKind::DPU, VPU::ExecutorKind::SHAVE_ACT});
    barrierInfo.buildTaskQueueTypeMap();

    bool modifiedIR = false;
    ExecutionGroupAnalysis execGroupAnalysis(netFunc, /* ignoreVariantLimit */ true, /* ignoreInvariantLimit */ false);

    auto getExecutionGroups = [&](vpux::VPU::ExecutorKind executorKind) {
        if (executorKind == VPU::ExecutorKind::DPU) {
            return execGroupAnalysis.getDPUExecutionGroups();
        } else if (executorKind == VPU::ExecutorKind::SHAVE_ACT) {
            return execGroupAnalysis.getActShvExecutionGroups();
        } else {
            VPUX_THROW("Unsupported executor kind for non-WLM legalization '{0}'", executorKind);
        }
    };

    std::vector<vpux::VPU::ExecutorKind> executors = {VPU::ExecutorKind::DPU, VPU::ExecutorKind::SHAVE_ACT};
    for (auto executorKind : executors) {
        size_t newBarriers = 0;

        auto execGroups = getExecutionGroups(executorKind);
        for (size_t taskBlockIndex = 0; taskBlockIndex < barrierInfo.getControlGraphBlockCount(); ++taskBlockIndex) {
            newBarriers += barrierInfo.createBarrierDependenciesBetweenExecutionGroups(taskBlockIndex, execGroups);
        }

        if (newBarriers > 0) {
            modifiedIR = true;
            log.info("Inserted {0} barriers between {1} tasks on same FIFO.", newBarriers,
                     stringifyExecutorKind(executorKind));
        }
    }

    barrierInfo.clearTaskQueueTypeMap();

    if (modifiedIR) {
        VPURT::orderExecutionTasksAndBarriers(netFunc, barrierInfo, log);
        VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");

        execGroupAnalysis =
                ExecutionGroupAnalysis(netFunc, /* ignoreVariantLimit */ true, /* ignoreInvariantLimit */ false);
        VPUX_THROW_UNLESS(barrierInfo.verifyBarriersForTaskDescriptorFetch(execGroupAnalysis.getExecutionGroups(),
                                                                           /* wlmEnabled */ false),
                          "Encountered execution group without required barrier for task descriptor fetch.");
    }

    VPURT::postProcessBarrierOps(netFunc);
}

bool verifyBarriersForTaskDescriptorFetch(BarrierInfo& barrierInfo, mlir::func::FuncOp func, bool wlmFlag,
                                          std::optional<WorkloadManagementMode> wlmMode) {
    bool ignoreVariantLimit = false;
    if (!wlmFlag) {
        ignoreVariantLimit = true;
    }
    auto execGroupAnalysis = ExecutionGroupAnalysis(func, ignoreVariantLimit, /* ignoreInvariantLimit */ false);
    auto validateEachTileSeparately = true;
    if (wlmMode.has_value() && wlmMode.value() == WorkloadManagementMode::PWLM_V0_LCA) {
        validateEachTileSeparately = false;
    }
    auto execGroups = validateEachTileSeparately ? execGroupAnalysis.getExecutionGroups()
                                                 : execGroupAnalysis.getExecutionGroupsForTile(0);
    return barrierInfo.verifyBarriersForTaskDescriptorFetch(execGroups, wlmFlag);
}

}  // namespace vpux
