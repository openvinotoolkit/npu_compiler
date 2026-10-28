//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPURT/utils/wlm_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

namespace vpux::VPURT {

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
    auto newBarrierOp = VPURT::DeclareVirtualBarrierOp::create(builder, mlir::UnknownLoc::get(builder.getContext()));
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
            /*isOutOfOrder*/ false, /*isCritical*/ false, /*dmaHwpId*/ nullptr,
            /*dmaProfilingMetaData*/ nullptr, fetchDMAAttr);

    return fetchDMAOp->getParentOfType<VPURT::TaskOp>();
}

VPURT::TaskOp createSkipDMA(mlir::OpBuilder& builder, mlir::Value input, mlir::Value output, int port,
                            VPUIP::SkipDMAAttr skipDMAAttr, llvm::StringLiteral opName) {
    auto* ctx = builder.getContext();
    auto skipDmaLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, opName));
    auto portAttr = vpux::getIntAttr(ctx, port);

    auto skipDMAOp = VPURT::wrapIntoTaskOp<VPUIP::SkipDMAOp>(builder, {}, {}, skipDmaLoc, input, output, portAttr,
                                                             /*isOutOfOrder*/ false, /*isCritical*/ false,
                                                             /*dmaHwpId*/ nullptr,
                                                             /*dmaProfilingMetaData*/ nullptr, skipDMAAttr);

    return skipDMAOp->getParentOfType<VPURT::TaskOp>();
}

VPUIP::FetchDMAAttr getFetchDMAAttr(int64_t groupIdxOrLogicalTaskIdx, BarrierInfo& barrierInfo, size_t taskIndex,
                                    size_t tileIdx, size_t listIdx, int64_t descId, bool isLogicalTask) {
    auto taskOp = barrierInfo.getTaskOpAtIndex(taskIndex);
    const auto ctx = taskOp->getContext();

    config::ExecutorKindAttr executorKindAttr = nullptr;
    mlir::IntegerAttr execGroupIdxAttr = nullptr;
    mlir::IntegerAttr logicalTaskIdxAttr = nullptr;
    mlir::IntegerAttr descIdAttr = nullptr;
    VPUIP::FetchTypeAttr fetchTypeAttr = nullptr;
    auto tileIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), 0);
    auto listIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), 0);

    // If this is a logical task then we're fetching single DMA Descriptor
    if (isLogicalTask) {
        tileIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), tileIdx);
        listIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), listIdx);
        logicalTaskIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), groupIdxOrLogicalTaskIdx);
        descIdAttr = mlir::IntegerAttr::get(getInt64Type(ctx), descId);
        fetchTypeAttr = VPUIP::FetchTypeAttr::get(ctx, VPUIP::FetchType::SingleDescriptor);
        executorKindAttr = config::ExecutorKindAttr::get(ctx, config::ExecutorKind::DMA_NN);
    } else {
        auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);
        tileIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), VPURT::getTileIndexForDpuOrShv(taskOp, taskQueueType));
        listIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), VPURT::getListIndexForDpuOrShv(taskOp));
        execGroupIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), groupIdxOrLogicalTaskIdx);
        fetchTypeAttr = VPUIP::FetchTypeAttr::get(ctx, VPUIP::FetchType::DescriptorGroup);
        executorKindAttr = config::ExecutorKindAttr::get(ctx, taskQueueType.type);
    }
    return VPUIP::FetchDMAAttr::get(ctx, executorKindAttr, tileIdxAttr, listIdxAttr, fetchTypeAttr, execGroupIdxAttr,
                                    logicalTaskIdxAttr, descIdAttr);
}

VPUIP::SkipDMAAttr getSkipDMAAttr(BarrierInfo& barrierInfo, size_t taskIndex, size_t logicalTaskIdx, int64_t descId) {
    auto taskOp = barrierInfo.getTaskOpAtIndex(taskIndex);
    const auto ctx = taskOp->getContext();
    auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);
    auto tileIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), VPURT::getTileIndexForDpuOrShv(taskOp, taskQueueType));
    auto listIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), VPURT::getListIndexForDpuOrShv(taskOp));

    auto logicalTaskIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), logicalTaskIdx);
    auto descIdAttr = mlir::IntegerAttr::get(getInt64Type(ctx), descId);

    return VPUIP::SkipDMAAttr::get(ctx, tileIdxAttr, listIdxAttr, logicalTaskIdxAttr, descIdAttr);
}

bool verifyBarriersForTaskDescriptorFetch(BarrierInfo& barrierInfo, mlir::func::FuncOp func,
                                          std::optional<WorkloadManagementMode> wlmMode) {
    auto execGroupAnalysis = ExecutionGroupAnalysis(func);
    auto validateEachTileSeparately = true;
    if (wlmMode.has_value() && wlmMode.value() <= WorkloadManagementMode::PWLM_V0_1_PAGES) {
        validateEachTileSeparately = false;
    }
    auto execGroups = validateEachTileSeparately ? execGroupAnalysis.getExecutionGroups()
                                                 : execGroupAnalysis.getExecutionGroupsForTile(0);
    return barrierInfo.verifyBarriersForTaskDescriptorFetch(execGroups);
}

// Verify that Fetch DMAs have correct dependencies and that FetchDMA fro GroupN
// is guaranteed to start only after Group[N-2] has finished and before last
// task of Group[N-1] starts executing:
//
// Group[N-2].lastTask -> ... -> Group[N].FetchDMA -> ... -> Group[N-1].lastTask
//
bool verifyFetchDmaDependencies(mlir::func::FuncOp func, BarrierInfo& barrierInfo,
                                ExecutionGroupListMap& executionGroupListMap, vpux::Logger& log) {
    auto isFifoPerShvEnabled = config::isFifoPerShaveEngineEnabled(func);

    SmallVector<size_t> fetchTasks;
    DenseMap<VPUIP::FetchDMAAttr, size_t> fetchTaskMap;

    std::optional<size_t> blockIdxOfTaskControlMap;
    std::pair<SmallVector<llvm::BitVector>, size_t> taskControlMapAndOffset;

    // Get all Fetch Tasks
    func.walk([&](VPURT::TaskOp taskOp) {
        if (auto fetchDMAOp = taskOp.getInnerTaskOpOfType<VPUIP::FetchDMAOp>()) {
            fetchTasks.push_back(barrierInfo.getIndex(taskOp));
        }
    });

    // Create a map for lookup
    for (auto fetchTask : fetchTasks) {
        auto fetchTaskOp = barrierInfo.getTaskOpAtIndex(fetchTask);
        auto fetchDMAOp = fetchTaskOp.getInnerTaskOpOfType<VPUIP::FetchDMAOp>();
        fetchTaskMap[fetchDMAOp.getFetchDmaAttr()] = fetchTask;
    }

    // TODO: Compile time optimization suggestion:
    // Iterate over control blocks to limit the number of rebuilds of barrierInfo TaskControlMap

    // For all taskOpQueues go over each execution group and verify the correctness of corresponding FetchTask in the IR
    for (auto& [taskQueueType, executionGroups] : executionGroupListMap) {
        // 0 and 1 can be fetch simultaneously
        if (executionGroups.size() < 3) {
            continue;
        }

        // In case of SHV tasks with FIFO per SHV disabled, we need to verify dependencies
        // not only against last task but also previous to last task
        size_t numOfGroupLastTasksToCheck = 1;
        if (taskQueueType.type == config::ExecutorKind::SHAVE_ACT && !isFifoPerShvEnabled) {
            numOfGroupLastTasksToCheck = 2;
        }

        for (size_t groupIdx = 2; groupIdx < executionGroups.size(); groupIdx++) {
            auto grandParentGroup = executionGroups[groupIdx - 2];
            auto parentGroup = executionGroups[groupIdx - 1];
            auto travelingGroup = executionGroups[groupIdx];

            auto fetchTaskAttr = getFetchDMAAttr(groupIdx, barrierInfo, travelingGroup.front());
            auto fetchTask = fetchTaskMap[fetchTaskAttr];
            // Is FetchTask for Group N dependent on last task of Group N-2 (Grand Parent Group)
            auto grandParentGroupSize = grandParentGroup.size();
            auto numOfTasksToCheck = std::min(numOfGroupLastTasksToCheck, grandParentGroupSize);
            for (size_t taskFromBack = 0; taskFromBack < numOfTasksToCheck; ++taskFromBack) {
                VPUX_THROW_UNLESS(grandParentGroupSize >= 1 + taskFromBack,
                                  "Index is out of bounds for grand parent group of size {0}, task from back {1}",
                                  grandParentGroupSize, taskFromBack);
                size_t index = grandParentGroupSize - 1 - taskFromBack;
                auto grandParentGroupTaskIdx = grandParentGroup[index];
                if (!barrierInfo.isDepFromTaskAToTaskB(grandParentGroupTaskIdx, fetchTask, taskControlMapAndOffset,
                                                       blockIdxOfTaskControlMap)) {
                    log.warning("Fetch task {0} is not dependent on last task of Group N-2 {1}", fetchTask,
                                grandParentGroupTaskIdx);
                    return false;
                }
            }

            // Is last task of Group N-1 (Parent Group) dependent on FetchTask for Group N
            auto parentGroupSize = parentGroup.size();
            numOfTasksToCheck = std::min(numOfGroupLastTasksToCheck, parentGroupSize);
            for (size_t taskFromBack = 0; taskFromBack < numOfTasksToCheck; ++taskFromBack) {
                VPUX_THROW_UNLESS(parentGroupSize >= 1 + taskFromBack,
                                  "Index is out of bounds for parent group of size {0}, task from back {1}",
                                  parentGroupSize, taskFromBack);
                size_t index = parentGroupSize - 1 - taskFromBack;
                auto parentGroupTaskIdx = parentGroup[index];
                if (!barrierInfo.isDepFromTaskAToTaskB(fetchTask, parentGroupTaskIdx, taskControlMapAndOffset,
                                                       blockIdxOfTaskControlMap)) {
                    log.warning("Last task of Group N-1 {0} is not dependent on fetch task {1}", parentGroupTaskIdx,
                                fetchTask);
                    return false;
                }
            }
        }
    }
    return true;
}

}  // namespace vpux::VPURT
