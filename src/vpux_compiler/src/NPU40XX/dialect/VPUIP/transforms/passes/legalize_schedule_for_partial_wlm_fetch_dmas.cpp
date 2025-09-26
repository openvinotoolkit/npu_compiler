//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/workload_management_status_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/wlm_legalization_utils.hpp"

#include <queue>

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_LEGALIZESCHEDULEFORPARTIALWLMFETCHDMAS
#define GEN_PASS_DEF_LEGALIZESCHEDULEFORPARTIALWLMFETCHDMAS
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;
namespace {

//
//  LegalizeScheduleForPartialWlmFetchDmasPass
//

struct DummyDMAData {
    size_t insertionPoint;
    SmallVector<IndexType> consumes;
    SmallVector<IndexType> producesIn;
};

class LegalizeScheduleForPartialWlmFetchDmasPass final :
        public VPUIP::arch40xx::impl::LegalizeScheduleForPartialWlmFetchDmasBase<
                LegalizeScheduleForPartialWlmFetchDmasPass> {
public:
    explicit LegalizeScheduleForPartialWlmFetchDmasPass(const int virtualBarrierThreshold, Logger log)
            : _virtualBarrierThreshold(virtualBarrierThreshold) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    int _virtualBarrierThreshold;
    void safeRunOnFunc() final;

    bool isValidDMA(BarrierInfo& barrierInfo, size_t dmaIdx);
    VPURT::TaskOp findLastDmaBeforeExecGroup(BarrierInfo& barrierInfo, ExecutionGroup& executionGroup);
    VPURT::TaskOp findFirstDmaAfterExecGroup(BarrierInfo& barrierInfo, ExecutionGroup& executionGroup);
    VPURT::TaskOp findDMAsThroughBarriersBFS(size_t startBarrier, BarrierInfo& barrierInfo, MinMaxOption option,
                                             bool bfsDirUp);

    void planDummyDMAAndBarriersInsertion(DenseMap<VPURT::TaskQueueType, ExecutionGroupList>& executionGroups,
                                          BarrierInfo& barrierInfo, SmallVector<std::pair<size_t, size_t>>& blockRange);

    void planDummyDMAAndBarriersInsertionPerQueue(ExecutionGroupList& executionGroup, BarrierInfo& barrierInfo,
                                                  SmallVector<std::pair<size_t, size_t>>& blockRange,
                                                  VPURT::TaskQueueType queueType);

    SmallVector<size_t> getDmasUpdatingBarriers(llvm::DenseSet<size_t>& barriers, BarrierInfo& barrierInfo);

    void realizePlannedInsertions(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                  mlir::Operation* bufferInsertionPoint, SmallVector<VPURT::TaskOp>& dummyDmas);

private:
    // Will be initialized in safeRunOnFunc() or relevant function, this is done to suppress the UNINIT_CTOR warning
    size_t _numAllTaskOps = 0;
    size_t _newBarrierIndex = 0;

    VPURT::TaskOp _firstDMATaskOp;
    VPURT::TaskOp _lastDMATaskOp;

    SmallVector<DummyDMAData> _dummyDMAsToInsert;
    SmallVector</*insertionPoint */ size_t> _newBarriersToInsert;
    llvm::DenseMap<IndexType, std::pair<SmallVector<IndexType>, SmallVector<IndexType>>>
            _barrierRemoveConsumerProducerMap;
    llvm::DenseMap<IndexType, std::pair<SmallVector<IndexType>, SmallVector<IndexType>>> _barrierAddConsumerProducerMap;

    SmallVector<VPURT::DeclareVirtualBarrierOp> _dummyBarriers;
    SmallVector<VPURT::TaskOp> _dummyDMAs;
};

// Returns a DMA which copies 0 len data from DDR to DDR
VPURT::TaskOp createDummyDma(mlir::OpBuilder& builder, mlir::Value inputBuf, mlir::Value outputBuf,
                             BarrierInfo& barrierInfo, SmallVector<VPURT::TaskOp>& dummyDmas) {
    auto newDMA = VPUIP::createSyncDMA(builder, inputBuf, outputBuf, 0, {}, {}, "wlm_fetch_dummy_dma");

    barrierInfo.addNewTaskOp(newDMA);
    dummyDmas.push_back(newDMA);
    return newDMA;
}

void finalizeBarrierInfo(BarrierInfo& barrierInfo, SmallVector<VPURT::TaskOp>& dummyDmas, mlir::func::FuncOp netFunc,
                         Logger& log) {
    // Apply the changes now as we need to make sure the new DMAs don't break the schedule
    barrierInfo.updateIR();

    // Log the number of inserted DMAs for fetch legalization
    if (!dummyDmas.empty()) {
        log.info("Inserted '{0}' DMAs for fetch legalization", dummyDmas.size());
    }

    // Correct the position of new dmas after realizing the changes from barrierInfo
    for (auto dummyDma : dummyDmas) {
        auto dummyDmaIdx = barrierInfo.getIndex(dummyDma);
        auto waitBarriers = barrierInfo.getWaitBarriers(dummyDmaIdx);
        if (waitBarriers.empty()) {
            continue;
        }

        auto lastTaskToUpdate = findLastTaskToUpdate(waitBarriers, barrierInfo);
        if (dummyDma->isBeforeInBlock(lastTaskToUpdate)) {
            dummyDma->moveAfter(lastTaskToUpdate);
        }
    }

    // Reorder barriers in production order, this will also verify the schedule
    VPURT::orderExecutionTasksAndBarriers(netFunc, barrierInfo, log);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(netFunc);
}

// Once we know the insertion point of DMAs this function creates actual DMAs in IR while also keeps a map of [index,
// DMAOp This map is used later to refer to real DMAOp and get real task-index from barrierInfo
void LegalizeScheduleForPartialWlmFetchDmasPass::realizePlannedInsertions(mlir::OpBuilder& builder,
                                                                          BarrierInfo& barrierInfo,
                                                                          mlir::Operation* bufferInsertionPoint,
                                                                          SmallVector<VPURT::TaskOp>& dummyDmas) {
    auto inBuffer = VPUIP::createDummyBuffer(builder, bufferInsertionPoint);
    auto outBuffer = VPUIP::createDummyBuffer(builder, bufferInsertionPoint);

    for (const auto& [newBarrierIndex, value] : _newBarriersToInsert | indexed) {
        auto valueOp = barrierInfo.getBarrierOpAtIndex(value);
        auto newBarrierOp = createNewBarrier(builder, barrierInfo, valueOp, nullptr, nullptr);
        _dummyBarriers.push_back(newBarrierOp);
    }

    for (const auto& [dummyDmaIndex, value] : _dummyDMAsToInsert | indexed) {
        auto insertionPointOp = barrierInfo.getTaskOpAtIndex(value.insertionPoint);
        builder.setInsertionPointAfter(insertionPointOp);
        auto dummyDMA = createDummyDma(builder, inBuffer, outBuffer, barrierInfo, dummyDmas);
        _dummyDMAs.push_back(dummyDMA);
    }

    // We have created the new DMAs and barriers, adjust dependencies
    for (const auto& [dummyDmaIndex, value] : _dummyDMAsToInsert | indexed) {
        SmallVector<size_t> realProducesIn;
        SmallVector<size_t> realConsumes;
        for (auto produce : value.producesIn) {
            auto realBarrierIdx = getIndexOfBarrier(produce, _dummyBarriers, barrierInfo);
            realProducesIn.push_back(realBarrierIdx);
        }
        for (auto consume : value.consumes) {
            auto realBarrierIdx = getIndexOfBarrier(consume, _dummyBarriers, barrierInfo);
            realConsumes.push_back(realBarrierIdx);
        }
        updateBarriersForDma(realConsumes, realProducesIn, _dummyDMAs[dummyDmaIndex], barrierInfo);
    }

    for (const auto& [indexType, value] : _barrierRemoveConsumerProducerMap) {
        auto realBarrierIdx = getIndexOfBarrier(indexType, _dummyBarriers, barrierInfo);
        for (auto consumer : value.first) {
            auto realTaskIdx = getIndexOfTask(consumer, _dummyDMAs, barrierInfo);
            barrierInfo.removeConsumer(realBarrierIdx, realTaskIdx);
        }
        for (auto producer : value.second) {
            auto realTaskIdx = getIndexOfTask(producer, _dummyDMAs, barrierInfo);
            barrierInfo.removeProducer(realBarrierIdx, realTaskIdx);
        }
    }

    for (const auto& [indexType, value] : _barrierAddConsumerProducerMap) {
        auto realBarrierIdx = getIndexOfBarrier(indexType, _dummyBarriers, barrierInfo);
        for (auto consumer : value.first) {
            auto realTaskIdx = getIndexOfTask(consumer, _dummyDMAs, barrierInfo);
            barrierInfo.addConsumer(realBarrierIdx, realTaskIdx);
        }
        for (auto producer : value.second) {
            auto realTaskIdx = getIndexOfTask(producer, _dummyDMAs, barrierInfo);
            barrierInfo.addProducer(realBarrierIdx, realTaskIdx);
        }
    }
}

bool LegalizeScheduleForPartialWlmFetchDmasPass::isValidDMA(BarrierInfo& barrierInfo, size_t dmaIdx) {
    auto taskOp = barrierInfo.getTaskOpAtIndex(dmaIdx);
    return taskOp.getExecutorKind() == VPU::ExecutorKind::DMA_NN && isDMAOnSupportedPortAndChannel(taskOp) &&
           dmaIdx < _numAllTaskOps;
}

VPURT::TaskOp LegalizeScheduleForPartialWlmFetchDmasPass::findDMAsThroughBarriersBFS(size_t startBarrier,
                                                                                     BarrierInfo& barrierInfo,
                                                                                     MinMaxOption option,
                                                                                     bool bfsDirUp) {
    std::queue<size_t> barriersToExplore;
    barriersToExplore.push(startBarrier);
    std::unordered_set<size_t> visitedBarriers;
    SmallVector<size_t> possibleDMAs;

    while (!barriersToExplore.empty()) {
        auto currentBarrier = barriersToExplore.front();
        barriersToExplore.pop();

        if (visitedBarriers.count(currentBarrier)) {
            continue;  // Skip already visited barriers to avoid cycles
        }
        visitedBarriers.insert(currentBarrier);

        SmallVector<size_t> currentOps = bfsDirUp ? to_small_vector(barrierInfo.getBarrierProducers(currentBarrier))
                                                  : to_small_vector(barrierInfo.getBarrierConsumers(currentBarrier));

        llvm::sort(currentOps, [&](const size_t& lhs, const size_t& rhs) {
            return compareVPURTOpPosition(lhs, rhs, barrierInfo);
        });

        for (auto op : currentOps) {
            auto nextBarriers = bfsDirUp ? barrierInfo.getWaitBarriers(op) : barrierInfo.getUpdateBarriers(op);

            auto filteredRange = currentOps | vpux::filtered([this, &barrierInfo](size_t dmaIdx) {
                                     return isValidDMA(barrierInfo, dmaIdx);
                                 });
            auto filteredVector = to_small_vector(filteredRange);
            possibleDMAs.insert(possibleDMAs.end(), filteredVector.begin(), filteredVector.end());

            if (!possibleDMAs.empty()) {
                return findMinMaxPosition(possibleDMAs, barrierInfo, option);
            }

            for (auto barrier : nextBarriers) {
                barriersToExplore.push(barrier);
            }
        }
    }

    return (option == MinMaxOption::Max) ? _firstDMATaskOp : _lastDMATaskOp;
}

/*
  452              650
   |                |
   v                v
  +------------------+
  | Execution   Group|
  +------------------+
9184               9277

Find First DMA for a group tries to find a DMA which is the first DMA that can be used as a marker to tell the
Execution Group has finished execution e.g. in above case 9277 is the last barrier in execution group. The function
looks for a DMA which is on tile 0 list 0 and waits for 9277

If there is no DMA which waits for 9277 using BFS check if the user's barrier (barrier->task->barrier) has a DMA user
and use it as first DMA
*/
VPURT::TaskOp LegalizeScheduleForPartialWlmFetchDmasPass::findFirstDmaAfterExecGroup(BarrierInfo& barrierInfo,
                                                                                     ExecutionGroup& executionGroup) {
    SmallVector<VPURT::DeclareVirtualBarrierOp> updateBarriers;
    for (const auto& taskIndex : executionGroup) {
        auto upBarriers = barrierInfo.getUpdateBarriers(taskIndex);
        for (auto updateBarrier : upBarriers) {
            auto bOpInterface = barrierInfo.getBarrierOpAtIndex(updateBarrier);
            auto updateBarrierOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(bOpInterface.getOperation());
            updateBarriers.push_back(updateBarrierOp);
        }
    }

    llvm::sort(updateBarriers, [&](const auto& lhs, const auto& rhs) {
        return compareVPURTOpPosition(lhs, rhs, barrierInfo);
    });

    auto lastBarrier = barrierInfo.getIndex(updateBarriers[updateBarriers.size() - 1]);
    SmallVector<size_t> possibleWaitingDMAs;

    auto consumers = to_small_vector(barrierInfo.getBarrierConsumers(lastBarrier));
    auto filteredRange = consumers | vpux::filtered([this, &barrierInfo](size_t dmaIdx) {
                             return isValidDMA(barrierInfo, dmaIdx);
                         });

    auto filteredVector = to_small_vector(filteredRange);
    possibleWaitingDMAs.insert(possibleWaitingDMAs.end(), filteredVector.begin(), filteredVector.end());

    if (!possibleWaitingDMAs.empty()) {
        return findMinMaxPosition(possibleWaitingDMAs, barrierInfo, MinMaxOption::Min);
    }

    return findDMAsThroughBarriersBFS(lastBarrier, barrierInfo, MinMaxOption::Min, /*bfsDirUp=*/false);
}

/*
  452                650
   |                |
   v                v
  +------------------+
  | Execution Group  |
  +------------------+
9184               9277

Find Last DMA for a group tries to find a DMA which is the last DMA that can be used as a marker to tell the Execution
Group should start execution e.g. in above case 9184 is the first barrier in execution group. The function looks for a
DMA which is on tile 0 list 0 and updates 9184

If there is no DMA which updates 9184 using BFS check if the user's barrier (barrier<-task<-barrier) has a DMA user and
use it as last DMA
*/
VPURT::TaskOp LegalizeScheduleForPartialWlmFetchDmasPass::findLastDmaBeforeExecGroup(BarrierInfo& barrierInfo,
                                                                                     ExecutionGroup& executionGroup) {
    SmallVector<VPURT::DeclareVirtualBarrierOp> waitBarriers;
    SmallVector<size_t> possibleUpdatingDMAs;

    for (const auto& taskIdx : executionGroup) {
        auto wBarriers = barrierInfo.getUpdateBarriers(taskIdx);
        for (auto waitBarrier : wBarriers) {
            auto bOpInterface = barrierInfo.getBarrierOpAtIndex(waitBarrier);
            auto waitBarrierOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(bOpInterface.getOperation());
            waitBarriers.push_back(waitBarrierOp);
        }
    }

    // Collect possible updating DMAs from all wait barriers
    for (const auto& waitBarrier : waitBarriers) {
        auto barrierIdx = barrierInfo.getIndex(waitBarrier);
        auto updatingDMAs = to_small_vector(barrierInfo.getBarrierProducers(barrierIdx));

        // Filter producers to retain only valid DMAs
        auto filteredRange = updatingDMAs | vpux::filtered([this, &barrierInfo](size_t dmaIdx) {
                                 return isValidDMA(barrierInfo, dmaIdx);
                             });

        possibleUpdatingDMAs.insert(possibleUpdatingDMAs.end(), filteredRange.begin(), filteredRange.end());
    }

    // If valid DMAs were found, return the one with the latest position
    if (!possibleUpdatingDMAs.empty()) {
        return findMinMaxPosition(possibleUpdatingDMAs, barrierInfo, MinMaxOption::Max);
    }

    // Initialize variable to track the latest DMA found across all barriers
    VPURT::TaskOp maxDmaOp;
    bool foundAnyDMA = false;

    // Search through all barriers using BFS to find the DMA with the latest position
    for (const auto& waitBarrier : waitBarriers) {
        auto barrierIdx = barrierInfo.getIndex(waitBarrier);
        auto dmaOp = findDMAsThroughBarriersBFS(barrierIdx, barrierInfo, MinMaxOption::Max, /*bfsDirUp=*/true);

        // Check if a DMA was found, and track the latest position DMA across all barriers
        if (dmaOp && (!foundAnyDMA || compareVPURTOpPosition(maxDmaOp, dmaOp, barrierInfo))) {
            maxDmaOp = dmaOp;
            foundAnyDMA = true;
        }
    }

    return maxDmaOp;
}

SmallVector<size_t> LegalizeScheduleForPartialWlmFetchDmasPass::getDmasUpdatingBarriers(
        llvm::DenseSet<size_t>& barriers, BarrierInfo& barrierInfo) {
    llvm::DenseSet<size_t> allTaskUpdatingBarriers;
    for (auto barrIdx : barriers) {
        auto allProducers = barrierInfo.getBarrierProducers(barrIdx);
        for (auto p : allProducers) {
            allTaskUpdatingBarriers.insert(p);
        }
    }
    auto updatingTasks = to_small_vector(allTaskUpdatingBarriers);
    auto filteredRange = updatingTasks | vpux::filtered([this, &barrierInfo](size_t dmaIdx) {
                             return isValidDMA(barrierInfo, dmaIdx);
                         });
    return to_small_vector(filteredRange);
}

// Function returns barriers used by parentGroup excluding the barriers used by travelingGroup
llvm::DenseSet<size_t> getExclusiveBarriersUsedByGroup(ExecutionGroup& parentGroup, ExecutionGroup& travelingGroup,
                                                       BarrierInfo& barrierInfo) {
    llvm::DenseSet<size_t> parentBarriersUsed;
    llvm::DenseSet<size_t> travelingBarriersUsed;
    llvm::DenseSet<size_t> exclusiveBarriers;

    auto getBarriersUsedByGroup = [&barrierInfo](ExecutionGroup& execGroup, llvm::DenseSet<size_t>& barriersUsed) {
        for (auto i : execGroup) {
            auto wBarr = barrierInfo.getWaitBarriers(i);
            auto uBarr = barrierInfo.getUpdateBarriers(i);

            std::for_each(wBarr.begin(), wBarr.end(), [&barriersUsed](size_t barrier) {
                barriersUsed.insert(barrier);
            });

            std::for_each(uBarr.begin(), uBarr.end(), [&barriersUsed](size_t barrier) {
                barriersUsed.insert(barrier);
            });
        }
    };

    getBarriersUsedByGroup(parentGroup, parentBarriersUsed);
    getBarriersUsedByGroup(travelingGroup, travelingBarriersUsed);

    for (auto barrier : parentBarriersUsed) {
        if (travelingBarriersUsed.find(barrier) == travelingBarriersUsed.end()) {
            exclusiveBarriers.insert(barrier);
        }
    }
    return exclusiveBarriers;
}

/*
Form legalization perspective of traveling group we care about the last tasks from grand-parent group and parent group.
Each group can have at max 32 Invariant/Kernel Invocation

    +----------------------------+   +--------------------+   +---------------------+
    | GrandParentGroup    |lastOp|   | ParentGroup |lastOp|   |      TravelingGroup |
    +----------------------------+   +--------------------+   +---------------------+

Case 1: Last task of grand parent and last task of parent have common wait and update barriers
B:9184 and B:9277 represent the barriers associated with last task of parent group and also last task of grand parent
group

  +--------------------+         +------------------+
  |LT GrandParent Group|         | LT Parent Group  |
  +--------------------+         +------------------+
B:9184               B:9277   B:9184             B:9277

                        | |
                         v

  +--------------------+         +------------------+
  |LT GrandParent Group|         | LT Parent Group  |
  +--------------------+         +------------------+
B:9184              BNew1      BNew2                B:9277
                    |            ^
                    |            |
                    |            |
                    v            |
                +-----+       +-----+
                | D1  |.......| D2  |
                +-----+       +-----+


Case 2: Last task of grand parent and last task of parent group share barriers
B:9277 and B:9353 represent the barriers associated with last task of parent group
B:9184 and B:9277 represent the barriers associated with last task of grand parent group

  +--------------------+         +------------------+
  |LT GrandParent Group|         | LT Parent Group  |
  +--------------------+         +------------------+
B:9184              B:9277     B:9277              B:9353

                            | |
                             v

  +--------------------+         +------------------+
  |LT GrandParent Group|         | LT Parent Group  |
  +--------------------+         +------------------+
B:9184              B:New      B:9277             B:9353
                    |            ^
                    |            |
                    |            |
                    v            |
                +-----+        +-----+
                | D1  |........| D2  |
                +-----+        +-----+


The barriers are common we cannot use them as wait and update and need to insert a
new barriers to delay traveling group

Case 3: Last task of grand parent and last task of parent have exclusive barriers
B:9729 and B:9353 represent the barriers associated with last task of parent group
B:9184 and B:9277 is the barriers associated with last task of grand parent group

  +--------------------+         +------------------+
  |LT GrandParent Group|         | LT Parent Group  |
  +--------------------+         +------------------+
B:9184               B:9277     B:9279              B:9353

                        | |
                         v

  +--------------------+         +------------------+
  |LT GrandParent Group|         | LT Parent Group  |
  +--------------------+         +------------------+
B:9184              B:9277     B:9279               B:9353
                    |            ^
                    |            |
                    |            |
                    v            |
                +-----+          +-----+
                | D1  |..........| D2  |
                +-----+          +-----+

*/
void LegalizeScheduleForPartialWlmFetchDmasPass::planDummyDMAAndBarriersInsertion(
        DenseMap<VPURT::TaskQueueType, ExecutionGroupList>& executionGroups, BarrierInfo& barrierInfo,
        SmallVector<std::pair<size_t, size_t>>& blockRange) {
    for (auto& [queueType, executionGroup] : executionGroups) {
        planDummyDMAAndBarriersInsertionPerQueue(executionGroup, barrierInfo, blockRange, queueType);
    }
}

void LegalizeScheduleForPartialWlmFetchDmasPass::planDummyDMAAndBarriersInsertionPerQueue(
        ExecutionGroupList& executionGroup, BarrierInfo& barrierInfo,
        SmallVector<std::pair<size_t, size_t>>& blockRange, VPURT::TaskQueueType queueType) {
    if (executionGroup.size() < 3) {
        return;
    }

    auto parentGroup = executionGroup[0];
    ExecutionGroup grandParentGroup;

    size_t groupIdx = 1;
    auto travelingGroup = executionGroup[groupIdx];

    while (groupIdx < executionGroup.size()) {
        auto hasGrandParent = !grandParentGroup.empty();
        auto firstGrandParentDma = hasGrandParent ? findFirstDmaAfterExecGroup(barrierInfo, grandParentGroup) : nullptr;
        auto insertionDma = hasGrandParent ? firstGrandParentDma : findLastDmaBeforeExecGroup(barrierInfo, parentGroup);
        auto insertionIndex = barrierInfo.getIndex(insertionDma);

        bool legalizationRequired = true;
        if (hasGrandParent) {
            // Check if the DMA is updating any barriers used by parent group
            // Function returns exclusive barriers used by parent group
            auto barriersByParentGroup = getExclusiveBarriersUsedByGroup(parentGroup, travelingGroup, barrierInfo);
            auto dmasUpdatingBarriers = getDmasUpdatingBarriers(barriersByParentGroup, barrierInfo);
            if (!dmasUpdatingBarriers.empty()) {
                auto lastDmaToUpdateBarrierInParentGroup =
                        barrierInfo.getIndex(findMinMaxPosition(dmasUpdatingBarriers, barrierInfo, MinMaxOption::Max));
                if (lastDmaToUpdateBarrierInParentGroup > insertionIndex) {
                    legalizationRequired = false;
                }
            }
        }

        if (legalizationRequired && hasGrandParent) {
            size_t parentTaskIdx = parentGroup.size() - 1;
            auto lastParentTaskOpIdx = parentGroup[parentTaskIdx];
            auto lastGrandParentTaskIdx = grandParentGroup[grandParentGroup.size() - 1];

            auto lastGrandParentTaskOpIdx = lastGrandParentTaskIdx;
            auto lastGrandParentTaskOp = barrierInfo.getTaskOpAtIndex(lastGrandParentTaskOpIdx);

            auto lastGrandParentTaskUpdateBarriers = barrierInfo.getUpdateBarriers(lastGrandParentTaskOpIdx);
            auto lastGrandParentTaskWaitBarriers = barrierInfo.getWaitBarriers(lastGrandParentTaskOpIdx);

            auto lastParentTaskWaitBarriers = barrierInfo.getWaitBarriers(lastParentTaskOpIdx);
            auto lastParentTaskUpdateBarriers = barrierInfo.getUpdateBarriers(lastParentTaskOpIdx);

            if (!inSameTaskBlock(lastParentTaskOpIdx, lastGrandParentTaskIdx, blockRange)) {
                grandParentGroup = parentGroup;
                parentGroup = travelingGroup;

                ++groupIdx;
                if (groupIdx < executionGroup.size()) {
                    travelingGroup = executionGroup[groupIdx];
                }
                continue;
            }

            auto commonWaitBarrierOpt =
                    getEarliestCommonBarrier(lastGrandParentTaskWaitBarriers, lastParentTaskWaitBarriers, barrierInfo);
            auto commonUpdateBarrierOpt = getEarliestCommonBarrier(lastGrandParentTaskUpdateBarriers,
                                                                   lastParentTaskUpdateBarriers, barrierInfo);
            if (commonWaitBarrierOpt && commonUpdateBarrierOpt) {
                size_t insertionBarrierIndex = commonWaitBarrierOpt.value();
                auto insertionBarrier = barrierInfo.getBarrierOpAtIndex(insertionBarrierIndex);
                size_t barOneDummyIdx = _newBarrierIndex++;
                size_t barTwoDummyIdx = _newBarrierIndex++;

                _newBarriersToInsert.push_back(insertionBarrierIndex);
                _newBarriersToInsert.push_back(insertionBarrierIndex);

                for (auto barrIdx : lastParentTaskWaitBarriers) {
                    auto barrOp = barrierInfo.getBarrierOpAtIndex(barrIdx);

                    auto& consumerToRemove =
                            _barrierRemoveConsumerProducerMap[{barrierInfo.getIndex(barrOp), Type::Real}].first;
                    consumerToRemove.push_back({lastParentTaskOpIdx, Type::Real});

                    auto& consumersToAdd = _barrierAddConsumerProducerMap[{barTwoDummyIdx, Type::Dummy}].first;
                    consumersToAdd.push_back({lastParentTaskOpIdx, Type::Real});
                }

                DummyDMAData dummyDMAOneAttr;
                dummyDMAOneAttr.insertionPoint = lastGrandParentTaskOpIdx;
                dummyDMAOneAttr.consumes = {{barOneDummyIdx, Type::Dummy}};
                dummyDMAOneAttr.producesIn = {{barTwoDummyIdx, Type::Dummy}};
                _dummyDMAsToInsert.push_back(dummyDMAOneAttr);

                for (auto barrIdx : lastGrandParentTaskUpdateBarriers) {
                    auto barrOp = barrierInfo.getBarrierOpAtIndex(barrIdx);
                    auto& producerToRemove =
                            _barrierRemoveConsumerProducerMap[{barrierInfo.getIndex(barrOp), Type::Real}].second;
                    producerToRemove.push_back({lastGrandParentTaskIdx, Type::Real});

                    auto& producerToAdd = _barrierAddConsumerProducerMap[{barOneDummyIdx, Type::Dummy}].second;
                    producerToAdd.push_back({lastGrandParentTaskIdx, Type::Real});
                }

                DummyDMAData dummyDMATwoAttr;
                dummyDMATwoAttr.insertionPoint = lastGrandParentTaskOpIdx;
                dummyDMATwoAttr.consumes = {{barOneDummyIdx, Type::Dummy}};
                dummyDMATwoAttr.producesIn = {{barTwoDummyIdx, Type::Dummy}};
                _dummyDMAsToInsert.push_back(dummyDMATwoAttr);

                /*
                    Since DMAX position in FIFO is before DMA1 and since the barriers are same for GP and TG we
                    would end up enqueuing them at same barrier this leads to inference hang as TG wouldn't be
                    fetched

                    To overcome this we must also update all DPU/SW task that waits on the same barrier as GP (except
                    the tasks in GP)

                    Bar0[--DMAX--]73
                                 73[----GP----]X
                                               X-DMA1-Y
                                               X-DMA2-Y
                                                      Y[----PG----]74
                                                     *Y[----TG----]74

                */
                auto allConsumersOfWaitBarrier = to_small_vector(barrierInfo.getBarrierConsumers(insertionBarrier));
                auto validUser = [&](size_t taskIdx) -> bool {
                    auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);

                    // Only change barrier deps for tasks after the last task from the grandparent execution group
                    if (taskOp->isBeforeInBlock(lastGrandParentTaskOp) ||
                        taskIdx == barrierInfo.getIndex(lastGrandParentTaskOp)) {
                        return false;
                    }

                    const auto kind = taskOp.getExecutorKind();

                    // Only allow same executor kind as GrandParent or DMA_NN
                    if (kind != queueType.type && kind != VPU::ExecutorKind::DMA_NN) {
                        return false;
                    }

                    // For non-DMA tasks, queue ID must match
                    if (kind != VPU::ExecutorKind::DMA_NN &&
                        VPURT::getTaskQueueType(taskOp, false).id != static_cast<int64_t>(queueType.id)) {
                        return false;
                    }

                    return true;
                };

                auto filteredRange = allConsumersOfWaitBarrier | vpux::filtered(std::move(validUser));
                auto filteredVector = to_small_vector(filteredRange);
                for (auto consumer : filteredVector) {
                    auto& consumerToRemove =
                            _barrierRemoveConsumerProducerMap[{insertionBarrierIndex, Type::Real}].first;
                    consumerToRemove.push_back({consumer, Type::Real});

                    auto& consumerToAdd = _barrierAddConsumerProducerMap[{barTwoDummyIdx, Type::Dummy}].first;
                    consumerToAdd.push_back({consumer, Type::Real});
                }

            } else if (auto commonBarrOpt = getEarliestCommonBarrier(lastGrandParentTaskUpdateBarriers,
                                                                     lastParentTaskWaitBarriers, barrierInfo)) {
                auto commonBarrierIndex = commonBarrOpt.value();
                size_t barOneDummyIdx = _newBarrierIndex++;

                DummyDMAData dummyDMAOneAttr;
                dummyDMAOneAttr.insertionPoint = lastGrandParentTaskOpIdx;
                dummyDMAOneAttr.consumes.push_back({barOneDummyIdx, Type::Dummy});
                dummyDMAOneAttr.producesIn.push_back({commonBarrierIndex, Type::Real});
                _dummyDMAsToInsert.push_back(dummyDMAOneAttr);

                _newBarriersToInsert.push_back(commonBarrierIndex);
                auto& producerToRemove = _barrierRemoveConsumerProducerMap[{commonBarrierIndex, Type::Real}].second;
                producerToRemove.push_back({lastGrandParentTaskIdx, Type::Real});
                auto& producerToAdd = _barrierAddConsumerProducerMap[{barOneDummyIdx, Type::Dummy}].second;
                producerToAdd.push_back({lastGrandParentTaskIdx, Type::Real});

                DummyDMAData dummyDMATwoAttr;
                dummyDMATwoAttr.insertionPoint = lastGrandParentTaskOpIdx;
                dummyDMATwoAttr.consumes.push_back({barOneDummyIdx, Type::Dummy});
                dummyDMATwoAttr.producesIn.push_back({commonBarrierIndex, Type::Real});
                _dummyDMAsToInsert.push_back(dummyDMATwoAttr);

            } else {
                /*
                    Special Case:
                    Example:
                    We have a TaskOp that waits on Barrier `C`, and `C` is also a barrier that the last task
                    of the parent group waits on. In this case, dummy DMAs cannot produce in `C`.

                    Solution:
                    Depending on the availability of barriers, we can resolve this in one of two ways:

                    Case 1: If all consumers of all wait barrier for last task of parent are before grand parent
                            we need to create a barrier

                    Case 2: If we have atleast 1 barrier from wait barrier of last parent task which has all users after
                    grand parent Then this barrier can be used for legalization but other can't e.g. we can use barrier
                    D as all users are after last grand parent task

                    Examples:

                    Case 1:
                    C[--TaskOp1--]X
                    D[--TaskOp2--]Y
                    A[--Last of GP--]B
                                            B[--DummyDMA1--]NewBarrier
                                            B[--DummyDMA2--]NewBarrier
                    NewBarrier,C,D[--Last of PG--]E

                    Case 2:
                    C[--TaskOp1--]X
                    A[--Last of GP--]B
                                            B[--DummyDMA1--]D
                                            B[--DummyDMA2--]D
                    D[--TaskOp2--]Y
                    C,D[--Last of PG--]E

                */
                Type type = Type::Real;
                auto lastParentWaitBarriersIdx = to_small_vector(barrierInfo.getWaitBarriers(lastParentTaskOpIdx));
                auto lastGrandParentUpdateBarriersIdx =
                        to_small_vector(barrierInfo.getUpdateBarriers(lastGrandParentTaskIdx));

                auto validUser = [&](size_t taskIdx) -> bool {
                    auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
                    // We must update all DMAs
                    if (VPURT::getTaskQueueType(taskOp, false).id == static_cast<int64_t>(queueType.id) ||
                        taskOp.getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
                        return true;
                    }
                    return false;
                };

                // Collect all barriers which can be used for legalizing
                SmallVector<size_t> barrierIndexesToUpdateByDummyDma;
                for (auto barrierIdx : lastParentWaitBarriersIdx) {
                    bool isUsedBeforeGrandParent = false;
                    for (auto barrierConsumer : barrierInfo.getBarrierConsumers(barrierIdx)) {
                        if (validUser(barrierConsumer) && barrierConsumer < lastGrandParentTaskIdx) {
                            isUsedBeforeGrandParent = true;
                            break;
                        }
                    }
                    if (!isUsedBeforeGrandParent) {
                        barrierIndexesToUpdateByDummyDma.push_back(barrierIdx);
                    }
                }

                // If no barriers were available for use, create a new barrier
                if (barrierIndexesToUpdateByDummyDma.empty()) {
                    type = Type::Dummy;
                    // Need to create new barrier for dummyDma -> lastParentTask dependency
                    size_t barOneDummyIdx = _newBarrierIndex++;
                    _newBarriersToInsert.push_back(lastParentWaitBarriersIdx[0]);

                    auto& consumerToAdd = _barrierAddConsumerProducerMap[{barOneDummyIdx, Type::Dummy}].first;
                    consumerToAdd.push_back({lastParentTaskOpIdx, Type::Real});
                    barrierIndexesToUpdateByDummyDma.push_back(barOneDummyIdx);
                }

                DummyDMAData dummyDMAOneAttr;
                dummyDMAOneAttr.insertionPoint = lastGrandParentTaskOpIdx;
                for (size_t barIdx : lastGrandParentUpdateBarriersIdx) {
                    dummyDMAOneAttr.consumes.push_back({barIdx, Type::Real});
                }
                for (size_t barrIdx : barrierIndexesToUpdateByDummyDma) {
                    dummyDMAOneAttr.producesIn.push_back({barrIdx, type});
                }
                _dummyDMAsToInsert.push_back(dummyDMAOneAttr);

                DummyDMAData dummyDMATwoAttr;
                dummyDMATwoAttr.insertionPoint = lastGrandParentTaskOpIdx;
                for (size_t barrIdx : lastGrandParentUpdateBarriersIdx) {
                    dummyDMATwoAttr.consumes.push_back({barrIdx, Type::Real});
                }
                for (size_t barrIdx : barrierIndexesToUpdateByDummyDma) {
                    dummyDMATwoAttr.producesIn.push_back({barrIdx, type});
                }
                _dummyDMAsToInsert.push_back(dummyDMATwoAttr);
            }
        }

        grandParentGroup = parentGroup;
        parentGroup = travelingGroup;

        ++groupIdx;
        if (groupIdx < executionGroup.size()) {
            travelingGroup = executionGroup[groupIdx];
        }
    }
}

void LegalizeScheduleForPartialWlmFetchDmasPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto module = netFunc->getParentOfType<mlir::ModuleOp>();
    auto barriersOps = netFunc.getOps<VPURT::DeclareVirtualBarrierOp>();
    auto numVirtualBarriers = static_cast<int64_t>(std::distance(barriersOps.begin(), barriersOps.end()));
    if (numVirtualBarriers > _virtualBarrierThreshold) {
        _log.info("Skip schedule legalization due to high number of barriers: {0}", numVirtualBarriers);
        VPU::setWorkloadManagementStatus(module, VPU::WorkloadManagementStatus::FAILED);
        return;
    }

    mlir::OpBuilder builder(netFunc);

    // Reuse the same Decl Buffer for all Dummy DMAs
    mlir::Value inBuffer = nullptr;
    mlir::Value outBuffer = nullptr;

    // Identify existing position of DeclareBufferOp, will be used as insertion point
    // for new tasks that will be inserted in IR
    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    auto bufferInsertionPoint = !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

    // Check for presense of atleast one DMA on port 0 channel 0
    auto taskOps = netFunc.getOps<VPURT::TaskOp>();
    VPUX_THROW_WHEN(taskOps.empty(), "Can not find TaskOp");

    auto dmaOps = taskOps | vpux::filtered([&](VPURT::TaskOp taskOp) {
                      return isDMAOnSupportedPortAndChannel(taskOp);
                  });
    auto filteredVector = to_small_vector(dmaOps);
    // No DMA is present, then create a DMA
    if (filteredVector.empty()) {
        inBuffer = VPUIP::createDummyBuffer(builder, bufferInsertionPoint);
        outBuffer = VPUIP::createDummyBuffer(builder, bufferInsertionPoint);
        auto firstTaskOp = *taskOps.begin();
        builder.setInsertionPoint(firstTaskOp);

        // It will be part of barrier Info later during analysis creation and dependencies will be added accordingly
        VPUIP::createSyncDMA(builder, inBuffer, outBuffer, 0, {}, {});
    }

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    SmallVector<std::pair<size_t, size_t>> blockRange;
    for (size_t blockIdx = 0; blockIdx < barrierInfo.getControlGraphBlockCount(); ++blockIdx) {
        auto [blockStartInd, blockEndInd] = barrierInfo.getControlGraphBlockTaskRange(
                blockIdx, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ true);
        blockRange.push_back({blockStartInd, blockEndInd});
    }

    _numAllTaskOps = barrierInfo.getNumOfTasks();
    VPURT::orderExecutionTasksAndBarriers(netFunc, barrierInfo, _log, true);

    // Build task queue type map for all queues in order to test paths between tasks on different FIFOs.
    barrierInfo.initializeTaskQueueTypeMap(
            {VPU::ExecutorKind::DMA_NN, VPU::ExecutorKind::DPU, VPU::ExecutorKind::SHAVE_ACT});
    barrierInfo.buildTaskQueueTypeMap();

    // Will have a map for each cluster along with task index of the task
    auto taskQueues = VPURT::getTaskOpQueues(netFunc, barrierInfo);

    VPURT::TaskQueueType queueType;
    queueType.type = VPU::ExecutorKind::DMA_NN;
    queueType.id = getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR);

    auto dQueue = taskQueues[queueType];
    _firstDMATaskOp = barrierInfo.getTaskOpAtIndex(dQueue[0]);
    _lastDMATaskOp = barrierInfo.getTaskOpAtIndex(dQueue[dQueue.size() - 1]);

    auto& execGroupAnalysis = getAnalysis<ExecutionGroupAnalysis>();
    execGroupAnalysis.logExecutionGroupTasks(_log);
    auto dpuGroups = execGroupAnalysis.getDPUExecutionGroups();
    auto swGroups = execGroupAnalysis.getActShvExecutionGroups();

    planDummyDMAAndBarriersInsertion(dpuGroups, barrierInfo, blockRange);
    planDummyDMAAndBarriersInsertion(swGroups, barrierInfo, blockRange);

    SmallVector<VPURT::TaskOp> dummyDmas;
    realizePlannedInsertions(builder, barrierInfo, bufferInsertionPoint, dummyDmas);
    finalizeBarrierInfo(barrierInfo, dummyDmas, netFunc, _log);
}

}  // namespace

//
// createLegalizeScheduleForPartialWlmFetchDmasPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createLegalizeScheduleForPartialWlmFetchDmasPass(
        const int virtualBarrierThreshold, Logger log) {
    return std::make_unique<LegalizeScheduleForPartialWlmFetchDmasPass>(virtualBarrierThreshold, log);
}
