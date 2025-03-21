//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/utils/wlm_legalization_utils.hpp"

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_LEGALIZESCHEDULEFORWLMFETCHDMAS
#define GEN_PASS_DEF_LEGALIZESCHEDULEFORWLMFETCHDMAS
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;
namespace {

//
//  LegalizeScheduleForWlmFetchDmasPass
//
class LegalizeScheduleForWlmFetchDmasPass final :
        public VPUIP::arch40xx::impl::LegalizeScheduleForWlmFetchDmasBase<LegalizeScheduleForWlmFetchDmasPass> {
public:
    explicit LegalizeScheduleForWlmFetchDmasPass(const int virtualBarrierThreshold,
                                                 WorkloadManagementMode workloadManagementMode, Logger log)
            : _virtualBarrierThreshold(virtualBarrierThreshold),
              _legalizeEachTileSeparately(workloadManagementMode != WorkloadManagementMode::PWLM_V0_LCA) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    int _virtualBarrierThreshold;
    bool _legalizeEachTileSeparately;
    void safeRunOnFunc() final;

    bool isValidDMA(BarrierInfo& barrierInfo, size_t dmaIdx);
    VPURT::TaskOp findLastDmaBeforeExecGroup(BarrierInfo& barrierInfo, ExecutionGroup& executionGroup);
    VPURT::TaskOp findFirstDmaAfterExecGroup(BarrierInfo& barrierInfo, ExecutionGroup& executionGroup);
    VPURT::TaskOp findDMAsThroughBarriersBFS(size_t startBarrier, BarrierInfo& barrierInfo, MinMaxOption option,
                                             bool bfsDirUp);

    VPURT::TaskOp createDummyDma(mlir::OpBuilder& builder, mlir::Value inputBuf, mlir::Value outputBuf,
                                 BarrierInfo& barrierInfo, SmallVector<VPURT::TaskOp>& dummyDmas);

    void insertDMAForFetchTasks(DenseMap<VPURT::TaskQueueType, ExecutionGroupList>& listOfExecutionGroups,
                                VPU::ExecutorKind executorKind, mlir::Operation* bufferInsertionPoint,
                                mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                SmallVector<VPURT::TaskOp>& dummyDmas, size_t tilesCount,
                                SmallVector<std::pair<size_t, size_t>>& blockRange);

    void legalizeForWorkloadManagementMode(DenseMap<VPURT::TaskQueueType, ExecutionGroupList>& listOfExecutionGroups,
                                           VPU::ExecutorKind executorKind, mlir::Operation* bufferInsertionPoint,
                                           mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                           SmallVector<VPURT::TaskOp>& dummyDmas, size_t tilesCount,
                                           SmallVector<std::pair<size_t, size_t>>& blockRange,
                                           VPURT::TaskQueueType queueType, mlir::Value& inBuffer,
                                           mlir::Value& outBuffer, size_t iterTile = 0);

    SmallVector<size_t> getDmasUpdatingBarriers(llvm::DenseSet<size_t>& barriers, BarrierInfo& barrierInfo);

private:
    // Will be initialized in safeRunOnFunc(), this is done to suppress the UNINIT_CTOR warning
    size_t _numAllTaskOps = 0;

    VPURT::TaskOp _firstDMATaskOp;
    VPURT::TaskOp _lastDMATaskOp;
};

// Returns a DMA which copies 0 len data from DDR to DDR
VPURT::TaskOp LegalizeScheduleForWlmFetchDmasPass::createDummyDma(mlir::OpBuilder& builder, mlir::Value inputBuf,
                                                                  mlir::Value outputBuf, BarrierInfo& barrierInfo,
                                                                  SmallVector<VPURT::TaskOp>& dummyDmas) {
    auto ctx = builder.getContext();
    auto dummyDmaLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "wlm_dummy_dma"));

    auto newDMAOp =
            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, mlir::ValueRange({}), mlir::ValueRange({}), dummyDmaLoc,
                                                  inputBuf, outputBuf, 0, false, false, nullptr, nullptr);
    auto newDMA = newDMAOp->getParentOfType<VPURT::TaskOp>();
    barrierInfo.addNewTaskOp(newDMA);
    dummyDmas.push_back(newDMA);
    return newDMA;
}

bool LegalizeScheduleForWlmFetchDmasPass::isValidDMA(BarrierInfo& barrierInfo, size_t dmaIdx) {
    auto taskOp = barrierInfo.getTaskOpAtIndex(dmaIdx);
    return taskOp.getExecutorKind() == VPU::ExecutorKind::DMA_NN && isDMAOnSupportedPortAndChannel(taskOp) &&
           dmaIdx < _numAllTaskOps;
}

VPURT::TaskOp LegalizeScheduleForWlmFetchDmasPass::findDMAsThroughBarriersBFS(size_t startBarrier,
                                                                              BarrierInfo& barrierInfo,
                                                                              MinMaxOption option, bool bfsDirUp) {
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
VPURT::TaskOp LegalizeScheduleForWlmFetchDmasPass::findFirstDmaAfterExecGroup(BarrierInfo& barrierInfo,
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
VPURT::TaskOp LegalizeScheduleForWlmFetchDmasPass::findLastDmaBeforeExecGroup(BarrierInfo& barrierInfo,
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

SmallVector<size_t> LegalizeScheduleForWlmFetchDmasPass::getDmasUpdatingBarriers(llvm::DenseSet<size_t>& barriers,
                                                                                 BarrierInfo& barrierInfo) {
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
void LegalizeScheduleForWlmFetchDmasPass::insertDMAForFetchTasks(
        DenseMap<VPURT::TaskQueueType, ExecutionGroupList>& listOfExecutionGroups, VPU::ExecutorKind executorKind,
        mlir::Operation* bufferInsertionPoint, mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
        SmallVector<VPURT::TaskOp>& dummyDmas, size_t tilesCount, SmallVector<std::pair<size_t, size_t>>& blockRange) {
    VPURT::TaskQueueType queueType;
    queueType.type = executorKind;
    queueType.id = 0;

    // Reuse the same Decl Buffer for all Dummy DMAs
    mlir::Value inBuffer = nullptr;
    mlir::Value outBuffer = nullptr;

    if (_legalizeEachTileSeparately) {
        for (size_t iterTile = 0; iterTile < tilesCount; ++iterTile) {
            queueType.id = iterTile;
            legalizeForWorkloadManagementMode(listOfExecutionGroups, executorKind, bufferInsertionPoint, builder,
                                              barrierInfo, dummyDmas, tilesCount, blockRange, queueType, inBuffer,
                                              outBuffer, iterTile);
        }
    } else {
        legalizeForWorkloadManagementMode(listOfExecutionGroups, executorKind, bufferInsertionPoint, builder,
                                          barrierInfo, dummyDmas, tilesCount, blockRange, queueType, inBuffer,
                                          outBuffer);
    }
}

void LegalizeScheduleForWlmFetchDmasPass::legalizeForWorkloadManagementMode(
        DenseMap<VPURT::TaskQueueType, ExecutionGroupList>& listOfExecutionGroups, VPU::ExecutorKind executorKind,
        mlir::Operation* bufferInsertionPoint, mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
        SmallVector<VPURT::TaskOp>& dummyDmas, size_t tilesCount, SmallVector<std::pair<size_t, size_t>>& blockRange,
        VPURT::TaskQueueType queueType, mlir::Value& inBuffer, mlir::Value& outBuffer, size_t iterTile) {
    auto executionGroupListForTile = listOfExecutionGroups[queueType];
    if (executionGroupListForTile.size() < 3) {
        return;
    }

    auto parentGroup = executionGroupListForTile[0];
    ExecutionGroup grandParentGroup;

    size_t groupIdx = 1;
    auto travelingGroup = executionGroupListForTile[groupIdx];

    while (groupIdx < executionGroupListForTile.size()) {
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

            auto lastGrandParentTaskOpIdx =
                    _legalizeEachTileSeparately
                            ? lastGrandParentTaskIdx
                            : getSiblingTaskOpOnTile(lastGrandParentTaskIdx, barrierInfo, tilesCount);
            auto lastGrandParentTaskOp = barrierInfo.getTaskOpAtIndex(lastGrandParentTaskOpIdx);

            auto lastGrandParentTaskUpdateBarriers = barrierInfo.getUpdateBarriers(lastGrandParentTaskOpIdx);
            auto lastGrandParentTaskWaitBarriers = barrierInfo.getWaitBarriers(lastGrandParentTaskOpIdx);

            auto lastParentTaskWaitBarriers = barrierInfo.getWaitBarriers(lastParentTaskOpIdx);
            auto lastParentTaskUpdateBarriers = barrierInfo.getUpdateBarriers(lastParentTaskOpIdx);

            if (!inSameTaskBlock(lastParentTaskOpIdx, lastGrandParentTaskIdx, blockRange)) {
                grandParentGroup = parentGroup;
                parentGroup = travelingGroup;

                ++groupIdx;
                if (groupIdx < executionGroupListForTile.size()) {
                    travelingGroup = executionGroupListForTile[groupIdx];
                }
                continue;
            }

            inBuffer = inBuffer != nullptr ? inBuffer : VPUIP::createDummyBuffer(builder, bufferInsertionPoint);
            outBuffer = outBuffer != nullptr ? outBuffer : VPUIP::createDummyBuffer(builder, bufferInsertionPoint);

            auto commonWaitBarrierOpt =
                    getEarliestCommonBarrier(lastGrandParentTaskWaitBarriers, lastParentTaskWaitBarriers, barrierInfo);
            auto commonUpdateBarrierOpt = getEarliestCommonBarrier(lastGrandParentTaskUpdateBarriers,
                                                                   lastParentTaskUpdateBarriers, barrierInfo);

            if (commonWaitBarrierOpt && commonUpdateBarrierOpt) {
                auto insertionBarrier = barrierInfo.getBarrierOpAtIndex(commonWaitBarrierOpt.value());
                auto newBarrierOneOp = createNewBarrier(builder, barrierInfo, insertionBarrier, nullptr, nullptr);
                auto newBarrierTwoOp = createNewBarrier(builder, barrierInfo, insertionBarrier, nullptr, nullptr);

                for (auto barrIdx : lastParentTaskWaitBarriers) {
                    auto barrOp = barrierInfo.getBarrierOpAtIndex(barrIdx);
                    if (_legalizeEachTileSeparately) {
                        barrierInfo.removeConsumer(barrOp, lastParentTaskOpIdx);
                        barrierInfo.addConsumer(newBarrierTwoOp, lastParentTaskOpIdx);
                    } else {
                        for (size_t tile = 0; tile < tilesCount; ++tile) {
                            // Along with updating dependencies for lastParentTaskOpIdx, we must also update the
                            // dependencies for sibling on other tiles
                            /*
                                Example: If the original consumers for BarrierX, for legalization also needs to consume
                                BarrierY then all the siblings on other tiles must also consume BarrierY This helps to
                                eliminate the need to perform legalization per tile
                                            |---->DPU0_0                            |---->DPU0_0
                                            |---->DPU0_1                            |---->DPU0_1
                                BarrierX                        =>        BarrierY
                                            |---->DPU0_2                            |---->DPU0_2
                                            |---->DPU0_3                            |---->DPU0_3

                                This is done inorder to also legalize the siblings as FetchTasks are for tasks per tile
                            */
                            auto siblingIdxOnTile =
                                    getSiblingTaskOpOnTile(lastParentTaskOpIdx, barrierInfo, tilesCount, tile);
                            if (siblingIdxOnTile != SIZE_MAX) {
                                barrierInfo.removeConsumer(barrOp, siblingIdxOnTile);
                                barrierInfo.addConsumer(newBarrierTwoOp, siblingIdxOnTile);
                            }
                        }
                    }
                }

                builder.setInsertionPointAfter(lastGrandParentTaskOp);
                auto dummyDmaOne = createDummyDma(builder, inBuffer, outBuffer, barrierInfo, dummyDmas);

                SmallVector<mlir::Value> produceIn = {newBarrierTwoOp};
                SmallVector<mlir::Value> consumes = {newBarrierOneOp};
                updateBarriersForDma(consumes, produceIn, dummyDmaOne, barrierInfo);

                for (auto barrIdx : lastGrandParentTaskUpdateBarriers) {
                    auto barrOp = barrierInfo.getBarrierOpAtIndex(barrIdx);
                    if (_legalizeEachTileSeparately) {
                        barrierInfo.removeProducer(barrOp, lastGrandParentTaskIdx);
                        barrierInfo.addProducer(newBarrierOneOp, lastGrandParentTaskIdx);
                    } else {
                        for (size_t tile = 0; tile < tilesCount; ++tile) {
                            auto siblingIdxOnTile =
                                    getSiblingTaskOpOnTile(lastGrandParentTaskIdx, barrierInfo, tilesCount, tile);
                            if (siblingIdxOnTile != SIZE_MAX) {
                                barrierInfo.removeProducer(barrOp, siblingIdxOnTile);
                                barrierInfo.addProducer(newBarrierOneOp, siblingIdxOnTile);
                            }
                        }
                    }
                }

                builder.setInsertionPointAfter(dummyDmaOne);
                auto dummyDmaTwo = createDummyDma(builder, inBuffer, outBuffer, barrierInfo, dummyDmas);
                updateBarriersForDma(consumes, produceIn, dummyDmaTwo, barrierInfo);

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
                    // Only need to change the barrier deps for the tasks after last task of grand parent
                    if (taskOp->isBeforeInBlock(lastGrandParentTaskOp) ||
                        taskIdx == barrierInfo.getIndex(lastGrandParentTaskOp)) {
                        return false;
                    }

                    // If we are legalizing per tile only then check if the task is on same tile
                    if (_legalizeEachTileSeparately) {
                        if (VPURT::getTaskQueueType(taskOp, false).id != static_cast<int64_t>(iterTile)) {
                            return false;
                        }
                    }

                    // Don't modify DMA and tasks which doesn't not have same type as tasks in GP as they will be
                    // legalized with insertFetchTask for DPU/SW
                    if (taskOp.getExecutorKind() == VPU::ExecutorKind::DMA_NN ||
                        taskOp.getExecutorKind() != executorKind) {
                        return false;
                    }
                    return true;
                };

                auto filteredRange = allConsumersOfWaitBarrier | vpux::filtered(std::move(validUser));
                auto filteredVector = to_small_vector(filteredRange);
                for (auto consumer : filteredVector) {
                    barrierInfo.removeConsumer(insertionBarrier, consumer);
                    barrierInfo.addConsumer(newBarrierTwoOp, consumer);
                }

            } else if (auto commonBarrOpt = getEarliestCommonBarrier(lastGrandParentTaskUpdateBarriers,
                                                                     lastParentTaskWaitBarriers, barrierInfo)) {
                auto commonBarrierIndex = commonBarrOpt.value();
                auto commonBarrierOp = barrierInfo.getBarrierOpAtIndex(commonBarrierIndex);

                builder.setInsertionPointAfter(lastGrandParentTaskOp);
                auto dummyDmaOne = createDummyDma(builder, inBuffer, outBuffer, barrierInfo, dummyDmas);
                auto newBarrierOneOp = createNewBarrier(builder, barrierInfo, commonBarrierOp, nullptr, nullptr);

                if (_legalizeEachTileSeparately) {
                    barrierInfo.removeProducer(commonBarrierIndex, lastGrandParentTaskIdx);
                    barrierInfo.addProducer(newBarrierOneOp, lastGrandParentTaskIdx);
                } else {
                    for (size_t tile = 0; tile < tilesCount; ++tile) {
                        auto siblingIdxOnTile =
                                getSiblingTaskOpOnTile(lastGrandParentTaskIdx, barrierInfo, tilesCount, tile);
                        if (siblingIdxOnTile != SIZE_MAX) {
                            barrierInfo.removeProducer(commonBarrierIndex, siblingIdxOnTile);
                            barrierInfo.addProducer(newBarrierOneOp, siblingIdxOnTile);
                        }
                    }
                }

                SmallVector<mlir::Value> produceIn = {commonBarrierOp.getBarrier()};
                SmallVector<mlir::Value> consumes = {newBarrierOneOp};
                updateBarriersForDma(consumes, produceIn, dummyDmaOne, barrierInfo);

                builder.setInsertionPointAfter(dummyDmaOne);
                auto dummyDmaTwo = createDummyDma(builder, inBuffer, outBuffer, barrierInfo, dummyDmas);
                updateBarriersForDma(consumes, produceIn, dummyDmaTwo, barrierInfo);

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
                auto lastParentWaitBarriersIdx = to_small_vector(barrierInfo.getWaitBarriers(lastParentTaskOpIdx));
                auto lastGrandParentUpdateBarriersIdx =
                        to_small_vector(barrierInfo.getUpdateBarriers(lastGrandParentTaskIdx));

                auto validUser = [&](size_t taskIdx) -> bool {
                    // If we're not legalizing per tile i.e. wlm mode is LCA based, in that case all users are legal so
                    // return early
                    if (!_legalizeEachTileSeparately) {
                        return true;
                    }
                    auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
                    if (VPURT::getTaskQueueType(taskOp, false).id == static_cast<int64_t>(iterTile)) {
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
                    // Need to create new barrier for dummyDma -> lastParentTask dependency
                    auto insertionPointBarrierOp = barrierInfo.getBarrierOpAtIndex(lastParentWaitBarriersIdx[0]);
                    auto newBarrierOp =
                            createNewBarrier(builder, barrierInfo, insertionPointBarrierOp, nullptr, nullptr);
                    barrierInfo.addConsumer(newBarrierOp, lastParentTaskOpIdx);
                    barrierIndexesToUpdateByDummyDma.push_back(barrierInfo.getIndex(newBarrierOp));
                }

                builder.setInsertionPointAfter(lastGrandParentTaskOp);
                auto dummyDmaOne = createDummyDma(builder, inBuffer, outBuffer, barrierInfo, dummyDmas);
                updateBarriersForDma(/*consumes*/ lastGrandParentUpdateBarriersIdx,
                                     /*producesIn*/ barrierIndexesToUpdateByDummyDma, dummyDmaOne, barrierInfo);

                builder.setInsertionPointAfter(dummyDmaOne);
                auto dummyDmaTwo = createDummyDma(builder, inBuffer, outBuffer, barrierInfo, dummyDmas);
                updateBarriersForDma(/*consumes*/ lastGrandParentUpdateBarriersIdx,
                                     /*producesIn*/ barrierIndexesToUpdateByDummyDma, dummyDmaTwo, barrierInfo);

                /*
                    DMA Ordering:
                    `firstDMA`    <- DMA which waits for the update barrier of the grandparent group
                    `lastDMA`     <- DMA which updates the wait barrier of the traveling group (TG),
                                     either directly or through a FIFO dependency.

                    During workload-management pass, if `lastDMA` is positioned before or equal to `firstDMA`,
                    then FetchTask cannot be inserted.

                    Post Legalization:
                    - DMA1 waits for the barrier (e.g., barrier 30) which is the update barrier for the grandparent
                        group (GP).
                    - DMA2 updates the wait barrier (e.g., barrier 15) for the traveling group (TG).

                    To address this, we ensure DMA1 and DMA2 also update barrier 15, making DMA2 the last DMA
                    and DMA1 the first DMA.

                    Final Layout:
                            [---DMAX---]15
                    20[---GP---]30
                            30[--DMA1--]35, 15
                            30[--DMA2--]35, 15
                    35[---PG---]40
                    15[---TG---]

                */
                auto lastDma = findLastDmaBeforeExecGroup(barrierInfo, travelingGroup);
                if (lastDma->isBeforeInBlock(dummyDmaOne)) {
                    auto travelingGroupWaitBarriers = barrierInfo.getWaitBarriers(travelingGroup[0]);
                    for (auto waitBarrier : travelingGroupWaitBarriers) {
                        barrierInfo.addProducer(waitBarrier, barrierInfo.getIndex(dummyDmaOne));
                        barrierInfo.addProducer(waitBarrier, barrierInfo.getIndex(dummyDmaTwo));
                    }
                }
            }
        }

        grandParentGroup = parentGroup;
        parentGroup = travelingGroup;

        ++groupIdx;
        if (groupIdx < executionGroupListForTile.size()) {
            travelingGroup = executionGroupListForTile[groupIdx];
        }
    }
}

void LegalizeScheduleForWlmFetchDmasPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto module = netFunc->getParentOfType<mlir::ModuleOp>();

    // For all WLM Mode except PWLM_V0_LCA perform legalization per tile
    if (workloadManagementModeOpt.hasValue()) {
        _legalizeEachTileSeparately = workloadManagementModeOpt.getValue() != WorkloadManagementMode::PWLM_V0_LCA;
    }

    auto barriersOps = netFunc.getOps<VPURT::DeclareVirtualBarrierOp>();
    auto numVirtualBarriers = static_cast<int64_t>(std::distance(barriersOps.begin(), barriersOps.end()));
    if (numVirtualBarriers > _virtualBarrierThreshold) {
        _log.info("Skip schedule legalization due to high number of barriers: {0}", numVirtualBarriers);
        vpux::VPUIP::setWlmStatus(module, vpux::VPUIP::WlmStatus::FAILED);
        return;
    }

    mlir::OpBuilder builder(netFunc);
    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = static_cast<size_t>(IE::getTileExecutor(parentModule).getCount());

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    SmallVector<std::pair<size_t, size_t>> blockRange;
    for (size_t blockIdx = 0; blockIdx < barrierInfo.getControlGraphBlockCount(); ++blockIdx) {
        auto [blockStartInd, blockEndInd] = barrierInfo.getControlGraphBlockTaskRange(
                blockIdx, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ true);
        blockRange.push_back({blockStartInd, blockEndInd});
    }

    _numAllTaskOps = barrierInfo.getNumOfTasks();
    VPURT::orderExecutionTasksAndBarriers(netFunc, barrierInfo, _log, true);
    barrierInfo.buildTaskQueueTypeMap();

    // Identify existing position of DeclareBufferOp, will be used as insertion point
    // for new tasks that will be inserted in IR
    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    auto bufferInsertionPoint = !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

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
    auto listOfDPUExecutionGroups = execGroupAnalysis.getDPUExecutionGroups();
    auto listOfSWExecutionGroups = execGroupAnalysis.getActShvExecutionGroups();

    SmallVector<VPURT::TaskOp> dummyDmas;
    insertDMAForFetchTasks(listOfDPUExecutionGroups, VPU::ExecutorKind::DPU, bufferInsertionPoint, builder, barrierInfo,
                           dummyDmas, tilesCount, blockRange);

    insertDMAForFetchTasks(listOfSWExecutionGroups, VPU::ExecutorKind::SHAVE_ACT, bufferInsertionPoint, builder,
                           barrierInfo, dummyDmas, tilesCount, blockRange);

    // Apply the changes now as we need to make sure the new DMAs don't break the schedule
    barrierInfo.updateIR();

    // Log the number of inserted DMAs for fetch legalization
    if (!dummyDmas.empty()) {
        _log.info("Inserted '{0}' DMAs for fetch legalization", dummyDmas.size());
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
    VPURT::orderExecutionTasksAndBarriers(netFunc, barrierInfo, _log);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(netFunc);
}

}  // namespace

//
// createLegalizeScheduleForWlmFetchDmasPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createLegalizeScheduleForWlmFetchDmasPass(
        const int virtualBarrierThreshold, WorkloadManagementMode workloadManagementMode, Logger log) {
    return std::make_unique<LegalizeScheduleForWlmFetchDmasPass>(virtualBarrierThreshold, workloadManagementMode, log);
}
