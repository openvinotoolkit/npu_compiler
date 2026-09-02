//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/wlm_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/options.hpp"

#include <queue>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_ADDPLACEHOLDERFETCHDMASPWLM
#define GEN_PASS_DEF_ADDPLACEHOLDERFETCHDMASPWLM
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
namespace {

//
//  createAddPlaceholderFetchDMAsPWLMPass
//

// This pass will insert placeholder FetchDMAs into the PWLM schedule for execution groups
// For DPU it will insert FetchDMAs between last grand parent group task (gpN-1) and last
// parent group task (pN-1) and make sure proper barrier dependencies are added if needed
// DPU: gpN-1 -> .. -> FetchDMA -> .. -> pN-1
// For SHV it will insert FetchDMAs between last parent group task (pN-1) and last
// parent group task (pN-1) and make sure proper barrier dependencies are added if needed.
// Since in PWLM there is no explicit compiler SHV FIFO control compiler needs to make sure
// previous to last grand parent task (gpN-2) has dependency to FetchDMA and previous
// to last parent task (pN-2) depends on FetchDMA
// SHV:  gpN-2  gpN-1                         pN-2  pN-1
//           \-----\--> .. -> FetchDMA -> .. ->/-----/

using BlockRange = SmallVector<std::pair<size_t, size_t>>;
struct FetchDMAData {
    size_t insertionPoint = 0;
    SmallVector<VPURT::IndexType> consumes;
    SmallVector<VPURT::IndexType> producesIn;
    VPUIP::FetchDMAAttr fetchDmaAttr;
};
struct PlannedInsertionsData {
    size_t newBarrierIndex = 0;
    mlir::Operation* bufferInsertionPoint = nullptr;
    mlir::Operation* barrierInsertionPoint = nullptr;

    SmallVector<FetchDMAData> fetchDMAsToInsert;
    llvm::DenseMap<VPURT::IndexType, std::pair<SmallVector<VPURT::IndexType>, SmallVector<VPURT::IndexType>>>
            barrierAddConsumerProducerMap;
};
// Structure that contains information about FetchDMA barriers:
// waitBarProducers -> waitBar -> FetchDMA -> updateBar -> updateBarConsumers
struct FetchDmaBarrierData {
    llvm::SmallVector<size_t, 4> waitBarProducers;    // tasks that produce FetchDMA waitBar
    llvm::SmallVector<size_t, 4> updateBarConsumers;  // tasks that consume FetchDMA updateBar
};
class AddPlaceholderFetchDMAsPWLMPass final :
        public VPUIP::impl::AddPlaceholderFetchDMAsPWLMBase<AddPlaceholderFetchDMAsPWLMPass> {
public:
    explicit AddPlaceholderFetchDMAsPWLMPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    void planFetchDMAAndBarriersInsertionPerQueue(BlockRange& blockRange, ExecutionGroupList& executionGroups,
                                                  BarrierInfo& barrierInfo, VPURT::TaskOp firstTaskOp,
                                                  PlannedInsertionsData& emptyInsertions,
                                                  VPURT::TaskQueueType fetchDmaQueueType);
    void planFetchDMAAndBarriersInsertionPerQueueWithCommonSHVList(BlockRange& blockRange,
                                                                   ExecutionGroupList& executionGroups,
                                                                   BarrierInfo& barrierInfo, VPURT::TaskOp firstTaskOp,
                                                                   PlannedInsertionsData& emptyInsertions,
                                                                   VPURT::TaskQueueType fetchDmaQueueType);

    void realizePlannedInsertions(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                  PlannedInsertionsData& preparedInsertions);
};

// Returns a DMA which copies 0 len data from DDR to CMX
VPURT::TaskOp createFetchDma(mlir::OpBuilder& builder, mlir::Value inputBuf, mlir::Value outputBuf,
                             BarrierInfo& barrierInfo, VPUIP::FetchDMAAttr fetchDMAData) {
    auto newDMA = VPURT::createFetchDMA(builder, inputBuf, outputBuf, 0, {}, {}, fetchDMAData);
    barrierInfo.addNewTaskOp(newDMA);
    return newDMA;
}

void finalizeBarrierInfo(BarrierInfo& barrierInfo, mlir::func::FuncOp netFunc, Logger& log) {
    // Update IR, verify schedules
    VPURT::orderExecutionTasksAndBarriers(netFunc, barrierInfo, log);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(netFunc);
}

// Once we know the insertion point of DMAs this function creates actual DMAs in IR while also keeps a map of
// [index, DMAOp This map is used later to refer to real DMAOp and get real task-index from barrierInfo
void AddPlaceholderFetchDMAsPWLMPass::realizePlannedInsertions(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                                               PlannedInsertionsData& preparedInsertions) {
    auto inBuffer = VPUIP::createDummyBuffer(builder, preparedInsertions.bufferInsertionPoint, VPU::MemoryKind::DDR);
    auto outBuffer =
            VPUIP::createDummyBuffer(builder, preparedInsertions.bufferInsertionPoint, VPU::MemoryKind::CMX_NN);
    SmallVector<VPURT::TaskOp> fetchDMAs;

    SmallVector<VPURT::DeclareVirtualBarrierOp> dummyBarriers;
    dummyBarriers.reserve(preparedInsertions.newBarrierIndex);

    // Create as many dummy barriers as were indexed during scheduling
    for (size_t i = 0; i < preparedInsertions.newBarrierIndex; ++i) {
        auto newBarrierOp = VPURT::createNewBarrier(builder, barrierInfo, preparedInsertions.barrierInsertionPoint,
                                                    nullptr, nullptr);
        dummyBarriers.push_back(newBarrierOp);
    }

    for (const auto& [dummyDmaIndex, value] : preparedInsertions.fetchDMAsToInsert | indexed) {
        auto insertionPointOp = barrierInfo.getTaskOpAtIndex(value.insertionPoint);
        // Ensure fetch DMAs for first 2 groups are always first in the list
        if (value.fetchDmaAttr.getExecGroupIdx().getValue().getSExtValue() < 2) {
            builder.setInsertionPoint(insertionPointOp);
        } else {
            builder.setInsertionPointAfter(insertionPointOp);
        }
        auto dummyDMA = createFetchDma(builder, inBuffer, outBuffer, barrierInfo, value.fetchDmaAttr);
        fetchDMAs.push_back(dummyDMA);
    }

    // We have created the new DMAs and barriers, adjust dependencies
    for (const auto& [dummyDmaIndex, value] : preparedInsertions.fetchDMAsToInsert | indexed) {
        SmallVector<size_t> realProducesIn;
        SmallVector<size_t> realConsumes;
        for (auto produce : value.producesIn) {
            auto realBarrierIdx = getIndexOfBarrier(produce, dummyBarriers, barrierInfo);
            realProducesIn.push_back(realBarrierIdx);
        }
        for (auto consume : value.consumes) {
            auto realBarrierIdx = getIndexOfBarrier(consume, dummyBarriers, barrierInfo);
            realConsumes.push_back(realBarrierIdx);
        }
        updateBarriersForDma(realConsumes, realProducesIn, fetchDMAs[dummyDmaIndex], barrierInfo);
    }

    for (const auto& [indexType, value] : preparedInsertions.barrierAddConsumerProducerMap) {
        auto realBarrierIdx = getIndexOfBarrier(indexType, dummyBarriers, barrierInfo);
        for (auto consumer : value.first) {
            auto realTaskIdx = getIndexOfTask(consumer, fetchDMAs, barrierInfo);
            barrierInfo.addConsumer(realBarrierIdx, realTaskIdx);
        }
        for (auto producer : value.second) {
            auto realTaskIdx = getIndexOfTask(producer, fetchDMAs, barrierInfo);
            barrierInfo.addProducer(realBarrierIdx, realTaskIdx);
        }
    }
}

// Prepare data for inserting Fetch DMA for DPU group in such a way that Fetch DMA
// would depend on last task of grand parent group and last task of parent group would depend on Fetch DMA:
//
//  lastTaskGrandParentGroup -|-> ... --> FetchDMA --> ... ->-|-> lastTaskGrandParentGroup
//
// If possible reuse existing barriers and dependencies between DPU FIFOs and DMA FIFO where Fetch DMA is inserted
// Otherwise prepare data for new barriers
void AddPlaceholderFetchDMAsPWLMPass::planFetchDMAAndBarriersInsertionPerQueue(
        BlockRange& blockRange, ExecutionGroupList& executionGroup, BarrierInfo& barrierInfo, VPURT::TaskOp firstTaskOp,
        PlannedInsertionsData& emptyInsertions, VPURT::TaskQueueType fetchDmaQueueType) {
    // Always populate for whatever is available for first 2 groups
    for (size_t groupIdx = 0; groupIdx < std::min<size_t>(2, executionGroup.size()); ++groupIdx) {
        FetchDMAData fetchDMAData;
        auto insertionIndex = barrierInfo.getIndex(firstTaskOp);
        fetchDMAData.insertionPoint = insertionIndex;
        fetchDMAData.fetchDmaAttr = VPURT::getFetchDMAAttr(groupIdx, barrierInfo, executionGroup[groupIdx].front());
        emptyInsertions.fetchDMAsToInsert.push_back(fetchDMAData);
    }

    // If less than 3 groups, skip the rest of the logic that depends on both being present
    if (executionGroup.size() < 3) {
        return;
    }

    std::optional<size_t> blockIdxOfTaskControlMap;
    std::pair<SmallVector<llvm::BitVector>, size_t> taskControlMapAndOffset;

    size_t groupIdx = 2;
    auto grandParentGroup = executionGroup.front();
    auto parentGroup = executionGroup[1];
    auto currentGroup = executionGroup[groupIdx];
    while (groupIdx < executionGroup.size()) {
        _log.trace("Planning FetchDMA for execution group {0}", groupIdx);
        _log = _log.nest();
        VPUX_THROW_WHEN(grandParentGroup.empty(), "GrandParent execution group is empty");
        VPUX_THROW_WHEN(parentGroup.empty(), "Parent execution group is empty");

        auto firstTaskParentGroup = parentGroup[0];
        auto lastTaskParentGroup = parentGroup.back();
        auto lastTaskGrandParentGroup = grandParentGroup.back();
        _log.trace("lastTaskGrandParentGroup {0}, firstTaskParentGroup {1}, lastTaskParentGroup {2}",
                   lastTaskGrandParentGroup, firstTaskParentGroup, lastTaskParentGroup);

        // Find if there is an existing DMA that can be reused as insertion point
        // lastTaskGrandParentGroup -> ... -> SomeDMA -> ... -> lastTaskParentGroup
        // FetchDMA for current group can be placed right before such SomeDMA
        std::optional<size_t> reuseDma = std::nullopt;
        auto dmaTask1Opt = barrierInfo.getNextTaskOnQueue(lastTaskGrandParentGroup, fetchDmaQueueType);
        auto dmaTask2Opt = barrierInfo.getPrevTaskOnQueue(lastTaskParentGroup, fetchDmaQueueType);
        if (dmaTask1Opt.has_value() && dmaTask2Opt.has_value() && dmaTask1Opt.value() <= dmaTask2Opt.value()) {
            _log.trace("Check if any DMA between {0} and {1} can be reused", dmaTask1Opt.value(), dmaTask2Opt.value());

            auto currentDma = dmaTask1Opt;
            // Scan if there is any DMA between dmaTask1Opt and dmaTask2Opt that can be reused
            while (currentDma.has_value() && currentDma.value() <= dmaTask2Opt.value()) {
                _log.trace("Considering DMA task at index {0}", currentDma.value());
                if (barrierInfo.isDepFromTaskAToTaskB(lastTaskGrandParentGroup, currentDma.value(),
                                                      taskControlMapAndOffset, blockIdxOfTaskControlMap) &&
                    barrierInfo.isDepFromTaskAToTaskB(currentDma.value(), lastTaskParentGroup, taskControlMapAndOffset,
                                                      blockIdxOfTaskControlMap)) {
                    _log.nest().trace("Reusing existing DMA task at index {0} for FetchDMA of group {1}",
                                      currentDma.value(), groupIdx);
                    reuseDma = currentDma;
                    break;
                }
                currentDma = barrierInfo.getNextTaskOnQueue(currentDma.value(), fetchDmaQueueType);
            }
        }

        FetchDMAData fetchDMAData;
        if (reuseDma.has_value()) {
            //  reuseDmaWaitBar -> FetchDMA → reuseDMA
            fetchDMAData.insertionPoint = reuseDma.value() - 1;
            for (auto waitBar : barrierInfo.getWaitBarriers(reuseDma.value())) {
                _log.nest().trace("Adding existing wait barrier {0} to FetchDMA", waitBar);
                fetchDMAData.consumes.push_back({waitBar, VPURT::Type::Real});
            }
        } else {
            auto dummyBarrierOneProducer = lastTaskGrandParentGroup;
            auto dummyBarrierTwoConsumer = lastTaskParentGroup;

            // If both tasks are in different blocks, we may need a sync task as consumer for PlaceholderFetchDMA
            if (!VPURT::inSameTaskBlock(lastTaskParentGroup, dummyBarrierOneProducer, blockRange)) {
                auto syncPoint = barrierInfo.getControlGraphSyncPoint(dummyBarrierOneProducer);
                // dummyBarrierOneProducer is NOT the sync task — safe to use sync as consumer
                if (dummyBarrierOneProducer != syncPoint.value()) {
                    dummyBarrierTwoConsumer = syncPoint.value();
                } else {
                    // dummyBarrierOneProducer IS the sync task
                    auto blockInd1 = barrierInfo.getControlGraphBlockIndex(dummyBarrierOneProducer);
                    auto blockInd2 = barrierInfo.getControlGraphBlockIndex(lastTaskParentGroup);

                    if (blockInd1 + 1 == blockInd2) {
                        // Tasks are in consecutive blocks — use parent directly as consumer
                        dummyBarrierTwoConsumer = lastTaskParentGroup;
                    } else {
                        // Need next block's sync point as consumer
                        auto nextSync = barrierInfo.getNextBlockSyncPoint(dummyBarrierOneProducer);
                        VPUX_THROW_UNLESS(nextSync.has_value(), "No next block sync point found for FetchDMA consumer");
                        dummyBarrierTwoConsumer = nextSync.value();
                    }
                }
            }

            // 1. LastOfGrandParent → B1
            _log.nest().trace("Add new barrier to create dependency from gpN_1 {0} to FetchDMA",
                              lastTaskGrandParentGroup);
            size_t barOneDummyIdx = emptyInsertions.newBarrierIndex++;
            auto& producersToAdd =
                    emptyInsertions.barrierAddConsumerProducerMap[{barOneDummyIdx, VPURT::Type::Dummy}].second;
            producersToAdd.push_back({dummyBarrierOneProducer, VPURT::Type::Real});

            // 2. B1 → FetchDMA → B2
            fetchDMAData.insertionPoint = dummyBarrierOneProducer;
            fetchDMAData.consumes = {{barOneDummyIdx, VPURT::Type::Dummy}};

            size_t barTwoDummyIdx = emptyInsertions.newBarrierIndex++;
            fetchDMAData.producesIn = {{barTwoDummyIdx, VPURT::Type::Dummy}};

            // 3. B2 → LastOfParent (or syncTask)
            _log.nest().trace("Add new barrier to create dependency from FetchDMA to pN_1 {0}", lastTaskParentGroup);
            auto& consumersToAdd =
                    emptyInsertions.barrierAddConsumerProducerMap[{barTwoDummyIdx, VPURT::Type::Dummy}].first;
            consumersToAdd.push_back({dummyBarrierTwoConsumer, VPURT::Type::Real});
        }

        fetchDMAData.fetchDmaAttr = VPURT::getFetchDMAAttr(groupIdx, barrierInfo, currentGroup.front());
        emptyInsertions.fetchDMAsToInsert.push_back(fetchDMAData);

        grandParentGroup = parentGroup;
        parentGroup = currentGroup;

        ++groupIdx;
        if (groupIdx < executionGroup.size()) {
            currentGroup = executionGroup[groupIdx];
        }
        _log = _log.unnest();
    }
}

// Prepare data for inserting Fetch DMA for SHV group in such a way that Fetch DMA
// would depend on 2 last tasks of grand parent group and 2 last task of parent group would depend on Fetch DMA:
//
// secondLastTaskGrandParentGroup ---|                               |-------> lastTaskGrandParentGroup
//         lastTaskGrandParentGroup -|-> ... --> FetchDMA --> ... ->-|-> secondLastTaskGrandParentGroup
//
// If possible reuse existing barriers and dependencies between SHV FIFOs and DMA FIFO where Fetch DMA is inserted
// Otherwise prepare data for new barriers
// SHV tasks require specific handling because of shared HW FIFO between SHV engines on same tile used in PWLM
// Compiler cannot assume that SHV[tileX][taskN+1] is blocked by SHV[tileX][taskN] and because of that compiler
// needs to guarantee that both last and last-1 tasks of a group are a dependency to or have dependency from Fetch DMA
// Such approach for descriptor fetch guarantee is possible because backend will always put SHV[tileX][taskN] and
// SHV[tileX][taskN-1] on different enqueue WorkItem tasks
void AddPlaceholderFetchDMAsPWLMPass::planFetchDMAAndBarriersInsertionPerQueueWithCommonSHVList(
        BlockRange& blockRange, ExecutionGroupList& executionGroup, BarrierInfo& barrierInfo, VPURT::TaskOp firstTaskOp,
        PlannedInsertionsData& emptyInsertions, VPURT::TaskQueueType fetchDmaQueueType) {
    // Always populate for whatever is available for first 2 groups
    for (size_t groupIdx = 0; groupIdx < std::min<size_t>(2, executionGroup.size()); ++groupIdx) {
        FetchDMAData fetchDMAData;
        auto insertionIndex = barrierInfo.getIndex(firstTaskOp);
        fetchDMAData.insertionPoint = insertionIndex;
        fetchDMAData.fetchDmaAttr = VPURT::getFetchDMAAttr(groupIdx, barrierInfo, executionGroup[groupIdx].front());
        emptyInsertions.fetchDMAsToInsert.push_back(fetchDMAData);
    }

    // If less than 3 groups, skip the rest of the logic that depends on both being present
    if (executionGroup.size() < 3) {
        return;
    }

    std::optional<size_t> blockIdxOfTaskControlMap;
    std::pair<SmallVector<llvm::BitVector>, size_t> taskControlMapAndOffset;

    size_t groupIdx = 2;
    auto grandParentGroup = executionGroup.front();
    auto parentGroup = executionGroup[1];
    auto travelingGroup = executionGroup[groupIdx];
    while (groupIdx < executionGroup.size()) {
        _log.trace("Planning FetchDMA for execution group {0}", groupIdx);
        _log = _log.nest();
        // If a current group exists we're sure that both grandParent and parent groups have at least 2 tasks
        const auto grandParentGroupSize = grandParentGroup.size();
        VPUX_THROW_WHEN(grandParentGroupSize < 2,
                        "For group {0} GrandParent execution group is incomplete. Has {1} tasks.", groupIdx,
                        grandParentGroupSize);

        const size_t secondLastTaskGrandParentGroup = *(grandParentGroup.rbegin() + 1);
        const size_t lastTaskGrandParentGroup = *(grandParentGroup.rbegin());

        const auto parentGroupSize = parentGroup.size();
        VPUX_THROW_WHEN(parentGroupSize < 2, "For group {0} Parent execution group is incomplete. Has {1} tasks.",
                        groupIdx, parentGroupSize);

        const size_t secondLastTaskParentGroup = *(parentGroup.rbegin() + 1);
        const size_t lastTaskParentGroup = *(parentGroup.rbegin());

        _log.trace("secondLastTaskGrandParentGroup {0}, lastTaskGrandParentGroup {1}, secondLastTaskParentGroup {2}, "
                   "lastTaskParentGroup {3}",
                   secondLastTaskGrandParentGroup, lastTaskGrandParentGroup, secondLastTaskParentGroup,
                   lastTaskParentGroup);

        // Find if there is an existing DMA that can be reused
        std::optional<size_t> reuseDma = std::nullopt;
        auto dmaTask1Opt = barrierInfo.getNextTaskOnQueue(lastTaskGrandParentGroup, fetchDmaQueueType);
        auto dmaTask2Opt = barrierInfo.getPrevTaskOnQueue(secondLastTaskParentGroup, fetchDmaQueueType);
        if (dmaTask1Opt.has_value() && dmaTask2Opt.has_value() && dmaTask1Opt.value() <= dmaTask2Opt.value()) {
            _log.trace("Check if any DMA between {0} and {1} can be reused", dmaTask1Opt.value(), dmaTask2Opt.value());

            auto currentDma = dmaTask1Opt;
            // Scan if there is any DMA between dmaTask1Opt and dmaTask2Opt that can be reused
            while (currentDma.has_value() && currentDma.value() <= dmaTask2Opt.value()) {
                _log.trace("Considering DMA task at index {0}", currentDma.value());
                if (barrierInfo.isDepFromTaskAToTaskB(lastTaskGrandParentGroup, currentDma.value(),
                                                      taskControlMapAndOffset, blockIdxOfTaskControlMap) &&
                    barrierInfo.isDepFromTaskAToTaskB(currentDma.value(), secondLastTaskParentGroup,
                                                      taskControlMapAndOffset, blockIdxOfTaskControlMap)) {
                    _log.nest().trace("Reusing existing DMA task at index {0} for FetchDMA of group {1}",
                                      currentDma.value(), groupIdx);
                    reuseDma = currentDma;
                    break;
                }
                currentDma = barrierInfo.getNextTaskOnQueue(currentDma.value(), fetchDmaQueueType);
            }
        }

        FetchDMAData fetchDMAData;
        if (reuseDma.has_value()) {
            //  reuseDmaWaitBar -> FetchDMA → reuseDMA
            fetchDMAData.insertionPoint = reuseDma.value() - 1;
            auto fetchDmaWaitBars = barrierInfo.getWaitBarriers(reuseDma.value());
            for (auto waitBar : fetchDmaWaitBars) {
                _log.nest().trace("Adding existing wait barrier {0} to FetchDMA", waitBar);
                fetchDMAData.consumes.push_back({waitBar, VPURT::Type::Real});
            }

            // Check if there is a dependency from secondLastOfGrandParent to lastOfGrandParent or from
            // secondLastOfGrandParent to reuseDMA. If there is then there is no need to add extra barrier
            if (!barrierInfo.isDepFromTaskAToTaskB(secondLastTaskGrandParentGroup, lastTaskGrandParentGroup,
                                                   taskControlMapAndOffset, blockIdxOfTaskControlMap) &&
                !barrierInfo.isDepFromTaskAToTaskB(secondLastTaskGrandParentGroup, reuseDma.value(),
                                                   taskControlMapAndOffset, blockIdxOfTaskControlMap)) {
                // If there is no dependency from secondLastOfGrandParent to FetchDMA, we need to add it
                // Use existing FetchDMA wait barrier
                auto fetchDmaWaitBar = *fetchDmaWaitBars.begin();

                _log.nest().trace(
                        "Add new dependency from secondLastTaskGrandParentGroup task {0} to FetchDMA wait barrier {1}",
                        secondLastTaskGrandParentGroup, fetchDmaWaitBar);
                auto& producersToAdd =
                        emptyInsertions.barrierAddConsumerProducerMap[{fetchDmaWaitBar, VPURT::Type::Real}].second;
                producersToAdd.push_back({secondLastTaskGrandParentGroup, VPURT::Type::Real});
            }

            // Check if there is a dependency from secondLastTaskParentGroup to lastTaskParentGroup or from
            // reuseDMA to lastTaskParentGroup. If there is then there is no need to add extra barrier
            if (!barrierInfo.isDepFromTaskAToTaskB(secondLastTaskParentGroup, lastTaskParentGroup,
                                                   taskControlMapAndOffset, blockIdxOfTaskControlMap) &&
                !barrierInfo.isDepFromTaskAToTaskB(reuseDma.value(), lastTaskParentGroup, taskControlMapAndOffset,
                                                   blockIdxOfTaskControlMap)) {
                // If there is no dependency from FetchDMA to lastTaskParentGroup, we need to add it
                // Make lastTaskParentGroup wait of same barriers as secondLastTaskParentGroup
                auto secondLastTaskParentGroupWaitBars = barrierInfo.getWaitBarriers(secondLastTaskParentGroup);
                for (auto waitBar : secondLastTaskParentGroupWaitBars) {
                    auto& consumersToAdd =
                            emptyInsertions.barrierAddConsumerProducerMap[{waitBar, VPURT::Type::Real}].first;
                    consumersToAdd.push_back({lastTaskParentGroup, VPURT::Type::Real});
                }
            }

        } else {
            auto dummyBarrierOneProducer = lastTaskGrandParentGroup;
            auto dummyBarrierTwoConsumer = secondLastTaskParentGroup;

            // If both tasks are in different blocks, we may need a sync task as consumer for PlaceholderFetchDMA
            if (!VPURT::inSameTaskBlock(dummyBarrierOneProducer, dummyBarrierTwoConsumer, blockRange)) {
                _log.nest().trace("Tasks {0} and {1} are in different blocks, checking for sync task",
                                  dummyBarrierOneProducer, dummyBarrierTwoConsumer);

                auto syncPoint = barrierInfo.getControlGraphSyncPoint(dummyBarrierOneProducer);
                _log.nest().trace("Found sync point at task {0}", syncPoint.value());
                // dummyBarrierOneProducer is NOT the sync task — safe to use sync as consumer
                if (dummyBarrierOneProducer != syncPoint.value()) {
                    dummyBarrierTwoConsumer = syncPoint.value();
                    _log.nest().trace("Using sync point {0} as consumer for FetchDMA", dummyBarrierTwoConsumer);
                } else {
                    // dummyBarrierOneProducer IS the sync task
                    auto blockInd1 = barrierInfo.getControlGraphBlockIndex(dummyBarrierOneProducer);
                    auto blockInd2 = barrierInfo.getControlGraphBlockIndex(dummyBarrierTwoConsumer);
                    _log.nest().trace("dummyBarrierOneProducer is the sync task. Block indices {0}, {1}", blockInd1,
                                      blockInd2);

                    if (blockInd1 + 1 == blockInd2) {
                        // Tasks are in consecutive blocks — use parent directly as consumer
                        dummyBarrierTwoConsumer = secondLastTaskParentGroup;
                        _log.nest().trace("Tasks are in consecutive blocks, using secondLastTaskParentGroup {0} as "
                                          "consumer for FetchDMA",
                                          dummyBarrierTwoConsumer);
                    } else {
                        // Need next block's sync point as consumer
                        auto nextSync = barrierInfo.getNextBlockSyncPoint(dummyBarrierOneProducer);
                        VPUX_THROW_UNLESS(nextSync.has_value(), "No next block sync point found for FetchDMA consumer");
                        dummyBarrierTwoConsumer = nextSync.value();
                        _log.nest().trace("Using next block sync point {0} as consumer for FetchDMA",
                                          dummyBarrierTwoConsumer);
                    }
                }
            }

            // 1. LastOfGrandParent → B1
            _log.nest().trace("Add new barrier to create dependency from lastTaskGrandParentGroup {0} to FetchDMA",
                              lastTaskGrandParentGroup);
            size_t barOneDummyIdx = emptyInsertions.newBarrierIndex++;
            auto& producersToAdd =
                    emptyInsertions.barrierAddConsumerProducerMap[{barOneDummyIdx, VPURT::Type::Dummy}].second;
            producersToAdd.push_back({dummyBarrierOneProducer, VPURT::Type::Real});

            if (!barrierInfo.isDepFromTaskAToTaskB(secondLastTaskGrandParentGroup, lastTaskGrandParentGroup,
                                                   taskControlMapAndOffset, blockIdxOfTaskControlMap)) {
                // If there is no dependency from secondLastOfGrandParent to lastTaskGrandParentGroup, we need to add it
                // Use existing FetchDMA wait barrier - barOneDummyIdx
                producersToAdd.push_back({secondLastTaskGrandParentGroup, VPURT::Type::Real});
            }

            // 2. B1 → FetchDMA → B2
            fetchDMAData.insertionPoint = dummyBarrierOneProducer;
            fetchDMAData.consumes = {{barOneDummyIdx, VPURT::Type::Dummy}};
            size_t barTwoDummyIdx = emptyInsertions.newBarrierIndex++;
            fetchDMAData.producesIn = {{barTwoDummyIdx, VPURT::Type::Dummy}};

            // 3. B2 → SecondLastOfParent (or syncTask)
            _log.nest().trace("Add new barrier to create dependency from FetchDMA to secondLastTaskParentGroup {0}",
                              secondLastTaskParentGroup);
            auto& consumersToAdd =
                    emptyInsertions.barrierAddConsumerProducerMap[{barTwoDummyIdx, VPURT::Type::Dummy}].first;
            consumersToAdd.push_back({dummyBarrierTwoConsumer, VPURT::Type::Real});

            // Check if there is a dependency from secondLastTaskParentGroup to lastTaskParentGroup or if
            // fetch DMA insertion point block index is different than lastTaskParentGroup block index.
            if (VPURT::inSameTaskBlock(fetchDMAData.insertionPoint + 1, lastTaskParentGroup, blockRange) &&
                !barrierInfo.isDepFromTaskAToTaskB(secondLastTaskParentGroup, lastTaskParentGroup,
                                                   taskControlMapAndOffset, blockIdxOfTaskControlMap)) {
                // If there is no dependency from secondLastTaskParentGroup to lastTaskParentGroup, we need to add it
                // Make lastTaskParentGroup wait on same barriers as secondLastTaskParentGroup
                consumersToAdd.push_back({lastTaskParentGroup, VPURT::Type::Real});
            }
        }

        fetchDMAData.fetchDmaAttr = VPURT::getFetchDMAAttr(groupIdx, barrierInfo, travelingGroup.front());
        emptyInsertions.fetchDMAsToInsert.push_back(fetchDMAData);

        grandParentGroup = parentGroup;
        parentGroup = travelingGroup;

        ++groupIdx;
        if (groupIdx < executionGroup.size()) {
            travelingGroup = executionGroup[groupIdx];
        }
        _log = _log.unnest();
    }
}

void AddPlaceholderFetchDMAsPWLMPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    mlir::OpBuilder builder(netFunc);
    PlannedInsertionsData dmaFetchInsertions;
    // Identify existing position of DeclareBufferOp, will be used as insertion point
    // for new tasks that will be inserted in IR
    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    dmaFetchInsertions.bufferInsertionPoint =
            !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

    auto barrierOps = netFunc.getOps<VPURT::DeclareVirtualBarrierOp>();
    dmaFetchInsertions.barrierInsertionPoint =
            !barrierOps.empty() ? *barrierOps.begin() : &netFunc.getBody().front().front();

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    const int numBarriersBefore = barrierInfo.getNumOfBarrierOps();

    // Get the blockRanges to check we don't add deps between blocks
    BlockRange blockRange;
    for (size_t blockIdx = 0; blockIdx < barrierInfo.getControlGraphBlockCount(); ++blockIdx) {
        auto [blockStartInd, blockEndInd] = barrierInfo.getControlGraphBlockTaskRange(
                blockIdx, /* blockStartSyncPoint */ false, /* blockEndSyncPoint */ true);
        blockRange.push_back({blockStartInd, blockEndInd});
    }

    // Build task queue type map for all queues in order to test paths between tasks on different FIFOs.
    barrierInfo.buildTaskQueueTypeMap();

    // Will have a map for each cluster along with task index of the task
    auto taskQueues = VPURT::getTaskOpQueues(netFunc, barrierInfo);

    // Initialize fetchDmaQueueType for DMA_NN executor on DDR channel, port 0
    VPURT::TaskQueueType fetchDmaQueueType{config::ExecutorKind::DMA_NN,
                                           getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR)};
    // firstTaskOp is used as insertion point for FetchDMAs for initial 2 execution groups
    // If we have any DMAs on supported port and channel then FetchDMAs must be placed before them
    // If we don't have any DMAs on suported port and channel we can just place FetchDMA before first TaskOp
    auto taskOps = netFunc.getOps<VPURT::TaskOp>();
    VPUX_THROW_WHEN(taskOps.empty(), "Can not find TaskOp");

    VPURT::TaskOp firstTaskOp = *taskOps.begin();
    if (!taskQueues[fetchDmaQueueType].empty()) {
        firstTaskOp = barrierInfo.getTaskOpAtIndex(taskQueues[fetchDmaQueueType].front());
    }

    auto& execGroupAnalysis = getAnalysis<ExecutionGroupAnalysis>();
    auto dpuGroups = execGroupAnalysis.getDPUExecutionGroups();
    auto swGroups = execGroupAnalysis.getActShvExecutionGroups();

    for (auto& [taskQueueType, executionGroups] : dpuGroups) {
        _log.trace("Planning FetchDMA and Barriers for queue {0}:{1}", stringifyEnum(taskQueueType.type),
                   taskQueueType.id);
        planFetchDMAAndBarriersInsertionPerQueue(blockRange, executionGroups, barrierInfo, firstTaskOp,
                                                 dmaFetchInsertions, fetchDmaQueueType);
    }
    for (auto& [taskQueueType, executionGroups] : swGroups) {
        _log.trace("Planning FetchDMA and Barriers for queue {0}:{1}", stringifyEnum(taskQueueType.type),
                   taskQueueType.id);
        planFetchDMAAndBarriersInsertionPerQueueWithCommonSHVList(blockRange, executionGroups, barrierInfo, firstTaskOp,
                                                                  dmaFetchInsertions, fetchDmaQueueType);
    }

    if (dmaFetchInsertions.fetchDMAsToInsert.empty()) {
        _log.info("No FetchDMAs to insert");
        barrierInfo.clearAttributes();
        return;
    }

    realizePlannedInsertions(builder, barrierInfo, dmaFetchInsertions);
    finalizeBarrierInfo(barrierInfo, netFunc, _log);

    // Log the number of inserted FetchDMAs
    _log.info("Inserted '{0}' FetchDMAs", dmaFetchInsertions.fetchDMAsToInsert.size());
    const int numBarriersAfter = barrierInfo.getNumOfBarrierOps();
    _log.info("Inserted '{0}' barriers, before: '{1}', after '{2}'", numBarriersAfter - numBarriersBefore,
              numBarriersBefore, numBarriersAfter);

    // After insertion perform verification if FetchDMAs satisfy constraints
    barrierInfo = BarrierInfo{netFunc};
    barrierInfo.buildTaskQueueTypeMap();

    execGroupAnalysis = ExecutionGroupAnalysis{netFunc};
    dpuGroups = execGroupAnalysis.getDPUExecutionGroups();
    swGroups = execGroupAnalysis.getActShvExecutionGroups();

    VPUX_THROW_WHEN(!verifyFetchDmaDependencies(netFunc, barrierInfo, dpuGroups, _log),
                    "Unsafe dependencies for Fetch DMA around DPUs");
    VPUX_THROW_WHEN(!verifyFetchDmaDependencies(netFunc, barrierInfo, swGroups, _log),
                    "Unsafe dependencies for Fetch DMA around SHVs");
    barrierInfo.clearAttributes();
}

}  // namespace

//
// createAddPlaceholderFetchDMAsPWLMPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAddPlaceholderFetchDMAsPWLMPass(Logger log) {
    return std::make_unique<AddPlaceholderFetchDMAsPWLMPass>(log);
}
