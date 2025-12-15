//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/wlm_legalization_utils.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_ADDPLACEHOLDERFETCHDMAS
#define GEN_PASS_DEF_ADDPLACEHOLDERFETCHDMAS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
namespace {

//
//  AddPlaceholderFetchDMAsPass
//

using BlockRange = SmallVector<std::pair<size_t, size_t>>;

struct FetchDMAData {
    size_t insertionPoint = 0;
    SmallVector<IndexType> consumes;
    SmallVector<IndexType> producesIn;
    VPUIP::FetchDMAAttr fetchDmaAttr;
};

struct PlannedInsertionsData {
    size_t newBarrierIndex = 0;
    mlir::Operation* bufferInsertionPoint = nullptr;
    mlir::Operation* barrierInsertionPoint = nullptr;

    SmallVector<FetchDMAData> fetchDMAsToInsert;
    llvm::DenseMap<IndexType, std::pair<SmallVector<IndexType>, SmallVector<IndexType>>> barrierAddConsumerProducerMap;
};

class AddPlaceholderFetchDMAsPass final : public VPUIP::impl::AddPlaceholderFetchDMAsBase<AddPlaceholderFetchDMAsPass> {
public:
    explicit AddPlaceholderFetchDMAsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    void planFetchDMAAndBarriersInsertionPerQueue(BlockRange& blockRange, ExecutionGroupList& executionGroup,
                                                  BarrierInfo& barrierInfo, VPURT::TaskOp firstTaskOp,
                                                  PlannedInsertionsData& emptyInsertions);
    void realizePlannedInsertions(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                  PlannedInsertionsData& preparedInsertions);
};

VPURT::TaskOp createFetchDma(mlir::OpBuilder& builder, mlir::Value inputBuf, mlir::Value outputBuf,
                             BarrierInfo& barrierInfo, VPUIP::FetchDMAAttr fetchDMAData) {
    auto newDMA = createFetchDMA(builder, inputBuf, outputBuf, 0, {}, {}, fetchDMAData);
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

// Once we know the insertion point of DMAs this function creates actual DMAs in IR while also keeps a map of [index,
// DMAOp This map is used later to refer to real DMAOp and get real task-index from barrierInfo
void AddPlaceholderFetchDMAsPass::realizePlannedInsertions(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                                           PlannedInsertionsData& preparedInsertions) {
    auto inBuffer = VPUIP::createDummyBuffer(builder, preparedInsertions.bufferInsertionPoint);
    auto outBuffer = VPUIP::createDummyBuffer(builder, preparedInsertions.bufferInsertionPoint);
    SmallVector<VPURT::TaskOp> fetchDMAs;

    SmallVector<VPURT::DeclareVirtualBarrierOp> dummyBarriers;
    dummyBarriers.reserve(preparedInsertions.newBarrierIndex);

    // Create as many dummy barriers as were indexed during scheduling
    for (size_t i = 0; i < preparedInsertions.newBarrierIndex; ++i) {
        auto newBarrierOp =
                createNewBarrier(builder, barrierInfo, preparedInsertions.barrierInsertionPoint, nullptr, nullptr);
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

// Insert barrier and FetchDMA for each group
// LastOfGrandParent/FirstOfParent --> NewBarrier1 --> FetchDMA --> NewBarrier2 --> LastOfParent/SyncTask
void AddPlaceholderFetchDMAsPass::planFetchDMAAndBarriersInsertionPerQueue(BlockRange& blockRange,
                                                                           ExecutionGroupList& executionGroup,
                                                                           BarrierInfo& barrierInfo,
                                                                           VPURT::TaskOp firstTaskOp,
                                                                           PlannedInsertionsData& emptyInsertions) {
    // Always populate for whatever is available for first 2 groups
    for (size_t groupIdx = 0; groupIdx < std::min<size_t>(2, executionGroup.size()); ++groupIdx) {
        FetchDMAData fetchDMAData;
        auto insertionIndex = barrierInfo.getIndex(firstTaskOp);
        fetchDMAData.insertionPoint = insertionIndex;
        fetchDMAData.fetchDmaAttr = getFetchDMAAttr(groupIdx, barrierInfo, executionGroup[groupIdx].front());
        emptyInsertions.fetchDMAsToInsert.push_back(fetchDMAData);
    }

    // If less than 3 groups, skip the rest of the logic that depends on both being present
    if (executionGroup.size() < 3) {
        return;
    }

    size_t groupIdx = 2;
    auto grandParentGroup = executionGroup.front();
    auto parentGroup = executionGroup[1];
    auto travelingGroup = executionGroup[groupIdx];
    while (groupIdx < executionGroup.size()) {
        auto firstTaskParentGroup = parentGroup[0];
        auto lastTaskParentGroup = parentGroup.back();
        auto lastTaskGrandParentGroup = grandParentGroup[grandParentGroup.size() - 1];

        auto dummyBarrierOneProducer = firstTaskParentGroup;
        // If parent group only has one task, then we cannot enqueue in parallel to parent
        if (firstTaskParentGroup == lastTaskParentGroup) {
            dummyBarrierOneProducer = lastTaskGrandParentGroup;
        }

        auto dummyBarrierTwoConsumer = lastTaskParentGroup;

        FetchDMAData fetchDMAData;
        // If both tasks are in different blocks, we may need a sync task as consumer for PlaceholderFetchDMA
        if (!inSameTaskBlock(lastTaskParentGroup, dummyBarrierOneProducer, blockRange)) {
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

        // 1. LastOfGrandParent/FirstOfParent → B1
        size_t barOneDummyIdx = emptyInsertions.newBarrierIndex++;
        auto& producersToAdd = emptyInsertions.barrierAddConsumerProducerMap[{barOneDummyIdx, Type::Dummy}].second;
        producersToAdd.push_back({dummyBarrierOneProducer, Type::Real});

        // 2. B1 → FetchDMA → B2
        fetchDMAData.insertionPoint = dummyBarrierOneProducer;
        fetchDMAData.consumes = {{barOneDummyIdx, Type::Dummy}};

        size_t barTwoDummyIdx = emptyInsertions.newBarrierIndex++;
        fetchDMAData.producesIn = {{barTwoDummyIdx, Type::Dummy}};

        // 3. B2 → LastOfParent (or syncTask)
        auto& consumersToAdd = emptyInsertions.barrierAddConsumerProducerMap[{barTwoDummyIdx, Type::Dummy}].first;
        consumersToAdd.push_back({dummyBarrierTwoConsumer, Type::Real});

        fetchDMAData.fetchDmaAttr = getFetchDMAAttr(groupIdx, barrierInfo, travelingGroup.front());
        emptyInsertions.fetchDMAsToInsert.push_back(fetchDMAData);

        grandParentGroup = parentGroup;
        parentGroup = travelingGroup;

        ++groupIdx;
        if (groupIdx < executionGroup.size()) {
            travelingGroup = executionGroup[groupIdx];
        }
    }
}

void AddPlaceholderFetchDMAsPass::safeRunOnFunc() {
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

    // Get the blockRanges to check we don't add deps between blocks
    BlockRange blockRange;
    for (size_t blockIdx = 0; blockIdx < barrierInfo.getControlGraphBlockCount(); ++blockIdx) {
        auto [blockStartInd, blockEndInd] = barrierInfo.getControlGraphBlockTaskRange(
                blockIdx, /* blockStartSyncPoint */ false, /* blockEndSyncPoint */ true);
        blockRange.push_back({blockStartInd, blockEndInd});
    }

    // Build task queue type map for all queues in order to test paths between tasks on different FIFOs.
    barrierInfo.initializeTaskQueueTypeMap(
            {VPU::ExecutorKind::DMA_NN, VPU::ExecutorKind::DPU, VPU::ExecutorKind::SHAVE_ACT});
    barrierInfo.buildTaskQueueTypeMap();

    // Will have a map for each cluster along with task index of the task
    auto taskQueues = VPURT::getTaskOpQueues(netFunc, barrierInfo);

    // Initialize fetchDmaQueueType for DMA_NN executor on DDR channel, port 0
    VPURT::TaskQueueType fetchDmaQueueType{VPU::ExecutorKind::DMA_NN,
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
    execGroupAnalysis.logExecutionGroupTasks(_log);
    auto dpuGroups = execGroupAnalysis.getDPUExecutionGroups();
    auto swGroups = execGroupAnalysis.getActShvExecutionGroups();

    for (auto& [_, executionGroups] : dpuGroups) {
        planFetchDMAAndBarriersInsertionPerQueue(blockRange, executionGroups, barrierInfo, firstTaskOp,
                                                 dmaFetchInsertions);
    }
    for (auto& [_, executionGroups] : swGroups) {
        planFetchDMAAndBarriersInsertionPerQueue(blockRange, executionGroups, barrierInfo, firstTaskOp,
                                                 dmaFetchInsertions);
    }

    realizePlannedInsertions(builder, barrierInfo, dmaFetchInsertions);
    finalizeBarrierInfo(barrierInfo, netFunc, _log);

    // Log the number of inserted FetchDMAs
    if (!dmaFetchInsertions.fetchDMAsToInsert.empty()) {
        _log.info("Inserted '{0}' FetchDMAs", dmaFetchInsertions.fetchDMAsToInsert.size());
    }
}

}  // namespace

//
// createAddPlaceholderFetchDMAsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAddPlaceholderFetchDMAsPass(Logger log) {
    return std::make_unique<AddPlaceholderFetchDMAsPass>(log);
}
