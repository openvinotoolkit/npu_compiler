//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/wlm_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include <map>
#include <set>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_PREPARESHAVESUBMITDMAS
#define GEN_PASS_DEF_PREPARESHAVESUBMITDMAS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
using PortChannelKey = std::pair<uint32_t, VPUIP::DmaChannelType>;

namespace {

enum class DMAType { Fetch = 0, Sync = 1 };

struct DMAData {
    size_t insertionPoint;
    uint32_t port;
    VPUIP::DmaChannelType channelType;
    SmallVector<VPURT::IndexType> consumes;
    SmallVector<VPURT::IndexType> producesIn;
    DMAType dmaType;
    VPUIP::FetchDMAAttr fetchDmaAttr;
    bool isShvSyncDma = false;
    int64_t logicalTaskIndex = -1;
};

struct PlannedInsertionsData {
    size_t newDmaIndex = 0;
    size_t newBarrierIndex = 0;
    mlir::Operation* bufferInsertionPoint = nullptr;
    mlir::Operation* barrierInsertionPoint = nullptr;

    SmallVector<DMAData> dmasToInsert;
    llvm::DenseMap<VPURT::IndexType, std::pair<SmallVector<VPURT::IndexType>, SmallVector<VPURT::IndexType>>>
            barrierAddConsumerProducerMap;
};

class PrepareShaveSubmitDMAsPass final : public VPUIP::impl::PrepareShaveSubmitDMAsBase<PrepareShaveSubmitDMAsPass> {
public:
    explicit PrepareShaveSubmitDMAsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    void planLegalization(BarrierInfo& barrierInfo, const SmallVector<size_t>& shvTasksWithDma,
                          PlannedInsertionsData& preparedInsertions, uint32_t numDmaPorts,
                          SmallVector<VPURT::TaskOp>& tasksToVerify, Logger& log);

    void realizePlannedInsertions(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                  PlannedInsertionsData& preparedInsertions, SmallVector<VPURT::TaskOp>& tasksToVerify,
                                  Logger& log);

    void legalizeDependenciesToControlBlockBoundaries(BarrierInfo& barrierInfo,
                                                      const std::set<size_t>& impactedControlBlockIndices,
                                                      uint32_t numDmaPorts, Logger& log);
};

void dumpShvTasksWithDma(const SmallVector<size_t>& shvTasksWithDma, Logger& log) {
    log.trace("SHV TaskOps with DMAs found at the following task indices:");
    for (auto taskIdx : shvTasksWithDma) {
        log.trace(" - Task index {0}", taskIdx);
    }
}

void findShvTasksWithDmaAndImpactedControlBlocks(SmallVector<size_t>& shvTasksWithDma,
                                                 std::set<size_t>& impactedControlBlockIndices,
                                                 BarrierInfo& barrierInfo, Logger& log) {
    log.trace("Finding SHV TaskOps with DMAs and grouping them by logical task index");
    for (size_t taskIdx = 0; taskIdx < barrierInfo.getNumOfTasks(); taskIdx++) {
        if (barrierInfo.getTaskQueueType(taskIdx).type != config::ExecutorKind::SHAVE_ACT) {
            continue;
        }
        auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
        auto swKernelOp = mlir::cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp());
        if (!isIoDmaSwKernel(swKernelOp)) {
            continue;
        }
        shvTasksWithDma.push_back(taskIdx);
        impactedControlBlockIndices.insert(barrierInfo.getControlGraphBlockIndex(taskIdx));
    }
}

// Returns a DMA which copies 0 len data from DDR/CMX to DDR
VPURT::TaskOp createDmaForGivenType(mlir::OpBuilder& builder, mlir::Value inputBuf, mlir::Value outputBuf,
                                    BarrierInfo& barrierInfo, const DMAData& dmaData) {
    // Create sync DMA based on queue type
    VPURT::TaskOp newDMA;
    switch (dmaData.dmaType) {
    case DMAType::Sync: {
        const auto logicalTaskAttr =
                dmaData.isShvSyncDma ? builder.getI64IntegerAttr(dmaData.logicalTaskIndex) : nullptr;
        newDMA = VPUIP::createSyncDMA(builder, inputBuf, outputBuf, dmaData.port, {}, {},
                                      dmaData.isShvSyncDma ? "shv_submit_sync_dma" : "shv_submit_guard_sync_dma",
                                      logicalTaskAttr);
        break;
    }
    case DMAType::Fetch:
        newDMA = VPURT::createFetchDMA(builder, inputBuf, outputBuf, dmaData.port, {}, {}, dmaData.fetchDmaAttr,
                                       "shv_submit_fetch_dma");
        break;
    default:
        VPUX_THROW("Unknown DMAType");
        break;
    }
    barrierInfo.addNewTaskOp(newDMA);
    return newDMA;
}

// Once we know the insertion point of DMAs this function creates actual DMAs in IR while also keeps a map of [index,
// DMAOp This map is used later to refer to real DMAOp and get real task-index from barrierInfo
void PrepareShaveSubmitDMAsPass::realizePlannedInsertions(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                                          PlannedInsertionsData& preparedInsertions,
                                                          SmallVector<VPURT::TaskOp>& tasksToVerify, Logger& log) {
    log.trace("Realizing planned insertions of Fetch/Sync DMAs and barriers in IR");

    auto toMemoryKind = [](VPUIP::DmaChannelType type) -> VPU::MemoryKind {
        switch (type) {
        case VPUIP::DmaChannelType::DDR:
            return VPU::MemoryKind::DDR;

        case VPUIP::DmaChannelType::CMX:
            return VPU::MemoryKind::CMX_NN;

        case VPUIP::DmaChannelType::NOT_SPECIFIED:
            VPUX_THROW("DmaChannelType::NOT_SPECIFIED is not valid here");

        default:
            VPUX_THROW("Unknown DmaChannelType");
        }
    };

    mlir::Value inBuffer;
    mlir::Value outBuffer = VPUIP::createDummyBuffer(builder, preparedInsertions.bufferInsertionPoint);

    SmallVector<VPURT::TaskOp> insertedDMAs;
    insertedDMAs.reserve(preparedInsertions.newDmaIndex);

    SmallVector<VPURT::DeclareVirtualBarrierOp> dummyBarriers;
    dummyBarriers.reserve(preparedInsertions.newBarrierIndex);

    // Create as many dummy barriers as were indexed during scheduling
    for (size_t i = 0; i < preparedInsertions.newBarrierIndex; ++i) {
        auto newBarrierOp = VPURT::createNewBarrier(builder, barrierInfo, preparedInsertions.barrierInsertionPoint,
                                                    nullptr, nullptr);
        dummyBarriers.push_back(newBarrierOp);
    }

    // This holds the buffer for each channel type which is used in newly created DMAs. We create one buffer per channel
    // type and reuse it for all DMAs of that channel type
    DenseMap<PortChannelKey, mlir::Value> taskQueueBufferMap;
    for (const auto& [_, value] : preparedInsertions.dmasToInsert | indexed) {
        auto insertionPointOp = barrierInfo.getTaskOpAtIndex(value.insertionPoint);
        builder.setInsertionPoint(insertionPointOp);

        PortChannelKey key{value.port, value.channelType};
        auto it = taskQueueBufferMap.find(key);
        if (it != taskQueueBufferMap.end()) {
            log.trace("Reusing buffer for port {0}, channel {1}", value.port, value.channelType);
            inBuffer = it->second;
        } else {
            log.trace("Creating buffer for port {0}, channel {1}", value.port, value.channelType);

            inBuffer = VPUIP::createDummyBuffer(builder, preparedInsertions.bufferInsertionPoint,
                                                toMemoryKind(value.channelType));
            taskQueueBufferMap[key] = inBuffer;
        }

        auto dummyDMA = createDmaForGivenType(builder, inBuffer, outBuffer, barrierInfo, value);
        tasksToVerify.push_back(dummyDMA);
        insertedDMAs.push_back(dummyDMA);
    }

    // We have created the new DMAs and barriers, adjust dependencies
    for (const auto& [dummyDmaIndex, value] : preparedInsertions.dmasToInsert | indexed) {
        SmallVector<size_t> realProducesIn;
        SmallVector<size_t> realConsumes;

        for (auto produce : value.producesIn) {
            auto realBarrierIdx = getIndexOfBarrier(produce, dummyBarriers, barrierInfo);
            log.trace("Add dependency between task at index {0} and barrier at index {1}", dummyDmaIndex,
                      realBarrierIdx);
            realProducesIn.push_back(realBarrierIdx);
        }
        for (auto consume : value.consumes) {
            auto realBarrierIdx = getIndexOfBarrier(consume, dummyBarriers, barrierInfo);
            log.trace("Add dependency between task at index {0} and barrier at index {1}", dummyDmaIndex,
                      realBarrierIdx);
            realConsumes.push_back(realBarrierIdx);
        }
        updateBarriersForDma(realConsumes, realProducesIn, insertedDMAs[dummyDmaIndex], barrierInfo);
    }

    for (const auto& [indexType, value] : preparedInsertions.barrierAddConsumerProducerMap) {
        auto realBarrierIdx = getIndexOfBarrier(indexType, dummyBarriers, barrierInfo);
        for (auto consumer : value.first) {
            auto realTaskIdx = getIndexOfTask(consumer, insertedDMAs, barrierInfo);
            log.trace("Add dependency between barrier at index {0} and task at index {1}", realBarrierIdx, realTaskIdx);
            barrierInfo.addConsumer(realBarrierIdx, realTaskIdx);
        }
        for (auto producer : value.second) {
            auto realTaskIdx = getIndexOfTask(producer, insertedDMAs, barrierInfo);
            log.trace("Add dependency between barrier at index {0} and task at index {1}", realBarrierIdx, realTaskIdx);
            barrierInfo.addProducer(realBarrierIdx, realTaskIdx);
        }
    }
}

/*
This function plans legalization of SHV tasks submitting DMAs by inserting required Fetch along with Sync DMAs
and barriers to ensure that all fetches are completed and DMA Descriptors for Skip are in CMX before SHV starts
executing.

        -> Fetch(Skip_list0_DDR) -|
        -> Fetch(Skip_list1_DDR) -|-> BAR -> |-> SyncDMA(shv_sync, DDR) [Skip Position] -> [optional release SyncDMA]
        -> Fetch(Skip_list0_CMX) -|          |-> SyncDMA(shv_sync, CMX) [Skip Position] -> [optional release SyncDMA]
        -> Fetch(Skip_list1_CMX) -|          |
                                             |-----------------------> SHV[list0](withDMA)
                                             |-----------------------> SHV[list1](withDMA)
*/
void PrepareShaveSubmitDMAsPass::planLegalization(BarrierInfo& barrierInfo, const SmallVector<size_t>& shvTasksWithDma,
                                                  PlannedInsertionsData& preparedInsertions, uint32_t numDmaPorts,
                                                  SmallVector<VPURT::TaskOp>& tasksToVerify, Logger& log) {
    log.trace("Planning legalization of SHV submit Fetch/Sync DMAs");

    std::map<size_t, SmallVector<size_t>> shvGroups;

    // Group by logical task index
    for (auto taskIdx : shvTasksWithDma) {
        auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
        tasksToVerify.push_back(taskOp);
        auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp());
        VPUX_THROW_UNLESS(swKernelOp != nullptr, "Expected SwKernelOp inside SHV TaskOp");

        const auto logicalTaskAttr = swKernelOp->getAttr(VPUIP::LOGICAL_TASK_INDEX_ATTR_NAME);
        VPUX_THROW_UNLESS(logicalTaskAttr != nullptr,
                          "SwKernelOp at index '{0}' is missing '{1}' IntegerAttr required by split SHV submit pass",
                          taskIdx, VPUIP::LOGICAL_TASK_INDEX_ATTR_NAME);

        const auto logicalIdx = mlir::cast<mlir::IntegerAttr>(logicalTaskAttr).getValue().getSExtValue();
        shvGroups[logicalIdx].push_back(taskIdx);
    }

    if (_log.isActive(LogLevel::Trace)) {
        dumpShvTasksWithDma(shvTasksWithDma, log);
    }

    log.trace("Planning insertions for SHV tasks with DMAs. Number of logical groups: {0}", shvGroups.size());
    const auto channelDDR = static_cast<size_t>(VPUIP::DmaChannelType::DDR);
    const auto channelCMX = static_cast<size_t>(VPUIP::DmaChannelType::CMX);
    size_t descId = 0;

    // Go over SHV tasks with same logicalTaskIndex and create DMAs and barriers for them. The plan is to create one
    // barrier per logical task, then add fetch DMAs as producers to the barrier and SHV tasks as consumers. This way we
    // ensure that all fetches are done before SHV starts executing.
    for (auto& [logicalIdx, taskIndices] : shvGroups) {
        auto earliestIdx = taskIndices.front();
        auto latestIdx = taskIndices.back();
        auto blockIdx = barrierInfo.getControlGraphBlockIndex(earliestIdx);

        // One barrier per logical task
        auto groupBarrierIdx = preparedInsertions.newBarrierIndex++;
        auto releaseBarrierIdx = preparedInsertions.newBarrierIndex++;

        // Temporary storage to enforce ordering later
        SmallVector<DMAData> plannedFetches;
        mlir::OpBuilder builder(barrierInfo.getTaskOpAtIndex(earliestIdx));

        for (auto taskIdx : taskIndices) {
            log.trace("Planning Fetches and skips for logical task index {0} with {1} SHV tasks", logicalIdx,
                      taskIndices.size());
            SmallVector<mlir::Attribute> descIdAttrs;
            auto swTaskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
            auto swOp = mlir::cast<VPUIP::SwKernelOp>(swTaskOp.getInnerTaskOp());
            auto taskQueueType = VPURT::getTaskQueueType(swTaskOp, false);

            auto tile = VPURT::getTileIndexForDpuOrShv(swTaskOp, taskQueueType);
            auto list = VPURT::getListIndexForDpuOrShv(swTaskOp);

            for (uint32_t port = 0; port < numDmaPorts; ++port) {
                for (auto channel : {channelDDR, channelCMX}) {
                    VPUX_UNUSED(channel);
                    const size_t curDescId = descId++;
                    descIdAttrs.push_back(builder.getI64IntegerAttr(curDescId));

                    DMAData fetch;
                    preparedInsertions.newDmaIndex++;
                    fetch.insertionPoint = earliestIdx;
                    fetch.channelType = VPUIP::DmaChannelType::DDR;
                    fetch.port = port;
                    fetch.dmaType = DMAType::Fetch;
                    fetch.fetchDmaAttr =
                            VPURT::getFetchDMAAttr(logicalIdx, barrierInfo, taskIdx, tile, list, curDescId, true);
                    fetch.producesIn = {{groupBarrierIdx, VPURT::Type::Dummy}};
                    plannedFetches.push_back(fetch);
                }
            }
            swOp.setSkipDescIdsAttr(builder.getArrayAttr(descIdAttrs));
        }

        // Insert Fetches in preparedInsertions first so that they are guaranteed to be before Sync and Skip DMAs in IR
        for (auto& fetch : plannedFetches) {
            preparedInsertions.dmasToInsert.push_back(fetch);
        }

        // Insert Sync DMAs for both channels. These DMAs will be used to update the barrier and ensure that SHV waits
        // for fetches to complete and descriptors to be ready in CMX before starting execution
        log.trace("Planning Sync DMAs for logical task index {0} ", logicalIdx);
        SmallVector<DMAData> plannedRelease;
        for (uint32_t port = 0; port < numDmaPorts; ++port) {
            DMAData syncDDR;
            preparedInsertions.newDmaIndex++;
            syncDDR.insertionPoint = earliestIdx;
            syncDDR.channelType = VPUIP::DmaChannelType::DDR;
            syncDDR.port = port;
            syncDDR.dmaType = DMAType::Sync;
            syncDDR.isShvSyncDma = true;
            syncDDR.logicalTaskIndex = logicalIdx;
            syncDDR.consumes = {{groupBarrierIdx, VPURT::Type::Dummy}};
            preparedInsertions.dmasToInsert.push_back(syncDDR);

            DMAData syncCMX;
            preparedInsertions.newDmaIndex++;
            syncCMX.insertionPoint = earliestIdx;
            syncCMX.channelType = VPUIP::DmaChannelType::CMX;
            syncCMX.port = port;
            syncCMX.dmaType = DMAType::Sync;
            syncCMX.isShvSyncDma = true;
            syncCMX.logicalTaskIndex = logicalIdx;
            syncCMX.consumes = {{groupBarrierIdx, VPURT::Type::Dummy}};
            preparedInsertions.dmasToInsert.push_back(syncCMX);

            DMAData releaseSyncDDR;
            preparedInsertions.newDmaIndex++;
            releaseSyncDDR.insertionPoint = latestIdx;
            releaseSyncDDR.channelType = VPUIP::DmaChannelType::DDR;
            releaseSyncDDR.port = port;
            releaseSyncDDR.dmaType = DMAType::Sync;
            releaseSyncDDR.logicalTaskIndex = logicalIdx;
            releaseSyncDDR.consumes = {{releaseBarrierIdx, VPURT::Type::Dummy}};
            plannedRelease.push_back(releaseSyncDDR);

            DMAData releaseSyncCMX;
            preparedInsertions.newDmaIndex++;
            releaseSyncCMX.insertionPoint = latestIdx;
            releaseSyncCMX.channelType = VPUIP::DmaChannelType::CMX;
            releaseSyncCMX.port = port;
            releaseSyncCMX.dmaType = DMAType::Sync;
            releaseSyncCMX.logicalTaskIndex = logicalIdx;
            releaseSyncCMX.consumes = {{releaseBarrierIdx, VPURT::Type::Dummy}};
            plannedRelease.push_back(releaseSyncCMX);
        }

        // Insert Fetches and ShaveSync in preparedInsertions first followed by ReleaseSync
        for (auto& release : plannedRelease) {
            preparedInsertions.dmasToInsert.push_back(release);
        }
        // Ask SHV to wait for fetches by adding them as dependencies to the barrier, and then add SHV tasks as
        // consumers of the barrier as well
        for (auto taskIdx : taskIndices) {
            log.trace("dependency for release {0} on barrier {1} added", taskIdx, releaseBarrierIdx);
            auto& producersToAdd =
                    preparedInsertions.barrierAddConsumerProducerMap[{releaseBarrierIdx, VPURT::Type::Dummy}].second;
            producersToAdd.push_back({taskIdx, VPURT::Type::Real});

            if (blockIdx < barrierInfo.getControlGraphBlockIndex(taskIdx)) {
                // No need to add dependency as rest of SHV tasks are in different block and there is guaranteed
                // dependency through block sync task
                continue;
            }
            log.trace("dependency for task {0} on barrier {1} added", taskIdx, groupBarrierIdx);
            auto& consumersToAdd =
                    preparedInsertions.barrierAddConsumerProducerMap[{groupBarrierIdx, VPURT::Type::Dummy}].first;
            consumersToAdd.push_back({taskIdx, VPURT::Type::Real});
        }
    }
}

// To not break control block boundaries this method ensures that:
// - first tasks on each FIFO depends on previous block sync task update barrier
//      - this is not required for first block (block 0)
// - last task on each FIFO has dependency to sync-task wait barrier
//      - for last block dependency is needed to final barrier but final barrier will be added later in compilation
//   If SHV Sync DMA is the the last task on queue then need to insert additional
//   sync DMA on same queue after it, this acts as release DMA for Skip DMAs
void PrepareShaveSubmitDMAsPass::legalizeDependenciesToControlBlockBoundaries(
        BarrierInfo& barrierInfo, const std::set<size_t>& impactedControlBlockIndices, uint32_t numDmaPorts,
        Logger& log) {
    log.trace("Legalizing dependencies to control block boundaries");

    mlir::DenseSet<VPURT::TaskQueueType> impactedQueues;
    for (size_t dmaPort = 0; dmaPort < numDmaPorts; ++dmaPort) {
        for (auto channel : {VPUIP::DmaChannelType::DDR, VPUIP::DmaChannelType::CMX}) {
            impactedQueues.insert({config::ExecutorKind::DMA_NN, getDMAQueueIdEncoding(dmaPort, channel)});
        }
    }

    for (auto blockIdx : impactedControlBlockIndices) {
        log.trace("Legalizing dependencies for control block {0}", blockIdx);

        for (auto& queueType : impactedQueues) {
            log.trace(" - Impacted queue: {0}:{1}", queueType.type, queueType.id);

            // Link previous block -> first task
            if (blockIdx > 0) {
                // Make sure first tasks on each FIFO has dependency from previous block sync task update barrier
                auto prevBlockSyncTaskOpt = barrierInfo.getControlGraphSyncPointForBlock(blockIdx - 1);
                VPUX_THROW_WHEN(!prevBlockSyncTaskOpt.has_value(), "Previous block sync point not found for block {0}",
                                blockIdx - 1);
                auto prevBlockSyncTaskIdx = prevBlockSyncTaskOpt.value();
                auto prevBlockSyncTaskQueueType = barrierInfo.getTaskQueueType(prevBlockSyncTaskIdx);

                if (prevBlockSyncTaskQueueType != queueType) {
                    auto updateBarrier = barrierInfo.getUpdateBarriers(prevBlockSyncTaskIdx);
                    auto firstTaskOpt = barrierInfo.getNextTaskOnQueue(prevBlockSyncTaskIdx, queueType);
                    VPUX_THROW_WHEN(!firstTaskOpt.has_value(), "First task not found after sync task at index {0}",
                                    prevBlockSyncTaskIdx);

                    auto firstTaskIdx = firstTaskOpt.value();

                    if (barrierInfo.getWaitBarriers(firstTaskIdx).empty()) {
                        auto firstTaskOp = barrierInfo.getTaskOpAtIndex(firstTaskIdx);
                        VPUX_THROW_UNLESS(
                                !mlir::isa<VPUIP::SkipDMAOp>(firstTaskOp.getInnerTaskOp()),
                                "Unexpected SkipDMAOp at task index {0}, first task on queue {1}:{2} after sync task. "
                                "SkipDMAOp must be preceded by SyncDMA before this stage",
                                firstTaskIdx, queueType.type, queueType.id);
                        barrierInfo.addConsumer(*updateBarrier.begin(), firstTaskIdx);
                    }
                }
            }

            // Link last task -> current block sync barrier, or handle queue tail in the last block.
            auto blockSyncTaskOpt = barrierInfo.getControlGraphSyncPointForBlock(blockIdx);
            std::optional<size_t> updateBarrierOpt;

            if (!blockSyncTaskOpt.has_value()) {
                // No block sync task in this block, can happen for last block
                continue;
            }
            auto blockSyncTaskIdx = blockSyncTaskOpt.value();
            auto blockSyncTaskQueueType = barrierInfo.getTaskQueueType(blockSyncTaskIdx);
            if (blockSyncTaskQueueType == queueType) {
                // Update barrier not needed
                continue;
            }

            auto waitBarriers = barrierInfo.getWaitBarriers(blockSyncTaskIdx);
            // Possibly the sync task was optimized out and doesn't have wait barrier of it's own
            // In such case use the previous task on the same queue with wait barrier to link to the last task on
            // this queue
            if (waitBarriers.empty()) {
                auto prevTaskOpt = barrierInfo.getPrevTaskOnQueueWithWaitBar(blockSyncTaskIdx, blockSyncTaskQueueType);
                if (prevTaskOpt.has_value()) {
                    waitBarriers = barrierInfo.getWaitBarriers(prevTaskOpt.value());
                }
            }
            VPUX_THROW_WHEN(waitBarriers.empty(),
                            "Expected at least 1 wait barrier for control-block sync task {0} on queue {1}:{2}",
                            blockSyncTaskIdx, blockSyncTaskQueueType.type, blockSyncTaskQueueType.id);
            updateBarrierOpt = *std::max_element(waitBarriers.begin(), waitBarriers.end());

            // Find last on this queue in this block
            auto lastTaskOpt = barrierInfo.getPrevTaskOnQueue(blockSyncTaskIdx, queueType);
            VPUX_THROW_WHEN(!lastTaskOpt.has_value(), "Last task not found before sync task at index {0}",
                            blockSyncTaskIdx);

            barrierInfo.addProducer(updateBarrierOpt.value(), lastTaskOpt.value());
        }
    }
}

void PrepareShaveSubmitDMAsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    // E#213511 Until the kernels support multi DMA engine support, limit the legalization and subsequent passes to only
    // submit SKips for one DMA Engine
    const auto numDmaPorts = numDmaEnginesOpt.hasValue() ? numDmaEnginesOpt.getValue() : 1;

    mlir::OpBuilder builder(netFunc);
    PlannedInsertionsData preparedInsertions;

    // Identify existing position of DeclareBufferOp, will be used as insertion point
    // for new tasks that will be inserted in IR
    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    preparedInsertions.bufferInsertionPoint =
            !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

    auto barrierOps = netFunc.getOps<VPURT::DeclareVirtualBarrierOp>();
    preparedInsertions.barrierInsertionPoint =
            !barrierOps.empty() ? *barrierOps.begin() : &netFunc.getBody().front().front();

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    // Build task queue type map for all queues in order to test paths between tasks on different FIFOs.
    barrierInfo.buildTaskQueueTypeMap();

    // Store information about SHV tasks which can submit DMA ops
    SmallVector<size_t> shvTasksWithDma;
    SmallVector<VPURT::TaskOp> tasksToVerify;
    std::set<size_t> impactedControlBlockIndices;
    findShvTasksWithDmaAndImpactedControlBlocks(shvTasksWithDma, impactedControlBlockIndices, barrierInfo, _log);
    if (shvTasksWithDma.empty()) {
        _log.trace("No SHV TaskOps with DMAs found, skipping split Fetch/Release legalization");
        barrierInfo.clearAttributes();
        return;
    }
    planLegalization(barrierInfo, shvTasksWithDma, preparedInsertions, numDmaPorts, tasksToVerify, _log);
    realizePlannedInsertions(builder, barrierInfo, preparedInsertions, tasksToVerify, _log);
    barrierInfo.updateIR();
    barrierInfo = BarrierInfo{netFunc};
    barrierInfo.buildTaskQueueTypeMap();
    legalizeDependenciesToControlBlockBoundaries(barrierInfo, impactedControlBlockIndices, numDmaPorts, _log);
    VPURT::orderExecutionTasksAndBarriers(netFunc, barrierInfo, _log);

    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(netFunc);
}
}  // namespace

//
// createLegalizeShaveSubmitDMAsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createPrepareShaveSubmitDMAsPass(Logger log) {
    return std::make_unique<PrepareShaveSubmitDMAsPass>(log);
}
