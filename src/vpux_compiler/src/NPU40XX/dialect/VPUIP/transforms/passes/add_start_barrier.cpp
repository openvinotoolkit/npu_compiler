//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/logging.hpp"

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_ADDSTARTBARRIER
#define GEN_PASS_DEF_ADDSTARTBARRIER
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;

namespace {

std::optional<uint32_t> getFirstNonFetchDMAIdx(ArrayRef<uint32_t> dmaTaskIndices, const BarrierInfo& barrierInfo) {
    for (auto dmaIdx : dmaTaskIndices) {
        auto taskOp = barrierInfo.getTaskOpAtIndex(dmaIdx);
        auto innerOp = taskOp.getInnerTaskOp();

        if (!mlir::isa<VPUIP::FetchDMAOp>(innerOp)) {
            return dmaIdx;
        }
    }
    return std::nullopt;
}

std::pair<VPURT::TaskOp, VPURT::DeclareVirtualBarrierOp> getFirstDmaAndStartBarrierCandidate(
        BarrierInfo& barrierInfo, VPURT::TaskOpQueues& taskQueueTypeMap, Logger log) {
    VPURT::TaskOp firstDmaOp;
    VPURT::DeclareVirtualBarrierOp startBarrierCandidateOp;

    // Check all queues and find a start barrier. Following conditions need to be met:
    //
    // 1. Start barrier is produced by DMA P0 Channel DDR to allow insertion of WLM Fetch DMAs
    //    before it. Fetch DMAs are currently inserted only on DMA P0.
    //
    // 2. Start barrier can be consumed only by DMA tasks, as those task do not require descriptor
    //    fetching and start barrier will be used as an earliest point for DPU/SHV enqueues
    //
    // 3. Start barrier consumption cannot depend on DPU/SHV tasks execution as those tasks would be enqueued
    //    at earliest at start barrier
    //
    // 4. [Optional] In case of compiler barrier programming, start barrier can't be produced by any other tasks than
    // firstDMA (DMA P0 Channel DDR). It's needed for avoid race condition between service tasks from DMA P0 Channel DDR
    // and DMAs from other lists.

    const VPURT::TaskQueueType dmaP0ChDddrQueueType = {VPU::ExecutorKind::DMA_NN,
                                                       getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR)};

    // 1. Find a first DMA on P0 CH:DDR which will be used
    // as a DMA to consume a start barrier if such needs to be created
    // when no start barrier candidate is found in IR
    // Find also first DMA on that FIFO that updates a barrier.
    // This DMA is the candidate to produce a start barrier

    const auto& dmaTaskIndices = taskQueueTypeMap[dmaP0ChDddrQueueType];
    auto firstP0ChDdrDmaIdx = getFirstNonFetchDMAIdx(dmaTaskIndices, barrierInfo);

    // If no DMA was found then it needs to be created, do early return
    if (!firstP0ChDdrDmaIdx.has_value()) {
        log.trace("No first DMA found");
        return std::make_pair(nullptr, nullptr);
    }

    log.trace("First DMA candidate index - {0}", firstP0ChDdrDmaIdx.value());
    firstDmaOp = barrierInfo.getTaskOpAtIndex(firstP0ChDdrDmaIdx.value());

    auto firstP0ChDdrDmaToUpdateBarrierIdxIt =
            llvm::find_if(taskQueueTypeMap[dmaP0ChDddrQueueType], [&](size_t taskIdx) {
                if (barrierInfo.getUpdateBarriers(taskIdx).empty()) {
                    return false;
                }
                return true;
            });

    std::optional<uint32_t> firstP0ChDdrDmaToUpdateBarrierIdx =
            (firstP0ChDdrDmaToUpdateBarrierIdxIt != taskQueueTypeMap[dmaP0ChDddrQueueType].end()
                     ? std::make_optional(*firstP0ChDdrDmaToUpdateBarrierIdxIt)
                     : std::nullopt);

    // If there is no DMA on P0 CH:DDR that updates any barrier, then
    // start barrier needs to be created, return early
    if (!firstP0ChDdrDmaToUpdateBarrierIdx.has_value()) {
        log.trace("First DMA to update barriers not found");
        return std::make_pair(firstDmaOp, nullptr);
    }

    auto startBarrierCandidatesVec =
            to_small_vector(barrierInfo.getUpdateBarriers(firstP0ChDdrDmaToUpdateBarrierIdx.value()));

    log.trace("Initial start barrier candidates - {0}", startBarrierCandidatesVec);

    // 2. Remove candidates which are consumed by non-DMA tasks
    startBarrierCandidatesVec.erase(
            llvm::remove_if(startBarrierCandidatesVec,
                            [&](size_t barrierIdx) {
                                auto barrierConsumedByNonDmaTask = false;
                                for (auto barrierConsumerIdx : barrierInfo.getBarrierConsumers(barrierIdx)) {
                                    auto barrierConsumerOp = barrierInfo.getTaskOpAtIndex(barrierConsumerIdx);
                                    if (barrierConsumerOp.getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
                                        barrierConsumedByNonDmaTask = true;
                                        break;
                                    }
                                }
                                if (barrierConsumedByNonDmaTask) {
                                    log.trace("Start barrier candidate {0} consumed by non DMA task", barrierIdx);
                                }
                                return barrierConsumedByNonDmaTask;
                            }),
            startBarrierCandidatesVec.end());

    if (startBarrierCandidatesVec.empty()) {
        log.trace("No start barrier candidates left");
        return std::make_pair(firstDmaOp, nullptr);
    }

    // Build dependency data for first block. No need to analyze other blocks as blocks with N > 0
    // are guaranteed to be dependant on tasks from block index 0
    auto taskControlMapAndOffset = barrierInfo.buildTaskControlMap(0);

    // 3. Check if start barrier candidate does not depend in any way on
    // non DMA tasks (DPU/SHV). If such dependency exists then this is not a start barrier
    for (auto& [queueType, taskVec] : taskQueueTypeMap) {
        if (queueType.type == VPU::ExecutorKind::DMA_NN || taskVec.empty()) {
            continue;
        }

        auto firstTaskOnQueueIdx = taskVec[0];

        // If given task is in next control block then it depends on start barrier
        // No need to check dependency
        if (barrierInfo.getControlGraphBlockIndex(firstTaskOnQueueIdx) > 0) {
            continue;
        }

        // Check if a consumer of start barrier candidate depends on first operation from non-DMA queue
        // In that case start barrier consumption depenends on non DMA task
        // This would be a deadlock as such task would be enqueued
        // at earliest on this start barrier
        startBarrierCandidatesVec.erase(
                llvm::remove_if(startBarrierCandidatesVec,
                                [&](size_t barrierIdx) {
                                    for (auto barrierConsumerIdx : barrierInfo.getBarrierConsumers(barrierIdx)) {
                                        if (barrierInfo.getControlGraphBlockIndex(barrierConsumerIdx) > 0) {
                                            continue;
                                        }

                                        auto dependencyExistsFromNonDmaTaskToBarrierCandidateConsumer =
                                                barrierInfo.controlPathExistsBetweenTasksInSameBlock(
                                                        taskControlMapAndOffset.first,
                                                        firstTaskOnQueueIdx - taskControlMapAndOffset.second,
                                                        barrierConsumerIdx - taskControlMapAndOffset.second, false);
                                        if (dependencyExistsFromNonDmaTaskToBarrierCandidateConsumer) {
                                            // If given barrier candidate depends on non DMA task then
                                            // it cannot be treated as start barrier
                                            log.trace("Start barrier candidate {0} depends on non DMA task",
                                                      barrierIdx);
                                            return true;
                                        }
                                    }
                                    return false;
                                }),
                startBarrierCandidatesVec.end());
    }

    // 4. Remove candidates which are produced by any other tasks than firstDMA. Special case which needed only for
    // compiler barrier programming for avoid race condition between DMA tasks
    startBarrierCandidatesVec.erase(
            llvm::remove_if(startBarrierCandidatesVec,
                            [&](size_t barrierIdx) {
                                auto barrierUpdatedNonFirstDmaTask = false;
                                for (auto barrierProducerIdx : barrierInfo.getBarrierProducers(barrierIdx)) {
                                    auto barrierProducerOp = barrierInfo.getTaskOpAtIndex(barrierProducerIdx);
                                    if (barrierProducerOp != firstDmaOp) {
                                        barrierUpdatedNonFirstDmaTask = true;
                                        break;
                                    }
                                }
                                return barrierUpdatedNonFirstDmaTask;
                            }),
            startBarrierCandidatesVec.end());

    if (startBarrierCandidatesVec.empty()) {
        log.trace("No start barrier candidates left");
        return std::make_pair(firstDmaOp, nullptr);
    }

    // No candidates left, return
    if (startBarrierCandidatesVec.empty()) {
        log.trace("No start barrier candidates left");
        return std::make_pair(firstDmaOp, nullptr);
    }

    // Pick candidate with smaller index
    auto startBarrierCandidateOpInd =
            *std::min_element(std::begin(startBarrierCandidatesVec), std::end(startBarrierCandidatesVec));

    log.trace("Found start DMA task index {0} and start barrier index {1}", firstP0ChDdrDmaToUpdateBarrierIdx.value(),
              startBarrierCandidateOpInd);

    // Valid first dma and start barrier were found
    startBarrierCandidateOp =
            mlir::cast<VPURT::DeclareVirtualBarrierOp>(barrierInfo.getBarrierOpAtIndex(startBarrierCandidateOpInd));
    firstDmaOp = barrierInfo.getTaskOpAtIndex(firstP0ChDdrDmaToUpdateBarrierIdx.value());

    return std::make_pair(firstDmaOp, startBarrierCandidateOp);
}

// In case of compiler barrier programming, we need to add explicit guard for all parallel DMA engines. It allow us to
// finish all necessary service DMAs like barrier programming, LUT programming and so on. This logic based on assumption
// that no tasks start before their barriers are programmed. Example:
//  DMA P0 CH:DDR ->|
//                  |-> Bar0 -> ...   will transfer to
//  DMA P1 CH:DDR ->|
//
//  SyncDMA P0 CH:DDR -> |                  |-> DMA P0 CH:DDR -> |
//                       |-> StartBarrier ->|                    | -> Bar0 ->
//                                          |-> DMA P1 CH:DDR -> |

// It will allows us insert service DMAs into P0 CH:DDR and guarantee their execution

void addExplicitDependencyBetweenDmaListsAndStartBarrier(mlir::func::FuncOp func, BarrierInfo& barrierInfo,
                                                         VPURT::TaskOpQueues& taskQueueTypeMap,
                                                         VPURT::DeclareVirtualBarrierOp startBarrierOp, Logger log) {
    const auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto dmaPortNum = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();
    auto dmaChannels = getDMAChannelsWithIndependentLinkAgents(config::getArch(module));
    for (auto dmaPortIdx : irange(dmaPortNum)) {
        for (auto dmaChannel : dmaChannels) {
            // We skip queue P0 CH:DDR because is queue where we have DMA that's responsible for handling start barrier.
            // Not needed to set an extra dependency
            if (dmaPortIdx == 0 && dmaChannel == VPUIP::DmaChannelType::DDR) {
                continue;
            }
            const VPURT::TaskQueueType dmaQueueType = {VPU::ExecutorKind::DMA_NN,
                                                       getDMAQueueIdEncoding(/*port*/ dmaPortIdx, dmaChannel)};

            auto firstDmaInSpecificQueue = std::begin(taskQueueTypeMap[dmaQueueType]);
            if (firstDmaInSpecificQueue == taskQueueTypeMap[dmaQueueType].end()) {
                continue;
            }

            auto firstDMAOpFromQueue = barrierInfo.getTaskOpAtIndex(*firstDmaInSpecificQueue);
            auto waitBarriers = firstDMAOpFromQueue.getWaitBarriersMutable();
            if (waitBarriers.empty()) {
                log.trace("Add {0} barrier as a waitBarrier for {1}", startBarrierOp.getBarrier(),
                          firstDMAOpFromQueue->getName());
                waitBarriers.append(startBarrierOp.getBarrier());
            }
        }
    }
}

class AddStartBarrierPass final : public VPUIP::arch40xx::impl::AddStartBarrierBase<AddStartBarrierPass> {
public:
    explicit AddStartBarrierPass(std::optional<WorkloadManagementMode> workloadManagementMode, Logger log)
            : _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    std::optional<WorkloadManagementMode> _workloadManagementMode;
};

void AddStartBarrierPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& barrierInfo = getAnalysis<BarrierInfo>();
    barrierInfo.buildTaskQueueTypeMap();
    auto taskQueueTypeMap = VPURT::getTaskOpQueues(func, barrierInfo);
    auto [firstDmaOp, startBarrierCandidateOp] =
            getFirstDmaAndStartBarrierCandidate(barrierInfo, taskQueueTypeMap, _log);

    if (startBarrierCandidateOp == nullptr) {
        auto insertPoint = &func.getBody().front().front();
        mlir::OpBuilder builder(func);
        builder.setInsertionPoint(insertPoint);
        startBarrierCandidateOp = builder.create<VPURT::DeclareVirtualBarrierOp>(insertPoint->getLoc());
        _log.trace("Add new start barrier {0}", startBarrierCandidateOp->getLoc());
        barrierInfo.addNewBarrier(startBarrierCandidateOp);

        auto buffers = func.getOps<VPURT::DeclareBufferOp>();
        VPUX_THROW_WHEN(buffers.empty(), "Can not find DeclareBufferOp");
        auto firstDeclareBufferOp = *buffers.begin();
        auto inBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp);
        auto outBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp);

        if (firstDmaOp == nullptr || !firstDmaOp.getWaitBarriers().empty()) {
            // Create a SyncDMA op as the first DMA
            auto taskOps = func.getOps<VPURT::TaskOp>();
            VPUX_THROW_WHEN(taskOps.empty(), "Can not find TaskOp");
            auto firstTaskOp = *taskOps.begin();
            _log.trace("Add Sync DMA that will consume start barrier");
            builder.setInsertionPoint(firstTaskOp);
            firstDmaOp = VPUIP::createSyncDMA(builder, inBuffer, outBuffer, 0, {}, {});
        }

        _log.trace("Add Sync DMA that will update start barrier consumed by DMA {0}", firstDmaOp->getLoc());
        builder.setInsertionPoint(firstDmaOp);
        VPUIP::createSyncDMA(builder, inBuffer, outBuffer, 0, {}, {startBarrierCandidateOp.getBarrier()},
                             "start_barrier_sync_dma");
        firstDmaOp.getWaitBarriersMutable().append(startBarrierCandidateOp.getBarrier());
    }

    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(&getContext(), "start_barrier"));
    startBarrierCandidateOp->setLoc(loc);
    startBarrierCandidateOp.setIsStartBarrier(true);
    addExplicitDependencyBetweenDmaListsAndStartBarrier(func, barrierInfo, taskQueueTypeMap, startBarrierCandidateOp,
                                                        _log);

    barrierInfo.clearAttributes();

    if (!_workloadManagementMode.has_value() || _workloadManagementMode <= WorkloadManagementMode::PWLM_V2_PAGES) {
        VPURT::verifyBarrierSlots(func, _log);
    }
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createAddStartBarrierPass(
        std::optional<WorkloadManagementMode> workloadManagementMode, Logger log) {
    return std::make_unique<AddStartBarrierPass>(workloadManagementMode, log);
}
