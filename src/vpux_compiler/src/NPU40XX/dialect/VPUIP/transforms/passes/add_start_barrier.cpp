//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <algorithm>
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/logging.hpp"

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_ADDSTARTBARRIER
#define GEN_PASS_DEF_ADDSTARTBARRIER
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;

namespace {

std::pair<VPURT::TaskOp, VPURT::DeclareVirtualBarrierOp> getFirstDmaAndStartBarrierCandidate(mlir::func::FuncOp funcOp,
                                                                                             BarrierInfo& barrierInfo,
                                                                                             Logger log) {
    auto taskQueueTypeMap = VPURT::getTaskOpQueues(funcOp, barrierInfo);

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

    const VPURT::TaskQueueType dmaP0ChDddrQueueType = {VPU::ExecutorKind::DMA_NN,
                                                       getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR)};

    // 1. Find a first DMA on P0 CH:DDR which will be used
    // as a DMA to consume a start barrier if such needs to be created
    // when no start barrier candidate is found in IR
    // Find also first DMA on that FIFO that updates a barrier.
    // This DMA is the candidate to produce a start barrier

    auto firstP0ChDdrDmaIdxIt = std::begin(taskQueueTypeMap[dmaP0ChDddrQueueType]);

    std::optional<uint32_t> firstP0ChDdrDmaIdx = (firstP0ChDdrDmaIdxIt != taskQueueTypeMap[dmaP0ChDddrQueueType].end()
                                                          ? std::make_optional(*firstP0ChDdrDmaIdxIt)
                                                          : std::nullopt);

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

VPURT::TaskOp createSyncDMA(mlir::OpBuilder& builder, mlir::Operation* insertPoint, mlir::Value input,
                            mlir::Value outputBuf, mlir::ValueRange updateBarriers) {
    auto ctx = builder.getContext();
    auto syncDmaLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "sync_dma"));
    auto zeroAttr = vpux::getIntAttr(ctx, 0);

    VPUX_THROW_WHEN(insertPoint == nullptr, "Invaild insert point");
    builder.setInsertionPoint(insertPoint);
    auto syncDMATask = VPURT::wrapIntoTaskOp<VPUIP::SyncDMAOp>(
            builder, {}, updateBarriers, syncDmaLoc, input, outputBuf, /*port*/ zeroAttr,
            /*isOutOfOrder*/ nullptr, /*isCritical*/ nullptr, /*dmaHwpId*/ nullptr,
            /*dmaProfilingMetaData*/ nullptr);
    return syncDMATask->getParentOfType<VPURT::TaskOp>();
}

class AddStartBarrierPass final : public VPUIP::arch40xx::impl::AddStartBarrierBase<AddStartBarrierPass> {
public:
    explicit AddStartBarrierPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddStartBarrierPass::safeRunOnFunc() {
    auto func = getOperation();

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    barrierInfo.buildTaskQueueTypeMap();
    auto [firstDmaOp, startBarrierCandidateOp] = getFirstDmaAndStartBarrierCandidate(func, barrierInfo, _log);
    barrierInfo.clearAttributes();

    if (startBarrierCandidateOp != nullptr) {
        auto loc = mlir::NameLoc::get(mlir::StringAttr::get(&getContext(), "start_barrier"));
        startBarrierCandidateOp->setLoc(loc);
        startBarrierCandidateOp.setIsStartBarrier(true);
        return;
    }

    // Need create new barrer and sync dma task
    auto insertPoint = &func.getBody().front().front();
    mlir::OpBuilder builder(func);
    builder.setInsertionPoint(insertPoint);
    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(&getContext(), "start_barrier"));
    auto barrierOp = builder.create<VPURT::DeclareVirtualBarrierOp>(loc);
    barrierOp.setIsStartBarrier(true);
    _log.trace("Add new start barrier {0}", barrierOp->getLoc());

    // Create dummy input and output buffer
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
        firstDmaOp = createSyncDMA(builder, firstTaskOp, inBuffer, outBuffer, {});
    }

    _log.trace("Add Sync DMA that will update start barrier consumed by DMA {0}", firstDmaOp->getLoc());
    createSyncDMA(builder, firstDmaOp, inBuffer, outBuffer, {barrierOp.getBarrier()});
    firstDmaOp.getWaitBarriersMutable().append(barrierOp.getBarrier());

    VPURT::verifyBarrierSlots(func, _log);
}
}  // namespace

//
// createAddStartBarrierPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createAddStartBarrierPass(Logger log) {
    return std::make_unique<AddStartBarrierPass>(log);
}
