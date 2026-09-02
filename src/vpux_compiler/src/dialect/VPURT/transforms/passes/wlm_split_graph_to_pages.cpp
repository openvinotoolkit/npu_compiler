//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/shave.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_WLMSPLITGRAPHTOPAGES
#define GEN_PASS_DEF_WLMSPLITGRAPHTOPAGES
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class WlmSplitGraphToPagesPass final : public VPURT::impl::WlmSplitGraphToPagesBase<WlmSplitGraphToPagesPass> {
public:
    explicit WlmSplitGraphToPagesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    void createDummyBuffersIfNeeded(mlir::OpBuilder& builder, mlir::Value& dummyInBuffer, mlir::Value& dummyOutBuffer,
                                    mlir::func::FuncOp func);
    bool scheduleLegalizeForShvWithDpu(BarrierInfo& barrierInfo, mlir::OpBuilder& builder, mlir::func::FuncOp func,
                                       mlir::Value& dummyInBuffer, mlir::Value& dummyOutBuffer, size_t numClusters);
};

void WlmSplitGraphToPagesPass::createDummyBuffersIfNeeded(mlir::OpBuilder& builder, mlir::Value& dummyInBuffer,
                                                          mlir::Value& dummyOutBuffer, mlir::func::FuncOp func) {
    if (dummyInBuffer && dummyOutBuffer) {
        return;
    }

    auto buffers = func.getOps<VPURT::DeclareBufferOp>();
    VPUX_THROW_WHEN(buffers.empty(), "Cannot find DeclareBufferOp");
    auto firstDeclareBufferOp = *buffers.begin();
    dummyInBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp, VPU::MemoryKind::DDR);
    dummyOutBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp);
}

// Check if model has pattern SHV(withDPU) -> BAR -> DPU (sync) and if yes then insert dummy DMA and barrier in between
// SHV and sync DPU task so that later WLM enqueue pass can insert enqueue DMA without breaking control block split
// restrictions: SHV(withDPU) -> BAR -> dummyDMA -> newBar -> DPU (sync)
bool WlmSplitGraphToPagesPass::scheduleLegalizeForShvWithDpu(BarrierInfo& barrierInfo, mlir::OpBuilder& builder,
                                                             mlir::func::FuncOp func, mlir::Value& dummyInBuffer,
                                                             mlir::Value& dummyOutBuffer, size_t numClusters) {
    bool newBarsInserted = false;
    for (size_t blockInd = 0; blockInd < barrierInfo.getControlGraphBlockCount(); blockInd++) {
        auto syncTaskOpt = barrierInfo.getControlGraphSyncPointForBlock(blockInd);
        if (!syncTaskOpt.has_value()) {
            break;
        }
        auto syncTaskInd = syncTaskOpt.value();
        auto syncTaskOp = barrierInfo.getTaskOpAtIndex(syncTaskInd);
        auto syncTaskQueueType = VPURT::getTaskQueueType(syncTaskOp, false);
        if (syncTaskQueueType.type != config::ExecutorKind::DPU) {
            continue;
        }

        // Identify syncTask which is a DPU task, check if its wait barrier is updated by SHV with DPU
        auto waitBarriers = barrierInfo.getWaitBarriers(syncTaskInd);
        if (waitBarriers.empty()) {
            auto prevTaskOpt = barrierInfo.getPrevTaskOnQueueWithWaitBar(syncTaskInd, syncTaskQueueType);
            if (prevTaskOpt.has_value()) {
                waitBarriers = barrierInfo.getWaitBarriers(prevTaskOpt.value());
            }
        }
        VPUX_THROW_WHEN(waitBarriers.empty(), "Expected at least 1 wait barrier for sync task {0} which is a DPU task",
                        syncTaskInd);

        auto latestWaitBar = *std::max_element(waitBarriers.begin(), waitBarriers.end());

        // Check if this barrier is produced by SHV with DPU task
        auto producerTasks = barrierInfo.getBarrierProducers(latestWaitBar);
        bool needToInsertNewBar = false;
        for (auto producerTaskInd : producerTasks) {
            if (!isDpuShaveKernelType(barrierInfo.getTaskOpAtIndex(producerTaskInd))) {
                continue;
            }

            auto shvTaskQueueType = barrierInfo.getTaskQueueType(producerTaskInd);
            auto tileIndex = vpux::getShaveTileIndexFromEncodedId(shvTaskQueueType.id, numClusters);

            // If DPU is on different tile, then can be ignored
            if (tileIndex != syncTaskQueueType.id) {
                continue;
            }

            // If its on same tile then we identified following sequence
            // SHVwithDPU (producer) -> wait barrier -> syncDPUtask (consumer)
            // This case needs to be legalized
            needToInsertNewBar = true;
            break;
        }

        if (!needToInsertNewBar) {
            continue;
        }
        _log.trace("Identified SHV with DPU task producing barrier {0} which is consumed by DPU sync task {1}. Insert "
                   "dummy DMA and barrier in between to legalize scheduling",
                   latestWaitBar, syncTaskInd);

        // Legalize the following pattern:
        //
        // SHVwithDPU -> BAR_A -> syncDPUtask
        //        ... -> ...      /
        //        ... -> BAR_B ->/
        // Into:
        // SHVwithDPU -> BAR_A -> dummyDMA -> newBar -> syncDPUtask
        //        ... -> ...      /
        //        ... -> BAR_B ->/
        createDummyBuffersIfNeeded(builder, dummyInBuffer, dummyOutBuffer, func);

        builder.setInsertionPoint(syncTaskOp);

        auto dummyDmaOp = VPUIP::createSyncDMA(builder, dummyInBuffer, dummyOutBuffer, 0, {}, {},
                                               "dummy_dma_before_sync_task_for_shv_with_dpu");

        auto dummyDmaOpTaskInd = barrierInfo.addNewTaskOp(dummyDmaOp);

        barrierInfo.setWaitBarriers(dummyDmaOpTaskInd, waitBarriers);

        auto latestWaitBarOp = barrierInfo.getBarrierOpAtIndex(latestWaitBar);

        builder.setInsertionPointAfter(latestWaitBarOp);
        auto newBar = builder.create<VPURT::DeclareVirtualBarrierOp>(syncTaskOp.getLoc());
        auto newBarInd = barrierInfo.addNewBarrier(newBar);

        barrierInfo.addProducer(newBarInd, dummyDmaOpTaskInd);

        BarrierInfo::TaskSet newBarIndSet = {newBarInd};
        barrierInfo.setWaitBarriers(syncTaskInd, newBarIndSet);
        newBarsInserted = true;
    }

    return newBarsInserted;
}

void WlmSplitGraphToPagesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    mlir::OpBuilder builder(func);
    mlir::Value inBuffer;
    mlir::Value outBuffer;

    auto numClusters = config::getTileExecutor(module).getCount();
    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);

    barrierInfo.buildTaskQueueTypeMap();

    // Before splitting graph into pages make sure that in case of SHV with DPU closest production barrier is not
    // consumed by DPU which is also a sync-task. This is needed to prevent cases where after splitting graph into pages
    // and delaying DPU enqueue because of SHV with DPU there is no way of adding dependency from enqueue DMA to
    // sync-task
    bool newBarsInserted = scheduleLegalizeForShvWithDpu(barrierInfo, builder, func, inBuffer, outBuffer, numClusters);
    if (newBarsInserted) {
        VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);
    }

    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandler(func, barrierInfo, numBarriers, _log);
    barrierPagesSplitHandler.initializeForAssignment(func);
    barrierPagesSplitHandler.assignPagesToBarriersInIr();
    barrierPagesSplitHandler.assignPagesToTasksInIr();

    // Identify pages which do not have any tasks
    // All subsequent passes expect at least 1 task in page
    auto dummyDmaDataForPagesWithNoTasks = barrierPagesSplitHandler.getDummyDmaDataForPagesWithNoTasks();
    if (!dummyDmaDataForPagesWithNoTasks.empty()) {
        _log.trace("There are {0} pages with no tasks", dummyDmaDataForPagesWithNoTasks.size());

        createDummyBuffersIfNeeded(builder, inBuffer, outBuffer, func);

        const size_t port = 0;
        const VPURT::TaskQueueType dummyDmaQueueType = {config::ExecutorKind::DMA_NN,
                                                        getDMAQueueIdEncoding(port, VPUIP::DmaChannelType::DDR)};

        DenseMap<size_t, VPURT::TaskOp> newDummyDmaOpsPerPage;
        for (auto dummyDmaDataForPageWithNoTasks : dummyDmaDataForPagesWithNoTasks) {
            auto pageInd = dummyDmaDataForPageWithNoTasks.pageInd;
            auto waitBar = dummyDmaDataForPageWithNoTasks.waitBar;
            auto updateBar = dummyDmaDataForPageWithNoTasks.updateBar;
            auto insertBefore = dummyDmaDataForPageWithNoTasks.insertBefore;

            _log.trace("Page {0} has no tasks assigned. Create dummy task between barriers {1} and {2} before task {3}",
                       pageInd, waitBar, updateBar, insertBefore);

            auto insertionPointOp = barrierInfo.getTaskOpAtIndex(insertBefore);
            builder.setInsertionPoint(insertionPointOp);

            auto dummyDmaOp =
                    VPUIP::createSyncDMA(builder, inBuffer, outBuffer, port, {}, {}, "dummy_dma_page_completing");
            dummyDmaOp.setWlmPage(pageInd);

            auto dummyDmaOpTaskInd = barrierInfo.addNewTaskOp(dummyDmaOp);

            barrierInfo.addConsumer(dummyDmaDataForPageWithNoTasks.waitBar, dummyDmaOpTaskInd);
            barrierInfo.addProducer(dummyDmaDataForPageWithNoTasks.updateBar, dummyDmaOpTaskInd);

            newDummyDmaOpsPerPage[pageInd] = dummyDmaOp;
        }

        // After all the dummyDMAs in empty pages were inserted update task page assignment
        // This is done as separate step to prevent cases where task which is used as insertion
        // point by multiple dummyDMAs gets shifted after insertion of first one. Such case
        // can happen when multiple consecutive pages are empty.
        for (auto dummyDmaDataForPageWithNoTasks : dummyDmaDataForPagesWithNoTasks) {
            auto pageInd = dummyDmaDataForPageWithNoTasks.pageInd;
            auto insertBefore = dummyDmaDataForPageWithNoTasks.insertBefore;

            barrierPagesSplitHandler.updateTaskPageAssignmentForQueue(insertBefore, pageInd, dummyDmaQueueType,
                                                                      newDummyDmaOpsPerPage[pageInd]);
        }

        VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);
    }

    // In case of SHV tasks make sure that SHV tasks on same tile do not have decreasing page number
    // for consecutive tasks in IR. This is already guaranteed if there are dedicated SHV FIFOs, but
    // in case its not then dedicated handling is needed.
    if (!config::isFifoPerShaveEngineEnabled(func)) {
        _log.trace("Separate SHV FIFOs not supported - check SHV tasks on same cluster never have decreasing pages");
        VPURT::BarrierPagesSplitHandler barrierPagesSplitHandler(func, barrierInfo, numBarriers, _log);
        barrierPagesSplitHandler.initializeForLegalization(/* allowDepsChange */ true);
        barrierPagesSplitHandler.initializeTaskQueueTypeMap(func);
        barrierPagesSplitHandler.updateTaskPageAssignmentForShvInCaseOfNoDedicatedShvFifos();
        barrierInfo = barrierPagesSplitHandler.getUpdatedBarrierInfo();
        VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);
    }

    barrierInfo.clearAttributes();
}
}  // namespace

//
// createWlmSplitGraphToPagesPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createWlmSplitGraphToPagesPass(Logger log) {
    return std::make_unique<WlmSplitGraphToPagesPass>(log);
}
