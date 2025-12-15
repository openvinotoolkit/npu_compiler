//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/options.hpp"

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
};

void WlmSplitGraphToPagesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    if (config::getWorkloadManagementStatus(module) != WorkloadManagementStatus::ENABLED) {
        // WLM is not supported, no need to run this pass
        return;
    }

    if (!VPURT::verifyOneWaitBarrierPerTask(func, _log)) {
        _log.warning("WLM cannot be enabled as not all tasks have 1 wait barrier");
        config::setWorkloadManagementStatus(module, WorkloadManagementStatus::FAILED);
        return;
    }

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);

    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandler(func, barrierInfo, numBarriers, _log);
    barrierPagesSplitHandler.initializeForAssignment(func);
    barrierPagesSplitHandler.assignPagesToBarriersInIr();
    barrierPagesSplitHandler.assignPagesToTasksInIr();

    // Identify pages which do not have any tasks
    // All subsequent passes expect at least 1 task in page
    auto dummyDmaDataForPagesWithNoTasks = barrierPagesSplitHandler.getDummyDmaDataForPagesWithNoTasks();
    if (!dummyDmaDataForPagesWithNoTasks.empty()) {
        _log.trace("There are {0} pages with no tasks", dummyDmaDataForPagesWithNoTasks.size());

        mlir::OpBuilder builder(func);
        auto buffers = func.getOps<VPURT::DeclareBufferOp>();
        VPUX_THROW_WHEN(buffers.empty(), "Cannot find DeclareBufferOp");
        auto firstDeclareBufferOp = *buffers.begin();
        auto inBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp, VPU::MemoryKind::DDR);
        auto outBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp);

        const size_t port = 0;
        const VPURT::TaskQueueType dummyDmaQueueType = {VPU::ExecutorKind::DMA_NN,
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
    barrierInfo.clearAttributes();
}
}  // namespace

//
// createWlmSplitGraphToPagesPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createWlmSplitGraphToPagesPass(Logger log) {
    return std::make_unique<WlmSplitGraphToPagesPass>(log);
}
