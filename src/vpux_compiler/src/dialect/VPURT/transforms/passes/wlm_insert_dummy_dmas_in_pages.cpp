//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_WLMINSERTDUMMYDMASINPAGES
#define GEN_PASS_DEF_WLMINSERTDUMMYDMASINPAGES
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class WlmInsertDummyDmasInPagesPass final :
        public VPURT::impl::WlmInsertDummyDmasInPagesBase<WlmInsertDummyDmasInPagesPass> {
public:
    explicit WlmInsertDummyDmasInPagesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void WlmInsertDummyDmasInPagesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto arch = config::getArch(func);

    if (config::getWorkloadManagementStatus(module) != WorkloadManagementStatus::ENABLED) {
        // WLM is not supported, no need to run this pass
        return;
    }

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    // Create dummy input and output buffer that will be needed for creating dummy DMAs
    mlir::OpBuilder builder(func);
    auto buffers = func.getOps<VPURT::DeclareBufferOp>();
    VPUX_THROW_WHEN(buffers.empty(), "Cannot find DeclareBufferOp");
    auto firstDeclareBufferOp = *buffers.begin();
    auto inDdrBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp, VPU::MemoryKind::DDR);
    auto inCmxBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp, VPU::MemoryKind::CMX_NN);
    auto outBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp);

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandler(func, barrierInfo, numBarriers, _log);
    barrierPagesSplitHandler.initializeForLegalization();

    // Before performing main logic for dummy DMA insertion, first identify pages which have only 1 barrier
    // It is a prerequisite for rest of the logic to be able to correctly perform legalization to have
    // at least two barrier in the page.
    // In this initial step for such pages, created a dummy barrier and dummy DMA and insert them in page
    // Before:
    //  Task1(PageN-1) -> Bar(PageN) -> Task2(PageN)
    // After:
    //  Task1(PageN-1) -> Bar(PageN) -> DummyDma(PageN) -> DummyBar(PageN) -> Task2(PageN)
    auto barrierOfSingleBarrierPagesVec = barrierPagesSplitHandler.getBarrierOfSingleBarrierPages();
    if (!barrierOfSingleBarrierPagesVec.empty()) {
        _log.trace("Number of pages with single barrier: {0}", barrierOfSingleBarrierPagesVec.size());
        for (auto barInd : barrierOfSingleBarrierPagesVec) {
            auto barOp = barrierInfo.getBarrierOpAtIndex(barInd);
            auto pageInd = barOp.getWlmPage().value();
            _log.nest().trace("Single barrier {0} in page {1}", barInd, pageInd);

            auto barProducers = barrierInfo.getBarrierProducers(barInd);
            auto barConsumers = barrierInfo.getBarrierConsumers(barInd);

            // Create new barrier that will be consumed by original barrier consumers
            builder.setInsertionPointAfter(barOp);
            auto newBarOp = builder.create<VPURT::DeclareVirtualBarrierOp>(barOp->getLoc());
            newBarOp.setWlmPage(pageInd);
            barrierInfo.addNewBarrier(newBarOp);
            auto newBarrierIdx = barrierInfo.getIndex(newBarOp);
            barrierInfo.addConsumers(newBarrierIdx, barConsumers);

            // Create dummy DMA Op that will consume original barrier and produce into new barrier
            // TODO: Identify what are the users of barrier and if it is already a DMA, create a dummy DMA
            // of different type. This way number of dummy DMAs inserted in subsequent steps will reduce
            builder.setInsertionPointAfter(
                    barrierInfo.getTaskOpAtIndex(*std::max_element(barProducers.begin(), barProducers.end())));
            auto dummyDmaOp = VPUIP::createSyncDMA(builder, inDdrBuffer, outBuffer, 0, {}, {},
                                                   "dummy_dma_increasing_bar_count_for_wlm_page");
            dummyDmaOp.setWlmPage(pageInd);
            auto dummyDmaOpTaskInd = barrierInfo.addNewTaskOp(dummyDmaOp);

            barrierInfo.addProducer(newBarrierIdx, dummyDmaOpTaskInd);

            barrierInfo.removeConsumers(barInd, barConsumers);
            barrierInfo.addConsumer(barInd, dummyDmaOpTaskInd);
        }

        VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);
        barrierPagesSplitHandler =
                VPURT::BarrierPagesSplitHandler{func, barrierInfo, static_cast<size_t>(numBarriers), _log};
        barrierPagesSplitHandler.initializeForLegalization();
    }

    auto dummyDmaInsertionDataVec = barrierPagesSplitHandler.getAndLegalizeDummyDmaInsertionData();

    if (dummyDmaInsertionDataVec.empty()) {
        return;
    }

    _log.trace("Insert {0} dummy DMAs in pages", dummyDmaInsertionDataVec.size());

    // Retrieving dummy DMA data may introduce new dependencies which need to be updated in IR
    // Recreate barrier info to have latest status of the IR
    barrierInfo = barrierPagesSplitHandler.getUpdatedBarrierInfo();
    barrierInfo.updateIR();

    for (auto dummyDmaInsertionData : dummyDmaInsertionDataVec) {
        auto pageInd = dummyDmaInsertionData.pageInd;
        auto queueType = dummyDmaInsertionData.queueType;
        auto insertionPointOp = barrierInfo.getTaskOpAtIndex(dummyDmaInsertionData.insertAfter);
        auto port = getDMAPortFromEncodedId(queueType.id);

        _log.trace("Insert new DMA[{0}][{1}] in page {2}", port, getDMAChannelTypeAsString(queueType.id, arch),
                   pageInd);

        builder.setInsertionPointAfter(insertionPointOp);

        auto inBuffer = (getDMAChannelTypeFromEncodedId(queueType.id, arch) == VPUIP::DmaChannelType::DDR)
                                ? inDdrBuffer
                                : inCmxBuffer;
        auto dummyDmaOp = VPUIP::createSyncDMA(builder, inBuffer, outBuffer, port, {}, {},
                                               "dummy_dma_fifo_completing_for_wlm_page");
        dummyDmaOp.setWlmPage(pageInd);

        auto dummyDmaOpTaskInd = barrierInfo.addNewTaskOp(dummyDmaOp);

        llvm::for_each(dummyDmaInsertionData.waitBars, [&](auto waitBar) {
            _log.nest().trace("wait bar {0}", waitBar);
            barrierInfo.addConsumer(waitBar, dummyDmaOpTaskInd);
        });

        llvm::for_each(dummyDmaInsertionData.updateBars, [&](auto updateBar) {
            _log.nest().trace("update bar {0}", updateBar);
            barrierInfo.addProducer(updateBar, dummyDmaOpTaskInd);
        });
    }

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);
    barrierPagesSplitHandler =
            VPURT::BarrierPagesSplitHandler{func, barrierInfo, static_cast<size_t>(numBarriers), _log};
    barrierPagesSplitHandler.initializeForLegalization();

    // In case during previous legalization there are now some last tasks on FIFO in page without update
    // barrier, then find update barriers for them.
    // TODO: Review the need for it as part of E#177151 activity
    auto lastTaskTypePerPageWithNoUpdBar = barrierPagesSplitHandler.getLastTasksOnFifoPerPageWithNoUpdBar();
    if (!lastTaskTypePerPageWithNoUpdBar.empty()) {
        _log.trace("Legalize last task on page with no update barrier");
        barrierPagesSplitHandler.addUpdateBarriersForLastTaskOnFifoInPage(lastTaskTypePerPageWithNoUpdBar);
        // After legalization get updated BarrierInfo, update IR and recreate BarrierPagesSplitHandler to have up to
        // date state
        barrierInfo = barrierPagesSplitHandler.getUpdatedBarrierInfo();
        VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);
        barrierPagesSplitHandler =
                VPURT::BarrierPagesSplitHandler{func, barrierInfo, static_cast<size_t>(numBarriers), _log};
        barrierPagesSplitHandler.initializeForLegalization();
    }

    // Perform final checks after legalization
    barrierPagesSplitHandler.verifyTaskBarrierPagesAreValid();
    barrierPagesSplitHandler.verifyNoCyclicDeps();
    VPUX_THROW_UNLESS(barrierPagesSplitHandler.isSplitToPagesValid(), "Split to pages is not valid");
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");

    barrierInfo.clearAttributes();
}
}  // namespace

//
// createWlmInsertDummyDmasInPagesPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createWlmInsertDummyDmasInPagesPass(Logger log) {
    return std::make_unique<WlmInsertDummyDmasInPagesPass>(log);
}
