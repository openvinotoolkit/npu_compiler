//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_FINDWLMENQUEUEDMASBARRIER
#define GEN_PASS_DEF_FINDWLMENQUEUEDMASBARRIER
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class FindWlmEnqueueDmasBarrierPass final :
        public VPURT::impl::FindWlmEnqueueDmasBarrierBase<FindWlmEnqueueDmasBarrierPass> {
public:
    explicit FindWlmEnqueueDmasBarrierPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FindWlmEnqueueDmasBarrierPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    auto& barrierInfo = getAnalysis<BarrierInfo>();

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);

    barrierInfo.buildTaskQueueTypeMap();

    // If no compute tasks in the model then there is no need for enqueue DMAs
    // DMA tasks are all enqueued at bootstrap
    if (barrierInfo.getNumOfTasks(config::ExecutorKind::DMA_NN) == barrierInfo.getNumOfTasks()) {
        barrierInfo.clearAttributes();
        return;
    }

    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandler(func, barrierInfo, numBarriers, _log);
    barrierPagesSplitHandler.initializeForEnqueue(func);

    mlir::DenseSet<vpux::config::ExecutorKind> executorEnqAtBootstrap{vpux::config::ExecutorKind::DMA_NN};

    // Get Execution Groups on each queue
    auto& execGroupAnalysis = getAnalysis<ExecutionGroupAnalysis>();
    auto enqueueDmaDataVec = barrierPagesSplitHandler.getEnqueueDmaData(execGroupAnalysis, executorEnqAtBootstrap);

    VPUX_THROW_WHEN(enqueueDmaDataVec.empty(), "No enqueue DMA data created");

    const auto arch = config::getArch(module);
    auto numClusters = config::getTileExecutor(module).getCount();

    // Create dummy input and output buffer that will be needed for creating dummy DMAs
    mlir::OpBuilder builder(func);
    auto ctx = builder.getContext();
    auto buffers = func.getOps<VPURT::DeclareBufferOp>();
    VPUX_THROW_WHEN(buffers.empty(), "Cannot find DeclareBufferOp");
    auto firstDeclareBufferOp = *buffers.begin();
    auto inBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp, VPU::MemoryKind::DDR);
    auto outBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp);

    const VPURT::TaskQueueType enqueueDmaQueueType = {config::ExecutorKind::DMA_NN,
                                                      getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR)};

    mlir::DenseSet<size_t> barriersConsumedByEnqueueDmas;
    // Process provided enqueue DMA data.
    // enqueueDmaDataVec order of enqueue DMAs follows the order of tasks on same HW FIFO
    // It is important to keep this order when inserting those ops in the IR, especially
    // when insertion point is the same for consecutive enqueue DMAs
    for (const auto& enqueueDmaData : enqueueDmaDataVec) {
        auto pageInd = enqueueDmaData.pageInd;
        auto queueType = enqueueDmaData.queueType;
        auto waitBars = enqueueDmaData.waitBars;
        auto startTaskIdx = enqueueDmaData.startTaskIdx;
        auto endTaskIdx = enqueueDmaData.endTaskIdx;
        auto insertBefore = enqueueDmaData.insertBefore;

        barriersConsumedByEnqueueDmas.insert(waitBars.begin(), waitBars.end());

        _log.trace("Enqueue DMA data: pageInd={0}, queueType={1}:{2}, startTaskIdx={3}, endTaskIdx={4}, "
                   "waitBars={5}, insertBefore={6}",
                   pageInd, config::stringifyExecutorKind(queueType.type), queueType.id, startTaskIdx, endTaskIdx,
                   to_small_vector(waitBars), insertBefore);

        auto insertionPointOp = barrierInfo.getTaskOpAtIndex(insertBefore);

        builder.setInsertionPoint(insertionPointOp);

        auto [tileIdx, listIdx] = VPURT::getTileAndListIndex(queueType, numClusters, arch);

        auto executorKindAttr = config::ExecutorKindAttr::get(ctx, queueType.type);
        auto tileIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), tileIdx);
        auto listIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), listIdx);
        auto startTaskIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), startTaskIdx);
        auto endTaskIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), endTaskIdx);
        auto enqueueDMAAttr = VPUIP::EnqueueDMAAttr::get(ctx, executorKindAttr, tileIdxAttr, listIdxAttr,
                                                         startTaskIdxAttr, endTaskIdxAttr);

        auto enqDmaTaskOp = VPUIP::createEnqueueDMA(builder, inBuffer, outBuffer, 0, {}, {}, enqueueDMAAttr);

        enqDmaTaskOp.setWlmPage(pageInd);

        auto enqDMATaskInd = barrierInfo.addNewTaskOp(enqDmaTaskOp);
        llvm::for_each(waitBars, [&](auto waitBar) {
            barrierInfo.addConsumer(waitBar, enqDMATaskInd);
        });

        barrierPagesSplitHandler.updateTaskPageAssignmentForQueue(insertBefore, pageInd, enqueueDmaQueueType,
                                                                  enqDmaTaskOp);
    }

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);

    // Optimize out redundant barrier dependencies inserted which were the result of inserting enqueue DMAs
    // Do such limited targeted modification instead of performing general optimization on whole IR to save compile time
    auto barriersConsumedByEnqueueDmasVec = to_small_vector(barriersConsumedByEnqueueDmas);
    for (auto barInd : barriersConsumedByEnqueueDmasVec) {
        // Remove redundant consumers of the barrier
        auto barCons = barrierInfo.getBarrierConsumers(barInd);
        SmallVector<size_t> dmaP0ChDDrDmas;
        for (auto cons : barCons) {
            auto taskOp = barrierInfo.getTaskOpAtIndex(cons);
            if (VPURT::getTaskQueueType(taskOp, false) == enqueueDmaQueueType) {
                dmaP0ChDDrDmas.push_back(cons);
            }
        }
        llvm::sort(dmaP0ChDDrDmas);
        // Leave only first consumer of barrier, all other are redundant and can be removed
        for (size_t i = 1; i < dmaP0ChDDrDmas.size(); i++) {
            barrierInfo.removeConsumer(barInd, dmaP0ChDDrDmas[i]);
            barrierInfo.updateIRForTask(dmaP0ChDDrDmas[i]);
        }
    }

    barrierPagesSplitHandler =
            VPURT::BarrierPagesSplitHandler{func, barrierInfo, static_cast<size_t>(numBarriers), _log};
    barrierPagesSplitHandler.initializeForEnqueue(func);

    // In case enqueue DMAs were added as last tasks on FIFO perform legalization
    // to satisfy WLM page split restrictions
    auto lastTaskTypePerPageWithNoUpdBar = barrierPagesSplitHandler.getLastTasksOnFifoPerPageWithNoUpdBar();
    if (!lastTaskTypePerPageWithNoUpdBar.empty()) {
        _log.trace("Legalize last task on page with no update barrier");
        barrierPagesSplitHandler.addUpdateBarriersForLastTaskOnFifoInPage(lastTaskTypePerPageWithNoUpdBar);
        // After legalization get updated BarrierInfo and update IR
        barrierInfo = barrierPagesSplitHandler.getUpdatedBarrierInfo();
        VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);
    }

    barrierInfo.clearAttributes();
}
}  // namespace

//
// createFindWlmEnqueueDmasBarrierPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createFindWlmEnqueueDmasBarrierPass(Logger log) {
    return std::make_unique<FindWlmEnqueueDmasBarrierPass>(log);
}
