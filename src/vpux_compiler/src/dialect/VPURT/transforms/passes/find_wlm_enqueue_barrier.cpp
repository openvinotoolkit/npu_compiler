//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/enqueue_barrier.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_FINDWLMENQUEUEBARRIER
#define GEN_PASS_DEF_FINDWLMENQUEUEBARRIER
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class FindWlmEnqueueBarrierPass final : public VPURT::impl::FindWlmEnqueueBarrierBase<FindWlmEnqueueBarrierPass> {
public:
    explicit FindWlmEnqueueBarrierPass(WorkloadManagementMode workloadManagementMode, bool disableDmaSwFifo, Logger log)
            : _workloadManagementMode(workloadManagementMode), _disableDmaSwFifo(disableDmaSwFifo) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
#if defined(__clang__)
    [[maybe_unused]] WorkloadManagementMode _workloadManagementMode;
#else
    WorkloadManagementMode _workloadManagementMode;
#endif
    bool _disableDmaSwFifo;
    void safeRunOnFunc() final;
};

void FindWlmEnqueueBarrierPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    if (config::getWorkloadManagementStatus(module) != WorkloadManagementStatus::ENABLED) {
        // WLM is not supported, no need to run this pass
        return;
    }

    if (!VPURT::verifyOneWaitBarrierPerTask(func, _log)) {
        _log.warning("WLM cannot be enabled as not all tasks have 1 wait barrier");
        config::setWorkloadManagementStatus(module, WorkloadManagementStatus::FAILED);
        signalPassFailure();
        return;
    }

    auto& barrierInfo = getAnalysis<BarrierInfo>();

    // Order barriers following barrier consumption order for WLM as this impacts work item ordering
    // and work items are triggered on barrier consumption events
    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);

    VPURT::EnqueueBarrierHandler enqueueBarrier(func, barrierInfo, _disableDmaSwFifo, _log);

    const auto enqDmaAtBootstrap = enqDmaAtBootstrapOpt.hasValue() ? enqDmaAtBootstrapOpt.getValue() : false;

    mlir::DenseSet<vpux::VPU::ExecutorKind> executorEnqAtBootstrap;
    if (enqDmaAtBootstrap) {
        executorEnqAtBootstrap.insert(vpux::VPU::ExecutorKind::DMA_NN);
    }

    const auto res = enqueueBarrier.calculateEnqueueBarriers(executorEnqAtBootstrap);
    if (mlir::failed(res)) {
        _log.warning("Enqueue algorithm failed. Need to switch to nonWLM");
        config::setWorkloadManagementStatus(module, WorkloadManagementStatus::FAILED);
        signalPassFailure();
        return;
    }

    func.walk([&](VPURT::TaskOp taskOp) {
        auto enqBar = enqueueBarrier.getEnqueueBarrier(taskOp);

        auto taskInd = barrierInfo.getIndex(taskOp);
        auto waitBars = barrierInfo.getWaitBarriers(taskInd);
        _log.trace("Enqueue task {0} with wait barrier {1} at barrier {2}", taskInd,
                   (waitBars.empty() ? "NONE" : std::to_string(*waitBars.begin())),
                   (enqBar == nullptr
                            ? "BOOTSTRAP"
                            : std::to_string(barrierInfo.getIndex(enqBar.getDefiningOp<VPURT::ConfigureBarrierOp>()))));
        if (enqBar != nullptr) {
            taskOp.getEnqueueBarrierMutable().assign(enqBar);
        }
    });

    barrierInfo.clearAttributes();
}
}  // namespace

//
// createFindWlmEnqueueBarrierPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createFindWlmEnqueueBarrierPass(WorkloadManagementMode workloadManagementMode,
                                                                         bool disableDmaSwFifo, Logger log) {
    return std::make_unique<FindWlmEnqueueBarrierPass>(workloadManagementMode, disableDmaSwFifo, log);
}
