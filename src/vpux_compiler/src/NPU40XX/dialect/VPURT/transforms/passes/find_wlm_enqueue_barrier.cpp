//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPURT/interfaces/enqueue_barrier.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

namespace vpux::VPURT::arch40xx {
#define GEN_PASS_DECL_FINDWLMENQUEUEBARRIER
#define GEN_PASS_DEF_FINDWLMENQUEUEBARRIER
#include "vpux/compiler/NPU40XX/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT::arch40xx

using namespace vpux;

namespace {

class FindWlmEnqueueBarrierPass final :
        public VPURT::arch40xx::impl::FindWlmEnqueueBarrierBase<FindWlmEnqueueBarrierPass> {
public:
    explicit FindWlmEnqueueBarrierPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FindWlmEnqueueBarrierPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    if (vpux::VPUIP::getWlmStatus(module) != vpux::VPUIP::WlmStatus::ENABLED) {
        // WLM is not supported, no need to run this pass
        return;
    }

    if (!VPURT::verifyOneWaitBarrierPerTask(func, _log)) {
        _log.warning("WLM cannot be enabled as not all tasks have 1 wait barrier");
        vpux::VPUIP::setWlmStatus(module, vpux::VPUIP::WlmStatus::FAILED);
        signalPassFailure();
        return;
    }

    auto& barrierInfo = getAnalysis<BarrierInfo>();

    // Order barriers following barrier consumption order for WLM as this impacts work item ordering
    // and work items are triggered on barrier consumption events
    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);

    VPURT::EnqueueBarrierHandler enqueueBarrier(func, barrierInfo, _log);

    const auto res = enqueueBarrier.calculateEnqueueBarriers();
    if (mlir::failed(res)) {
        _log.warning("Enqueue algorithm failed. Need to switch to nonWLM");
        vpux::VPUIP::setWlmStatus(module, vpux::VPUIP::WlmStatus::FAILED);
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

std::unique_ptr<mlir::Pass> vpux::VPURT::arch40xx::createFindWlmEnqueueBarrierPass(Logger log) {
    return std::make_unique<FindWlmEnqueueBarrierPass>(log);
}
