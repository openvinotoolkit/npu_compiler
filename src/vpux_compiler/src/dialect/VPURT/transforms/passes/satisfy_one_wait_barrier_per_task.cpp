//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/wlm_legalization_utils.hpp"

#include <llvm/ADT/SetOperations.h>

namespace vpux::VPURT {
#define GEN_PASS_DECL_SATISFYONEWAITBARRIERPERTASK
#define GEN_PASS_DEF_SATISFYONEWAITBARRIERPERTASK
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class SatisfyOneWaitBarrierPerTaskPass final :
        public VPURT::impl::SatisfyOneWaitBarrierPerTaskBase<SatisfyOneWaitBarrierPerTaskPass> {
public:
    explicit SatisfyOneWaitBarrierPerTaskPass(std::optional<int> virtualBarrierThresholdForWlm,
                                              const bool unevenVariantSplitFlag,
                                              std::optional<WorkloadManagementMode> workloadManagementMode, Logger log)
            : _virtualBarrierThresholdForWlm(virtualBarrierThresholdForWlm),
              _unevenVariantSplitFlag(unevenVariantSplitFlag),
              _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool _mergeWaitBarriersIteratively = false;
    bool _considerTaskExecutorType = false;
    std::optional<int> _virtualBarrierThresholdForWlm;
    bool _unevenVariantSplitFlag;
    std::optional<WorkloadManagementMode> _workloadManagementMode;
};

void SatisfyOneWaitBarrierPerTaskPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto& barrierInfo = getAnalysis<BarrierInfo>();
    if (_unevenVariantSplitFlag) {
        barrierInfo.enableUnevenVariantSplit();
    }

    auto wlmFlag = config::getWorkloadManagementStatus(module) == WorkloadManagementStatus::ENABLED;

    // In case of WLM all tasks need to be driven by single barrier as this is one of the constraints
    // to make each schedule feasible for WLM enabling
    // If WLM is enabled but number of barriers is above threshold do not force it as WLM will not be
    // enabled later nevertheless
    if (wlmFlag && _virtualBarrierThresholdForWlm.has_value() &&
        barrierInfo.getNumOfBarrierOps() > static_cast<size_t>(_virtualBarrierThresholdForWlm.value())) {
        _log.trace("WLM flag turned off because number of barrier is above threshold {0} > {1}",
                   barrierInfo.getNumOfBarrierOps(), _virtualBarrierThresholdForWlm.value());
        wlmFlag = false;
        config::setWorkloadManagementStatus(module, WorkloadManagementStatus::FAILED);
        legalizeScheduleForNonWlm(func, barrierInfo, _log);
    }

    const auto maxAvailableSlots = maxVariantCount.hasValue() ? checked_cast<size_t>(maxVariantCount.getValue())
                                                              : VPUIP::getBarrierMaxVariantCount(func);
    const auto maxSlotsSum = VPUIP::getBarrierMaxVariantSum(func);
    _log.trace("There are {0} slots for each barrier",
               maxSlotsSum < maxAvailableSlots ? maxSlotsSum : maxAvailableSlots);

    const auto availableSlots = vpux::VPUIP::getAvailableSlots(maxSlotsSum, maxAvailableSlots);
    auto mergeBarriersIteratively = mergeWaitBarriersIteratively.hasValue()
                                            ? checked_cast<bool>(mergeWaitBarriersIteratively.getValue())
                                            : _mergeWaitBarriersIteratively;

    if (wlmFlag) {
        mergeBarriersIteratively = true;
        // For some models, strictly enforcing 1-wait barrier per task can lead to performance regression when tasks
        // executor type is not taken into account when batches of tasks must be linearized. Taking into account tasks
        // executor type can help avoid placing tasks from same engine under different barriers, thus not preventing
        // them to run in parallel.
        _considerTaskExecutorType = true;
    }

    // merge parallel wait barriers
    bool modifiedIR = barrierInfo.ensureTasksDrivenBySingleBarrier(availableSlots, mergeBarriersIteratively,
                                                                   _considerTaskExecutorType);

    if (!modifiedIR) {
        // IR was not modified
        barrierInfo.clearAttributes();
        return;
    }

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(func);
    if (!_workloadManagementMode.has_value() ||
        _workloadManagementMode.value() <= WorkloadManagementMode::PWLM_V2_PAGES) {
        VPUX_THROW_UNLESS(VPURT::verifyBarrierSlots(func, _log), "Barrier slot count check failed");
    }
    auto hasOneWaitBarrierPerTask = VPURT::verifyOneWaitBarrierPerTask(func, _log);
    if (mergeBarriersIteratively) {
        VPUX_THROW_UNLESS(hasOneWaitBarrierPerTask, "Encountered task with more than one wait barrier");
    }
}

}  // namespace

//
// createSatisfyOneWaitBarrierPerTaskPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createSatisfyOneWaitBarrierPerTaskPass(
        std::optional<int> virtualBarrierThresholdForWlm, const bool unevenVariantSplitFlag,
        std::optional<WorkloadManagementMode> workloadManagementMode, Logger log) {
    return std::make_unique<SatisfyOneWaitBarrierPerTaskPass>(virtualBarrierThresholdForWlm, unevenVariantSplitFlag,
                                                              workloadManagementMode, log);
}
