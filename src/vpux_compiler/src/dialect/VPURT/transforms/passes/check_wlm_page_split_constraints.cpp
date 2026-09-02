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
#include "vpux/compiler/dialect/VPURT/utils/wlm_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_CHECKWLMPAGESPLITCONSTRAINTS
#define GEN_PASS_DEF_CHECKWLMPAGESPLITCONSTRAINTS
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class CheckWlmPageSplitConstraintsPass final :
        public VPURT::impl::CheckWlmPageSplitConstraintsBase<CheckWlmPageSplitConstraintsPass> {
public:
    explicit CheckWlmPageSplitConstraintsPass(std::optional<WorkloadManagementMode> workloadManagementMode, Logger log)
            : _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    std::optional<WorkloadManagementMode> _workloadManagementMode;
};

void CheckWlmPageSplitConstraintsPass::safeRunOnFunc() {
    auto func = getOperation();

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    barrierInfo.buildTaskQueueTypeMap();
    barrierInfo.buildWlmTasksDependencies(true);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");

    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandler(func, barrierInfo, numBarriers, _log);
    barrierPagesSplitHandler.initializeForVerification(func);
    barrierPagesSplitHandler.verifyTaskBarrierPagesAreValid();
    barrierPagesSplitHandler.verifyTaskPagesAreNeverDecreasingOnQueue();
    barrierPagesSplitHandler.verifyNoCyclicDeps();
    VPUX_THROW_UNLESS(barrierPagesSplitHandler.isSplitToPagesValid(), "Split to pages is not valid");

    barrierPagesSplitHandler.verifyPhysicalBarsDependencies();

    const auto wlmMode = _workloadManagementMode.value();
    if (wlmMode == WorkloadManagementMode::FWLM_V1_PAGES) {
        barrierPagesSplitHandler.verifyBarProgDmaDependencies(func);
        barrierPagesSplitHandler.verifyEnqueueDmas(func);
        barrierPagesSplitHandler.verifyEnqueueOfDmas(func);
    }
    // Once we have enqueues inserted, check if FetchTasks have all required dependencies
    auto& execGroupAnalysis = getAnalysis<ExecutionGroupAnalysis>();
    auto dpuGroups = execGroupAnalysis.getDPUExecutionGroups();
    auto swGroups = execGroupAnalysis.getActShvExecutionGroups();
    VPUX_THROW_WHEN(!VPURT::verifyFetchDmaDependencies(func, barrierInfo, dpuGroups, _log),
                    "Unsafe dependencies for Fetch DMA around DPUs");
    VPUX_THROW_WHEN(!VPURT::verifyFetchDmaDependencies(func, barrierInfo, swGroups, _log),
                    "Unsafe dependencies for Fetch DMA around SHVs");

    barrierInfo.clearAttributes();
}
}  // namespace

//
// createCheckWlmPageSplitConstraintsPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createCheckWlmPageSplitConstraintsPass(
        WorkloadManagementMode workloadManagementMode, Logger log) {
    return std::make_unique<CheckWlmPageSplitConstraintsPass>(workloadManagementMode, log);
}
