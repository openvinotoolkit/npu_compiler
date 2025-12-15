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
#define GEN_PASS_DECL_WLMLEGALIZESPLITGRAPHTOPAGES
#define GEN_PASS_DEF_WLMLEGALIZESPLITGRAPHTOPAGES
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class WlmLegalizeSplitGraphToPagesPass final :
        public VPURT::impl::WlmLegalizeSplitGraphToPagesBase<WlmLegalizeSplitGraphToPagesPass> {
public:
    explicit WlmLegalizeSplitGraphToPagesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void WlmLegalizeSplitGraphToPagesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto numBarriers =
            numBarriersOpt.hasValue() ? numBarriersOpt.getValue() : VPUIP::getNumAvailableBarriers(func);

    if (config::getWorkloadManagementStatus(module) != WorkloadManagementStatus::ENABLED) {
        // WLM is not supported, no need to run this pass
        return;
    }

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::BarrierPagesSplitHandler barrierPagesSplitHandler(func, barrierInfo, numBarriers, _log);
    barrierPagesSplitHandler.initializeForLegalization();
    // TODO: Initialization can update dependencies. Separate it to make it clear
    // when and what methods modify dependencies and their call requires updating IR - E#177834
    barrierInfo = barrierPagesSplitHandler.getUpdatedBarrierInfo();
    barrierInfo.updateIR();

    _log.trace("Check and legalize schedule for barrier page split");

    if (!barrierPagesSplitHandler.areNoDepsGoingBeyondNeighborPage()) {
        _log.trace("Legalize long dependencies");
        barrierPagesSplitHandler.legalizeLongDependenciesForBarrierPagesSplit();
        barrierPagesSplitHandler.ensureBarrierHasProducer();
        auto foundRedundantBarriers = barrierPagesSplitHandler.cleanupRedundantBarriers();

        // After legalization get updated BarrierInfo, update IR and recreate BarrierPagesSplitHandler to have up to
        // date state
        barrierInfo = barrierPagesSplitHandler.getUpdatedBarrierInfo();
        if (foundRedundantBarriers) {
            barrierInfo.updateIR();
            VPURT::postProcessBarrierOps(func);
            barrierInfo = vpux::BarrierInfo{func};
        }
        VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);
        barrierPagesSplitHandler =
                VPURT::BarrierPagesSplitHandler{func, barrierInfo, static_cast<size_t>(numBarriers), _log};
        barrierPagesSplitHandler.initializeForLegalization();
    }

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

    if (!barrierPagesSplitHandler.areBoundaryTasksFromNeighborPagesDependent()) {
        _log.trace("Legalize dependency for boundary tasks");
        barrierPagesSplitHandler.legalizeBoundaryTasksForBarrierPagesSplit();
        barrierPagesSplitHandler.ensureBarrierHasProducer();
        auto foundRedundantBarriers = barrierPagesSplitHandler.cleanupRedundantBarriers();

        // After legalization get updated BarrierInfo, update IR and recreate BarrierPagesSplitHandler to have up to
        // date state
        barrierInfo = barrierPagesSplitHandler.getUpdatedBarrierInfo();
        if (foundRedundantBarriers) {
            barrierInfo.updateIR();
            VPURT::postProcessBarrierOps(func);
            barrierInfo = vpux::BarrierInfo{func};
        }
        VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);

        barrierPagesSplitHandler =
                VPURT::BarrierPagesSplitHandler{func, barrierInfo, static_cast<size_t>(numBarriers), _log};
        barrierPagesSplitHandler.initializeForLegalization();
    }

    // Perform final checks after legalization
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");

    barrierInfo = vpux::BarrierInfo{func};
    barrierPagesSplitHandler =
            VPURT::BarrierPagesSplitHandler{func, barrierInfo, static_cast<size_t>(numBarriers), _log};
    barrierPagesSplitHandler.initializeForLegalization();
    barrierPagesSplitHandler.verifyTaskBarrierPagesAreValid();
    barrierPagesSplitHandler.verifyNoCyclicDeps();
    VPUX_THROW_UNLESS(barrierPagesSplitHandler.isSplitToPagesValid(), "Split to pages is not valid");

    barrierInfo.clearAttributes();
}
}  // namespace

//
// createWlmLegalizeSplitGraphToPagesPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createWlmLegalizeSplitGraphToPagesPass(Logger log) {
    return std::make_unique<WlmLegalizeSplitGraphToPagesPass>(log);
}
