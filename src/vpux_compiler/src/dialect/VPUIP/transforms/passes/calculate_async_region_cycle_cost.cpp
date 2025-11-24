//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/cycle_cost_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CALCULATEASYNCREGIONCYCLECOST
#define GEN_PASS_DEF_CALCULATEASYNCREGIONCYCLECOST
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

class CalculateAsyncRegionCycleCostPass final :
        public VPUIP::impl::CalculateAsyncRegionCycleCostBase<CalculateAsyncRegionCycleCostPass> {
public:
    explicit CalculateAsyncRegionCycleCostPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void CalculateAsyncRegionCycleCostPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    const auto arch = config::getArch(module);
    auto maybeCostModelAnalysis = getCachedParentAnalysis<VPU::CostModelAnalysis>(module);
    auto costModel = VPU::CostModelAnalysis::getOrCreateCostModel(maybeCostModelAnalysis, arch, _log);
    CycleCostInfo cycleCostInfo(std::move(costModel), funcOp);
    funcOp->walk([&](mlir::async::ExecuteOp execOp) {
        if (auto costInterface = mlir::dyn_cast_or_null<VPUIP::CycleCostInterface>(execOp.getOperation())) {
            auto cycleCost = cycleCostInfo.getCycleCost(costInterface);
            execOp->setAttr(cycleCostAttrName, getIntAttr(execOp->getContext(), cycleCost));
        }
    });
}
}  // namespace

//
// createCalculateAsyncRegionCycleCostPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createCalculateAsyncRegionCycleCostPass(Logger log) {
    return std::make_unique<CalculateAsyncRegionCycleCostPass>(log);
}
