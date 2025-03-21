//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/cycle_cost_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

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
    CycleCostInfo cycleCostInfo(funcOp);
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
