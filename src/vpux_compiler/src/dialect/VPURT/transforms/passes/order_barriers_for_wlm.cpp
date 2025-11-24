//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_ORDERBARRIERSFORWLM
#define GEN_PASS_DEF_ORDERBARRIERSFORWLM
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class OrderBarriersForWlmPass final : public VPURT::impl::OrderBarriersForWlmBase<OrderBarriersForWlmPass> {
public:
    explicit OrderBarriersForWlmPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OrderBarriersForWlmPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    if (config::getWorkloadManagementStatus(module) != WorkloadManagementStatus::ENABLED) {
        // WLM is not supported, no need to run this pass
        return;
    }

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);
    barrierInfo.clearAttributes();
}
}  // namespace

//
// createOrderBarriersForWlmPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createOrderBarriersForWlmPass(Logger log) {
    return std::make_unique<OrderBarriersForWlmPass>(log);
}
