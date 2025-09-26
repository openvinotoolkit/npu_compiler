//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_ADDFINALBARRIER
#define GEN_PASS_DEF_ADDFINALBARRIER
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class AddFinalBarrierPass final : public VPURT::impl::AddFinalBarrierBase<AddFinalBarrierPass> {
public:
    explicit AddFinalBarrierPass(std::optional<WorkloadManagementMode> workloadManagementMode, Logger log)
            : _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    std::optional<WorkloadManagementMode> _workloadManagementMode;
    void safeRunOnFunc() final;
};

void AddFinalBarrierPass::safeRunOnFunc() {
    auto func = getOperation();

    VPURT::addFinalBarrierIfNotExists(func, _log);

    if (!_workloadManagementMode.has_value() || _workloadManagementMode <= WorkloadManagementMode::PWLM_V2_PAGES) {
        VPURT::verifyBarrierSlots(func, _log);
    }
}
}  // namespace

//
// createAddFinalBarrierPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createAddFinalBarrierPass(
        std::optional<WorkloadManagementMode> workloadManagementMode, Logger log) {
    return std::make_unique<AddFinalBarrierPass>(workloadManagementMode, log);
}
