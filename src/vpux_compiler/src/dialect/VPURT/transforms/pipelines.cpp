//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// BarrierLegalization
//

void vpux::VPURT::buildBarrierLegalizationPipeline(mlir::OpPassManager& pm,
                                                   std::optional<bool> workloadManagementEnabled,
                                                   const bool unevenVariantSplitFlag, Logger log) {
    bool wlmEnabled = workloadManagementEnabled.has_value() && workloadManagementEnabled.value() == true;

    if (!wlmEnabled) {
        pm.addPass(VPURT::createSplitExceedingBarrierSlotCountPass(log));
        pm.addPass(VPURT::createReduceExceedingActiveCountBarriersPass(unevenVariantSplitFlag, log));
    }
}

//
// registerVPURTPipelines
//

void VPURT::registerVPURTPipelines() {
    mlir::PassPipelineRegistration<>("barrier-legalization", "Barrier Legalization", [](mlir::OpPassManager& pm) {
        VPURT::buildBarrierLegalizationPipeline(pm);
    });
}
