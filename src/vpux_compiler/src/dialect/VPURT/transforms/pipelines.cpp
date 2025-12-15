//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
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
                                                   std::optional<int> virtualBarrierThresholdForWlm,
                                                   std::optional<WorkloadManagementMode> workloadManagementMode,
                                                   const bool unevenVariantSplitFlag, Logger log) {
    if (!workloadManagementMode.has_value() || workloadManagementMode.value() < WorkloadManagementMode::FWLM_V1_PAGES) {
        pm.addPass(VPURT::createSplitExceedingBarrierSlotCountPass(log));
    }
    pm.addPass(VPURT::createSatisfyOneWaitBarrierPerTaskPass(virtualBarrierThresholdForWlm, unevenVariantSplitFlag,
                                                             workloadManagementMode, log));
    if (!workloadManagementMode.has_value() || workloadManagementMode.value() < WorkloadManagementMode::FWLM_V1_PAGES) {
        pm.addPass(VPURT::createReduceExceedingActiveCountBarriersPass(
                virtualBarrierThresholdForWlm, workloadManagementMode, unevenVariantSplitFlag, log));
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
