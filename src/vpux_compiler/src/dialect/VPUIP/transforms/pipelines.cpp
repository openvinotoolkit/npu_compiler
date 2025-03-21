//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// AsyncScheduling
//

void vpux::VPUIP::buildAsyncSchedulingPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(Core::createMoveDeclarationsToTopPass(log));
    pm.addPass(VPUIP::createWrapIntoAsyncRegionsPass(log));
    pm.addPass(VPUIP::createMoveViewOpsIntoAsyncRegionsPass(log));
    pm.addPass(VPUIP::createMoveWaitResultToAsyncBlockArgsPass(log));
}

//
// HardwareAdaptation
//

void vpux::VPUIP::buildHardwareAdaptationPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(VPUIP::createBreakDataFlowPass(log));
    pm.addPass(VPUIP::createConvertAllocationsToDeclarationsPass(log));
    pm.addPass(VPUIP::createLinearizeCallOpsPass(log));
    pm.addPass(VPUIP::createConvertAsyncOpsToTasksPass(log));
    pm.addPass(VPUIP::createConvertFuncArgsToDeclarationsPass(log));
    pm.addPass(VPUIP::createConvertViewOpsToDeclarationsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(Core::createMoveDeclarationsToTopPass(log));
}

//
// registerVPUIPPipelines
//

void VPUIP::registerVPUIPPipelines() {
    mlir::PassPipelineRegistration<>("async-scheduling", "Asynchronous Scheduling", [](mlir::OpPassManager& pm) {
        VPUIP::buildAsyncSchedulingPipeline(pm);
    });

    mlir::PassPipelineRegistration<>("hardware-adaptation", "Hardware Adaptation", [](mlir::OpPassManager& pm) {
        VPUIP::buildHardwareAdaptationPipeline(pm);
    });
}
