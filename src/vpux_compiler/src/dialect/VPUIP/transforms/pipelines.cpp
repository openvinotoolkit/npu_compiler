//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
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
// DMAUnrollingPipeline
//
void vpux::VPUIP::buildDMAUnrollingPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(VPUIP::createUnrollDMAAnalysisPass(log));
    pm.addPass(VPUIP::createUnrollDepthToSpaceDMAPass(log));
    pm.addPass(VPUIP::createUnrollSpaceToDepthDMAPass(log));
    pm.addPass(VPUIP::createUnrollPermuteDMAPass(log));

    pm.addPass(VPUIP::createUnrollUpsamplingDMAPass(log));
    pm.addPass(VPUIP::createUnrollExpandDMAPass(log));
    pm.addPass(VPUIP::createUnrollPerAxisTileDMAPass(log));
    pm.addPass(VPUIP::createUnrollGatherDMAPass(log));
    pm.addPass(VPUIP::createInvalidateUnrollDMAAnalysisPass(log));
}

//
// OptimizeCopies
//

void vpux::VPUIP::buildOptimizeCopiesPipeline(mlir::OpPassManager& pm, const VPUIP::OptimizeCopiesOptionsBase& options,
                                              Logger log) {
    pm.addPass(VPUIP::createOptimizeCopiesPass(log));
    pm.addPass(VPUIP::createUniquifyWeightsTableCopiesPass(log));
    pm.addPass(VPUIP::createOptimizeConcatViewCopiesPass(log));
    pm.addPass(VPUIP::createFuseDDRCopiesIntoConcats(log));
    pm.addPass(VPUIP::createOptimizeParallelCopiesPass(options.enableOptimizeConstCopies, log));
    pm.addPass(VPUIP::createOptimizeSubviewCopiesPass(log));
    pm.addPass(VPUIP::createFuseLastCopyPass(log));
    if (options.enableOpsAsDMA) {
        pm.addPass(VPUIP::createOptimizeTileOpAsNNDMAPass(log));
    }
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

    mlir::PassPipelineRegistration<VPUIP::OptimizeCopiesOptionsBase>(
            "optimize-copies-pipeline", "Optimize Copies Pipeline",
            [](mlir::OpPassManager& pm, const VPUIP::OptimizeCopiesOptionsBase& options) {
                VPUIP::buildOptimizeCopiesPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<>("dma-unrolling", "DMA unrolling", [](mlir::OpPassManager& pm) {
        VPUIP::buildDMAUnrollingPipeline(pm);
    });
}
