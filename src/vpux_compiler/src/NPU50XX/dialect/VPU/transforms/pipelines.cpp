//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU50XX/conversion.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::VPU::arch50xx::buildIncrementalPipeline(mlir::OpPassManager& pm, const vpux::MCAndTilingOptionsBase& options,
                                                   Logger log) {
    pm.addPass(VPU::createDecomposeMVNPass(log));

    pm.addPass(VPU::createMultiClusterStrategyAssignmentPass(options.enablePrefetching, options.opTilingCacheThreshold,
                                                             options.mcOptimizationScope, log));
    if (options.enablePrintStatistics) {
        pm.addPass(VPU::createPrintNNCacheStatisticsPass(log, "multi-cluster-strategy-assignment"));
    }
    pm.addPass(VPU::createConvertNCEInterpolateToDWPass(log));

    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                  options.readStrategyFromJson, readStrategyFileLocation,
                                                  options.dumpStrategyToLog, false, log));
    pm.addPass(VPU::createSplitGRUSequencePass(log));
    pm.addPass(VPU::createApplyTilingMVN1SumPass(options.enablePrefetching, log));
    pm.addPass(VPU::createTileLSTMSequencePass(log));

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(/*enableOutputEnsurance=*/true,
                                                           /*enableDequantWeightEnsuranceBeforeStrategy=*/false,
                                                           /*skipNonConvOC=*/false, log));
    pm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ true,
                                             /*disablePassOnEntryFunctionForHostCompile=*/false, log));

    VPU::buildTilingPipeline(pm, VPU::TilingOptions(options), log);

    if (options.enableScfComputeOpsOutlining) {
        VPU::buildScfComputeOpsOutliningPipeline(pm, options.loopUnrollFactor, log);
    }

    auto& nestedPm = options.enableScfComputeOpsOutlining ? pm.nest<mlir::ModuleOp>() : pm;

    nestedPm.addPass(VPU::createBoundedTensorsToDynamicDimsMaskPass(log));

    nestedPm.addPass(VPU::createMakeOpsWithDistributedTensorPass(options.enableExplicitDistributionInfoAttr, log));

    nestedPm.addPass(VPU::createRelocateWeightTableForReusePass(log));
    nestedPm.addPass(VPU::createComputeInterpolateCoordinatesPass(options.enableExplicitDistributionInfoAttr, log));
    nestedPm.addPass(VPU::createRemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(log));

    nestedPm.addPass(VPU::createMakeDistributedCopiesPass(log));
    nestedPm.addPass(VPU::createAdjustDistributedTensorAroundOpsPass(log));
}

//
// DefaultHWPipeline
//

void vpux::VPU::arch50xx::buildDefaultHWPipeline(mlir::OpPassManager& pm,
                                                 const VPU::arch50xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    /*
        Memory reservation for CMX has to happen as early in VPU as possible. It is required because memory reservation
        decreases usable CMX size which can result in different tiling decisions. If different passes see different
        effective CMX size different failures which can be hard to diagnose can happen. Examples of such failures
       include:
        - Fail during compilation if additional memory was reserved after tiling but before scheduling since tiles
        selected by tiling pipeline won't fit CMX anymore
        - Memory corruption if additional memory is reserved after scheduler since additional memory will overlap
        addresses allocated by the scheduler Currently there is no validation if memory is not reserved before the first
        call to getTotalCMXSize.
    */

    if (options.enableCompressActivationSpill) {
        pm.addPass(VPU::createCompressDmaReserveMemPass(log));
    }

    if (options.enableProfiling) {
        pm.addPass(VPU::createDMATaskProfilingReserveMemPass(options.enableDMAProfiling.getValue(), log));
    }

    // TODO: E#140041 enable profiling with outlining
    if (options.enableConcatRepeatingBlockOutlining && canOutlineFromProfilingPerspective(options)) {
        pm.addPass(VPU::createConcatRepeatingBlocksOutliningPass(options.concatRepeatingBlockOutliningSeqLength, log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }

    pm.addPass(VPU::createConvertOpToDMAForPerformantExecutionPass(log));
    pm.addPass(VPU::createMoveConvertAroundViewLikeOpsPass(log));
    pm.addPass(VPU::createAdjustForOptimizedLayersPass(log));
    pm.addPass(VPU::createDetectionOutputDecompositionPass(log));
    pm.addPass(VPU::createSplitRealDFTOpsPass(log));
    pm.addPass(VPU::createAdjustLSTMCellInputsPass(log));
    if (options.enableSEPtrsOperations || options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPU::createSplitSEOpsPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
        pm.addPass(VPU::createLowerOpsToSENCEPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
    }

    pm.addPass(VPU::createFuseClampPass(log));

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(options.enableOutputEnsurance,
                                                           options.enableDequantWeightEnsuranceBeforeStrategy,
                                                           /*skipNonConvOC=*/true, log));
    pm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ true,
                                             /*disablePassOnEntryFunctionForHostCompile=*/false, log));

    if (options.enableWeightsSparsity) {
        VPU::buildWeightsSparsityPipeline(pm, VPU::WeightsSparsityOptions(options), log);
    }
    if (VPU::isActSparsityEnabled(options.enableActivationSparsity)) {
        VPU::buildActivationSparsityPipeline(pm, VPU::ActivationSparsityOptions(options), log);
        pm.addPass(VPU::createLowerSparsityOpsPass(/*fakeSparsify=*/false, log));
    }

    pm.addPass(VPU::arch50xx::createAutopadChannelsPass(log));
    pm.addPass(VPU::createAddExplicitPaddingBeforeNCEPermutePass(log));

    if (options.enableInPlaceEltwise) {
        pm.addPass(VPU::createDetectInPlaceEltwisePass(log));
    }

    pm.addPass(VPU::createCostModelAnalysisConstructPass(log));
    if (options.enableSMPipeline) {
        VPU::buildSMPipeline(pm, vpux::MCAndTilingOptionsBase(options), log);
    } else {
        VPU::arch50xx::buildIncrementalPipeline(pm, vpux::MCAndTilingOptionsBase(options), log);
    }

    auto& nestedPm = options.enableScfComputeOpsOutlining ? pm.nest<mlir::ModuleOp>() : pm;

    nestedPm.addPass(VPU::createAdjustMemorySpacePass(log));
    nestedPm.addPass(VPU::createOptimizeSharedInputCopyForConcatPass(log));
    nestedPm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ true,
                                                   options.disablePassOnEntryFunctionForHostCompile, log));
    nestedPm.addPass(mlir::createCanonicalizerPass(grc));

    nestedPm.addPass(VPU::createCMXConcatPass(log));
    nestedPm.addPass(mlir::createCanonicalizerPass(grc));
    nestedPm.addPass(VPU::createMoveReflectPadToCMXPass(log));

    nestedPm.addPass(vpux::VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    if (options.enablePrintStatistics) {
        nestedPm.addPass(VPU::createPrintNNCacheStatisticsPass(log, "split-NCE-ops-onto-workloads"));
    }
    nestedPm.addPass(VPU::createCorrectNCEWorkloadsPass(log));
    nestedPm.addPass(VPU::createComputeNCEInputWorkloadsPass(log));
    nestedPm.addPass(VPU::createShiftOutputWorkloadsForHaloPass(log));
    if (options.enableShaveCodeGen) {
        VPU::buildShaveCodeGenPipeline(nestedPm);
    }
    nestedPm.addPass(mlir::createCanonicalizerPass(grc));
    nestedPm.addPass(createAdjustDynamicOpsBeforeBufferizationPass());
    nestedPm.addPass(VPU::createLegalizeDynamicShapeConcatForSWLayersPass(log));
    nestedPm.addPass(VPU::createAdjustMemorySpaceForSHVOpsPass(log));
    if (options.enableEntireMainContentOutlining && canOutlineFromProfilingPerspective(options)) {
        nestedPm.addPass(VPU::createOutlineEntireMainContentPass(log));
    }
}

void vpux::VPU::arch50xx::registerVPUPipelines() {
    mlir::PassPipelineRegistration<VPU::arch50xx::DefaultHWOptions>(
            "default-hw-mode-vpu", "VPU dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const VPU::arch50xx::DefaultHWOptions& options) {
                VPU::arch50xx::buildDefaultHWPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<vpux::arch40xx::MCAndTilingOptionsDevice>(
            "incremental-pipeline", "Apply Incremental Pipeline",
            [](mlir::OpPassManager& pm, const vpux::arch40xx::MCAndTilingOptionsDevice& options) {
                VPU::arch50xx::buildIncrementalPipeline(pm, vpux::MCAndTilingOptionsBase(options));
            });

    mlir::PassPipelineRegistration<vpux::arch40xx::MCAndTilingOptionsDevice>(
            "sm-pipeline", "Apply SM Pipeline",
            [](mlir::OpPassManager& pm, const vpux::arch40xx::MCAndTilingOptionsDevice& options) {
                VPU::buildSMPipeline(pm, vpux::MCAndTilingOptionsBase(options));
            });
}
