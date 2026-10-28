//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::VPU::arch40xx::buildIncrementalPipeline(mlir::OpPassManager& pm, const vpux::MCAndTilingOptionsBase& options,
                                                   Logger log) {
    pm.addPass(VPU::createDecomposeMVNPass(log));

    pm.addPass(VPU::createMultiClusterStrategyAssignmentPass(options.enablePrefetching, options.opTilingCacheThreshold,
                                                             options.mcOptimizationScope,
                                                             options.enableTilingFullSearchSpace, log));
    pm.addPass(VPU::createConvertNCEInterpolateToDWPass(log));
    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyDefaultFileLocation,
                                                  options.readStrategyFromJson, readStrategyDefaultFileLocation,
                                                  options.dumpStrategyToLog, false, log));
    pm.addPass(VPU::createSplitGRUSequencePass(log));
    pm.addPass(VPU::createApplyTilingMVN1SumPass(options.enablePrefetching, log));
    pm.addPass(VPU::createTileLSTMSequencePass(log));

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(/*enableOutputEnsurance=*/true,
                                                           /*enableDequantWeightEnsuranceBeforeStrategy=*/false,
                                                           /*skipConvOC=*/SkipOCMode::SKIP_NONE,
                                                           /*skipEltwiseOC=*/SkipOCMode::SKIP_NONE,
                                                           /*enableSplitChannelForDynamicDequantize=*/false, log));
    pm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ false,
                                             /*disablePassOnEntryFunctionForHostCompile=*/false, log));

    VPU::buildTilingPipeline(pm, VPU::TilingOptions(options), log);

    if (options.enableScfComputeOpsOutlining) {
        VPU::buildScfComputeOpsOutliningPipeline(
                pm, options.loopUnrollFactor, options.enableProfiling, options.enableCascadedUnrolling,
                options.enableBacktrackingBeyondResidualKernel, options.tailOverlapBacktrackMarginPercent,
                options.autoUnrollingMode, options.enableWeightsExtraction, log);
    }

    auto& nestedPm = options.enableScfComputeOpsOutlining ? pm.nest<mlir::ModuleOp>() : pm;
    if (options.enableBoundedTensorsToDynamicDimsMask) {
        nestedPm.addPass(VPU::createBoundedTensorsToDynamicDimsMaskPass(log));
    }

    nestedPm.addPass(VPU::createMakeOpsWithDistributedTensorPass(options.enableExplicitDistributionInfoAttr, log));

    nestedPm.addPass(VPU::createComputeInterpolateCoordinatesPass(options.enableExplicitDistributionInfoAttr, log));
    // E#183249 - reintroduce RemoveOutputSparseToAvoidSuboptimalDPUWorkloads pass after perf regression
    // is investigated

    nestedPm.addPass(VPU::createMakeDistributedCopiesPass(log));
    nestedPm.addPass(VPU::createAdjustDistributedTensorAroundOpsPass(log));

    if (options.enableBoundedTensorsToDynamicDimsMask) {
        // restore dynamic tensor representation after MC tiling
        nestedPm.addPass(VPU::createDynamicDimsMaskToBoundedTensorsPass(log));
    }
}

//
// NCEWorkloadsPipeline
//
// Splits NCE operations onto DPU workloads and legalizes them. When SCF tiling is enabled the workloads are
// generated on the still-rolled loops and the loops are fully unrolled only after all workload legalization
// passes have run, so that every unrolled tile inherits an already correct workload.
//

void vpux::VPU::arch40xx::buildNCEWorkloadsPipeline(mlir::OpPassManager& pm, const VPU::NCEWorkloadsOptions& options,
                                                    Logger log) {
    if (options.enableSCFTiling) {
        pm.addPass(VPU::createWorkloadsForNCEOpsSCFPass(log));
    }

    pm.addPass(VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    pm.addPass(VPU::createCorrectNCEWorkloadsPass(log));
    pm.addPass(VPU::createComputeNCEInputWorkloadsPass(log));
    pm.addPass(VPU::createShiftOutputWorkloadsForHaloPass(log));

    if (options.enableSCFTiling) {
        pm.addPass(VPU::createFullUnrollSCFLoopPass(log));
    }
}

//
// DefaultHWPipeline
//

void vpux::VPU::arch40xx::buildDefaultHWPipeline(mlir::OpPassManager& pm,
                                                 const VPU::arch40xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (options.enableConcatRepeatingBlockOutlining) {
        pm.addPass(VPU::createConcatRepeatingBlocksOutliningPass(options.concatRepeatingBlockOutliningSeqLength,
                                                                 options.concatRepeatingBlockOutliningSdpaHeadThreshold,
                                                                 log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }

    pm.addPass(VPU::createConvertOpToDMAForPerformantExecutionPass(log));
    pm.addPass(VPU::createMoveConvertAroundViewLikeOpsPass(log));
    pm.addPass(VPU::createAdjustForOptimizedLayersPass(log));
    pm.addPass(VPU::createDetectionOutputDecompositionPass(log));
    pm.addPass(VPU::createSplitRealDFTOpsPass(log));
    pm.addPass(VPU::createAdjustLSTMCellInputsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableSEPtrsOperations || options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPU::createSplitSEOpsPass(log));
        pm.addPass(VPU::createLowerOpsToSENCEPass(log));
    }

    pm.addPass(VPU::createFuseConvertPass(log));

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(options.enableOutputEnsurance,
                                                           options.enableDequantWeightEnsuranceBeforeStrategy,
                                                           /*skipConvOC=*/SkipOCMode::SKIP_NONE,
                                                           /*skipEltwiseOC=*/SkipOCMode::SKIP_NONE,
                                                           /*enableSplitChannelForDynamicDequantize=*/false, log));
    pm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ false,
                                             /*disablePassOnEntryFunctionForHostCompile=*/false, log));
    if (options.enableWeightsSparsity) {
        VPU::buildWeightsSparsityPipeline(pm, VPU::WeightsSparsityOptions(options), log);
    }
    if (VPU::isActSparsityEnabled(options.enableActivationSparsity)) {
        VPU::buildActivationSparsityPipeline(pm, VPU::ActivationSparsityOptions(options), log);
        pm.addPass(VPU::createLowerSparsityOpsPass(/*fakeSparsify=*/false, log));
    }

    pm.addPass(VPU::createAddExplicitPaddingBeforeNCEPermutePass(log));

    pm.addPass(VPU::createDetectInPlaceEltwisePass(log));

    pm.addPass(VPU::createCostModelAnalysisConstructPass(log));

    if (options.enableShaveCodeGen) {
        ShaveCodeGen::buildShaveCodeGenPipelineVPU(pm, log, options.enableShaveCodeGenTiling);
    }

    if (options.enableSMPipeline) {
        VPU::buildSMPipeline(pm, vpux::MCAndTilingOptionsBase(options), log);
    } else {
        VPU::arch40xx::buildIncrementalPipeline(pm, vpux::MCAndTilingOptionsBase(options), log);
    }
    auto& nestedPm = options.enableScfComputeOpsOutlining ? pm.nest<mlir::ModuleOp>() : pm;

    nestedPm.addPass(VPU::createAdjustMemorySpacePass(log));
    nestedPm.addPass(VPU::createOptimizeSharedInputCopyForConcatPass(log));
    nestedPm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ false,
                                                   options.disablePassOnEntryFunctionForHostCompile, log));
    nestedPm.addPass(mlir::createCanonicalizerPass(grc));

    nestedPm.addPass(VPU::createCMXConcatPass(log));
    nestedPm.addPass(mlir::createCanonicalizerPass(grc));
    nestedPm.addPass(VPU::createMoveReflectPadToCMXPass(log));
    nestedPm.addPass(VPU::createMoveTensorOpsToCMXPass(log));

    VPU::arch40xx::buildNCEWorkloadsPipeline(nestedPm, VPU::NCEWorkloadsOptions(options), log);

    nestedPm.addPass(mlir::createCanonicalizerPass(grc));
    nestedPm.addPass(createAdjustDynamicOpsBeforeBufferizationPass());
    nestedPm.addPass(VPU::createLegalizeDynamicShapeConcatForSWLayersPass(log));
    nestedPm.addPass(VPU::createAdjustMemorySpaceForSHVOpsPass(log));
    if (options.enableEntireMainContentOutlining) {
        nestedPm.addPass(VPU::createOutlineEntireMainContentPass(log));
    }
}

void vpux::VPU::arch40xx::registerVPUPipelines() {
    mlir::PassPipelineRegistration<VPU::arch40xx::DefaultHWOptions>(
            "default-hw-mode-vpu", "VPU dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const VPU::arch40xx::DefaultHWOptions& options) {
                VPU::arch40xx::buildDefaultHWPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<vpux::arch40xx::MCAndTilingOptionsDevice>(
            "incremental-pipeline", "Apply Incremental Pipeline",
            [](mlir::OpPassManager& pm, const vpux::arch40xx::MCAndTilingOptionsDevice& options) {
                VPU::arch40xx::buildIncrementalPipeline(pm, vpux::MCAndTilingOptionsBase(options));
            });

    mlir::PassPipelineRegistration<vpux::arch40xx::MCAndTilingOptionsDevice>(
            "sm-pipeline", "Apply SM Pipeline",
            [](mlir::OpPassManager& pm, const vpux::arch40xx::MCAndTilingOptionsDevice& options) {
                VPU::buildSMPipeline(pm, vpux::MCAndTilingOptionsBase(options));
            });

    mlir::PassPipelineRegistration<VPU::NCEWorkloadsOptions>(
            "nce-workloads", "Split NCE operations onto DPU workloads and legalize them",
            [](mlir::OpPassManager& pm, const VPU::NCEWorkloadsOptions& options) {
                VPU::arch40xx::buildNCEWorkloadsPipeline(pm, options);
            });
}
