//
// Copyright (C) 2024-2026 Intel Corporation
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
                                                   Logger log, bool enableLegacyConvSplitOverIC) {
    // When RunMVNNormalizeOnDPU is enabled, force-decompose all NHWC MVN ops so that the
    // resulting MVN1NormalizeOp can be converted to NCE.MaxPool even when per-channel tiles
    // fit in CMX (which is the common case for K2-style models with large spatial MVN ops).
    const bool forceDecompose = options.enableRunMVNNormalizeOnDPU;
    pm.addPass(VPU::createDecomposeMVNPass(log, forceDecompose));
    if (options.enableRunMVNNormalizeOnDPU) {
        pm.addPass(VPU::createRunMVNNormalizeOnDPUPass(log));
    }
    pm.addPass(VPU::createMultiClusterStrategyAssignmentPass(options.enablePrefetching, options.opTilingCacheThreshold,
                                                             options.mcOptimizationScope,
                                                             options.enableTilingFullSearchSpace, log));
    if (options.enablePrintStatistics) {
        pm.addPass(VPU::createPrintNNCacheStatisticsPass(log, "multi-cluster-strategy-assignment"));
    }
    pm.addPass(VPU::createConvertNCEInterpolateToDWPass(log));

    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyDefaultFileLocation,
                                                  options.readStrategyFromJson, readStrategyDefaultFileLocation,
                                                  options.dumpStrategyToLog, false, log));
    pm.addPass(VPU::createSplitGRUSequencePass(log));
    pm.addPass(VPU::createApplyTilingMVN1SumPass(options.enablePrefetching, log));
    pm.addPass(VPU::createTileLSTMSequencePass(log));
    pm.addPass(VPU::createSplitGatedDeltaNetPass(log));

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(/*enableOutputEnsurance=*/true,
                                                           /*enableDequantWeightEnsuranceBeforeStrategy=*/false,
                                                           /*skipConvOC=*/SkipOCMode::SKIP_LARGE_SPATIAL,
                                                           /*skipEltwiseOC=*/SkipOCMode::SKIP_LARGE_SPATIAL,
                                                           /*enableSplitChannelForDynamicDequantize=*/true, log));
    pm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ true,
                                             /*disablePassOnEntryFunctionForHostCompile=*/false, log));

    auto tilingOpts = VPU::TilingOptions(options);
    tilingOpts.enableLegacyConvSplitOverIC = enableLegacyConvSplitOverIC;
    VPU::buildTilingPipeline(pm, tilingOpts, log);

    if (options.enableScfComputeOpsOutlining) {
        VPU::buildScfComputeOpsOutliningPipeline(
                pm, options.loopUnrollFactor, options.enableProfiling, options.enableCascadedUnrolling,
                options.enableBacktrackingBeyondResidualKernel, options.tailOverlapBacktrackMarginPercent,
                options.autoUnrollingMode, options.enableWeightsExtraction, log);
    }

    auto& nestedPm = options.enableScfComputeOpsOutlining ? pm.nest<mlir::ModuleOp>() : pm;

    // Re-run RunMVNNormalizeOnDPU (which includes RunAddBroadcastOnDPU) after tiling and
    // ScfComputeOpsOutlining. Tiling does not create new MVN1NormalizeOps (those are fixed before
    // tiling), but tiling of a force-decomposed MVN1Normalize can produce broadcast VPU.AddOps that
    // RunAddBroadcastOnDPU (a sub-pattern of RunMVNNormalizeOnDPU) must convert to DPU NCE.MaxPool.
    // Note: RunMVNNormalizeOnDPU is a FunctionPass; adding it to a ModuleOp-scoped nestedPm is
    // valid -- OpPassManager::addPass() implicitly nests it under func::FuncOp (standard MLIR behaviour).
    if (options.enableRunMVNNormalizeOnDPU) {
        nestedPm.addPass(VPU::createRunMVNNormalizeOnDPUPass(log));
    }

    if (options.enableBoundedTensorsToDynamicDimsMask) {
        nestedPm.addPass(VPU::createBoundedTensorsToDynamicDimsMaskPass(log));
    }

    nestedPm.addPass(VPU::createMakeOpsWithDistributedTensorPass(options.enableExplicitDistributionInfoAttr, log));

    nestedPm.addPass(VPU::createRelocateWeightTableForReusePass(log));
    nestedPm.addPass(VPU::createComputeInterpolateCoordinatesPass(options.enableExplicitDistributionInfoAttr, log));
    nestedPm.addPass(VPU::createRemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(log));

    nestedPm.addPass(VPU::createMakeDistributedCopiesPass(log));
    nestedPm.addPass(VPU::createAdjustDistributedTensorAroundOpsPass(log));

    if (options.enableBoundedTensorsToDynamicDimsMask) {
        // restore dynamic tensor representation after MC tiling
        nestedPm.addPass(VPU::createDynamicDimsMaskToBoundedTensorsPass(log));
    }
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
    if (options.enableSoftmaxDecomposition) {
        pm.addPass(VPU::arch50xx::createDecomposeSoftmaxInSdpaPass(log));
    }
    pm.addPass(VPU::createAdjustForOptimizedLayersPass(log));
    pm.addPass(VPU::createDetectionOutputDecompositionPass(log));
    pm.addPass(VPU::createSplitRealDFTOpsPass(log));
    pm.addPass(VPU::createAdjustLSTMCellInputsPass(log));
    if (options.enableSEPtrsOperations || options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPU::createSplitSEOpsPass(log));
        pm.addPass(VPU::createLowerOpsToSENCEPass(log));
    }

    pm.addPass(VPU::createFuseConvertPass(log));

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(options.enableOutputEnsurance,
                                                           options.enableDequantWeightEnsuranceBeforeStrategy,
                                                           /*skipConvOC=*/SkipOCMode::SKIP_LARGE_SPATIAL,
                                                           /*skipEltwiseOC=*/SkipOCMode::SKIP_ALL,
                                                           /*enableSplitChannelForDynamicDequantize=*/true, log));
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

    pm.addPass(VPU::createDetectInPlaceEltwisePass(log));

    pm.addPass(VPU::createCostModelAnalysisConstructPass(log));

    if (options.enableShaveCodeGen) {
        ShaveCodeGen::buildShaveCodeGenPipelineVPU(pm, log, options.enableShaveCodeGenTiling);
    }

    if (options.enableSMPipeline) {
        VPU::buildSMPipeline(pm, vpux::MCAndTilingOptionsBase(options), log);
    } else {
        VPU::arch50xx::buildIncrementalPipeline(pm, vpux::MCAndTilingOptionsBase(options), log,
                                                options.enableLegacyConvSplitOverIC);
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
    nestedPm.addPass(VPU::createMoveTensorOpsToCMXPass(log));

    if (options.enableSCFTiling) {
        nestedPm.addPass(VPU::createWorkloadsForNCEOpsSCFPass(log));
        nestedPm.addPass(VPU::createFullUnrollSCFLoopPass(log));
    }

    nestedPm.addPass(vpux::VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    if (options.enablePrintStatistics) {
        nestedPm.addPass(VPU::createPrintNNCacheStatisticsPass(log, "split-NCE-ops-onto-workloads"));
    }
    nestedPm.addPass(VPU::createCorrectNCEWorkloadsPass(log));
    nestedPm.addPass(VPU::createComputeNCEInputWorkloadsPass(log));
    nestedPm.addPass(VPU::createShiftOutputWorkloadsForHaloPass(log));

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
                VPU::arch50xx::buildIncrementalPipeline(pm, vpux::MCAndTilingOptionsBase(options), Logger::global(),
                                                        /*enableLegacyConvSplitOverIC=*/false);
            });

    mlir::PassPipelineRegistration<vpux::arch40xx::MCAndTilingOptionsDevice>(
            "sm-pipeline", "Apply SM Pipeline",
            [](mlir::OpPassManager& pm, const vpux::arch40xx::MCAndTilingOptionsDevice& options) {
                VPU::buildSMPipeline(pm, vpux::MCAndTilingOptionsBase(options));
            });
}
