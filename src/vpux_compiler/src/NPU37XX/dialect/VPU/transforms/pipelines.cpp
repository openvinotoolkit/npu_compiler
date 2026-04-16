//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::VPU::arch37xx::buildIncrementalPipeline(mlir::OpPassManager& pm, const vpux::MCAndTilingOptionsBase& options,
                                                   Logger log) {
    pm.addPass(VPU::createDecomposeMVNPass(log));

    pm.addPass(VPU::createMultiClusterStrategyAssignmentPass(options.enablePrefetching, options.opTilingCacheThreshold,
                                                             options.mcOptimizationScope, log));

    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                  options.readStrategyFromJson, readStrategyFileLocation,
                                                  options.dumpStrategyToLog, false, log));

    pm.addPass(VPU::createSplitGRUSequencePass(log));
    pm.addPass(VPU::createApplyTilingMVN1SumPass(options.enablePrefetching, log));

    VPU::buildTilingPipeline(pm, VPU::TilingOptions(options), log);

    pm.addPass(VPU::createBoundedTensorsToDynamicDimsMaskPass(log));
    pm.addPass(VPU::createMakeOpsWithDistributedTensorPass(options.enableExplicitDistributionInfoAttr, log));

    pm.addPass(VPU::createComputeInterpolateCoordinatesPass(options.enableExplicitDistributionInfoAttr, log));
    pm.addPass(VPU::createRemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(log));

    pm.addPass(VPU::createMakeDistributedCopiesPass(log));
    pm.addPass(VPU::createAdjustDistributedTensorAroundOpsPass(log));
}

//
// DefaultHWPipeline
//

void vpux::VPU::arch37xx::buildDefaultHWPipeline(mlir::OpPassManager& pm,
                                                 const VPU::arch37xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    /*
      Memory reservation for CMX has to happen as early in VPU as possible. It is required because memory reservation
      decreases usable CMX size which can result in different tiling decisions. If different passes see different
      effective CMX size different failures which can be hard to diagnose can happen. Examples of such failures include:
      - Fail during compilation if additional memory was reserved after tiling but before scheduling since tiles
      selected by tiling pipeline won't fit CMX anymore
      - Memory corruption if additional memory is reserved after scheduler since additional memory will overlap
      addresses allocated by the scheduler Currently there is no validation if memory is not reserved before the first
      call to getTotalCMXSize.
    */
    if (options.enableProfiling) {
        pm.addPass(VPU::createDMATaskProfilingReserveMemPass(options.enableDMAProfiling.getValue(), log));
    }

    /*
      Call this pass after all other memory reservation has already been done. This pass checks if there is 1KiB
      of reserved memory at the end of CMX and extends it if some is missing. So to not waste CMX memory make sure
      as much as possible is allocated in that 1KiB region. Exception to this rule is memory reserved for SW kernel IO
      for such memory make sure to reserve it after this pass to allow data prefetching.
    */
    pm.addPass(VPU::createSWKernelDataPrefetchReserveMemPass(log));

    // TODO: E#140041 enable profiling with outlining
    if (options.enableConcatRepeatingBlockOutlining && canOutlineFromProfilingPerspective(options)) {
        pm.addPass(VPU::createConcatRepeatingBlocksOutliningPass(options.concatRepeatingBlockOutliningSeqLength, log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }

    pm.addPass(VPU::createAdjustForOptimizedLayersPass(log));

    pm.addPass(VPU::createDetectionOutputDecompositionPass(log));
    pm.addPass(VPU::createSplitRealDFTOpsPass(log));

    if (options.enableSEPtrsOperations || options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPU::createSplitSEOpsPass(log));
        pm.addPass(VPU::createLowerOpsToSENCEPass(log));
    }

    pm.addPass(VPU::createFuseClampPass(log));

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(/*enableOutputEnsurance=*/true,
                                                           /*enableDequantWeightEnsuranceBeforeStrategy=*/false,
                                                           /*skipOC=*/false, log));
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

    if (options.enableInPlaceEltwise) {
        pm.addPass(VPU::createDetectInPlaceEltwisePass(log));
    }

    pm.addPass(VPU::createCostModelAnalysisConstructPass(log));
    if (options.enableSMPipeline) {
        VPU::buildSMPipeline(pm, vpux::MCAndTilingOptionsBase(options), log);
    } else {
        VPU::arch37xx::buildIncrementalPipeline(pm, vpux::MCAndTilingOptionsBase(options), log);
    }

    pm.addPass(VPU::createAdjustMemorySpacePass(log));
    pm.addPass(VPU::createOptimizeSharedInputCopyForConcatPass(log));
    pm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ false,
                                             /*disablePassOnEntryFunctionForHostCompile=*/false, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createCMXConcatPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    pm.addPass(VPU::createCorrectNCEWorkloadsPass(log));
    pm.addPass(VPU::createResolveEltwiseWithZTiledWorkloadsPass(log));
    if (options.enableShaveCodeGen) {
        ShaveCodeGen::buildShaveCodeGenPipelineVPU(pm);
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(createAdjustDynamicOpsBeforeBufferizationPass());
    pm.addPass(VPU::createLegalizeDynamicShapeConcatForSWLayersPass(log));
    pm.addPass(VPU::createAdjustMemorySpaceForSHVOpsPass(log));
    pm.addPass(VPU::createOutlineEntireMainContentPass(log));
}

void vpux::VPU::arch37xx::buildReferenceSWPipeline(mlir::OpPassManager& pm,
                                                   const VPU::arch37xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Create DMA HWP scratch buffer
    pm.addPass(VPU::createDMATaskProfilingReserveMemPass("false", log));
    pm.addPass(VPU::createSWKernelDataPrefetchReserveMemPass(log));
    pm.addPass(VPU::createDetectionOutputDecompositionPass(log));
    pm.addPass(VPU::createSplitRealDFTOpsPass(log));
    pm.addPass(VPU::createSplitGRUSequencePass(log));
    pm.addPass(VPU::createDecomposeMVNPass(log));

    pm.addPass(VPU::createFlashSDPATilingStrategyEstimationPass(log));
    pm.addPass(VPU::createTilingStrategyAssignmentPass(
            /*enablePrefetchTiling=*/false, /*enableVPUNNCostForTiling*/ false,
            /*enableShaveDDRAccessOptimization*/ "true", /*enableDynAlignment=*/false, log));
    pm.addPass(VPU::createApplyTilingMVN1SumPass(/*enablePrefetchTiling=*/false, log));
    pm.addPass(VPU::createApplyTilingPass(/*enableSCFTiling=*/false, /*enableDynAlignment=*/false, log));
    pm.addPass(VPU::createComputeInterpolateCoordinatesPass(/*enableExplicitDistributionInfoAttr*/ false, log));

    pm.addPass(VPU::createUnrollFlashSDPAPass(log));
    pm.addPass(VPU::createBoundedTensorsToDynamicDimsMaskPass(log));
    if (options.enableShaveCodeGen) {
        ShaveCodeGen::buildShaveCodeGenPipelineVPU(pm);
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(createAdjustDynamicOpsBeforeBufferizationPass());
    pm.addPass(VPU::createLegalizeDynamicShapeConcatForSWLayersPass(log));
    pm.addPass(VPU::createAdjustMemorySpaceForSHVOpsPass(log));
}

void vpux::VPU::arch37xx::registerVPUPipelines() {
    mlir::PassPipelineRegistration<VPU::arch37xx::DefaultHWOptions>(
            "default-hw-mode-vpu", "VPU dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const VPU::arch37xx::DefaultHWOptions& options) {
                VPU::arch37xx::buildDefaultHWPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<vpux::arch37xx::MCAndTilingOptionsDevice>(
            "incremental-pipeline", "Apply Incremental Pipeline",
            [](mlir::OpPassManager& pm, const vpux::arch37xx::MCAndTilingOptionsDevice& options) {
                VPU::arch37xx::buildIncrementalPipeline(pm, vpux::MCAndTilingOptionsBase(options));
            });

    mlir::PassPipelineRegistration<vpux::arch37xx::MCAndTilingOptionsDevice>(
            "sm-pipeline", "Apply SM Pipeline",
            [](mlir::OpPassManager& pm, const vpux::arch37xx::MCAndTilingOptionsDevice& options) {
                VPU::buildSMPipeline(pm, vpux::MCAndTilingOptionsBase(options));
            });

    mlir::PassPipelineRegistration<VPU::arch37xx::DefaultHWOptions>(
            "reference-sw-mode-vpu", "VPU dialect part of Reference SW pipeline",
            [](mlir::OpPassManager& pm, const VPU::arch37xx::DefaultHWOptions& options) {
                VPU::arch37xx::buildReferenceSWPipeline(pm, options);
            });
}
