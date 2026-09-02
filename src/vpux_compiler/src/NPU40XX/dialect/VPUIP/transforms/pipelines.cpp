//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/version.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::VPUIP::arch40xx::buildMemoryAllocationPipeline(mlir::OpPassManager& pm,
                                                          const VPUIP::arch40xx::MemoryAllocationOptions& options,
                                                          Logger log) {
    pm.addPass(VPUIP::createFeasibleAllocationPass(
            VPU::getMemKind<VPU::MemoryKind::CMX_NN>, VPU::getMemKind<VPU::MemoryKind::DDR>, options.linearizeSchedule,
            options.enableLoopAllocation, options.enablePipelining, options.enablePrefetching,
            options.optimizeFragmentation, options.optimizeDynamicSpilling, options.enableMultiScheduleHeuristic,
            options.enableVfUndefinedScheduler, log));
    if (options.enablePrintStatistics) {
        pm.addPass(VPU::createPrintNNCacheStatisticsPass(log, "feasible-allocation"));
    }

    if (options.enableCompressActivationSpill) {
        pm.addPass(VPUIP::createAdjustSpillSizePass(log));
    }

    pm.addPass(VPUIP::createQueryArgsAllocationAnalysisPass());
    pm.addPass(VPUIP::createUngroupBoundedBuffersPass(log));
    pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));
}

void vpux::VPUIP::arch40xx::buildDefaultHWPipeline(mlir::OpPassManager& pm,
                                                   const VPUIP::arch40xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Ensure the cost model analysis is constructed
    // The analysis is constructed if it does not exist yet and reused otherwise,
    // so it does not really affect the Default pipeline where the cache is created in VPU dialect,
    // but it helps specifically for the WS where VPUIP pipeline can be run separately by dedicated PassManager
    pm.addPass(VPU::createCostModelAnalysisConstructPass(log));
    if (options.enableShaveCodeGen) {
        ShaveCodeGen::buildShaveCodeGenPipelineVPUIP(pm);
    }

    pm.addPass(VPUIP::createTileActShaveKernelTaskPass(log));

    // This pass is a part of "copy optimization pipeline", but need to be done before because
    // WrapWithPermuteAsNNDMA depends on it.
    pm.addPass(VPUIP::createMovePureViewOpBeforeCopyPass(log));

    if (options.enableOpsAsDMA) {
        pm.addPass(VPUIP::createWrapWithPermuteAsNNDMAPass(log));
    }
    pm.addPass(VPUIP::createOptimizeExpandSubviewPass(log));
    pm.addPass(VPUIP::createConvertExpandPass(options.deferExpandToExpandDMA, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPUIP::createConvertEltwiseToInPlacePass(log));

    // Level 2 : Abstract RunTime

    pm.addPass(VPUIP::createSetMemorySpacePass(VPU::getMemKind<VPU::MemoryKind::DDR>,
                                               options.setMemorySpaceForFunctionBoundaries, log));

    if (options.enableSEPtrsOperations || options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPUIP::createComputeSEBasePtrsPass(log));
        pm.addPass(VPUIP::createConvertSETablesToConstantsPass(log));
    }
    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createPropagateSparsityCompressionPass(log));
    }

    if (options.enableWeightsSparsity || VPU::isActSparsityEnabled(options.enableActivationSparsity) ||
        options.enableSEPtrsOperations || options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPUIP::createUngroupBufferSectionRewriterExecutorPass(options.enableSEPtrsOperations ||
                                                                         options.enableExperimentalSEPtrsOperations));
    }

    pm.addPass(VPUIP::createUngroupBoundedBuffersPass(log));

    VPUIP::buildOptimizeCopiesPipeline(pm, VPUIP::OptimizeCopiesOptionsBase(options), log);

    pm.addPass(VPUIP::createConvertDynamicReshapeToInPlacePass(log));
    pm.addPass(VPUIP::createInsertCopyForEltwiseInPlaceInputPass(log));
    pm.addPass(VPUIP::createOptimizeConvertDMAOpPass(log));

    if (options.enableOpsAsDMA) {
        pm.addPass(VPUIP::createConvertToDMAPass(log));
    }

    pm.addPass(VPUIP::createAddCopyBetweenSWKernelsAndNetworkIOPass(log));
    pm.addPass(VPUIP::createConvertVPUIPCopyToSWCopyPass(log));
    pm.addPass(VPUIP::createCopyOpTilingPass(log));

    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(VPUIP::createConvWeightsCompressionPass(log));

    if (VPU::isActSparsityEnabled(options.enableActivationSparsity)) {
        pm.addPass(VPUIP::createComputeSESizesPass(/*onlyInputsConcatOverC=*/true, log));
    }

    pm.addPass(VPUIP::createFuseConstantsPass(log));

    if (options.enableWeightsSwizzling || options.enableActivationSwizzling) {
        pm.addPass(VPUIP::createSwizzlingPass(options.enableWeightsSwizzling, options.enableActivationSwizzling, log));
    }

    // Note: this pass introduces necessary VPUIP.Copy operations, thus, it must
    // be called *after* all copy optimizations are run (to ensure the
    // introduced copies are not optimized out).

    // Batch compile method 'debatch' adheres to another function calling consideration
    // based on FucntionInput/Output sections, which prevent a demand that
    // repeating block same function arguments must have a same DDR offset,
    // so that we turn this legalization unless the batch compile method is different
    if (!DebatcherOptions::isAvailable(options)) {
        pm.addPass(VPUIP::createLegalizeRepeatingFuncCallsPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPUIP::createConvertTransferOpsToDMAsPass(log));

    pm.addPass(VPUIP::createLegalizeStridedDMAsPass(log));

    if (options.enableProfiling && options.enableDPUProfiling) {
        pm.addPass(VPUIP::createDPUProfilingPass(log));
    }

    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(log));
    }

    VPUIP::buildAsyncSchedulingPipeline(pm, log);
    if (options.enableAsyncRegionOutlining && canOutlineFromProfilingPerspective(options)) {
        pm.addPass(VPUIP::createAsyncRegionsOutliningPass(options.asyncRegionOutliningMinOpsInBlock, log));
    }

    pm.addPass(VPUIP::createCalculateAsyncRegionCycleCostPass(log));

    VPUIP::arch40xx::buildMemoryAllocationPipeline(pm, VPUIP::arch40xx::MemoryAllocationOptions(options), log);

    pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));

    if (options.enablePopulateWeightTableWithShave) {
        pm.addPass(VPUIP::createPatchPopulateWeightTableWithShavePass(log));
    }

    // Handle WeightsTable, which requires statically allocated memory
    pm.addPass(VPUIP::createPatchWeightsTablePass(log));

    pm.addPass(VPUIP::createAddSwKernelCacheHandlingOpsPass(log));

    VPUIP::buildHardwareAdaptationPipeline(pm, log);

    // Level 1 : VPU RunTime
    pm.addPass(VPUIP::createAssignShvLogicalTaskIndexPass(log));
    pm.addPass(VPUIP::createUnrollSwKernelPass(log));

    pm.addPass(VPUIP::createUnrollDistributedOpsPass(log, options.enableSegmentedDmaFusion));
    pm.addPass(VPUIP::createBatchMatMulToMatMulPass(log));
    pm.addPass(VPUIP::createDetectDMASplitCandidatePass(log));
    pm.addPass(VPUIP::createNNDMATilingPass(log));
    pm.addPass(VPUIP::createSegmentHalosPass(log));
    pm.addPass(VPUIP::createComputeHaloRegionForDPUTaskOpPass(log));

    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createFlattenSparseWeightsTypesPass(log));
    }
    if (VPU::isActSparsityEnabled(options.enableActivationSparsity) || options.enableSEPtrsOperations ||
        options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPUIP::createComputeSESizesPass(/*onlyInputsConcatOverC=*/false, log));
    }
    if (options.enableSEPtrsOperations || options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPUIP::createAdjustInputDataForExplicitSETablePass(log));
    }

    VPUIP::buildDMAUnrollingPipeline(pm, log);

    if (options.enableWeightsSwizzling || options.enableActivationSwizzling) {
        pm.addPass(Const::createApplySwizzlingPass());
        pm.addPass(VPUIP::createResolveDMAWithSwizzlingPass(log));
    }

    if (options.enableCompressWeightsBTC) {
        pm.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }

    pm.addPass(VPUIP::createSplitDMAToBalanceLoadPass(log));
    if (options.enableSegmentedDmaFusion) {
        pm.addPass(VPUIP::createFuseSegmentedDmaPass(log));
    }

    const bool isInliningRequired = isOutliningEnabled(options);
    if (isInliningRequired) {
        if (options.enableBarrierSchedWithFunctionOutlining) {
            pm.addPass(VPURT::createInsertSyncTasksPass(log));
        } else {
            if (auto debatcherReorderingOptionsPtr = IE::DebatcherOpReorderingOptions::create(options, log)) {
                pm.addPass(IE::createRevertTileExecutorNumPass(*debatcherReorderingOptionsPtr, log));
            }
            pm.addPass(VPUIP::createDispatchedInlinerPass(log));
        }
    }

    pm.addPass(VPUIP::createLegalizeNNDMADimSizesPass(log));
    if (options.enableProfiling) {
        pm.addPass(VPUIP::createDMATaskProfilingHwDdrPass(options.enableDMAProfiling, log));
    }

    pm.addPass(VPURT::createSplitControlGraphPass(options.controlGraphSplitBlockSize, log));

    if (!options.linearizeSchedule) {
        pm.addPass(VPUIP::createBarrierOptimizationPass(options.workloadManagementMode, log));
    }

    pm.addPass(
            VPURT::createSimplifySchedulePass(options.reduceParallelControlFlows, options.workloadManagementMode, log));

    auto dpuDryRunMode = VPU::getDPUDryRunMode(options.dpuDryRun);
    if (dpuDryRunMode == VPU::DPUDryRunMode::STRIP || options.shaveDryRun) {
        pm.addPass(VPUIP::createComputeTaskStrippingPass(log, dpuDryRunMode, options.shaveDryRun));
    }

    pm.addPass(VPUIP::createAddSwKernelInstructionPrefetchPass(log));

    if (options.workloadManagementMode == WorkloadManagementMode::FWLM_V1_PAGES) {
        pm.addPass(VPUIP::createPrepareShaveSubmitDMAsPass(log));
    }
    pm.addPass(VPUIP::createSplitLargeInvariantsPass(log));
    pm.addPass(VPURT::createInsertBarrierToMarkTheEndOfDescriptorGroupPass(options.workloadManagementMode, log));

    if (options.workloadManagementMode == WorkloadManagementMode::FWLM_V1_PAGES) {
        pm.addPass(VPUIP::createAddPlaceholderFetchDMAsPass(log));
    }
    if (options.workloadManagementMode == WorkloadManagementMode::PWLM_V0_1_PAGES) {
        pm.addPass(VPUIP::createAddPlaceholderFetchDMAsPWLMPass(log));
    }

    if (!isInliningRequired || !options.enableBarrierSchedWithFunctionOutlining) {
        // In case of outlining, final barrier is added after legalization in order to avoid its duplication.
        // Possible TODO for adding the barrier before legalization is to convert the pass to a Module pass.
        pm.addPass(VPURT::createAddFinalBarrierPass(options.workloadManagementMode, log));
    }
    VPURT::buildBarrierLegalizationPipeline(pm, /* workloadManagementEnabled */ true, /* unevenVariantSplitFlag */ true,
                                            log);

    if (isInliningRequired && options.enableBarrierSchedWithFunctionOutlining) {
        if (auto debatcherReorderingOptionsPtr = IE::DebatcherOpReorderingOptions::create(options, log)) {
            pm.addPass(IE::createRevertTileExecutorNumPass(*debatcherReorderingOptionsPtr, log));
        }
        pm.addPass(VPUIP::createDispatchedInlinerPass(log));
        pm.addPass(VPURT::createOptimizeSyncTasksPass(log));
        // In case of outlining, add final barrier after legalization in order to avoid duplication of the final barrier
        // attribute.
        pm.addPass(VPURT::createAddFinalBarrierPass(options.workloadManagementMode, log));
    }

    pm.addPass(VPUIP::createAddStartBarrierPass(options.workloadManagementMode, log));

    pm.addPass(VPURT::createWlmSplitGraphToPagesPass(log));
    // TODO: E#146544: Add a pass that will insert dummy tasks
    pm.addPass(VPURT::createWlmLegalizeSplitGraphToPagesPass(log));
    if (options.workloadManagementMode == WorkloadManagementMode::FWLM_V1_PAGES) {
        pm.addPass(VPURT::createWlmInsertDummyDmasInPagesPass(log));
    }
    if (options.workloadManagementBarrierProgrammingMode ==
        WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED) {
        pm.addPass(VPURT::createWlmLegalizePagesForBarrierDmasPass(log));
        pm.addPass(VPURT::createWlmInsertDummyBarriersInPagesPass(log));
    }

    if (options.enableCompressActivationSpill) {
        pm.addPass(VPUIP::createCompressSpillDmaPass(log));
    }

    pm.addPass(VPUIP::createDMAOutOfOrderOptimizationPass(options.workloadManagementMode, log));

    if (options.enableProfiling) {
        if (options.enableDPUProfiling) {
            pm.addPass(VPUIP::createConstantDpuProfHwpBasePass(log));
        }
        pm.addPass(VPUIP::createCaptureWorkpointPass(log));
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(Core::createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(options.workloadManagementMode, log));

    pm.addPass(VPURT::createOptimizeBarriersSlotsUsagePass(log));

    if (options.workloadManagementMode == WorkloadManagementMode::FWLM_V1_PAGES) {
        pm.addPass(VPURT::createFindWlmEnqueueDmasBarrierPass(log));
    }

    pm.addPass(VPUIP::createUpdateSwKernelParamsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableIntermediateBufferOutput) {
        pm.addPass(VPURT::createIntermediateBufferOutputPass(log));
    }

    if (options.workloadManagementMode == WorkloadManagementMode::PWLM_V0_1_PAGES) {
        pm.addPass(VPURT::createFindWlmEnqueueBarrierWithPagesPass(log));
    }

    if (options.workloadManagementMode == WorkloadManagementMode::FWLM_V1_PAGES) {
        pm.addPass(VPUIP::createInsertShaveSubmitSkipDMAsPass(log));
    }
    pm.addPass(
            VPURT::createInferenceExecutionAnalysisPass(options.scheduleTraceFile, options.enableScheduleTrace, log));
    pm.addPass(VPU::createCostModelAnalysisDestroyPass(log));
    if (options.enableDumpTaskStats) {
        // Force logging if dump-task-stats was enabled explicitly on the command line
        pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(
                log, options.enableDumpTaskStats.hasValue() && options.enableDumpTaskStats));
    }

    // [E170237] Temporary keep it under developer mode due to long compilation time
    if (isDeveloperBuild()) {
        // At the end of scheduling verify if WLM constraints are satisfied
        pm.addPass(VPURT::createCheckWlmPageSplitConstraintsPass(options.workloadManagementMode, log));
    }
}

void vpux::VPUIP::arch40xx::buildReferenceSWPipeline(mlir::OpPassManager& pm,
                                                     const VPUIP::arch40xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (options.enableShaveCodeGen) {
        ShaveCodeGen::buildShaveCodeGenPipelineVPUIP(pm);
    }

    pm.addPass(VPUIP::createSetMemorySpacePass(VPU::getMemKind<VPU::MemoryKind::DDR>,
                                               options.setMemorySpaceForFunctionBoundaries, log));

    pm.addPass(VPUIP::createAddCopyBetweenSWKernelsAndNetworkIOPass(log));

    pm.addPass(VPUIP::createCopyOpTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(log));
    }

    pm.addPass(VPUIP::createUngroupBoundedBuffersPass(log));

    pm.addPass(VPUIP::createConvertTransferOpsToDMAsPass(log));

    pm.addPass(VPUIP::createLegalizeStridedDMAsPass(log));

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(VPUIP::createLinearizationPass(log));
    pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));

    pm.addPass(VPUIP::createAddSwKernelCacheHandlingOpsPass(log));

    VPUIP::buildHardwareAdaptationPipeline(pm, log);

    pm.addPass(VPURT::createInsertBarrierToMarkTheEndOfDescriptorGroupPass(options.workloadManagementMode, log));

    if (options.workloadManagementMode == WorkloadManagementMode::FWLM_V1_PAGES) {
        pm.addPass(VPUIP::createAddPlaceholderFetchDMAsPass(log));
    }
    if (options.workloadManagementMode == WorkloadManagementMode::PWLM_V0_1_PAGES) {
        pm.addPass(VPUIP::createAddPlaceholderFetchDMAsPWLMPass(log));
    }

    pm.addPass(VPURT::createAddFinalBarrierPass(options.workloadManagementMode, log));
    VPURT::buildBarrierLegalizationPipeline(pm, /* workloadManagementEnabled */ true, /* unevenVariantSplitFlag */ true,
                                            log);
    pm.addPass(VPUIP::createAddStartBarrierPass(options.workloadManagementMode, log));

    // Level 1 : VPU RunTime

    if (options.workloadManagementMode == WorkloadManagementMode::PWLM_V0_1_PAGES) {
        pm.addPass(VPURT::createWlmSplitGraphToPagesPass(log));
        pm.addPass(VPURT::createWlmLegalizeSplitGraphToPagesPass(log));
    }

    if (options.enableProfiling) {
        pm.addPass(VPUIP::createCaptureWorkpointPass(log));
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(Core::createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(options.workloadManagementMode, log));

    pm.addPass(VPURT::createOptimizeBarriersSlotsUsagePass(log));

    pm.addPass(VPUIP::createUpdateSwKernelParamsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.workloadManagementMode == WorkloadManagementMode::PWLM_V0_1_PAGES) {
        pm.addPass(VPURT::createFindWlmEnqueueBarrierWithPagesPass(log));
    }
}

void vpux::VPUIP::arch40xx::registerVPUIPPipelines() {
    mlir::PassPipelineRegistration<VPUIP::arch40xx::MemoryAllocationOptions>(
            "memory-allocation", "Memory Allocation",
            [](mlir::OpPassManager& pm, const VPUIP::arch40xx::MemoryAllocationOptions& options) {
                VPUIP::arch40xx::buildMemoryAllocationPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<VPUIP::arch40xx::DefaultHWOptions>(
            "default-hw-mode-vpuip", "VPUIP dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const VPUIP::arch40xx::DefaultHWOptions& options) {
                VPUIP::arch40xx::buildDefaultHWPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<VPUIP::arch40xx::DefaultHWOptions>(
            "reference-sw-mode-vpuip", "VPUIP dialect part of reference SW pipeline",
            [](mlir::OpPassManager& pm, const VPUIP::arch40xx::DefaultHWOptions& options) {
                VPUIP::arch40xx::buildReferenceSWPipeline(pm, options);
            });
}
