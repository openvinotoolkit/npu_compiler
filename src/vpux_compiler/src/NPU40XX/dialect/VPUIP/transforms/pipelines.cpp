//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
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
            options.enablePipelining, options.enablePrefetching, options.optimizeFragmentation,
            options.optimizeDynamicSpilling, log));

    if (options.enableCompressActivationSpill) {
        pm.addPass(VPUIP::createAdjustSpillSizePass(log));
    }

    if (options.enableGroupAsyncExecuteOps) {
        pm.addPass(VPUIP::createGroupAsyncExecuteOpsPass(log));
    }

    pm.addPass(VPUIP::createQueryArgsAllocationAnalysisPass());
    pm.addPass(VPUIP::createUngroupBoundedBuffersPass(log));
    pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));
}

void vpux::VPUIP::arch40xx::buildDefaultHWPipeline(mlir::OpPassManager& pm,
                                                   const VPUIP::arch40xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (options.enableShaveCodeGen) {
        VPUIP::buildShaveCodeGenPipeline(pm);
    }

    if (options.enableShaveKernelTiling) {
        pm.addPass(VPUIP::createTileActShaveKernelTaskPass(log));
    }
    if (options.enableOptimizeCopies || options.enableOpsAsDMA) {
        // This pass is a part of "copy optimization pipeline", but need to be done before because
        // WrapWithPermuteAsNNDMA depends on it.
        pm.addPass(VPUIP::createMovePureViewOpBeforeCopyPass(log));
    }
    if (options.enableOpsAsDMA) {
        pm.addPass(VPUIP::createWrapWithPermuteAsNNDMAPass(log));
    }
    pm.addPass(VPUIP::createOptimizeExpandSubviewPass(log));
    pm.addPass(VPUIP::createConvertExpandPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPUIP::createConvertEltwiseToInPlacePass(log));

    // Level 2 : Abstract RunTime

    pm.addPass(VPUIP::createSetMemorySpacePass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));

    if (options.enableSEPtrsOperations || options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPUIP::createMoveSubViewBeforeSparseBufferPass(log));
        pm.addPass(VPUIP::createComputeSEBasePtrsPass(log));
        pm.addPass(VPUIP::createConvertSETablesToConstantsPass(log));
    }
    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createPropagateSparsityCompressionPass(log));
    }
    if (options.enableWeightsSparsity || VPU::isActSparsityEnabled(options.enableActivationSparsity)) {
        pm.addPass(VPUIP::createUngroupSparseBuffersPass(log));
    }

    pm.addPass(VPUIP::createUngroupBoundedBuffersPass(log));

    VPUIP::arch37xx::buildOptimizeCopiesPipeline(pm, VPUIP::arch37xx::OptimizeCopiesOptions(options), log);

    pm.addPass(VPUIP::createConvertDynamicReshapeToInPlacePass(log));
    pm.addPass(VPUIP::createInsertCopyForEltwiseInPlaceInputPass(log));
    pm.addPass(VPUIP::arch40xx::createOptimizeConvertDMAOpPass(log));

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

    if (options.enableConstantFusion) {
        pm.addPass(VPUIP::createFuseConstantsPass(log));
    }

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

    if (options.enableProfiling && options.enableDPUProfiling) {
        pm.addPass(VPUIP::createDPUProfilingPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    VPUIP::buildAsyncSchedulingPipeline(pm, log);
    if (options.enableAsyncRegionOutlining) {
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

    pm.addPass(VPUIP::arch37xx::createAddSwKernelCacheHandlingOpsPass(log));

    VPUIP::buildHardwareAdaptationPipeline(pm, log);

    // Level 1 : VPU RunTime
    pm.addPass(VPUIP::createUnrollSwKernelPass(log));

    pm.addPass(VPUIP::arch40xx::createUnrollDistributedOpsPass(log, options.enableSegmentedDmaFusion));
    pm.addPass(VPUIP::createBatchMatMulToMatMulPass(log));
    pm.addPass(VPUIP::arch40xx::createDetectDMASplitCandidatePass(log));
    pm.addPass(VPUIP::createNNDMATilingPass(log));
    pm.addPass(VPUIP::createSegmentHalosPass(log));
    pm.addPass(VPUIP::arch40xx::createComputeHaloRegionForDPUTaskOpPass(log));

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

    if (options.enableDpuFromShaveControl) {
        pm.addPass(VPUIP::createSyncShvDpuPass(log));
    }

    if (options.enableWeightsSwizzling || options.enableActivationSwizzling) {
        pm.addPass(Const::createApplySwizzlingPass());
        pm.addPass(VPUIP::createResolveDMAWithSwizzlingPass(log));
    }

    if (options.enableCompressWeightsBTC) {
        pm.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }

    pm.addPass(VPUIP::arch40xx::createSplitDMAToBalanceLoadPass(log));
    if (options.enableSegmentedDmaFusion) {
        pm.addPass(VPUIP::arch40xx::createFuseSegmentedDmaPass(log));
    }

    // TODO: E#140041 enable profiling with outlining
    bool isOutliningEnabled = (options.functionOutlining.hasValue() || options.enableVerticalFusionOutlining) &&
                              (!options.enableProfiling || options.enableProfilingWithOutlining);

    if (isOutliningEnabled) {
        if (options.enableBarrierSchedWithFunctionOutlining) {
            pm.addPass(VPURT::createInsertSyncTasksPass(log));
        } else {
            if (auto debatcherReorderingOptionsPtr = IE::DebatcherOpReorderingOptions::create(options, log)) {
                pm.addPass(IE::createRevertTileExecutorNumPass(*debatcherReorderingOptionsPtr, log));
            }
            pm.addPass(VPUIP::createDispatchedInlinerPass(log));
        }
    }

    if (options.enableProfiling) {
        pm.addPass(VPUIP::arch40xx::createDMATaskProfilingHwDdrPass(options.enableDMAProfiling, log));
    }

    if (options.enableControlGraphSplit) {
        pm.addPass(VPURT::createSplitControlGraphPass(options.controlGraphSplitBlockSize, log));
    }

    if (!options.linearizeSchedule) {
        pm.addPass(VPUIP::createDMABarrierOptimizationPass(log));
    }

    if (options.enableSimpleSchedule) {
        pm.addPass(VPURT::createSimplifySchedulePass(options.reduceParallelControlFlows, options.workloadManagementMode,
                                                     log));
    }

    auto dpuDryRunMode = VPU::getDPUDryRunMode(options.dpuDryRun);
    if (dpuDryRunMode == VPU::DPUDryRunMode::STRIP || options.shaveDryRun == true) {
        pm.addPass(VPUIP::arch40xx::createComputeTaskStrippingPass(log, dpuDryRunMode, options.shaveDryRun));
    }

    if (options.enableSWKernelInstructionPrefetch) {
        pm.addPass(vpux::VPUIP::createAddSwKernelInstructionPrefetchPass(log));
    }

    // Ensures legal schedule in the case of a WLM rollback
    pm.addPass(VPURT::createInsertBarrierToMarkTheEndOfDescriptorGroupPass(
            options.workloadManagementBarrierCountThreshold, options.workloadManagementMode, log));

    if (options.workloadManagementEnable) {
        pm.addPass(VPUIP::arch40xx::createLegalizeScheduleForPartialWlmFetchDmasPass(
                options.workloadManagementBarrierCountThreshold, log));
    }

    if (!isOutliningEnabled || !options.enableBarrierSchedWithFunctionOutlining) {
        // In case of outlining, final barrier is added after legalization in order to avoid its duplication.
        // Possible TODO for adding the barrier before legalization is to convert the pass to a Module pass.
        pm.addPass(VPURT::createAddFinalBarrierPass(options.workloadManagementMode, log));
    }
    VPURT::buildBarrierLegalizationPipeline(pm, options.workloadManagementBarrierCountThreshold,
                                            options.workloadManagementMode,
                                            /* unevenVariantSplitFlag */ true, log);

    if (isOutliningEnabled && options.enableBarrierSchedWithFunctionOutlining) {
        if (auto debatcherReorderingOptionsPtr = IE::DebatcherOpReorderingOptions::create(options, log)) {
            pm.addPass(IE::createRevertTileExecutorNumPass(*debatcherReorderingOptionsPtr, log));
        }
        pm.addPass(VPUIP::createDispatchedInlinerPass(log));
        pm.addPass(VPURT::createOptimizeSyncTasksPass(log));
        // In case of outlining, add final barrier after legalization in order to avoid duplication of the final barrier
        // attribute.
        pm.addPass(VPURT::createAddFinalBarrierPass(options.workloadManagementMode, log));
    }

    pm.addPass(VPUIP::arch40xx::createAddStartBarrierPass(options.workloadManagementMode, log));

    if (options.workloadManagementEnable && options.workloadManagementMode >= WorkloadManagementMode::PWLM_V2_PAGES) {
        pm.addPass(VPURT::createWlmSplitGraphToPagesPass(log));
        // TODO: E#146544: Add a pass that will insert dummy tasks
        pm.addPass(VPURT::createWlmLegalizeSplitGraphToPagesPass(log));
        if (options.workloadManagementBarrierProgrammingMode ==
            WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED) {
            pm.addPass(VPURT::createWlmLegalizePagesForBarrierDmasPass(log));
        }
        if (options.workloadManagementBarrierProgrammingMode ==
            WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED) {
            pm.addPass(VPURT::createWlmInsertDummyBarriersInPagesPass(log));
        }
    }

    if (options.enableCompressActivationSpill) {
        pm.addPass(VPUIP::arch40xx::createCompressSpillDmaPass(log));
    }

    if (options.enableDmaOutOfOrder) {
        pm.addPass(VPUIP::arch40xx::createDMAOutOfOrderOptimizationPass(options.workloadManagementMode, log));
    }

    if (options.enableProfiling) {
        if (options.enableDPUProfiling) {
            pm.addPass(VPUIP::arch40xx::createConstantDpuProfHwpBasePass(log));
        }
        pm.addPass(VPUIP::createCaptureWorkpointPass(log));
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(Core::createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(options.enableColorBinPhysicalBarrierAssignment,
                                                       options.workloadManagementMode,
                                                       options.workloadManagementBarrierCountThreshold, log));

    if (options.workloadManagementEnable && options.workloadManagementMode >= WorkloadManagementMode::PWLM_V2_PAGES) {
        pm.addPass(VPURT::createOptimizeBarriersSlotsUsagePass(log));
    }

    if (!options.workloadManagementEnable || options.workloadManagementMode <= WorkloadManagementMode::PWLM_V2_PAGES) {
        // BarrierSimulation pass performs final verification of barrier schedule but does not change the IR.
        // To save compilation time it may be skipped - TODO E#178177
        // For FWLM skip this pass as FWLM passes use barrier consumption order which is not fully supported by this
        // pass
        pm.addPass(VPURT::createBarrierSimulationPass(log));
    }
    pm.addPass(VPUIP::createUpdateSwKernelParamsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableIntermediateBufferOutput) {
        pm.addPass(VPURT::createIntermediateBufferOutputPass(log));
    }

    if (options.enableActivityFactor || options.enableScheduleTrace) {
        pm.addPass(VPURT::createInferenceExecutionAnalysisPass(options.scheduleTraceFile, options.enableScheduleTrace,
                                                               options.enableActivityFactor, log));
    }
    pm.addPass(VPU::createCostModelAnalysisDestroyPass(log));
    if (options.enableDumpTaskStats) {
        // Force logging if dump-task-stats was enabled explicitly on the command line
        pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(
                log, options.enableDumpTaskStats.hasValue() && options.enableDumpTaskStats));
    }

    // [E170237] Temporary keep it under developer mode due to long compilation time
    if (isDeveloperBuild()) {
        // At the end of scheduling verify if WLM constraints are satisfied
        if (options.workloadManagementEnable &&
            options.workloadManagementMode >= WorkloadManagementMode::PWLM_V2_PAGES) {
            pm.addPass(VPURT::createCheckWlmPageSplitConstraintsPass(options.workloadManagementMode, log));
        }
    }
}

void vpux::VPUIP::arch40xx::buildReferenceSWPipeline(mlir::OpPassManager& pm,
                                                     const VPUIP::arch40xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (options.enableShaveCodeGen) {
        VPUIP::buildShaveCodeGenPipeline(pm);
    }

    pm.addPass(VPUIP::createSetMemorySpacePass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));

    pm.addPass(VPUIP::createAddCopyBetweenSWKernelsAndNetworkIOPass(log));

    pm.addPass(VPUIP::createCopyOpTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    pm.addPass(VPUIP::createUngroupBoundedBuffersPass(log));

    pm.addPass(VPUIP::createConvertTransferOpsToDMAsPass(log));

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(VPUIP::createLinearizationPass(log));
    pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));

    pm.addPass(VPUIP::arch37xx::createAddSwKernelCacheHandlingOpsPass(log));

    VPUIP::buildHardwareAdaptationPipeline(pm, log);

    pm.addPass(VPUIP::arch40xx::createAddStartBarrierPass(options.workloadManagementMode, log));
    pm.addPass(VPURT::createAddFinalBarrierPass(options.workloadManagementMode, log));

    // Level 1 : VPU RunTime

    if (options.enableProfiling) {
        pm.addPass(VPUIP::createCaptureWorkpointPass(log));
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(Core::createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(options.enableColorBinPhysicalBarrierAssignment, std::nullopt,
                                                       std::nullopt, log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createUpdateSwKernelParamsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::VPUIP::arch40xx::registerVPUIPPipelines() {
    mlir::PassPipelineRegistration<VPUIP::arch37xx::OptimizeCopiesOptions>(
            "optimize-copies-pipeline", "Optimize Copies Pipeline",
            [](mlir::OpPassManager& pm, const VPUIP::arch37xx::OptimizeCopiesOptions& options) {
                VPUIP::arch37xx::buildOptimizeCopiesPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<VPUIP::arch40xx::MemoryAllocationOptions>(
            "memory-allocation", "Memory Allocation",
            [](mlir::OpPassManager& pm, const VPUIP::arch40xx::MemoryAllocationOptions& options) {
                VPUIP::arch40xx::buildMemoryAllocationPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<>("dma-unrolling", "DMA unrolling", [](mlir::OpPassManager& pm) {
        VPUIP::buildDMAUnrollingPipeline(pm);
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
