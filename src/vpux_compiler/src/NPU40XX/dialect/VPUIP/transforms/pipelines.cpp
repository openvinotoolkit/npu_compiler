//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPURT/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/profiling/common.hpp"

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
    pm.addPass(mlir::createCanonicalizerPass(grc));

    VPUIP::arch37xx::buildOptimizeCopiesPipeline(pm, VPUIP::arch37xx::OptimizeCopiesOptions(options), log);

    pm.addPass(VPUIP::createInsertCopyForEltwiseInPlaceInputPass(log));
    pm.addPass(VPUIP::arch40xx::createOptimizeConvertDMAOpPass(log));

    if (options.enableOpsAsDMA) {
        pm.addPass(VPUIP::createConvertToDMAPass(log));
    }

    pm.addPass(VPUIP::createAddCopyBetweenSWKernelsAndNetworkIOPass(log));
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

    if (options.enableCompressActivationSpill) {
        pm.addPass(VPUIP::createCompressDmaReserveMemPass(log));
    }

    auto profilingMode = getDMAProfilingMode(VPU::ArchKind::NPU40XX, options.enableDMAProfiling.getValue());
    pm.addPass(VPUIP::createDMATaskProfilingReserveMemPass(profilingMode, log));

    if (options.enableSWKernelPrefetchingReserveMem) {
        pm.addPass(VPUIP::createSWKernelPrefetchingReserveMemPass(log));
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

    pm.addPass(VPUIP::arch40xx::createUnrollClusterTilingPass(log, options.enableSegmentedDmaFusion));
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

    VPUIP::arch40xx::buildDMAUnrollingPipeline(pm, log);

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

    if (options.enableCompressActivationSpill) {
        pm.addPass(VPUIP::arch40xx::createCompressSpillDmaPass(log));
    }

    // TODO: E#140041 enable profiling with outlining
    bool isOutliningEnabled =
            (options.functionOutlining.hasValue() || options.enableVerticalFusionOutlining) && !options.enableProfiling;

    if (isOutliningEnabled) {
        if (options.enableBarrierSchedWithFunctionOutlining) {
            pm.addPass(VPURT::arch40xx::createInsertSyncTasksPass(log));
        } else {
            if (auto debatcherReorderingOptionsPtr = IE::DebatcherOpReorderingOptions::create(options, log)) {
                pm.addPass(IE::createRevertTileExecutorNumPass(*debatcherReorderingOptionsPtr, log));
            }
            pm.addPass(mlir::createInlinerPass());
        }
    }

    if (options.enableProfiling) {
        auto dmaProfilingMode = getDMAProfilingMode(VPU::ArchKind::NPU40XX, options.enableDMAProfiling.getValue());
        pm.addPass(VPUIP::arch40xx::createDMATaskProfilingHwDdrPass(dmaProfilingMode, log));
    }

    if (options.enableControlGraphSplit) {
        pm.addPass(VPURT::createSplitControlGraphPass(options.controlGraphSplitBlockSize, log));
    }

    if (!options.linearizeSchedule) {
        pm.addPass(VPUIP::createDMABarrierOptimizationPass(log));
    }

    if (options.enableSimpleSchedule) {
        pm.addPass(VPURT::createSimplifySchedulePass(options.shareWaitAndUpdateBarriers,
                                                     options.reduceParallelControlFlows, log));
    }

    auto dpuDryRunMode = VPU::getDPUDryRunMode(options.dpuDryRun);
    if (dpuDryRunMode == VPU::DPUDryRunMode::STRIP || options.shaveDryRun == true) {
        pm.addPass(VPUIP::arch40xx::createComputeTaskStrippingPass(log, dpuDryRunMode, options.shaveDryRun));
    }

    // Ensures legal schedule in the case of a WLM rollback
    pm.addPass(VPURT::createInsertBarrierToMarkTheEndOfDescriptorGroupPass(
            options.workloadManagementBarrierCountThreshold, options.workloadManagementMode, log));

    if (options.workloadManagementEnable) {
        pm.addPass(VPUIP::arch40xx::createLegalizeScheduleForWlmFetchDmasPass(
                options.workloadManagementBarrierCountThreshold, options.workloadManagementMode, log));
    }

    VPURT::buildBarrierLegalizationPipeline(pm, options.workloadManagementBarrierCountThreshold,
                                            options.workloadManagementMode,
                                            /* unevenVariantSplitFlag */ true, log);

    if (isOutliningEnabled && options.enableBarrierSchedWithFunctionOutlining) {
        if (auto debatcherReorderingOptionsPtr = IE::DebatcherOpReorderingOptions::create(options, log)) {
            pm.addPass(IE::createRevertTileExecutorNumPass(*debatcherReorderingOptionsPtr, log));
        }
        pm.addPass(mlir::createInlinerPass());
        pm.addPass(VPURT::arch40xx::createOptimizeSyncTasksPass(log));
    }

    pm.addPass(VPUIP::arch40xx::createAddStartBarrierPass(log));
    pm.addPass(VPURT::arch37xx::createAddFinalBarrierPass(log));

    if (options.enableDmaOutOfOrder) {
        pm.addPass(VPUIP::arch40xx::createDMAOutOfOrderOptimizationPass(log));
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
                                                       options.workloadManagementBarrierCountThreshold, log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createUpdateSwKernelParamsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableIntermediateBufferOutput) {
        pm.addPass(VPURT::createIntermediateBufferOutputPass(log));
    }

    if (options.enableActivityFactor || options.enableScheduleTrace) {
        pm.addPass(VPURT::createInferenceExecutionAnalysisPass(options.scheduleTraceFile, options.enableScheduleTrace,
                                                               options.enableActivityFactor, log));
    }
    if (options.enableDumpTaskStats) {
        // Force logging if dump-task-stats was enabled explicitly on the command line
        pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(
                log, options.enableDumpTaskStats.hasValue() && options.enableDumpTaskStats));
    }
}

//
// DMAUnrollingPipeline
//

void vpux::VPUIP::arch40xx::buildDMAUnrollingPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(VPUIP::createUnrollDMAAnalysisPass(log));
    pm.addPass(VPUIP::arch40xx::createUnrollDepthToSpaceDMAPass(log));
    pm.addPass(VPUIP::arch40xx::createUnrollSpaceToDepthDMAPass(log));
    pm.addPass(VPUIP::createUnrollPermuteToNNDMAPass(log));

    pm.addPass(VPUIP::createUnrollUpsamplingDMAPass(log));
    pm.addPass(VPUIP::createUnrollExpandDMAPass(log));
    pm.addPass(VPUIP::createUnrollPerAxisTileDMAPass(log));
    pm.addPass(VPUIP::createInvalidateUnrollDMAAnalysisPass(log));
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
        VPUIP::arch40xx::buildDMAUnrollingPipeline(pm);
    });

    mlir::PassPipelineRegistration<VPUIP::arch40xx::DefaultHWOptions>(
            "default-hw-mode-vpuip", "VPUIP dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const VPUIP::arch40xx::DefaultHWOptions& options) {
                VPUIP::arch40xx::buildDefaultHWPipeline(pm, options);
            });
}
