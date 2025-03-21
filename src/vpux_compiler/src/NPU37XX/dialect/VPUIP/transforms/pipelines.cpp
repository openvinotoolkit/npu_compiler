//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/profiling/common.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::VPUIP::arch37xx::buildOptimizeCopiesPipeline(mlir::OpPassManager& pm,
                                                        const VPUIP::arch37xx::OptimizeCopiesOptions& options,
                                                        Logger log) {
    pm.addPass(VPUIP::createUnwrapClusterTilingPass(log));
    if (options.enableOptimizeCopies) {
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
}

void vpux::VPUIP::arch37xx::buildMemoryAllocationPipeline(mlir::OpPassManager& pm,
                                                          const VPUIP::arch37xx::MemoryAllocationOptions& options,
                                                          Logger log) {
    pm.addPass(VPUIP::createFeasibleAllocationPass(
            VPU::getMemKind<VPU::MemoryKind::CMX_NN>, VPU::getMemKind<VPU::MemoryKind::DDR>, options.linearizeSchedule,
            options.enablePipelining, options.enablePrefetching, options.optimizeFragmentation,
            options.optimizeDynamicSpilling, log));

    if (options.enableGroupAsyncExecuteOps) {
        pm.addPass(VPUIP::createGroupAsyncExecuteOpsPass(log));
    }

    pm.addPass(VPUIP::createQueryArgsAllocationAnalysisPass());
    pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));
}

void vpux::VPUIP::arch37xx::buildDefaultHWPipeline(mlir::OpPassManager& pm,
                                                   const VPUIP::arch37xx::DefaultHWOptions& options, Logger log) {
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
    pm.addPass(VPUIP::createSetMemorySpacePass(vpux::VPU::getMemKind<VPU::MemoryKind::DDR>, log));

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

    pm.addPass(VPUIP::createSwizzlingPass(options.enableWeightsSwizzling, options.enableActivationSwizzling, log));

    // Note: this pass introduces necessary VPUIP.Copy operations, thus, it must
    // be called *after* all copy optimizations are run (to ensure the
    // introduced copies are not optimized out).
    pm.addPass(VPUIP::createLegalizeRepeatingFuncCallsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPUIP::createConvertTransferOpsToDMAsPass(log));

    if (options.enableProfiling && options.enableDPUProfiling) {
        pm.addPass(VPUIP::createDPUProfilingPass(vpux::VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(vpux::VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    if (options.enableProfiling) {
        auto dmaProfilingMode = getDMAProfilingMode(VPU::ArchKind::NPU37XX, options.enableDMAProfiling.getValue());
        pm.addPass(VPUIP::createDMATaskProfilingReserveMemPass(dmaProfilingMode, log));
    }

    if (options.enableSWKernelPrefetchingReserveMem) {
        pm.addPass(VPUIP::createSWKernelPrefetchingReserveMemPass(log));
    }

    pm.addPass(VPUIP::createCalculateAsyncRegionCycleCostPass(log));

    VPUIP::arch37xx::buildMemoryAllocationPipeline(pm, VPUIP::arch37xx::MemoryAllocationOptions(options), log);

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

    pm.addPass(VPUIP::arch37xx::createUnrollClusterTilingPass(log));
    pm.addPass(VPUIP::createNNDMATilingPass(log));
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

    VPUIP::arch37xx::buildDMAUnrollingPipeline(pm, log);

    // TODO: E#140041 enable profiling with outlining
    bool isOutliningEnabled =
            (options.functionOutlining.hasValue() || options.enableVerticalFusionOutlining) && !options.enableProfiling;

    // TODO: E#118869 For now put the pass before barrier scheduling
    if (isOutliningEnabled) {
        pm.addPass(mlir::createInlinerPass());
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

    pm.addPass(VPURT::createInsertBarrierToMarkTheEndOfDescriptorGroupPass(std::nullopt, std::nullopt, log));

    VPURT::buildBarrierLegalizationPipeline(pm, std::nullopt, std::nullopt,
                                            /* unevenVariantSplitFlag */ false, log);

    pm.addPass(VPURT::arch37xx::createAddFinalBarrierPass(log));

    pm.addPass(Const::createApplySwizzlingPass());

    pm.addPass(VPUIP::createResolveDMAWithSwizzlingPass(log));

    if (options.enableCompressWeightsBTC) {
        pm.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }

    if (options.enableProfiling) {
        auto dmaProfilingMode = getDMAProfilingMode(VPU::ArchKind::NPU37XX, options.enableDMAProfiling.getValue());
        pm.addPass(VPUIP::createDMATaskProfilingAfterBarrierSchedPass(dmaProfilingMode, log));
        pm.addPass(VPUIP::createCaptureWorkpointPass(log));
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(Core::createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(options.enableColorBinPhysicalBarrierAssignment, std::nullopt,
                                                       log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createUpdateSwKernelParamsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(Const::createConstantFoldingPass());

    // TODO: #-120399 This is a temporary solution to remove strides from const.declare operations. Ideally,
    // this would be done by a custom canonicalizer by matching the different dialect's subview operations
    // and their constant inputs. Strides in constants should have never reached this point in the first place!
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

void vpux::VPUIP::arch37xx::buildDMAUnrollingPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(VPUIP::createUnrollDMAAnalysisPass(log));
    pm.addPass(VPUIP::arch37xx::createUnrollDepthToSpaceDMAPass(log));
    pm.addPass(VPUIP::arch37xx::createUnrollSpaceToDepthDMAPass(log));
    pm.addPass(VPUIP::createUnrollPermuteToNNDMAPass(log));

    pm.addPass(VPUIP::createUnrollUpsamplingDMAPass(log));
    pm.addPass(VPUIP::createUnrollExpandDMAPass(log));
    pm.addPass(VPUIP::createUnrollPerAxisTileDMAPass(log));
    pm.addPass(VPUIP::createInvalidateUnrollDMAAnalysisPass(log));
}

void vpux::VPUIP::arch37xx::registerVPUIPPipelines() {
    mlir::PassPipelineRegistration<VPUIP::arch37xx::OptimizeCopiesOptions>(
            "optimize-copies-pipeline", "Optimize Copies Pipeline",
            [](mlir::OpPassManager& pm, const VPUIP::arch37xx::OptimizeCopiesOptions& options) {
                VPUIP::arch37xx::buildOptimizeCopiesPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<VPUIP::arch37xx::MemoryAllocationOptions>(
            "memory-allocation", "Memory Allocation",
            [](mlir::OpPassManager& pm, const VPUIP::arch37xx::MemoryAllocationOptions& options) {
                VPUIP::arch37xx::buildMemoryAllocationPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<>("dma-unrolling", "DMA unrolling", [](mlir::OpPassManager& pm) {
        VPUIP::arch37xx::buildDMAUnrollingPipeline(pm);
    });

    mlir::PassPipelineRegistration<VPUIP::arch37xx::DefaultHWOptions>(
            "default-hw-mode-vpuip", "VPUIP dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const VPUIP::arch37xx::DefaultHWOptions& options) {
                VPUIP::arch37xx::buildDefaultHWPipeline(pm, options);
            });
}
