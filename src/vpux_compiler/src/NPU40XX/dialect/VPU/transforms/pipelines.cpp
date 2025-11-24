//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::VPU::arch40xx::buildIncrementalPipeline(mlir::OpPassManager& pm, const vpux::MCAndTilingOptionsBase& options,
                                                   Logger log) {
    pm.addPass(VPU::createDecomposeMVNPass(log));

    pm.addPass(VPU::createMultiClusterStrategyAssignmentPass(options.enablePrefetching, options.opTilingCacheThreshold,
                                                             options.mcOptimizationScope, log));
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
    pm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ false,
                                             /*disablePassOnEntryFunctionForHostCompile=*/false, log));

    VPU::buildTilingPipeline(pm, VPU::TilingOptions(options), log);

    if (options.enableScfComputeOpsOutlining) {
        VPU::buildScfComputeOpsOutliningPipeline(pm, log);
    }

    pm.addPass(VPU::createBoundedTensorsToDynamicDimsMaskPass(log));
    pm.addPass(VPU::createMakeOpsWithDistributedTensorPass(options.enableExplicitDistributionInfoAttr, log));

    pm.addPass(VPU::createComputeInterpolateCoordinatesPass(options.enableExplicitDistributionInfoAttr, log));
    // E#183249 - reintroduce RemoveOutputSparseToAvoidSuboptimalDPUWorkloads pass after perf regression
    // is investigated

    pm.addPass(VPU::createMakeDistributedCopiesPass(log));
    pm.addPass(VPU::createAdjustDistributedTensorAroundOpsPass(log));
}

//
// DefaultHWPipeline
//

void vpux::VPU::arch40xx::buildDefaultHWPipeline(mlir::OpPassManager& pm,
                                                 const VPU::arch40xx::DefaultHWOptions& options, Logger log) {
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

    // Unconditional on NPU40xx due to DMA HWP scratch range requirement
    pm.addPass(VPU::createDMATaskProfilingReserveMemPass(
            options.enableProfiling ? options.enableDMAProfiling.getValue() : "false", log));

    // Make sure to run this after SWKernelDataPrefetchReserveMem which ensures we have enough
    // memory at the end of CMX to allow SW kernel data prefetch.
    if (options.enableSWKernelInstructionPrefetch) {
        pm.addPass(VPU::createSWKernelInstructionPrefetchReserveMemForDummyKernelsPass(log));
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
    pm.addPass(VPU::createAddSwOpAuxiliaryBufferPass(log));
    pm.addPass(VPU::createAdjustLSTMCellInputsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

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
                                                           /*skipNonConvOC=*/false, log));
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

    if (options.enableM2I) {
        pm.addPass(VPU::createFuseM2IOpsPass(log));
        pm.addPass(VPU::createConvertM2IOpsPass(log));
    }

    pm.addPass(VPU::createCostModelAnalysisConstructPass(log));
    if (options.enableSMPipeline) {
        VPU::buildSMPipeline(pm, vpux::MCAndTilingOptionsBase(options), log);
    } else {
        VPU::arch40xx::buildIncrementalPipeline(pm, vpux::MCAndTilingOptionsBase(options), log);
    }

    pm.addPass(VPU::createAdjustMemorySpacePass(log));
    pm.addPass(VPU::createOptimizeSharedInputCopyForConcatPass(log));
    pm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ false,
                                             options.disablePassOnEntryFunctionForHostCompile, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createCMXConcatPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(VPU::createMoveReflectPadToCMXPass(log));

    pm.addPass(VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    pm.addPass(VPU::createCorrectNCEWorkloadsPass(log));
    pm.addPass(VPU::createResolveEltwiseWithZTiledWorkloadsPass(log));
    pm.addPass(VPU::createComputeNCEInputWorkloadsPass(log));
    pm.addPass(VPU::createShiftOutputWorkloadsForHaloPass(log));
    if (options.enableEntireMainContentOutlining && canOutlineFromProfilingPerspective(options)) {
        pm.addPass(VPU::createOutlineEntireMainContentPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));
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
}
