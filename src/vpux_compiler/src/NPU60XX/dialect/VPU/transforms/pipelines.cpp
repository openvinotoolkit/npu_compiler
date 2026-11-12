//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/Shave/transforms/passes.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// NCEWorkloadsPipeline
//
// Splits NCE operations onto DPU workloads and legalizes them. When SCF tiling is enabled the workloads are
// generated on the still-rolled loops and the loops are fully unrolled only after all workload legalization
// passes have run, so that every unrolled tile inherits an already correct workload.
//

void vpux::VPU::arch60xx::buildNCEWorkloadsPipeline(mlir::OpPassManager& pm, const VPU::NCEWorkloadsOptions& options,
                                                    Logger log) {
    if (options.enableSCFTiling) {
        pm.addPass(VPU::createWorkloadsForNCEOpsSCFPass(log));
    }

    pm.addPass(VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    pm.addPass(VPU::createCorrectNCEWorkloadsPass(log));
    pm.addPass(VPU::createCreateNewWeightTablesDataPass(log));
    pm.addPass(VPU::createComputeNCEInputWorkloadsPass(log));
    pm.addPass(VPU::createShiftOutputWorkloadsForHaloPass(log));

    if (options.enableSCFTiling) {
        pm.addPass(VPU::createFullUnrollSCFLoopPass(log));
    }
}

//
// DefaultHWPipeline
//

void vpux::VPU::arch60xx::buildDefaultHWPipeline(mlir::OpPassManager& pm,
                                                 const VPU::arch60xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Custom Kernel scratch and replication handling
    pm.addPass(Shave::createMaterializeScratchBuffersPass(log));
    pm.addPass(Shave::createHandleExternalKernelTilingRequirementsPass(log));

    if (options.enableConcatRepeatingBlockOutlining) {
        pm.addPass(VPU::createConcatRepeatingBlocksOutliningPass(
                options.concatRepeatingBlockOutliningSeqLength, options.concatRepeatingBlockOutliningSdpaMinSeqLength,
                options.concatRepeatingBlockOutliningSdpaHeadThreshold, log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }

    pm.addPass(VPU::createAdjustForOptimizedLayersPass(log));
    pm.addPass(VPU::createConvertTileWithEltwiseToNCEPoolPass(log));
    pm.addPass(VPU::createConvertOpToDMAForPerformantExecutionPass(log));
    pm.addPass(VPU::createMoveConvertAroundViewLikeOpsPass(log));
    pm.addPass(VPU::createDetectionOutputDecompositionPass(log));
    pm.addPass(VPU::createSplitRealDFTOpsPass(log));
    pm.addPass(VPU::createAdjustLSTMCellInputsPass(log));

    if (options.enableSEPtrsOperations || options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPU::createSplitSEOpsPass(log));
        pm.addPass(VPU::createLowerOpsToSENCEPass(log));
    }

    if (options.enableSoftmaxDecomposition) {
        pm.addPass(VPU::arch50xx::createDecomposeSoftmaxInSdpaPass(log));
    }
    pm.addPass(VPU::arch60xx::createExtractReduceMaxFromExpStabilizedPass(log));
    pm.addPass(VPU::createFuseReduceMinMaxToDpuPass(log));

    pm.addPass(VPU::createFuseConvertPass(log));

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(options.enableOutputEnsurance,
                                                           options.enableDequantWeightEnsuranceBeforeStrategy,
                                                           /*skipConvOC=*/SkipOCMode::SKIP_LARGE_SPATIAL,
                                                           /*skipEltwiseOC=*/SkipOCMode::SKIP_ALL,
                                                           /*enableSplitChannelForDynamicDequantize=*/true, log));
    pm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ false, log));

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
                                                /*enableLegacyConvSplitOverIC=*/false);
    }

    auto& nestedPm = options.enableScfComputeOpsOutlining ? pm.nest<mlir::ModuleOp>() : pm;

    nestedPm.addPass(VPU::createAdjustMemorySpacePass(log));
    nestedPm.addPass(VPU::createOptimizeSharedInputCopyForConcatPass(log));
    nestedPm.addPass(VPU::createOptimizeConcatPass(/*optimizeOnlyOuterConcat*/ false, log));
    nestedPm.addPass(mlir::createCanonicalizerPass(grc));

    nestedPm.addPass(VPU::createCMXConcatPass(log));
    nestedPm.addPass(mlir::createCanonicalizerPass(grc));
    nestedPm.addPass(VPU::createMoveReflectPadToCMXPass(log));
    nestedPm.addPass(VPU::createMoveTensorOpsToCMXPass(log));

    VPU::arch60xx::buildNCEWorkloadsPipeline(nestedPm, VPU::NCEWorkloadsOptions(options), log);

    nestedPm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableSCFTiling) {
        // Must run after the tensor.pad readers (MoveTensorOpsToCMX, WorkloadsForNCEOpsSCF) and
        // before bufferization.
        nestedPm.addPass(VPU::createConvertTensorPadToVPUPadPass(log));
    }
    nestedPm.addPass(createAdjustDynamicOpsBeforeBufferizationPass());
    nestedPm.addPass(VPU::createLegalizeDynamicShapeConcatForSWLayersPass(log));
    nestedPm.addPass(Shave::createExternalKernelMemSpaceAdjustmentPass(log));
    nestedPm.addPass(VPU::createAdjustMemorySpaceForSHVOpsPass(log));
    if (options.enableEntireMainContentOutlining) {
        nestedPm.addPass(VPU::createOutlineEntireMainContentPass(log));
    }
}

void vpux::VPU::arch60xx::registerVPUPipelines() {
    mlir::PassPipelineRegistration<VPU::arch60xx::DefaultHWOptions>(
            "default-hw-mode-vpu", "VPU dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const VPU::arch60xx::DefaultHWOptions& options) {
                VPU::arch60xx::buildDefaultHWPipeline(pm, options);
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

    mlir::PassPipelineRegistration<VPU::NCEWorkloadsOptions>(
            "nce-workloads", "Split NCE operations onto DPU workloads and legalize them",
            [](mlir::OpPassManager& pm, const VPU::NCEWorkloadsOptions& options) {
                VPU::arch60xx::buildNCEWorkloadsPipeline(pm, options);
            });
}
