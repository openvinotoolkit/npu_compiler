//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/Shave/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/locverif/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// Outlining
//

void vpux::IE::arch40xx::buildLowPrecisionPipeline(mlir::OpPassManager& pm, const LowPrecisionOptions& options,
                                                   Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createOptimizeUnalignedQDQSeqPass(log));
    pm.addPass(IE::createSwapFakeQuantWithReshapeAndStridedSlicePass(log));
    pm.addPass(IE::createSwapConvertWithReshapeKindOpsPass(log));
    if (options.enableAlignScales) {
        pm.addPass(IE::createAlignScalesPass(log));
    }
    if (options.enableAdjustNonZeroFakeQuant) {
        pm.addPass(IE::createAdjustNonZeroFakeQuantPass(log));
    }
    if (options.enableMatmulMixedPrecisionDecomposition) {
        pm.addPass(IE::createProcessAsymmetricZeroPointsForMatmulPass(
                options.matmulMixedPrecisionDecompositionRatio.getValue(), log));
    }

    pm.addPass(IE::createSplitFakeQuantPass(log));

    pm.addPass(mlir::createCanonicalizerPass());  // Note: folds constants before convert-to-dequantize
    pm.addPass(IE::createConvertToQuantizedOpsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createPropagateAndFuseQuantizeDequantizePass(log));
    pm.addPass(IE::createFuseConvertWithQDQPass(log));
    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    pm.addPass(IE::createPropagateDequantThroughConcatPass(log));
    pm.addPass(IE::createConvertWeightsToU8Pass(log));
    pm.addPass(IE::createFuseQuantizedOpsPass(log));

    // Enable sequence FuseQuantizedOps->FuseActivationOps->FuseOutstandingDequant->ConvertToMixedPrecision->
    // ConvertQuantizeOpsToNceOps. The sequence allows Conv->Quantize->Dequantize->LeakyReLU->Quantize->Dequantize
    // fused into a single Conv.
    pm.addPass(IE::createFuseActivationOpsPass(log));

    if (options.enableConvertWeightsToU8I4) {
        pm.addPass(IE::createConvertWeightsToI8Pass(log));
    }
    if (options.enableConvertToPalletizationLUT) {
        pm.addPass(IE::createConvertToPalletizationLUT(log));
    }
    pm.addPass(IE::createConvertToMixedPrecision(isOptionEnabled(options.enableFloatInQuantWeightsMixedMode), log));
    if (options.enableQuantDequantRemoval) {
        pm.addPass(IE::createRemoveQuantDequantSeqPass(log));
    }
    if (options.enableConvertWeightsToU8I4) {
        pm.addPass(IE::createConvertWeightsToU8Pass(log));
        pm.addPass(IE::createConvertWeightsToI4Pass(log));
    }
    // After the execution of ConvertWeightsToU8 could appear new cases when FuseQuantizedOps and
    // ConvertToMixedPrecision can be applied. The execution ConvertWeightsToU8 can align the data type of the operands
    // of NCE operations to U8, condition that is necessary in rewriters like: FuseWithEltwiseConverter,
    // FloatOutAddRewriter where it required that the operands to have the same data type and this happens only after
    // execution of ConvertWeightsToU8.
    pm.addPass(IE::createFuseQuantizedOpsPass(log));
    if (options.enableFuseOutstandingDequant) {
        if (!options.functionOutlining.hasValue()) {
            pm.addPass(IE::createFuseOutstandingDequant(log));
        }

        // This is a short term solution to call ConvertToMixedPrecision when we have
        //     Original subgraph
        //         Conv -> FQ1 -> FQ2 -> Conv (FQ1 and FQ2 have different params)
        //     At this point
        //        (Conv-Q1-DQ1) -> Q2 -> (DQ2-Conv)
        // In long term need to consider a new pass to fuse FQs with different params
    }
    // Note: this ConvertToMixedPrecision call serves both FuseQuantizedOps and
    // FuseOutstandingDequant
    pm.addPass(IE::createConvertToMixedPrecision(isOptionEnabled(options.enableFloatInQuantWeightsMixedMode), log));

    if (options.enableFuseOutstandingQuant) {
        pm.addPass(IE::createFuseOutstandingQuantPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(options.runtimeDequantizationLimit,
                                             isOptionEnabled(options.enableRuntimeDequant), log));
    if (options.enableLogDynamicQuant) {
        pm.addPass(IE::createLoggingWeightsQuantFusedIntoTaskPass(log));
    }
    pm.addPass(IE::createOptimizePrecisionAcrossFunctionCallsPass(log));

    // E#176434: remove option
    if (options.enableConvertQuantizeOpsToNceOps) {
        pm.addPass(IE::createConvertQuantizeOpsToNceOpsPass(log));
    }

    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::IE::arch40xx::buildFinalTransformationPipeline(mlir::OpPassManager& pm,
                                                          const IE::arch40xx::DefaultHWOptions& options, Logger log) {
    pm.addPass(IE::createAdaptODUPermutePass(log));
    pm.addPass(IE::createFuseInefficientTileForAddPass(log));
    // Operation Conversions
    if (options.enableConvertExpandToConvPass) {
        pm.addPass(IE::createConvertExpandToConvPass(log));
    }

    // Operation Fusions
    pm.addPass(IE::createOptimizeIdentityPoolPass(log));
    if (options.enableFuseD2SExpand) {
        pm.addPass(IE::createFuseD2SExpandChannelsPass(log));
    }

    // Operation optimizations
    pm.addPass(IE::createPropagateShapeCastPass(log));
    pm.addPass(IE::createPropagatePermuteCastPass(log));
    pm.addPass(IE::createMoveDynamicDequantizeToUserPass(log));
}

//
// AttentionProcessingPipeline
//

void vpux::IE::arch40xx::buildAttentionProcessingPipeline(mlir::OpPassManager& pm,
                                                          const IE::AttentionProcessingOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (options.enableFlashSDPAConversion) {
        pm.addPass(IE::createConvertSDPAToFlashSDPAPass(log));
    }
    pm.addPass(IE::createResolveStridedSlicePass(log));
    if (options.enableConvertToAttention) {
        pm.addPass(IE::createFuseAttentionPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }
    if (options.enableFuseSoftwareSDPA) {
        pm.addPass(IE::createFuseSDPAPass(log));
    }
    if (options.enableDecomposeAttention) {
        pm.addPass(IE::createDecomposeAttentionPass(log));
    }
    pm.addPass(IE::createReshapeMatMulInputsPass(options.enableGroupedMatMul, log));
}

//
// DefaultHWPipeline
//

void vpux::IE::arch40xx::buildDefaultHWPipeline(mlir::OpPassManager& pm, const IE::arch40xx::DefaultHWOptions& options,
                                                Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(locverif::createStartLocationVerifierPass(log, options.locationsVerificationMode));
    pm.addPass(IE::createForbidFourBitOutputsPass(log));

    IE::buildOutliningPipeline(pm, options, log);

    // No passes should be run before this pipeline, with very few exceptions.
    IE::buildPostImportPipeline(pm, log);
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(IE::createDumpStatisticsOfIeOpsPass("Start of IE pipeline statistics", log));

    if (options.enableReduceNumTilesForSmallModelsPass) {
        pm.addPass(IE::createReduceNumTilesForSmallModelsPass(log));
    }

    // Level 3 : Topology
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }

    if (options.enableDynamicShapeTransformationsPipeline) {
        IE::buildDynamicShapeTransformationsPipeline(pm, IE::DynamicShapeTransformOptions(options), log);
    }
    IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::arch40xx::buildAttentionProcessingPipeline(pm, IE::AttentionProcessingOptions(options), log);
    IE::buildInitialTransformationsPipeline(pm, IE::TransformOptions(options), log);

    if (options.enableAdjustPrecisionPipeline) {
        IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);
    }

    // Couldn't move the pass before convert_precision_to_fp16 because of regressions, extra conversions are added
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));
    IE::buildOperationConversionPipeline(pm, IE::OperationConversionOptions(options), log);

    IE::buildAdjustShapePipeline(pm, log);
    IE::buildSplitLargeOpsPipeline(pm, log);
    IE::buildConvertToEfficientOpsPipeline(pm, IE::ConvertToEfficientOpsOptions(options), log);

    IE::buildAdjustForVPUPipeline(pm, log);

    pm.addPass(mlir::createCSEPass());

    IE::buildHandleHyperParametersPipeline(pm, log);
    IE::buildConvertToConvolutionPipeline(pm, log);
    IE::buildReorderFakeQuantizePipeline(pm, IE::ReorderFakeQuantizeOptions(options), log);

    pm.addPass(mlir::createCanonicalizerPass(grc));
    IE::buildScaleShiftProcessingPipeline(pm, log);

    IE::arch40xx::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
    pm.addPass(IE::createConvertShapeTo4DPass(isOptionEnabled(options.forceConvertGatherTo4D), log));
    pm.addPass(IE::createSwapViewOpAndClampPass(log));

    IE::buildOptimizeActivationsPipeline(pm, IE::OptimizeActivationsOptions(options), log);

    IE::buildSplitAndMapBilinearInterpolateOnDPUPipeline(pm, IE::SplitAndMapBilinearInterpolateOnDPUOptions(options),
                                                         log);

    IE::buildBatchTransformationPipeline(pm, BatchUnrollOptions::create(options, log), log);

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    IE::buildOptimizeMemPermuteAndActivationChannelsExpandPipeline(pm, IE::ExpandActivationChannelsOptions(options),
                                                                   log);

    pm.addPass(locverif::createStopLocationVerifierPass(log));

    IE::buildOptimizeViewLikeOpsPipeline(pm, log);

    IE::buildOptimizeSliceOpPipeline(pm, options.disableSliceToConvMinHWThreshold, log);

    IE::buildDimensionAlignmentPipeline(pm, log);

    IE::arch40xx::buildFinalTransformationPipeline(pm, options, log);

    // Shave related optimization
    pm.addPass(Shave::createLoadExternalKernelResourcesPass(log));
    if (options.enableShaveCodeGen) {
        ShaveCodeGen::buildShaveCodeGenPipelineIE(pm, log);
    }

    // Logging of optimizations post-check at the end of the pipeline
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }

    pm.addPass(IE::createDumpStatisticsOfIeOpsPass("End of IE pipeline statistics", log));
}

void vpux::IE::arch40xx::buildReferenceSWPipeline(mlir::OpPassManager& pm,
                                                  const IE::arch40xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // No passes should be run before this pipeline, with very few exceptions.
    IE::buildPostImportPipeline(pm, log);

    // Level 3 : Topology

    IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::arch40xx::buildAttentionProcessingPipeline(pm, IE::AttentionProcessingOptions(options), log);
    IE::buildInitialTransformationsPipeline(pm, IE::TransformOptions(options), log);
    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    // Couldn't move the pass before convert_precision_to_fp16 because of regressions, extra conversions are added
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    // Resolve group quant MatMul pattern
    pm.addPass(mlir::createCSEPass());
    pm.addPass(IE::createUniquifySimilarOpsPass(log));
    pm.addPass(IE::createMergeParallelFullyConnectedPass(log));
    pm.addPass(IE::createUnrollGroupQuantizePass(log));
    pm.addPass(IE::createUnrollFullyConnectedPass(log));
    pm.addPass(IE::createMergeFullyConnectedPass(isOptionEnabled(options.mergeUnrolledMatmulForLargeOC), log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }

    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertStridedSlice2ConvPass(log));
    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(isOptionEnabled(options.forceConvertGatherTo4D), log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertToSpatialOpPass(false, log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    IE::buildAdjustForVPUPipeline(pm, log);

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(options.runtimeDequantizationLimit,
                                             isOptionEnabled(options.enableRuntimeDequant), log));
    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableShaveCodeGen) {
        ShaveCodeGen::buildShaveCodeGenPipelineIE(pm, log);
    }
}

//
// registerIEPipelines
//

void vpux::IE::arch40xx::registerIEPipelines() {
    mlir::PassPipelineRegistration<IE::arch40xx::DefaultHWOptions>(
            "default-hw-mode-ie", "IE dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const IE::arch40xx::DefaultHWOptions& options) {
                IE::arch40xx::buildDefaultHWPipeline(pm, options);
            });
    mlir::PassPipelineRegistration<IE::LowPrecisionTransformOptions>(
            "initial-low-precision-transformations",
            "[LEGALIZATION] Initial Low Precision Transformations, convert initial low precision IR operations to "
            "equivalent operations supported by the lower compilation levels",
            [](mlir::OpPassManager& pm, const IE::LowPrecisionTransformOptions& options) {
                IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<IE::AttentionProcessingOptions>(
            "attention-processing",
            "[OPTIMIZATION] Attention processing transformations: fuse patterns into AttentionOp and apply "
            "attention decomposition strategies",
            [](mlir::OpPassManager& pm, const IE::AttentionProcessingOptions& options) {
                IE::arch40xx::buildAttentionProcessingPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<LowPrecisionOptions>(
            "low-precision", "[OPTIMIZATION] Low precision transformations",
            [](mlir::OpPassManager& pm, const LowPrecisionOptions& options) {
                IE::arch40xx::buildLowPrecisionPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<DynamicShapeTransformOptions>(
            "dynamic-shape-transformations", "[LEGALIZATION] Introduces operation to handle dynamic shapes",
            [](mlir::OpPassManager& pm, const DynamicShapeTransformOptions& options) {
                IE::buildDynamicShapeTransformationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<IE::arch40xx::DefaultHWOptions>(
            "reference-sw-mode-ie", "IE dialect part of Reference SW pipeline",
            [](mlir::OpPassManager& pm, const IE::arch40xx::DefaultHWOptions& options) {
                IE::arch40xx::buildReferenceSWPipeline(pm, options);
            });
}
