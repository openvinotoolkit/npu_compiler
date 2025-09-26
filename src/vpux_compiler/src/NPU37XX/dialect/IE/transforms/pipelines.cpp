//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::IE::arch37xx::buildOptimizeActivationsPipeline(mlir::OpPassManager& pm,
                                                          const OptimizeActivationsOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    pm.addPass(IE::createInsertIdentityPoolBeforeOpPass(log));
    pm.addPass(IE::createSwapMaxPoolWithActivation(log));

    if (options.enableDPUF16ToF32Convert) {
        if (options.enableSwapConvertWithSWOp) {
            pm.addPass(IE::createSwapConvertWithSWOpPass(log));
        }
        pm.addPass(IE::createRunF16ToF32ConvertOnDPUPass(log));
    }

    pm.addPass(IE::createFuseActivationOpsPass(options.enableFuseClampOperations, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::IE::arch37xx::buildMemPermutePositioningPipeline(mlir::OpPassManager& pm,
                                                            const MemPermutePositioningOptions& /*options*/,
                                                            Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createPropagateMemPermuteThroughSoftMaxPass(log));
    pm.addPass(IE::createMovePermutePostEltwisePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createLegalizeNDMemPermutePass(log));
    pm.addPass(IE::createPropagateMemPermuteBeforeOpPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createPropagateMemPermuteThroughEltwisePass(log));
    pm.addPass(IE::createUniquifyOpsPass(log));
    pm.addPass(IE::createAdjustMemPermuteAroundOpPass(log));
}

void vpux::IE::arch37xx::buildMemPermuteProcessingPipeline(mlir::OpPassManager& pm,
                                                           const ExpandActivationChannelsOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(IE::createSwapMemPermuteAndExpandPass(log));
    pm.addPass(IE::createPropagateMemPermuteBeforeOpPass(log));
    pm.addPass(IE::createOptimizeConcatWithConvPass(log));
    if (options.enableAdjustConvShapePass) {
        // For such Convolution which used to optimize Concat need to be processed by AdjustConvolutionShapePass for
        // efficiency.
        pm.addPass(IE::createAdjustConvolutionShapePass(log));
    }
    pm.addPass(IE::createInsertIdentityPoolBeforeOpPass(log));
    pm.addPass(IE::createFuseMemPermutePass(log));
    pm.addPass(IE::createConvertMemPermuteToOpPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createUniquifyOpsPass(log));
}

void vpux::IE::arch37xx::buildExpandAndOptimizeActivationChannelsPipeline(
        mlir::OpPassManager& pm, const ExpandActivationChannelsOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    if (options.enableExpandActivationChannels) {
        if (options.enableAdjustConvShapePass) {
            pm.addPass(IE::createOptimizeAvgPoolWithUnalignedChannelsPass(log));
            pm.addPass(IE::createAdjustConvolutionShapePass(log));
        }
        pm.addPass(IE::createExpandActivationChannelsPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations), log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::createOptimizeSliceExpandPass(log));
        }

        pm.addPass(IE::createAdjustConvolutionWeightsPass(log));
        pm.addPass(IE::createAdjustConvolutionInputShapePass(log));
        pm.addPass(IE::createAdjustInputShapePass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::createOptimizeSliceExpandPass(log));
        }
        pm.addPass(IE::createUniquifyOpsPass(log));
        pm.addPass(IE::createHandleEltwiseWithSmallHeightPass(log));
        pm.addPass(IE::createPropagateAffineReshapePass(log));
        pm.addPass(IE::createUniquifyBranchesPass(log));

        if (options.enableFusePermuteQuantizeExpand) {
            pm.addPass(IE::createPropagateExpandPass(log));
            pm.addPass(IE::createFusePermuteQuantizeExpandPass(log));
            // Convert reorder to permuteQuantize after OptimizeReordersPass if it's feasible
            // Such as Reorder(ui8)-Convert(ui->f16)-Expand to Convert(ui8->f16)-Expand-Reorder(f16)
            pm.addPass(IE::createConvertReorderToPermuteQuantizePass(log));
        }
    }

    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertSplitConcatToTransposePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::IE::arch37xx::buildOptimizeMemPermuteAndActivationChannelsExpandPipeline(
        mlir::OpPassManager& pm, const ExpandActivationChannelsOptions& options, Logger log) {
    IE::arch37xx::buildMemPermutePositioningPipeline(pm, IE::MemPermutePositioningOptions(options), log);
    IE::arch37xx::buildExpandAndOptimizeActivationChannelsPipeline(pm, options, log);
    IE::arch37xx::buildMemPermuteProcessingPipeline(pm, options, log);
}

//
// InitialLowPrecisionTransformations
//

void vpux::IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(
        mlir::OpPassManager& pm, const IE::LowPrecisionTransformOptions& options, Logger log) {
    pm.addPass(IE::createConvertScalarToTensorPass(log));
    pm.addPass(IE::createConsolidateNF4WeightsPatternPass(log));
    pm.addPass(IE::createDecomposeMultiZPQuantizationPatternPass(log));
    pm.addPass(IE::createWeightsDequantizeToFakeQuantizePass(log));
    pm.addPass(IE::createFuseInputScaleShiftPass(log));
    pm.addPass(IE::createConsolidateWeightsDequantizationPass(options, log));
    pm.addPass(IE::createConvertMinMaxToClampPass(log));
    pm.addPass(IE::createFoldActivationBeforeFQPass(log));
}

//
// InitialTransformations
//

void vpux::IE::arch37xx::buildInitialTransformationsPipeline(mlir::OpPassManager& pm,
                                                             const IE::TransformOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createFuseDynamicQuantizePass(log));
    pm.addPass(IE::createFuseRMSNormPass(log));
    pm.addPass(IE::createUniquifyOpsPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createOptimizeParallelLayersPass(log));
    pm.addPass(IE::createDecomposeLSTMSequencePass(log));
    if (options.enableDecomposeGRUSequence) {
        pm.addPass(IE::createDecomposeGRUSequencePass(log));
    }
    pm.addPass(IE::createDecomposeLSTMCellPass(log));
    pm.addPass(IE::createDecomposeGRUCellPass(log));
    pm.addPass(IE::createDecomposeNormalizeL2Pass(log));
    pm.addPass(IE::createReshapeMatMulInputsPass(options.enableGroupedMatMul, log));
    pm.addPass(IE::createAdjustFakeQuantizeParamsPass(log));
    pm.addPass(IE::createAdjustFakeQdqParamsPass(log));
    pm.addPass(IE::createFuseFQAndMulPass(options.fuseFQAndMulWithNonConstInput, log));
    pm.addPass(IE::createHandleU16FakeQuantizePass(log));
    pm.addPass(IE::createFuseSoftmaxPass(log));
    pm.addPass(IE::createSwishFusionPass(log));
    pm.addPass(IE::createFuseSDPAPass(log));
    pm.addPass(IE::createFuseRoPEPass(log));
    pm.addPass(IE::createEltwiseFakeQuantizeFusionPass(options.enableAdaptiveStripping, log));
    pm.addPass(IE::createUnrollTensorIteratorPass(log));
    pm.addPass(IE::createNormalizeL2FusionPass(log));
    pm.addPass(IE::createMVNFusionPass(log));
    if (options.enableConvertFFTToConv) {
        pm.addPass(IE::createConvertFFTToConvPass(log));
    }
    pm.addPass(IE::createMoveMultiplyDividePostOpPass(log));
    pm.addPass(IE::createShrinkMatmulGroupsPass(log));
    pm.addPass(IE::createMatMulInputsTo2dPass(options.enableGroupedMatMul, log));
    pm.addPass(IE::createPropagateOpThroughBatchConcatPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.fuseMvn6ScaleBias) {
        pm.addPass(IE::createFuseMvn6ScaleBiasPass(log));
    }
    pm.addPass(IE::createConvertMVN6ToMVN1Pass(log));
    pm.addPass(IE::createConvertSubGRUSequenceToConvPass(log));
    pm.addPass(IE::createConvertConvBackpropDataToConvOrTransposedConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDilatedConvConvertPass(log));
}

//
// LowPrecision
//

void vpux::IE::arch37xx::buildLowPrecisionPipeline(mlir::OpPassManager& pm, const LowPrecisionOptions& options,
                                                   Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createOptimizeUnalignedQDQSeqPass(log));
    pm.addPass(IE::createSwapFakeQuantWithReshapeAndStridedSlicePass(log));
    pm.addPass(IE::createSwapConvertWithReshapeKindOpsPass(log));
    if (options.enableAlignScales) {
        pm.addPass(IE::createAlignScalesPass(isOptionEnabled(options.enableSEPtrsOperations), log));
    }
    if (options.enableAdjustNonZeroFakeQuant) {
        pm.addPass(IE::createAdjustNonZeroFakeQuantPass(log));
    }
    if (options.enableConvolutionMixedPrecisionDecomposition) {
        pm.addPass(IE::createProcessAsymmetricZeroPointsForConvolutionPass(log));
    }
    if (options.enableMatmulMixedPrecisionDecomposition) {
        pm.addPass(IE::createProcessAsymmetricZeroPointsForMatmulPass(
                options.matmulMixedPrecisionDecompositionRatio.getValue(), log));
    }

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(IE::createConvertToDequantizePass(options, log));
    if (options.enablePropagateQuantDequant) {
        pm.addPass(mlir::createCanonicalizerPass(grc));
        pm.addPass(
                IE::createPropagateAndFuseQuantizeDequantizePass(isOptionEnabled(options.enableSEPtrsOperations), log));
    }
    pm.addPass(IE::createFuseConvertWithQDQPass(log));
    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    pm.addPass(IE::createPropagateDequantThroughConcatPass(log));
    pm.addPass(IE::createFuseQuantizedOpsPass(
            /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
            /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));

    // Enable sequence FuseQuantizedOps->FuseActivationOps->FuseOutstandingDequant->ConvertToMixedPrecision->
    // ConvertQuantizeOpsToNceOps. The sequence allows Conv->Quantize->Dequantize->LeakyReLU->Quantize->Dequantize
    // fused into a single Conv.
    pm.addPass(IE::createFuseActivationOpsPass(options.enableFuseClampOperations, log));

    if (options.enableFP16ToU8MixedMode) {
        pm.addPass(IE::createOptimizeNetworkInputConvertPass(log));
    }

    if (options.enableConvertWeightsToU8I4) {
        pm.addPass(IE::createConvertWeightsToI8Pass(log));
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
    pm.addPass(IE::createFuseQuantizedOpsPass(
            /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
            /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
    pm.addPass(IE::createConvertToMixedPrecision(isOptionEnabled(options.enableFloatInQuantWeightsMixedMode), log));
    if (options.enableFuseOutstandingDequant) {
        pm.addPass(IE::createFuseOutstandingDequant(log));

        // This is a short term solution when we have
        //     Original subgraph
        //         Conv -> FQ1 -> FQ2 -> Conv (FQ1 and FQ2 have different params)
        //     At this point
        //        (Conv-Q1-DQ1) -> Q2 -> (DQ2-Conv)
        // In long term need to consider a new pass to fuse FQs with different params
        pm.addPass(IE::createConvertToMixedPrecision(isOptionEnabled(options.enableFloatInQuantWeightsMixedMode), log));
    }
    if (options.enableFuseOutstandingQuant) {
        pm.addPass(IE::createFuseOutstandingQuantPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(options.runtimeDequantizationLimit,
                                             isOptionEnabled(options.enableRuntimeDequant), log));
    if (options.enableDynamicQuant) {
        pm.addPass(IE::createWeightsQuantFusedIntoTaskPass(log));
    }
    pm.addPass(IE::createOptimizePrecisionAcrossFunctionCallsPass(log));

    // E#176434: remove option
    if (options.enableConvertQuantizeOpsToNceOps) {
        pm.addPass(IE::createConvertQuantizeOpsToNceOpsPass(log));
    }

    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// AdjustLayout
//

void vpux::IE::arch37xx::buildAdjustLayoutPipeline(mlir::OpPassManager& pm, const AdjustLayoutOptions& options,
                                                   Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (options.enableForceZMajorConcat) {
        pm.addPass(IE::createInsertReorderBetweenLayerAndConcatPass(log));
    }

    pm.addPass(IE::createPropagateAffineReshapePass(log));
    pm.addPass(IE::createPropagateTransposePass(log));
    pm.addPass(IE::createUniquifyBranchesPass(log));
    pm.addPass(IE::createSwapTransposeConcatPass(log));
    pm.addPass(IE::createTransposeToPermuteCastPass(log));
    pm.addPass(IE::createAdjustLayoutsPass(
            /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
            /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableOptimizeReorders) {
        pm.addPass(IE::createFuseReshapeMvnPass(log));
        pm.addPass(IE::createOptimizeReordersPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
        pm.addPass(IE::createOptimizeReordersAcrossFunctionCallsPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
        pm.addPass(IE::createUniquifyOpsPass(log));
        pm.addPass(IE::createUniquifyBranchesPass(log));
        pm.addPass(IE::createPropagateReorderToNCEPass(log));
        pm.addPass(IE::createFuseReordersPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }
}

void vpux::IE::arch37xx::buildDynamicShapeTransformationsPipeline(mlir::OpPassManager& pm,
                                                                  const DynamicShapeTransformOptions& options,
                                                                  Logger log) {
    pm.addPass(IE::createOptDynamicEltwiseWithShapeOfPass(log));
    pm.addPass(IE::createPopulateDynamicDimensionsHWPass(log));
    pm.addPass(IE::createPopulateDynamicDimensionsGenericPass(log));
    if (isOptionEnabled(options.enableApplyDynamicBoundaryCorrection)) {
        pm.addPass(IE::createApplyDynamicBoundaryCorrectionPass(log));
    }
    pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
    pm.addPass(IE::createLegalizeReifyResultShapesResidualsPass(log));
    pm.addPass(IE::createPadDynamicInputsPass(log));
}

//
// DefaultHWPipeline
//

void vpux::IE::arch37xx::buildDefaultHWPipeline(mlir::OpPassManager& pm, const IE::arch37xx::DefaultHWOptions& options,
                                                Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(Core::createStartLocationVerifierPass(log, options.locationsVerificationMode));

    bool isOutliningEnabled = options.functionOutlining.hasValue();
    if (options.enableProfiling && !options.enableProfilingWithOutlining) {
        // TODO: E#140041 enable profiling with outlining
        log.warning("Outlining was disabled due to profiling being enabled");
        isOutliningEnabled = false;
    }
    if (isOutliningEnabled) {
        if (options.enableLoopOutliner) {
            pm.addPass(IE::createLoopOutlinerPass(log));
        }
        pm.addPass(mlir::createCanonicalizerPass(grc));
        pm.addPass(IE::createOutlinerPass(options, log));
        pm.addPass(IE::createDuplicateFQAcrossFunctionCallsPass(log));
    }

    // No passes should be run before this pipeline, with very few exceptions.
    IE::buildPostImportPipeline(pm, IE::PostImportOptions(options), log);
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Level 3 : Topology
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }

    if (options.enableOnlineSDPAConversion) {
        pm.addPass(IE::createConvertSDPAToOnlineSDPAPass(log));
        pm.addPass(IE::createTileOnlineSDPAPass(log));
        pm.addPass(IE::createDecomposeOnlineSDPAPass(options.disableIncrementalSDPADecomposition, log));
        pm.addPass(IE::createTileIncrementalSDPAPass(log));
        if (!options.disableIncrementalSDPADecomposition) {
            pm.addPass(IE::createDecomposeIncrementalSDPAPass(log));
        }
    }

    IE::arch37xx::buildDynamicShapeTransformationsPipeline(pm, IE::DynamicShapeTransformOptions(), log);
    pm.addPass(IE::createReshapeMatMulInputsPass(options.enableGroupedMatMul, log));
    IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::arch37xx::buildInitialTransformationsPipeline(pm, IE::TransformOptions(options), log);
    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    IE::buildOperationConversionPipeline(pm, IE::OperationConversionOptions(options), log);

    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    pm.addPass(IE::createUnrollConv3dToConv2dPass(log));
    pm.addPass(IE::createReshapeMaxPoolPass(log));
    if (options.enableHandleLargeKernel) {
        pm.addPass(IE::createAdjustMaxPoolInputShapePass(log));
        pm.addPass(IE::createHandleLargeKernelsPass(log));
    }
    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }

    pm.addPass(IE::createConvertDivideToMultiplyPass(log));
    pm.addPass(IE::createReassociateMultiplyPass(log));
    pm.addPass(IE::createAdaptShapesForScaleShiftPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createSwapTransposeConcatPass(log));
    pm.addPass(IE::createConvertSplitConcatToTransposePass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(IE::createSplitInterpolateAxesPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertGatherElementsToGatherPass(log));

    pm.addPass(IE::createConvertToSpatialOpPass(false, isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertSubtractToAddPass(log));
    pm.addPass(IE::createConvertBranchesConcatToConvPass(log));
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    pm.addPass(IE::createSwapPadLayerPass(log));
    // Note: apply FuseStaticScale after ConvertDivideToMultiply to increase
    // the applicability
    pm.addPass(IE::createFuseStaticScalePass(log));
    pm.addPass(IE::createBroadcastInputForAddPass(log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    // E#79878: Solve eltwise single layer test failure.
    // SwapOperations pass may generate non-4D AddOp.
    // If AddOp appears here means that it cannot be fused into NCE task.
    // So convert it's shape to 4D and then convert this AddOp to ScaleShift.
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(IE::createConvertToScaleShiftPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));

    pm.addPass(IE::createConvertDepth2SpaceToTransposedConvPass(log));
    pm.addPass(IE::createSwapD2SAndScaleShiftPass(log));
    pm.addPass(IE::createConvertReverseToDWConvPass(log));
    pm.addPass(IE::createConvertDeformableConvToConvPass(log));

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);
    pm.addPass(Core::createStopLocationVerifierPass(log));

    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));

    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    if (options.enableSplitConvWithMultipleFQ) {
        pm.addPass(IE::createSplitConvWithMultipleFQPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableHandleLargeKernel) {
        pm.addPass(IE::createHandleLargeKernelsPass(log));
    }
    if (options.enableHandleLargeStrides) {
        pm.addPass(IE::createHandleLargeStridesPass(log));
    }
    if (options.enableHandleAsymmetricStrides) {
        pm.addPass(IE::createHandleAsymmetricStridesPass(log));
    }
    if (options.enableHandleLargePads) {
        pm.addPass(IE::createHandleLargePadsPass(log));
    }
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableOptimizeScaleShiftToDWConv) {
        IE::buildScaleShiftProcessingPipeline(pm, log);
    }

    pm.addPass(IE::createFuseActivationOpsPass(options.enableFuseClampOperations, log));
    pm.addPass(IE::createConvertStridedSlice2ConvPass(log));
    if (options.enableLowPrecision) {
        IE::arch37xx::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
        pm.addPass(IE::createConvertShapeTo4DPass(log));
        pm.addPass(IE::createSwapViewOpAndClampPass(log));
    }
    IE::arch37xx::buildOptimizeActivationsPipeline(pm, IE::OptimizeActivationsOptions(options), log);

    pm.addPass(IE::createOptimizeTileOpPass(log));

    if (options.enableSEPtrsOperations && options.enableSplitBilinerIntoHAndW) {
        pm.addPass(IE::createSplitBilinerIntoHAndWPass(log));
    }

    if (options.enableBilinearInterpolateOnDPU) {
        pm.addPass(IE::createMapBilinearInterpolateOnDPUPass(isOptionEnabled(options.enableSEPtrsOperations), log));
    }

    pm.addPass(IE::createConvertBatchedLayerTo1NPass(log));
    pm.addPass(IE::createConvertBroadcastToTilePass(log));
    pm.addPass(IE::createUnrollBatchPass(log, isOptionEnabled(options.skipUnrollBatch)));

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }
    pm.addPass(IE::createConvertBranchesConcatToConvPass(log));

    pm.addPass(IE::createFuseConvWithSlicePass(log));

    pm.addPass(IE::createSwapMVNWithTransposePass(log));

    IE::arch37xx::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    if (options.enableFusePermuteQuantize) {
        pm.addPass(IE::createFusePermuteQuantizePass(false, log));
        pm.addPass(IE::createConvertReorderToPermuteQuantizePass(log));
    }

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createAdjustGroupConvShapePass(log));
    }
    IE::arch37xx::buildOptimizeMemPermuteAndActivationChannelsExpandPipeline(
            pm, IE::ExpandActivationChannelsOptions(options), log);

    pm.addPass(IE::createRemoveViewLikeOpsChainPass(log));
    pm.addPass(IE::createOptimizeOpSlicePass(log));
    pm.addPass(IE::createUniquifyOpsPass(log));

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createExpandActivationWidthPass(log));
        pm.addPass(IE::createAdjustInputShapePass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
        pm.addPass(IE::createPropagateAffineReshapePass(log));
        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::createOptimizeSliceExpandPass(log));
        }
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }
    if (options.enableOptimizeSliceWithStride) {
        pm.addPass(IE::createOptimizeSliceWithStridePass(log));
        if (options.enableAdjustConvShapePass) {
            pm.addPass(IE::createAdjustConvolutionShapePass(log));
        }
    }
    if (options.enableConvertExpandToConvPass) {
        pm.addPass(IE::createConvertExpandToConvPass(log));
    }
    pm.addPass(IE::createPropagateShapeCastPass(log));
    pm.addPass(IE::createOptimizeIdentityPoolPass(log));
    pm.addPass(IE::createPropagatePermuteCastThroughDequantizePass(log));
    pm.addPass(IE::createMoveDynamicDequantizeToUserPass(log));
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }
}

void vpux::IE::arch37xx::buildReferenceSWPipeline(mlir::OpPassManager& pm,
                                                  const IE::arch37xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    // No passes should be run before this pipeline, with very few exceptions.
    IE::buildPostImportPipeline(pm, IE::PostImportOptions(options), log);

    // Level 3 : Topology
    IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::arch37xx::buildInitialTransformationsPipeline(pm, IE::TransformOptions(options), log);
    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    // Resolve group quant MatMul pattern
    pm.addPass(IE::createUniquifyOpsPass(log));
    pm.addPass(IE::createMergeParallelFullyConnectedPass(log));
    pm.addPass(IE::createUnrollGroupQuantizePass(log));
    pm.addPass(IE::createUnrollFullyConnectedPass(log));
    if (options.fuseScalesToAccumulate) {
        pm.addPass(IE::createFuseScalesToAccumulatePass(log));
    }
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }

    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertStridedSlice2ConvPass(log));
    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertToSpatialOpPass(false, isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(options.runtimeDequantizationLimit,
                                             isOptionEnabled(options.enableRuntimeDequant), log));
    if (options.enableMergeFakeQuant) {
        pm.addPass(IE::createMergeFakeQuantPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::arch37xx::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// registerIEPipelines
//

void vpux::IE::arch37xx::registerIEPipelines() {
    mlir::PassPipelineRegistration<IE::arch37xx::DefaultHWOptions>(
            "default-hw-mode-ie", "IE dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const IE::arch37xx::DefaultHWOptions& options) {
                IE::arch37xx::buildDefaultHWPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<IE::LowPrecisionTransformOptions>(
            "initial-low-precision-transformations",
            "[LEGALIZATION] Initial Low Precision Transformations, convert initial low precision IR operations to "
            "equivalent operations supported by the lower compilation levels",
            [](mlir::OpPassManager& pm, const IE::LowPrecisionTransformOptions& options) {
                IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<IE::TransformOptions>(
            "initial-transformations",
            "[LEGALIZATION] Initial Transformations, convert initial IR operations to another and tries to reduce the "
            "number of op types used in the graph",
            [](mlir::OpPassManager& pm, const IE::TransformOptions& options) {
                IE::arch37xx::buildInitialTransformationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<OptimizeActivationsOptions>(
            "optimize-activations", "[OPTIMIZATION] Optimize activations for VPU target",
            [](mlir::OpPassManager& pm, const OptimizeActivationsOptions& options) {
                IE::arch37xx::buildOptimizeActivationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<MemPermutePositioningOptions>(
            "mempermute-positioning",
            "[OPTIMIZATION] MemPermute positioning is responsible for handling data transfromations ops (Transpose, "
            "Reshape etc), transform it to MemPermute and reorder the op to optimize final subgraph to avoid "
            "unnecessary data permutations",
            [](mlir::OpPassManager& pm, const MemPermutePositioningOptions& options) {
                IE::arch37xx::buildMemPermutePositioningPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ExpandActivationChannelsOptions>(
            "expand-and-optimize-activation-channels", "[OPTIMIZATION] Expand and optimize activation channels",
            [](mlir::OpPassManager& pm, const ExpandActivationChannelsOptions& options) {
                IE::arch37xx::buildExpandAndOptimizeActivationChannelsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ExpandActivationChannelsOptions>(
            "mempermute-processing",
            "[OPTIMIZATION] MemPermute processing is responsible for handling mempermute op and optimize final "
            "subgraph to avoid unnecessary data "
            "permutations",
            [](mlir::OpPassManager& pm, const ExpandActivationChannelsOptions& options) {
                IE::arch37xx::buildMemPermuteProcessingPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ExpandActivationChannelsOptions>(
            "optimize-mempermute-and-activation-channels-expand",
            "[OPTIMIZATION] Optimize MemPermute and activation channels expand",
            [](mlir::OpPassManager& pm, const ExpandActivationChannelsOptions& options) {
                IE::arch37xx::buildOptimizeMemPermuteAndActivationChannelsExpandPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<LowPrecisionOptions>(
            "low-precision", "[OPTIMIZATION] Low precision transformations",
            [](mlir::OpPassManager& pm, const LowPrecisionOptions& options) {
                IE::arch37xx::buildLowPrecisionPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<AdjustLayoutOptions>(
            "adjust-layout", "[LEGALIZATION] Adjust IR layout for VPU target",
            [](mlir::OpPassManager& pm, const AdjustLayoutOptions& options) {
                IE::arch37xx::buildAdjustLayoutPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions>(
            "dynamic-shape-transformations", "[LEGALIZATION] Introduces operation to handle dynamic shapes",
            [](mlir::OpPassManager& pm) {
                IE::arch37xx::buildDynamicShapeTransformationsPipeline(pm, DynamicShapeTransformOptions());
            });

    mlir::PassPipelineRegistration<IE::arch37xx::DefaultHWOptions>(
            "reference-sw-mode-ie", "IE dialect part of Reference SW pipeline",
            [](mlir::OpPassManager& pm, const IE::arch37xx::DefaultHWOptions& options) {
                IE::arch37xx::buildReferenceSWPipeline(pm, options);
            });
}
