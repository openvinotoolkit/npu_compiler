//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/locverif/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/Quant/Transforms/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// OptimizeViewLikeOps
//

void vpux::IE::buildOptimizeViewLikeOpsPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(IE::createRemoveViewLikeOpsChainPass(log));
    pm.addPass(mlir::createCSEPass());
    pm.addPass(IE::createUniquifySimilarOpsPass(log));
}

//
// DimensionAlignment
//

void vpux::IE::buildDimensionAlignmentPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(IE::createAlignDimensionsForDPUPass(log));
    pm.addPass(IE::createAdjustInputShapePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createPropagateAffineReshapePass(log));
    pm.addPass(IE::createOptimizeSliceExpandPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// OptimizeSliceOp
//

void vpux::IE::buildOptimizeSliceOpPipeline(mlir::OpPassManager& pm, bool disableSliceToConvMinHWThreshold,
                                            Logger log) {
    pm.addPass(IE::createFuseConvWithSlicePass(log));
    pm.addPass(IE::createOptimizeOpSlicePass(log));
    pm.addPass(IE::createConvertParallelSlicesToGatherPass(log));
    pm.addPass(IE::createOptimizeSliceWithStridePass(disableSliceToConvMinHWThreshold, log));
    // Note: createAdjustConvolutionShapePass is needed after slice optimizations
    // to fix convolution shapes if input slices were changed
    pm.addPass(IE::createAdjustConvolutionShapePass(log));
}

//
// BatchTransformationPipeline
//

void vpux::IE::buildBatchTransformationPipeline(mlir::OpPassManager& pm,
                                                const std::unique_ptr<BatchUnrollOptions>& batchUnrollOptions,
                                                Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(IE::createConvertBatchedLayerTo1NPass(log));
    pm.addPass(IE::createConvertBroadcastToTilePass(log));
    if (batchUnrollOptions) {
        pm.addPass(IE::createUnrollBatchPass(log, isOptionEnabled(batchUnrollOptions->skipUnrollBatch)));
    }
    pm.addPass(IE::createUpstreamSlicePass(log));
    pm.addPass(IE::createConvertBranchesConcatToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// SplitAndMapBilinearInterpolateOnDPUPipeline
//

void vpux::IE::buildSplitAndMapBilinearInterpolateOnDPUPipeline(
        mlir::OpPassManager& pm, const vpux::IE::SplitAndMapBilinearInterpolateOnDPUOptions& options, Logger log) {
    if (options.enableSEPtrsOperations && options.enableSplitBilinearIntoHAndW) {
        pm.addPass(IE::createSplitBilinearIntoHAndWPass(log));
    }

    pm.addPass(IE::createMapBilinearInterpolateOnDPUPass(log));
}

//
// ExpandAndOptimizeActivationChannels
//

void vpux::IE::buildExpandAndOptimizeActivationChannelsPipeline(mlir::OpPassManager& pm,
                                                                const ExpandActivationChannelsOptions& options,
                                                                Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    if (options.enableAdjustConvShapePass) {
        pm.addPass(IE::createOptimizeAvgPoolWithUnalignedChannelsPass(log));
        pm.addPass(IE::createAdjustConvolutionShapePass(log));
        pm.addPass(IE::createAdjustGroupConvShapePass(log));
    }
    pm.addPass(IE::createExpandActivationChannelsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(IE::createOptimizeSliceExpandPass(log));
    pm.addPass(IE::createAdjustConvolutionWeightsPass(log));
    pm.addPass(IE::createAdjustConvolutionInputShapePass(log));
    pm.addPass(IE::createAdjustInputShapePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createOptimizeSliceExpandPass(log));
    pm.addPass(mlir::createCSEPass());
    pm.addPass(IE::createUniquifySimilarOpsPass(log));
    pm.addPass(IE::createHandleEltwiseWithSmallHeightPass(log));
    pm.addPass(IE::createPropagateAffineReshapePass(log));
    pm.addPass(IE::createUniquifyBranchesPass(log));

    pm.addPass(IE::createPropagateExpandPass(log));
    pm.addPass(IE::createFusePermuteQuantizeExpandPass(log));

    pm.addPass(IE::createSwapOperationsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// OptimizeActivations
//

void vpux::IE::buildOptimizeActivationsPipeline(mlir::OpPassManager& pm, const OptimizeActivationsOptions& options,
                                                Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (options.enableSwapConvertWithSWOp) {
        pm.addPass(IE::createSwapConvertWithSWOpPass(log));
    }
    pm.addPass(IE::createRunF16ToF32ConvertOnDPUPass(log));

    pm.addPass(IE::createOptimizeActivationsPipelineRewriterExecutorPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createOptimizeTileOpPass(log));
}

//
// InitialTransformations
//

void vpux::IE::buildInitialTransformationsPipeline(mlir::OpPassManager& pm, const IE::TransformOptions& options,
                                                   Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createFuseDynamicQuantizePass(log));
    pm.addPass(IE::createFuseRMSNormPass(log));
    pm.addPass(mlir::createCSEPass());
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createOptimizeParallelLayersPass(log));
    pm.addPass(IE::createDecomposeISTFTPass(log));
    pm.addPass(IE::createDecomposeLSTMSequencePass(log));
    if (options.enableDecomposeGRUSequence) {
        pm.addPass(IE::createDecomposeGRUSequencePass(log));
    }
    pm.addPass(IE::createDecomposeLSTMCellPass(log));
    pm.addPass(IE::createDecomposeGRUCellPass(log));
    pm.addPass(IE::createDecomposeL2OpsPass(log));
    pm.addPass(IE::createDecomposeSoftPlusPass(log));
    pm.addPass(IE::createFuseRoPEPass(log));
    pm.addPass(IE::createReshapeMatMulInputsPass(options.enableGroupedMatMul, log));
    pm.addPass(IE::createSwishFusionPass(log));
    pm.addPass(IE::createFuseColorConversionPass(options.enableYuvToRgbShaveScale, log));
    pm.addPass(IE::createEltwiseFakeQuantizeFusionPass(log));
    pm.addPass(IE::createChunkGatedDeltaRuleLoopPass(log));
    pm.addPass(IE::createUnrollTensorIteratorPass(log));
    pm.addPass(IE::createNormalizeL2FusionPass(log));
    pm.addPass(IE::createMVNFusionPass(log));
    pm.addPass(IE::createFuseOneHotSelectPass(log));
    pm.addPass(IE::createFuseOpsToMatMulPass(options.enableGroupedMatMul, options.enableConvertCumSumToMatMul, log));
    pm.addPass(IE::createDecomposeSTFTPass(log));
    if (options.enableConvertFFTToConv) {
        pm.addPass(IE::createConvertFFTToConvPass(log));
    }
    pm.addPass(IE::createMoveMultiplyDividePostOpPass(log));
    pm.addPass(IE::createFuseNormalizeL2ToRMSPass(log));
    pm.addPass(IE::createUnrollSDPAPatternPass(log));
    pm.addPass(IE::createShrinkMatmulGroupsPass(log));
    pm.addPass(IE::createBatchOpProcessingPipelineRewriterExecutorPass(options, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertMVN6ToMVN1Pass(log));
    pm.addPass(IE::createConvertSubGRUSequenceToConvPass(log));
    pm.addPass(IE::createLegalizeConvBackpropDataPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDilatedConvConvertPass(log));
    pm.addPass(IE::createConvertSelectToEltwisePass(log));
}

//
// AdjustShape
//

void vpux::IE::buildAdjustShapePipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    pm.addPass(IE::createReshapeMaxPoolPass(log));
    pm.addPass(IE::createAdjustMaxPoolInputShapePass(log));
    pm.addPass(IE::createAdaptShapesForScaleShiftPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
}

//
// AdjustLayout
//

void vpux::IE::buildAdjustLayoutPipeline(mlir::OpPassManager& pm, const AdjustLayoutOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createSwapMVNWithTransposePass(log));

    if (options.enableForceZMajorConcat) {
        pm.addPass(IE::createInsertReorderBetweenLayerAndConcatPass(log));
    }

    pm.addPass(IE::createPropagateAffineReshapePass(log));
    pm.addPass(IE::createPropagateTransposePass(log));
    pm.addPass(IE::createUniquifyBranchesPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createSwapTransposeConcatPass(log));
    pm.addPass(IE::createTransposeToPermuteCastPass(log));
    pm.addPass(IE::createAdjustLayoutsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableOptimizeReorders) {
        pm.addPass(IE::createFuseReshapeMvnPass(log));
        pm.addPass(IE::createOptimizeReordersPass(log));
        pm.addPass(IE::createOptimizeReordersAcrossFunctionCallsPass(log));
        pm.addPass(mlir::createCSEPass());
        pm.addPass(IE::createUniquifyBranchesPass(log));
        pm.addPass(IE::createPropagateReorderToNCEPass(log));
        pm.addPass(IE::createFuseReordersPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }
}

void vpux::IE::buildDynamicShapeTransformationsPipeline(mlir::OpPassManager& pm,
                                                        const DynamicShapeTransformOptions& options, Logger log) {
    pm.addPass(IE::createOptDynamicEltwiseWithShapeOfPass(log));
    pm.addPass(IE::createPopulateDynamicDimensionsHWPass(log));
    pm.addPass(IE::createPopulateDynamicDimensionsGenericPass(log));
    if (isOptionEnabled(options.enableApplyDynamicBoundaryCorrection)) {
        pm.addPass(IE::createApplyDynamicBoundaryCorrectionPass(log));
    }
    // The verifier is disabled before the MLIR pass which cannot be adjusted and then re-enabled after the
    // locations are fixed to ensure their uniqueness in this part of the pipeline.
    pm.addPass(locverif::createStopLocationVerifierPass(log));
    pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
    pm.addPass(IE::createFixDynamicOpsLocationsPass(log));
    pm.addPass(locverif::createStartLocationVerifierPass(log, options.locationsVerificationMode));
    pm.addPass(IE::createLegalizeReifyResultShapesResidualsPass(log));
    pm.addPass(IE::createPadDynamicInputsPass(log));
}

void vpux::IE::buildOptimizeMemPermuteAndActivationChannelsExpandPipeline(
        mlir::OpPassManager& pm, const ExpandActivationChannelsOptions& options, Logger log) {
    pm.addPass(IE::createFusePermuteQuantizePass(true, log));
    pm.addPass(IE::createConvertReorderToPermuteQuantizePass(log));
    IE::buildMemPermutePositioningPipeline(pm, IE::MemPermutePositioningOptions(options), log);
    IE::buildExpandAndOptimizeActivationChannelsPipeline(pm, options, log);
    IE::buildMemPermuteProcessingPipeline(pm, options, log);
}

void vpux::IE::buildOutliningPipeline(mlir::OpPassManager& pm, const DefaultHWOptionsBase& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    bool isOutliningEnabled = options.functionOutlining.hasValue();
    if (!canOutlineFromProfilingPerspective(options)) {
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
}

void vpux::IE::buildDebatcherPipeline(mlir::OpPassManager& pm, const vpux::BatchCompileOptionsAdapter& options,
                                      Logger log) {
    log.debug("Checks whether DebatcherOptions available or not");
    auto debatcherOptionsPtr = DebatcherOptions::create(options);
    if (debatcherOptionsPtr == nullptr) {
        return;
    }

    log.info("Will proceed with the options for batch processing: {0}", debatcherOptionsPtr->to_string());
    pm.addPass(IE::createDebatcherPass(*debatcherOptionsPtr, log));
    DefaultHWOptionsBase batchingOutlining;
    batchingOutlining.functionOutlining = "batching=";
    pm.addPass(IE::createOutlinerPass(batchingOutlining, log));
    pm.addPass(IE::createDeDebatcherPass(*debatcherOptionsPtr, log));
    if (auto reorderingOptionsPtr = IE::DebatcherOpReorderingOptions::create(*debatcherOptionsPtr, log)) {
        pm.addPass(IE::createOverrideTileExecutorNumPass(*reorderingOptionsPtr, log));
    }
}

//
// MemPermutePositioning
//

void vpux::IE::buildMemPermutePositioningPipeline(mlir::OpPassManager& pm, const MemPermutePositioningOptions& options,
                                                  Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createPropagateMemPermuteThroughSoftMaxPass(log));
    pm.addPass(IE::createOptimizeReduceOpsWithMemPermutePass(options.enableFuseReduceMinMaxToDpu, log));
    if (options.enableMovePermutePostEltwise) {
        pm.addPass(IE::createMovePermutePostEltwisePass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createLegalizeNDMemPermutePass(log));
    pm.addPass(IE::createPropagateMemPermuteBeforeOpPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enablePropagateMemPermuteThroughEltwise) {
        pm.addPass(IE::createPropagateMemPermuteThroughEltwisePass(log));
    }
    pm.addPass(mlir::createCSEPass());
    pm.addPass(IE::createUniquifySimilarOpsPass(log));
    if (options.enableAdjustMemPermuteAroundOp) {
        pm.addPass(IE::createAdjustMemPermuteAroundOpPass(log));
    }
}

//
// MemPermuteProcessing
//

void vpux::IE::buildMemPermuteProcessingPipeline(mlir::OpPassManager& pm,
                                                 const ExpandActivationChannelsOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createMemPermuteProcessingPipelineRewriterExecutorPass(options, log));
    pm.addPass(IE::createConvertMemPermuteToOpPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(mlir::createCSEPass());
    pm.addPass(IE::createUniquifySimilarOpsPass(log));
}

//
// PostImport
//

void vpux::IE::buildPostImportPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(IE::createInputQuantizationRestorationPass(log));
    pm.addPass(IE::createPropagateAndCleanUpFQPass(log));
    pm.addPass(IE::createDecomposeMatMulThroughSlicePass(log));
    pm.addPass(IE::createConvertVariadicSplitToStridedSlicePass(log));
}

//
// AdjustPrecision
//

void vpux::IE::buildAdjustPrecisionPipeline(mlir::OpPassManager& pm, const AdjustPrecisionOptions& options,
                                            Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createConvertPrecisionToFPPass(log, options.computeLayersWithHigherPrecision));
    pm.addPass(IE::createConvertPrecisionToI32Pass(log));
    pm.addPass(IE::createUseUserPrecisionPass(log));
    pm.addPass(IE::createAdjustSoftwareOpsPrecisionPass(log));
    pm.addPass(IE::createAdjustNCEOpsWithI32InputsPass(log, options.enableConvertFCToConv));
    pm.addPass(IE::createLegalizeEpsilonUsagePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// AdjustForVPU
//
// E-184685: Group more passes into this pipeline, rename it into a proper name
void vpux::IE::buildAdjustForVPUPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // passes using walk drivers or conversion drivers
    // First run: legalize pre-existing dilated Conv/GroupConv before downstream conversion passes.
    pm.addPass(IE::createLegalizeDilatedConvolutionPass(log));
    pm.addPass(IE::createConvertPaddingsToFloorModePass(log));
    pm.addPass(IE::createConvertNearestToBroadCastOrStridedConcatPass(log));
    pm.addPass(IE::createConvertBilinearToStridedConcatAndConvPass(log));
    pm.addPass(IE::createConvertBroadcastToTilePass(log));
    pm.addPass(IE::createConvertScatterPass(log));
    pm.addPass(IE::createConvertTransposedConv2DToConv2DPass(log));
    pm.addPass(IE::createConvertGroupTransposedConvToGroupConvPass(log));
    // Second run: legalize dilated Conv/GroupConv ops introduced by the two conversion passes above (E#222712).
    pm.addPass(IE::createLegalizeDilatedConvolutionPass(log));
    pm.addPass(IE::createConvertGroupTransposedConvToTransposedConvPass(log));
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(IE::createConvertUpsamplingToStridedConcatPass(log));
    pm.addPass(IE::createConvertNegativePadToSlicePass(log));
    pm.addPass(IE::createConvertNonConstantPadToSliceAndConcatPass(log));

    // passes using rewriter drivers
    pm.addPass(vpux::IE::createAdjustForVPUPipelineRewriterExecutorPass(log));

    // E-184685: Relocate the following passes to other pipelines
    pm.addPass(IE::createConvertPadToConcatPass(log));
    pm.addPass(IE::createConvertDepth2SpaceLayerPass(log));
    pm.addPass(IE::createConvertSpace2DepthLayerPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createOptimizeOpSlicePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::IE::buildScaleShiftProcessingPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createAdjustScaleShiftForDWConvPass(log));
    pm.addPass(IE::createConvertBroadcastToTilePass(log));
    pm.addPass(IE::createConvertScaleShiftToDWPass(log));

    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::IE::buildOperationConversionPipeline(mlir::OpPassManager& pm, const IE::OperationConversionOptions& options,
                                                Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    // Resolve group quant MatMul pattern
    pm.addPass(mlir::createCSEPass());
    pm.addPass(IE::createMergeParallelFullyConnectedPass(log));

    // this pass is added here so that every DynamicDequant op will have a quantized input.
    // For now Dequantize op can only receive quantized inputs, with the current refactoring ongoing (Deq->DynamicDeq
    // and quant inputs -> raw data types) there were regressions observed because ConvertDynamicDequantizeToDequantize
    // received raw data types at input and needed to insert a QuantCast for Dequant. This led to
    // MergeFullyConnectedPass not being able to merge the FCs because of the QuantCast in between. The consensus
    // reached was that there should be no QuantCast inserted for Dequantize op, and the op should be extended to also
    // receive raw data types through which probably it assumes that they are dummy quant casts (scale=1 zp =0). This is
    // a temporary solution until the refactoring is completed. Ticket to address this issue:
    // TODO #E-225955
    pm.addPass(IE::createConvertDQRawDataTypeToQuantizedPass(log));

    pm.addPass(IE::createUnrollGroupQuantizePass(log));
    pm.addPass(IE::createUnrollFullyConnectedPass(log));
    if (options.convertDynamicDequantize) {
        pm.addPass(IE::createConvertDynamicDequantizeToDequantizePass(log));
    }
    pm.addPass(IE::createMoveMultiplyDividePostOpPass(log));
    pm.addPass(IE::createSwapOperationsWithGatherAndSlicePass(log));
    pm.addPass(IE::createMergeFullyConnectedPass(isOptionEnabled(options.mergeUnrolledMatmulForLargeOC), log));

    pm.addPass(IE::createConvertMatMulToConvPass(log));
    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }

    pm.addPass(IE::createDecomposeConcatMatMulPass(log));
    pm.addPass(IE::createConvertExtractImagePatchesPass(log));
    pm.addPass(IE::createSwapEltwiseAndReducePass(log));
    pm.addPass(IE::createConvertReduceSumToConvPass(log));
    pm.addPass(IE::createUnrollReduceMinAllAxesPass(log));
    pm.addPass(IE::createConvertReduceToPoolingPass(options.enableFuseReduceMinMaxToDpu, log));
    pm.addPass(IE::createConvertPowerToMultPass(log));
    pm.addPass(IE::createConvertGatherPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::IE::buildSplitLargeOpsPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(IE::createUnrollConv3dToConv2dPass(log));
    pm.addPass(IE::createSplitInterpolateAxesPass(log));
    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    pm.addPass(IE::createHandleLargeKernelsPass(log));
}

void vpux::IE::buildConvertToEfficientOpsPipeline(mlir::OpPassManager& pm, const ConvertToEfficientOpsOptions& options,
                                                  Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createConvertDivideToMultiplyPass(log));
    // NOTE: ReassociateMultiply relies on ConvertDivideToMultiply
    pm.addPass(IE::createReassociateMultiplyPass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(isOptionEnabled(options.forceConvertGatherTo4D), log));
    // NOTE: Canonicalizer required after ConvertShapeTo4DPass
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertGatherElementsToGatherPass(log));
    pm.addPass(IE::createConvertToSpatialOpPass(false, log));
    pm.addPass(IE::createConvertSubtractToAddPass(log));
    pm.addPass(IE::createSwapTransposeConcatPass(log));
    pm.addPass(IE::createConvertSplitConcatToAffineReshapePass(log));
    pm.addPass(IE::createConvertBranchesConcatToConvPass(log));
    pm.addPass(IE::createSwapOperationsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createSwapPadLayerPass(log));
    // NOTE: apply FuseScale after ConvertDivideToMultiply to increase
    // the applicability
    pm.addPass(IE::createFuseScalePass(log));
    pm.addPass(IE::createSwapOperationsPass(log));
    pm.addPass(IE::createBroadcastInputForAddPass(log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    pm.addPass(IE::createConvertToScaleShiftPass(/*enableNCEEltwiseMultiply=*/false, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    // NOTE: Required to avoid performance regression related to AddOp
    pm.addPass(IE::createSwapOperationsPass(log));
    if (options.enableD2SToTransposedConvConversion) {
        pm.addPass(IE::createConvertDepth2SpaceToTransposedConvPass(log));
    }
    // NOTE: SwapD2SAndScaleShift depends on ConvertDepth2SpaceToTransposedConv
    pm.addPass(IE::createSwapD2SAndScaleShiftPass(log));
    pm.addPass(IE::createConvertReverseToDWConvPass(log));
}

void vpux::IE::buildHandleHyperParametersPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createHandleLargeKernelsPass(log));
    pm.addPass(IE::createHandleLargeStridesPass(log));
    pm.addPass(IE::createHandleLargePadsPass(log));
}

void vpux::IE::buildReorderFakeQuantizePipeline(mlir::OpPassManager& pm, const ReorderFakeQuantizeOptions& options,
                                                Logger log) {
    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    pm.addPass(IE::createSplitConvWithMultipleFQPass(log));
}

void vpux::IE::buildConvertToConvolutionPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(IE::createConvertStridedSlice2ConvPass(log));
}

//
// registerIEPipelines
//

void vpux::IE::registerIEPipelines() {
    mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions>(
            "post-import",
            "[LEGALIZATION] The post import pipeline contains passes that were historically ngraph passes. It's "
            "considered a legalization step because it converts the imported IR into an IR format that is supported by "
            "the other passes. No other passes should be run before it (with very few exceptions).",
            [](mlir::OpPassManager& pm) {
                IE::buildPostImportPipeline(pm);
            });

    mlir::PassPipelineRegistration<AdjustPrecisionOptions>(
            "adjust-precision", "[LEGALIZATION] Adjust IR precision for VPU target",
            [](mlir::OpPassManager& pm, const AdjustPrecisionOptions& options) {
                IE::buildAdjustPrecisionPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions>(
            "adjust-for-vpu", "[LEGALIZATION] Adjust IE Dialect IR for VPU target", [](mlir::OpPassManager& pm) {
                IE::buildAdjustForVPUPipeline(pm);
            });

    mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions>(
            "scaleshift-processing",
            "[OPTIMIZATION] scaleshift processing is responsible for handling scaleshift ops, transform it to"
            "depthwise convolution and optimize final subgraph to run more efficiently",
            [](mlir::OpPassManager& pm) {
                IE::buildScaleShiftProcessingPipeline(pm);
            });

    mlir::PassPipelineRegistration<OperationConversionOptions>(
            "operation-conversion",
            "[OPTIMIZATION] Operation Conversion pipeline is responsible for changing type of existing operations."
            "Main purpose is reducing subset of ops"
            "which using in our graph for improve pattern matching of next passes ",
            [](mlir::OpPassManager& pm, const OperationConversionOptions& options) {
                IE::buildOperationConversionPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<IE::TransformOptions>(
            "initial-transformations",
            "[LEGALIZATION] Initial Transformations, convert initial IR operations to another and tries to reduce the "
            "number of op types used in the graph",
            [](mlir::OpPassManager& pm, const IE::TransformOptions& options) {
                IE::buildInitialTransformationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<OptimizeActivationsOptions>(
            "optimize-activations", "[OPTIMIZATION] Optimize activations for VPU target",
            [](mlir::OpPassManager& pm, const OptimizeActivationsOptions& options) {
                IE::buildOptimizeActivationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<MemPermutePositioningOptions>(
            "mempermute-positioning",
            "[OPTIMIZATION] MemPermute positioning is responsible for handling data transfromations ops (Transpose, "
            "Reshape etc), transform it to MemPermute and reorder the op to optimize final subgraph to avoid "
            "unnecessary data permutations",
            [](mlir::OpPassManager& pm, const MemPermutePositioningOptions& options) {
                IE::buildMemPermutePositioningPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ExpandActivationChannelsOptions>(
            "expand-and-optimize-activation-channels", "[OPTIMIZATION] Expand and optimize activation channels",
            [](mlir::OpPassManager& pm, const ExpandActivationChannelsOptions& options) {
                IE::buildExpandAndOptimizeActivationChannelsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ExpandActivationChannelsOptions>(
            "mempermute-processing",
            "[OPTIMIZATION] MemPermute processing is responsible for handling mempermute op and optimize final "
            "subgraph to avoid unnecessary data "
            "permutations",
            [](mlir::OpPassManager& pm, const ExpandActivationChannelsOptions& options) {
                IE::buildMemPermuteProcessingPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ExpandActivationChannelsOptions>(
            "optimize-mempermute-and-activation-channels-expand",
            "[OPTIMIZATION] Optimize MemPermute and activation channels expand",
            [](mlir::OpPassManager& pm, const ExpandActivationChannelsOptions& options) {
                IE::buildOptimizeMemPermuteAndActivationChannelsExpandPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<AdjustLayoutOptions>(
            "adjust-layout", "[LEGALIZATION] Adjust IR layout for VPU target",
            [](mlir::OpPassManager& pm, const AdjustLayoutOptions& options) {
                IE::buildAdjustLayoutPipeline(pm, options);
            });
}
