//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::IE::arch40xx::buildLowPrecisionPipeline(mlir::OpPassManager& pm, const LowPrecisionOptions& options,
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
// DefaultHWPipeline
//

void vpux::IE::arch40xx::buildDefaultHWPipeline(mlir::OpPassManager& pm, const IE::arch40xx::DefaultHWOptions& options,
                                                Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(Core::createStartLocationVerifierPass(log, options.locationsVerificationMode));

    // Blob compilation using 'debatcher' method leverages 'outlining' feature so that
    // it will be turned on unless it was already enabled
    bool isOutliningEnabled = options.functionOutlining.hasValue() || DebatcherOptions::isAvailable(options);
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

        auto debatcherOptionsPtr = DebatcherOptions::create(options);
        if (debatcherOptionsPtr) {
            pm.addPass(IE::createDebatcherPass(*debatcherOptionsPtr, log));
            pm.addPass(IE::createOutlinerPass(options, log));
            pm.addPass(IE::createDeDebatcherPass(*debatcherOptionsPtr, log));
            if (auto reorderingOptionsPtr = IE::DebatcherOpReorderingOptions::create(*debatcherOptionsPtr, log)) {
                pm.addPass(IE::createOverrideTileExecutorNumPass(*reorderingOptionsPtr, log));
            }
        } else {
            pm.addPass(IE::createOutlinerPass(options, log));
            pm.addPass(IE::createDuplicateFQAcrossFunctionCallsPass(log));
        }
    }

    // No passes should be run before this pipeline, with very few exceptions.
    IE::buildPostImportPipeline(pm, log);
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableReduceNumTilesForSmallModelsPass) {
        pm.addPass(IE::createReduceNumTilesForSmallModelsPass(log));
    }

    // Level 3 : Topology
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }

    if (options.enableFlashSDPAConversion) {
        pm.addPass(IE::createConvertSDPAToFlashSDPAPass(log));
    }

    if (options.enableDynamicShapeTransformationsPipeline) {
        IE::buildDynamicShapeTransformationsPipeline(pm, IE::DynamicShapeTransformOptions(options), log);
    }
    pm.addPass(IE::createReshapeMatMulInputsPass(options.enableGroupedMatMul, log));
    IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::buildInitialTransformationsPipeline(pm, IE::TransformOptions(options), log);

    if (options.enableAdjustPrecisionPipeline) {
        IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);
    }

    IE::buildOperationConversionPipeline(pm, IE::OperationConversionOptions(options), log);

    if (options.enableM2I) {
        pm.addPass(IE::createM2IBatchNormFusionPass());
    }

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
    pm.addPass(IE::createConvertShapeTo4DPass(isOptionEnabled(options.forceConvertGatherTo4D), log));
    pm.addPass(IE::createSplitInterpolateAxesPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertGatherElementsToGatherPass(log));
    //  [Tracking number: E#101595]
    // This temporary check is necessary for m2i interpolate functional tests and it will be removed as part of
    // E#101595
    pm.addPass(IE::createConvertToSpatialOpPass(isOptionEnabled(options.enableM2I),
                                                isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertSubtractToAddPass(log));
    pm.addPass(IE::createConvertBranchesConcatToConvPass(log));
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    pm.addPass(IE::createSwapPadLayerPass(log));
    // Note: apply FuseStaticScale after ConvertDivideToMultiply to increase
    // the applicability
    pm.addPass(IE::createFuseStaticScalePass(log));
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    pm.addPass(IE::createBroadcastInputForAddPass(log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    // E#79878: Solve eltwise single layer test failure.
    // SwapOperations pass may generate non-4D AddOp.
    // If AddOp appears here means that it cannot be fused into NCE task.
    // So convert it's shape to 4D and then convert this AddOp to ScaleShift.
    pm.addPass(IE::createConvertShapeTo4DPass(isOptionEnabled(options.forceConvertGatherTo4D), log));
    pm.addPass(IE::createConvertToScaleShiftPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));

    if (options.enableD2SToTransposedConvConversion) {
        pm.addPass(IE::createConvertDepth2SpaceToTransposedConvPass(log));
    }
    pm.addPass(IE::createSwapD2SAndScaleShiftPass(log));
    pm.addPass(IE::createConvertReverseToDWConvPass(log));
    if (options.enableConvertDeformableConvToConv) {
        pm.addPass(IE::createConvertDeformableConvToConvPass(log));
    }

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    pm.addPass(mlir::createCSEPass());

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
    pm.addPass(Core::createStopLocationVerifierPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableOptimizeScaleShiftToDWConv) {
        IE::buildScaleShiftProcessingPipeline(pm, log);
    }

    pm.addPass(IE::createFuseActivationOpsPass(options.enableFuseClampOperations, log));
    pm.addPass(IE::createConvertStridedSlice2ConvPass(log));
    if (options.enableLowPrecision) {
        IE::arch40xx::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
        pm.addPass(IE::createConvertShapeTo4DPass(isOptionEnabled(options.forceConvertGatherTo4D), log));
        pm.addPass(IE::createSwapViewOpAndClampPass(log));
    }
    IE::buildOptimizeActivationsPipeline(pm, IE::OptimizeActivationsOptions(options), log);

    pm.addPass(IE::createOptimizeTileOpPass(log));

    if (options.enableSEPtrsOperations && options.enableSplitBilinerIntoHAndW) {
        pm.addPass(IE::createSplitBilinerIntoHAndWPass(log));
    }

    if (options.enableBilinearInterpolateOnDPU) {
        pm.addPass(IE::createMapBilinearInterpolateOnDPUPass(isOptionEnabled(options.enableSEPtrsOperations), log));
    }

    pm.addPass(IE::createConvertBatchedLayerTo1NPass(log));
    pm.addPass(IE::createConvertBroadcastToTilePass(log));

    if (auto batchUnrollOptions = BatchUnrollOptions::create(options, log); batchUnrollOptions != nullptr) {
        pm.addPass(IE::createUnrollBatchPass(log, isOptionEnabled(batchUnrollOptions->skipUnrollBatch)));
    }

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }

    pm.addPass(IE::createConvertBranchesConcatToConvPass(log));

    pm.addPass(IE::createSwapMVNWithTransposePass(log));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    pm.addPass(IE::createFuseConvWithSlicePass(log));

    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    if (options.enableFusePermuteQuantize) {
        pm.addPass(IE::createFusePermuteQuantizePass(true, log));
        pm.addPass(IE::createConvertReorderToPermuteQuantizePass(log));
    }

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createAdjustGroupConvShapePass(log));
    }

    IE::buildOptimizeMemPermuteAndActivationChannelsExpandPipeline(pm, IE::ExpandActivationChannelsOptions(options),
                                                                   log);

    pm.addPass(IE::createRemoveViewLikeOpsChainPass(log));
    pm.addPass(IE::createOptimizeOpSlicePass(log));
    pm.addPass(IE::createConvertParallelSlicesToGatherPass(log));
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
    pm.addPass(IE::createLoadExternalKernelResourcesPass(log));

    if (options.enableShaveCodeGen) {
        IE::buildShaveCodeGenPipeline(pm, log);
    }

    if (options.enableFuseD2SExpand) {
        pm.addPass(IE::createFuseD2SExpandChannelsPass(log));
    }
}

void vpux::IE::arch40xx::buildReferenceSWPipeline(mlir::OpPassManager& pm,
                                                  const IE::arch40xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // No passes should be run before this pipeline, with very few exceptions.
    IE::buildPostImportPipeline(pm, log);

    // Level 3 : Topology

    IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::buildInitialTransformationsPipeline(pm, IE::TransformOptions(options), log);
    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    // Resolve group quant MatMul pattern
    pm.addPass(IE::createUniquifyOpsPass(log));
    pm.addPass(IE::createMergeParallelFullyConnectedPass(log));
    pm.addPass(IE::createUnrollGroupQuantizePass(log));
    pm.addPass(IE::createUnrollFullyConnectedPass(log));
    pm.addPass(IE::createMergeFullyConnectedPass(log));
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
    pm.addPass(IE::createConvertShapeTo4DPass(isOptionEnabled(options.forceConvertGatherTo4D), log));
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

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableShaveCodeGen) {
        IE::buildShaveCodeGenPipeline(pm, log);
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
