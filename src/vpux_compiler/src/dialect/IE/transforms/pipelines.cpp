//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// ExpandAndOptimizeActivationChannels
//

void vpux::IE::buildExpandAndOptimizeActivationChannelsPipeline(mlir::OpPassManager& pm,
                                                                const ExpandActivationChannelsOptions& options,
                                                                Logger log) {
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

//
// OptimizeActivations
//

void vpux::IE::buildOptimizeActivationsPipeline(mlir::OpPassManager& pm, const OptimizeActivationsOptions& options,
                                                Logger log) {
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

//
// InitialTransformations
//

void vpux::IE::buildInitialTransformationsPipeline(mlir::OpPassManager& pm, const IE::TransformOptions& options,
                                                   Logger log) {
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
    pm.addPass(IE::createDecomposeL2OpsPass(log));
    pm.addPass(IE::createReshapeMatMulInputsPass(options.enableGroupedMatMul, log));
    pm.addPass(IE::createAdjustFakeQuantizeParamsPass(log));
    pm.addPass(IE::createAdjustFakeQdqParamsPass(log));
    pm.addPass(IE::createFuseFQAndMulPass(options.fuseFQAndMulWithNonConstInput, log));
    pm.addPass(IE::createHandleU16FakeQuantizePass(log));
    pm.addPass(IE::createSwishFusionPass(log));
    pm.addPass(IE::createFuseSoftmaxPass(log));
    pm.addPass(IE::createFuseColorConversionPass(log));
    if (options.enableConvertToSdpaExtended) {
        pm.addPass(IE::createFuseSDPAExtendedPass(log));
    }
    pm.addPass(IE::createFuseSDPAPass(log));
    pm.addPass(IE::createFuseRoPEPass(log));
    pm.addPass(IE::createEltwiseFakeQuantizeFusionPass(log));
    pm.addPass(IE::createUnrollTensorIteratorPass(log));
    pm.addPass(IE::createNormalizeL2FusionPass(log));
    pm.addPass(IE::createMVNFusionPass(log));
    pm.addPass(IE::createFuseReduceMeanSquarePass(log));
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
    pm.addPass(IE::createLegalizeConvBackpropDataPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDilatedConvConvertPass(log));
}

//
// AdjustLayout
//

void vpux::IE::buildAdjustLayoutPipeline(mlir::OpPassManager& pm, const AdjustLayoutOptions& options, Logger log) {
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

void vpux::IE::buildDynamicShapeTransformationsPipeline(mlir::OpPassManager& pm,
                                                        const DynamicShapeTransformOptions& options, Logger log) {
    pm.addPass(IE::createOptDynamicEltwiseWithShapeOfPass(log));
    pm.addPass(IE::createPopulateDynamicDimensionsHWPass(log));
    pm.addPass(IE::createPopulateDynamicDimensionsGenericPass(log));
    if (isOptionEnabled(options.enableApplyDynamicBoundaryCorrection)) {
        // ApplyDynamicBoundaryCorrectionPass creates operations that are later processed by the MLIR pass, which
        // results in creation of duplicate locations. All duplicate operations are removed later in the pipeline after
        // the CSE pass.
        pm.addPass(IE::createApplyDynamicBoundaryCorrectionPass(log));
        // The verifier is disabled before the MLIR pass which cannot be adjusted and then re-enabled after the
        // locations are fixed to ensure their uniqueness in this part of the pipeline.
        pm.addPass(Core::createStopLocationVerifierPass(log));
        pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
        pm.addPass(IE::createFixDynamicOpsLocationsPass(log));
        pm.addPass(Core::createStartLocationVerifierPass(log, options.locationsVerificationMode));
    } else {
        pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
    }
    pm.addPass(IE::createLegalizeReifyResultShapesResidualsPass(log));
    pm.addPass(IE::createPadDynamicInputsPass(log));
    pm.addPass(IE::createDynamicConcatToScatterNDUpdatePass(log));
}

void vpux::IE::buildOptimizeMemPermuteAndActivationChannelsExpandPipeline(
        mlir::OpPassManager& pm, const ExpandActivationChannelsOptions& options, Logger log) {
    IE::buildMemPermutePositioningPipeline(pm, IE::MemPermutePositioningOptions(options), log);
    IE::buildExpandAndOptimizeActivationChannelsPipeline(pm, options, log);
    IE::buildMemPermuteProcessingPipeline(pm, options, log);
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
    pm.addPass(IE::createMovePermutePostEltwisePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createLegalizeNDMemPermutePass(log));
    pm.addPass(IE::createPropagateMemPermuteBeforeOpPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enablePropagateMemPermuteThroughEltwise) {
        pm.addPass(IE::createPropagateMemPermuteThroughEltwisePass(log));
    }
    pm.addPass(IE::createUniquifyOpsPass(log));
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
    pm.addPass(IE::createSwapMemPermuteAndExpandPass(log));
    pm.addPass(IE::createPropagateMemPermuteBeforeOpPass(log));
    pm.addPass(IE::createOptimizeConcatWithConvPass(log));
    if (options.enableAdjustConvShapePass) {
        // For such Convolution which used to optimize Concat need to be processed by AdjustConvolutionShapePass for
        // efficiency.
        pm.addPass(IE::createAdjustConvolutionShapePass(log));
    }
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    pm.addPass(IE::createInsertIdentityPoolBeforeOpPass(log));
    pm.addPass(IE::createOptimizeInnermostConcatPass(log));
    pm.addPass(IE::createFuseMemPermutePass(log));
    pm.addPass(IE::createConvertMemPermuteToOpPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createUniquifyOpsPass(log));
}

//
// PostImport
//

void vpux::IE::buildPostImportPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(IE::createInputQuantizationRestorationPass(log));
    pm.addPass(IE::createPropagateAndCleanUpFQPass(log));
    pm.addPass(IE::createConvertVariadicSplitToStridedSlicePass(log));
}

//
// AdjustPrecision
//

void vpux::IE::buildAdjustPrecisionPipeline(mlir::OpPassManager& pm, const AdjustPrecisionOptions& options,
                                            Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (options.enableConvertPrecisionToFP16) {
        pm.addPass(IE::createConvertPrecisionToFP16Pass(log, options.computeLayersWithHigherPrecision));
    }
    pm.addPass(IE::createConvertPrecisionToI32Pass(log));
    pm.addPass(IE::createUseUserPrecisionPass(log));
    pm.addPass(IE::createAdjustSoftwareOpsPrecisionPass(log));
    pm.addPass(IE::createAdjustNCEOpsWithI32InputsPass(log, options.enableConvertFCToConv));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// AdjustForVPU
//
// E-184685: Group more passes into this pipeline, rename it into a proper name
void vpux::IE::buildAdjustForVPUPipeline(mlir::OpPassManager& pm, const AdjustForVPUOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // passes using walk drivers or conversion drivers
    pm.addPass(
            IE::createLegalizeDilatedConvolutionPass(isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
    pm.addPass(IE::createConvertPaddingsToFloorModePass(log));
    pm.addPass(IE::createConvertNearestToBroadCastOrStridedConcatPass(
            /*interpolateAsSEOp=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertBilinearToStridedConcatAndConvPass(
            /*interpolateAsSEOp=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertBroadcastToTilePass(log));
    pm.addPass(IE::createConvertScatterNDUpdateToStridedConcatPass(log));
    pm.addPass(IE::createConvertTransposedConv2DToConv2DPass(
            /*enableSEPTransposedConv=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertGroupTransposedConvToGroupConvPass(
            /*enableSEPTransposedConv=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertGroupTransposedConvToTransposedConvPass(
            /*enableSEPTransposedConv=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(IE::createConvertUpsamplingToStridedConcatPass(log));
    if (options.enableConvertNonConstantPadToSliceAndConcat) {
        pm.addPass(IE::createConvertNonConstantPadToSliceAndConcatPass(
                /*enableSEPPad=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    }

    // passes using rewriter drivers
    pm.addPass(vpux::IE::createAdjustForVPUPipelineRewriterExecutorPass(options, log));

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
    pm.addPass(IE::createUniquifyOpsPass(log));
    pm.addPass(IE::createMergeParallelFullyConnectedPass(log));
    pm.addPass(IE::createUnrollGroupQuantizePass(log));
    pm.addPass(IE::createUnrollFullyConnectedPass(log, options.accumulateMatmulWithDPU));
    pm.addPass(IE::createConvertDynamicDequantizeToDequantizePass(log));
    pm.addPass(IE::createMoveMultiplyDividePostOpPass(log));
    pm.addPass(IE::createSwapOperationWithGatherPass(log));
    if (options.mergeUnrolledMatmul) {
        pm.addPass(IE::createMergeFullyConnectedPass(log));
    }
    if (options.fuseScalesToAccumulate) {
        pm.addPass(IE::createFuseScalesToAccumulatePass(log));
    }

    pm.addPass(IE::createConvertMatMulToConvPass(log));
    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }

    pm.addPass(IE::createDecomposeConcatMatMulPass(log));
    pm.addPass(IE::createConvertExtractImagePatchesPass(log));
    pm.addPass(IE::createConvertReduceSumToConvPass(log));
    pm.addPass(IE::createUnrollReduceMinAllAxesPass(log));
    pm.addPass(IE::createDecomposeSTFTPass(log));
    pm.addPass(IE::createConvertReduceToPoolingPass(log));
    pm.addPass(IE::createConvertPowerToMultPass(log));
    pm.addPass(IE::createConvertGatherToSlicePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// ShaveCodeGen specific passes included in DefaultHW and ReferenceSW
//

void vpux::IE::buildShaveCodeGenPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(ShaveCodeGen::createEncapsulateCodeGenOpsPass());
    pm.addPass(ShaveCodeGen::createEarlyCodeGenCapsuleFusionPass());

    ShaveCodeGen::buildLowerSwLayers2LinalgPipeline(pm, log);
    pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
    pm.addPass(mlir::createCanonicalizerPass());

    pm.addPass(ShaveCodeGen::createOutlineCodeGenCapsulesPass());
    pm.addPass(ShaveCodeGen::createFlattenEltwiseKernelPass());
    pm.addPass(ShaveCodeGen::createLinalgTileAndFuseSwLayersPass());
    pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
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

    mlir::PassPipelineRegistration<AdjustForVPUOptions>(
            "adjust-for-vpu", "[LEGALIZATION] Adjust IE Dialect IR for VPU target",
            [](mlir::OpPassManager& pm, const AdjustForVPUOptions& options) {
                IE::buildAdjustForVPUPipeline(pm, options);
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

    mlir::PassPipelineRegistration<>("shavecodegen-ie", "Shavecodegen specific passes", [](mlir::OpPassManager& pm) {
        IE::buildShaveCodeGenPipeline(pm);
    });
}
