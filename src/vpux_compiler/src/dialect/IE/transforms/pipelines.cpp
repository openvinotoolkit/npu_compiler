//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/public_options.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// PostImport
//

void vpux::IE::buildPostImportPipeline(mlir::OpPassManager& pm, const PostImportOptions& options, Logger log) {
    // moved HandleU16FakeQuantize in the beginning of the pipeline, it needed for u16->u8 lowering + the other passes
    // before it are needed for fixing the pipeline passes are under the flag enableQDQOptimizationAggressive not to
    // affect other models; issues are tracked here #E-181373
    if (options.enableQDQOptimizationAggressive) {
        const auto grc = getDefaultGreedyRewriteConfig();
        pm.addPass(mlir::createCanonicalizerPass(grc));
        pm.addPass(IE::createReshapeMatMulInputsPass(false, log));
        pm.addPass(IE::createConvertScalarToTensorPass(log));
        pm.addPass(IE::createAdjustFakeQuantizeParamsPass(log));
        pm.addPass(IE::createAdjustFakeQdqParamsPass(log));
        pm.addPass(IE::createFuseFQAndMulPass(options.fuseFQAndMulWithNonConstInput, log));
        pm.addPass(IE::createHandleU16FakeQuantizePass(log));
    }
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

void vpux::IE::buildAdjustForVPUPipeline(mlir::OpPassManager& pm, const AdjustForVPUOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(
            IE::createLegalizeDilatedConvolutionPass(isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
    pm.addPass(IE::createPerAxisFQConcatPass(log));
    pm.addPass(IE::createConvertPaddingsToFloorModePass(log));
    pm.addPass(IE::createConvertShuffleChannelsPass(log));
    pm.addPass(IE::createConvertNearestToBroadCastOrStridedConcatPass(
            /*interpolateAsSEOp=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertBilinearToStridedConcatAndConvPass(
            /*interpolateAsSEOp=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertBroadcastToTilePass(log));
    pm.addPass(IE::createMergeTileWithSlicePass(log));
    pm.addPass(IE::createConvertScatterNDUpdateToStridedConcatPass(log));
    pm.addPass(IE::createConvertTransposedConv2DToConv2DPass(
            /*enableSEPTransposedConv=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertGroupTransposedConvToGroupConvPass(
            /*enableSEPTransposedConv=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertGroupTransposedConvToTransposedConvPass(
            /*enableSEPTransposedConv=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(IE::createConvertLargeConvToMultiConvWithAddPass(log));
    pm.addPass(IE::createConvertUpsamplingToStridedConcatPass(log));
    pm.addPass(IE::createMergeWeightsSharedConvPass(log));
    if (options.enableConvertNonConstantPadToSliceAndConcat) {
        pm.addPass(IE::createConvertNonConstantPadToSliceAndConcatPass(
                /*enableSEPPad=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    }
    pm.addPass(IE::createFusePadOpsPass(log));
    pm.addPass(IE::createConvertPadToConcatPass(log));
    pm.addPass(IE::createConvertDepth2SpaceLayerPass(log));
    pm.addPass(IE::createConvertSpace2DepthLayerPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createFuseActivationOpsPass(options.enableFuseClampOperations, log));
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

    pm.addPass(IE::createConvertExtractImagePatchesPass(log));
    pm.addPass(IE::createConvertReduceSumToConvPass(log));
    pm.addPass(IE::createUnrollReduceMinAllAxesPass(log));
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
    mlir::PassPipelineRegistration<PostImportOptions>(
            "post-import",
            "[LEGALIZATION] The post import pipeline contains passes that were historically ngraph passes. It's "
            "considered a legalization step because it converts the imported IR into an IR format that is supported by "
            "the other passes. No other passes should be run before it (with very few exceptions).",
            [](mlir::OpPassManager& pm, const PostImportOptions& options) {
                IE::buildPostImportPipeline(pm, options);
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

    mlir::PassPipelineRegistration<>("shavecodegen-ie", "Shavecodegen specific passes", [](mlir::OpPassManager& pm) {
        IE::buildShaveCodeGenPipeline(pm);
    });
}
