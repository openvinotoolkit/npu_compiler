//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

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
    if (options.enableProfiling) {
        // TODO: E#140041 enable profiling with outlining
        log.warning("Outlining was disabled due to profiling being enabled");
        isOutliningEnabled = false;
    }
    if (isOutliningEnabled) {
        if (options.enableLoopOutliner) {
            pm.addPass(IE::createLoopOutlinerPass(log));
        }
        pm.addPass(mlir::createCanonicalizerPass(grc));

        auto debatcherOptionsPtr = DebatcherOptions::create(options, log);
        if (debatcherOptionsPtr) {
            log.info("DebatcherOptions: {0}", debatcherOptionsPtr->to_string());
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

    // Level 3 : Topology
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }

    IE::arch37xx::buildDynamicShapeTransformationsPipeline(pm, log);
    pm.addPass(IE::createReshapeMatMulInputsPass(options.enableGroupedMatMul, log));
    IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::arch37xx::buildInitialTransformationsPipeline(pm, IE::TransformOptions(options), log);
    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

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

    pm.addPass(IE::createAdaptShapesForScaleShiftPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createSwapTransposeConcatPass(log));
    pm.addPass(IE::createConvertSplitConcatToTransposePass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    //  [Tracking number: E#101595]
    // This temporary check is necessary for m2i interpolate functional tests and it will be removed as part of
    // E#101595
    pm.addPass(IE::createConvertToSpatialOpPass(isOptionEnabled(options.enableM2I),
                                                isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
    pm.addPass(IE::createConvertSubtractToAddPass(log));
    pm.addPass(IE::createConvertBranchesConcatToConvPass(log));
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    pm.addPass(IE::createSwapPadLayerPass(log));
    pm.addPass(IE::createConvertDivideToMultiplyPass(log));
    // Note: apply FuseStaticScale after ConvertDivideToMultiply to increase
    // the applicability
    pm.addPass(IE::arch37xx::createFuseStaticScalePass(log, false));
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
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
        pm.addPass(IE::arch40xx::createMapBilinearInterpolateOnDPUPass(isOptionEnabled(options.enableSEPtrsOperations),
                                                                       log));
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

    IE::arch37xx::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    pm.addPass(IE::createFuseConvWithSlicePass(log));

    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    if (options.enableFusePermuteQuantize) {
        pm.addPass(IE::createFusePermuteQuantizePass(true, log));
        pm.addPass(IE::createConvertReorderToPermuteQuantizePass(log));
    }

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createAdjustGroupConvShapePass(log));
    }

    IE::arch37xx::buildOptimizeMemPermuteAndActivationChannelsExpandPipeline(
            pm, IE::ExpandActivationChannelsOptions(options), log);

    if (auto batchUnrollOptions = BatchUnrollOptions::create(options, log); batchUnrollOptions != nullptr) {
        pm.addPass(IE::createUnrollBatchPass(log, isOptionEnabled(batchUnrollOptions->skipUnrollBatch)));
    }

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
            pm.addPass(IE::arch37xx::createOptimizeSliceExpandPass(log));
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

    mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions>(
            "mempermute-processing",
            "[OPTIMIZATION] MemPermute processing is responsible for handling mempermute op and optimize final "
            "subgraph to avoid unnecessary data "
            "permutations",
            [](mlir::OpPassManager& pm) {
                IE::arch37xx::buildMemPermuteProcessingPipeline(pm);
            });

    mlir::PassPipelineRegistration<ExpandActivationChannelsOptions>(
            "optimize-mempermute-and-activation-channels-expand",
            "[OPTIMIZATION] Optimize MemPermute and activation channel expand",
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
                IE::arch37xx::buildDynamicShapeTransformationsPipeline(pm);
            });
}
