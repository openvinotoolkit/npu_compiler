//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/Pass.h>

namespace vpux {
namespace IE {

//
// AdjustPrecision
//

struct AdjustPrecisionOptions : mlir::PassPipelineOptions<AdjustPrecisionOptions> {
    StrOption computeLayersWithHigherPrecision{*this, "compute-layers-with-higher-precision",
                                               llvm::cl::desc("Enable compute layers with higher precision"),
                                               llvm::cl::init("")};

    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     llvm::cl::init(true)};

    AdjustPrecisionOptions() = default;

    template <class OtherOptions>
    explicit AdjustPrecisionOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

void buildAdjustPrecisionPipeline(mlir::OpPassManager& pm, const AdjustPrecisionOptions& options,
                                  Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createLoggingWeightsQuantFusedIntoTaskPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInsertIdentityPoolBeforeOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateExpandPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseReordersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseScalePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSubGRUSequenceToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseOutstandingDequant(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFusePermuteQuantizeExpandPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertWeightsToI8Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createReduceNumTilesForSmallModelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createReduceNumTilesForSmallModelsPass(bool enableSDPAContributions,
                                                                   Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPrecisionToFPPass(Logger log = Logger::global(),
                                                           StringRef computeLayersWithHigherPrecision = "");
std::unique_ptr<mlir::Pass> createConvertPrecisionToI32Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUseUserPrecisionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustSoftwareOpsPrecisionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustNCEOpsWithI32InputsPass(Logger log = Logger::global(),
                                                                bool enableConvertFCToConv = true);
std::unique_ptr<mlir::Pass> createPropagateReorderToNCEPass(Logger log = Logger::global());

//
// AdjustLayout
//

struct AdjustLayoutOptions : mlir::PassPipelineOptions<AdjustLayoutOptions> {
    BoolOption enableOptimizeReorders{*this, "optimize-reorders", llvm::cl::desc("Enable optimize-reorders pass"),
                                      llvm::cl::init(true)};

    BoolOption enableForceZMajorConcat{*this, "force-z-major-concat",
                                       llvm::cl::desc("Enable transpose-reorder-concat pass"), llvm::cl::init(true)};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    AdjustLayoutOptions() = default;

    template <class OtherOptions>
    explicit AdjustLayoutOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

std::unique_ptr<mlir::Pass> createAdjustLayoutsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseReshapeMvnPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseRMSNormPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseNormalizeL2ToRMSPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseRoPEPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseColorConversionPass(const bool enableYuvToRgbShaveScale = false,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseSDPAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseAttentionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeAttentionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseDynamicQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSelectToEltwisePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseReduceSquarePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeParallelLayersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createEliminateSliceInSoftmaxMatMulPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeReordersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeReordersAcrossFunctionCallsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizePrecisionAcrossFunctionCallsPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createUniquifySimilarOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeIdentityPoolPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertFFTToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeSTFTPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeISTFTPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToMemPermutePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLegalizeNDMemPermutePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createTransposeToPermuteCastPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdaptShapesForScaleShiftPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSplitConcatToAffineReshapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSplitConcatToTransposePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOutlinerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOutlinerPass(const DefaultHWOptionsBase& outlingOptions,
                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDuplicateFQAcrossFunctionCallsPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createDebatcherPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDebatcherPass(const DebatcherOptions& options, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDeDebatcherPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDeDebatcherPass(const DebatcherOptions& options, Logger log = Logger::global());

struct DebatcherOpReorderingOptions : mlir::PassPipelineOptions<DebatcherOpReorderingOptions> {
    StrOption overideToTilesPerBatchMode{*this, "override-to-tiles-per-batch-mode",
                                         llvm::cl::desc("'apply' or 'revert'"), llvm::cl::init("apply")};
    DebatcherOpReorderingOptions() {
    }
    ~DebatcherOpReorderingOptions() {
    }
    DebatcherOpReorderingOptions(const DebatcherOpReorderingOptions& src);
    DebatcherOpReorderingOptions& operator=(const DebatcherOpReorderingOptions& src);

    static std::unique_ptr<DebatcherOpReorderingOptions> create(const DebatcherOptions& options,
                                                                Logger log = Logger::global());
    static std::unique_ptr<DebatcherOpReorderingOptions> create(const BatchCompileOptionsAdapter& options,
                                                                Logger log = Logger::global());

    template <class Options>
    static std::unique_ptr<DebatcherOpReorderingOptions> create(const Options& options, Logger log = Logger::global()) {
        if constexpr (vpux::detail::has_getBatchCompileAdapter<Options>::value) {
            return create(options.getBatchCompileAdapter(), log);
        } else {
            return nullptr;
        }
    }

    static bool isAvailable(const DebatcherOptions& options);

    static std::string getDefaultOptions();
};

std::unique_ptr<mlir::Pass> createOverrideTileExecutorNumPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createOverrideTileExecutorNumPass(const DebatcherOpReorderingOptions& options,
                                                              Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRevertTileExecutorNumPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRevertTileExecutorNumPass(const DebatcherOpReorderingOptions& options,
                                                            Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createUnrollBatchPass(Logger log = Logger::global(), const bool skipUnrollBatch = false);

std::unique_ptr<mlir::Pass> createUnrollSDPAPatternPass(Logger log = Logger::global());

//
// AdjustForVPU
//

void buildAdjustForVPUPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustForVPUPipelineRewriterExecutorPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAssignReadValueToReturnsAndInputs(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertScalarToTensorPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertMinMaxToClampPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertShapeTo4DPass(bool forceConvertGatherTo4D, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertShapeTo4DPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapOperationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapViewOpAndClampPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapTransposeConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapPadLayerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertNceOpsTo4DPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGroupConvToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollConv3dToConv2dPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLoopOutlinerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createChunkGatedDeltaRuleLoopPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollTensorIteratorPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPaddingsToFloorModePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLegalizeDilatedConvolutionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveStridedSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertStridedSlice2ConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRunF16ToF32ConvertOnDPUPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSliceWithStridePass(bool disableMinHWThreshold = false,
                                                              Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeTileOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseActivationOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPadToConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapMaxPoolWithActivation(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapConvertWithSWOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertTransposedConv2DToConv2DPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGroupTransposedConvToGroupConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGroupTransposedConvToTransposedConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertUpsamplingToStridedConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDepth2SpaceToTransposedConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapD2SAndScaleShiftPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseSoftMaxConvertPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseLogSoftmaxVariantsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDepth2SpaceLayerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSpace2DepthLayerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDeformableConvToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInsertReorderBetweenLayerAndConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleEltwiseWithSmallHeightPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateAffineReshapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateShapeCastPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateTransposePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapTransposeWithFQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateDequantThroughConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapConvertWithReshapeKindOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGatherPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToScaleShiftPass(bool enableNCEEltwiseMultiply = false,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createReassociateMultiplyPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeLSTMSequencePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeGRUSequencePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeLSTMCellPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeGRUCellPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeL2OpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeSoftPlusPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDilatedConvConvertPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSubtractToAddPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeOpSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertBroadcastToTilePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGRNToNormalizeL2Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUniquifyBranchesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapMVNWithTransposePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustMemPermuteAroundOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateMemPermuteThroughEltwisePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateMemPermuteBeforeOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateMemPermuteThroughSoftMaxPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeReduceOpsWithMemPermutePass(bool enableFuseReduceMinMaxToDpu = false,
                                                                      Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createReshapeMaxPoolPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertParallelSlicesToGatherPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGatherElementsToGatherPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDumpStatisticsOfIeOpsPass(StringRef title = {}, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSDPAToFlashSDPAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMapBilinearInterpolateOnDPUPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToMixedPrecision(bool enableFloatInQuantWeightsMixedMode = true,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSliceExpandPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseD2SExpandChannelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLegalizeEpsilonUsagePass(Logger log = Logger::global());

struct MemPermutePositioningOptions : mlir::PassPipelineOptions<MemPermutePositioningOptions> {
    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};
    BoolOption enablePropagateMemPermuteThroughEltwise{
            *this, "enable-propagate-mem-permute-through-eltwise",
            llvm::cl::desc("Enable propagation of MemPermute through Eltwise."), llvm::cl::init(true)};
    BoolOption enableAdjustMemPermuteAroundOp{*this, "enable-adjust-mem-permute-around-op",
                                              llvm::cl::desc("Enable adjustment of MemPermute around operations."),
                                              llvm::cl::init(true)};
    BoolOption enableMovePermutePostEltwise{*this, "enable-move-permute-post-eltwise",
                                            llvm::cl::desc("Enable moving MemPermute after Eltwise operations."),
                                            llvm::cl::init(true)};
    BoolOption enableFuseReduceMinMaxToDpu{
            *this, "enable-fuse-reduce-minmax-to-dpu",
            llvm::cl::desc("Enable fusing ReduceMin/ReduceMax operations into the parent DPU op"),
            llvm::cl::init(false)};

    MemPermutePositioningOptions() = default;

    template <class OtherOptions>
    explicit MemPermutePositioningOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

//
// OptimizeActivations
//

struct OptimizeActivationsOptions : mlir::PassPipelineOptions<OptimizeActivationsOptions> {
    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableFuseClampOperations{*this, "enable-fuse-clamp-op", ::llvm::cl::desc("Enable FuseClamp operations"),
                                         ::llvm::cl::init(false)};

    BoolOption enableSwapConvertWithSWOp{*this, "swap-convert-with-sw-op",
                                         llvm::cl::desc("Enable swap-convert-with-sw-op pass"), llvm::cl::init(true)};

    OptimizeActivationsOptions() = default;

    template <class OtherOptions>
    explicit OptimizeActivationsOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

void buildOptimizeActivationsPipeline(mlir::OpPassManager& pm, const OptimizeActivationsOptions& options,
                                      Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeActivationsPipelineRewriterExecutorPass(Logger log = Logger::global());

//
// LowPrecision
//

struct LowPrecisionOptions : mlir::PassPipelineOptions<LowPrecisionOptions> {
    BoolOption enableQuantDequantRemoval{*this, "quant-dequant-removal",
                                         llvm::cl::desc("Enable quantize->dequantize sequence removal"),
                                         llvm::cl::init(true)};

    BoolOption enableFuseOutstandingDequant{*this, "fuse-outstanding-dequant",
                                            llvm::cl::desc("Fuse outstanding dequantize after NCE task"),
                                            llvm::cl::init(true)};

    BoolOption enableFuseOutstandingQuant{*this, "fuse-outstanding-quant",
                                          llvm::cl::desc("Fuse outstanding quantize before two-input Eltwise task"),
                                          llvm::cl::init(true)};

    BoolOption enableSwapTransposeWithFQ{*this, "swap-transpose-with-fq",
                                         ::llvm::cl::desc("Enable SwapTransposeWithFQ pass"), ::llvm::cl::init(true)};

    BoolOption enableConvertToPalletizationLUT{*this, "enable-convert-to-palletization-lut",
                                               llvm::cl::desc("Enable conversion of certain types to palletized LUT"),
                                               llvm::cl::init(false)};

    BoolOption enableFloatInQuantWeightsMixedMode{
            *this, "enable-float-in-quant-weights-mixed-mode",
            llvm::cl::desc("Enable mixed mode for NCE tasks with float input and quantized weights"),
            llvm::cl::init(true)};

    BoolOption enableAlignScales{*this, "enable-align-scales", llvm::cl::desc("Enable align scales"),
                                 llvm::cl::init(true)};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableAdjustNonZeroFakeQuant{*this, "adjust-non-zero-fake-quant",
                                            llvm::cl::desc("Enable adjust non zero fake quant"), llvm::cl::init(true)};

    BoolOption enableLogDynamicQuant{*this, "enable-log-dynamic-quant",
                                     llvm::cl::desc("Enable logging dynamic quant weights signal pass."),
                                     llvm::cl::init(false)};

    BoolOption enableRuntimeDequant{*this, "enable-runtime-dequant",
                                    llvm::cl::desc("Enable runtime dequantization of asymmetrically quantized weight"),
                                    llvm::cl::init(false)};
    Int64Option runtimeDequantizationLimit{
            *this, "runtime-dequantization-limit",
            llvm::cl::desc("Lower limit on weight size for runtime dequantization"
                           "Weights smaller than the limit will be statically dequantized"),
            llvm::cl::init(524'288)};  // 512kb
    BoolOption enableMatmulMixedPrecisionDecomposition{
            *this, "enable-matmul-mixed-precision-decomposition",
            llvm::cl::desc("Enable mixed precision decomposition for matmul"), llvm::cl::init(false)};
    DoubleOption matmulMixedPrecisionDecompositionRatio{
            *this, "matmul-mixed-precision-decomposition-ratio",
            llvm::cl::desc("Determines when to enable Matmul Mixed Precision Decomposition"
                           "Ratio = (MatMul input size)/(Sum of Inputs of newly added ops by decomposition)"),
            llvm::cl::init(250.0)};

    BoolOption enableConvertWeightsToU8I4{*this, "enable-convert-weights-to-u8-i4",
                                          llvm::cl::desc("Enable ConvertWeightsToU8I4 pass in pipeline. This pass is "
                                                         "disabled for WS init function compilation."),
                                          llvm::cl::init(true)};

    BoolOption enableConvertQuantizeOpsToNceOps{
            *this, "enable-convert-quantize-ops-to-nce",
            llvm::cl::desc("Enable Quantize/Dequantize ops conversion to DPU operation"), llvm::cl::init(true)};

    BoolOption enableFuseClampOperations{*this, "enable-fuse-clamp-op", llvm::cl::desc("Enable fuse clamp operations"),
                                         llvm::cl::init(false)};

    StrOption functionOutlining{*this, "function-outlining",
                                llvm::cl::desc("Define a list of outlining modes and their parameters where the next "
                                               "outlining mode is the fallback mode of the previous one."
                                               "Example: function-outlining=' repeating-blocks=max-num-iterations=30 "
                                               "min-ops-in-block=16, naive=num-parts=2'")};
    LowPrecisionOptions() = default;

    template <class OtherOptions>
    explicit LowPrecisionOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

struct TransformOptions : mlir::PassPipelineOptions<TransformOptions> {
    TransformOptions() = default;

    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     llvm::cl::init(true)};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};

    BoolOption enableConvertFFTToConv{*this, "convert-fft-to-conv", llvm::cl::desc("Enable convert-fft-to-conv pass"),
                                      llvm::cl::init(true)};
    BoolOption enableDecomposeGRUSequence{*this, "decompose-gru-sequence",
                                          llvm::cl::desc("Enable decompose-gru-sequence pass"), llvm::cl::init(true)};

    BoolOption enableConvertToReduceSquare{*this, "convert-to-reduce-square",
                                           llvm::cl::desc("Enable fuse-reduce-square pass"), llvm::cl::init(true)};

    BoolOption enableYuvToRgbShaveScale{*this, "yuv-to-rgb-shave-scale",
                                        llvm::cl::desc("Enable YUV to RGB SHAVE scale conversion"),
                                        llvm::cl::init(false)};

    BoolOption enableConvertCumSumToMatMul{*this, "convert-cumsum-to-matmul",
                                           llvm::cl::desc("Enable convert-cumsum-to-matmul rewriter"),
                                           llvm::cl::init(false)};

    template <class OtherOptions>
    explicit TransformOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

struct AttentionProcessingOptions : mlir::PassPipelineOptions<AttentionProcessingOptions> {
    AttentionProcessingOptions() = default;

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};

    BoolOption enableFlashSDPAConversion{*this, "enable-flash-sdpa-conversion",
                                         llvm::cl::desc("Enable convert-sdpa-to-flash-sdpa pass"),
                                         llvm::cl::init(false)};

    BoolOption enableConvertToAttention{*this, "convert-to-attention",
                                        llvm::cl::desc("Enable conversion to Attention operation"),
                                        llvm::cl::init(false)};

    BoolOption enableFuseSoftwareSDPA{*this, "fuse-software-sdpa", llvm::cl::desc("Enable fuse-sdpa pass"),
                                      llvm::cl::init(false)};

    BoolOption enableDecomposeAttention{*this, "decompose-attention", llvm::cl::desc("Enable decompose-attention pass"),
                                        llvm::cl::init(true)};

    template <class OtherOptions>
    explicit AttentionProcessingOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

std::unique_ptr<mlir::Pass> createBatchOpProcessingPipelineRewriterExecutorPass(const TransformOptions& options,
                                                                                Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBatchOpProcessingPipelineRewriterExecutorPass(Logger log = Logger::global());

struct LowPrecisionTransformOptions : mlir::PassPipelineOptions<LowPrecisionTransformOptions> {
    LowPrecisionTransformOptions() = default;

    BoolOption fuseFQAndMulWithNonConstInput{*this, "fuse-fq-and-mul-with-non-const-input",
                                             llvm::cl::desc("Enable fuse-fq-and-mul pass with non const input"),
                                             llvm::cl::init(false)};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};

    BoolOption enableDynamicQuantizationForStaticCase{*this, "enable-dynamic-quantization-for-static-case",
                                                      llvm::cl::desc("Enable dynamic quantization for static case."),
                                                      llvm::cl::init(false)};

    template <class OtherOptions>
    explicit LowPrecisionTransformOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
        // TODO: E#179583 The structure will be reused or removed in ticket if determined to not be necessary.
    }
};

struct ExpandActivationChannelsOptions : mlir::PassPipelineOptions<ExpandActivationChannelsOptions> {
    ExpandActivationChannelsOptions() = default;

    BoolOption enableAdjustConvShapePass{*this, "adjust-convolution-shape",
                                         llvm::cl::desc("Enable adjust-convolution-shape pass"), llvm::cl::init(true)};

    Int64Option preferredSpatialAlignment{
            *this, "preferred-spatial-alignment",
            llvm::cl::desc("Preferred spatial (W) alignment for reshape in adjust-convolution-input-shape pass"),
            llvm::cl::init(VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT)};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};

    BoolOption enablePropagateMemPermuteThroughEltwise{
            *this, "enable-propagate-mem-permute-through-eltwise",
            llvm::cl::desc("Enable propagation of MemPermute through Eltwise."), llvm::cl::init(true)};
    BoolOption enableAdjustMemPermuteAroundOp{*this, "enable-adjust-mem-permute-around-op",
                                              llvm::cl::desc("Enable adjustment of MemPermute around operations."),
                                              llvm::cl::init(true)};
    BoolOption enableMovePermutePostEltwise{*this, "enable-move-permute-post-eltwise",
                                            llvm::cl::desc("Enable moving MemPermute after Eltwise operations."),
                                            llvm::cl::init(true)};

    template <class OtherOptions>
    explicit ExpandActivationChannelsOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

std::unique_ptr<mlir::Pass> createMemPermuteProcessingPipelineRewriterExecutorPass(
        const ExpandActivationChannelsOptions& options, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMemPermuteProcessingPipelineRewriterExecutorPass(Logger log = Logger::global());

struct DynamicShapeTransformOptions : mlir::PassPipelineOptions<DynamicShapeTransformOptions> {
    DynamicShapeTransformOptions() = default;

    BoolOption enableApplyDynamicBoundaryCorrection{*this, "enable-apply-dynamic-boundary-correction",
                                                    llvm::cl::desc("Enable apply-dynamic-boundary-correction pass"),
                                                    llvm::cl::init(false)};

    StrOption locationsVerificationMode{*this, "verify-locations",
                                        llvm::cl::desc("Selects location verification mode. Possible "
                                                       "options are off/fast/full/thorough"),
                                        llvm::cl::init(vpux::isDeveloperBuild() ? "fast" : "off")};

    template <class OtherOptions>
    explicit DynamicShapeTransformOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

//
// OperationConversionOptions
//

struct OperationConversionOptions : mlir::PassPipelineOptions<OperationConversionOptions> {
    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     llvm::cl::init(true)};

    BoolOption mergeUnrolledMatmulForLargeOC{*this, "merge-unrolled-matmul-for-large-oc",
                                             llvm::cl::desc("Enable merging unrolled Matmul ops for large OC"),
                                             llvm::cl::init(false)};

    BoolOption convertDynamicDequantize{*this, "convert-dynamic-dequantize",
                                        llvm::cl::desc("Enable convert dynamic dequantize to dequantize ops"),
                                        llvm::cl::init(true)};
    BoolOption enableFuseReduceMinMaxToDpu{
            *this, "enable-fuse-reduce-minmax-to-dpu",
            llvm::cl::desc("Enable fusing ReduceMin/ReduceMax operations into the parent DPU op"),
            llvm::cl::init(false)};

    OperationConversionOptions() = default;

    template <class OtherOptions>
    explicit OperationConversionOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

void buildScaleShiftProcessingPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildOperationConversionPipeline(mlir::OpPassManager& pm, const IE::OperationConversionOptions& options,
                                      Logger log = Logger::global());

//
// SplitAndMapBilinearInterpolateOnDPUOptions
//

struct SplitAndMapBilinearInterpolateOnDPUOptions :
        mlir::PassPipelineOptions<SplitAndMapBilinearInterpolateOnDPUOptions> {
    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableSplitBilinearIntoHAndW{*this, "split-bilinear-into-H-and-W",
                                            llvm::cl::desc("Enable split-bilinear-into-H-and-W pass"),
                                            llvm::cl::init(false)};
    SplitAndMapBilinearInterpolateOnDPUOptions() = default;

    template <class OtherOptions>
    explicit SplitAndMapBilinearInterpolateOnDPUOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

void buildSplitAndMapBilinearInterpolateOnDPUPipeline(mlir::OpPassManager& pm,
                                                      const SplitAndMapBilinearInterpolateOnDPUOptions& options,
                                                      Logger log);

//
// ConvertToEfficientOpsOptions
//

struct ConvertToEfficientOpsOptions : mlir::PassPipelineOptions<ConvertToEfficientOpsOptions> {
    // E#180631: This is a workaround flag until proper 4D gather is supported
    BoolOption forceConvertGatherTo4D{*this, "force-convert-gather-to-4d",
                                      llvm::cl::desc("For WS Init schedule, GatherOp is forced to 4D in "
                                                     "ConvertShapeTo4D pass"),
                                      llvm::cl::init(false)};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations")};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableD2SToTransposedConvConversion{*this, "enable-d2s-to-transposed-conv-conversion",
                                                   llvm::cl::desc("Enable conversion from D2S to TransposedConv pass"),
                                                   llvm::cl::init(true)};

    BoolOption enableNCEEltwiseMultiply{*this, "enable-nce-eltwise-multiply",
                                        llvm::cl::desc("Enable NCEEltwise for Multiply ops with [1,C,1,1] shape"),
                                        llvm::cl::init(false)};

    ConvertToEfficientOpsOptions() = default;

    template <class OtherOptions>
    explicit ConvertToEfficientOpsOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

struct ReorderFakeQuantizeOptions : mlir::PassPipelineOptions<ReorderFakeQuantizeOptions> {
    BoolOption enableSwapTransposeWithFQ{*this, "swap-transpose-with-fq",
                                         ::llvm::cl::desc("Enable SwapTransposeWithFQ pass"), ::llvm::cl::init(true)};

    ReorderFakeQuantizeOptions() = default;

    template <class OtherOptions>
    explicit ReorderFakeQuantizeOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

std::unique_ptr<mlir::Pass> createConvertMVN6ToMVN1Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleU16FakeQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createQDQOptimizationAggressivePass(const bool fuseFQAndMulWithNonConstInput = false,
                                                                Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseQuantizationMultiplyPass(const bool fuseFQAndMulWithNonConstInput = false,
                                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createEltwiseFakeQuantizeFusionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConsolidateNF4WeightsPatternPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFoldActivationBeforeFQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptDynamicEltwiseWithShapeOfPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapFakeQuantWithReshapeAndStridedSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveScatterUpdateByTransposePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAlignScalesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitFakeQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateAndFuseQuantizeDequantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDequantizeConstPass(const int64_t runtimeDequantizationLimit = 0,
                                                      const bool enableRuntimeDequantization = false,
                                                      Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMergeFakeQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseQuantizedOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRemoveQuantDequantSeqPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeUnalignedQDQSeqPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToPalletizationLUT(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertWeightsToU8Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertWeightsToI4Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseConvertWithQDQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToQuantizedOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertQuantizeOpsToNceOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollGroupQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMergeFullyConnectedPass(const bool mergeUnrolledMatmulForLargeOC = false,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollFullyConnectedPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveMultiplyDividePostOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createReshapeMatMulInputsPass(const bool enableGroupedMatMul = false,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMergeParallelFullyConnectedPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseOutstandingQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createShrinkMatmulGroupsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwishFusionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertConstantDynamicDequantizeToDequantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDynamicDequantizeToDequantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDequantizeToDynamicDequantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapOperationWithGatherPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertVariadicSplitToStridedSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustFakeQdqParamsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInitialLowPrecisionTransformationsPipelineRewriterExecutorPass(
        const LowPrecisionTransformOptions& options, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInitialLowPrecisionTransformationsPipelineRewriterExecutorPass(
        const bool enableDynamicQuantizationForStaticCase = false, Logger log = Logger::global());

//
// Legalization for NCE
//

std::unique_ptr<mlir::Pass> createAdjustGroupConvShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustConvolutionShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustConvolutionWeightsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertBatchedLayerTo1NPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustConvolutionInputShapePass(
        int64_t preferredSpatialAlignment = VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustMaxPoolInputShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeAvgPoolWithUnalignedChannelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeConcatMatMulPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDivideToMultiplyPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertMatMulToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLegalizeConvBackpropDataPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertFCToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAvgPoolToDWConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustScaleShiftForDWConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertScaleShiftToDWPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToSpatialOpPass(const bool m2iEnabled = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertNearestToBroadCastOrStridedConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitBilinearIntoHAndWPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitInterpolateAxesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertBilinearToStridedConcatAndConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertScatterPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitConvWithMultipleFQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleLargeStridesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createProcessAsymmetricZeroPointsForMatmulPass(double decompositionEnablementRatio = 0.0,
                                                                           Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createExpandActivationChannelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleLargeKernelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseOpsToMatMulPass(const bool enableGroupedMatMul = false,
                                                      const bool enableConvertCumSumToMatMul = false,
                                                      Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapEltwiseAndReducePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertReduceSumToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertReduceToPoolingPass(bool enableFuseReduceMinMaxToDpu = false,
                                                             Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollReduceMinAllAxesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleExcludePadForAvgPoolPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPowerToMultPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAlignDimensionsForDPUPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFusePermuteQuantizePass(const bool dpuOnly = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustInputShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMovePermutePostEltwisePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertExtractImagePatchesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBroadcastInputForAddPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseInefficientTileForAddPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBroadcastInputForMultiplyPass(const bool broadcastInputForMultiply = false,
                                                                Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertReorderToPermuteQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdaptODUPermutePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRemoveViewLikeOpsChainPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleLargePadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createNormalizeL2FusionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertMemPermuteToOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLogOpOptimizationsPass();
std::unique_ptr<mlir::Pass> createAdjustNonZeroFakeQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseConvWithSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMVNFusionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertBranchesConcatToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagatePermuteCastPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveDynamicDequantizeToUserPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPopulateDynamicDimensionsHWPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPopulateDynamicDimensionsGenericPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPadDynamicInputsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertReverseToDWConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createApplyDynamicBoundaryCorrectionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFixDynamicOpsLocationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDynamicConcatToScatterNDUpdatePass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createLegalizeReifyResultShapesResidualsPass(Logger log = Logger::global());

//
// Generic Optimizations
//

std::unique_ptr<mlir::Pass> createFuseInputScaleShiftPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateAndCleanUpFQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUpstreamSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertNegativePadToSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertNonConstantPadToSliceAndConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertExpandToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInputQuantizationRestorationPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createForbidFourBitOutputsPass(Logger log = Logger::global());

//
// DefaultHWOptions(for all devices)
//

struct DefaultHWOptionsDialectBase : public virtual vpux::DefaultHWOptionsBase {
    BoolOption enableSplitInterpolateAxes{*this, "split-interpolate-axes",
                                          llvm::cl::desc("Enable split-interpolate-axes pass"), llvm::cl::init(false)};

    BoolOption enableAdjustConvShapePass{*this, "adjust-convolution-shape",
                                         llvm::cl::desc("Enable adjust-convolution-shape pass"), llvm::cl::init(true)};

    Int64Option preferredSpatialAlignment{
            *this, "preferred-spatial-alignment",
            llvm::cl::desc("Preferred spatial (W) alignment for reshape in adjust-convolution-input-shape pass"),
            llvm::cl::init(VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT)};

    BoolOption enableConvertExpandToConvPass{*this, "convert-expand-to-conv",
                                             llvm::cl::desc("Enable convert-expand-to-conv pass"),
                                             llvm::cl::init(true)};

    BoolOption disableSliceToConvMinHWThreshold{
            *this, "disable-slice-to-conv-min-hw-threshold",
            llvm::cl::desc("Disable H*W threshold check in optimize-slice-with-stride. "
                           "When true, Slice is converted to Convolution regardless of spatial size."),
            llvm::cl::init(false)};

    BoolOption logOpOptimizations{*this, "log-op-optimizations",
                                  llvm::cl::desc("Log potential operation optimizations that can be done"),
                                  llvm::cl::init(false)};

    // TransformOptions
    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     llvm::cl::init(true)};

    BoolOption enableDecomposeAttention{*this, "decompose-attention", llvm::cl::desc("Enable decompose-attention pass"),
                                        llvm::cl::init(true)};

    BoolOption enableConvertCumSumToMatMul{*this, "convert-cumsum-to-matmul",
                                           llvm::cl::desc("Enable convert-cumsum-to-matmul rewriter"),
                                           llvm::cl::init(false)};

    // AdjustLayoutOptions

    BoolOption enableOptimizeReorders{*this, "optimize-reorders", llvm::cl::desc("Enable optimize-reorders pass"),
                                      llvm::cl::init(true)};

    BoolOption enableForceZMajorConcat{*this, "force-z-major-concat",
                                       llvm::cl::desc("Enable transpose-reorder-concat pass"), llvm::cl::init(true)};

    BoolOption enableSwapTransposeWithFQ{*this, "swap-transpose-with-fq",
                                         ::llvm::cl::desc("Enable SwapTransposeWithFQ pass"), ::llvm::cl::init(true)};

    BoolOption enableAlignScales{*this, "enable-align-scales", llvm::cl::desc("Enable align scales"),
                                 llvm::cl::init(true)};

    BoolOption enableAdjustNonZeroFakeQuant{*this, "adjust-non-zero-fake-quant",
                                            llvm::cl::desc("Enable adjust non zero fake quant"), llvm::cl::init(true)};

    BoolOption enableFloatInQuantWeightsMixedMode{*this, "enable-float-in-quant-weights-mixed-mode",
                                                  llvm::cl::desc("Enable mixed mode for NCE tasks with float input "
                                                                 "and quantized weights"),
                                                  llvm::cl::init(true)};

    // Common
    BoolOption enableFuseClampOperations{*this, "enable-fuse-clamp-op", llvm::cl::desc("Enable fuse clamp operations"),
                                         llvm::cl::init(false)};

    BoolOption enableLogDynamicQuant{*this, "enable-log-dynamic-quant",
                                     llvm::cl::desc("Enable logging dynamic quant weights signal pass."),
                                     llvm::cl::init(false)};

    BoolOption enableQuantDequantRemoval{*this, "quant-dequant-removal",
                                         llvm::cl::desc("Enable quantize->dequantize sequence removal"),
                                         llvm::cl::init(true)};

    BoolOption enableFuseOutstandingDequant{*this, "fuse-outstanding-dequant",
                                            llvm::cl::desc("Fuse outstanding dequantize after NCE task"),
                                            llvm::cl::init(true)};

    BoolOption enableFuseOutstandingQuant{*this, "fuse-outstanding-quant",
                                          llvm::cl::desc("Fuse outstanding quantize before two-input Eltwise task"),
                                          llvm::cl::init(true)};

    BoolOption enableAdjustPrecisionPipeline{*this, "enable-adjust-precision-pipeline",
                                             llvm::cl::desc("Enable the AdjustPrecision pipeline. This pipeline "
                                                            "is disabled for WS init function compilation."),
                                             llvm::cl::init(true)};

    BoolOption enableConvertWeightsToU8I4{*this, "enable-convert-weights-to-u8-i4",
                                          llvm::cl::desc("Enable ConvertWeightsToU8I4 pass in pipeline. This pass is "
                                                         "disabled for WS init function compilation."),
                                          llvm::cl::init(true)};
    BoolOption enableFlashSDPAConversion{*this, "enable-flash-sdpa-conversion",
                                         llvm::cl::desc("Convert SDPA layer into FlashSDPA that implements "
                                                        "FlashAttention-2 approach"),
                                         llvm::cl::init(false)};

    BoolOption enableApplyDynamicBoundaryCorrection{*this, "enable-apply-dynamic-boundary-correction",
                                                    llvm::cl::desc("Enable apply-dynamic-boundary-correction pass"),
                                                    llvm::cl::init(false)};

    BoolOption enableConvertQuantizeOpsToNceOps{*this, "enable-convert-quantize-ops-to-nce",
                                                llvm::cl::desc("Enable Quantize/Dequantize ops conversion to DPU "
                                                               "operation"),
                                                llvm::cl::init(true)};

    BoolOption enableDynamicShapeTransformationsPipeline{*this, "enable-dynamic-shape-transformations",
                                                         llvm::cl::desc("Enable DynamicShapeTransformations Pipeline"),
                                                         llvm::cl::init(true)};
    BoolOption enableD2SToTransposedConvConversion{*this, "enable-d2s-to-transposed-conv-conversion",
                                                   llvm::cl::desc("Enable conversion from D2S to TransposedConv pass"),
                                                   llvm::cl::init(true)};

    // E#180631: This is a workaround flag until proper 4D gather is supported
    BoolOption forceConvertGatherTo4D{*this, "force-convert-gather-to-4d",
                                      llvm::cl::desc("For WS Init schedule, GatherOp is forced to 4D in "
                                                     "ConvertShapeTo4D pass"),
                                      llvm::cl::init(false)};

    Int64Option runtimeDequantizationLimit{*this, "runtime-dequantization-limit",
                                           llvm::cl::desc("Lower limit on weight size for runtime dequantization"
                                                          "Weights smaller than the limit will be statically "
                                                          "dequantized"),
                                           llvm::cl::init(524'288)};  // 512kb

    BoolOption enableDecomposeSDPA{*this, "enable-decompose-sdpa",
                                   llvm::cl::desc("Enable ngraph passes decomposing SDPA like ops"),
                                   llvm::cl::init(true)};
};

//
// Pipelines
//

void buildDebatcherPipeline(mlir::OpPassManager& pm, const BatchCompileOptionsAdapter& options,
                            Logger log = Logger::global());

void buildOptimizeMemPermuteAndActivationChannelsExpandPipeline(mlir::OpPassManager& pm,
                                                                const ExpandActivationChannelsOptions& options,
                                                                Logger log = Logger::global());
void buildOutliningPipeline(mlir::OpPassManager& pm, const DefaultHWOptionsBase& options,
                            Logger log = Logger::global());

void buildMemPermuteProcessingPipeline(mlir::OpPassManager& pm, const ExpandActivationChannelsOptions& options,
                                       Logger log = Logger::global());
void buildMemPermutePositioningPipeline(mlir::OpPassManager& pm, const MemPermutePositioningOptions& options,
                                        Logger log = Logger::global());
void buildDimensionAlignmentPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildOptimizeSliceOpPipeline(mlir::OpPassManager& pm, bool disableSliceToConvMinHWThreshold = false,
                                  Logger log = Logger::global());
void buildExpandAndOptimizeActivationChannelsPipeline(mlir::OpPassManager& pm,
                                                      const ExpandActivationChannelsOptions& options,
                                                      Logger log = Logger::global());
void buildInitialTransformationsPipeline(mlir::OpPassManager& pm, const TransformOptions& options,
                                         Logger log = Logger::global());
void buildDynamicShapeTransformationsPipeline(mlir::OpPassManager& pm, const IE::DynamicShapeTransformOptions& options,
                                              Logger log = Logger::global());
void buildPostImportPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildAdjustLayoutPipeline(mlir::OpPassManager& pm, const AdjustLayoutOptions& options,
                               Logger log = Logger::global());
void buildAdjustShapePipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildSplitLargeOpsPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildConvertToEfficientOpsPipeline(mlir::OpPassManager& pm, const ConvertToEfficientOpsOptions& options,
                                        Logger log = Logger::global());
void buildOptimizeViewLikeOpsPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildHandleHyperParametersPipeline(mlir::OpPassManager& pm, Logger log);
void buildReorderFakeQuantizePipeline(mlir::OpPassManager& pm, const ReorderFakeQuantizeOptions& options, Logger log);
void buildConvertToConvolutionPipeline(mlir::OpPassManager& pm, Logger log);
void buildBatchTransformationPipeline(mlir::OpPassManager& pm, const std::unique_ptr<BatchUnrollOptions>& options,
                                      Logger log = Logger::global());

//
// Registration
//

void registerIEPipelines();
void registerPasses();

}  // namespace IE
}  // namespace vpux
