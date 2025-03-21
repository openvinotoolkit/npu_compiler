//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"

#include "vpux/compiler/utils/options.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassOptions.h>

#include <memory>

namespace vpux {
namespace IE {

//
// PostImport
//

void buildPostImportPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

//
// AdjustPrecision
//

struct AdjustPrecisionOptions : mlir::PassPipelineOptions<AdjustPrecisionOptions> {
    BoolOption enableConvertPrecisionToFP16{*this, "convert-precision-to-fp16",
                                            llvm::cl::desc("Enable convert-precision-to-fp16 pass"),
                                            llvm::cl::init(true)};

    StrOption computeLayersWithHigherPrecision{*this, "compute-layers-with-higher-precision",
                                               llvm::cl::desc("Enable compute layers with higher precision"),
                                               llvm::cl::init("")};

    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     llvm::cl::init(true)};

    AdjustPrecisionOptions() = default;

    template <class OtherOptions>
    explicit AdjustPrecisionOptions(const OtherOptions& options) {
        enableConvertFCToConv = options.enableConvertFCToConv;
        enableConvertPrecisionToFP16 = options.enableConvertPrecisionToFP16;
        computeLayersWithHigherPrecision = options.computeLayersWithHigherPrecision;
    }
};

void buildAdjustPrecisionPipeline(mlir::OpPassManager& pm, const AdjustPrecisionOptions& options,
                                  Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertPrecisionToFP16Pass(Logger log = Logger::global(),
                                                             StringRef computeLayersWithHigherPrecision = "");
std::unique_ptr<mlir::Pass> createConvertPrecisionToI32Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUseUserPrecisionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustSoftwareOpsPrecisionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustNCEOpsWithI32InputsPass(Logger log = Logger::global(),
                                                                bool enableConvertFCToConv = true);

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
        enableOptimizeReorders = options.enableOptimizeReorders;
        enableForceZMajorConcat = options.enableForceZMajorConcat;
        enableSEPtrsOperations = options.enableSEPtrsOperations;
        enableExperimentalSEPtrsOperations = options.enableExperimentalSEPtrsOperations;
    }
};

std::unique_ptr<mlir::Pass> createAdjustLayoutsPass(const bool seOpsEnabled = false,
                                                    const bool seExperimentalOpsEnabled = false,
                                                    Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseReshapeMvnPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseRMSNormPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseRoPEPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeReordersPass(const bool seOpsEnabled = false,
                                                       const bool seExperimentalOpsEnabled = false,
                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeReordersAcrossFunctionCallsPass(const bool seOpsEnabled = false,
                                                                          const bool seExperimentalOpsEnabled = false,
                                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizePrecisionAcrossFunctionCallsPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createUniquifyOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeIdentityPoolPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToMemPermutePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapMemPermuteAndExpandPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLegalizeNDMemPermutePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createTransposeToPermuteCastPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdaptShapesForScaleShiftPass(Logger log = Logger::global());
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
    static std::unique_ptr<DebatcherOpReorderingOptions> create(const DefaultHWOptionsBase& options,
                                                                Logger log = Logger::global());
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

//
// AdjustForVPU
//

struct AdjustForVPUOptions : mlir::PassPipelineOptions<AdjustForVPUOptions> {
    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableFuseClampOperations{*this, "enable-fuse-clamp-op", llvm::cl::desc("Enable fuse clamp operations"),
                                         llvm::cl::init(false)};

    AdjustForVPUOptions() = default;

    template <class OtherOptions>
    explicit AdjustForVPUOptions(const OtherOptions& options) {
        enableSEPtrsOperations = options.enableSEPtrsOperations;
        enableExperimentalSEPtrsOperations = options.enableExperimentalSEPtrsOperations;
        enableFuseClampOperations = options.enableFuseClampOperations;
    }
};

void buildAdjustForVPUPipeline(mlir::OpPassManager& pm, const AdjustForVPUOptions& options,
                               Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertAssignReadValueToReturnsAndInputs(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertScalarToTensorPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertMinMaxToClampPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertShapeTo4DPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapOperationsPass(const bool seOpsEnabled = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapViewOpAndClampPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapTransposeConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapPadLayerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertNceOpsTo4DPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGroupConvToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertLargeConvToMultiConvWithAddPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollConv3dToConv2dPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLoopOutlinerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollTensorIteratorPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPaddingsToFloorModePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertShuffleChannelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLegalizeDilatedConvolutionPass(const bool enableDilatedGroupConv = false,
                                                                 Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveStridedSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertStridedSlice2ConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRunF16ToF32ConvertOnDPUPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSliceWithStridePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeTileOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseActivationOpsPass(const bool enableFuseClamp = false,
                                                        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFusePadOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPadToConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapMaxPoolWithActivation(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertTransposedConv2DToConv2DPass(const bool enableSEPTransposedConv = false,
                                                                      Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGroupTransposedConvToGroupConvPass(const bool enableSEPTransposedConv = false,
                                                                            Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGroupTransposedConvToTransposedConvPass(
        const bool enableSEPTransposedConv = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertUpsamplingToStridedConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDepth2SpaceToTransposedConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapD2SAndScaleShiftPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDepth2SpaceLayerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSpace2DepthLayerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInsertReorderBetweenLayerAndConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateAffineReshapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateShapeCastPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateTransposePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapTransposeWithFQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateFqThroughConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapConvertWithTransposeReshapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPerAxisFQConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGatherToSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToScaleShiftPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeLSTMSequencePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeLSTMCellPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeGRUCellPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeNormalizeL2Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDilatedConvConvertPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSubtractToAddPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeOpSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertBroadcastToTilePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGRNToNormalizeL2Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUniquifyBranchesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapMVNWithTransposePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustMemPermuteAroundOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateMemPermuteThroughAddPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateMemPermuteBeforeOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeConcatWithConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateMemPermuteThroughSoftMaxPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateOpThroughBatchConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createReshapeMaxPoolPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertParallelSlicesToGatherPass(Logger log = Logger::global());

struct MemPermutePositioningOptions : mlir::PassPipelineOptions<MemPermutePositioningOptions> {
    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};

    MemPermutePositioningOptions() = default;

    template <class OtherOptions>
    explicit MemPermutePositioningOptions(const OtherOptions& options) {
        enableGroupedMatMul = options.enableGroupedMatMul;
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

    BoolOption enableDPUF16ToF32Convert{*this, "enable-dpu-f16-to-f32-convert",
                                        llvm::cl::desc("Enable running F16 -> F32 converts on DPU."),
                                        llvm::cl::init(false)};

    OptimizeActivationsOptions() = default;

    template <class OtherOptions>
    explicit OptimizeActivationsOptions(const OtherOptions& options) {
        enableSEPtrsOperations = options.enableSEPtrsOperations;
        enableFuseClampOperations = options.enableFuseClampOperations;
        enableDPUF16ToF32Convert = options.enableDPUF16ToF32Convert;
    }
};

void buildOptimizeActivationsPipeline(mlir::OpPassManager& pm, const OptimizeActivationsOptions& options,
                                      Logger log = Logger::global());

//
// LowPrecision
//

struct LowPrecisionOptions : mlir::PassPipelineOptions<LowPrecisionOptions> {
    BoolOption enableQuantDequantRemoval{*this, "quant-dequant-removal",
                                         llvm::cl::desc("Enable quantize->dequantize sequence removal"),
                                         llvm::cl::init(false)};

    BoolOption enableFuseOutstandingDequant{*this, "fuse-outstanding-dequant",
                                            llvm::cl::desc("Fuse outstanding dequantize after NCE task"),
                                            llvm::cl::init(false)};

    BoolOption enableFuseOutstandingQuant{*this, "fuse-outstanding-quant",
                                          llvm::cl::desc("Fuse outstanding quantize before two-input Eltwise task"),
                                          llvm::cl::init(false)};

    BoolOption enableSwapTransposeWithFQ{*this, "swap-transpose-with-fq",
                                         ::llvm::cl::desc("Enable SwapTransposeWithFQ pass"), ::llvm::cl::init(true)};

    BoolOption enablePropagateQuantDequant{*this, "propagate-quant-dequant",
                                           llvm::cl::desc("Enable Propagate Quantize Dequantize pass"),
                                           llvm::cl::init(true)};

    BoolOption enableFP16ToU8MixedMode{
            *this, "enable-fp16-to-u8-mixed-mode",
            llvm::cl::desc("Enable mixed mode for NCE tasks with FP16 input and quantized output"),
            llvm::cl::init(false)};

    BoolOption enableFloatInQuantWeightsMixedMode{
            *this, "enable-float-in-quant-weights-mixed-mode",
            llvm::cl::desc("Enable mixed mode for NCE tasks with float input and quantized weights"),
            llvm::cl::init(true)};

    BoolOption enableAlignScales{*this, "enable-align-scales", llvm::cl::desc("Enable align scales"),
                                 llvm::cl::init(true)};

    BoolOption enableAdaptiveStripping{*this, "enable-adaptive-stripping", llvm::cl::desc("Enable adaptive stripping"),
                                       llvm::cl::init(false)};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableAdjustNonZeroFakeQuant{*this, "adjust-non-zero-fake-quant",
                                            llvm::cl::desc("Enable adjust non zero fake quant"), llvm::cl::init(true)};

    BoolOption enableU16FQToScaleShiftConversion{*this, "enable-u16-fake-quantize-to-scale-shift-conversion",
                                                 llvm::cl::desc("Enable u16 fake quantize to scale shift conversion"),
                                                 llvm::cl::init(false)};

    BoolOption enableConvolutionMixedPrecisionDecomposition{
            *this, "enable-convolution-mixed-precision-decomposition",
            llvm::cl::desc("Enable mixed precision decomposition for convolution"), llvm::cl::init(false)};

    BoolOption enableWDBlockArgumentInput{
            *this, "enable-wd-blockarg-input",
            llvm::cl::desc("Enable WeightsDequantizeToFakeQuantizePass on structures with BlockArgument input"),
            llvm::cl::init(true)};

    BoolOption enableDynamicQuant{*this, "enable-dynamic-quant",
                                  llvm::cl::desc("Enable dynamic quant weights signal pass."), llvm::cl::init(false)};

    BoolOption enableRuntimeDequant{*this, "enable-runtime-dequant",
                                    llvm::cl::desc("Enable runtime dequantization of asymmetricly quantized weight"),
                                    llvm::cl::init(false)};
    Int64Option runtimeDequantizationLimit{
            *this, "runtime-dequantization-limit",
            llvm::cl::desc("Lower limit on weight size for runtime dequantization"
                           "Weights smaller than the limit will be statically dequantized"),
            llvm::cl::init(524'288)};  // 512kb

    LowPrecisionOptions() = default;

    template <class OtherOptions>
    explicit LowPrecisionOptions(const OtherOptions& options) {
        enableU16FQToScaleShiftConversion = options.enableU16FQToScaleShiftConversion;
        enableQuantDequantRemoval = options.enableQuantDequantRemoval;
        enableFuseOutstandingDequant = options.enableFuseOutstandingDequant;
        enableFuseOutstandingQuant = options.enableFuseOutstandingQuant;
        enableSwapTransposeWithFQ = options.enableSwapTransposeWithFQ;
        enablePropagateQuantDequant = options.enablePropagateQuantDequant;
        enableFP16ToU8MixedMode = options.enableFP16ToU8MixedMode;
        enableFloatInQuantWeightsMixedMode = options.enableFloatInQuantWeightsMixedMode;
        enableAlignScales = options.enableAlignScales;
        enableSEPtrsOperations = options.enableSEPtrsOperations;
        enableExperimentalSEPtrsOperations = options.enableExperimentalSEPtrsOperations;
        enableAdjustNonZeroFakeQuant = options.enableAdjustNonZeroFakeQuant;
        enableConvolutionMixedPrecisionDecomposition = options.enableConvolutionMixedPrecisionDecomposition;
        enableWDBlockArgumentInput = options.enableWDBlockArgumentInput;
        enableDynamicQuant = options.enableDynamicQuant;
        enableRuntimeDequant = options.enableRuntimeDequant;
        runtimeDequantizationLimit = options.runtimeDequantizationLimit;

        if (options.enableAdaptiveStripping) {
            enableU16FQToScaleShiftConversion = true;
            enableQuantDequantRemoval = true;
            enableFuseOutstandingDequant = true;
            enableFuseOutstandingQuant = true;
        }
    }
};

struct TransformOptions : mlir::PassPipelineOptions<TransformOptions> {
    TransformOptions() = default;

    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     llvm::cl::init(true)};

    BoolOption enableWDBlockArgumentInput{
            *this, "enable-wd-blockarg-input",
            llvm::cl::desc("Enable WeightsDequantizeToFakeQuantizePass on structures with BlockArgument input"),
            llvm::cl::init(true)};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};

    BoolOption enableConvertFFTToConv{*this, "convert-fft-to-conv", llvm::cl::desc("Enable convert-fft-to-conv pass"),
                                      llvm::cl::init(true)};

    BoolOption fuseMvn6ScaleBias{*this, "fuse-mvn6-scale-bias", llvm::cl::desc("Enable fuse-mvn6-scale-bias pass"),
                                 llvm::cl::init(false)};

    BoolOption enableAdaptiveStripping{*this, "enable-adaptive-stripping", llvm::cl::desc("Enable adaptive stripping"),
                                       llvm::cl::init(false)};

    BoolOption enableU16FQToScaleShiftConversion{*this, "enable-u16-fake-quantize-to-scale-shift-conversion",
                                                 llvm::cl::desc("Enable u16 fake quantize to scale shift conversion"),
                                                 llvm::cl::init(false)};
    template <class OtherOptions>
    explicit TransformOptions(const OtherOptions& options) {
        enableConvertFCToConv = options.enableConvertFCToConv;
        enableWDBlockArgumentInput = options.enableWDBlockArgumentInput;
        enableGroupedMatMul = options.enableGroupedMatMul;
        fuseMvn6ScaleBias = options.fuseMvn6ScaleBias;
        enableConvertFFTToConv = options.enableConvertFFTToConv;
        enableAdaptiveStripping = options.enableAdaptiveStripping;
        enableU16FQToScaleShiftConversion = options.enableU16FQToScaleShiftConversion;

        if (options.enableAdaptiveStripping) {
            enableU16FQToScaleShiftConversion = true;
        }
    }
};

struct LowPrecisionTransformOptions : mlir::PassPipelineOptions<LowPrecisionTransformOptions> {
    LowPrecisionTransformOptions() = default;

    BoolOption enableWDBlockArgumentInput{
            *this, "enable-wd-blockarg-input",
            llvm::cl::desc("Enable WeightsDequantizeToFakeQuantizePass on structures with BlockArgument input"),
            llvm::cl::init(true)};

    BoolOption enableWeightsDynamicDequantization{*this, "enable-weights-dynamic-dequantization",
                                                  llvm::cl::desc("Enable dynamic dequantization for weights as input"),
                                                  llvm::cl::init(false)};

    template <class OtherOptions>
    explicit LowPrecisionTransformOptions(const OtherOptions& options) {
        enableWDBlockArgumentInput = options.enableWDBlockArgumentInput;
        enableWeightsDynamicDequantization = options.enableWeightsDynamicDequantization;
    }
};

struct ExpandActivationChannelsOptions : mlir::PassPipelineOptions<ExpandActivationChannelsOptions> {
    ExpandActivationChannelsOptions() = default;

    BoolOption enableExpandActivationChannels{*this, "expand-activation-channels",
                                              llvm::cl::desc("Enable expand-activation-channels pass"),
                                              llvm::cl::init(true)};
    BoolOption enableAdjustConvShapePass{*this, "adjust-convolution-shape",
                                         llvm::cl::desc("Enable adjust-convolution-shape pass"), llvm::cl::init(true)};

    BoolOption enableOptimizeSliceExpand{*this, "optimize-slice-expand",
                                         llvm::cl::desc("Enable optimize-slice-expand pass"), llvm::cl::init(true)};

    BoolOption enableFusePermuteQuantizeExpand{*this, "fuse-permute-quantize-expand",
                                               llvm::cl::desc("Enable fuse-permute-quantize-expand pass"),
                                               llvm::cl::init(true)};
    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};

    template <class OtherOptions>
    explicit ExpandActivationChannelsOptions(const OtherOptions& options) {
        enableExpandActivationChannels = options.enableExpandActivationChannels;
        enableAdjustConvShapePass = options.enableAdjustConvShapePass;
        enableOptimizeSliceExpand = options.enableOptimizeSliceExpand;
        enableFusePermuteQuantizeExpand = options.enableFusePermuteQuantizeExpand;
        enableSEPtrsOperations = options.enableSEPtrsOperations;
        enableExperimentalSEPtrsOperations = options.enableExperimentalSEPtrsOperations;
        enableGroupedMatMul = options.enableGroupedMatMul;
    }
};

//
// OperationConversionOptions
//

struct OperationConversionOptions : mlir::PassPipelineOptions<OperationConversionOptions> {
    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     llvm::cl::init(true)};

    BoolOption accumulateMatmulWithDPU{*this, "accumulate-matmul-with-dpu",
                                       llvm::cl::desc("Accumulate unrolled Matmul results with DPU"),
                                       llvm::cl::init(false)};

    BoolOption fuseScalesToAccumulate{
            *this, "fuse-scales-to-accumulate",
            llvm::cl::desc("Enable scales fusing to following Accumulate op from GPTQ Matmul unrolling"),
            llvm::cl::init(false)};

    BoolOption mergeUnrolledMatmul{*this, "merge-unrolled-matmul", llvm::cl::desc("Enable merging urolled Matmul ops"),
                                   llvm::cl::init(true)};

    OperationConversionOptions() = default;

    template <class OtherOptions>
    explicit OperationConversionOptions(const OtherOptions& options) {
        enableConvertFCToConv = options.enableConvertFCToConv;
        accumulateMatmulWithDPU = options.accumulateMatmulWithDPU;
        fuseScalesToAccumulate = options.fuseScalesToAccumulate;
        mergeUnrolledMatmul = options.mergeUnrolledMatmul;
    }
};

void buildScaleShiftProcessingPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildOperationConversionPipeline(mlir::OpPassManager& pm, const IE::OperationConversionOptions& options,
                                      Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createFuseMvn6ScaleBiasPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertMVN6ToMVN1Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleU16FakeQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleU16FakeQuantizePass(const IE::TransformOptions& options,
                                                            Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseFQAndMulPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createEltwiseFakeQuantizeFusionPass(const bool adaptiveStrippingEnabled = false,
                                                                Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWeightsDequantizeToFakeQuantizePass();
std::unique_ptr<mlir::Pass> createWeightsDequantizeToFakeQuantizePass(const IE::LowPrecisionTransformOptions& options,
                                                                      Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConsolidateWeightsDequantizationPass();
std::unique_ptr<mlir::Pass> createConsolidateWeightsDequantizationPass(const IE::LowPrecisionTransformOptions& options,
                                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFoldActivationBeforeFQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptDynamicEltwiseWithShapeOfPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapFakeQuantWithReshapeAndStridedSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveScatterUpdateByTransposePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAlignScalesPass(const bool seOpsEnabled = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitFakeQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateQuantizeDequantizePass(const bool seOpsEnabled = false,
                                                                  Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDequantizeConstPass(const int64_t runtimeDequantizationLimit = 0,
                                                      const bool enableRuntimeDequantization = false,
                                                      Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMergeFakeQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseQuantizedOpsPass(const bool seOpsEnabled = false,
                                                       const bool enableExperimentalSEPtrsOperations = false,
                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRemoveQuantDequantSeqPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeUnalignedQDQSeqPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertWeightsToU8Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertWeightsToI4Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseConvertWithQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToDequantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToDequantizePass(const IE::LowPrecisionOptions& options,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertQuantizeOpsToNceOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollGroupQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMergeFullyConnectedPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollFullyConnectedPass(Logger log = Logger::global(),
                                                           bool accumulateMatmulWithDPU = false);
std::unique_ptr<mlir::Pass> createFuseScalesToAccumulatePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveMultiplyPostOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createReshapeMatMulInputsPass(const bool enableGroupedMatMul = false,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMergeTileWithSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMergeParallelFullyConnectedPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseOutstandingQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMergeWeightsSharedConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createShrinkMatmulGroupsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwishFusionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDynamicDequantizeToDequantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapOperationWithGatherPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertVariadicSplitToStridedSlicePass(Logger log = Logger::global());

//
// Legalization for NCE
//

std::unique_ptr<mlir::Pass> createAdjustGroupConvShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustConvolutionShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustConvolutionWeightsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertBatchedLayerTo1NPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustConvolutionInputShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustMaxPoolInputShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeAvgPoolWithUnalignedChannelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMatMulInputsTo2dPass(const bool enableGroupedMatMul = false,
                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDivideToMultiplyPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertMatMulToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertConvBackpropDataToTransposedConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertFCToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAvgPoolToDWConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustScaleShiftForDWConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertScaleShiftToDWPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToSpatialOpPass(const bool m2iEnabled = false,
                                                         const bool seExperimentalOpsEnabled = false,
                                                         Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertNearestToBroadCastOrStridedConcatPass(const bool interpolateAsSEOp = false,
                                                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitBilinerIntoHAndWPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertBilinearToStridedConcatAndConvPass(const bool interpolateAsSEOp = false,
                                                                            Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertScatterNDUpdateToStridedConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitConvWithMultipleFQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleLargeStridesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleAsymmetricStridesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createExpandActivationChannelsPass(const bool seOpsEnabled = false,
                                                               const bool seExperimentalOpsEnabled = false,
                                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleLargeKernelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertReduceSumToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertReduceToPoolingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollReduceMinAllAxesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleExcludePadForAvgPoolPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPowerToMultPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createExpandActivationWidthPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFusePermuteQuantizePass(const bool dpuOnly = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustInputShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMovePermutePostEltwisePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertExtractImagePatchesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBroadcastInputForAddPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBroadcastInputForMultiplyPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertReorderToPermuteQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseMemPermutePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRemoveViewLikeOpsChainPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleLargePadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createNormalizeL2FusionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createM2IBatchNormFusionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertMemPermuteToPoolPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLogOpOptimizationsPass();
std::unique_ptr<mlir::Pass> createAdjustNonZeroFakeQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseConvWithSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMVNFusionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertBranchesConcatToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagatePermuteCastThroughDequantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveDynamicDequantizeToUserPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPopulateDynamicDimensionsHWPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPopulateDynamicDimensionsGenericPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPadDynamicInputsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertReverseToDWConvPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createLegalizeReifyResultShapesResidualsPass(Logger log = Logger::global());

//
// Generic Optimizations
//

std::unique_ptr<mlir::Pass> createCleanupFQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseInputScaleShiftPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateFQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUpstreamSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertNonConstantPadToSliceAndConcatPass(const bool enableSEPPad = false,
                                                                            Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertExpandToConvPass(Logger log = Logger::global());

//
// DefaultHWOptions(for all devices)
//

struct DefaultHWOptionsDialectBase : public virtual vpux::DefaultHWOptionsBase {
    BoolOption enableConvertAvgPoolToDWConv{*this, "convert-avg-pool-to-dw-conv",
                                            llvm::cl::desc("Enable convert-avg-pool-to-dw-conv pass"),
                                            llvm::cl::init(true)};

    BoolOption enableU16FQToScaleShiftConversion{*this, "enable-u16-fake-quantize-to-scale-shift-conversion",
                                                 llvm::cl::desc("Enable u16 fake quantize to scale shift conversion"),
                                                 llvm::cl::init(false)};

    BoolOption enableOptimizeScaleShiftToDWConv{*this, "optimize-scale-shift-to-depthwise",
                                                llvm::cl::desc("Enable optimize-scale-shift-to-depthwise pass"),
                                                llvm::cl::init(true)};

    BoolOption enableSplitConvWithMultipleFQ{*this, "split-conv-with-multiple-fq",
                                             llvm::cl::desc("Enable split-conv-with-multiple-fq pass"),
                                             llvm::cl::init(true)};

    BoolOption enableHandleLargeKernel{*this, "handle-large-kernel", llvm::cl::desc("Enable handle-large-kernel pass"),
                                       llvm::cl::init(true)};

    BoolOption enableHandleLargeStrides{*this, "handle-large-strides",
                                        llvm::cl::desc("Enable handle-large-strides pass"), llvm::cl::init(true)};

    BoolOption enableHandleLargePads{*this, "handle-large-pads", llvm::cl::desc("Enable handle-large-pads pass"),
                                     llvm::cl::init(true)};

    BoolOption enableHandleAsymmetricStrides{*this, "handle-asymmetric-strides",
                                             llvm::cl::desc("Enable handle-asymmetric-strides pass"),
                                             llvm::cl::init(true)};

    BoolOption enableBilinearInterpolateOnDPU{*this, "map-interpolate-on-dpu",
                                              llvm::cl::desc("Enable map-interpolate-on-dpu pass"),
                                              llvm::cl::init(true)};

    BoolOption enableSplitBilinerIntoHAndW{*this, "split-bilinear-into-H-and-W",
                                           llvm::cl::desc("Enable split-bilinear-into-H-and-W pass"),
                                           llvm::cl::init(false)};

    BoolOption enableUpstreamSlice{*this, "upstream-slice", llvm::cl::desc("Enable upstream-slice pipeline building"),
                                   llvm::cl::init(true)};

    BoolOption enableExpandActivationChannels{*this, "expand-activation-channels",
                                              llvm::cl::desc("Enable expand-activation-channels pass"),
                                              llvm::cl::init(true)};

    BoolOption enableAdjustConvShapePass{*this, "adjust-convolution-shape",
                                         llvm::cl::desc("Enable adjust-convolution-shape pass"), llvm::cl::init(true)};

    BoolOption enableOptimizeSliceExpand{*this, "optimize-slice-expand",
                                         llvm::cl::desc("Enable optimize-slice-expand pass"), llvm::cl::init(true)};

    BoolOption enableOptimizeSliceWithStride{*this, "optimize-slice-with-stride",
                                             llvm::cl::desc("Enable optimize-slice-with-stride pass"),
                                             llvm::cl::init(true)};

    BoolOption enableConvertExpandToConvPass{*this, "convert-expand-to-conv",
                                             llvm::cl::desc("Enable convert-expand-to-conv pass"),
                                             llvm::cl::init(true)};

    BoolOption logOpOptimizations{*this, "log-op-optimizations",
                                  llvm::cl::desc("Log potential operation optimizations that can be done"),
                                  llvm::cl::init(false)};

    // AdjustPrecisionOptions
    BoolOption enableConvertPrecisionToFP16{*this, "convert-precision-to-fp16",
                                            llvm::cl::desc("Enable convert-precision-to-fp16 pass"),
                                            llvm::cl::init(true)};

    // TransformOptions
    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     llvm::cl::init(true)};

    BoolOption enableWDBlockArgumentInput{
            *this, "enable-wd-blockarg-input",
            llvm::cl::desc("Enable WeightsDequantizeToFakeQuantizePass on structures with BlockArgument input"),
            llvm::cl::init(true)};

    BoolOption fuseMvn6ScaleBias{*this, "fuse-mvn6-scale-bias", llvm::cl::desc("Enable fuse-mvn6-scale-bias pass"),
                                 llvm::cl::init(false)};
    // AdjustLayoutOptions

    BoolOption enableOptimizeReorders{*this, "optimize-reorders", llvm::cl::desc("Enable optimize-reorders pass"),
                                      llvm::cl::init(true)};

    BoolOption enableForceZMajorConcat{*this, "force-z-major-concat",
                                       llvm::cl::desc("Enable transpose-reorder-concat pass"), llvm::cl::init(true)};

    // LowPrecisionOptions
    BoolOption enableLowPrecision{*this, "low-precision", llvm::cl::desc("Enable low-precision pipeline building"),
                                  llvm::cl::init(true)};

    BoolOption enableQuantDequantRemoval{*this, "quant-dequant-removal",
                                         llvm::cl::desc("Enable quantize->dequantize sequence removal"),
                                         llvm::cl::init(false)};

    BoolOption enableFuseOutstandingDequant{*this, "fuse-outstanding-dequant",
                                            llvm::cl::desc("Fuse outstanding dequantize after NCE task"),
                                            llvm::cl::init(false)};

    BoolOption enableFuseOutstandingQuant{*this, "fuse-outstanding-quant",
                                          llvm::cl::desc("Fuse outstanding quantize before two-input Eltwise task"),
                                          llvm::cl::init(false)};

    BoolOption enableSwapTransposeWithFQ{*this, "swap-transpose-with-fq",
                                         ::llvm::cl::desc("Enable SwapTransposeWithFQ pass"), ::llvm::cl::init(true)};

    BoolOption enablePropagateQuantDequant{*this, "propagate-quant-dequant",
                                           llvm::cl::desc("Enable Propagate Quantize Dequantize pass"),
                                           llvm::cl::init(true)};

    BoolOption enableAlignScales{*this, "enable-align-scales", llvm::cl::desc("Enable align scales"),
                                 llvm::cl::init(true)};

    BoolOption enableAdjustNonZeroFakeQuant{*this, "adjust-non-zero-fake-quant",
                                            llvm::cl::desc("Enable adjust non zero fake quant"), llvm::cl::init(true)};

    // LowPrecisionOptions(only for 37XX)
    BoolOption enableFP16ToU8MixedMode{
            *this, "enable-fp16-to-u8-mixed-mode",
            llvm::cl::desc("Enable mixed mode for NCE tasks with FP16 input and quantized output"),
            llvm::cl::init(false)};

    BoolOption enableFloatInQuantWeightsMixedMode{
            *this, "enable-float-in-quant-weights-mixed-mode",
            llvm::cl::desc("Enable mixed mode for NCE tasks with float input and quantized weights"),
            llvm::cl::init(true)};

    // LowPrecisionOptions(37XX+)
    BoolOption enableConvolutionMixedPrecisionDecomposition{
            *this, "enable-convolution-mixed-precision-decomposition",
            llvm::cl::desc("Enable mixed precision decomposition for convolution"), llvm::cl::init(false)};

    // Common
    BoolOption enableFuseClampOperations{*this, "enable-fuse-clamp-op", llvm::cl::desc("Enable fuse clamp operations"),
                                         llvm::cl::init(false)};

    BoolOption enableDynamicQuant{*this, "enable-dynamic-quant",
                                  llvm::cl::desc("Enable dynamic quant weights signal pass."), llvm::cl::init(false)};
};

//
// Registration
//

void registerIEPipelines();
void registerPasses();

}  // namespace IE
}  // namespace vpux
