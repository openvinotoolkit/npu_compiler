//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>

#include <functional>
#include <memory>
#include <optional>

namespace vpux {
namespace VPU {

using SparsityProfileCreateFunc = std::function<std::optional<VPU::ActivationSparsityProfile>(StringRef)>;

//
// Activation sparsity options
//

struct ActivationSparsityOptions : mlir::PassPipelineOptions<ActivationSparsityOptions> {
    StrOption enableActivationSparsity{*this, "enable-activation-sparsity",
                                       llvm::cl::desc("Enable activation sparsity"), llvm::cl::init("auto")};
    StrOption actSparsityProfile{*this, "act-sparsity-profile", llvm::cl::desc("Activation sparsity profile"),
                                 llvm::cl::init("NONE")};

    ActivationSparsityOptions() = default;

    template <class OtherOptions>
    explicit ActivationSparsityOptions(const OtherOptions& options) {
        enableActivationSparsity = options.enableActivationSparsity;
        actSparsityProfile = options.actSparsityProfile;
    }
};

//
// Weights sparsity options
//

struct WeightsSparsityOptions : mlir::PassPipelineOptions<WeightsSparsityOptions> {
    StrOption weightsSparsityHeuristic{*this, "weights-sparsity-heuristic",
                                       llvm::cl::desc("Weights sparsity heuristic (ratio or cmx)"),
                                       llvm::cl::init("ratio")};
    DoubleOption weightsSparsityThreshold{*this, "weights-sparsity-threshold",
                                          llvm::cl::desc("Weights sparsity threshold")};
    Int64Option weightsSparsityLargeConstThreshold{*this, "weights-sparsity-large-const-threshold",
                                                   llvm::cl::desc("Weights sparsity large const threshold")};
    Int64Option weightsSparsityComputeOpThreshold{
            *this, "weights-sparsity-compute-op-threshold",
            llvm::cl::desc("Minimum number of compute operations where fragmentation is likely")};
    BoolOption enableWeightSwizzling{*this, "enable-weights-swizzling", ::llvm::cl::desc("Enable weights swizzling"),
                                     ::llvm::cl::init(true)};
    WeightsSparsityOptions() = default;

    template <class OtherOptions>
    explicit WeightsSparsityOptions(const OtherOptions& options) {
        weightsSparsityHeuristic = options.weightsSparsityHeuristic;
        weightsSparsityThreshold = options.weightsSparsityThreshold;
        weightsSparsityLargeConstThreshold = options.weightsSparsityLargeConstThreshold;
        weightsSparsityComputeOpThreshold = options.weightsSparsityComputeOpThreshold;
        enableWeightSwizzling = options.enableWeightsSwizzling;
    }
};

//
// Tiling options
//

struct TilingOptions : mlir::PassPipelineOptions<TilingOptions> {
    BoolOption enablePrefetchTiling{*this, "enable-prefetch", llvm::cl::desc("Enable prefetch mode"),
                                    llvm::cl::init(true)};

    BoolOption enableVPUNNCostForTiling{*this, "enable-vpunn-cost-for-tiling",
                                        llvm::cl::desc("Use VPUNN cost model to get the best tiling strategy"),
                                        llvm::cl::init(false)};

    BoolOption enableOutputPipelining{*this, "output-pipelining", llvm::cl::desc("Enable output pipelining"),
                                      llvm::cl::init(false)};

    BoolOption enableVerticalFusion{*this, "vertical-fusion", llvm::cl::desc("Enable vertical fusion feature"),
                                    llvm::cl::init(false)};

    BoolOption enableVerticalFusionPipelining{*this, "vertical-fusion-pipelining",
                                              llvm::cl::desc("Enable vertical fusion pipelining"),
                                              llvm::cl::init(false)};

    IntOption opTilingCacheThreshold{
            *this, "op-tiling-cache-threshold",
            llvm::cl::desc("threshold for number of clustered ops for tiling cache optimization"),
            llvm::cl::init(CLUSTERED_OP_THRESHOLD_FOR_TILING_CACHE)};

    IntOption vfOutliningInstanceThreshold{
            *this, "vf-outlining-instance-threshold",
            llvm::cl::desc("Threshold for number of instances (slices of the graph) to perform outlining"),
            llvm::cl::init(5)};

    IntOption vfOutliningTileThreshold{
            *this, "vf-outlining-tile-threshold",
            llvm::cl::desc("Threshold for outlining vertical fusion regions with accumulated number of tiles"),
            llvm::cl::init(10)};

    BoolOption enableVerticalFusionOutlining{*this, "vf-outlining", llvm::cl::desc("Enable vertical fusion outlining"),
                                             llvm::cl::init(true)};

    BoolOption enableProfiling{*this, "profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};

    StrOption enableShaveDDRAccessOptimization{
            *this, "enable-shave-ddr-access-optimization",
            llvm::cl::desc("SHAVE DDR access optimization option (true, false or auto)"), llvm::cl::init("true")};

    // Extended Tiling options - Incremental Pipeline
    BoolOption readStrategyFromJson{*this, "read-strategy-from-json",
                                    llvm::cl::desc("Read the multiclustering and tiling strategy from a JSON file"),
                                    llvm::cl::init(false)};

    BoolOption writeStrategyToJson{*this, "write-strategy-to-json",
                                   llvm::cl::desc("Write the multiclustering and tiling strategy to a JSON file"),
                                   llvm::cl::init(false)};

    StrOption modelHash{*this, "model-hash", llvm::cl::desc("Hash of model XML architecture"), llvm::cl::init("")};

    TilingOptions() = default;

    template <class OtherOptions>
    explicit TilingOptions(const OtherOptions& options) {
        enablePrefetchTiling = options.enablePrefetching;
        enableVPUNNCostForTiling = options.enableVPUNNCostForTiling;
        enableOutputPipelining = options.enableOutputPipelining;
        enableVerticalFusion = options.enableVerticalFusion;
        opTilingCacheThreshold = options.opTilingCacheThreshold;
        vfOutliningInstanceThreshold = options.vfOutliningInstanceThreshold;
        vfOutliningTileThreshold = options.vfOutliningTileThreshold;
        enableVerticalFusionOutlining = options.enableVerticalFusionOutlining;
        enableProfiling = options.enableProfiling;
        enableVerticalFusionPipelining = options.enablePipelining;
        enableShaveDDRAccessOptimization = options.enableShaveDDRAccessOptimization;
        readStrategyFromJson = options.readStrategyFromJson;
        writeStrategyToJson = options.writeStrategyToJson;
        modelHash = options.modelHash;
    }
};

//
// InitCompiler options
//

struct InitCompilerOptions : mlir::PassPipelineOptions<InitCompilerOptions> {
    // InitResources pass options
    StrOption arch{*this, "vpu-arch", ::llvm::cl::desc("VPU architecture to compile for")};
    StrOption compilationMode{*this, "compilation-mode",
                              ::llvm::cl::desc("[Optional] Set compilation mode as `ReferenceSW` or `DefaultHW`"),
                              ::llvm::cl::init("DefaultHW")};
    IntOption revisionID{*this, "revision-id", ::llvm::cl::desc("[Optional] Revision ID of the platform")};
    IntOption numberOfDPUGroups{*this, "num-of-dpu-groups",
                                ::llvm::cl::desc("[Optional] Number of available DPU groups")};
    IntOption numberOfDMAPorts{*this, "num-of-dma-ports", ::llvm::cl::desc("[Optional] Number of available DMA ports")};
    IntOption availableCMXMemory{*this, "available-cmx-memory", ::llvm::cl::desc("[Optional] Available CMX memory")};
    BoolOption allowCustomValues{*this, "allow-custom-values",
                                 ::llvm::cl::desc("[Optional] Allows keep predefined values in IR")};

    // SetupBarrierVariantConstraints pass options
    BoolOption workloadManagementEnable{*this, "workload-management-enable",
                                        llvm::cl::desc("Enable partial workload management"), llvm::cl::init(true)};
    BoolOption wlmRollback{
            *this, "wlm-rollback",
            llvm::cl::desc("When compilation with WLM fails, automatically switches to WLM-disabled pipeline"),
            llvm::cl::init(true)};

    // SetupChannelsAutoPadding pass options
    BoolOption enableAutoPaddingODU{*this, "enable-auto-padding-odu",
                                    llvm::cl::desc("Enable auto padding for output channels"), llvm::cl::init(false)};

    // SetupChannelsAutoPadding pass options
    BoolOption enableAutoPaddingIDU{*this, "enable-auto-padding-idu",
                                    llvm::cl::desc("Enable auto padding for output channels"), llvm::cl::init(false)};

    // SetupIsReduceSupported pass options
    BoolOption enableIsReduceSupported{*this, "enable-is-reduce-supported",
                                       ::llvm::cl::desc("[Optional] Set IsReduceSupported for NCE to true/false"),
                                       ::llvm::cl::init(false)};

    // SetupEnableFP16CompressedConv pass option
    BoolOption enableFP16CompressedConvolution{*this, "enable-fp16-compressed-convolution",
                                               llvm::cl::desc("Enable FP16 Compressed convolution op"),
                                               llvm::cl::init(false)};

    StrOption ppeVersion{
            *this, "ppe-version",
            ::llvm::cl::desc("Specifies the compiler's target PPE hardware version ['Auto', 'IntPPE', 'FpPPE']. When "
                             "set to 'Auto', the latest PPE version available on the target architecture is picked."),
            ::llvm::cl::init("Auto")};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableWeightsDynamicDequantization{*this, "enable-weights-dynamic-dequantization",
                                                  llvm::cl::desc("Enable weights dequantization for weights as input"),
                                                  llvm::cl::init(false)};

    BoolOption enableAdaptiveStripping{*this, "enable-adaptive-stripping", llvm::cl::desc("Enable adaptive stripping"),
                                       llvm::cl::init(false)};

    InitCompilerOptions() = default;

    template <class OtherOptions>
    InitCompilerOptions(ArchKind archParam, CompilationMode compilationModeParam, const OtherOptions& options) {
        arch = std::string(VPU::stringifyEnum(archParam));
        compilationMode = std::string(VPU::stringifyEnum(compilationModeParam));

        this->matchAndCopyOptionValuesFrom(options);
    }

    InitCompilerOptions(ArchKind archParam, CompilationMode compilationModeParam,
                        std::optional<int> revisionIDParam = std::nullopt,
                        std::optional<int> numberOfDPUGroupsParam = std::nullopt,
                        std::optional<int> numberOfDMAPortsParam = std::nullopt,
                        std::optional<bool> wlmRollbackParam = std::nullopt,
                        std::optional<Byte> availableCMXMemoryParam = std::nullopt,
                        std::optional<bool> enableFP16CompressedConvolutionParam = std::nullopt,
                        std::optional<bool> enableAutoPaddingIDUParam = std::nullopt,
                        std::optional<bool> enableAutoPaddingODUParam = std::nullopt,
                        std::optional<bool> enableSEPtrsOperationsParam = std::nullopt,
                        std::optional<bool> enableExperimentalSEPtrsOperationsParam = std::nullopt,
                        std::optional<bool> enableReduceOperationsParam = std::nullopt,
                        std::optional<bool> compilerDynamicQuantizationParam = std::nullopt,
                        std::optional<bool> enableAdaptiveStrippingParam = std::nullopt) {
        arch = std::string(VPU::stringifyEnum(archParam));
        compilationMode = std::string(VPU::stringifyEnum(compilationModeParam));

        maybeSetValue(revisionID, revisionIDParam);
        maybeSetValue(numberOfDPUGroups, numberOfDPUGroupsParam);
        maybeSetValue(numberOfDMAPorts, numberOfDMAPortsParam);
        maybeSetValue(wlmRollback, wlmRollbackParam);
        maybeSetValue(enableFP16CompressedConvolution, enableFP16CompressedConvolutionParam);
        setAvailableCMXMemory(availableCMXMemoryParam);
        maybeSetValue(enableAutoPaddingIDU, enableAutoPaddingIDUParam);
        maybeSetValue(enableAutoPaddingODU, enableAutoPaddingODUParam);
        maybeSetValue(enableSEPtrsOperations, enableSEPtrsOperationsParam);
        maybeSetValue(enableExperimentalSEPtrsOperations, enableExperimentalSEPtrsOperationsParam);
        maybeSetValue(enableIsReduceSupported, enableReduceOperationsParam);
        maybeSetValue(enableWeightsDynamicDequantization, compilerDynamicQuantizationParam);
        maybeSetValue(enableAdaptiveStripping, enableAdaptiveStrippingParam);
    }

public:
    void setAvailableCMXMemory(std::optional<Byte> maybeAvailableCMXMemory) {
        if (maybeAvailableCMXMemory.has_value()) {
            availableCMXMemory = maybeAvailableCMXMemory.value().count();
        }
    }

    void setNumberOfDPUGroups(std::optional<int> maybeNumberOfDPUGroups) {
        maybeSetValue(numberOfDPUGroups, maybeNumberOfDPUGroups);
    }

    void setNumberOfDMAPorts(std::optional<int> maybeNumberOfDMAPorts) {
        maybeSetValue(numberOfDMAPorts, maybeNumberOfDMAPorts);
    }

private:
    template <typename OptionType, typename ValType>
    static void maybeSetValue(OptionType& option, std::optional<ValType> value) {
        if (value.has_value()) {
            option = value.value();
        }
    }
};

//
// Passes
//

std::unique_ptr<mlir::Pass> createInitResourcesPass();
std::unique_ptr<mlir::Pass> createInitResourcesPass(const InitCompilerOptions& initCompilerOptions,
                                                    Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createOptimizeSharedInputCopyForConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCMXConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitNCEOpsOntoWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveEltwiseWithZTiledWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createShiftOutputWorkloadsForHaloPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMakeOpsWithDistributedTensorPass(bool enableExplicitDistributionInfoAttr = false,
                                                                   Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMakeDistributedCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustDistributedTensorAroundOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapDistributedOpsInNCEClusterTiling(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustMemorySpacePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMultiClusterStrategyAssignmentPass(
        bool enablePrefetchTiling = true, bool enableMcSideLoadingDump = false,
        const int clusteredOpThreshold = CLUSTERED_OP_THRESHOLD_FOR_TILING_CACHE,
        StringRef mcOptimizationScope = "subgraph", StringRef modelHash = "", Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createManualStrategyUtilsPass();
std::unique_ptr<mlir::Pass> createManualStrategyUtilsPass(bool writeStrategyToJSON,
                                                          StringRef writeStrategyFileLocation = "strategy_out.json",
                                                          bool readStrategyFromJSON = false,
                                                          StringRef readStrategyFileLocation = "strategy_in.json",
                                                          bool enableSideLoadDump = false, StringRef modelHash = "",
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createManualStrategyUtilsPass(bool writeStrategyToJSON,
                                                          StringRef writeStrategyFileLocation = "strategy_out.json",
                                                          bool readStrategyFromJSON = false,
                                                          StringRef readStrategyFileLocation = "strategy_in.json",
                                                          bool updateStrategyForOutputPipelining = false,
                                                          bool enableSideLoadDump = false, StringRef modelHash = "",
                                                          Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createDetectionOutputDecompositionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitGRUSequencePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddSwOpAuxiliaryBufferPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createTileLSTMSequencePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustLSTMCellInputsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createComputeInterpolateCoordinatesPass(bool enableExplicitDistributionInfoAttr = false,
                                                                    Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRelocateWeightTableForReusePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDetectInPlaceEltwisePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseNCEInterpolateConsumersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddExplicitPaddingBeforeNCEPermutePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOutputPipelineTilingPass(bool enablePrefetchTiling = true,
                                                           Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLegalizeDynamicShapeConcatForSWLayersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertConstArgsToMultiConstantsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConcatRepeatingBlocksOutliningPass(int64_t minSeqLength = 1,
                                                                     const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createOutlineEntireMainContentPass(const Logger& log = Logger::global());

void buildInitCompilerPipeline(mlir::OpPassManager& pm, const VPU::InitCompilerOptions& options,
                               Logger log = Logger::global());

//
// Sparsity
//

std::unique_ptr<mlir::Pass> createSparsifyWeightsPass(
        VPU::WeightsSparsityHeuristic heuristic = VPU::WeightsSparsityHeuristic::RATIO,
        std::optional<double> manualThreshold = std::nullopt,
        int64_t largeConstThreshold = (200_MB).to<vpux::Byte>().count(), int64_t computeOpThreshold = 350,
        bool enableWeightSwizzling = true, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRecomputeSparsityPtrsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseSparsityOpsPass(std::optional<bool> fuseSparsify = std::nullopt,
                                                      Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createOptimizeSparsifyDesparsifyPairsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSparsifyDesparsifyPairsPass(const VPU::ActivationSparsityOptions& options,
                                                                      Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createOptimizeSparsityOpsPass(SparsityProfileCreateFunc sparsityProfileCreateCb,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapOpsInSparsifyDesparsifyPairsPass();
std::unique_ptr<mlir::Pass> createWrapOpsInSparsifyDesparsifyPairsPass(
        VPU::EnableActivationSparsityMode enableActivationSparsityMode,
        VPU::ActivationSparsityProfile actSparsityProfile, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddSparsityMapToSparseActivationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLowerSparsityOpsPass(std::optional<bool> fakeSparsify = std::nullopt,
                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitSEOpsPass(const bool seOpsEnabled = false,
                                                 const bool enableExperimentalSEPtrsOperations = false,
                                                 Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLowerOpsToSENCEPass(const bool seOpsEnabled = false,
                                                      const bool enableExperimentalSEPtrsOperations = false,
                                                      Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertOpToDMAForPerformantExecutionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createTileGatherPass(Logger log = Logger::global());

//
// Tiling
//

std::unique_ptr<mlir::Pass> createTilingStrategyAssignmentPass(bool enablePrefetchTiling = true,
                                                               bool enableVpunnCostForTiling = false,
                                                               StringRef enableShaveDDRAccessOptimization = "true",
                                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createApplyTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapVerticalFusionRegionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveViewOpsToVerticalFusionPass(Logger log = Logger::global());
// Tracking number [E#76838]
// Turn on the enableVerticalFusionPipelining when VF pipelining is enabled
std::unique_ptr<mlir::Pass> createMergeVfSubgraphsPass(bool enableVerticalFusionPipelining = false,
                                                       bool enablePrefetchTiling = true, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createVfTilingPass(bool enableVerticalFusionPipelining = false,
                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createVerticalFusionOutliningPass();
std::unique_ptr<mlir::Pass> createVerticalFusionOutliningPass(const TilingOptions& options,
                                                              Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollUnusedVerticalFusionRegionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createEnsureNCEOpsSizeRequirementsPass(bool enableOutputEnsurance = true,
                                                                   Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseClampPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createStrategyManagerImplPass(bool enablePrefetchTiling = true,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createEfficientIROrderPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(Logger log = Logger::global());

void buildActivationSparsityPipeline(mlir::OpPassManager& pm, const VPU::ActivationSparsityOptions& options,
                                     Logger log = Logger::global());

void buildWeightsSparsityPipeline(mlir::OpPassManager& pm, const VPU::WeightsSparsityOptions& options,
                                  Logger log = Logger::global());
void buildTilingPipeline(mlir::OpPassManager& pm, const VPU::TilingOptions& options, Logger log = Logger::global());

//
// Strategy Pipeline
//

void buildVFPipeline(mlir::OpPassManager& pm, const VPU::TilingOptions& options, Logger log = Logger::global());
void buildSMPipeline(mlir::OpPassManager& pm, const vpux::MCAndTilingOptionsBase& options,
                     Logger log = Logger::global());

//
// Setup Pipeline Options
//

std::unique_ptr<mlir::Pass> createSetupPipelineOptionsPass();
std::unique_ptr<mlir::Pass> createSetupPipelineOptionsPass(const InitCompilerOptions& initCompilerOptions,
                                                           Logger log = Logger::global());

//
// Npu Constraints
//

std::unique_ptr<mlir::Pass> createSetupNpuConstraintPass();
std::unique_ptr<mlir::Pass> createSetupNpuConstraintPass(const InitCompilerOptions& initCompilerOptions,
                                                         Logger log = Logger::global());

//
// Setup Max Kernel Size
//

std::unique_ptr<mlir::Pass> createSetupMaxKernelSizePass();
std::unique_ptr<mlir::Pass> createSetupMaxKernelSizePass(const InitCompilerOptions& initCompilerOptions,
                                                         Logger log = Logger::global());

//
// Channels Auto Padding
//

std::unique_ptr<mlir::Pass> createSetupChannelsAutoPaddingPass();
std::unique_ptr<mlir::Pass> createSetupChannelsAutoPaddingPass(const InitCompilerOptions& initCompilerOptions,
                                                               Logger log = Logger::global());

//
// Reduce Operation
//

std::unique_ptr<mlir::Pass> createSetupIsReduceSupportedPass();
std::unique_ptr<mlir::Pass> createSetupIsReduceSupportedPass(const InitCompilerOptions& initCompilerOptions,
                                                             Logger log = Logger::global());

//
// FP16 Compressed Convolution
//

std::unique_ptr<mlir::Pass> createSetupEnableFP16CompressedConvPass();
std::unique_ptr<mlir::Pass> createSetupEnableFP16CompressedConvPass(const InitCompilerOptions& initCompilerOptions,
                                                                    Logger log = Logger::global());

//
// SEPtrs Operations
//

std::unique_ptr<mlir::Pass> createSetupEnableSEPtrsOperationsPass();
std::unique_ptr<mlir::Pass> createSetupEnableSEPtrsOperationsPass(const InitCompilerOptions& initCompilerOptions,
                                                                  Logger log = Logger::global());

//
// Weights separation
//

std::unique_ptr<mlir::Pass> createIntroduceInitFunctionPass(const Logger& log = Logger::global());

//
// Adaptive Stripping
//

std::unique_ptr<mlir::Pass> createSetupEnableAdaptiveStrippingPass();
std::unique_ptr<mlir::Pass> createSetupEnableAdaptiveStrippingPass(const InitCompilerOptions& initCompilerOptions,
                                                                   Logger log = Logger::global());

//
// DefaultHWOptions(for all devices)
//
struct DefaultHWOptionsDialectBase : public virtual vpux::DefaultHWOptionsBase {
    BoolOption enableInPlaceEltwise{*this, "enable-in-place-eltwise",
                                    llvm::cl::desc("Enable inplace eltwise op execution"), llvm::cl::init(true)};

    BoolOption enableSMPipeline{*this, "enable-SM-Pipeline", llvm::cl::desc("Enable Strategy Manager pipeline"),
                                llvm::cl::init(false)};

    // WeightsSparsityOptions
    StrOption weightsSparsityHeuristic{*this, "weights-sparsity-heuristic",
                                       llvm::cl::desc("Weights sparsity heuristic (RATIO or CMX)"),
                                       llvm::cl::init("RATIO")};

    DoubleOption weightsSparsityThreshold{*this, "weights-sparsity-threshold",
                                          llvm::cl::desc("Threshold for ratio of sparse weights values"),
                                          llvm::cl::init(-1.0)};
    Int64Option weightsSparsityLargeConstThreshold{
            *this, "weights-sparsity-large-const-threshold",
            llvm::cl::desc(
                    "Sparsify weights using a single thread if the constant's size is larger than this threshold."),
            llvm::cl::init((200_MB).to<vpux::Byte>().count())};
    IntOption weightsSparsityComputeOpThreshold{
            *this, "weights-sparsity-compute-op-threshold",
            llvm::cl::desc("Minimum number of compute operations where fragmentation is likely"), llvm::cl::init(350)};

    // TilingOptions
    BoolOption enableVerticalFusion{*this, "vertical-fusion", llvm::cl::desc("Enable vertical fusion feature"),
                                    llvm::cl::init(true)};

    BoolOption readStrategyFromJson{*this, "read-strategy-from-json",
                                    llvm::cl::desc("Read the multiclustering and tiling strategy from a JSON file"),
                                    llvm::cl::init(false)};

    BoolOption writeStrategyToJson{*this, "write-strategy-to-json",
                                   llvm::cl::desc("Write the multiclustering and tiling strategy to a JSON file"),
                                   llvm::cl::init(false)};

    StrOption enableShaveDDRAccessOptimization{
            *this, "enable-shave-ddr-access-optimization",
            llvm::cl::desc("SHAVE DDR access optimization option (true, false or auto)"), llvm::cl::init("true")};
};

//
// Registration
//

void registerVPUPipelines();
void registerPasses();

}  // namespace VPU
}  // namespace vpux
