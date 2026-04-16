//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>

#include <functional>
#include <memory>
#include <optional>

namespace vpux::VPU {
enum class ActivationSparsityProfile : uint64_t;
enum class WeightsSparsityHeuristic : uint64_t;
}  // namespace vpux::VPU

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
        this->matchAndCopyOptionValuesFrom(options);
    }
};

//
// Weights sparsity options
//

struct WeightsSparsityOptions : mlir::PassPipelineOptions<WeightsSparsityOptions> {
    StrOption weightsSparsityHeuristic{*this, "weights-sparsity-heuristic",
                                       llvm::cl::desc("Weights sparsity heuristic (RATIO or CMX)"),
                                       llvm::cl::init("RATIO")};
    DoubleOption weightsSparsityThreshold{*this, "weights-sparsity-threshold",
                                          llvm::cl::desc("Threshold for ratio of sparse weights values"),
                                          llvm::cl::init(-1.0)};
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
        this->matchAndCopyOptionValuesFrom(options);
    }
};

//
// Tiling options
//

struct TilingOptions : mlir::PassPipelineOptions<TilingOptions> {
    BoolOption enablePrefetchTiling{*this, "prefetching", llvm::cl::desc("Enable prefetch mode"), llvm::cl::init(true)};

    BoolOption enableVPUNNCostForTiling{*this, "enable-vpunn-cost-for-tiling",
                                        llvm::cl::desc("Use VPUNN cost model to get the best tiling strategy"),
                                        llvm::cl::init(false)};

    BoolOption enableOutputPipelining{*this, "output-pipelining", llvm::cl::desc("Enable output pipelining"),
                                      llvm::cl::init(false)};

    BoolOption enableVerticalFusion{*this, "vertical-fusion", llvm::cl::desc("Enable vertical fusion feature"),
                                    llvm::cl::init(false)};

    BoolOption enableVerticalFusionPipelining{*this, "pipelining", llvm::cl::desc("Enable vertical fusion pipelining"),
                                              llvm::cl::init(false)};

    BoolOption enableSCFTiling{*this, "scf-tiling", llvm::cl::desc("Enable tiling using SCF dialect"),
                               llvm::cl::init(false)};

    BoolOption enableDynamicDimAlignment{
            *this, "dynamic-dim-alignment",
            llvm::cl::desc(
                    "Align dynamic dimensions during tiling to make it easier to merge strided DMAs into batched DMAs"),
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

    BoolOption enableVFScheduleTrace{*this, "enable-vf-schedule-trace",
                                     llvm::cl::desc("Enable vertical fusion scheduling trace"), llvm::cl::init(false)};

    BoolOption enableVerticalFusionOutlining{*this, "vf-outlining", llvm::cl::desc("Enable vertical fusion outlining"),
                                             llvm::cl::init(true)};

    BoolOption enableProfiling{*this, "profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};
    BoolOption enableProfilingWithOutlining{*this, "profiling-with-outlining",
                                            llvm::cl::desc("Enable profiling with outlining"), llvm::cl::init(false)};

    StrOption enableShaveDDRAccessOptimization{
            *this, "enable-shave-ddr-access-optimization",
            llvm::cl::desc("SHAVE DDR access optimization option (true, false or auto)"), llvm::cl::init("true")};

    BoolOption enableReorderConcatBranches{
            *this, "enable-reorder-concat-branches",
            llvm::cl::desc("Reorder branches of concat to make sure it is executed branch by branch"),
            llvm::cl::init(false)};

    BoolOption enablePrintStatistics{*this, "enable-print-statistics", ::llvm::cl::desc("Enable print statistics"),
                                     ::llvm::cl::init(vpux::isDeveloperBuild())};

    // Extended Tiling options - Incremental Pipeline
    BoolOption readStrategyFromJson{*this, "read-strategy-from-json",
                                    llvm::cl::desc("Read the multiclustering and tiling strategy from a JSON file"),
                                    llvm::cl::init(false)};

    BoolOption writeStrategyToJson{*this, "write-strategy-to-json",
                                   llvm::cl::desc("Write the multiclustering and tiling strategy to a JSON file"),
                                   llvm::cl::init(false)};
    BoolOption dumpStrategyToLog{*this, "dump-strategy-to-log",
                                 llvm::cl::desc("Dump the multiclustering and tiling strategy to log info"),
                                 llvm::cl::init(false)};

    mlir::detail::PassOptions::Option<WorkloadManagementMode> workloadManagementMode{
            *this, "workload-management-mode",
            ::llvm::cl::desc("Option for enabling WLM enqueue barriers search algorithm at VPURT. To be used only for "
                             "experiments."),
            ::llvm::cl::init(WorkloadManagementMode::PWLM_V0_1_PAGES),
            ::llvm::cl::values(clEnumValN(WorkloadManagementMode::PWLM_V2_PAGES, "PWLM_V2_PAGES",
                                          "WLM with split into subgraphs (pages)"),
                               clEnumValN(WorkloadManagementMode::PWLM_V1_BARRIER_FIFO, "PWLM_V1_BARRIER_FIFO",
                                          "WLM enqueue barriers search algorithm at VPURT ENABLED"),
                               clEnumValN(WorkloadManagementMode::PWLM_V0_1_PAGES, "PWLM_V0_1_PAGES",
                                          "PWLM with split into subgraphs (pages)"),
                               clEnumValN(WorkloadManagementMode::PWLM_V0_LCA, "PWLM_V0_LCA",
                                          "WLM enqueue barriers search algorithm at VPURT DISABLED. Use LCA based "
                                          "enqueue algorithm at VPUMI"))};

    TilingOptions() = default;

    template <class OtherOptions>
    explicit TilingOptions(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
    }
};

//
// ScfComputeOpsOutliningOptions
//

struct ScfComputeOpsOutliningOptions : mlir::PassPipelineOptions<ScfComputeOpsOutliningOptions> {
    StrOption loopUnrollFactor{*this, "loop-unroll-factor", llvm::cl::desc("Loop unroll factor"), llvm::cl::init("")};

    BoolOption enableProfiling{*this, "profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};

    BoolOption enableCascadedUnrolling{*this, "cascaded-unrolling", llvm::cl::desc("Enable cascaded loop unrolling"),
                                       llvm::cl::init(false)};

    ScfComputeOpsOutliningOptions() = default;
};

//
// InitCompiler options
//

struct InitCompilerOptions : mlir::PassPipelineOptions<InitCompilerOptions> {
    // InitResources pass options

    // 'platform' represent the compilation target and supersedes 'arch'.
    // 'arch' is used in test scenarios where specific target might not be relevant
    // Either 'platform' or 'vpu-arch' shall be set.
    StrOption platform{*this, "platform", ::llvm::cl::desc("Target NPU platform")};
    StrOption arch{*this, "vpu-arch", ::llvm::cl::desc("Target NPU architecture")};
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

    BoolOption enableSwKernelFifoPerShaveEngine{*this, "enable-sw-kernel-fifo-per-shave-engine",
                                                llvm::cl::desc("Enable dedicated FIFO for each ActShave engine"),
                                                llvm::cl::init(false)};

    // SetupChannelsAutoPadding pass options
    BoolOption enableAutoPaddingODU{*this, "enable-auto-padding-odu",
                                    llvm::cl::desc("Enable auto padding for output channels"), llvm::cl::init(false)};

    // SetupChannelsAutoPadding pass options
    BoolOption enableAutoPaddingIDU{*this, "enable-auto-padding-idu",
                                    llvm::cl::desc("Enable auto padding for output channels"), llvm::cl::init(false)};

    // SetupSprLUT pass options
    BoolOption enableSprLUT{*this, "enable-sprlut", llvm::cl::desc("Enable sprLUT"), llvm::cl::init(false)};

    // SetupIsReduceSupported pass options
    BoolOption enableIsReduceSupported{*this, "enable-is-reduce-supported",
                                       ::llvm::cl::desc("[Optional] Set IsReduceSupported for NCE to true/false"),
                                       ::llvm::cl::init(false)};

    // SetupEnableFP16CompressedConv pass option
    BoolOption enableFP16CompressedConvolution{*this, "enable-fp16-compressed-convolution",
                                               llvm::cl::desc("Enable FP16 Compressed convolution op"),
                                               llvm::cl::init(false)};

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

    BoolOption enableQDQOptimizationAggressive{*this, "enable-qdq-optimization-aggressive",
                                               llvm::cl::desc("Enable aggressive QDQ optimizations"),
                                               llvm::cl::init(false)};

    BoolOption enableExtraStaticShapeOps{
            *this, "enable-extra-static-shape-ops",
            llvm::cl::desc("Attach StaticShapeOpInterface trait to operations that perform "
                           "faster when their shapes are static."),
            llvm::cl::init(true)};

    mlir::detail::PassOptions::Option<WeightsTableReuseMode> weightsTableReuseMode{
            *this, "weights-table-reuse-mode",
            ::llvm::cl::desc("Option for enabling weights table reuse for different modes."),
            ::llvm::cl::values(clEnumValN(WeightsTableReuseMode::ENABLED, "ENABLED",
                                          "Fully enable weights table reuse for all operations"),
                               clEnumValN(WeightsTableReuseMode::VF_ENABLED, "VF_ENABLED",
                                          "Enable weights table reuse for pure vertical fusion region only"),
                               clEnumValN(WeightsTableReuseMode::DISABLED, "DISABLED",
                                          "Disable weights table reuse for all operations")),
            ::llvm::cl::init(WeightsTableReuseMode::DISABLED)};

    // SetupEnableVPUNNPreSplit pass option
    BoolOption enableVPUNNPreSplit{*this, "enable-vpunn-pre-split", llvm::cl::desc("Enable VPUNN pre-split API"),
                                   llvm::cl::init(false)};

    // SetupEnableODULocalRegion pass option
    BoolOption enableODULocalRegion{*this, "enable-odu-local-region", llvm::cl::desc("Enable ODU local region"),
                                    llvm::cl::init(false)};

    // SetupEnableDCIM pass options
    BoolOption enableDCIM{*this, "enable-dcim", ::llvm::cl::desc("Enable DCIM"), ::llvm::cl::init(true)};
    BoolOption powerOptimized{*this, "power-optimized", llvm::cl::desc("Enable PowerOptimized"), llvm::cl::init(false)};

    BoolOption enableAsymmetricPerTensorZP{
            *this, "enable-asymmetric-per-tensor-zp",
            llvm::cl::desc("Enable asymmetric weights with per tensor zp to run on NCE without dequantization"),
            llvm::cl::init(false)};

    BoolOption enableAsymmetricPerChannelZP{
            *this, "enable-asymmetric-per-channel-zp",
            llvm::cl::desc("Enable asymmetric weights with per channel zp to run on NCE without dequantization"),
            llvm::cl::init(false)};

    BoolOption enableProfiling{*this, "enable-profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};

    InitCompilerOptions() = default;

    // options setup
    template <class OtherOptions>
    InitCompilerOptions(config::Platform platformParam, config::CompilationMode compilationModeParam,
                        const OtherOptions& options) {
        platform = std::string(config::stringifyEnum(platformParam));
        compilationMode = std::string(config::stringifyEnum(compilationModeParam));

        this->matchAndCopyOptionValuesFrom(options);
    }

    // lit-tests
    template <class OtherOptions>
    InitCompilerOptions(config::ArchKind archParam, config::CompilationMode compilationModeParam,
                        const OtherOptions& options) {
        arch = std::string(config::stringifyEnum(archParam));
        compilationMode = std::string(config::stringifyEnum(compilationModeParam));

        this->matchAndCopyOptionValuesFrom(options);
    }

    // PSS tests
    InitCompilerOptions(config::ArchKind archParam, config::CompilationMode compilationModeParam) {
        arch = std::string(config::stringifyEnum(archParam));
        compilationMode = std::string(config::stringifyEnum(compilationModeParam));
        allowCustomValues = true;
    }

public:
    // PSS tests
    void setAvailableCMXMemory(std::optional<Byte> maybeAvailableCMXMemory) {
        if (maybeAvailableCMXMemory.has_value()) {
            availableCMXMemory = maybeAvailableCMXMemory.value().count();
        }
    }

    // PSS tests
    void setNumberOfDPUGroups(std::optional<int> maybeNumberOfDPUGroups) {
        maybeSetValue(numberOfDPUGroups, maybeNumberOfDPUGroups);
    }

    // PSS tests
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

std::unique_ptr<mlir::Pass> createMoveConvertAroundViewLikeOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createComputeNCEInputWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createApplyTilingMVN1SumPass(bool enablePrefetchTiling = true,
                                                         Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeMVNPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitRealDFTOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustForOptimizedLayersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInitResourcesPass();
std::unique_ptr<mlir::Pass> createInitResourcesPass(const InitCompilerOptions& initCompilerOptions,
                                                    Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createDMATaskProfilingReserveMemPass(const std::string& enableDMAProfiling = "false",
                                                                 Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCompressDmaReserveMemPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSWKernelInstructionPrefetchReserveMemForDummyKernelsPass(
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSWKernelDataPrefetchReserveMemPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createShaveStacksReserveMemPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createOptimizeSharedInputCopyForConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCMXConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitNCEOpsOntoWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCorrectNCEWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCreateNewWeightTablesDataPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveEltwiseWithZTiledWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createShiftOutputWorkloadsForHaloPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMakeOpsWithDistributedTensorPass(bool enableExplicitDistributionInfoAttr = false,
                                                                   Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMakeDistributedCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustDistributedTensorAroundOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustMemorySpacePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustMemorySpaceForSHVOpsPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createCostModelAnalysisConstructPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCostModelAnalysisDestroyPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPrintNNCacheStatisticsPass(Logger log = Logger::global(), StringRef passName = "");
std::unique_ptr<mlir::Pass> createMultiClusterStrategyAssignmentPass(
        bool enablePrefetchTiling = true, const int clusteredOpThreshold = CLUSTERED_OP_THRESHOLD_FOR_TILING_CACHE,
        StringRef mcOptimizationScope = "subgraph", Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createManualStrategyUtilsPass();
std::unique_ptr<mlir::Pass> createManualStrategyUtilsPass(bool writeStrategyToJSON,
                                                          StringRef writeStrategyFileLocation = "strategy_out.json",
                                                          bool readStrategyFromJSON = false,
                                                          StringRef readStrategyFileLocation = "strategy_in.json",

                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createManualStrategyUtilsPass(bool writeStrategyToJSON,
                                                          StringRef writeStrategyFileLocation = "strategy_out.json",
                                                          bool readStrategyFromJSON = false,
                                                          StringRef readStrategyFileLocation = "strategy_in.json",
                                                          bool updateStrategyForOutputPipelining = false,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createManualStrategyUtilsPass(
        bool writeStrategyToJSON, StringRef writeStrategyFileLocation = "strategy_out.json",
        bool readStrategyFromJSON = false, StringRef readStrategyFileLocation = "strategy_in.json",
        bool dumpStrategyToLog = false, bool updateStrategyForOutputPipelining = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createManualStrategyUtilsPass(
        bool writeStrategyToJSON, StringRef writeStrategyFileLocation = "strategy_out.json",
        bool readStrategyFromJSON = false, StringRef readStrategyFileLocation = "strategy_in.json",
        bool dumpStrategyToLog = false, bool updateStrategyForOutputPipelining = false,
        std::string contextId = "default", Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDetectionOutputDecompositionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitGRUSequencePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createTileLSTMSequencePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustLSTMCellInputsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createComputeInterpolateCoordinatesPass(bool enableExplicitDistributionInfoAttr = false,
                                                                    Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRelocateWeightTableForReusePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCorrectStorageElementTableSeSizeForSEPDWConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDetectInPlaceEltwisePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseNCEInterpolateConsumersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddExplicitPaddingBeforeNCEPermutePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOutputPipelineTilingPass(bool enablePrefetchTiling = true,
                                                           Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSCFVerticalFusionPass(bool enableDynamicDimAlignment = false,
                                                        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSCFMulticlusteringPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSCFFuseLastViewLikeOpPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createLegalizeDynamicShapeConcatForSWLayersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertConstArgsToMultiConstantsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConcatRepeatingBlocksOutliningPass(int64_t minSeqLength = 1,
                                                                     const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createOutlineEntireMainContentPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createBoundedTensorsToDynamicDimsMaskPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveReflectPadToCMXPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveTensorOpsToCMXPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRunMVNNormalizeOnDPUPass(Logger log = Logger::global());

void buildInitCompilerPipeline(mlir::OpPassManager& pm, const VPU::InitCompilerOptions& options,
                               Logger log = Logger::global());

//
// Sparsity
//

std::unique_ptr<mlir::Pass> createSparsifyWeightsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSparsifyWeightsPass(VPU::WeightsSparsityHeuristic heuristic,
                                                      std::optional<double> manualThreshold = std::nullopt,
                                                      int64_t largeConstThreshold = (200_MB).to<vpux::Byte>().count(),
                                                      int64_t computeOpThreshold = 350,
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
std::unique_ptr<mlir::Pass> createSplitSEOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLowerOpsToSENCEPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertNCEInterpolateToDWPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertOpToDMAForPerformantExecutionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createTileGatherPass(Logger log = Logger::global());

//
// Tiling
//

std::unique_ptr<mlir::Pass> createTilingStrategyAssignmentPass(bool enablePrefetchTiling = true,
                                                               bool enableVpunnCostForTiling = false,
                                                               StringRef enableShaveDDRAccessOptimization = "true",
                                                               bool enableDynamicDimAlignment = false,
                                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createApplyTilingPass(bool enableSCFTiling = false, bool enableDynamicDimAlignment = false,
                                                  Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapVerticalFusionRegionPass(
        const WorkloadManagementMode workloadManagementMode = WorkloadManagementMode::PWLM_V0_1_PAGES,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveViewOpsToVerticalFusionPass(
        const WorkloadManagementMode workloadManagementMode = WorkloadManagementMode::PWLM_V0_1_PAGES,
        Logger log = Logger::global());
// Tracking number [E#76838]
// Turn on the enableVerticalFusionPipelining when VF pipelining is enabled
std::unique_ptr<mlir::Pass> createMergeVfSubgraphsPass(
        bool enableVerticalFusionPipelining = false, bool enablePrefetchTiling = true,
        const WorkloadManagementMode workloadManagementMode = WorkloadManagementMode::PWLM_V0_1_PAGES,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createVfTilingPass(
        bool enableVerticalFusionPipelining = false, bool enableVFScheduleTrace = false,
        const WorkloadManagementMode workloadManagementMode = WorkloadManagementMode::PWLM_V0_1_PAGES,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createVerticalFusionOutliningPass();
std::unique_ptr<mlir::Pass> createVerticalFusionOutliningPass(const TilingOptions& options,
                                                              Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollUnusedVerticalFusionRegionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createEnsureNCEOpsSizeRequirementsPass(
        bool enableOutputEnsurance = true, bool enableDequantWeightEnsuranceBeforeStrategy = true, bool skipOC = false,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvolutionSplitOverInputChannelPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseClampPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseConvertPass(Logger log = Logger::global());

// If optimizeOnlyOuterConcat is true, only optimize when concat dimension is the highest dimension
std::unique_ptr<mlir::Pass> createOptimizeConcatPass(bool optimizeOnlyOuterConcat = false,
                                                     bool disablePassOnEntryFunction = false,
                                                     Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createStrategyManagerImplPass(bool enablePrefetchTiling = true,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createEfficientIROrderPass(bool enableReorderConcatBranches = false,
                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFlashSDPATilingStrategyEstimationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollFlashSDPAPass(Logger log = Logger::global());

void buildActivationSparsityPipeline(mlir::OpPassManager& pm, const VPU::ActivationSparsityOptions& options,
                                     Logger log = Logger::global());

void buildWeightsSparsityPipeline(mlir::OpPassManager& pm, const VPU::WeightsSparsityOptions& options,
                                  Logger log = Logger::global());
void buildTilingPipeline(mlir::OpPassManager& pm, const VPU::TilingOptions& options, Logger log = Logger::global());

//
// Scf Compute Ops outlining Pipeline
//

void buildScfComputeOpsOutliningPipeline(mlir::OpPassManager& pm, const vpux::StrOption& loopUnrollFactor,
                                         bool enableProfiling, const vpux::BoolOption& enableCascadedUnrolling,
                                         Logger log = Logger::global());

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
// Target Independent Options
//

std::unique_ptr<mlir::Pass> createSetTargetIndependentPassOptionsPass();
std::unique_ptr<mlir::Pass> createSetTargetIndependentPassOptionsPass(const InitCompilerOptions& initCompilerOptions,
                                                                      Logger log = Logger::global());
//
// Tiling related contraints
//

std::unique_ptr<mlir::Pass> createSetupTilingConstraintPass();
std::unique_ptr<mlir::Pass> createSetupTilingConstraintPass(const InitCompilerOptions& initCompilerOptions,
                                                            Logger log = Logger::global());

//
// Weights separation
//

std::unique_ptr<mlir::Pass> createConstructWsAnalysisPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createDestructWsAnalysisPass(const Logger& log = Logger::global());

std::unique_ptr<mlir::Pass> createQueryWSInfoPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createQueryWSInfoPass(std::optional<Byte> memLimit, const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createIntroduceInitFunctionPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createIntroduceInitFunctionPass(StringRef wsExtractionModeString,
                                                            std::optional<int64_t> initPart,
                                                            std::optional<Byte> memLimit,
                                                            const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createConcatInitInputsPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createConcatInitResultsPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createConcatInitResultsPass(StringRef wsExtractionModeString,
                                                        std::optional<int64_t> initPart, std::optional<Byte> memLimit,
                                                        const Logger& log = Logger::global());

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

    BoolOption dumpStrategyToLog{*this, "dump-strategy-to-log",
                                 llvm::cl::desc("Dump the multiclustering and tiling strategy to log info"),
                                 llvm::cl::init(false)};

    StrOption enableShaveDDRAccessOptimization{
            *this, "enable-shave-ddr-access-optimization",
            llvm::cl::desc("SHAVE DDR access optimization option (true, false or auto)"), llvm::cl::init("true")};
};

//
// Host compilation pipeline passes
//

std::unique_ptr<mlir::Pass> createScfComputeOpsOutliningPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFinalizeComputeFunctionBoundariesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDynamicToStaticKernelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertVPUOpsToUpstreamOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRestorePadAttrAfterSCFTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustBlockSizeForScfTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollSCFLoopPass(StringRef loopUnrollFactor = "",
                                                    bool enableCascadedUnrolling = true, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFullUnrollSCFLoopPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSCFLoopAnalysisAndDebugPass(Logger log = Logger::global());

//
// Registration
//

void registerVPUPipelines();
void registerPasses();

}  // namespace VPU
}  // namespace vpux
