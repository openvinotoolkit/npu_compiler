//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/developer_build_utils.hpp"
#include "vpux/compiler/core/public_options.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <string>

namespace vpux {
//
// DefaultHWOptionsBase
// This class must be inherited by all dialect-base options
// to avoid confusion when we have the same option for IE and the VPU dialect, but with a different value
//

struct BatchUnrollOptions;
struct DebatcherOptions;

template <class Suboption>
std::string getDefaultValueOfStrSubOption() {
    return Suboption::getDefaultOptions();
}

struct DefaultHWOptionsBase : mlir::PassPipelineOptions<DefaultHWOptionsBase>, public virtual PublicOptions {
    BoolOption enableDPUF16ToF32Convert{*this, "enable-dpu-f16-to-f32-convert",
                                        llvm::cl::desc("Enable running F16 -> F32 converts on DPU."),
                                        llvm::cl::init(true)};

    BoolOption enableVerifiers{*this, "enable-verifiers", llvm::cl::desc("Enable verifiers execution after each pass"),
                               llvm::cl::init(isDeveloperBuild())};

    BoolOption enableMemoryUsageCollector{*this, "enable-memory-usage-collector",
                                          llvm::cl::desc("Enable peak memory usage instrumentation after each pass"),
                                          llvm::cl::init(isDeveloperBuild())};

    BoolOption enableFunctionStatisticsInstrumentation{
            *this, "enable-function-statistics-instrumentation",
            llvm::cl::desc("Enable printing statistics for functions after each pass"), llvm::cl::init(false)};

    StrOption functionOutlining{*this, "function-outlining",
                                llvm::cl::desc("Define a list of outlining modes and their parameters where the next "
                                               "outlining mode is the fallback mode of the previous one."
                                               "Example: function-outlining=' repeating-blocks=max-num-iterations=30 "
                                               "min-ops-in-block=16, naive=num-parts=2'")};

    BoolOption enableLoopOutliner{*this, "loop-outlining", llvm::cl::desc("Apply outlining for body of Loop op"),
                                  llvm::cl::init(false)};

    BoolOption enableDummyOpReplacement{*this, "dummy-op-replacement",
                                        llvm::cl::desc("Replace unsupported SW Kernel ops with Dummy ones"),
                                        llvm::cl::init(false)};

    BoolOption constantFoldingInBackground{*this, "constant-folding-in-background",
                                           llvm::cl::desc("Fold constants in background threads"),
                                           llvm::cl::init(false)};

    IntOption constantFoldingInBackgroundNumThreads{
            *this, "constant-folding-in-background-num-threads",
            llvm::cl::desc("Number of background threads to use for constant folding in background. Ignored if "
                           "`constant-folding-in-background` is disabled."),
            llvm::cl::init(1)};

    BoolOption constantFoldingInBackgroundCollectStatistics{
            *this, "constant-folding-in-background-collect-statistics",
            llvm::cl::desc("Toggle for the collection of statistics when folding constants in background. Ignored if "
                           "`constant-folding-in-background` is disabled."),
            llvm::cl::init(false)};

    IntOption constantFoldingInBackgroundMemoryUsageLimit{
            *this, "constant-folding-in-background-memory-usage-limit",
            llvm::cl::desc("Fold constants in background memory usage limit (in MB)"), llvm::cl::init(3 * 1024)};

    DoubleOption constantFoldingInBackgroundCacheCleanThreshold{
            *this, "constant-folding-in-background-cache-clean-threshold",
            llvm::cl::desc("Cache will be cleaned to this threshold when reach the memory usage limit"),
            llvm::cl::init(0.8)};

    BoolOption enableProfiling{*this, "profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};

    // This is a temporary option to enable running profiling passes along with outlining passes.
    // It will be removed after all profiling engines are updated to support outlined functions.
    // Ticket: E#159100
    BoolOption enableProfilingWithOutlining{*this, "profiling-with-outlining",
                                            llvm::cl::desc("Enable profiling with outlining"), llvm::cl::init(false)};

    BoolOption enableIntermediateBufferOutput{
            *this, "enable-intermediate-buffer-output",
            llvm::cl::desc("Enable intermediate output of defined operation buffer at specified insertion place"),
            llvm::cl::init(false)};

    BoolOption enableActivityFactor{*this, "enable-activity-factor",
                                    llvm::cl::desc("Enable activity factor and inference time estimation"),
                                    llvm::cl::init(true)};

    StrOption scheduleTraceFile{*this, "schedule-trace-file-name",
                                llvm::cl::desc("Compile time schedule JSON trace file name"),
                                llvm::cl::init("compileTimeScheduleTrace.json")};

    BoolOption enablePrefetching{*this, "prefetching",
                                 llvm::cl::desc("Enable prefetch tiling pass and prefetch scheduling"),
                                 llvm::cl::init(true)};

    BoolOption enablePipelining{*this, "pipelining",
                                llvm::cl::desc("Enable vertical fusion pipelining pass and schedule pipelining"),
                                llvm::cl::init(true)};
    StrOption mcOptimizationScope{
            *this, "mc-optimization-scope", llvm::cl::desc("Determine multi-clustering optimization scope"),
            llvm::cl::desc("Multi-cluster strategy optimization scope (local, subgraph)"), llvm::cl::init("subgraph")};

    IntOption concatRepeatingBlockOutliningSeqLength{
            *this, "concat-repeating-block-outlining-min-seq-length",
            llvm::cl::desc("Threshold for length of concat input sequence for repeating blocks outlining"),
            llvm::cl::init(5)};

    BoolOption enableConcatRepeatingBlockOutlining{*this, "concat-repeating-block-outlining",
                                                   llvm::cl::desc("Enable concat input as repeating blocks outlining"),
                                                   llvm::cl::init(true)};
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

    BoolOption enableSCFTiling{*this, "scf-tiling", llvm::cl::desc("Enable tiling using SCF dialect"),
                               llvm::cl::init(false)};

    BoolOption enableScfComputeOpsOutlining{*this, "scf-compute-ops-outlining",
                                            llvm::cl::desc("Outline SCF compute ops"), llvm::cl::init(false)};

    BoolOption enableAsyncRegionOutlining{*this, "async-region-outlining",
                                          llvm::cl::desc("Enable async region outlining"), llvm::cl::init(false)};

    IntOption asyncRegionOutliningMinOpsInBlock{
            *this, "async-region-outlining-min-ops-in-block",
            llvm::cl::desc("Threshold for number of ops in each instance of async region outlining"),
            llvm::cl::init(100)};

    BoolOption enableEntireMainContentOutlining{
            *this, "outline-entire-main-content",
            llvm::cl::desc("Enable outlining of all non-call ops in the main function"), llvm::cl::init(true)};

    BoolOption fuseScalesToAccumulate{
            *this, "fuse-scales-to-accumulate",
            llvm::cl::desc("Enable scales fusing to following Accumulate op from GPTQ Matmul unrolling"),
            llvm::cl::init(false)};

    BoolOption allowCustomValues{*this, "allow-custom-values",
                                 ::llvm::cl::desc("[Optional] Allows keep predefined values in IR")};

    // VPURT
    BoolOption enableControlGraphSplit{*this, "enable-control-graph-split",
                                       llvm::cl::desc("Enable split of control graph to simplify barrier scheduling"),
                                       llvm::cl::init(true)};
    IntOption controlGraphSplitBlockSize{
            *this, "control-graph-split-block-size",
            llvm::cl::desc("Maximal number of tasks in each block that control graph will be split into. Used to "
                           "reduce memory consumption of barrier legalization pipeline for big models. Memory usage is "
                           "roughly (control-graph-split-block-size)^2/8"),
            llvm::cl::init(CONTROL_GRAPH_SPLIT_BLOCK_SIZE)};

    BoolOption enableSimpleSchedule{*this, "simple-schedule", llvm::cl::desc("Enable schedule simplification"),
                                    llvm::cl::init(true)};

    BoolOption reduceParallelControlFlows{*this, "reduce-parallel-control-flows",
                                          llvm::cl::desc("Reduce parallel overlapping control flows where possible"),
                                          llvm::cl::init(true)};

    Int64Option constantFoldingSizeThreshold{
            *this, "constant-folding-threshold",
            llvm::cl::desc("Fold constants in single threading if the size is larger than the threshold."),
            llvm::cl::init(300 * 1024 * 1024)};  // 300MB

    // Option to control locations verification. Possible options are:
    // off - no verification
    // fast - verifies locations only after last* pass by thorough algorithm
    // full - verifies locations after each pass using fast algorithm and thorough after last* pass
    // thorough - does thorough verification after each** pass
    // This feature is in process of development, details: #E81319
    // *last in sequence of fixed passes. Some passes is not fixed yet, so may break compilation because of locations
    // reuse
    // **each of fixed passes
    StrOption locationsVerificationMode{
            *this, "verify-locations",
            llvm::cl::desc("Selects location verification mode. Possible options are off/fast/full/thorough"),
            llvm::cl::init(vpux::isDeveloperBuild() ? "fast" : "off")};

    BoolOption enableDmaOutOfOrder{*this, "dma-ooo", llvm::cl::desc("Enable out-of-order DMA"), llvm::cl::init(true)};

    BoolOption enableColorBinPhysicalBarrierAssignment{
            *this, "enable-color-bin-physical-barrier-assignment",
            llvm::cl::desc("Enable physical barrier assignment optimization"), llvm::cl::init(false)};

    BoolOption enablePopulateWeightTableWithShave{*this, "enable-populate-weight-table-with-shave",
                                                  llvm::cl::desc("Enable populating weights table with Shave"),
                                                  llvm::cl::init(false)};

    BoolOption enableFP16CompressedConvolution{*this, "enable-fp16-compressed-convolution",
                                               llvm::cl::desc("Enable FP16 Compressed convolution op"),
                                               llvm::cl::init(false)};

    BoolOption enableWeightsDynamicDequantization{*this, "enable-weights-dynamic-dequantization",
                                                  llvm::cl::desc("Enable weights dequantization for weights as input"),
                                                  llvm::cl::init(false)};
    BoolOption enableInPlaceBufferization{
            *this, "enable-in-place-bufferization",
            llvm::cl::desc("Enable in-place bufferization. Might eliminate some redundant buffer allocations at the "
                           "cost of longer compile time"),
            llvm::cl::init(false)};

    BoolOption enableExtraStaticShapeOps{
            *this, "enable-extra-static-shape-ops",
            llvm::cl::desc("Attach StaticShapeOpInterface trait to operations that perform "
                           "faster when their shapes are static."),
            llvm::cl::init(true)};

    BoolOption useMemrefForHostFunctionBufferization{
            *this, "use-memref-for-host-function-bufferization",
            llvm::cl::desc("Enable memref bufferization for host function ops"), llvm::cl::init(false)};

    BoolOption enableMergeFakeQuant{*this, "merge-fake-quant", llvm::cl::desc("Enable merge-fake-quant pass"),
                                    llvm::cl::init(true)};

    BoolOption enableCompressActivationSpill{*this, "compress-activation-spill",
                                             ::llvm::cl::desc("Enable compress-activation-spill feature"),
                                             ::llvm::cl::init(false)};

    BoolOption enableSWKernelInstructionPrefetch{*this, "enable-sw-kernels-instruction-prefetch",
                                                 llvm::cl::desc("Enable SW kernels instruction prefetch"),
                                                 llvm::cl::init(true)};

    // TODO: Ideally, some of the passes in the HostCompile pipeline should not operate on the main function.
    // The following option will skip processing of the main function in these passes. This will be refactored in the
    // future. Track: E#168311
    BoolOption disablePassOnEntryFunctionForHostCompile{
            *this, "disable-pass-on-entry-function",
            llvm::cl::desc("Disable certain passes for entry function operations for HostCompile pipeline"),
            llvm::cl::init(false)};

    BoolOption enableShaveCodeGen{*this, "enable-shave-code-gen",
                                  llvm::cl::desc("Enable Shave Code Generation - JIT kernels compilation"),
                                  llvm::cl::init(false)};
};

//
// MCAndTilingOptionsBase options
//

struct MCAndTilingOptionsBase : mlir::PassPipelineOptions<MCAndTilingOptionsBase> {
    BoolOption enablePrefetching{*this, "prefetching", llvm::cl::desc("Enable prefetch mode"), llvm::cl::init(true)};

    BoolOption enableVerticalFusion{*this, "vertical-fusion", llvm::cl::desc("Enable vertical fusion feature"),
                                    llvm::cl::init(false)};

    BoolOption enablePipelining{*this, "pipelining", llvm::cl::desc("Enable vertical fusion pipelining"),
                                llvm::cl::init(false)};

    IntOption opTilingCacheThreshold{
            *this, "op-tiling-cache-threshold",
            llvm::cl::desc("threshold for number of clustered ops for tiling cache optimization"),
            llvm::cl::init(CLUSTERED_OP_THRESHOLD_FOR_TILING_CACHE)};
    StrOption mcOptimizationScope{
            *this, "mc-optimization-scope", llvm::cl::desc("Determine multi-clustering optimization scope"),
            llvm::cl::desc("Multi-cluster strategy optimization scope (local, subgraph)"), llvm::cl::init("subgraph")};

    IntOption vfOutliningInstanceThreshold{
            *this, "vf-outlining-instance-threshold",
            llvm::cl::desc("Threshold for number of instances (slices of the graph) to perform outlining"),
            llvm::cl::init(5)};

    IntOption vfOutliningTileThreshold{
            *this, "vf-outlining-tile-threshold",
            llvm::cl::desc("Threshold for outlining vertical fusion regions with accumulated number of tiles"),
            llvm::cl::init(10)};

    BoolOption enableVerticalFusionOutlining{*this, "vf-outlining", llvm::cl::desc("Enable vertical fusion outlining"),
                                             llvm::cl::init(false)};

    BoolOption enableProfiling{*this, "profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};

    // This is a temporary option to enable running profiling passes along with outlining passes.
    // It will be removed after all profiling engines are updated to support outlined functions.
    // Ticket: E#159100
    BoolOption enableProfilingWithOutlining{*this, "profiling-with-outlining",
                                            llvm::cl::desc("Enable profiling with outlining"), llvm::cl::init(false)};

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

    BoolOption enableVPUNNCostForTiling{*this, "enable-vpunn-cost-for-tiling",
                                        llvm::cl::desc("Use VPUNN cost model to get the best tiling strategy"),
                                        llvm::cl::init(false)};

    BoolOption enableOutputPipelining{*this, "output-pipelining", llvm::cl::desc("Enable output pipelining"),
                                      llvm::cl::init(false)};

    BoolOption enableSCFTiling{*this, "scf-tiling", llvm::cl::desc("Enable tiling using SCF dialect"),
                               llvm::cl::init(false)};

    BoolOption enableScfComputeOpsOutlining{*this, "scf-compute-ops-outlining",
                                            llvm::cl::desc("Outline SCF compute ops"), llvm::cl::init(false)};

    StrOption enableShaveDDRAccessOptimization{
            *this, "enable-shave-ddr-access-optimization",
            llvm::cl::desc("SHAVE DDR access optimization option. (true, false or auto)"), llvm::cl::init("true")};

    BoolOption enableExplicitDistributionInfoAttr{
            *this, "enable-explicit-distributed-attr",
            llvm::cl::desc("Enable DistributionInfoAttr with explicit per cluster memory/compute shapes & offsets"),
            llvm::cl::init(false)};

    mlir::detail::PassOptions::Option<WorkloadManagementMode> workloadManagementMode{
            *this, "workload-management-mode",
            ::llvm::cl::desc("Option for enabling WLM enqueue barriers search algorithm at VPURT. To be used only for "
                             "experiments."),
            ::llvm::cl::init(WorkloadManagementMode::PWLM_V0_LCA),
            ::llvm::cl::values(clEnumValN(WorkloadManagementMode::PWLM_V2_PAGES, "PWLM_V2_PAGES",
                                          "WLM with split into subgraphs (pages)"),
                               clEnumValN(WorkloadManagementMode::PWLM_V1_BARRIER_FIFO, "PWLM_V1_BARRIER_FIFO",
                                          "WLM enqueue barriers search algorithm at VPURT ENABLED"),
                               clEnumValN(WorkloadManagementMode::PWLM_V0_LCA, "PWLM_V0_LCA",
                                          "WLM enqueue barriers search algorithm at VPURT DISABLED. Use LCA based "
                                          "enqueue algorithm at VPUMI"))};

    MCAndTilingOptionsBase() = default;

    template <class OtherOptions>
    explicit MCAndTilingOptionsBase(const OtherOptions& options) {
        enablePrefetching = options.enablePrefetching;
        enableVerticalFusion = options.enableVerticalFusion;
        enablePipelining = options.enablePipelining;
        mcOptimizationScope = options.mcOptimizationScope;
        enableVPUNNCostForTiling = options.enableVPUNNCostForTiling;
        opTilingCacheThreshold = options.opTilingCacheThreshold;
        vfOutliningInstanceThreshold = options.vfOutliningInstanceThreshold;
        vfOutliningTileThreshold = options.vfOutliningTileThreshold;
        enableVerticalFusionOutlining = options.enableVerticalFusionOutlining;
        enableProfiling = options.enableProfiling;
        enableProfilingWithOutlining = options.enableProfilingWithOutlining;
        enableOutputPipelining = options.enableOutputPipelining;
        enableShaveDDRAccessOptimization = options.enableShaveDDRAccessOptimization;
        readStrategyFromJson = options.readStrategyFromJson;
        writeStrategyToJson = options.writeStrategyToJson;
        dumpStrategyToLog = options.dumpStrategyToLog;
        enableExplicitDistributionInfoAttr = options.enableExplicitDistributionInfoAttr;
        workloadManagementMode = options.workloadManagementMode;
        enableSCFTiling = options.enableSCFTiling;
        enableScfComputeOpsOutlining = options.enableSCFTiling;
    }
};

template <typename T>
struct BackendCompilationOptionsBase : mlir::PassPipelineOptions<T> {
    BoolOption enableMemorySideCache{*this, "enable-memory-side-cache", llvm::cl::desc("Enable memory side cache"),
                                     llvm::cl::init(false)};
    BoolOption workloadManagementEnable{*this, "workload-management-enable",
                                        llvm::cl::desc("Enable partial workload management"), llvm::cl::init(true)};
    StrOption enableDMAProfiling{*this, "dma-profiling",
                                 llvm::cl::desc("Enable DMA task profiling (true|static|false)"),
                                 llvm::cl::init("false")};

    IntOption workloadManagementBarrierCountThreshold{*this, "workload-management-barrier-count-threshold",
                                                      llvm::cl::desc("Threshold for WLM optimization"),
                                                      llvm::cl::init(VIRTUAL_BARRIER_THRESHOLD_WLM)};

    mlir::detail::PassOptions::Option<WorkloadManagementMode> workloadManagementMode{
            *this, "workload-management-mode",
            ::llvm::cl::desc("Option for enabling WLM enqueue barriers search algorithm at VPURT. To be used only for "
                             "experiments."),
            ::llvm::cl::init(WorkloadManagementMode::PWLM_V0_LCA),
            ::llvm::cl::values(
                    clEnumValN(WorkloadManagementMode::PWLM_V2_PAGES, "PWLM_V2_PAGES",
                               "Partial WLM with split into pages"),
                    clEnumValN(WorkloadManagementMode::PWLM_V1_BARRIER_FIFO, "PWLM_V1_BARRIER_FIFO",
                               "Partial WLM, enqueue barriers search algorithm at VPURT ENABLED"),
                    clEnumValN(WorkloadManagementMode::PWLM_V0_LCA, "PWLM_V0_LCA",
                               "Partial WLM, enqueue barriers search algorithm at VPURT DISABLED. Use LCA based "
                               "enqueue algorithm at VPUMI"))};

    mlir::detail::PassOptions::Option<DMAFifoType> workloadManagementDmaFifoType{
            *this, "workload-management-dma-fifo-type",
            ::llvm::cl::desc("Option to switch behaviour between software and hardware DMA FIFO types"),
            ::llvm::cl::init(DMAFifoType::SW),
            ::llvm::cl::values(clEnumValN(DMAFifoType::SW, "SW", "Enable SW DMA FIFO upfront HW DMA FIFO type"),
                               clEnumValN(DMAFifoType::HW, "HW", "Use HW DMA FIFO directly"))};

    StrOption enableShaveDDRAccessOptimization{
            *this, "enable-shave-ddr-access-optimization",
            llvm::cl::desc("SHAVE DDR access optimization option (true, false or auto)"), llvm::cl::init("true")};

    BoolOption enableDumpStatisticsOfWlmOps{*this, "enable-dump-wlm-ops-stats",
                                            llvm::cl::desc("Enable dump of WLM ops statistics"), llvm::cl::init(false)};

    mlir::detail::PassOptions::Option<AllocateShaveStackFrames> allocateShaveStackFrames{
            *this, "allocate-shave-stack-frames",
            ::llvm::cl::desc("Enable the computation and allocation of a new section which "
                             "will be used as stack frame form shaves."),
            ::llvm::cl::init(AllocateShaveStackFrames::DISABLED),
            ::llvm::cl::values(
                    clEnumValN(AllocateShaveStackFrames::ENABLED, "ENABLED",
                               "Allocate DDR buffer to be used as shave stack frames."),
                    clEnumValN(AllocateShaveStackFrames::DISABLED, "DISABLED", "Stack frames allocated by FW."))};

    mlir::detail::PassOptions::Option<WorkloadManagementBarrierProgrammingMode>
            workloadManagementBarrierProgrammingMode{
                    *this, "workload-management-barrier-programming-mode",
                    ::llvm::cl::desc(
                            "Option for enabling different barrier programming algorithms. To be used only for "
                            "experiments."),
                    ::llvm::cl::values(
                            clEnumValN(WorkloadManagementBarrierProgrammingMode::LEGACY, "LEGACY", "Legacy Mode"),
                            clEnumValN(WorkloadManagementBarrierProgrammingMode::NO_BARRIER_DMAS_SCHEDULED,
                                       "NO_BARRIER_DMAS_SCHEDULED", "RT handles barrier programming"),
                            clEnumValN(WorkloadManagementBarrierProgrammingMode::INITIAL_BARRIER_DMAS_SCHEDULED,
                                       "INITIAL_BARRIER_DMAS_SCHEDULED",
                                       "Compiler generates DMA to program initial barriers"),
                            clEnumValN(WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED,
                                       "ALL_BARRIER_DMAS_SCHEDULED",
                                       "Compiler generates DMAs to program all barriers"))};

    IntOption modelIdentifier{
            *this, "model-identifier",
            llvm::cl::desc("Unique identifier for the compiled model, used for debugging and troubleshooting"),
            llvm::cl::init(0)};

    Int64Option workspaceAddr{*this, "cmx-workspace-addr", llvm::cl::desc("CMX workspace start address."),
                              llvm::cl::init(0)};
    Int64Option workspaceSize{*this, "cmx-workspace-size", llvm::cl::desc("CMX workspace size."), llvm::cl::init(0)};
    Int64Option metadataAddr{*this, "cmx-metadata-addr", llvm::cl::desc("CMX metadata start address."),
                             llvm::cl::init(0)};
    Int64Option metadataSize{*this, "cmx-metadata-size", llvm::cl::desc("CMX metadata size."), llvm::cl::init(0)};
};

struct BatchCompileOptionsAdapter {
    BatchCompileOptionsAdapter(mlir::detail::PassOptions& parent);
    StrOption batchCompileMethod;
    StrOption debatchCompileMethodSettings;
    StrOption batchUnrollCompileMethodSettings;

    void updateBatchCompileOptionsFromString(std::string_view strOptions);
};

struct BatchCompilerOptionsAdapterView {
    static std::optional<BatchCompilerOptionsAdapterView> tryExtractFromString(std::string_view strOptions);
    std::string inject(const std::string& originalStrOptions) const;
    const BatchCompileOptionsAdapter& get() const;
    std::string print() const;
    struct Occurence {
        std::string::size_type pos;
        std::string::size_type length;

        friend bool operator<(const Occurence& l, const Occurence& r) {
            // sort by pos only
            return l.pos < r.pos;
        }
    };

private:
    BatchCompilerOptionsAdapterView() = default;
    using ScopeGuard = mlir::detail::PassOptions;
    std::unique_ptr<ScopeGuard> guard;
    std::unique_ptr<BatchCompileOptionsAdapter> optionDataPtr;
    std::vector<std::optional<Occurence>> optionDataMemberViews;
};

struct DebatcherOptions : mlir::PassPipelineOptions<DebatcherOptions> {
    StrOption debatcherInliningMethod;
    StrOption debatcherIntputCoeffPartitions;
    IntOption modelOpsNumberEnableThreshold;
    IntOption maxBatchNumberDisableLimit;
    DebatcherOptions();

    static std::unique_ptr<DebatcherOptions> create(const BatchCompileOptionsAdapter& options);
    static bool isAvailable(const BatchCompileOptionsAdapter& options);
    static std::string getDefaultOptions();
    static std::string getDefaultDebatchInputCoeffPartitionsValue();
    std::string to_string() const;
};

struct BatchUnrollOptions : mlir::PassPipelineOptions<BatchUnrollOptions> {
    BoolOption skipUnrollBatch{*this, "skip-unroll-batch", llvm::cl::desc("Skip unroll on batch dimension"),
                               llvm::cl::init(false)};
    BatchUnrollOptions() = default;

    static std::unique_ptr<BatchUnrollOptions> create(const BatchCompileOptionsAdapter& options,
                                                      Logger log = Logger::global());
    static bool isAvailable(const BatchCompileOptionsAdapter& options);
    static std::string getDefaultOptions();
};

}  // namespace vpux
