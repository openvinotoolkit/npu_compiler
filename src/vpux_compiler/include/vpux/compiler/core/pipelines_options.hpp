//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>
#include <mlir/Transforms/Passes.h>
#include "vpux/compiler/core/public_options.hpp"
#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/developer_build_utils.hpp"
#include "vpux/utils/core/type/float16.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <memory>
#include <string>
#include <type_traits>

namespace vpux {

// Noise margin above FP16 lowest value to account for residual noise on disabled signal lines.
// Default mask threshold = std::numeric_limits<vpux::type::float16>::lowest() + this margin.
constexpr double SOFTMAX_MASK_DISABLED_NOISE_MARGIN = 500.0;

//
// BatchCompileOptionsAdapter
//

struct BatchCompileOptionsAdapter {
    BatchCompileOptionsAdapter(mlir::detail::PassOptions& parent);
    StrOption batchCompileMethod;
    StrOption debatchCompileMethodSettings;
    StrOption batchUnrollCompileMethodSettings;

    void updateBatchCompileOptionsFromString(std::string_view strOptions);
};

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
private:
    // Use composition instead of inheritance to avoid virtual inheritance complexity
    BatchCompileOptionsAdapter _batchCompileAdapter;

public:
    DefaultHWOptionsBase(): _batchCompileAdapter(static_cast<mlir::detail::PassOptions&>(*this)) {
    }

    // Accessors for batch compile options
    const BatchCompileOptionsAdapter& getBatchCompileAdapter() const {
        return _batchCompileAdapter;
    }
    BatchCompileOptionsAdapter& getBatchCompileAdapter() {
        return _batchCompileAdapter;
    }

    // Compiler infrastructure options applied before pipeline construction. Start.

    BoolOption enableVerifiers{*this, "enable-verifiers", llvm::cl::desc("Enable verifiers execution after each pass"),
                               llvm::cl::init(isDeveloperBuild())};

    BoolOption enableMemoryUsageCollector{*this, "enable-memory-usage-collector",
                                          llvm::cl::desc("Enable peak memory usage instrumentation after each pass"),
                                          llvm::cl::init(isDeveloperBuild())};
    StrOption compilerProfilerTool{*this, "compiler-profiler-tool",
                                   llvm::cl::desc("Select the compiler profiler tool to use. Example: "
                                                  "compiler-profiler-tool='mlir-profiler=path=./dump.json'"),
                                   llvm::cl::init("")};

    BoolOption enableDecomposeSDPA{*this, "enable-decompose-sdpa",
                                   llvm::cl::desc("Enable ngraph passes decomposing SDPA like ops"),
                                   llvm::cl::init(true)};

    BoolOption enableDummyOpReplacement{*this, "dummy-op-replacement",
                                        llvm::cl::desc("Replace unsupported SW Kernel ops with Dummy ones"),
                                        llvm::cl::init(false)};

    BoolOption enableProfiling{*this, "profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};

    BoolOption enablePipelinedCmdListRecording{
            *this, "enable-pipelined-cmd-list-recording",
            llvm::cl::desc("Enable pipelined command list recording and inference execution"),
            llvm::cl::init(vpux::HostExec::defaultEnablePipelinedCmdListRecording)};

    BoolOption enableDynamicDimAlignment{
            *this, "dynamic-dim-alignment",
            llvm::cl::desc(
                    "Align dynamic dimensions during tiling to make it easier to merge strided DMAs into batched DMAs"),
            llvm::cl::init(false)};

    BoolOption enableFunctionStatisticsInstrumentation{
            *this, "enable-function-statistics-instrumentation",
            llvm::cl::desc("Enable printing statistics for functions after each pass"), llvm::cl::init(false)};

    // Compiler infrastructure options applied before pipeline construction. End.

    StrOption functionOutlining{*this, "function-outlining",
                                llvm::cl::desc("Define a list of outlining modes and their parameters where the next "
                                               "outlining mode is the fallback mode of the previous one."
                                               "Example: function-outlining=' repeating-blocks=max-num-iterations=30 "
                                               "min-ops-in-block=16, naive=num-parts=2'")};

    BoolOption enableLoopOutliner{*this, "loop-outlining", llvm::cl::desc("Apply outlining for body of Loop op"),
                                  llvm::cl::init(false)};

    // This is a temporary option to enable running profiling passes along with outlining passes.
    // It will be removed after all profiling engines are updated to support outlined functions.
    // Ticket: E#159100
    BoolOption enableProfilingWithOutlining{*this, "profiling-with-outlining",
                                            llvm::cl::desc("Enable profiling with outlining"), llvm::cl::init(false)};

    BoolOption enableIntermediateBufferOutput{
            *this, "enable-intermediate-buffer-output",
            llvm::cl::desc("Enable intermediate output of defined operation buffer at specified insertion place"),
            llvm::cl::init(false)};

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

    BoolOption enableVFScheduleTrace{*this, "enable-vf-schedule-trace",
                                     llvm::cl::desc("Enable vertical fusion scheduling trace"), llvm::cl::init(false)};

    BoolOption enableSCFTiling{*this, "scf-tiling", llvm::cl::desc("Enable tiling using SCF dialect"),
                               llvm::cl::init(false)};

    BoolOption enableYuvToRgbShaveScale{*this, "yuv-to-rgb-shave-scale",
                                        llvm::cl::desc("Enable YUV to RGB SHAVE scale conversion"),
                                        llvm::cl::init(false)};

    BoolOption enableBoundedTensorsToDynamicDimsMask{*this, "bounded-tensors-to-dynamic-dims-mask",
                                                     llvm::cl::desc("Enable BoundedTensorsToDynamicDimsMask pass"),
                                                     llvm::cl::init(true)};

    BoolOption enableScfComputeOpsOutlining{*this, "scf-compute-ops-outlining",
                                            llvm::cl::desc("Outline SCF compute ops"), llvm::cl::init(false)};

    BoolOption enableWeightsExtraction{*this, "weights-extraction",
                                       llvm::cl::desc("Extract repeated weights and concat host constants in the "
                                                      "SCF compute-ops outlining pipeline"),
                                       llvm::cl::init(false)};

    BoolOption enableAsyncRegionOutlining{*this, "async-region-outlining",
                                          llvm::cl::desc("Enable async region outlining"), llvm::cl::init(false)};

    IntOption asyncRegionOutliningMinOpsInBlock{
            *this, "async-region-outlining-min-ops-in-block",
            llvm::cl::desc("Threshold for number of ops in each instance of async region outlining"),
            llvm::cl::init(100)};

    BoolOption enableEntireMainContentOutlining{
            *this, "outline-entire-main-content",
            llvm::cl::desc("Enable outlining of all non-call ops in the main function"), llvm::cl::init(true)};

    BoolOption enablePrintStatistics{*this, "enable-print-statistics", ::llvm::cl::desc("Enable print statistics"),
                                     ::llvm::cl::init(vpux::isDeveloperBuild())};

    BoolOption allowCustomValues{*this, "allow-custom-values",
                                 ::llvm::cl::desc("[Optional] Allows keep predefined values in IR")};

    // VPURT
    IntOption controlGraphSplitBlockSize{
            *this, "control-graph-split-block-size",
            llvm::cl::desc("Maximal number of tasks in each block that control graph will be split into. Used to "
                           "reduce memory consumption of barrier legalization pipeline for big models. Memory usage is "
                           "roughly (control-graph-split-block-size)^2/8"),
            llvm::cl::init(CONTROL_GRAPH_SPLIT_BLOCK_SIZE)};

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

    BoolOption enablePopulateWeightTableWithShave{*this, "enable-populate-weight-table-with-shave",
                                                  llvm::cl::desc("Enable populating weights table with Shave"),
                                                  llvm::cl::init(false)};

    BoolOption enableFP16CompressedConvolution{*this, "enable-fp16-compressed-convolution",
                                               llvm::cl::desc("Enable FP16 Compressed convolution op"),
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

    BoolOption enableCompressActivationSpill{*this, "compress-activation-spill",
                                             ::llvm::cl::desc("Enable compress-activation-spill feature"),
                                             ::llvm::cl::init(false)};

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

    BoolOption enableShaveCodeGenTiling{*this, "enable-shave-code-gen-tiling",
                                        llvm::cl::desc("Enable tiling of Shave Code Generation JIT kernels"),
                                        llvm::cl::init(false)};

    BoolOption enableFuseD2SExpand{*this, "enable-fuse-d2s-expand", llvm::cl::desc("Enable Fuse D2S Expand pass"),
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

    StrOption loopUnrollFactor{*this, "loop-unroll-factor",
                               llvm::cl::desc("List of unroll factors for SCF loops to unroll by."),
                               llvm::cl::init("1,1,1,1")};

    BoolOption enableCascadedUnrolling{
            *this, "enable-cascaded-unrolling",
            llvm::cl::desc("Enable cascaded unrolling with decreasing factors (e.g., 10 -> 5 -> 2)"),
            llvm::cl::init(true)};

    mlir::detail::PassOptions::Option<AutoUnrollingMode> autoUnrollingMode{
            *this, "auto-unrolling-mode",
            llvm::cl::desc("Auto-unrolling mode: disabled | inner | outer | all | biggest"),
            llvm::cl::init(AutoUnrollingMode::DISABLED),
            llvm::cl::values(clEnumValN(AutoUnrollingMode::DISABLED, "disabled", "Auto-unrolling off"),
                             clEnumValN(AutoUnrollingMode::INNER, "inner",
                                        "Unroll innermost loops; skip outer if inner already unrolled"),
                             clEnumValN(AutoUnrollingMode::OUTER, "outer",
                                        "Unroll outermost loops; skip inner if parent loop will be unrolled"),
                             clEnumValN(AutoUnrollingMode::ALL, "all", "Unroll all loops without anti-nesting guard"),
                             clEnumValN(AutoUnrollingMode::BIGGEST, "biggest",
                                        "Unroll only the loop with the largest auto-assigned factor"))};

    BoolOption enableBacktrackingBeyondResidualKernel{
            *this, "enable-backtracking-beyond-residual-kernel",
            llvm::cl::desc("Enable tail-overlap backtracking for cascaded loop unrolling (runtime-selected stage)"),
            llvm::cl::init(false)};

    Int64Option tailOverlapBacktrackMarginPercent{
            *this, "tail-overlap-backtrack-margin-percent",
            llvm::cl::desc(
                    "Allowed deficit for tail-overlap backtracking as a percentage (0..100) of one overlap tile step"),
            llvm::cl::init(15)};

    StrOption disabledPasses{*this, "disabled-passes",
                             llvm::cl::desc("Regex for disabling specific passes in the compiler pipeline"),
                             llvm::cl::init("")};

    BoolOption constantTracing{*this, "constant-tracing",
                               llvm::cl::desc("Flag for enabling constant tracing in the compiler pipeline"),
                               llvm::cl::init(false)};

    mlir::detail::PassOptions::Option<VFMergeConfiguration> vfMergeConfiguration{
            *this, "vf-merge-configuration", ::llvm::cl::desc("Option for configuring vertical fusion merge strategy."),
            ::llvm::cl::init(VFMergeConfiguration::COST_BASED),
            ::llvm::cl::values(
                    clEnumValN(VFMergeConfiguration::COST_BASED, "COST_BASED",
                               "Merge based on cost comparison between merged and non-merged cases"),
                    clEnumValN(VFMergeConfiguration::GREEDY, "GREEDY", "Merge greedily without considering cost"))};

    mlir::detail::PassOptions::Option<AllocateDDRStackFrames> allocateDDRStackFrames{
            *this, "allocate-ddr-stack-frames",
            ::llvm::cl::desc("Enable the computation and allocation of a new section which "
                             "will be used as stack frames for shaves."),
            ::llvm::cl::init(AllocateDDRStackFrames::DISABLED),
            ::llvm::cl::values(clEnumValN(AllocateDDRStackFrames::ENABLED, "ENABLED",
                                          "Allocate DDR buffer to be used as shave stack frames."),
                               clEnumValN(AllocateDDRStackFrames::DISABLED, "DISABLED", "Shave stack frames in CMX."))};

    BoolOption enableOutdatedPassDetection{*this, "enable-outdated-pass-detection",
                                           llvm::cl::desc("Enable detection of outdated passes"),
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

    BoolOption enableTilingFullSearchSpace{*this, "enable-tiling-full-search-space",
                                           llvm::cl::desc("Enable full search space for tiling"),
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

    BoolOption enableVFScheduleTrace{*this, "enable-vf-schedule-trace",
                                     llvm::cl::desc("Enable vertical fusion scheduling trace"), llvm::cl::init(false)};

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

    BoolOption enableBoundedTensorsToDynamicDimsMask{*this, "bounded-tensors-to-dynamic-dims-mask",
                                                     llvm::cl::desc("Enable BoundedTensorsToDynamicDimsMask pass"),
                                                     llvm::cl::init(true)};

    BoolOption enableDynamicDimAlignment{
            *this, "dynamic-dim-alignment",
            llvm::cl::desc(
                    "Align dynamic dimensions during tiling to make it easier to merge strided DMAs into batched DMAs"),
            llvm::cl::init(false)};

    BoolOption enableScfComputeOpsOutlining{*this, "scf-compute-ops-outlining",
                                            llvm::cl::desc("Outline SCF compute ops"), llvm::cl::init(false)};

    BoolOption enableWeightsExtraction{*this, "weights-extraction",
                                       llvm::cl::desc("Extract repeated weights and concat host constants in the "
                                                      "SCF compute-ops outlining pipeline (experimental)"),
                                       llvm::cl::init(false)};

    BoolOption enablePrintStatistics{*this, "enable-print-statistics", ::llvm::cl::desc("Enable print statistics"),
                                     ::llvm::cl::init(vpux::isDeveloperBuild())};

    StrOption enableShaveDDRAccessOptimization{
            *this, "enable-shave-ddr-access-optimization",
            llvm::cl::desc("SHAVE DDR access optimization option. (true, false or auto)"), llvm::cl::init("true")};

    BoolOption enableReorderConcatBranches{
            *this, "enable-reorder-concat-branches",
            llvm::cl::desc("Reorder branches of concat to make sure it is executed branch by branch"),
            llvm::cl::init(false)};

    BoolOption enableExplicitDistributionInfoAttr{
            *this, "enable-explicit-distributed-attr",
            llvm::cl::desc("Enable DistributionInfoAttr with explicit per cluster memory/compute shapes & offsets"),
            llvm::cl::init(false)};

    mlir::detail::PassOptions::Option<VFMergeConfiguration> vfMergeConfiguration{
            *this, "vf-merge-configuration", ::llvm::cl::desc("Option for configuring vertical fusion merge strategy."),
            ::llvm::cl::init(VFMergeConfiguration::COST_BASED),
            ::llvm::cl::values(
                    clEnumValN(VFMergeConfiguration::COST_BASED, "COST_BASED",
                               "Merge based on cost comparison between merged and non-merged cases"),
                    clEnumValN(VFMergeConfiguration::GREEDY, "GREEDY", "Merge greedily without considering cost"))};

    mlir::detail::PassOptions::Option<WorkloadManagementMode> workloadManagementMode{
            *this, "workload-management-mode",
            ::llvm::cl::desc("Option for enabling WLM enqueue barriers search algorithm at VPURT. To be used only for "
                             "experiments."),
            ::llvm::cl::init(WorkloadManagementMode::PWLM_V0_1_PAGES),
            ::llvm::cl::values(
                    clEnumValN(WorkloadManagementMode::PWLM_V0_1_PAGES, "PWLM_V0_1_PAGES",
                               "PWLM with split into subgraphs (pages)"),
                    clEnumValN(WorkloadManagementMode::PWLM_V0_1_PAGES, "PWLM_V0_LCA",
                               "This is a deprecated WLM mode which is no longer supported. The option is kept for "
                               "backwards compatibility only and the mode redirects to PWLM_V0_1_PAGES mode, which has "
                               "the same vpu-fw compatibility but offers numerous stability improvements."))};

    StrOption loopUnrollFactor{*this, "loop-unroll-factor",
                               llvm::cl::desc("List of unroll factors for SCF loops to unroll by."),
                               llvm::cl::init("1,1,1,1")};

    BoolOption enableCascadedUnrolling{
            *this, "enable-cascaded-unrolling",
            llvm::cl::desc("Enable cascaded unrolling with decreasing factors (e.g., 10 -> 5 -> 2)"),
            llvm::cl::init(true)};

    mlir::detail::PassOptions::Option<AutoUnrollingMode> autoUnrollingMode{
            *this, "auto-unrolling-mode",
            llvm::cl::desc("Auto-unrolling mode: disabled | inner | outer | all | biggest"),
            llvm::cl::init(AutoUnrollingMode::DISABLED),
            llvm::cl::values(clEnumValN(AutoUnrollingMode::DISABLED, "disabled", "Auto-unrolling off"),
                             clEnumValN(AutoUnrollingMode::INNER, "inner",
                                        "Unroll innermost loops; skip outer if inner already unrolled"),
                             clEnumValN(AutoUnrollingMode::OUTER, "outer",
                                        "Unroll outermost loops; skip inner if parent loop will be unrolled"),
                             clEnumValN(AutoUnrollingMode::ALL, "all", "Unroll all loops without anti-nesting guard"),
                             clEnumValN(AutoUnrollingMode::BIGGEST, "biggest",
                                        "Unroll only the loop with the largest auto-assigned factor"))};

    BoolOption enableBacktrackingBeyondResidualKernel{
            *this, "enable-backtracking-beyond-residual-kernel",
            llvm::cl::desc("Enable tail-overlap backtracking for cascaded loop unrolling (runtime-selected stage)"),
            llvm::cl::init(false)};

    Int64Option tailOverlapBacktrackMarginPercent{
            *this, "tail-overlap-backtrack-margin-percent",
            llvm::cl::desc(
                    "Allowed deficit for tail-overlap backtracking as a percentage (0..100) of one overlap tile step"),
            llvm::cl::init(15)};

    BoolOption enableRunMVNNormalizeOnDPU{*this, "enable-run-mvn-normalize-on-dpu",
                                          llvm::cl::desc("Enable RunMVNNormalizeOnDPU pass on DPU"),
                                          llvm::cl::init(false)};

    BoolOption enableSoftmaxDecomposition{*this, "enable-softmax-decomposition",
                                          llvm::cl::desc("Enable Softmax decomposition pass"), llvm::cl::init(true)};

    MCAndTilingOptionsBase() = default;

    template <class OtherOptions>
    explicit MCAndTilingOptionsBase(const OtherOptions& options) {
        this->matchAndCopyOptionValuesFrom(options);
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

    mlir::detail::PassOptions::Option<WorkloadManagementMode> workloadManagementMode{
            *this, "workload-management-mode",
            ::llvm::cl::desc("Option for enabling WLM enqueue barriers search algorithm at VPURT. To be used only for "
                             "experiments."),
            ::llvm::cl::init(WorkloadManagementMode::PWLM_V0_1_PAGES),
            ::llvm::cl::values(
                    clEnumValN(WorkloadManagementMode::FWLM_V1_PAGES, "FWLM_V1_PAGES",
                               "Full WLM with split into pages"),
                    clEnumValN(WorkloadManagementMode::PWLM_V0_1_PAGES, "PWLM_V0_1_PAGES",
                               "PWLM with split into subgraphs (pages)"),
                    clEnumValN(WorkloadManagementMode::PWLM_V0_1_PAGES, "PWLM_V0_LCA",
                               "This is a deprecated WLM mode which is no longer supported. The option is kept for "
                               "backwards compatibility only and the mode redirects to PWLM_V0_1_PAGES mode, which has "
                               "the same vpu-fw compatibility but offers numerous stability improvements."))};

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

    mlir::detail::PassOptions::Option<AllocateDDRStackFrames> allocateDDRStackFrames{
            *this, "allocate-ddr-stack-frames",
            ::llvm::cl::desc("Enable the computation and allocation of a new section which "
                             "will be used as stack frames for shaves."),
            ::llvm::cl::init(AllocateDDRStackFrames::DISABLED),
            ::llvm::cl::values(clEnumValN(AllocateDDRStackFrames::ENABLED, "ENABLED",
                                          "Allocate DDR buffer to be used as shave stack frames."),
                               clEnumValN(AllocateDDRStackFrames::DISABLED, "DISABLED", "Shave stack frames in CMX."))};

    mlir::detail::PassOptions::Option<WorkloadManagementBarrierProgrammingMode>
            workloadManagementBarrierProgrammingMode{
                    *this, "workload-management-barrier-programming-mode",
                    ::llvm::cl::desc(
                            "Option for enabling different barrier programming algorithms. To be used only for "
                            "experiments."),
                    ::llvm::cl::values(
                            clEnumValN(WorkloadManagementBarrierProgrammingMode::LEGACY, "LEGACY", "Legacy Mode"),
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

struct BatchCompilerOptionsAdapterView {
    static std::optional<BatchCompilerOptionsAdapterView> tryExtractFromString(std::string_view strOptions);
    std::string injectInto(const std::string& originalStrOptions) const;
    const BatchCompileOptionsAdapter& get() const;
    std::string print() const;

private:
    BatchCompilerOptionsAdapterView() = default;
    using ScopeGuard = mlir::detail::PassOptions;
    std::unique_ptr<ScopeGuard> guard;
    std::unique_ptr<BatchCompileOptionsAdapter> optionDataPtr;
};

namespace detail {

template <typename T, typename = void>
struct has_getBatchCompileAdapter : std::false_type {};

template <typename T>
struct has_getBatchCompileAdapter<T, std::void_t<decltype(std::declval<T>().getBatchCompileAdapter())>> :
        std::true_type {};

}  // namespace detail

struct DebatcherOptions : mlir::PassPipelineOptions<DebatcherOptions> {
    StrOption debatcherInliningMethod;
    StrOption debatcherInputCoeffPartitions;
    IntOption modelOpsNumberEnableThreshold;
    IntOption maxBatchNumberDisableLimit;
    DebatcherOptions();

    static std::unique_ptr<DebatcherOptions> create(const BatchCompileOptionsAdapter& options);

    template <class Options>
    static std::unique_ptr<DebatcherOptions> create(const Options& options) {
        if constexpr (detail::has_getBatchCompileAdapter<Options>::value) {
            return create(options.getBatchCompileAdapter());
        } else {
            return nullptr;
        }
    }

    static bool isAvailable(const BatchCompileOptionsAdapter& options);

    template <class Options>
    static bool isAvailable(const Options& options) {
        if constexpr (detail::has_getBatchCompileAdapter<Options>::value) {
            return isAvailable(options.getBatchCompileAdapter());
        } else {
            return false;
        }
    }

    static bool isExplicitlySpecified(const BatchCompileOptionsAdapter& options);
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

    template <class Options>
    static std::unique_ptr<BatchUnrollOptions> create(const Options& options, Logger log = Logger::global()) {
        if constexpr (detail::has_getBatchCompileAdapter<Options>::value) {
            return create(options.getBatchCompileAdapter(), log);
        } else {
            return nullptr;
        }
    }

    static bool isAvailable(const BatchCompileOptionsAdapter& options);

    template <class Options>
    static bool isAvailable(const Options& options) {
        if constexpr (detail::has_getBatchCompileAdapter<Options>::value) {
            return isAvailable(options.getBatchCompileAdapter());
        } else {
            return false;
        }
    }

    static bool isExplicitlySpecified(const BatchCompileOptionsAdapter& options);
    static std::string getDefaultOptions();
};

template <class OptionsType>
bool canOutlineFromProfilingPerspective(const OptionsType& options) {
    // TODO: E#140041 enable profiling unconditionally
    return !options.enableProfiling || options.enableProfilingWithOutlining;
}

template <class OptionsType>
bool isOutliningEnabled(const OptionsType& options) {
    bool isAnyTypeOfOutliningEnabled = options.functionOutlining.hasValue() || DebatcherOptions::isAvailable(options) ||
                                       options.enableVerticalFusionOutlining ||
                                       options.enableConcatRepeatingBlockOutlining ||
                                       options.enableEntireMainContentOutlining || options.enableAsyncRegionOutlining;
    return isAnyTypeOfOutliningEnabled && canOutlineFromProfilingPerspective(options);
}
}  // namespace vpux
