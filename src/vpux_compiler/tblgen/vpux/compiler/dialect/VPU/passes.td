//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_PASSES
#define VPUX_COMPILER_DIALECT_VPU_PASSES

include "mlir/Pass/PassBase.td"

//
// InitResources
//

def InitResources : PassBase<"init-resources", "vpux::ModulePass"> {
    let summary = "Initializes compiler for VPU platforms";
    let description = [{
        This pass attaches VPU related compilation parameters to Module attributes and
        initializes run-time resources information.
    }];

    let constructor = "vpux::VPU::createInitResourcesPass()";

    let options = [
        Option<
            "archOpt", "vpu-arch",
            "std::string", "",
            "VPU architecture to compile for"
        >,
        Option<
            "compilationModeOpt", "compilation-mode",
            "std::string", [{"DefaultHW"}],
            "[Optional] Set compilation mode as `ReferenceSW` or `DefaultHW`"
        >,
        Option<
            "revisionIDOpt", "revision-id",
            "int", "",
            "[Optional] Revision ID of the platform"
        >,
        Option<
            "numberOfDPUGroupsOpt", "num-of-dpu-groups",
            "int", "",
            "[Optional] Number of available DPU groups"
        >,
        Option<
            "numberOfDMAPortsOpt", "num-of-dma-ports",
            "int", "",
            "[Optional] Number of available DMA ports"
        >,
        Option<
            "availableCMXMemoryOpt", "available-cmx-memory",
            "int", "",
            "[Optional] Available CMX memory"
        >,
        Option<
            "allowCustomValues", "allow-custom-values",
            "bool", "",
            "[Optional] Allows keep predefined values in IR"
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// DMATaskProfilingReserveMem
//

def DMATaskProfilingReserveMem : PassBase<"dma-task-profiling-reserve-mem", "vpux::ModulePass"> {
    let summary = "DMA task profiling memory reserving";

    let description = [{
        This pass adds in ModuleOp information about reserved memory for DMA profiling.
    }];

    let options = [
        Option<
            "enableDMAProfiling", "dma-profiling",
            "std::string", [{"false"}],
            "Enable DMA task profiling (true|static|false)"
        >
    ];

    let constructor = "vpux::VPU::createDMATaskProfilingReserveMemPass()";
}

//
// Reserve memory for compressed DMA
//

def CompressDmaReserveMem : PassBase<"compress-dma-reserve-mem", "vpux::ModulePass"> {
    let summary = "Reserve memory for additional compressedDMA metadata";

    let description = [{
        Reserve memory in CMX where additional metadata is stored for compressed DMAs handling activation spilling.
        In this memory Compressed DMA wil store a LUT with actual compressed data sizee that is later to be used
        by decompress DMA
    }];

    let constructor = "vpux::VPU::createCompressDmaReserveMemPass()";
}

//
// SWKernelDataPrefetchReserveMem
//

def SWKernelDataPrefetchReserveMem : PassBase<"sw-kernel-data-prefetch-reserve-mem", "vpux::ModulePass"> {
    let summary = "Reserve memory for SW Kernel data prefetching";

    let description = [{
        SW Kernel reads extra few bytes of data for better performance.
        When input buffer is at the end of CMX, reading extra data would cause ACT SHAVE read violation.
        Now we reserve CMX for profiling and activation compression at the end of CMX.
        This pass checks the total reserved memory size and inserts a dummy section if necessary, to ensure
        the total reserved memory size is safe for SW Kenel data prefetching.

        This pass eliminates the risk of ACT SHAVE read violation, but model performance might be impacted.
        One case is described in E#122488, where DMA and DPU execution pipeline is impacted by the memory layout adjustment.
        This can be optimized in the future by CMX allocation improvement.
    }];

    let constructor = "vpux::VPU::createSWKernelDataPrefetchReserveMemPass()";
}

//
// SWKernelInstructionPrefetchReserveMemForDummyKernels
//

def SWKernelInstructionPrefetchReserveMemForDummyKernels : PassBase<"sw-kernel-instruction-prefetch-reserve-mem-for-dummy-kernels", "vpux::ModulePass"> {
    let summary = "Reserve I/O memory for dummy SW kernels used instruction prefetching";

    let description = [{
        Dummy SW kernel is used to mimic SW instruction prefetch feature.
        To build these tasks up some reserved memory is required as dummy I/O space.
    }];

    let constructor = "vpux::VPU::createSWKernelInstructionPrefetchReserveMemForDummyKernelsPass()";
}

//
// CloneReservedResourcesFromTopModule
//

def CloneReservedResourcesFromTopModule : PassBase<"clone-reserved-resources-from-top-module", "vpux::ModulePass"> {
    let summary = "Clone reserved memory resources from top module into outlined nested modules";

    let description = [{
        Outlining done at VPU stage creates modules without reserved memory resources which can cause compilation
        issues down the pipeline as scheduling will work with different memory constraints to tiling. This pass clones
        reserved resources to avoid this issue.
    }];

    let constructor = "vpux::VPU::createCloneReservedResourcesFromTopModulePass()";
}

//
// SCFVerticalFusion
//

def SCFVerticalFusion : PassBase<"scf-vertical-fusion", "vpux::FunctionPass"> {
    let summary = "Apply VF to operations and convert it to scf representation";

    let description = [{
        Applies scf tiling to operation and searches for its parents to merge
        to the loop.
        For all operations which might be fused the optimal number of tiles is calculated.
        For the largest dimension the number is chosen so that VF passes memory requirements to be optimally fused.

        Current algorithm doesn't compare cost of VF among different dimensions tiling and
        it doesn't compare the cost of VF vs operations without VF.
        Algorithm provides greedy approach to merge as much as possible based on same limitations which
        non-scf algorithm has (aligned MC stratgies). To be extended to cost comparison further.
    }];

    let constructor = "vpux::VPU::createSCFVerticalFusionPass()";
}

//
// OutputPipelineTiling
//

def OutputPipelineTiling : PassBase<"output-pipeline-tiling", "vpux::FunctionPass"> {
    let summary = "Increase the number of tiles so that sub-tiles can be executed in parallel";

    let description = [{
        Originally, tiling pipeline created tiles for a conv based operation to fit below operands on CMX with PIPELINING tiling mode:
        1) Activation of current tile
        2) Weights of current tile
        3) Result of current tile
        4) Activation or weights of next tile for prefetching

        As a result, DPU computation of next tile has to wait for previous tile's output spilling
        when there's no enough CMX memory size left for the next tile's result.

        Output pipelining is introduced in to optimize such case by creating more and smaller tiles
        to fit all operands of 2 sub-tiles on CMX at the same time.
        This allows for better overlap between the cost of DMA data transmission and DPU computation.
    }];

    let constructor = "vpux::VPU::createOutputPipelineTilingPass()";

    let options = [
        Option<
            "tilingMode", "tiling-mode",
            "std::string", "",
            "[Optional] Set tiling mode as `ISOLATED` or `PREFETCH`"
        >
    ];
}

//
// Cost model analysis construct
//

def CostModelAnalysisConstruct : PassBase<"cost-model-analysis-construct", "vpux::ModulePass"> {
    let summary = "This pass constructs a vpunn cost model instance and preserve it in an analysis";

    let description = [{
        The pass constructs a VPUNN cost model according to option `version` and preserves it in analysis.
        The cost model instance exists until the AnalysisManager is cleared
        or the cost model analysis is cleared explicitly by pass `CostModelAnalysisDestroy`.
    }];

    let constructor = "vpux::VPU::createCostModelAnalysisConstructPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Cost model analysis destroy
//

def CostModelAnalysisDestroy : PassBase<"cost-model-analysis-destroy", "vpux::ModulePass"> {
    let summary = "This pass destroys a vpunn cost model instance if it exists";

    let description = [{
        The pass destroys the VPUNN cost model instance if it exists
        The destroyed cost model version is chosen by option `version`
    }];

    let constructor = "vpux::VPU::createCostModelAnalysisDestroyPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// NN Cache Statistics
//
def PrintNNCacheStatistics : PassBase<"print-nn-cache-statistics", "vpux::ModulePass"> {
    let summary = "This pass prints nn cache statistics log";

    let description = [{
        Since the cost model is singleton, the functional passA could affect the cache statistics for functional passB,
        which could make nn cahce statistics log inaccurtate in passB.
        This pass is used to synchronize parallel functions between functional passes, thus making nn cache statistics log more accurate.
    }];

    let constructor = "vpux::VPU::createPrintNNCacheStatisticsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Multi-cluster strategy assignment
//

def MultiClusterStrategyAssignment : PassBase<"multi-cluster-strategy-assignment", "vpux::FunctionPass"> {
    let summary = "This pass compute the hardware efficiency of layer that is executed as SOH or SOK and assigns the most optimal strategy";

    let constructor = "vpux::VPU::createMultiClusterStrategyAssignmentPass()";

    let options = [
        Option<
            "tilingMode", "tiling-mode",
            "std::string", [{"PREFETCH"}],
            "[Optional] Set tiling mode as `ISOLATED` or `PREFETCH`. `PREFETCH` is set by default"
        >,
        Option<
            "mcOptimizationScope", "mc-optimization-scope",
            "std::string", [{"subgraph"}],
            "[Optional] Set optimization scope as `local`, `subgraph` or `global`. `subgraph` is set by default"
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Make individual ops with distributed tensor for input and or output
//

def MakeOpsWithDistributedTensor : PassBase<"make-ops-with-distributed-tensor", "vpux::FunctionPass"> {
    let summary = "This pass creates vpu operations that should be executed across multiple clusters";

    let description = [{
        This pass builds an IR in order to represent multi-cluster compilation. It performs a number of functions.
        It creates variations of distributed tensors depending on the multi-cluster strategy of the layer.
    }];

    let constructor = "vpux::VPU::createMakeOpsWithDistributedTensorPass()";

    let options = [
        Option<
            "enableExplicitDistributionInfoAttr", "enable-explicit-distributed-attr",
            "bool", "false",
            "Flag to enable generating explicit DistributionInfoAttr for Distributed data type."
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Create DMAs from/to distributed tensors
//

def MakeDistributedCopies : PassBase<"make-distributed-copies", "vpux::FunctionPass"> {
    let summary = "";

    let description = [{
        This pass creates DMA operations DDR->CMX and CMX to DDR from UnrolledType
    }];

    let constructor = "vpux::VPU::createMakeDistributedCopiesPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// AdjustDistributedTensorAroundOps
//

def AdjustDistributedTensorAroundOps : PassBase<"adjust-distributed-tensor-around-ops", "vpux::FunctionPass"> {
    let summary = "This pass adjusts distributed tensor type for ops that should be executed across multiple clusters";

    let description = [{
        This pass will adjust the activation distributed tensor type for ops that should be executed across multiple clusters,
        so that the copy op chains can be optimized in further passes.
        For subgraph:
           Op0-> Overlapped type0 -> Copy0(CMX2DDR) -> Copy1(DDR2CMX) -> Overlapped type1 -> Op1
        When type0's compute shape is equal to type1's memory shape
    }];

    let constructor = "vpux::VPU::createAdjustDistributedTensorAroundOpsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Reorder IR to archive more efficient order for scheduler
//

def EfficientIROrder : PassBase<"efficient-ir-order", "vpux::FunctionPass"> {
    let summary = "Change the IR order to execute operation more efficiently";

    let description = [{
        Reorder IR around eltwise pattern in order to reach better performance on scheduling stage
    }];

    let constructor = "vpux::VPU::createEfficientIROrderPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "enableReorderConcatBranches", "enable-reorder-concat-branches",
            "bool", "true",
            "Flag to enable reorder concat branches"
        >
    ];
}

//
// Wrap operations in VerticalFusionOp
//

def WrapVerticalFusionRegion : PassBase<"wrap-in-vertical-fusion", "vpux::FunctionPass"> {
    let summary = "This pass wraps vpu operations that might be tiled in order to implement VF";

    let description = [{
        Wrap operations to VerticalFusion block which match criterias
        1. Operation has activation tiling (activation doesn't fit in CMX)
        2. NCE operations don't have strides larger than 1
        3. Even if NCE operation doesn't have activation tiling, but its kernel is 1x1,
        it also might be wrapped because there is no additional computation cost of it
    }];

    let constructor = "vpux::VPU::createWrapVerticalFusionRegionPass()";

    let options = [
        Option<
            "workloadManagementModeOpt", "workload-management-mode",
            "vpux::WorkloadManagementMode", "",
            "Setting which controls mode of WLM",
            [{::llvm::cl::values(
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V0_LCA, "PWLM_V0_LCA", "WLM enqueue barriers search algorithm at VPURT enabled"),
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V1_BARRIER_FIFO, "PWLM_V1_BARRIER_FIFO", "WLM enqueue barriers search algorithm at VPURT disabled"),
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V2_PAGES, "PWLM_V2_PAGES", "WLM with split into subgraphs(pages)")
            )}]
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Move view like operations to VF
//

def MoveViewOpsToVF : PassBase<"move-view-ops-to-vf", "vpux::FunctionPass"> {
    let summary = "Move view like operations to VF subgraphs";

    let description = [{
        Move view like operations to nearest VF subgraphs,
        which allows to build larger ones on the next step.
    }];

    let constructor = "vpux::VPU::createMoveViewOpsToVerticalFusionPass()";

    let options = [
        Option<
            "workloadManagementModeOpt", "workload-management-mode",
            "vpux::WorkloadManagementMode", "",
            "Setting which controls mode of WLM",
            [{::llvm::cl::values(
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V0_LCA, "PWLM_V0_LCA", "WLM enqueue barriers search algorithm at VPURT ENABLED"),
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V1_BARRIER_FIFO, "PWLM_V1_BARRIER_FIFO", "WLM enqueue barriers search algorithm at VPURT DISABLED"),
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V2_PAGES, "PWLM_V2_PAGES", "WLM with split into subgraphs(pages)")
            )}]
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Build VF Subgraph
//

def MergeVfSubgraphs : PassBase<"merge-vertical-fusion-subgraphs", "vpux::FunctionPass"> {
    let summary = "Build subgraph from VF single regions";

    let description = [{
        Merge VF blocks and add operations to them recalculating tiling information
        and halo regions in following cases:
        1. All operations have compatible tensor types (distributed and not distributed)
        2. Region which is supposed to be added doesn't have any other users except current region or
        all its users point to current region too. In newest versions the merging process is not skipped for other operands.
        3. All operations in new region after merging fit in CMX when they are tiled for VF. In case they don't, number of tiles
        increases.
        5. Required CMX memory by constant weights shouldn't exceed the threshold to avoid spilling.

        There are two versions of the VF merging implementation, referred to as v1 and v2. The differences in VF pipelining between
        these versions are as follows:
        1. v1 only enables pipelining for the pattern NCE -> SW -> NCE.
        2. v2 supports generic pipelining for any subgraph that includes both DPU and ActShave executor
    }];

    let constructor = "vpux::VPU::createMergeVfSubgraphsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "enableVerticalFusionPipelining", "enable-vertical-fusion-pipelining",
            "bool", "false",
            "Flag to enable vertical fusion pipelining"
        >,

        Option<
            "tilingMode", "tiling-mode",
            "std::string", [{"PREFETCH"}],
            "[Optional] Set tiling mode as `ISOLATED` or `PREFETCH`. `PREFETCH` is set by default"
        >,

        Option<
            "workloadManagementModeOpt", "workload-management-mode",
            "vpux::WorkloadManagementMode", "",
            "Setting which controls mode of WLM",
            [{::llvm::cl::values(
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V0_LCA, "PWLM_V0_LCA", "WLM enqueue barriers search algorithm at VPURT ENABLED"),
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V1_BARRIER_FIFO, "PWLM_V1_BARRIER_FIFO", "WLM enqueue barriers search algorithm at VPURT DISABLED"),
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V2_PAGES, "PWLM_V2_PAGES", "WLM with split into subgraphs(pages)")
            )}]
        >
    ];
}


//
// VF outlining
//

def VerticalFusionOutlining : PassBase<"vertical-fusion-outlining", "vpux::ModulePass"> {
    let summary = "Apply outlining on graph with vertical fusion regions as separator";

    let description = [{
        Apply outlining on graph with vertical fusion regions as the point where we separate the graph.
        Flag options controll the outlining process based on thresholds.
        The following function:

        func.func @main (%arg) {
            %0 = Op0(%arg)
            %1 = VFOp(%0)
            %2 = Op1(%1)
            return %2
        }

        Will be outlined into:

        func.func @main_vf1 (%arg) {
            %0 = Op0(%arg)
            return %0
        }
        func.func @main_vf2 (%arg) {
            %0 = VFOp(%arg)
            return %0
        }
        func.func @main_vf3 (%arg) {
            %0 = Op1(%arg)
            return %0
        }
        func.func @main (%arg) {
            %0 = call @main_vf1(%arg)
            %1 = call @main_vf2(%0)
            %2 = call @main_vf3(%1)
            return %2
        }
    }];

    let constructor = "vpux::VPU::createVerticalFusionOutliningPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "verticalFusionTileThreshold", "vf-outlining-tile-threshold",
            "int", "",
            "Tiling threshold for outlining vertical fusion regions"
        >,
        Option<
            "numInstanceThreshold", "vf-outlining-instance-threshold",
            "int", "",
            "Threshold for number of instances to perform outlining"
        >
    ];
}

//
// VF Tiling
//

def VfTiling : PassBase<"vertical-fusion-tiling", "vpux::FunctionPass"> {
    let summary = "Apply tiling on VF subgraph";

    let description = [{
        Apply VF tiling on subgraph wrapped in VF region.
    }];

    let constructor = "vpux::VPU::createVfTilingPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "enableVerticalFusionPipelining", "enable-vertical-fusion-pipelining",
            "bool", "false",
            "Flag to enable vertical fusion pipelining"
        >,
        Option<
            "workloadManagementModeOpt", "workload-management-mode",
            "vpux::WorkloadManagementMode", "",
            "Setting which controls mode of WLM",
            [{::llvm::cl::values(
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V0_LCA, "PWLM_V0_LCA", "WLM enqueue barriers search algorithm at VPURT ENABLED"),
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V1_BARRIER_FIFO, "PWLM_V1_BARRIER_FIFO", "WLM enqueue barriers search algorithm at VPURT DISABLED"),
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V2_PAGES, "PWLM_V2_PAGES", "WLM with split into subgraphs(pages)")
            )}]
        >
    ];
}

//
// Unroll unused VerticalFusionOp
//

def UnrollUnusedVerticalFusionRegion : PassBase<"unroll-unused-vertical-fusion", "vpux::FunctionPass"> {
    let summary = "Unroll single VF blocks";

    let description = [{
        Unroll VF block in case it hasn't been assembled with other blocks in subgraph
    }];

    let constructor = "vpux::VPU::createUnrollUnusedVerticalFusionRegionPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Manual strategy utils
//

def ManualStrategyUtils : PassBase<"manual-strategy-utils", "vpux::FunctionPass"> {
    let summary = "Utils for reading or writing a json strategy";

    let description = [{
        Utility allowing to store and write as JSON the current selected strategy from the two strategy passes
        createMultiClusterStrategyAssignmentPass() and createPrefetchTilingPass(). And also to manually
        overwrite the strategy.
    }];

    let constructor = "vpux::VPU::createManualStrategyUtilsPass()";

    let options = [
        Option<
            "writeStrategyToJSON", "write-strategy-to-json",
            "bool", "false",
            "Flag to enable writing strategy to file"
        >,
        Option<
            "writeStrategyFileLocation", "write-strategy-file-location",
            "std::string", [{"strategy.json"}],
            "Location/path to write strategy file"
        >,
        Option<
            "readStrategyFromJSON", "read-strategy-from-json",
            "bool", "false",
            "Flag to enable reading strategy from file"
        >,
        Option<
            "readStrategyFileLocation", "read-strategy-file-location",
            "std::string", [{"strategy.json"}],
            "Location/path to read strategy file"
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SplitNCEOpsOntoWorkloads
//

def SplitNCEOpsOntoWorkloads : PassBase<"split-NCE-ops-onto-workloads", "vpux::FunctionPass"> {
    let summary = "Split VPU NCE operation onto workloads";

    let constructor = "vpux::VPU::createSplitNCEOpsOntoWorkloadsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// CorrectNCEWorkloads
//

def CorrectNCEWorkloads : PassBase<"correct-NCE-workloads", "vpux::FunctionPass"> {
    let summary = "Correct NCE workloads if they do not fit requirements";

    let description = [{
        The pass adjusts workload size for NCEDepthConvolution, NCEMaxPool and NCEAveragePool,
        as well as for NCE operations that produce sparse activations.

        NCEDepthConvolutionOp, NCEMaxPoolOp and NCEAveragePoolOp require the number of channels to be of the supported channel values for the platform.
        If the number of channels does not match, workload is split.

        NCE operations with sparse outputs must have all variants with the same number of channels. Additionally, if the NCE op shares a
        consumer with another NCE op (directly or indirectly), the number of channels of their variants must be aligned.
    }];

    let constructor = "vpux::VPU::createCorrectNCEWorkloadsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Strategy manager pass
//

def StrategyManagerImpl : PassBase<"strategy-manager", "vpux::FunctionPass"> {
    let summary = "Assignment and optimization multi-cluster strategies to operations";

    let description = [{
        Pass consists of two parts:
        1. Assignment of multicluster strategies and tiling strategies to each operation based on vpunn cost of each strategy.
        2. Optimization/adjustment of strategies based on one of common optimization algorithm.
    }];

    let constructor = "vpux::VPU::createStrategyManagerImplPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "tilingMode", "tiling-mode",
            "std::string", [{"PREFETCH"}],
            "[Optional] Set tiling mode as `ISOLATED` or `PREFETCH`. `PREFETCH` is set by default"
        >
    ];
}

//
// ResolveEltwiseWithZTiledWorkloads
//

def ResolveEltwiseWithZTiledWorkloads : PassBase<"resolve-eltwise-with-z-tiled-workloads", "vpux::FunctionPass"> {
    let summary = "Resolves Eltwise operations which have workloads tiled over Z";

    let description = [{
        Hardware Eltwise does not support variants tiled over the Z dimension. If such cases are encountered,
        these operations are split into separate Eltwise operations, each containing the workloads that cover
        a different subset of channels.

        For example, if the original Eltwise contains the following workloads:
            1. offset = [0, 0,  0, 0], sizes = [1, 64, 8, 16], cluster_id = 0
            2. offset = [0, 64, 0, 0], sizes = [1, 64, 8, 16], cluster_id = 0
            3. offset = [0, 0,  8, 0], sizes = [1, 64, 8, 16], cluster_id = 1
            4. offset = [0, 64, 8, 0], sizes = [1, 64, 8, 16], cluster_id = 1
        Two Eltwise operations will be created, the first one containing workloads 1 and 3, the other one
        workloads 2 and 4, with their channel offsets reset to zero. The correct subset of channels is
        sliced individually for each new Eltwise operation.

        In case the inputs are distributed types in CMX, manual copy operations that spill them to DDR are
        introduced, in order to avoid Slice operations that work with these types. These Slice operations
        would get lowered to copies where both the input and output are distributed types; such scenarios
        are not fully supported (E#78676).

        The outputs of the smaller Eltwise operations get copied to DDR in order to avoid accuracy degradation
        that takes place when the outputs are concatenated in CMX.
    }];

    let constructor = "vpux::VPU::createResolveEltwiseWithZTiledWorkloadsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ShiftOutputWorkloadsForHalo
//

def ShiftOutputWorkloadsForHalo : PassBase<"shift-dpu-workloads-start", "vpux::FunctionPass"> {
    let summary = "Adjust output workload start for each dpu task according to halo information";

    let description = [{
        For arch 40XX, output workload start offsets are computed relative to the full tensor. The start offsets need to be
        adjusted to reflect position in the cluster that computes the current section of output.

        In addition to that, in case of output Distributed OVERLAPPED, this pass will take care to shift the offsets to
        make space for any halo that needs to be put at the beginning of any dimension.
    }];

    let constructor = "vpux::VPU::createShiftOutputWorkloadsForHaloPass()";
    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// AdjustMemorySpace
//

def AdjustMemorySpace : PassBase<"adjust-memory-space", "vpux::FunctionPass"> {
    let summary = "Adjusts the tensor location for VPU-driven operations";

    let description = [{
        The pass adjusts the location of tensors that are used by hardware-driven operations

        Currently, it surrounds VPU-driven nodes with Copy operations to specify that all the data
        that they consume/produce must reside in CMX
    }];

    let constructor = "vpux::VPU::createAdjustMemorySpacePass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// AdjustMemorySpaceForSHVOps
//

def AdjustMemorySpaceForSHVOps : PassBase<"adjust-memory-space-for-shv-ops", "vpux::FunctionPass"> {
    let summary = "Adjusts the memory space for SHAVE operations";

    let description = [{
        The pass adjusts the memory space for SHAVE operations, by placing their tensors in either CMX or DDR.

        It identifies all operations that are known to be lowered to SHAVE and attempts to place all of the
        operands and results in CMX. In case not all tensors fit, the pass selects the ones which will
        maximize the CMX utilization. For example, if the operand is small and the result is large, the result
        is chosen to be placed in CMX.

        Copies are introduced to move the operands to CMX or to move the results back to DDR.
    }];

    let constructor = "vpux::VPU::createAdjustMemorySpaceForSHVOpsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// OptimizeSharedInputCopyForConcat
//

def OptimizeSharedInputCopyForConcat : PassBase<"optimize-shared-input-copy-for-concat", "vpux::FunctionPass"> {
    let summary = "Optimize duplicated input copy for concat";

    let constructor = "vpux::VPU::createOptimizeSharedInputCopyForConcatPass()";

    let description = [{
        Convert pattern:
        ```
                                  / Concat -> Slice - Copy(DDR2CMX)
            Copy(CMX2DDR) -> Concat
                                  \ Concat -> Slice - Copy(DDR2CMX)
        ```
        as
        ```
                                  / Slice -> Copy(DDR2CMX) -> CMXConcat
            Copy(CMX2DDR) -> Concat
                                  \ Slice ->  Copy(DDR2CMX) -> CMXConcat
        ```
        The original subgraph will try to copy to DDR multi times, and after this converion,
        the CMX2DDR copy will be reduced to only once.
    }];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}



//
// CMXConcat
//

def CMXConcat : PassBase<"cmx-concat", "vpux::FunctionPass"> {
    let summary = "Move Concat operations from DDR to NNCMX";

    let constructor = "vpux::VPU::createCMXConcatPass()";

    let description = [{
        This pass will try to check if a Concat operation can fit in NNCMX
        with few restrictions and if so move the concat from DDR to NNCMX.
    }];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// AddSwOpAuxiliaryBuffer
//

def AddSwOpAuxiliaryBuffer : PassBase<"add-sw-op-auxiliary-buffer", "vpux::FunctionPass"> {
    let summary = "Adds buffers needed for specific Software Operations.";

    let description = [{
        It requires the creation of a auxiliary input buffer, who has the role of storing
        the intermediate results obtained after the relevant operation.
        The results will be sorted, recalculated, and finally, the output will be extracted based on them.
        This auxiliary buffer has a necessary size depend by input parameters.
    }];

    let constructor = "vpux::VPU::createAddSwOpAuxiliaryBufferPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SplitGRUSequence
//

def SplitGRUSequence : PassBase<"split-gru-sequence", "vpux::FunctionPass"> {
    let summary = "Split GRUSequence into GRUSequenceFirstPart and GRUSequenceLastPart";

    let description = [{
        The pass can split GRUSequence into two parts to fit into CMX when tiling strategy can't be generated.
    }];

    let constructor = "vpux::VPU::createSplitGRUSequencePass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// DetectInPlaceEltwise
//

def DetectInPlaceEltwise : PassBase<"detect-in-place-eltwise", "vpux::FunctionPass"> {
    let summary = "Convert Eltwise operation to read and write to the same buffer in memory";

    let description = [{
        This pass will check if Eltwise operation has input and output buffers of the same size
        in memory and mark such Eltwise eligible for inplace execution.
        It will write the result into one of the inputs in memory.
    }];

    let constructor = "vpux::VPU::createDetectInPlaceEltwisePass()";
}


//=================================================================================
// Sparsity
//=================================================================================

//
// WrapOpsInSparsifyDesparsifyPairs
//

def WrapOpsInSparsifyDesparsifyPairs : PassBase<"wrap-ops-in-sparsify-pairs", "vpux::FunctionPass"> {
    let summary = "Wrap operations in pairs of Sparsify-Desparsify";

    let description = [{
        Wraps operations in pairs of Sparsify-Desparify ops. The sparsity profile
        will determine which operations will be wrapped:
        - profile S0: add SparsifyOp for each input and Sparsify-Desparsify chain for output
        - profile S1: add Sparsify-Desparsify chain both for inputs and output
    }];

    let constructor = "vpux::VPU::createWrapOpsInSparsifyDesparsifyPairsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "enableActivationSparsityMode", "enable-activation-sparsity-mode",
            "std::string", [{"false"}],
            "Activation sparsity enablement mode (auto, true or false)"
        >,
        Option<
            "sparsityProfile", "sparsity-profile",
            "std::string", [{""}],
            "Flag to choose sparsity profile"
        >
    ];
}

//
// FuseSparsityOps
//

def FuseSparsityOps : PassBase<"fuse-sparsity-ops", "vpux::FunctionPass"> {
    let summary = "Fuse subsequent [De]SparsifyOps with SparseOpInterface ops";

    let constructor = "vpux::VPU::createFuseSparsityOpsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "fuseSparsify", "fuse-sparsify",
            "bool", "false",
            "Flag to choose inputs or output will be handled"
        >
    ];
}

//
// OptimizeSparsifyDesparsifyPairs
//

def OptimizeSparsifyDesparsifyPairs : PassBase<"optimize-sparsify-desparsify-pairs", "vpux::FunctionPass"> {
    let summary = "Optimize common patterns of subsequent sparsify-desparsify ops to remove redundant conversions";

    let constructor = "vpux::VPU::createOptimizeSparsifyDesparsifyPairsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "actSparsityProfile", "act-sparsity-profile",
            "std::string", [{""}],
            "Flag to choose activation sparsity profile"
        >
    ];
}

//
// OptimizeSparsityOps
//

def OptimizeSparsityOps : PassBase<"optimize-sparsity-ops", "vpux::FunctionPass"> {
    let summary = "Optimize additional sparsity patterns";

    let description = [{
        Some optimizations such duplicated Sparsify ops for Eltwise, first Sparsify
        or last Desparsify cant be done during WrapOpsInSparsifyDesparsifyPairs pass
        until output sparsity wouldnt be fused
    }];

    let constructor = "vpux::VPU::createOptimizeSparsityOpsPass(vpux::VPU::symbolizeEnum<VPU::ActivationSparsityProfile>)";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "sparsityProfile", "sparsity-profile",
            "std::string", [{""}],
            "Flag to choose sparsity profile"
        >
    ];
}

//
// LowerSparsityOps
//

def LowerSparsityOps : PassBase<"lower-sparsity-ops", "vpux::FunctionPass"> {
    let summary = "Convert Sparsify/Desparsify ops to Eltwise or GroupSparseBufferOp";

    let constructor = "vpux::VPU::createLowerSparsityOpsPass()";

    let description = [{
        Converts Sparsify operations to Convolutions and Desparsify operations to Eltwise ops.

        In case the `fakeSparsity` flag is set to true, Sparsify operations are instead converted to a
        GroupSparseTensor operation whose sparsity map contains only values of 1. This lets the data be
        interpreted as a sparse one without actually removing the sparse values.
    }];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "fakeSparsify", "fake-sparsify",
            "bool", "false",
            "Flag to choose method of VPU.Sparsify lowering"
        >
    ];
}

//
// SparsifyWeights
//

def SparsifyWeights : PassBase<"sparsify-weights", "vpux::FunctionPass"> {
    let summary = "Sparsify weights for NCE ops";

    let description = [{
        Convert const parameters for NCE ops to sparse types depending on sparsify strategy.
    }];

    let constructor = "vpux::VPU::createSparsifyWeightsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

}

//
// RecomputeSparsityPtrs
//

def RecomputeSparsityPtrs : PassBase<"recompute-sparsity-ptrs", "vpux::FunctionPass"> {
    let summary = "Recomputes sparsity pointers";

    let description = [{
        Recomputes the sparsity pointers inside the weights table for sparse weights.
    }];

    let constructor = "vpux::VPU::createRecomputeSparsityPtrsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

}

//
// AddSparsityMapToSparseActivations
//

def AddSparsityMapToSparseActivations : PassBase<"add-sparsity-map-to-sparse-activations", "vpux::FunctionPass"> {
    let summary = "Update type of result for operations which produce SparseTensor type.";

    let description = [{
        Pass updates output type of operations which produce sparsified output. It adds sparsity_map to output tensor type.
        Then it propagates type to all users until sparse data consumer is reached.
    }];

    let constructor = "vpux::VPU::createAddSparsityMapToSparseActivationsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SplitSEOps
//

def SplitSEOps : PassBase<"split-se-ops", "vpux::FunctionPass"> {
    let summary = "Split compatible SE operations for better performance";

    let description = [{
        Finds operations that can be executed on hardware using the Storage Element pointers
        feature and splits them if they have benefits for performance.

        The list of supported operations:
        Interpolate Op that satisfied the following limitations:
        - The NCEInterpolate size is larger than the CMX size;
        - Both factors on H and W are larger than 4
    }];

    let constructor = "vpux::VPU::createSplitSEOpsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "seOpsEnabled", "se-ops-enabled",
            "bool", "false",
            "Flag to identify whether operations that can be executed using the Storage Element hardware feature are enabled"
        >,
        Option<
            "seExperimentalOpsEnabled", "se-experimental-ops-enabled",
            "bool", "false",
            "This flag identifies operations that are still a work in progress and can be executed using the Storage Element hardware feature."
        >
    ];
}

//
// LowerOpsToSENCE
//

def LowerOpsToSENCE : PassBase<"lower-ops-to-se-nce", "vpux::FunctionPass"> {
    let summary = "Converts compatible operations to SE NCE operations";

    let description = [{
        Finds operations that can be executed on hardware using the Storage Element pointers
        feature and lowers them to VPU.NCE.

        The list of supported operations:
        - Interpolate - mode: NEAREST
                        axes: H, W
                        coordinate_transformation_mode: all (except ALIGN_CORNERS)
                        nearest_mode: all
                        scale: integer only
                        padding: none

                      - mode: LINEAR, LINEAR_ONNX
                        axes: H, W
                        coordinate_transformation_mode: ASYMMETRIC, PYTORCH_HALF_PIXEL, HALF_PIXEL
                        scale: integer only
                          - values between [1-5] for PYTORCH_HALF_PIXEL and HALF_PIXEL
                            when the scale is an even number
                          - values between [1-11] for all other cases
                        padding: none
    }];

    let constructor = "vpux::VPU::createLowerOpsToSENCEPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "seOpsEnabled", "se-ops-enabled",
            "bool", "false",
            "Flag to identify whether operations that can be executed using the Storage Element hardware feature are enabled"
        >,
        Option<
            "seExperimentalOpsEnabled", "se-experimental-ops-enabled",
            "bool", "false",
            "This flag identifies operations that are still a work in progress and can be executed using the Storage Element hardware feature."
        >
    ];
}


//
// ConvertNCEInterpolateToDW
//

def ConvertNCEInterpolateToDW : PassBase<"convert-nce-interpolate-to-dw", "vpux::FunctionPass"> {
    let summary = "Converts SEP NCE.Interpolate into NCE.DepthConvolution with SEP input";

    let description = [{
        In certain configurations, running a DPU interpolate op on DW.Conv instead of Conv brings
        performance benefits. This pass makes the decision to switch from VPU.NCE.Interpolate (Conv)
        to VPU.NCE.DepthConvolution with SEP input when it is beneficial to do so.

        The pass takes into consideration the multiclustering strategy when deciding how to run the
        interpolate. As such it is placed after MC strategy is known, but before tiling, as some
        tiling might be needed to make workload channels supported for SEP DW.Conv. Experiments show
        that it could still be better performance-wise with the added tiling in some cases.
    }];

    let constructor = "vpux::VPU::createConvertNCEInterpolateToDWPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// FuseNCEInterpolateConsumers
//

def FuseNCEInterpolateConsumers : PassBase<"fuse-nce-interpolate-consumers", "vpux::FunctionPass"> {
    let summary = "Fuses NCE.Interpolate into consumer NCE.Convolution";

    let description = [{
        Fuses NCE.Interpolate with consumer NCE.Convolution.

        NCE.Interpolate with mode "nearest" is lowered to a dummy NCE.Convolution with
        SE table that upsamples tensor.

        We can simply pass the SE table to this consumer and avoid the dummy convolution.
    }];

    let constructor = "vpux::VPU::createFuseNCEInterpolateConsumersPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//=================================================================================
// Tiling
//=================================================================================

//
// Tiling Strategy Assignment
//

def TilingStrategyAssignment : PassBase<"tiling-strategy-assignment", "vpux::FunctionPass"> {
    let summary = "Assign tiling strategy for layers applicable";

    let description = [{
        The pass assigns tiling strategy for layers whose memory requirements exceed the capacity available.
        The pass only assigns strategy and do not perform any tiling actions, and if tiling strategy is set by
        ManualStrategyUtilsPass, it will not be processed by this pass.

        Isolated tiling: split each single layer in isolation, with no smarter heuristics such as
                         "allow running in parallel" or "allow continious computation in tiles" or any else.
        Prefetch tiling: tries to run tiles in parallel, and 'prefetch' means that the next tile could be loaded
                         in advance when the current tile is computing.

        The pass does not use any cost model to optimize the entire layer's processing time.
    }];

    let constructor = "vpux::VPU::createTilingStrategyAssignmentPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "vpux::VPUIP::VPUIPDialect"
    ];

    let options = [
        Option<
            "tilingMode", "tiling-mode",
            "std::string", [{"PREFETCH"}],
            "[Optional] Set tiling mode as `ISOLATED` or `PREFETCH`"
        >,
        Option<
            "enableVpunnCostForTiling", "enable-vpunn-cost-for-tiling",
            "bool", "false",
            "Use VPUNN cost model to determine tiling strategy, with this flag the pass will be utilizing tiling strategy selected based on least DMA+DPU cost"
        >,
        Option<
            "enableShaveDDRAccessOptimization", "enable-shave-ddr-access-optimization",
            "std::string", [{"true"}],
            "SHAVE DDR access optimization option (true, false or auto)"
        >
    ];
}

//
// Apply Tiling
//

def ApplyTiling : PassBase<"apply-tiling", "vpux::FunctionPass"> {
    let summary = "Apply tiling on layers with assigned tiling strategy";

    let description = [{
        The pass applies tiling strategy on layers with previously assigned strategy attribute.
    }];

    let constructor = "vpux::VPU::createApplyTilingPass()";

    let options = [
        Option<
            "enableSCFTiling", "enable-scf-tiling",
            "bool", "false",
            "Flag to enable tiling with scf dialect"
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// DetectionOutput decomposition
//

def DetectionOutputDecomposition : PassBase<"detection-output-decomposition", "vpux::FunctionPass"> {
    let summary = "Replace DetectionOutput operation with a subgraph of smaller operations";

    let description = [{
        Replace DetectionOutput operation
        ┌─────────┐   ┌────────────────┐  ┌──────────┐
        │BoxLogits│   │ClassPredictions│  │PriorBoxes│
        └────┬────┘   └───────┬────────┘  └─────┬────┘
             │                │                 │
             │         ┌──────┴────────┐        │
             └─────────┤DetectionOutput├────────┘
                       └───────────────┘

        with a subgraph (Reshapes and MemPermutes are ommited)
        ┌─────────┐  ┌──────────┐        ┌────────────────┐
        │BoxLogits│  │PriorBoxes│        │ClassPredictions│
        └───────┬─┘  └─┬────────┘        └───────┬────────┘
                │      │                         │
        ┌───────┴──────┴───────────┐  ┌──────────┴────────┐
        │DetectionOutputDecodeBoxes│  │DetectionOutputSort│
        └───────────────────────┬──┘  └──┬───┬───┬────────┘
                                │        │   │   │
                                │        │   │   │
                             ┌──┴────────┴───┴───┴───┐
                             │DetectionOutputNmsCaffe│
                             └─────────┬─┬─┬─────────┘
                                       │ │ │
                          ┌────────────┴─┴─┴────────────┐
                          │DetectionOutputCollectResults│
                          └─────────────────────────────┘
    }];

    let constructor = "vpux::VPU::createDetectionOutputDecompositionPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

def AdjustLSTMCellInputs : PassBase<"adjust-lstmcell-inputs", "vpux::FunctionPass"> {
    let summary = "Reorder and reshape the weights for LSTMCell";

    let description = [{
        To satisfy the software kernel constraints necessary for performance,
        this pass reorders the weights of LSTMCell as follows:

        weights - 4D tensor of shape [1, 4, hidden_size, input_size] and layout
        NWHC.
        recurrenceWeights - 4D tensor of shape [1, 4, hidden_size, hidden_size]
        and layout NWHC.
        biases - 4D tensor of shape [1, 1, 4, hidden_size] and layout NCWH.
    }];

    let constructor = "vpux::VPU::createAdjustLSTMCellInputsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Tile LSTMSequence
//

def TileLSTMSequence : PassBase<"tile-lstm-sequence", "vpux::FunctionPass"> {
    let summary = "Tile the LSTMSequenceOp sequentially";

    let description = [{
        Tile the LSTMSequenceOp sequentially, creating a chain of
        LSTMSequenceOps split on the seq_length dimension.
    }];

    let constructor = "vpux::VPU::createTileLSTMSequencePass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}


//
// ComputeInterpolateCoordinates
//

def ComputeInterpolateCoordinates : PassBase<"compute-interpolate-coordinates", "vpux::FunctionPass"> {
    let summary = "Compute interpolate coordinates and lambdas for the innermost axis.";

    let description = [{
        The interpolate coordinates and lambdas coefficients are computed at
        compile time to improve performance and simplify the kernel.

        This computation is only done for the innermost axis because this is
        where the kernel's main loop operates and where the most complexity
        lies. Without this pass, the kernel would need to calculate these
        constants multiple times to avoid excessive stack usage. Calculating
        these constants for other axes would make the multi-cluster and
        multi-shave tiling more restrictive.

        This pass has to run after the Interpolate operation is tiled to fit
        into CMX.
        Additionally, the interpolate operation must ensure that the constants
        calculated by this pass will also fit into CMX.

        The coordinates are byte offsets in the tiled input. The lambdas
        contain two interleaved values for each coordinate.

    }];

    let options = [
        Option<
            "enableExplicitDistributionInfoAttr", "enable-explicit-distributed-attr",
            "bool", "false",
            "Flag to enable generating explicit DistributionInfoAttr for Distributed data type."
        >
    ];

    let constructor = "vpux::VPU::createComputeInterpolateCoordinatesPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// RelocateWeightTableForReuse
//

def RelocateWeightTableForReuse : PassBase<"relocate-weight-table-for-reuse", "vpux::FunctionPass"> {
    let summary = "Relocate weight table constant for reuse.";

    let description = [{
        This pass relocates weight table constant with zero offset
        in order to be further reused.
        For weight table with multiclustered segmented case, each cluster
        has its content of weight table offseted from zero.
            For example:
                Looking at the following weight table const.Declare tensor<1152x1x1x4xsi32>
                The offsets before relocate pass are the following :
                    0, 256, 512,.........., 294400, 294656.
                If distribution mode is segmented and num_clusters 4, after relocate pass the offsets are changed like below:
                    0, 256,....., 73216, 73472, 0, 256,....., 73216, 73472, 0, 256,....., 73216, 73472, 0, 256,....., 73216, 73472.
    }];

    let constructor = "vpux::VPU::createRelocateWeightTableForReusePass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// CorrectStorageElementTableSeSizeForSEPDWConv
//

def CorrectStorageElementTableSeSizeForSEPDWConv : PassBase<"correct-storage-element-table-sesize-for-sep-dwconv", "vpux::FunctionPass"> {
    let summary = "Correct sesize attribute for VPU.StorageElementTable which is used as input for NCE DWConv";

    let description = [{
        For SEP DWConv, input has sparse tensor type with default seSize 16. While there is HW limitation which requires the workload offset start_z equals to 0,
        which means there should be only one workload per cluster.
        Through TilingStrategyAssignmentPass, the op is guaranteed to have only one workload per cluster, but every cluster may have different worload size on Z
        For exmaple,
        With numClusters = 2, seSize=16, SOK multi cluster strategy
        1. Original input Shape [1, 64, 8, 8], the updated seSize should be [32, 32]
            CL0: Workload [1, 32, 8, 8], seSize = 32
            CL1: Workload [1, 32, 8, 8], seSize = 32
        2. Original input Shape [1, 96, 8, 8], the updated seSize should be [64, 32]
            CL0: Workload [1, 64, 8, 8], seSize = 64
            CL1: Workload [1, 32, 8, 8], seSize = 32

    }];


    let constructor = "vpux::VPU::createCorrectStorageElementTableSeSizeForSEPDWConvPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// EnsureNCEOpsSizeRequirements
//

def EnsureNCEOpsSizeRequirements : PassBase<"ensure-nce-ops-size-requirements", "vpux::FunctionPass"> {
    let summary = "Ensure hw operations meet size requirements";

    let description = [{
        This pass ensures that hardware operations meet hardware size requirements:
        each operation need to have less than 8192 values per dimension. This is done
        by tiling such operations into smaller ones.
    }];

    let constructor = "vpux::VPU::createEnsureNCEOpsSizeRequirementsPass()";

    let options = [
        Option<
            "enableOutputEnsurance", "enable-output-ensurance",
            "bool", "true",
            "Flag to enable output size ensurance"
        >,
        Option<
            "enableDequantWeightEnsuranceBeforeStrategy", "enable-dequant-weight-ensurance-before-strategy",
            "bool", "false",
            "Skip ops where the weights are dequantized via an explicit operation before strategy is assigned"
        >,
        Option<
            "skipNonConvOC", "skip-non-conv-oc",
            "bool", "false",
            "Flag to skip checking the output channel of non-convolution op"
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ConvolutionSplitOverInputChannel
//

def ConvolutionSplitOverInputChannel : PassBase<"convolution-split-over-input-channel", "vpux::FunctionPass"> {
    let summary = "Split convolution over input channel to avoid 2D tiling";

    let description = [{
        This pass matches ConvOps that have 2D tilingStrategy, and large inputs/weights and small outputs.
        Then it splits the ConvOps over input channel and accumulate these new ConvOps using AddOps.
    }];

    let constructor = "vpux::VPU::createConvolutionSplitOverInputChannelPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// FuseClampPass
//

def FuseClampPass : PassBase<"fuse-clamp", "vpux::FunctionPass"> {
    let summary = "Fuses VPU.Clamp parameters into previous NCE operation";

    let description = [{
        This pass follows `SetupPPEPass` and fuses VPU.Clamp with already existing PPE task.
        1. Search for VPU.NCE -> VPU.Clamp pattern
        2. Fetch min and max parameters from VPU.Clamp
        3. Set clamp_low and clamp_high according to min, max and existing activation
        4. Remove VPU.Clamp from the graph
    }];

    let constructor = "vpux::VPU::createFuseClampPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// OptimizeConcat
//

def OptimizeConcat : PassBase<"optimize-concat", "vpux::FunctionPass"> {
    let summary = "Try to eliminate Concat for Concat-Slice pattern";

    let description = [{
        After `ApplyTilingPass` lots of `VPU.Concat`-`VPU.Slice` are introduced, `VPU.Concat` can be eliminated only if all its users are `VPU.Slice` and
        input tensor of each `VPU.Slice` is actually sub-tensor of one of input tensors of Concat.
    }];

    let constructor = "vpux::VPU::createOptimizeConcatPass()";

    let options = [
        Option<
            "optimizeOnlyOuterConcat", "optimize-only-outer-concat",
            "bool", "false",
            "Flag to check dimension and only optimize concat when the dimension is the highest."
        >,
        Option<
            "disablePassOnEntryFunction", "disable-pass-on-entry-function",
            "bool", "false",
            "Flag to disable the pass on entry function for host compilation."
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SetupPipelineOptions
//

def SetupPipelineOptions : PassBase<"setup-pipeline-options", "vpux::ModulePass"> {
    let summary = "Initialize the symbol table for global options in the model/IR";

    let description = [{
        Creates a PipelineOptionsOp structure in the IR so the constraints could be exposed globally
    }];

    let constructor = "vpux::VPU::createSetupPipelineOptionsPass()";

    let options = [
        Option<
            "archOpt", "vpu-arch",
            "std::string", "",
            "VPU architecture to compile for"
        >,
        Option<
            "allowCustomValues", "allow-custom-values",
            "bool", "",
            "[Optional] Allows keep predefined values in IR"
        >,
        Option<
            "ppeVersionOpt", "ppe-version",
            "std::string", [{"Auto"}],
            "Specifies the compiler's target PPE hardware version ['Auto', 'IntPPE', 'FpPPE']. When set to 'Auto', the latest PPE version available on the target architecture is picked."
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SetupNpuConstraint
//

def SetupNpuConstraint : PassBase<"setup-npu-constraint", "vpux::ModulePass"> {
    let summary = "Set WLM related constraints in the model/IR";

    let description = [{
        Uses the PipelineOptionsOp structure from the IR so the WLM constraints could be exposed globally.
        Sets the constraints like BarrierMaxVariantCount, BarrierMaxVariantSum, MetadataMaxInvariantCount, MetadataMaxKernelInvocationCount,
        MetadataMaxVariantCount, MetadataMaxKernelRangeCount, MetadataMaxMediaCount
    }];

    let constructor = "vpux::VPU::createSetupNpuConstraintPass()";

    let options = [
        Option<
            "workloadManagementStatus", "workload-management-status",
            "vpux::WorkloadManagementStatus", "vpux::WorkloadManagementStatus::ENABLED",
            "[Optional] Set workload management status",
            [{::llvm::cl::values(
                clEnumValN(vpux::WorkloadManagementStatus::ENABLED, "ENABLED", "workload management enabled"),
                clEnumValN(vpux::WorkloadManagementStatus::DISABLED, "DISABLED", "workload management disabled"),
                clEnumValN(vpux::WorkloadManagementStatus::FAILED, "FAILED", "workload management attempted, but failed")
            )}]
        >,
        Option<
            "allowCustomValues", "allow-custom-values",
            "bool", "",
            "[Optional] Allows keep predefined values in IR"
        >,
        Option<
            "wlmRollback", "wlm-rollback",
            "bool", "false",
            "[Optional] Indicates that compilation might rollback to wlm-disabled pipeline"
        >,
        Option<
            "enableSwKernelFifoPerShaveEngine", "enable-sw-kernel-fifo-per-shave-engine",
            "bool", "",
            "Enable separate SW kernel FIFO for each Shave engine"
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SetupMaxKernelSize
//

def SetupMaxKernelSize : PassBase<"setup-max-kernel-size", "vpux::ModulePass"> {
    let summary = "Set maximum kernel size allowed by arch in the model/IR";

    let description = [{
        Uses the PipelineOptionsOp structure from the IR so the constant could be exposed globally
    }];

    let constructor = "vpux::VPU::createSetupMaxKernelSizePass()";

    let options = [
        Option<
            "allowCustomValues", "allow-custom-values",
            "bool", "",
            "[Optional] Allows keep predefined values in IR"
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SetTargetIndependentPassOptions
//

def SetTargetIndependentPassOptions : PassBase<"set-target-independent-options", "vpux::ModulePass"> {
    let summary = "Enable pass options and store it in the Module/IR";

    let description = [{
        Uses the PipelineOptionsOp structure from the IR so the options could be exposed globally. Adds several Symbols aka Options to PipelineOptions sumbol table
    }];

    let constructor = "vpux::VPU::createSetTargetIndependentPassOptionsPass()";

    let options = [
        Option<
            "enableAutoPaddingODU", "enable-auto-padding-odu",
            "bool", "false",
            "[Optional] Set auto padding for ODU to true/false"
        >,
        Option<
            "enableAutoPaddingIDU", "enable-auto-padding-idu",
            "bool", "false",
            "[Optional] Set auto padding for IDU to true/false"
        >,
        Option<
            "enableIsReduceSupported", "enable-is-reduce-supported",
            "bool", "false",
            "[Optional] Set IsReduceSupported for NCE to true/false"
        >,
        Option<
            "enableFP16CompressedConvolution", "enable-fp16-compressed-convolution",
            "bool", "false",
            "Enable FP16 Compressed convolution op"
        >,
        Option<
            "enableVPUNNPreSplit", "enable-vpunn-pre-split",
            "bool", "false",
            "Enable VPUNN pre-split API"
        >,
        Option<
            "weightsTableReuseMode", "weights-table-reuse-mode",
            "vpux::WeightsTableReuseMode", "",
            "Setting the weights table reuse mode",
            [{::llvm::cl::values(
                clEnumValN(vpux::WeightsTableReuseMode::ENABLED, "ENABLED", "Weights table can be reused for all operations that support it"),
                clEnumValN(vpux::WeightsTableReuseMode::VF_ENABLED, "VF_ENABLED", "Weights table can be reused for operations in pure-vertical-fusion region, to avoid possible memory fragmentation"),
                clEnumValN(vpux::WeightsTableReuseMode::DISABLED, "DISABLED", "Weights table cannot be reused")
            )}]
        >,
        Option<
            "enableSEPtrsOperations", "enable-se-ptrs-operations",
            "bool", "false",
            "[Optional] Set SEPtrsOperations to true/false"
        >,
        Option<
            "enableExperimentalSEPtrsOperations", "enable-experimental-se-ptrs-operations",
            "bool", "false",
            "[Optional] Set ExperimentalSEPtrsOperations to true/false"
        >,
        Option<
            "enableQDQOptimizationAggressive", "enable-qdq-optimization-aggressive",
            "bool", "false",
            "[Optional] Set QDQOptimizationAggressive to true/false"
        >,
        Option<
            "enableAdaptiveStripping", "enable-adaptive-stripping",
            "bool", "false",
            "[Optional] Set AdaptiveStripping to true/false"
        >,
        Option<
            "enableExtraStaticShapeOps", "enable-extra-static-shape-ops",
            "bool", "true",
            "[Optional] Set ExtraStaticShapeOps to true/false"
        >,
        Option<
            "enableSprLUT", "enable-sprlut",
            "bool", "false",
            "[Optional] Set sprLUT to true/false"
        >,
        Option<
            "enableDCIM", "enable-dcim",
            "bool", "false",
            "[Optional] Set EnableDCIM for NCE to true/false"
        >,
        Option<
            "allowCustomValues", "allow-custom-values",
            "bool", "",
            "[Optional] Allows keep predefined values in IR"
        >,
        Option<
            "enableAsymmetricPerTensorZP", "enable-asymmetric-per-tensor-zp",
            "bool", "false",
            "[Optional] Enable asymmetric weights with per tensor zp to run on NCE without dequantization"
        >,
        Option<
            "enableAsymmetricPerChannelZP", "enable-asymmetric-per-channel-zp",
            "bool", "false",
            "[Optional] Enable asymmetric weights with per channel zp to run on NCE without dequantization"
        >,
        Option<
            "enableProfiling", "enable-profiling",
            "bool", "false",
            "Enable profiling"
        >,
        Option<
            "enableWeightsDynamicDequantization", "enable-weights-dynamic-dequantization",
            "bool", "false",
            "[Optional] Set WeightsDynamicDequantization to true/false"
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SetupTilingConstraint
//

def SetupTilingConstraint : PassBase<"setup-tiling-contraint", "vpux::ModulePass"> {
    let summary = "Set tiling related constraint by arch in the model/IR";

    let description = [{
        Uses the PipelineOptionsOp structure from the IR so the constant could be exposed globally. The constraint includes:
        - LargeFilterFragmentationThreshold: threshold to avoid memory fragmentation due to large filter size
    }];

    let constructor = "vpux::VPU::createSetupTilingConstraintPass()";

    let options = [
        Option<
            "allowCustomValues", "allow-custom-values",
            "bool", "",
            "[Optional] Allows keep predefined values in IR"
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ConvertOpToDMAForPerformantExecution
//

def ConvertOpToDMAForPerformantExecution : PassBase<"convert-op-to-dma-for-performant-execution", "vpux::FunctionPass"> {
    let summary = "Adapt and move ops to DMA if possible and efficient";

    let description = [{
        If op can be done with DMA efficiently this pass will, convert the op to DMA related op.
        Only added to VPU40XX+ pipelines as only op converted via this pass is Gather and is only supported
        on VPU40XX+ pipelines.

        Currently GatherOp -> GatherDMAOp is supported ( on LNL and newer architectures).
        Gather DMA has some limitation, the input size after axis should be less than 4096.
        On Adaptation side the pass will tile gather on the dimension after gather axis.
        Gather -> Reshape + Convert + (Optional) Copy + GatherDMA + Reshape + Copy.
    }];

    let constructor = "vpux::VPU::createConvertOpToDMAForPerformantExecutionPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

#endif

//
// RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass
//

def RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass : PassBase<"remove-output-sparse-to-avoid-suboptimal-dpu-workloads", "vpux::FunctionPass"> {
    let summary = "Remove output sparsity for SOK layer to avoid suboptimal dpu workloads";

    let description = [{
        This pass removes SOK layer's output sparsity if
        1. SOK layer has different split sizes on clusters excluding the last one. For example, we need to split OC = 128 on 6 tiles,
        the tiled size will be {32, 32, 16, 16, 16, 16}. If there's output sparsity, we need to split 32 into two pieces of 16 because
        we must have the same workload channel excluding the last one. However, two workloads with 16 channels have much worse
        performance than a workload with 32 channels. If there's no sparsity, we can keep the workload with 32 channels.
        This is only relevant to the senario with more than 2 tiles so it doesn't exist for NPU37XX.
        2. SOK layer's output is used by `VPU.Concat`

        Conv1_1 (OC = 256, SOK)  Conv1_2 (OC = 256, SOK)
             \                               /
                         Concat on C
                              |
                            Conv2

       Take above graph as an example, we need to split OC = 256 on 6 tiles, the tiled size will be {48, 48, 48, 48, 48, 16}.
       After concatenation, the combined workloads will be {48, 48, 48, 48, 48, 16, 48, 48, 48, 48, 48, 16}. If there's output sparsity for Conv1_1 and Conv1_2,
       we need to split 48 into three pieces of 16 because we must have the same workload channel excluding the last one. If there's no sparsity, we can
       keep the workload with 48 channels.
    }];

    let constructor = "vpux::VPU::createRemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// AddExplicitPaddingBeforeNCEPermute
//

def AddExplicitPaddingBeforeNCEPermute : PassBase<"add-explicit-padding-before-nce-permute", "vpux::FunctionPass"> {
    let summary = "Insert explicit padding before NCE Permute.";

    let description = [{
        When output element type of NCEPermuteOp is FP16 and expandedChannels is greater than Input Channels,
        there is no explicit padding done here, and when the consumer op tries to use the whole output tensor
        of the NCEPermuteOp we don't know if inside the memory for padding we have something allocated or
        we have NaN values. This will cause accuracy issues. This pass insert explicit padding in order to
        avoid such cases.
    }];

    let constructor = "vpux::VPU::createAddExplicitPaddingBeforeNCEPermutePass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

def LegalizeDynamicShapeConcatForSWLayers : PassBase<"legalize-dynamic-shape-concat-for-sw-layers", "vpux::FunctionPass"> {
    let summary = "Unroll a `VPU.Concat` with more than two dynamic operands.";

    let description = [{
        Express `VPU.Concat` with more then two dynamic inputs as a chain of concatenations.
        For example:
        ```
            %0 = VPU.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]]}
        ```
        becomes
        ```
            %0 = VPU.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]}
            %1 = VPU.Concat(%0, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]]}
        ```
        This transformation is necessary because a concatenation with dynamic inputs
        is executed as an activation shave kernel. The kernel expects only two inputs.
        This pass expresses a `VPU.Concat` that cannon be lowered into the kernel with
        a chain of concatenations that can be lowered.
    }];

    let constructor = "vpux::VPU::createLegalizeDynamicShapeConcatForSWLayersPass()";

    let options = [
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}
//
// ConvertConstArgsToMultiConstants
//

def ConvertConstArgsToMultiConstants : PassBase<"convert-const-args-to-multi-constants", "vpux::ModulePass"> {
    let summary = "Converts constant arguments of outlined functions to 'const.MultiDeclare' ops";

    let description = [{
        When an outlined function's argument value comes from a 'const.Declare' op on all call sites, this pass
        will move that 'const.Declare' op to a 'const.Rodata' op in the global scope and replace all occurances
        of the argument value in the function body with a 'const.MultiDeclare' op. Finally, the argument is
        deleted from the call site and the function signature.

        Example:
        ```MLIR
        func.func @main_fn1(%arg0: tensor<48x48x3x3xf32>, %arg1: tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32> {
            %0 = VPU.Add(%arg0, %arg1) : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
            return %0 : tensor<48x48x3x3xf32>
        }

        func.func @main(...) {
            ...
            %cst = const.Declare tensor<48x48x3x3xf32> = dense<1.0> : tensor<48x48x3x3xf32>
            %0 = call @main_fn1(%input, %cst) : (tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
            ...
        }
        ```
        would get converted into
        ```MLIR
        const.Data @Data {
            const.Rodata @rodata_0 dense<1.0> : tensor<48x48x3x3xf32>
        }

        const.BundleData @BundleData {
            const.RodataBundle @bundle_0 = [@Data::@rodata_0] : tensor<48x48x3x3xf32>
        }

        func.func @main_fn1(%arg0: tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32> {
            %mcst = const.MultiDeclare tensor<48x48x3x3xf32> = @BundleData::@bundle_0 : tensor<48x48x3x3xf32>
            %0 = VPU.Add(%arg0, %mcst) : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
            return %0 : tensor<48x48x3x3xf32>
        }

        func.func @main(...) {
            ...
            %0 = call @main_fn1(%input) : (tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
            ...
        }
        ```
    }];

    let constructor = "vpux::VPU::createConvertConstArgsToMultiConstantsPass()";

    let options = [
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}



//
// ConcatRepeatingBlocksOutlining
//

def ConcatRepeatingBlocksOutlining : PassBase<"concat-repeating-blocks-outlining", "vpux::ModulePass"> {
    let summary = "Outline concats with repeating input branches into functions";

    let description = [{
        Identifies concat operations that has identical input branches and outlines them into functions.
        For example, the following concat could be a candidate for outlining:
            A   A
            |   |
            B   B
            \   /
            Concat

        The pass can be configured with the following options:
        - min-seq-length: minimum number of compute operations that should be found on an input branch
        - single-function-per-concat: decides whether to outline each input branch into an individual function (false)
                                      or outline all input branches and the concat into a single function (true)
    }];

    let constructor = "vpux::VPU::createConcatRepeatingBlocksOutliningPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "minSeqLength", "min-seq-length",
            "int", "",
            "Minimum threshold for the length of an input branch"
        >,

        Option<
            "singleFunctionPerConcat", "single-function-per-concat",
            "bool", "true",
            "Outline a concat operation into a single function along with its branches"
        >,
    ];
}

//
// OutlineEntireMainContent
//

def OutlineEntireMainContent : PassBase<"outline-entire-main-content", "vpux::ModulePass"> {
    let summary = "Outline all operations from main into functions if other outlining instances exist";

    let description = [{
        In case some operations were already outlined before calling this pass, all the remaining
        operations from main will be outlined as well. The end-result will have main only containing
        call operations (and the return terminator).

        This is done to work-around a limitation in the scheduler, as it does not support scenarios
        where call operations are mixed with non-call operations.

        For cases where an input / output of a new function is set in CMX, copies to DDR are introduced
        so that all values passed between functions are in DDR. This is to accomodate another limitation
        in the scheduler, where each function / call operation must work with DDR values.
        For example, if the following Convolution is targeted for outlining in this pass:
            func @main(...) {
                ...
                %conv = VPU.NCEConvolution(...) -> !VPU.DistributedTensor<..., @CMX_NN, ...>
                ...
            }
        the resulting IR would contain:
            func @conv(...) -> tensor<..., @DDR> {
                %conv = VPU.NCEConvolution(...) -> !VPU.DistributedTensor<..., @CMX_NN>
                %copy = VPU.Copy(%conv) -> tensor<..., @DDR>
                return %copy
            }
            func @main(...) {
                ...
                call @conv(...) -> tensor<..., @DDR>
                ...
            }
    }];

    let constructor = "vpux::VPU::createOutlineEntireMainContentPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

def ConstructWsAnalysis : PassBase<"construct-ws-analysis", "vpux::ModulePass"> {
    let summary = "Create weights separation analysis";

    let description = [{
        This pass constructs weights separation analysis object used across
        weights separation passes.

        Note: The analysis is preserved until --destruct-ws-analysis pass.
    }];

    let constructor = "vpux::VPU::createConstructWsAnalysisPass()";
}

def DestructWsAnalysis : PassBase<"destruct-ws-analysis", "vpux::ModulePass"> {
    let summary = "Destroy weights separation analysis";

    let description = [{
        This pass destroys previously created weights separation analysis
        object.
    }];

    let constructor = "vpux::VPU::createDestructWsAnalysisPass()";
}

def IntroduceInitFunction : PassBase<"introduce-init-function", "vpux::ModulePass"> {
    let summary = "Introduce the `init` function into the IR";

    let description = [{
        IntroduceInitFunctionPass creates the init() function in the IR. The pass maps the transformation attributes of
        const.Declare ops to IE-level ops, so they can be executed at inference time on the accelerator.

        Every constants can have a list of transformation attributes. Most of these
        transformation attributes can be mapped to combinations IE-dialect level operations.
        A trivial example is
        ```MLIR
            %cst = const.Declare tensor<4x4xf32> = dense<[1, 2, ..., 16]> : tensor<4x4xf32>, [#const.Add<1.0 : f32>]
        ```
        , which can be mapped to
        ```MLIR
            %original = const.Declare tensor<4x4xf32> = dense<[1, 2, ..., 16]>
            %one = const.Declare tensor<1x1xf32> = dense<1.0> : tensor<1x1xf32>
            %1 = IE.Add(%original, %one) {...} : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
        ```
        The IntroduceInitFunctionPass packs these IE operations into a new init() function:
        ```MLIR
            func @init(%arg : tensor<4x4xf32>) -> tensor<4x4xf32> {
                %one = const.Declare tensor<1x1xf32> = dense<1.0> : tensor<1x1xf32>
                %0 = IE.Add(%arg, %one) {...} : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
                return %0 : tensor<4x4xf32>
            }

            %original = const.Declare tensor<4x4xf32> = dense<[1, 2, ..., 16]>
            %0 = call @init(%original) : (tensor<4x4xf32>) -> tensor<4x4xf32>
        ```

        This pass comes with 3 modes: GenMain, GenInit and GenAll. Their meaning is explained below.

        GenMain (generate main):
            This mode allows the user to produce a blob that only contains the main()-schedule.
            The network IO info is updated accordingly. Uses of the candidate DeclareOps in main() are replaced by
            the newly associated function inputs. Any transformations that do *not* belong into init() (e.g. SubView)
            are inserted between associated function argument and its use in main().

        GenInit (generate init):
            This mode allows the user to produce a blob that only contains the init()-schedule.
            The main() function is erased. The network IO info is set to reflect the function signature of init().

        Additionally, the pass also acknowledges a special memory limit option (see pass options).
        The memory limit is a threshold that controls how much memory an init schedule uses for input and output
        arguments (note: any additional memory allocated during schedule execution is not accounted for). When the
        threshold is exceeded, the init schedule is split into (several) "parts". Note that init generation procedure
        guarantees that every given input constant (e.g. OV weight) is only processed by a single init schedule part,
        regardless of the specified memory limit. This in turn guarantees that once a particular init schedule part is
        run, all inputs of that part are no longer needed by any other init schedule part and thus could be safely
        discarded (e.g. deallocated by the caller of the init schedule).
    }];

    let options = [
        Option<
            "wsExtractionMode", "ws-extraction-mode",
            "std::string", [{"gen-init"}],
            "Can be one of the following values: 'gen-init', 'gen-main'"
        >,
        Option<
            "initPart", "init-part", "int64_t", "-1", "Specifies which part of"
            "the init schedule to generate when it is split by the memory"
            "limit. Note that this option only works in 'gen-init' mode."
        >,
        Option<
            "memoryLimit", "memory-limit", "int64_t",
            "std::numeric_limits<int64_t>::max()", "Specifies the memory limit"
            "(in bytes) that init schedule must acknowledge when splitting"
            "inits into parts in 'gen-init' mode."
        >
    ];

    let constructor = "vpux::VPU::createIntroduceInitFunctionPass()";
}

def QueryWSInfo : PassBase<"query-ws-info", "vpux::ModulePass"> {
    let summary = "Analyze IR and store weights separation meta information";

    let description = [{
        This pass analyzes IR and produces weights separation meta information
        that is stored into the attributes of the entry point function. One
        could then access this information via the attributes. The following
        information is stored:

        * VPU.WsTotalInitPartCount - the amount of init parts for the specified
          memory limit (see options for details)
    }];

    let options = [
        Option<
            "memoryLimit", "memory-limit", "int64_t",
            "std::numeric_limits<int64_t>::max()", "Specifies the memory limit"
            "(in bytes) that init schedule must acknowledge when splitting"
            "inits into parts. This option's value MUST be aligned to the same"
            "option's value in --introduce-init-function pass"
        >
    ];

    let constructor = "vpux::VPU::createQueryWSInfoPass()";
}

def ConcatInitInputs : PassBase<"concat-init-inputs", "vpux::ModulePass"> {
    let summary = "Concatenates init function inputs into a single tensor";

    let description = [{
        This pass concatenates the inputs of init function into a single tensor,
        reducing the overall number of weights passed to init to a single
        concatenated weight (note: it is currently highly preferred to limit the
        amount of inputs, when possible, as every schedule input uses CMX-placed
        meta information).
    }];

    let constructor = "vpux::VPU::createConcatInitInputsPass()";
}

def ConcatInitResults : PassBase<"concat-init-results", "vpux::ModulePass"> {
    let summary = "Concatenates init function results into a single tensor";

    let description = [{
        This pass concatenates the results of init function into a single
        tensor, reducing the overall number of I/O between init and main
        schedules (note: it is currently highly preferred to limit the amount of
        inputs of main schedule, when possible, as every main schedule input
        uses CMX-placed meta information).

        Consequently, the pass also converts the single input (coming from init
        result) into correct multiple inputs inside the main schedule.
    }];

    let options = [
        Option<
            "wsExtractionMode", "ws-extraction-mode",
            "std::string", [{"gen-init"}],
            "Can be one of the following values: 'gen-init', 'gen-main'"
        >,
        Option<
            "initPart", "init-part", "int64_t", "-1", "Specifies which part of"
            "the init schedule to use when it is split by the memory limit."
            "Note that this option only works in 'gen-init' mode."
        >,
        Option<
            "memoryLimit", "memory-limit", "int64_t",
            "std::numeric_limits<int64_t>::max()", "Specifies the memory limit"
            "(in bytes) that init schedule must acknowledge when splitting"
            "inits into parts in 'gen-init' mode."
        >
    ];

    let constructor = "vpux::VPU::createConcatInitResultsPass()";
}

//
// BoundedTensorsToDynamicDimsMask
//

def BoundedTensorsToDynamicDimsMask : PassBase<"bounded-tensors-to-dynamic-dims-mask", "vpux::ModulePass"> {
    let summary = "Convert dynamic tensors with upper bounds to a static tensor with upper-bounds as shape";

    let description = [{
        Change dynamic tensor with upper-bound attribute from
            tensor<1x2x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 64, 512]>: tensor<4xsi64>}>
        to
            tensor<1x2x35x512xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 0]: tensor<4xsi64>}>

        Makes it possible to do cluster tiling on static upper-bounded shape by hiding dynamic nature of the tensor.
    }];

    let constructor = "vpux::VPU::createBoundedTensorsToDynamicDimsMaskPass()";

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::math::MathDialect",
        "mlir::linalg::LinalgDialect",

        "vpux::VPU::VPUDialect"
    ];
}

//
// SCF compute outlining
//

def ScfComputeOpsOutlining : PassBase<"scf-compute-ops-outlining", "vpux::ModulePass"> {
    let summary = "Outlines operations from the bodies of scf dialect operations";

    let description = [{
        The host dynamism approach separates scheduling (host) and compute functions in the main function.
        This pass moves all compute operations from the host main function into separate functions and
        rewrites the main function to call these newly created compute functions.

        The following function:

        func.func @main (%arg0) {
            %1 = scf.for ... {
                %2 = tensor.extract_slice
                %3 = scf.if %2 -> ... {
                    %4 = VPU.OP1
                    scf.yield
                } else {
                    %5 = VPU.OP2
                    scf.yield
                }
                scf.yield
            }
        }

        Will be outlined into:

        func.func @compute_func0 (%arg) {
            %0 = VPU.OP1
            return %0
        }

        func.func @compute_func1 (%arg) {
            %0 = VPU.OP2
            return %0
        }

        func.func @main (%arg0) {
            %1 = scf.for ... {
                %2 = tensor.extract_slice
                %3 = scf.if %2 -> ... {
                    %4 = call @compute_func0
                    scf.yield
                } else {
                    %5 = call @compute_func1
                    scf.yield
                }
                scf.yield
            }
        }
    }];

    let constructor = "vpux::VPU::createScfComputeOpsOutliningPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "mlir::scf::SCFDialect",
        "mlir::tensor::TensorDialect"
    ];
}

//
// FinalizeComputeFunctionBoundaries
//

def FinalizeComputeFunctionBoundaries : PassBase<"finalize-compute-function-boundaries", "vpux::ModulePass"> {
    let summary = "Finalize compute function boundaries for Host Pipeline";

    let description = [{
        This pass finalizes the compute function boundaries for the Host Pipeline.

        It ensures that all compute functions are properly outlined and that their
        boundaries are correctly defined. After executing this pass, @main function
        can contain only ops/types that can be lowered to something executable
        on the host.

        The pass performs several key transformations:

        1. NPU-specific encoding is preserved only inside compute functions
        2. NPU-specific attributes (layout, memory space, bounds) are stripped from
           tensor types in function signatures and @main function
        3. Tensor types are transformed to be compatible with upstream MLIR components
        4. Necessary bridging operations (like Core.ReinterpretCast) are inserted at
           function boundaries

        Important contract established by this pass:
        - NPU-specific information (layout, bounds) is preserved INSIDE compute functions
        - Function signatures and @main function use "clean" tensor types without NPU-specific attributes
        - For dynamic shapes with bounds, the bounds information is preserved only within
          compute functions but not in the function signatures or @main


        Examples of tensor type transformations:

        ```mlir
        // Static shape with layout
        // Before: NPU-specific tensor with NHWC layout
        tensor<1x16x720x1000xf16, {order = #NHWC}>

        // After: Standard tensor with default layout
        tensor<1x720x1000x16xf16>

        // Dynamic shape with bounds
        // Before: NPU-specific tensor with bounds and NHWC layout
        tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

        // After: Standard tensor without bounds, layout preserved through dimension reordering
        tensor<1x256x?x16xf16>
        ```

        Shape manipulation operations like `tensor.extract_slice` and `tensor.insert_slice` are special cases
        since they contain explicit offsets, sizes, and strides as operands. These operands must be permuted
        to match the new tensor layout:

        ```mlir
        // Before: extract_slice with NHWC ordering
        %0 = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 16, 90, 1000] [1, 1, 1, 1]
            : tensor<1x16x90x1000xf16, {order = #NHWC}>
            to tensor<1x16x90x1000xf16, {order = #NHWC}>

        // After: extract_slice with standard ordering, dimensions permuted
        %0 = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
            : tensor<1x90x1000x16xf16> to tensor<1x90x1000x16xf16>
        ```

        The pass bridges the host-NPU boundary by inserting Core.ReinterpretCast operations:

        ```mlir
        func.func private @main_func0(%arg0: tensor<1x256x?x16xf16>) -> tensor<1x256x?x16xf16> {
            // Convert from "clean" tensor to NPU-specific tensor with layout and bounds
            %0 = Core.ReinterpretCast(%arg0) : tensor<1x256x?x16xf16> ->
                tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

            // NPU operations with full NPU-specific type information
            %1 = VPU.NCE.Eltwise(%0, %0) {...} ->
                tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

            // Convert back to "clean" tensor for function return
            %2 = Core.ReinterpretCast(%1) :
                tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> ->
                tensor<1x256x?x16xf16>

            return %2 : tensor<1x256x?x16xf16>
        }
        ```
    }];

    let constructor = "vpux::VPU::createFinalizeComputeFunctionBoundariesPass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::scf::SCFDialect",
        "mlir::tensor::TensorDialect",
        "mlir::arith::ArithDialect",
        "vpux::VPU::VPUDialect"

    ];
}

//
// MoveReflectPadToCMX
//

def MoveReflectPadToCMX : PassBase<"move-reflect-pad-to-cmx", "vpux::FunctionPass"> {
    let summary = "Moves Reflect Pad ops to CMX";

    let description = [{
        This pass moves the Reflect Pad patterns to CMX for better performance. The patterns are formed by Subview ops,
        Copy ops and each ending with a ConcatView op. The input of the pattern that does padding on a certain dimension will be moved
        from DDR to CMX, all the ops forming the pattern will be executed in CMX, the output of the ConcatView op
        will be moved from CMX to DDR. The same will happen for each dimension. The copies are expected to be later optimized by OptimizeCopies.
    }];

    let constructor = "vpux::VPU::createMoveReflectPadToCMXPass()";
}

//
// ConvertDynamicToStaticKernels
//

def ConvertDynamicToStaticKernels : PassBase<"convert-dynamic-to-static-kernels", "vpux::ModulePass"> {
    let summary = "Converts compute function ops with dynamic tensor shapes to static shapes";

    let description = [{
        The pass identifies scf.for loops where the iteration space is determined by dynamic dimensions of input tensors for compute functions.
        Using the step size of the loop and tensor slice operations, the pass will convert the dynamic tensor shapes
        to static shapes by using index adjustments through backtracking. NPU compute function / kernel will be transformed
        to use static shapes instead of dynamic shapes, based on the static shape inferred from the previous step. For the cases
        where backtracking is not possible, the pass throws a runtime error. Pass currently does not support padding of the input tensors.

        This conversion enables better optimization and scheduling as static shapes allow for
        more effective tiling and resource allocation during compilation.
    }];

    let constructor = "vpux::VPU::createConvertDynamicToStaticKernelsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "mlir::tensor::TensorDialect",
        "mlir::arith::ArithDialect",
        "mlir::scf::SCFDialect",
        "mlir::cf::ControlFlowDialect",
    ];
}

//
// AdjustForOptimizedLayers
//

def AdjustForOptimizedLayers : PassBase<"adjust-for-optimized-layers", "vpux::FunctionPass"> {
    let summary = "Adjust layers shape or layout to utilize more compute engines or leverage optimized kernel implementation";

    let description = [{
        The pass adjusts layers shape or layout to utilize more compute engines and better leverage optimized SHAVE
        implementation
        Supported Optimizations:
        - Softmax:
          * Axis0: Adjust shape to leverage the optimized kernel for Softmax with axis=0
          * MultiSHAVEs: Adjust shape to gather dimensions on the tile dimension to utilize more SHAVE engines
        - Gelu & Multiply:
          * MultiSHAVEs & MultiClusters: Adjust shape to gather dimensions on the tile dimension to utilize more
          SHAVE engines and avoid Clustering strategy due to batch size
         - NCEPermute:
          * MultiClusters: Adjust shape to ensure there's enough dim size on height for MC split to utilize more DPU engines
    }];

    let constructor = "vpux::VPU::createAdjustForOptimizedLayersPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SplitRealDFTOpsPass
//

def SplitRealDFTOps : PassBase<"split-real-dft-ops", "vpux::FunctionPass"> {
    let summary = "Replace RDFT and IRDFT operations with a subgraph of smaller operations";

    let description = [{
        Replace RDFT and IRDFT operations with a subgraph of smaller operations.
        VPU.RDFT = {VPU.RDFTUncutOp->VPU.SliceOp}
        VPU.IRDFT = {VPU.IDFTOp->VPU.IRDFTLastAxisOp}
    }];

    let constructor = "vpux::VPU::createSplitRealDFTOpsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// DecomposeMVN
//

def DecomposeMVN : PassBase<"decompose-mvn", "vpux::FunctionPass"> {
    let summary = "Decompose MVN into 3 separate tiled functions";

    let description = [{
        The pass can Decompose MVN into 3 separate tiled functions when can't fit into CMX.
                IN -> MVN -> OUT
            will be decompose in:
                IN -> MVN1SumOp -> MVN1MeanVarOp -> MVN1NormalizeOp -> OUT
                    \_____________________________/
    }];

    let constructor = "vpux::VPU::createDecomposeMVNPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ApplyTilingMVN1Sum
//

def ApplyTilingMVN1Sum : PassBase<"apply-tiling-mvn1sum", "vpux::FunctionPass"> {
    let summary = "Tiling for mvn1_sum op";

    let description = [{
        MVN1SumOp tiling will be done in this pass. Because the current logic is to tile the output and after to back infer
        to the input. Output tensor size is NxCx2 and it can be 1x1x2, it is imposible to tile and back infer in this case.
        MVN1MeanVarOp does not need tiling because tensor is NxCx2.
    }];

    let constructor = "vpux::VPU::createApplyTilingMVN1SumPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "tilingMode", "tiling-mode",
            "std::string", [{"PREFETCH"}],
            "[Optional] Set tiling mode as `ISOLATED` or `PREFETCH`"
        >
    ];
}

//
// FuseM2IOps
//

def FuseM2IOps : PassBase<"fuse-m2i-ops", "vpux::FunctionPass"> {
    let summary = "Fuse M2I ops to VPU::M2ITask";

    let constructor = "vpux::VPU::createFuseM2IOpsPass()";

    let dependentDialects = [
       "vpux::VPU::VPUDialect"
    ];
}

//
// ConvertM2IOps
//

def ConvertM2IOps : PassBase<"convert-m2i-ops", "vpux::FunctionPass"> {
    let summary = "Convert individual M2I ops to VPU::M2ITask";

    let constructor = "vpux::VPU::createConvertM2IOpsPass()";

    let dependentDialects = [
       "vpux::VPU::VPUDialect"
    ];
}

//
// ComputeNCEInputWorkloads
//

def ComputeNCEInputWorkloads : PassBase<"compute-nce-input-workloads", "vpux::FunctionPass"> {
    let summary = "Compute input workloads start/end";

    let description = [{
        This pass will compute the input workloads start/end for each VPU.DPU.WorkloadOp, based on
        the output workloads' start/end.

        For arch 40XX, in cases where there are multiple consumers with different halo requirements
        from the parent op, we can select the largest halo size for the parent op, but we need to set
        the input workloads accordingly in the consumer ops.

        For example, an NCE op needs to have its input start at offset X on height to be able to compute
        its first output line. Due to kernel requirements, another NCE op that shares the same parent needs
        to have its input start at X - 2 height line. This situation can be resolved by having the previous
        NCE op put its output starting with line X - 2 and have the first NCE op set its inStart = 2.
    }];

    let constructor = "vpux::VPU::createComputeNCEInputWorkloadsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// MoveConvertAroundViewLikeOps
//

def MoveConvertAroundViewLikeOps : PassBase<"move-convert-around-viewlike-ops", "vpux::FunctionPass"> {
    let summary = "Move ConvertOp around Viewlike ops";

    let description = [{
        This is the base pass which contains all rewriters for moving convertOp around in the IR
        MoveConvertAfterPermuteCast -> Moves ConvertOp after PermuteCast so we can fuse the CopyOp and ConvertDMAOp and avoid DDR DMAs
        MoveConvertAfterShapeCast -> Moves ConvertOp after ShapeCastOp so we can fuse the CopyOp and ConvertDMAOp and avoid DDR DMAs
        MoveConvertAffineReshape -> Moves ConvertOp before AffineReshape with all 4 Dims
    }];

    let constructor = "vpux::VPU::createMoveConvertAroundViewLikeOpsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ConvertVPUOpsToUpstreamOps
//

def ConvertVPUOpsToUpstreamOps : PassBase<"convert-vpuops-to-upstream", "vpux::ModulePass"> {
    let summary = "Converts VPU operations to equivalent ones in MLIR upstream";

    let description = [{
        The pass converts operations in the compiler which might be represented as
        existing operations from MLIR upstream taking into consideration possible restrictions.
        For instance:
        VPU.SliceOp -> tensor.extract_slice (applicable for not sparse tensors only)
    }];

    let constructor = "vpux::VPU::createConvertVPUOpsToUpstreamOpsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "mlir::tensor::TensorDialect"
    ];
}

//
// RestorePadAttrAfterSCFTiling
//

def RestorePadAttrAfterSCFTiling : PassBase<"restore-tensor-pad-after-scf-tiling", "vpux::FunctionPass"> {
    let summary = "Restores tensor padding after SCF tiling";

    let description = [{
        In the HostCompile pipeline, the SCF tiling pass removes the padding attribute from convolution operations
        and replaces the implicit padding with explicit padding operations. Before outlining NCE operations in the
        host pipeline, we need to fold the padding into corresponding operations to avoid costly compute splits between
        NPU and CPU. This pass restores the padding information for convolution operations and removes tensor.pad operations.

    }];

    let constructor = "vpux::VPU::createRestorePadAttrAfterSCFTilingPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "mlir::tensor::TensorDialect",
        "mlir::arith::ArithDialect",
        "mlir::scf::SCFDialect",
    ];
}

//
// FlashSDPATilingStrategyEstimation
//

def FlashSDPATilingStrategyEstimation : PassBase<"flash-sdpa-tiling-strategy-estimation", "vpux::FunctionPass"> {
    let summary = "Computes Query and Key/Value tiling parameters";

    let description = [{
        Compute tiling dimensions for both Query and Key/Value tensors.
        Assign Key/Value tiling attribute to propagate it to the UnrollFlashSDPAPass.
        Isolated tiling algorithm will tile VPU.FlashSDPAOp on Query.
        The Key/Value tiling attribute is used to estimate CMX requirement for the final tiled operation.
    }];

    let constructor = "vpux::VPU::createFlashSDPATilingStrategyEstimationPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

// FuseLastViewLikeOp
//

def SCFFuseLastViewLikeOp : PassBase<"scf-fuse-last-viewlike-op", "vpux::FunctionPass"> {
    let summary = "Fuses the last view-like op into the preceding scf.for loop";

    let description = [{
        View ops placed after the last compute op in an scf.for loop will remain outside
        of it after the SCF VF pass. That will however lead to errors later on.

        This pass takes the final view ops and integrates them in the scf.for loop.
        Currently supporting only PermuteCast fusion.
    }];

    let constructor = "vpux::VPU::createSCFFuseLastViewLikeOpPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// UnrollFlashSDPA
//

def UnrollFlashSDPA : PassBase<"unroll-flash-sdpa", "vpux::FunctionPass"> {
    let summary = "Tiles VPU.FlashSDPA on Key and Value tensors";

    let description = [{
        Unroll VPU.FlashSDPAOp based on the pre-computed Key/Value tiling attribute.
        Propagate RunningMax, RunningSum and RunningOutput through each operation.
    }];

    let constructor = "vpux::VPU::createUnrollFlashSDPAPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// AdjustBlockSizeForScfTiling
//

def AdjustBlockSizeForScfTiling : PassBase<"adjust-block-size-for-scf-tiling", "vpux::ModulePass"> {
    let summary = "Adjusts block sizes for SCF tiling";

    let description = [{
        In the Host dynamism approach, the ApplyTiling pass tiles over output tensor dimensions using scf.for loops.
        The resulting block sizes for output and input tensors may not have fixed sizes, depending on the tensor dimensions
        and step sizes. This pass adjusts the block sizes for input and output tensors to static sizes by backtracking the
        index for the last tile.

        The pass does not handle scenarios where there are not enough elements to perform backtracking.
    }];

    let constructor = "vpux::VPU::createAdjustBlockSizeForScfTilingPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "mlir::tensor::TensorDialect",
        "mlir::arith::ArithDialect",
        "mlir::scf::SCFDialect",
        "mlir::cf::ControlFlowDialect",
    ];
}
