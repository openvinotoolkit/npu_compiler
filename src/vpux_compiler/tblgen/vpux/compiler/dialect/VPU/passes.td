//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
        initializes **IERT Dialect** run-time resources information.
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
        "vpux::IERT::IERTDialect",
        "vpux::VPU::VPUDialect"
    ];
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
// WrapDistributedOpsInNCEClusterTiling
//

def WrapDistributedOpsInNCEClusterTiling : PassBase<"wrap-distributed-ops-in-nceclustertiling", "vpux::FunctionPass"> {
    let summary = "Wraps operations that have distributed types in NCEClusterTiling";

    let description = [{
        This pass finds operations that have distributed types as inputs or outputs and wraps them into NCEClusterTiling.
        Only VPU.ClusteredOpInterface and VPU.Copy operations are covered, so that view-like operations can still use distributed types.
    }];

    let constructor = "vpux::VPU::createWrapDistributedOpsInNCEClusterTiling()";

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
        1. Number of operations which might increase computational cost does not exceed limit
        2. All operations have same multicluster strategy or don't have them at all
        3. Region which is supposed to be added doesn't have any other users except current region or
        all its users point to current region too.
        4. All operations in new region after merging fit in CMX when they are tiled for VF. In case they don't, number of tiles
        increases.
        5. Required CMX memory by constant weights shouldn't exceed the threshold to avoid spilling.
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
        into CMX and before the operation is wrapped into NCEClusterTiling.
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
        >
    ];

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
            "workloadManagementEnable", "workload-management-enable",
            "bool", "false",
            "[Optional] Set partial workload management to true/false"
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
// SetupChannelsAutoPadding
//

def SetupChannelsAutoPadding : PassBase<"setup-channels-auto-padding", "vpux::ModulePass"> {
    let summary = "Enable the auto padding for input/output channels and store it in the model/IR";

    let description = [{
        Uses the PipelineOptionsOp structure from the IR so the options could be exposed globally
    }];

    let constructor = "vpux::VPU::createSetupChannelsAutoPaddingPass()";

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
// SetupIsReduceSupported
//

def SetupIsReduceSupported : PassBase<"setup-is-reduce-supported", "vpux::ModulePass"> {
    let summary = "Enable Reduce operations to be supported on NCE";

    let description = [{
        Uses the PipelineOptionsOp structure from the IR so the options could be exposed globally
    }];

    let constructor = "vpux::VPU::createSetupIsReduceSupportedPass()";

    let options = [
        Option<
            "enableIsReduceSupported", "enable-is-reduce-supported",
            "bool", "false",
            "[Optional] Set IsReduceSupported for NCE to true/false"
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
// SetupEnableSEPtrsOperations
//

def SetupEnableSEPtrsOperations : PassBase<"setup-enable-se-ptrs-operations", "vpux::ModulePass"> {
    let summary = "Enable SEPtrs operations to be supported";

    let description = [{
        Uses the PipelineOptionsOp structure from the IR so the options could be exposed globally
    }];

    let constructor = "vpux::VPU::createSetupEnableSEPtrsOperationsPass()";

    let options = [
        Option<
            "enableSEPtrsOperations", "enable-se-ptrs-operations",
            "bool", "false",
            "[Optional] Set SEPtrsOperations to true/false"
        >,
        Option<
            "allowCustomValues", "allow-custom-values",
            "bool", "",
            "[Optional] Allows keep predefined values in IR"
        >,
        Option<
            "enableExperimentalSEPtrsOperations", "enable-experimental-se-ptrs-operations",
            "bool", "false",
            "[Optional] Set ExperimentalSEPtrsOperations to true/false"
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SetupEnableAdaptiveStripping
//

def SetupEnableAdaptiveStripping : PassBase<"setup-enable-adaptive-stripping", "vpux::ModulePass"> {
    let summary = "Enable adaptive stripping";

    let description = [{
        Uses the PipelineOptionsOp structure from the IR so the options could be exposed globally
    }];

    let constructor = "vpux::VPU::createSetupEnableAdaptiveStrippingPass()";

    let options = [
        Option<
            "enableAdaptiveStripping", "enable-adaptive-stripping",
            "bool", "false",
            "[Optional] Set AdaptiveStripping to true/false"
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
// ConvertOpToDMAForPerformantExecution
//

def ConvertOpToDMAForPerformantExecution : PassBase<"convert-op-to-dma-for-performant-execution", "vpux::FunctionPass"> {
    let summary = "Move ops to DMA if possible and effiecient";

    let description = [{
        If op can be done with DMA efficiently this pass will, convert the op to DMA related op.
        Only added to NPU40XX pipelines as only op converted via this pass is Gather and is only supported
        on NPU40XX pipelines.

        Currently GatherOp -> GatherDMAOp is supported ( on NPU40XX and newer architectures)
        Gather -> Reshape + Convert + (Optional) Copy + GatherDMA + Reshape + Copy
    }];

    let constructor = "vpux::VPU::createConvertOpToDMAForPerformantExecutionPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// TileGather
//

def TileGather : PassBase<"tile-gather", "vpux::FunctionPass"> {
    let summary = "Tile Gather Op";

    let description = [{
        GatherOp will convert to Gather DMA in NPU40XX pipelines.
        But Gather DMA has some limitation, the input size after axis shold less than 4096.
        The pass will tile gather from the dimention after axis
    }];

    let constructor = "vpux::VPU::createTileGatherPass()";

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
// SetupEnableFP16CompressedConv
//

def SetupEnableFP16CompressedConv : PassBase<"setup-enable-fp16-compressed-conv", "vpux::ModulePass"> {
    let summary = "Enable fp16 compressed convolution and store it in the model/IR";

    let description = [{
        Uses the PipelineOptionsOp structure from the IR so the options could be exposed globally
    }];

    let constructor = "vpux::VPU::createSetupEnableFP16CompressedConvPass()";

    let options = [
        Option<
            "enableFP16CompressedConvolution", "enable-fp16-compressed-convolution",
            "bool", "false",
            "Enable FP16 Compressed convolution op"
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

//
// IntroduceInitFunction
//

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

        GenAll (generate all i.e. init and main):
            This mode allows the user to produce a blob that contains both the init()- and the main()-schedule.
            The entry-point function calls in sequence init() and then main(), following GenInit and GenMain
            conventions.
    }];

    let options = [
        Option<
            "extractionMode", "extraction-mode",
            "std::string", "\"\"",
            "Can be one of the following values: 'gen-init', 'gen-main' or 'gen-all'"
        >
    ];

    let constructor = "vpux::VPU::createIntroduceInitFunctionPass()";
}
