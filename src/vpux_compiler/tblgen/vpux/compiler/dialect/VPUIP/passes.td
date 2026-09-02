//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_PASSES
#define VPUX_COMPILER_DIALECT_VPUIP_PASSES

include "mlir/Pass/PassBase.td"

//
// AddSwKernelInstructionPrefetch
//

def AddSwKernelInstructionPrefetch : PassBase<"add-sw-kernel-instruction-prefetch", "vpux::FunctionPass"> {
    let summary = "Add SW kernel instruction prefetch ops";

    let description = [{
        This pass adds instruction prefetch operations that fetch instructions into L2$ before
        kernel execution. The pass only prefetches in first interval in which all SHAVEs are idle.

        The pass supports 2 modes of prefetch: dummy kernels and dedicated prefetch kernel. Dedicated prefetch is a special
        kernel that fetches instructions from a given memory address. Dummy kernels are a special mode for older architectures
        without support for a dedicated prefetch kernel; the dummy kernel is simply a normal kernel of a given type that
        runs on a small pre-allocated buffer.

        Prefetch tasks are waiting for barrier updated by a previous group of SHAVE tasks and update the barrier of a next
        SHAVE task group. In a case prefetch tasks are inserted before any other kernel they don't wait for any barrier.
        For architectures that support dedicated prefetch kernel update barrier is necessary to ensure next kernel won't
        immediately cancel prefetch operation.

        Limitations:
        1. Only the first idle interval is used for prefetch. Adding prefetch kernels has been found to cause unpredictable
        changes in WLM schedule, resulting in regressions. Experimentally, prefetching only in the first interval performs best.
        2. Cache utilization is not analyzed by this pass. Skipping this analysis simplifies the algorithm, and it has been
        found that for the vast majority of networks L2$ can hold all instruction data.
        3. The pass is sensitive to cost model inaccuracies. Specifically, strided DMAs are known to be scored incorrectly,
        which can cause the compiler to miss idle intervals for prefetch.

        Example of schedule before and after this pass:

        Before:
        DMA             | DMA ... DMA             DMA ... DMA
        Cluster0 DPU    |             DPU ... DPU
        Cluster0 SHAVE0 |                                     softmax, convert
        Cluster0 SHAVE1 |                                     softmax
        Cluster1 DPU    |             DPU ... DPU
        Cluster1 SHAVE0 |                                     topk
        Cluster1 SHAVE1 |                                     topk

        After:
        DMA             | DMA ... DMA             DMA ... DMA
        Cluster0 DPU    |             DPU ... DPU
        Cluster0 SHAVE0 | prefetch softmax                    softmax, convert
        Cluster0 SHAVE1 | prefetch convert                    softmax
        Cluster1 DPU    |             DPU ... DPU
        Cluster1 SHAVE0 | prefetch topk                       topk
        Cluster1 SHAVE1 |                                     topk
    }];

    let options = [
        Option<
            "minimumShaveInitTime", "minimum-shave-init-time",
            "size_t", "200000",
            "Specifies the time needed for shave to finish initialization"
        >,
        Option<
            "minimumShaveInitTimeDummyKernels", "minimum-shave-init-time-dummy-kernels",
            "size_t", "100000",
            "Specifies the time needed for shave to finish initialization for dummy kernels"
        >,
        Option<
            "minimumShaveIdleTimeForPrefetch", "minimum-shave-idle-time-for-prefetch",
            "size_t", "50000",
            "Specifies the minimum idle time of all ActShaves that is needed for the pass to insert prefetch operations"
        >
    ];

    let constructor = "vpux::VPUIP::createAddSwKernelInstructionPrefetchPass()";
}

//
// MovePureViewOpBeforeCopy
//

def MovePureViewOpBeforeCopy : PassBase<"move-pure-view-op-before-copy", "vpux::FunctionPass"> {
    let summary = "Move pure view-like operations before copy";

    let description = [{
        AI-generated summary

        By moving pure view-like ops before Copy operations, this pass creates copy operation chains
        that can later be fused by downstream optimization passes.

        Rewriters:
        - MoveViewOpToTheFrontOfCopy: Moves PermuteCast, GenericReshape, QuantizeCast, and ShapeCast
          ops that appear after a CopyOp to before it. Skips ShapeCast when used with input channels
          compression (compressed convolution). Handles distributed buffer types and validates
          stride compatibility.
          Before: CopyOp -> PermuteCast -> GenericReshape -> CopyOp
          After: PermuteCast -> GenericReshape -> CopyOp -> CopyOp

        - MoveSubviewToTheFrontOfCopy: Moves SubViewOp that sits between two CopyOps to before
          the first CopyOp, reducing copied data size. Only applies when the parent CopyOp has a
          single SubView user and the SubView is compatible with any distributed buffer distribution.
          Before: CopyOp(full) -> SubView -> CopyOp
          After: SubView -> CopyOp(reduced)
    }];

    let constructor = "vpux::VPUIP::createMovePureViewOpBeforeCopyPass()";
}

//
// OptimizeCopies
//

def OptimizeCopies : PassBase<"optimize-copies", "vpux::FunctionPass"> {
    let summary = "Removes Copy Ops which are unnecessary";

    let description = [{
        AI-generated summary

        This pass applies multiple rewrite patterns to eliminate unnecessary Copy operations,
        reducing DMA transfers and intermediate buffer allocations.

        Rewrite patterns (applied in priority order):
        Phase 1 (greedy, with benefit ordering):
        - CopyOpSequence: Fuses chains of Copy operations (e.g. CMX->DDR->CMX) into a single Copy.
          Handles SubView and view-like op chains.
        - CMXToCMXCopy: Eliminates CMX-to-CMX copies by replacing the output buffer allocation
          with a NonDistributedCast or DistributedCast when memory layouts are compatible.
        - DDRToDDRCopy: Removes DDR-to-DDR copies by propagating the source buffer to all users.
        - ConcatViewWithCopy: Optimizes ConcatView followed by Copy by adjusting ConcatView
          output buffer allocation.
        - FuseCopyToTheFrontOfDistributedCopy: Fuses a plain Copy into a preceding distributed Copy.
        - FuseCopyToTheBackOfDistributedCopy: Fuses a plain Copy into a following distributed Copy.
        - SubViewWithDistributedCopy: Combines SubView with a distributed Copy to avoid
          materializing the full tensor.
        - DuplicatedCopyWithCMXCopy: Removes redundant CMX copy when source is already DUPLICATED
          in CMX.
        - FuseCopiesThroughReshape: Eliminates copies separated by a GenericReshape.
        - SubViewWithCopy: Replaces SubView -> (GenericReshape/ShapeCast) -> Copy pattern
          with Subview -> ViewOp -> (GenericReshape/ShapeCast), if the SubView is trivial and
          the CopyOp is within the same memory space.

        Phase 2 (separate greedy pass, depends on Phase 1 results):
        - CopyOpSequenceWithSubview: Handles Copy sequences with intervening SubViews,
          checking stride constraints after Phase 1 optimizations.
        - GatherDMAWithSubview: Fuses GatherDMA with a following SubView.
    }];

    let constructor = "vpux::VPUIP::createOptimizeCopiesPass()";
}

//
// UniquifyWeightsTableCopies
//

def UniquifyWeightsTableCopies : PassBase<"uniquify-weights-table-copies", "vpux::FunctionPass"> {
    let summary = "Removes Copy Ops of shared constant weights table";

    let description = [{
        AI-generated summary

        This pass deduplicates Copy operations of shared constant weights tables when weights table
        reuse is enabled (isZeroOffsetWeightsTable attribute is set).

        The pass finds NCEClusterTaskOp operations with IsZeroOffsetWeightsTable set, traces their
        weight table inputs to Const::DeclareOp sources through single-use CopyOps, groups all
        CopyOps that share the same Const::DeclareOp source and output type, then replaces all
        duplicate CopyOps with the first CopyOp in each group, erasing the redundant copies.
    }];

    let constructor = "vpux::VPUIP::createUniquifyWeightsTableCopiesPass()";
}

//
// OptimizeConcatViewCopies
//

def OptimizeConcatViewCopies : PassBase<"optimize-concat-view-copies", "vpux::FunctionPass"> {
    let summary = "Optimize ConcatView Op to remove unnecessary copies or reduce copies data size";

    let description = [{
        - Propagate SubView Op before ConcatView Op to reduce data size of copies
        Target Pattern: NCE Task 16 channels output -> Copy 16 channels -> Concat -> Subview -> 3 channels
        Optimization:   NCE Task 16 channels output -> Copy 3 channels -> Concat -> Subview -> 3 channels

        - Fuse ConcatView Ops to remove unnecessary copies, two conditions need to be satisfied:
        a) The Stride Level for each ConcatView input (after fusing) should be no more than 2;
           It's a runtime and HW limitation in order to get the right NNDMA descriptor, we support a maximum of 3D DMA transfers with 2 levels of striding.
        b) The number of inputs from the second ConcatView, which come from the output of the first ConcatView should no more than 1;
           For example, first ConcatView has M inputs, second ConcatView has N inputs, out of which P of them are the output of the first ConcatView
           After fusing, the number of input copies is: M * P + (N - P)
           Can't ensure we get benefit when P is of a large size. Limit optimization to P=1.

        - FuseGatherDMAWithSubViewCopy: Transforms a costly GatherDMAOp (with INDEXED mode) into Subview + CopyOps, reducing the amount of data copied.
          Turns subgraph:
        NCE_i  (CMX, DUPLICATED|SEGMENTED or SEGMENTED (on K only))
            |
        Copy  (CMX → SubView of alloc_concat, DDR)       ← one per NCE
              \   /
          ConcatViewOp  (DDR, shape 1xCxHxW, NHWC) -> GenericReshape  (1xCx(H*W)x1) -> PermuteCast  (mem_perm=[d1,d3,d0,d2], (H*W)xCx1x1) ->
                -> GenericReshape  (flat: numFlatRows x gatherRowSize) -> GatherDMAOp  (gathers K rows → K x gatherRowSize, CMX) ->
                -> GenericReshape  (K x gatherRowSize → 1 x K x gatherRowSize) -> Copy  (CMX -> DDR)
          into:
        For each row index r = indices[j]:
            NCE_inputIdx  (CMX, DUPLICATED/SEGMENTED on K) -> [ExtractFlatSliceOp for SEGMENTED on K] ->
                    -> SubView [0, channelOffset, h, w][1, gatherRowSize, 1, 1] -> [GenericReshape + PermuteCast to match DDR dest dims] ->
                    -> Copy  (CMX → row-j slot of downstream alloc, DDR)
    }];

    let constructor = "vpux::VPUIP::createOptimizeConcatViewCopiesPass()";
}

//
// OptimizeSubviewCopies
//

def OptimizeSubviewCopies : PassBase<"optimize-subview-copies", "vpux::FunctionPass"> {
    let summary = "Optimize SubView Op to remove unnecessary copies or reduce copies data size";

    let description = [{
        This pass handles the following pattern:
        %source -> n x Subview -> n x CopyOp -> n x (input act of NCEClusterTask<CONV>)

        This will become:
        %source -> Copyop -> n x Subview -> n x (input act of NCEClusterTask<CONV>)

        The following restrictions apply:
            * Subview can only have CopyOp consumers which go into input 0 of a DPU Conv
            * Subview is of kind: 1xCx1x1 -> VPUIP.Subview -> 1xC'x1x1, with C > C'
            * if Conv is distributed, the input **must** be DUPLICATED
            * all Convs must individually fit in CMX with a buff %source's size

        Additionally, if %source is the result of a compatible CMX -> DDR DMA we can also optimize it as well:
        %cmx_source -> DMA to DDR  -> n x Subview over C -> n x DMA to CMX ->  n x NCEConv
            to
        %cmx_source  -> (optional DistributedCast) -> n x Subview over C ->  n x NCEConv

        A compatible CMX -> DMA DMA must meet the following requirements:
            * if Distributed must be of DUPLICATED-like mode
            * the input of this DMA must fit in CMX with all the children Convs after the Subviews

    }];

    let constructor = "vpux::VPUIP::createOptimizeSubviewCopiesPass()";
}

//
// OptimizeParallelCopies
//

def OptimizeParallelCopies : PassBase<"optimize-parallel-copies", "vpux::FunctionPass"> {
    let summary = "Copy the data only once for all the tiles that share the same data";

    let description = [{
        AI-generated summary

        This pass merges parallel Copy operations that transfer the same data into a single Copy.

        Rewriter: ParallelCopiesRewriter
        The rewriter finds CopyOps consumed by tiles of one tiling subgraph that operate on the same
        weight or activation source. Merge candidates are validated against:
        - Compute operation distance: fuses copies only when consumer NCE tasks (in-place Eltwise
          or Conv) are within 3 compute steps of each other
        - Sibling filtering: the compute-distance heuristic is applied when at least one sibling
          compute op is in-place ELTWISE
        - Two-axis tiling distance: applies a distance cost of 2 for two-axis tiling scenarios
        - Constant copy optimization: optionally merges constant copies (controlled by
          enableOptimizeConstCopy flag)
        - Interpolate mode and coordinate mode compatibility for NCE operations
        - SubView and DistributedBuffer compatibility

        ParallelToDDRCopiesRewriter
          Fuses CMX->DDR CopyOps with the same input value and result type into a single copy,
          provided neither the root nor the sibling writes to a SubView or function-argument buffer.

        SplatConstRewriter
          Targets NCE tasks tiled over K that each receive a separate splat const.Declare and
          CopyOp for weight_table_scale / weight_table_bias.  Sibling NCE tasks are identified
          by tracing weights through SubView offsets back to a shared DDR buffer; copies whose
          splat values are equivalent are merged into a single copy.
    }];

    let constructor = "vpux::VPUIP::createOptimizeParallelCopiesPass()";
}

//
// FuseLastCopy
//

def FuseLastCopy : PassBase<"fuse-last-copy", "vpux::FunctionPass"> {
    let summary = "Remove last Copy operations that output to DDR";

    let description = [{
        AI-generated summary

        This pass eliminates the final Copy operations that write results to network output buffers
        (function block arguments) by making producers write directly to the output buffer.

        The pass walks Copy operations whose output buffer is a function block argument (network output).
        For each such Copy, it uses AliasesInfo to trace the input back to its root allocation. It then
        replaces the root allocation with the output buffer, eliminating the Copy.

        When cast operations (GenericReshape, QuantizeCast, PermuteCast, ShapeCast, ReinterpretCast)
        appear between the producer and the Copy, the pass reverses and applies the same casts to the
        output buffer so the producer can write directly to it. Chains of multiple casts are handled
        iteratively. ConcatView operations in the chain are also fused.

        Restrictions:
        - Input and output must be in the same memory space
        - Input and output must be compact (no strides)
        - No multiple output copy operations sharing the same source root
        - Strided network outputs are not supported
    }];

    let constructor = "vpux::VPUIP::createFuseLastCopyPass()";
}

//
// CopyOpTiling
//

def CopyOpTiling : PassBase<"tile-copies", "vpux::FunctionPass"> {
    let summary = "Legalizes Copy Ops which do not fit hardware capabilities";

    let description = [{
        AI-generated summary

        This pass splits Copy operations that exceed hardware DMA capabilities into multiple
        smaller tiles.

        Rewriters:
        - CopyOpTiling: Splits large CopyOps to fit DMA hardware constraints:
          * Maximum transfer size: 16MB (2^24 bytes)
          * Maximum number of planes: 256 (planes implement higher-dimensional stride DMA)
          Tiles are balanced across DMA ports for parallel execution. Does not handle
          distributed buffer types.

        - SingleCopyOpWithConcatOpUserTiling: For CopyOp → ConcatView patterns with strided
          copies, splits the CopyOp across DMA ports (typically 2) for parallel execution.
          Applied after CopyOpTiling in a separate partial conversion phase.
    }];

    let constructor = "vpux::VPUIP::createCopyOpTilingPass()";
}

//
// ConvertDynamicReshapeToInPlace
//

def ConvertDynamicReshapeToInPlace : PassBase<"convert-dynamic-reshape-to-in-place", "vpux::FunctionPass"> {
    let summary = "Convert DynamicReshape operation to read and write to the same buffer in memory";

    let description = [{
        AI-generated summary

        This pass converts DynamicReshape SwKernel operations to in-place operations that read and
        write to the same buffer when the input and output ranks are equal and memory kinds match.

        The pass walks SwKernelOp operations with kernel entry name "dynamic_reshape". For eligible
        operations, it inserts a VPUIP.ViewOp on the input to cast it to the output type, then
        creates a new SwKernelOp that uses the viewed buffer as both input and output. Dynamic
        input shapes (bounded buffers) are preserved through the transformation.
    }];

    let constructor = "vpux::VPUIP::createConvertDynamicReshapeToInPlacePass()";
}

//
// ConvertEltwiseToInPlace
//

def ConvertEltwiseToInPlace : PassBase<"convert-eltwise-to-in-place", "vpux::FunctionPass"> {
    let summary = "Convert Eltwise operation to read and write to the same buffer in memory";

    let description = [{
        This pass will check if Eltwise operation was selected for inplace execution
        and convert the Eltwise to write the result into one of the inputs in memory.
        A view operation is added to support different quantization parameters for input/output.
        Can be extended to support different input/output memory requirements with a subview.
    }];

    let constructor = "vpux::VPUIP::createConvertEltwiseToInPlacePass()";
}

//
// ConvertSprLUTToConst
//


def ConvertSprLUTToConst : PassBase<"convert-sprlut-to-const", "vpux::FunctionPass"> {
    let summary = "Convert sprLUT from PPE attribute to Const";

    let description = [{
        Software Programmable Lookup Table for NCE operation is treated in similar way as weights table,
        i.e. it's just yet another input. Introducing this input on VPU level would require changes for
        every possible NCE operation we have there, which is hard to maintain and test. In VPUIP all NCE
        operations are converted to the same NCEClusterTask, so here we need to change only op.
        This pass performs this conversion, extracting sprLUT from PPE attribute and making it Const.
    }];

    let constructor = "vpux::VPUIP::createConvertSprLUTToConstPass()";
}

//
// ConvertPalletLUTToConst
//

def ConvertPalletLUTToConst : PassBase<"convert-pallet-lut-to-const", "vpux::FunctionPass"> {
    let summary = "Convert pallet LUT to Const";

    let description = [{
        AI-generated summary

        Materializes palette Look-Up Table data as a Const.DeclareOp for NCE cluster tasks
        with palletized (quantile-quantized) weights. Fills a 64-entry 16-bit LUT from the
        quantile type definition and attaches it as the palletLookupTable input of
        VPUIP.NCEClusterTaskOp.
    }];

    let constructor = "vpux::VPUIP::createConvertPalletLUTToConstPass()";
}

//
// InsertCopyForEltwiseInPlaceInput
//

def InsertCopyForEltwiseInPlaceInput : PassBase<"insert-copy-for-eltwise-in-place-input", "vpux::FunctionPass"> {
    let summary = "Ensures input buffer that gets overwritten by output is not used by another op.";

    let description = [{
        This is a legalization pass which will check that the input buffer that gets overwritten by an in place
        Eltwise's output will not be used by another op. If such a case exist, we introduce a spill at
        Eltwise's input, thus ensuring that the other ops receive an input buffer that is not overwritten.

        This pass should be applied after copy optimization passes
    }];

    let constructor = "vpux::VPUIP::createInsertCopyForEltwiseInPlaceInputPass()";
}

//
// SetMemorySpace
//

// TODO:#10593: extract SetInternalMemorySpace pass
def SetMemorySpace : PassBase<"set-memory-space", "vpux::ModulePass"> {
    let summary = "Set specific memory space for all memory buffers";

    let description = [{
        AI-generated summary

        This pass assigns the specified memory space (e.g. DDR) to all internal memory buffers that
        do not yet have a memory space set.

        The pass operates in three phases per function:
        1. Function signature update (optional, controlled by setMemorySpaceForFunctionBoundaries):
           Updates function argument types and their aliases to the target memory space, then
           updates the function signature. When function-boundary updates are disabled, the pass
           keeps function signatures unchanged and still tags pure view-like aliases derived from
           boundary buffers so lower-level operations see explicit memory spaces.
        2. Allocation update: Walks memref::AllocOp operations without a memory space and updates
           all their aliases to the target memory space, including async::ValueType wrappers.
        3. Grouping operation update: Walks GroupedViewOpInterface operations (e.g.
           VPUIP.GroupBoundedBuffer) and propagates the memory space to operands that lack one
           when at least one operand already has the target memory space set.
    }];

    let constructor = [{
        vpux::VPUIP::createSetMemorySpacePass(vpux::VPU::symbolizeEnum<VPU::MemoryKind>)
    }];

    let options = [
        Option<
            "memSpaceName", "memory-space",
            "std::string", [{""}],
            "Memory space to perform allocation"
        >,
        Option<
            "setMemorySpaceForFunctionBoundaries", "set-memory-space-for-function-boundaries",
            "bool", "true",
            "Enable/Disable MemorySpace information in function signatures"
        >
    ];
}

//
// QueryArgsAllocationAnalysis
//

def QueryArgsAllocationAnalysis : PassBase<"query-args-allocation-analysis", "vpux::ModulePass"> {
    let summary = "Calculate and cache ReservedMemInfo analysis";

    let description = [{
        StaticAllocation pass should be function pass to achieve the best performance in multithreaded mode(when there are several functions in IR).
        At the same time, StaticAllocation should be aware whether some memory is reserved by the caller(for example, for arguments).
        Since caller's function is changing at the same time,
        the QueryArgsAllocationAnalysis prepare the necessary information and marks it preserved to be used in StaticAllocation.
    }];

    let constructor = [{
        vpux::VPUIP::createQueryArgsAllocationAnalysisPass()
    }];
}

//
// StaticAllocation
//

def StaticAllocation : PassBase<"static-allocation", "vpux::FunctionPass"> {
    let summary = "Replace dynamic allocations with static";

    let description = [{
        AI-generated summary

        Performs static memory allocation for buffers in the specified memory space (CMX or DDR)
        using a linear scan algorithm. Assigns fixed memory offsets to all buffer declarations at
        compile time, computes the total memory requirement, and verifies allocations fit within
        available memory capacity.
    }];

    let constructor = [{
        vpux::VPUIP::createStaticAllocationPass(vpux::VPU::symbolizeEnum<VPU::MemoryKind>)
    }];

    let options = [
        Option<
            "memSpaceName", "memory-space",
            "std::string", [{""}],
            "Memory space to perform allocation"
        >
    ];
}

//
// Patch Weight Table
//

def PatchWeightsTable : PassBase<"patch-weight-table", "vpux::FunctionPass"> {
    let summary = "Adjusts weights and sparsity pointers after memory scheduling";

    let description = [{
        AI-generated summary

        This pass patches weights table constants with correct memory pointers after memory
        scheduling has assigned buffer addresses.

        For each NCEClusterTaskOp with a weight table (excluding MaxPool, operations with
        populateWeightTableWithShave, and operations with isZeroOffsetWeightsTable), the pass:
        1. Traces the weight table operand through DeclareBufferOp to find the loading DMA and
           source Const::DeclareOp (resolving spill chains if weights were spilled).
        2. Computes per-cluster weights pointers from the DeclareBufferOp byte offset of the
           weights operand. For Duplicated|Segmented distributions, per-cluster offsets are
           computed based on the segmentation axis and strides.
        3. Computes the sparsity map pointer from the root buffer of the weightsSparsityMap
           operand.
        4. Handles fused constants by using the fused buffer base offset and computing the
           weight table size from the weights pointer minus base offset.
        5. Adds a relocateWeightsTablePointers transformation to the Const::DeclareOp content
           attribute, which patches the weight and sparsity pointer fields at runtime.
    }];

    let constructor = "vpux::VPUIP::createPatchWeightsTablePass()";
}

//
// PatchPopulateWeightTableWithShave
//

def PatchPopulateWeightTableWithShave : PassBase<"patch-populate-weight-table-with-shave", "vpux::FunctionPass"> {
    let summary = "Adjusts weights pointers after memory scheduling and before loosing data dependencies due to slices";

    let description = [{
        AI-generated summary

        Updates weight table entries for operations that execute on SHAVE processors by patching
        the weight pointer and sparsity pointer fields. Adjusts pointer offsets based on
        SHAVE-specific memory mapping requirements, which differ from DPU weight table pointer
        conventions.
    }];

    let constructor = "vpux::VPUIP::createPatchPopulateWeightTableWithShavePass()";
}

//
// Linearization
//

def Linearization : PassBase<"linearization", "vpux::ModulePass"> {
    let summary = "Perform linearization of the IR";

    let description = [{
        AI-generated summary

        Serializes the task execution order by inserting explicit dependencies between all tasks,
        creating a fully sequential execution schedule. Adds barrier dependencies that enforce an
        execution order matching the original IR order, used as a baseline scheduling strategy or
        for debugging.
    }];

    let constructor = "vpux::VPUIP::createLinearizationPass()";
}


//
// BreakDataFlow
//

def BreakDataFlow : PassBase<"break-data-flow", "vpux::FunctionPass"> {
    let summary = "Breaks the data flow in the graph";

    let description = [{
        AI-generated summary

        This pass breaks the data flow in the graph so that operation results are no longer directly
        used by subsequent operations. This is required for the VPURT dialect because all VPUIP
        dialect tasks will be wrapped inside the body of a TaskOp, and operation results from one
        TaskOp body cannot be used inside another TaskOp body.

        The pass performs two transformations:
        1. For all VPUIP::LayerOpInterface operations: replaces each operation result with the
           corresponding output buffer operand, breaking the def-use chain.
        2. For all VPUIP::ConcatViewOp operations: replaces the ConcatView result with its output
           buffer and erases the ConcatViewOp. Also cleans up single-use async.await operations
           feeding into the ConcatView inputs.
    }];

    let constructor = "vpux::VPUIP::createBreakDataFlowPass()";
}

//
// DMATaskProfilingAfterBarrierSched
//

def DMATaskProfilingAfterBarrierSched : PassBase<"dma-task-profiling-after-barrier", "vpux::ModulePass"> {
    let summary = "DMA task profiling handling after barrier scheduled";

    let description = [{
        AI-generated summary

        Inserts DMA profiling timestamp operations after barrier scheduling. Creates NNDMAOp
        tasks that read the hardware timer register and write timestamps to profiling buffer
        slots, integrating them into the existing barrier dependency graph.
    }];

    let constructor = "vpux::VPUIP::createDMATaskProfilingAfterBarrierSchedPass()";
}

//
// CaptureWorkpoint
//

def CaptureWorkpoint : PassBase<"capture-workpoint", "vpux::ModulePass"> {
    let summary = "Read value from WORKPOINT_CONFIG_MIRROR.FINAL_PLL_FREQ and store to profiling output";

    let description = [{
        AI-generated summary

        Inserts DMA operations that read the hardware workpoint register at inference start and
        end. The workpoint value captures the NPU clock frequency configuration, which profiling
        tools use to convert hardware cycle counts into wall-clock timing measurements.
    }];

    let constructor = "vpux::VPUIP::createCaptureWorkpointPass()";
}

//
// DPUProfiling
//

def DPUProfiling : PassBase<"dpu-profiling", "vpux::ModulePass"> {
    let summary = "DPU task profiling";

    let description = [{
        AI-generated summary

        Instruments DPU (NCE) tasks with profiling timestamp capture by allocating profiling
        buffer slots in CMX and inserting timestamp write operations. Each DPU task records start
        and end cycle counts, enabling per-layer performance analysis of convolution and pooling
        operations.
    }];

    let constructor = "vpux::VPUIP::createDPUProfilingPass()";

}

//
// M2IProfiling
//

def M2IProfiling : PassBase<"m2i-profiling", "vpux::ModulePass"> {
    let summary = "M2I task profiling";

    let description = [{
        AI-generated summary

        Allocates profiling buffer slots for M2I tasks and inserts timestamp capture operations.
        Records start and end cycle counts for M2I media preprocessing tasks (resize, color
        convert, normalize).
    }];

    let constructor = "vpux::VPUIP::createM2IProfilingPass()";
}

def LegalizeRepeatingFuncCalls : PassBase<"legalize-repeating-func-calls", "vpux::FunctionPass"> {
    let summary = "Ensure repeating func calls adhere to VPUIP constaints";

    let description = [{
        This pass adjusts repeating func.call invocations (call the same
        function multiple times) in a way that satisfies constraints of later
        passes (c.f. ConvertFuncArgsToDeclarations).
    }];

    let constructor = "vpux::VPUIP::createLegalizeRepeatingFuncCallsPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
    ];
}

//=================================================================================
// Asynchronous Scheduling
//=================================================================================

//
// WrapIntoAsyncRegions
//

def WrapIntoAsyncRegions : PassBase<"wrap-into-async-regions", "vpux::FunctionPass"> {
    let summary = "Wraps layer operations into asynchronous regions";

    let description = [{
        AI-generated summary

        Wraps each VPUIP.AsyncLayerOpInterface operation into its own mlir.async.execute
        region. Clones the operation into the async body, creates async.yield for results,
        and replaces original uses with async.await values. Preserves the executor attribute
        on each async.execute region for downstream scheduling.
    }];

    let constructor = "vpux::VPUIP::createWrapIntoAsyncRegionsPass()";

    let dependentDialects = [
        "mlir::async::AsyncDialect"
    ];
}

//
// AsyncRegionsOutlining
//

def AsyncRegionsOutlining : PassBase<"async-regions-outlining", "vpux::ModulePass"> {
    let summary = "Outlines async operations into functions based on AsyncDepsInfo";

    let description = [{
        This pass outlines async operations in main function into a number of sub-functions based on AsyncDepsInfo.
        Unlike the outlining in IE dialect and VPU dialect, this pass operates within the VPUIP dialect and after the
        increment pipeline. It allows for more precise control over DMA and compute operations within each sub-function.
        This precision is beneficial for memory scheduling that occurs during the feasible allocation process.
        This pass is executed before the feasible allocation pass, allowing feasible allocation to be performed more
        concurrently, thereby reducing compilation time. It helps mitigate long spills, such as shared weights, enabling
        feasible allocation to free up CMX occupied by live buffers for future use which helps enable more prefetching
        and pipelining case. This pass reduces the number of operations in each function, which in turn decreases memory
        fragmentation possibility during the feasible allocation pass.

        The behavior of the pass is as follows.
        Given the following function:
        func @main {
            %cst0 = const.Declare
            %cst1 = const.Declare
            %buf0 = memref.alloc()
            %buf1 = memref.alloc()
            %bufCMX0 = VPUIP.DistributedBuffer
            %bufCMX1 = VPUIP.DistributedBuffer
            %bufCMX2 = VPUIP.DistributedBuffer
            %bufCMX3 = VPUIP.DistributedBuffer
            %bufCMX4 = VPUIP.DistributedBuffer

            // weight DMA
            %t0, %r0 = async.execute index=0 {
                %0 = VPUIP.NNDMA inputs(%cst0) outputs(%bufCMX0)
                async.yield
            }

            // activation DMA
            %t1, %r1 = async.execute index=1 {
                %0 = VPUIP.NNDMA inputs(%buf0) outputs(%bufCMX1)
                async.yield
            }

            // Compute
            %t2, %r2 = async.execute [t0, t1] use [r0, r1] index=2 {
                %0 = NCE inputs(%bufCMX0, bufCMX1) outputs(%bufCMX2)
                async.yield
            }

            // weight DMA
            %t3, %r3 = async.execute index=3 {
                %0 = VPUIP.NNDMA inputs(%cst1) outputs(%bufCMX3)
                async.yield
            }

            // Compute
            %t4, %r4 = async.execute [t2, t3] use [r2, r3] index=4 {
                %0 = NCE inputs(%bufCMX2, bufCMX3) outputs(%bufCMX4)
                async.yield
            }

            // activation out DMA
            %t5, %r5 = async.execute [t4] use [r4] index=5 {
                %0 = VPUIP.NNDMA inputs(%bufCMX4) outputs(%buf1)
                async.yield
            }

            %r6 = async.await %r5 : !async.value
            return %r6
        }

        If the operations are divided into sets based on AsyncDepsInfo:
            set1 = {0, 1, 2}
            set2 = {3, 4, 5}
        The pass outlines these operations into separate functions and replaces
        the original function with the following:

        func @main_outlining1 {
            %cst0 = const.Declare
            %buf0 = memref.alloc()
            %buf1 = memref.alloc()
            %bufCMX0 = VPUIP.DistributedBuffer
            %bufCMX1 = VPUIP.DistributedBuffer
            %bufCMX2 = VPUIP.DistributedBuffer

            // weight DMA
            %t0, %r0 = async.execute index=0 {
                %0 = VPUIP.NNDMA inputs(%cst0) outputs(%bufCMX0)
                async.yield
            }

            // activation DMA
            %t1, %r1 = async.execute index=1 {
                %0 = VPUIP.NNDMA inputs(%buf0) outputs(%bufCMX1)
                async.yield
            }

            // Compute
            %t2, %r2 = async.execute [t0, t1] use [r0, r1] index=2 {
                %0 = NCE inputs(%bufCMX0, bufCMX1) outputs(%bufCMX2)
                async.yield
            }

            // added DMA to pass buffer through DDR
            %t3, %r3 = async.execute index=3 {
                %0 = VPUIP.NNDMA inputs(%bufCMX2) outputs(%buf1)
                async.yield
            }

            %r4 = async.await %r3 : !async.value
            return %r4
        }

        func @main_outlining2 {
            %cst0 = const.Declare
            %buf0 = memref.alloc()
            %buf1 = memref.alloc()
            %bufCMX0 = VPUIP.DistributedBuffer
            %bufCMX1 = VPUIP.DistributedBuffer
            %bufCMX2 = VPUIP.DistributedBuffer

            // weight DMA
            %t0, %r0 = async.execute index=0 {
                %0 = VPUIP.NNDMA inputs(%cst0) outputs(%bufCMX0)
                async.yield
            }

            // activation DMA
            %t1, %r1 = async.execute index=1 {
                %0 = VPUIP.NNDMA inputs(%buf0) outputs(%bufCMX1)
                async.yield
            }

            // Compute
            %t2, %r2 = async.execute [t0, t1] use [r0, r1] index=2 {
                %0 = NCE inputs(%bufCMX0, bufCMX1) outputs(%bufCMX2)
                async.yield
            }

            // activation out DMA
            %t3, %r3 = async.execute index=3 {
                %0 = VPUIP.NNDMA inputs(%bufCMX2) outputs(%buf1)
                async.yield
            }

            %r4 = async.await %r3 : !async.value
            return %r4
        }

        func main (%arg0) {
            %buf0 = memref.alloc()
            %buf1 = memref.alloc()

            %t0, %r0 = async.execute index=0 {
                %0 = Call @main_outlining1 inputs(%arg0) outputs(%buf0)
                async.yield
            }

            %t1, %r1 = async.execute [t0] uses [r0] index=1 {
                %0 = Call @main_outlining2 inputs(%buf0) outputs(%buf1)
                async.yield
            }

            %r2 = async.await %r1 : !async.value
            return %r2
        }
    }];

    let constructor = "vpux::VPUIP::createAsyncRegionsOutliningPass()";

    let dependentDialects = [
        "mlir::async::AsyncDialect",
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];

    let options = [
        Option<
            "asyncRegionOutliningMinOpsInBlock", "async-region-outlining-min-ops-in-block",
            "int", "",
            "Threshold for number of ops in each instance of async region outlining"
        >
    ];
}

//
// MoveWaitResultToAsyncBlockArgs
//

def MoveWaitResultToAsyncBlockArgs : PassBase<"move-wait-result-to-async-block-args", "vpux::FunctionPass"> {
    let summary = "Move async.await results into async.execute block arguments to optimize scheduling";

    let description = [{
        AI-generated summary

        Converts async.await results used inside async.execute regions into block arguments
        of those regions (via bodyOperands), eliminating the need for the await at the outer
        scope. Moves remaining await operations closer to their first user and erases unused
        ones.
    }];

    let constructor = "vpux::VPUIP::createMoveWaitResultToAsyncBlockArgsPass()";
}

//
// CalculateAsyncRegionCycleCost
//

def CalculateAsyncRegionCycleCost : PassBase<"calculate-async-region-cycle-cost", "vpux::FunctionPass"> {
    let summary = "Calculate and attach cycle cost attributes to async.execute operations";

    let description = [{
        AI-generated summary

        Computes the execution cycle cost for each async.execute operation using CycleCostInfo
        (backed by the VPU cost model) and stores the result as the cycleCost integer attribute
        on the operation.
    }];

    let constructor = "vpux::VPUIP::createCalculateAsyncRegionCycleCostPass()";
}

//
// MoveViewOpsIntoAsyncRegions
//

def MoveViewOpsIntoAsyncRegions : PassBase<"move-view-ops-into-async-regions", "vpux::FunctionPass"> {
    let summary = "Duplicate pure view operations into async.execute regions to reduce outer dependencies";

    let description = [{
        AI-generated summary

        Clones pure view-like operations (e.g., SubView, PermuteCast) that are defined outside
        async.execute regions but used inside them, placing the clone at the start of the
        async.execute body block. The original outer operation is left in place for other users.
    }];

    let constructor = "vpux::VPUIP::createMoveViewOpsIntoAsyncRegionsPass()";
}

//
// OptimizeAsyncDeps
//

def OptimizeAsyncDeps : PassBase<"optimize-async-deps", "vpux::FunctionPass"> {
    let summary = "Optimizes dependencies between 'async.execute' operations";

    let description = [{
        The pass tries to remove extra explicit `!async.token` based dependencies,
        if they are represented implicitly (as a result of transitive dependencies).
    }];

    let constructor = "vpux::VPUIP::createOptimizeAsyncDepsPass()";
}

//
// AddAsyncSchedulingDependencies
//

def AddAsyncSchedulingDependencies : PassBase<"add-async-scheduling-dependencies", "vpux::FunctionPass"> {
    let summary = "Add dependencies required by asynchronous scheduling constraints";

    let description = [{
        Provides a short-term solution for scheduler prioritization of input-only producers by
        adding artificial dependencies between consecutive DynamicDequantize and convolution
        groups. This keeps each DynamicDequantize producer close to its convolution consumer
        instead of scheduling producers from later groups too early.

        TODO(E227196): Remove this pass after the scheduler handles input-only producers
        just in time without artificial dependencies.
    }];

    let constructor = "vpux::VPUIP::createAddAsyncSchedulingDependenciesPass()";
}

//
// FeasibleAllocation
//

def FeasibleAllocation : PassBase<"feasible-allocation", "vpux::FunctionPass"> {
    let summary = "Feasible Memory Scheduling Pass";

    let description = [{
        AI-generated summary

        Performs feasible memory allocation using a spill-aware algorithm that handles cases where
        tensors do not fit in CMX. Attempts to allocate buffers in CMX and automatically inserts
        spill-to-DDR and prefetch-from-DDR DMA operations when memory pressure exceeds available
        CMX capacity, with configurable spill optimization strategies.
    }];

    let constructor = [{
        vpux::VPUIP::createFeasibleAllocationPass(
        [](vpux::StringRef memSpaceName) {
            VPUX_THROW_UNLESS(!memSpaceName.empty(), "Missing memory space option");
            return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpaceName);
        }, vpux::VPU::symbolizeEnum<VPU::MemoryKind>
        )
    }];

    let options = [
        Option<
            "memSpaceName", "memory-space",
            "std::string", [{""}],
            "Memory space to perform allocation"
        >,
        Option<
            "secondLvlMemSpaceName", "second-level-memory-space",
            "std::string", [{""}],
            "Second level memory space to perform spilling"
        >,
        Option<
            "linearizeSchedule", "linearize-schedule",
            "bool", "false",
            "Linearize all tasks on all engines"
        >,
        Option<
            "enableLoopAllocation", "enable-loop-allocation",
            "bool", "false",
            "Enables loop allocation for tiling and vertical fusion regions"
        >,
        Option<
            "enablePipelining", "pipelining",
            "bool", "true",
            "Enables pipelining"
        >,
        Option<
            "enablePrefetching", "prefetching",
            "bool", "true",
            "Enables prefetching"
        >,
        Option<
            "optimizeFragmentation", "optimize-fragmentation",
            "bool", "true",
            "Perform buffer sorting to optimize fragmentation"
        >,
        Option<
            "optimizeDynamicSpilling", "optimize-dynamic-spilling",
            "bool", "true",
            "Perform dynamic spill DMA optimization"
        >,
        Option<
            "enableMultiScheduleHeuristic", "enable-multi-schedule-heuristic",
            "bool", "false",
            "Enables compiler to schedule with different heuristic logics and compare costs"
        >,
        // E#216928 remove when feature fully enabled
        Option<
            "enableVfUndefinedScheduler", "enable-vf-undefined-scheduler",
            "bool", "false",
            "Route VF regions through UndefinedVF (currently empty schedule -> legacy fallback)"
        >
    ];

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// WrapWithPermuteAsNNDMA
//

def WrapWithPermuteAsNNDMA: PassBase<"wrap-with-permute-as-nndma", "vpux::FunctionPass"> {
    let summary = "Wrap op and permute operation as dma and fuse unnecessary copy ops";

    let description = [{
        AI-generated summary

        This pass fuses permutation, expansion, tiling, and depth/space conversion operations with
        adjacent Copy operations, converting them into efficient PermuteDMA or other DMA operations.

        Rewrite patterns:
        - FuseMemPermuteWithCopy: Fuses SwKernel MemPermute with a following Copy into PermuteDMA.
        - FuseExpandWithCopy: Fuses VPUIP.Expand with a following Copy into PermuteDMA when the
          expand can be expressed as a permutation.
        - FusePerAxisTileWithCopy: Fuses SwKernel PerAxisTile with a following Copy into
          PerAxisTileDMA.
        - FuseExpandWithUpsampling: Fuses VPUIP.Expand with a following UpsamplingDMA.
        - FuseExpandAndPermuteWithCopy: Fuses an Expand followed by a MemPermute and Copy
          into a single PermuteDMA.
        - FuseSpaceToDepthAndPermute: Fuses SwKernel SpaceToDepth with a MemPermute into
          SpaceToDepthDMA.
        - WrapDepthToSpaceAsDistributedNNDMA: Converts SwKernel DepthToSpace to
          DepthToSpaceDMA with distributed buffer support.
        - FuseSpaceToDepthWithDistributedCopy: Fuses SwKernel SpaceToDepth with a distributed
          Copy into SpaceToDepthDMA.
        - FuseDistributedCopyWithMemPermute: Fuses a distributed Copy with a following
          SwKernel MemPermute into PermuteDMA.
        - FuseDistributedMemPermuteWithViewLikeOps: Fuses a PermuteDMA with following
          view-like operations (PermuteCast, ShapeCast, GenericReshape) to avoid redundant copies.
    }];

    let constructor = "vpux::VPUIP::createWrapWithPermuteAsNNDMAPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// OptimizeTileOpAsNNDMA
//

def OptimizeTileOpAsNNDMA: PassBase<"optimize-tile-op-as-nndma", "vpux::FunctionPass"> {
    let summary = "Optimize tile op with concat op user by convert it as dma and fuse unnecessary copy ops";

    let description = [{
        AI-generated summary

        This pass optimizes Tile operations followed by ConcatView and clustered Copy by converting
        them into PerAxisTileDMA operations.

        Rewriter: FuseTileWithConcatClusteredCopy
        Matches the pattern: SwKernel(Tile) -> ConcatView -> CopyOp (with DistributedBuffer output)
        and replaces it with PerAxisTileDMA when:
        - The Tile repeats data along a single axis
        - The output is a distributed buffer (multi-cluster)
        - The tile and concat can be expressed as a single DMA transfer
    }];

    let constructor = "vpux::VPUIP::createOptimizeTileOpAsNNDMAPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// OptimizeExpandSubview
//

def OptimizeExpandSubview : PassBase<"optimize-expand-subview", "vpux::FunctionPass"> {
    let summary = "Optimize Expand-Subview patterns on the same dimension to minimize large data concatenation in DDR";

    let description = [{
        This pass optimizes the Expand-Subview pattern when both Expand and Subview operate on the same dimension
        A Subview within the non-expanded part can directly connect to the input of the Expand operation
        A Subview that includes the expanded part can connect with a small data concatenation
        This optimization avoids large data concatenation operations in DDR memory

        Target Pattern:
                                            Input (C: m)
                                                 |
                                            Expand (At C: m + q)
             ____________________________________|____________________________________________
            |                                    |                                            |
        Subview_0 (At C: p0 < m)   Subview_1 (At C: p0 + p1 < m)   ...   Subview_n (At C: p0 + p1 + ... + pn > m)

        Convert to:
                                            Input (C: m)
             ____________________________________|____________________________________________________________________________
            |                                    |                                               |                            |
        Subview_0 (At C: p0 < m)   Subview_1 (At C: p0 + p1 < m)   ...   Subview_n-1 (At C: p0 + p1 + ... + p_n-1 < m)     Subview_n
                                                                                                                              |
                                                                                                                            Expand
    }];

    let constructor = "vpux::VPUIP::createOptimizeExpandSubviewPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ConvertExpand
//

def ConvertExpand : PassBase<"convert-expand", "vpux::FunctionPass"> {
    let summary = "Convert Expand that cannot fuse with permute to copy and concat subgraph";

    let description = [{
        Converts supported VPUIP.Expand operations into Copy-from-zero-constant and ConcatView
        data movement sequences that zero-fill padding regions in the destination buffer.

        When `defer-to-expand-dma` is true, Expand operations with integral element types or
        integral quantized storage types and all-zero pads_begin are left in place for
        downstream ConvertToDMA lowering to VPUIP.ExpandDMA. When false (the default), every
        supported case is decomposed here.
    }];

    let constructor = "vpux::VPUIP::createConvertExpandPass()";

    let options = [
        Option<
            "deferToExpandDMA", "defer-to-expand-dma",
            "bool", "false",
            "When true, leave integral-storage Expand with all-zero pads_begin in place so "
            "ConvertToDMA can later lower them to VPUIP.ExpandDMA; when false, decompose them "
            "here into the Copy-from-zero-constant + ConcatView pattern."
        >
    ];

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ConvertToDMA
//

def ConvertToDMA: PassBase<"convert-to-dma", "vpux::FunctionPass"> {
    let summary = "Convert Permute and DepthToSpace from SW ops to DMA ops";

    let description = [{
        AI-generated summary

        This pass converts several SW kernel operations and VPUIP operations to DMA equivalents
        when it is both legal and beneficial for performance.

        Rewrite patterns:
        - SwKernelMemPermuteConverter: Converts SwKernel MemPermute to PermuteDMA.
        - SwKernelDepthToSpaceConverter: Converts SwKernel DepthToSpace to DepthToSpaceDMA.
        - SwKernelSpaceToDepthConverter: Converts SwKernel SpaceToDepth to SpaceToDepthDMA.
        - SwKernelPerAxisTileConverter: Converts SwKernel PerAxisTile to PerAxisTileDMA.
        - ExpandConverter: Converts VPUIP.ExpandOp to ExpandDMA (always converted, marked illegal).
        - UpsamplingOpConverter: Converts VPUIP.UpsamplingOp (DDR-to-DDR only) to UpsamplingDMA.

        Operations with bounded (dynamic) buffers or sub-byte quantized types with unaligned
        dimensions are excluded from conversion.
    }];

    let constructor = "vpux::VPUIP::createConvertToDMAPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ActShaveProfiling
//

def ActShaveProfiling : PassBase<"act-shave-profiling", "vpux::ModulePass"> {
    let summary = "ActShave task profiling";

    let description = [{
        AI-generated summary

        Allocates CMX profiling buffer slots for each SHAVE task and inserts timestamp capture
        operations that record start and end cycle counts.
    }];

    let constructor = "vpux::VPUIP::createActShaveProfilingPass()";

    let dependentDialects = [
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// GroupProfilingBuffers
//

def GroupProfilingBuffers : PassBase<"group-profiling-buffers", "vpux::ModulePass"> {
    let summary = "Group profiling buffers into single profiling output buffer";

    let description = [{
        AI-generated summary

        This pass merges individual per-engine profiling buffers (DPU, DMA, ActShave, M2I) into
        a single combined profiling output buffer.

        The pass:
        1. Collects sizes of all profiling buffer sections from the network's ProfilingOutputsInfo.
        2. Computes aligned offsets for each section within a combined buffer, maintaining
           PROFILING_SECTION_ALIGNMENT between sections.
        3. Creates a single U32 output buffer sized to hold all profiling sections.
        4. Records section metadata (execution type, offset, size) for postprocessing tools.
        5. Replaces individual profiling function arguments with SubView slices of the combined
           buffer at the computed offsets.
        6. Updates the function signature to expose the single combined profiling output.
    }];

    let constructor = "vpux::VPUIP::createGroupProfilingBuffersPass()";
}

//
// UpdateSwKernelParams
//

def UpdateSwKernelParams : PassBase<"update-sw-kernel-params", "vpux::FunctionPass"> {
    let summary = "Update parameters sent to sw kernel for selected operations";

    let description = [{
        AI-generated summary

        This pass updates SHAVE kernel parameters that depend on resolved tensor shapes.

        Rewriter: UpdateSwKernelParamsRewriter
        Applies to SwKernelOp operations with kernel entry names: softmax, log_softmax,
        log_softmax_topk, and log_softmax_peak. Updates the kernel run attributes with
        parameters computed from the final resolved output (or input for log_softmax_peak)
        shape, including axis, pad size, and dynamic shape information.
    }];

    let constructor = "vpux::VPUIP::createUpdateSwKernelParamsPass()";
}

//
// DumpStatisticsOfTaskOpsPass
//

def DumpStatisticsOfTaskOpsPass : PassBase<"dump-statistics-of-task-ops", "vpux::FunctionPass"> {
    let summary = "Dump the statistics of operations (used Task operations and weights compression)";

    let description = [{
        AI-generated summary

        This pass collects and reports statistics about the compiled model's operations, barriers,
        weights compression, and constant swizzling.

        The pass reports:
        - Input and output tensor sizes
        - DDR heap size
        - VPUIP task operation counts (DMA, DPU, SHAVE, etc.)
        - Barrier statistics (total count, producer/consumer counts)
        - Weights compression statistics (compressed vs. uncompressed sizes, compression ratios)
        - Constant swizzling statistics (count and sizes of swizzled constants)

        Operations not converted to DPU are flagged as warnings. Output is logged at info level.
    }];

    let constructor = "vpux::VPUIP::createDumpStatisticsOfTaskOpsPass()";
}

//
// CompressWeightsBTC
//

def CompressWeightsBTC : PassBase<"compress-weights-btc", "vpux::FunctionPass"> {
    let summary = "Compress binary data when possible using BitCompactor";

    let description = [{
        This pass applies bitcompactor to tensor binary data. The logic is the following:
        1. Find VPUIP::NNDMAOp with Const::DeclareOp source and VPURT::DeclareBufferOp target.
        2. Check that weights size matches minimal compression size.
        3. Compress weights.
        4. Wrap compressed weights to flat tensor shapes with UInt8 data type.
        5. Replace original VPUIP::NNDMAOp with VPUIP::DecompressDMAOp

        This pass also handles multicluster cases, where NNDMAOp has DistributedBuffer output type (with DUPLICATED).
        SOK case is supported as well since this pass is set to be executed after unroll-distributed-ops,
        which splits SEGMENTED buffers into per-cluster chunks.

        Compression is performed using the BitCompactor library.
        When huffmanEnable = true, the BitCompactor encoder is configured to use its Huffman path (where supported)
        instead of regular BTC compression.
    }];

    let constructor = "vpux::VPUIP::createCompressWeightsBTCPass()";

    let options = [
        Option<
            "huffmanEnable", "huffman-enable",
            "bool", "false",
            "Flag to enable BTC Huffman compression"
        >,
        Option<
            "failIfNoCompression", "fail-if-no-compression",
            "bool", "false",
            "Flag to enable failing the pass when no compression was applied"
        >
    ];
}

//
// UnrollExpandDMA
//

def UnrollExpandDMA : PassBase<"unroll-expand-dma", "vpux::FunctionPass"> {
    let summary = "Unroll expand task with several NN DMA tasks";

    let description = [{
        AI-generated summary

        This pass unrolls ExpandDMA tasks into per-cluster NNDMA operations.

        Rewrite patterns:
        - SingleClusterExpandDMARewriter: Handles single-cluster ExpandDMA by splitting the expand
          into multiple NNDMAs that copy original data and zero-fill padding regions.
        - MultiClusterExpandDMARewriter: Handles multi-cluster (DistributedBuffer) ExpandDMA by
          creating per-cluster expand operations with DMA port balancing.

        The pass checks UnrollDMAAnalysis to skip execution when no ExpandDMA operations exist.
    }];

    let constructor = "vpux::VPUIP::createUnrollExpandDMAPass()";
}

//
// UnrollGatherDMA
//

def UnrollGatherDMA : PassBase<"unroll-gather-dma", "vpux::FunctionPass"> {
    let summary = "Unroll gather DMA task with several NN DMA tasks";

    let description = [{
        AI-generated summary

        This pass unrolls GatherDMA tasks into per-cluster NNDMA operations for multi-cluster
        execution.

        Rewriter: GatherDMARewriter
        Handles DistributedBuffer inputs by splitting indices and output buffers per cluster.
        Supports both DUPLICATED indices (input is split per cluster) and SEGMENTED indices
        (input is shared across clusters). Assigns DMA ports in round-robin fashion and marks
        unpaired cluster operations as split candidates when indices alignment permits.

        The pass checks UnrollDMAAnalysis to skip execution when no GatherDMA operations exist.
    }];

    let constructor = "vpux::VPUIP::createUnrollGatherDMAPass()";
}

//
// UnrollPerAxisTileDMA
//

def UnrollPerAxisTileDMA : PassBase<"unroll-per-axis-tile-dma", "vpux::FunctionPass"> {
    let summary = "Split PerAxisTileDMA task with several NN DMA tasks";

    let description = [{
        This pass splits PerAxisTileDMA tasks into several NN DMA tasks, which are functionally equivalent.
        Each sub PerAxisTileDMA will be converted to a NNDMA.
    }];

    let constructor = "vpux::VPUIP::createUnrollPerAxisTileDMAPass()";
}

//
// UnrollUpsamplingDMA
//

def UnrollUpsamplingDMA : PassBase<"unroll-upsampling-dma", "vpux::FunctionPass"> {
    let summary = "Unroll upsampling task with several NN DMA tasks";

    let description = [{
        This pass unroll UpsamplingDMA tasks with several NN DMA tasks, which are functionally equivalent.
        Each sub UpsamplingDMA will be converted to a NNDMA.
    }];

    let constructor = "vpux::VPUIP::createUnrollUpsamplingDMAPass()";
}

//
// AssignShvLogicalTaskIndex
//

def AssignShvLogicalTaskIndex : PassBase<"assign-shv-logical-task-index", "vpux::ModulePass"> {
    let summary = "Assign a globally unique logical_task index to IO-DMA SHV kernels";

    let description = [{
        Tags every IO-DMA submitting SW kernel (a SwKernelOp whose entry is in SW_KERNELS_IO_DMA) with
        a `logical_task` integer attribute, handing out sequential indexes with a single module-level
        walk so the numbering is globally unique across vertical-fusion outlined sub-functions.
        UnrollSwKernel later copies the attribute onto the unrolled tasks, and LegalizeShaveSubmitDMAs
        groups SHV tasks by it (one barrier per logical task). Assigning at module scope (single
        threaded) keeps the indexes unique without the thread-unsafe cross-function access a
        per-function pass would otherwise need.
    }];

    let constructor = "vpux::VPUIP::createAssignShvLogicalTaskIndexPass()";
}

//
// UnrollSwKernel
//

def UnrollSwKernel : PassBase<"unroll-sw-kernel", "vpux::FunctionPass"> {
    let summary = "Unroll SwKernel task with several SwKernel.Run";

    let description = [{
        AI-generated summary

        This pass unrolls SwKernel tasks that contain multiple SwKernel.Run operations into
        individual SwKernel tasks, each wrapping a single SwKernel.Run.

        Rewriter: SwKernelRewriter
        For each SwKernelOp with multiple SwKernelRun regions, creates separate SwKernelOp
        instances per run with the corresponding input/output operands mapped from block
        arguments to the outer SwKernelOp operands. Handles:
        - Dynamic shapes (BoundedBuffer operands with shape maps)
        - Profiling metadata updates (per-tile profiling buffer indices)
        - Optional dedicated FIFO per SHAVE engine (controlled by
          enableSwKernelFifoPerShaveEngine option), assigning listIndex attributes
    }];

    let constructor = "vpux::VPUIP::createUnrollSwKernelPass()";

    let options = [
        Option<
            "enableSwKernelFifoPerShaveEngine", "enable-sw-kernel-fifo-per-shave-engine",
            "bool", "",
            "Enable separate SW kernel FIFO for each Shave engine"
        >
    ];

}

//
// ConvertTransferOpsToDMAs
//

def ConvertTransferOpsToDMAs : PassBase<"convert-transfer-ops-to-DMAs", "vpux::FunctionPass"> {
    let summary = "Convert CopyOp and TimestampOp operations to low-level NNDMAOp operations";

    let description = [{
        AI-generated summary

        Converts VPUIP.CopyOp to VPUIP.NNDMAOp and VPUIP.TimestampOp to NNDMAOp reading
        from the hardware timer register (architecture-specific address).
    }];

    let constructor = "vpux::VPUIP::createConvertTransferOpsToDMAsPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvertAllocationsToDeclarations
//

def ConvertAllocationsToDeclarations : PassBase<"convert-allocations-to-declarations", "vpux::FunctionPass"> {
    let summary = "Convert dynamic buffer allocations to static declarations";

    let description = [{
        AI-generated summary

        Converts VPUIP.StaticAllocOp to VPURT.DeclareBufferOp, replacing abstract memory
        allocations with buffer declarations that carry explicit memory section and byte
        offset. Required before codegen because the runtime uses flat buffer declarations,
        not alloc operations.
    }];

    let constructor = "vpux::VPUIP::createConvertAllocationsToDeclarationsPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvertFuncArgsToDeclarations
//

def ConvertFuncArgsToDeclarations : PassBase<"convert-func-args-to-declarations", "vpux::ModulePass"> {
    let summary = "Replace use of function arguments with result of DeclareBuffer";

    let description = [{
        Operands that are network arguments are replaced by the result of DeclareBuffer operation
        with the corresponding buffer section (NetworkInput/NetworkOutput)
    }];

    let constructor = "vpux::VPUIP::createConvertFuncArgsToDeclarationsPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvertViewOpsToDeclarations
//

def ConvertViewOpsToDeclarations : PassBase<"convert-view-ops-to-declarations", "vpux::FunctionPass"> {
    let summary = "Convert view-like operations to buffer declarations with computed offsets";

    let description = [{
        AI-generated summary

        Converts view operations (SubView, PermuteCast, GenericReshape, QuantizeCast, ShapeCast)
        to VPURT.DeclareBufferOp with computed byte offset and strides. Traces alias chains from
        ViewOp results back to root buffer declarations to compute the final memory offset.
    }];

    let constructor = "vpux::VPUIP::createConvertViewOpsToDeclarationsPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// LinearizeCallOps
//

def LinearizeCallOps : PassBase<"linearize-call-ops", "vpux::FunctionPass"> {
    let summary = "Ensure Call operations are executed sequentially";

    let description = [{
        Ensures the Call operations found in the function have dependencies between each other,
        so that they are executed sequentially. This is necessary to ensure that the
        operations found within the called functions are called in a deterministic order,
        after the calls are inlined.

        Note: this pass expects the function to contain only Call operations.
    }];

    let constructor = "vpux::VPUIP::createLinearizeCallOpsPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvertAsyncOpsToTasks
//

def ConvertAsyncOpsToTasks : PassBase<"convert-async-ops-to-tasks", "vpux::FunctionPass"> {
    let summary = "Convert Async Dialect operations to tasks";

    let description = [{
        This pass inlines 'async.execute' body to parent Block and replaces '!async.token' based dependencies with
        VPUIP virtual barriers.
    }];

    let constructor = "vpux::VPUIP::createConvertAsyncOpsToTasksPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// Swizzling
//

def Swizzling : PassBase<"swizzling", "vpux::FunctionPass"> {
    let summary = "Configure swizzling for eligible buffers";

    let description = [{
        AI-generated summary

        Enables swizzling for eligible buffers on all architectures with swizzling support
        (VPUX37XX, VPUX40XX, VPUX50XX, and later). The pass is controlled by two independent
        flags: `enableWeightsSwizzling` and `enableActivationSwizzling`.

        The pass performs two iterations over NCEClusterTask operations:
        1. Identifies operations whose weight constants (weights, weights table, weights sparsity
           map) can be swizzled, applying the swizzling transformation to constant content.
        2. Identifies operations whose activation output buffers can be swizzled (DPU-to-DPU
           buffers where both producer and all consumers are NCEClusterTask operations).

        For Eltwise operations, both inputs must have matching swizzling attributes. The pass
        iteratively disables activation swizzling on mismatched Eltwise inputs until consistency
        is achieved. Fused constants are handled separately with a dedicated enable flag.
        Before enabling swizzling, CMX usage is verified across all operand permutations to
        ensure the aligned buffer sizes do not exceed available CMX capacity.

        Swizzling requirements:
        - buffer needs to be properly aligned
        - swizzled buffers must be in CMX space with size a multiple of 512
        - activation buffer must be produced and consumed by DPU type tasks
        - weights require a swizzling transformation on constant content
        - 5-d tensors are excluded from swizzling

        Device supported swizzling keys:
        - 0: 16 bytes alignment
        - 1: 1024 bytes alignment
        - 2: 2048 bytes alignment
        - 3: 4096 bytes alignment
        - 4: 8192 bytes alignment
        - 5: 16384 bytes alignment
    }];

    let constructor = "vpux::VPUIP::createSwizzlingPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
    let options = [
        Option<
            "enableWeightsSwizzling", "enable-weights-swizzling",
            "bool", "true",
            "Enables weights swizzling"
        >,
        Option<
            "enableActivationSwizzling", "enable-activation-swizzling",
            "bool", "true",
            "Enables activation swizzling"
        >
    ];
}

//
// DMA Barrier Optimization
//

def BarrierOptimization : PassBase<"barrier-optimization", "vpux::FunctionPass"> {
    let summary = "Optimize and merge redundant barrier synchronization points";

    let description = [{
        AI-generated summary

        Merges barriers whose producers are FIFO-ordered (all producers of barrier B follow
        all producers of barrier A). Orders barriers by largest producer index and iteratively
        merges compatible pairs within each task block. Removes barriers with no producers
        or consumers after merging.
    }];

    let constructor = "vpux::VPUIP::createBarrierOptimizationPass()";
    let options = [
        Option<
            "workloadManagementModeOpt", "workload-management-mode",
            "vpux::WorkloadManagementMode", "",
            "Setting which controls mode of WLM",
            [{::llvm::cl::values(
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V0_1_PAGES, "PWLM_V0_1_PAGES", "PWLM with split into subgraphs (pages)"),
                clEnumValN(vpux::WorkloadManagementMode::FWLM_V1_PAGES, "FWLM_V1_PAGES", "Full WLM with split into pages")
            )}]
        >
    ];
}

//
// OperationStubbing
//

def OperationStubbing : PassBase<"operation-stubbing", "vpux::FunctionPass"> {
    let summary = "Replace selected operations with stub operations based on a condition predicate";

    let description = [{
        AI-generated summary

        Conditionally transforms operations into stub replacements (IE::StubOp, VPU::StubOp,
        or VPUIP::StubOp) that preserve type and operand information while bypassing actual
        execution. Supports operations from IE, VPU, and VPUIP dialects, enabling selective
        operation elimination for debugging and testing purposes.
    }];

    let constructor = "vpux::VPUIP::createOperationStubbingPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvWeightsCompression
//

def ConvWeightsCompression : PassBase<"conv-weights-compression", "vpux::FunctionPass"> {
    let summary = "Compress weights of Conv operation";

    let description = [{
        AI-generated summary

        This pass applies weights compression for NCEClusterTask convolution operations whose
        weights were previously padded with zeros along the input channels dimension.

        For each NCEClusterTaskOp with a weight table, the pass checks whether compression is
        applicable via canWeightsBeCompressed. Two cases are handled:

        1. Padding done only on IC (input channels): The pass strips the zero-padding from the
           weights constant by creating a SubView of the original shape, updates the weights
           table pointers to reflect the new per-OC strides, and creates a new CopyOp with the
           reduced constant. For distributed (tiled) weights, the DistributedBufferType and
           DistributionInfoAttr are recomputed with the new shape.

        2. General case: The pass sets cm_sp_pattern based on VPU_COMPRESSED_INPUT_CHANNEL_NUM
           without modifying the weights data.

        In both cases, a VPUIP.ShapeCast is inserted on the weights input of the NCEClusterTask
        to reshape the weights to [OC, channelAlignment, KY, KX] and the cm_sp_pattern attribute
        is set on the new NCEClusterTask clone.
    }];

    let constructor = "vpux::VPUIP::createConvWeightsCompressionPass()";
}

//
// FuseConstants
//

def FuseConstants : PassBase<"fuse-constants", "vpux::FunctionPass"> {
    let summary = "Fuse constant inputs of NCEClusterOp";

    let description = [{
        AI-generated summary

        Concatenates NCEClusterTaskOp constant inputs into a single fused constant in the
        following order: weight_table -> weights -> weights_sparsity_map.

        Rewriter: FuseConstants (greedy pattern on NCEClusterTaskOp)
        For each NCEClusterTaskOp with a weight table, the pass:
        1. Skips ELTWISE and AVEPOOL task types, operations already fused (constantsFused
           attribute set), and operations with isZeroOffsetWeightsTable.
        2. Locates the Const::DeclareOp and CopyOp for each of the three constants
           (weight_table, weights, weights_sparsity_map). All three CopyOps must have
           single use. If weights are not constants (e.g. activation tensors), fusion is skipped.
        3. Creates a new fused Const::DeclareOp using the Fuse content transformation, which
           concatenates the binary data of all constants into a single flat tensor (U8 or F16
           element type, chosen based on which dominates the total size).
        4. Allocates a CMX buffer for the fused constant (VPURT::AllocDistributed for tiled
           operations, memref::AllocOp for single-tile) and copies the fused constant into it.
        5. Replaces each original constant CopyOp with a SubView -> ViewOp chain that extracts
           the corresponding slice from the fused constant buffer.
        6. Sets the constantsFused attribute on the NCEClusterTaskOp.
    }];

    let constructor = "vpux::VPUIP::createFuseConstantsPass()";
}

//
// PropagateSparsityCompression
//

def PropagateSparsityCompression : PassBase<"propagate-compression-scheme", "vpux::FunctionPass"> {
    let summary = "Compresses the type of the sparse weights";

    let description = [{
        Propagates the compression scheme attribute from the sparse buffer type to the individual types
        that are grouped into a sparse buffer. Starting from the sparse weights constant, all types up to
        the consumer NCE operation will have the compression scheme present.
    }];

    let constructor = "vpux::VPUIP::createPropagateSparsityCompressionPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}


//
// FlattenSparseWeightsTypes
//

def FlattenSparseWeightsTypes : PassBase<"flatten-sparse-weights-types", "vpux::FunctionPass"> {
    let summary = "Flattens types that have a compression scheme";

    let description = [{
        AI-generated summary

        Flattens the types of sparse weights with compression schemes into binary buffers.
        The pass operates in three phases:

        1. Creates an explicit DeclareBuffer operand for each NCE weights operand that has a
           compression scheme, preserving the original dense type so the NCE task can correctly
           interpret the weights layout.
        2. Enables the compressed output flag on the Sparsify transformation of weight constants,
           allowing the constant's output type to be flattened.
        3. Walks all operations and flattens the shapes of operands and results that carry a
           compression scheme into 1-D U8 tensors sized to the compressed allocation. NCE weights
           operands are excluded from flattening because they retain the original dense type from
           phase 1.
    }];

    let constructor = "vpux::VPUIP::createFlattenSparseWeightsTypesPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ComputeSESizes
//

def ComputeSESizes : PassBase<"compute-se-sizes", "vpux::FunctionPass"> {
    let summary = "Computes the storage element sizes for sparse NCEClusterTask activations";

    let constructor = "vpux::VPUIP::createComputeSESizesPass()";

    let description = [{
        Computes the storage element sizes for the sparse activations of NCEClusterTasks.
        The pass should be called twice:
        - once with the `onlyInputsConcatOverC` set to true before use-def chains are lost,
          in order to populate the correct information for consumers of buffers that have been
          concatenated over channels; these can be explicit concat operations, broadcasted
          activations or produced by multiple variants;
        - once after cluster tiling operations are unrolled, to populate the sizes for the
          rest of the sparse activations based on the number of channels present after unrolling.

        Eltwise inputs must have both inputs generated with the same storage element size value.
    }];

    let options = [
        Option<
            "onlyInputsConcatOverC", "only-inputs-concat-over-c",
            "bool", "false",
            "Flag to choose whether to handle only inputs concatenated over channels on this pass call"
        >
    ];

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ComputeSEBasePtrs
//

def ComputeSEBasePtrs : PassBase<"compute-se-base-ptrs", "vpux::FunctionPass"> {
    let summary = "Computes the base pointers for storage elements";

    let description = [{
        Computes the base pointers for NCEClusterTasks and stores them into the
        input storage element table operation.

        The pass will identify the NCE operations that have the input storage element
        table present and whose input data is distributed. In case the input data
        is not distributed, the base pointers will be considered to be the implicit
        zero value.

        The input data distribution across clusters will determine the value of the
        base pointers. For example, let's take an interpolate operation whose input
        data is 3x3 and segmented over height across two clusters as follows:
        1 2 3   <- cluster 0
        4 5 6   <- cluster 0
        7 8 9   <- cluster 1

        If the interpolate configuration is nearest-neighbor with a scale of 2, the
        following elements would be identified by the storage element pointers:
        1 1 2 2 3 3
        1 1 2 2 3 3
        4 4 5 5 6 6
        4 4 5 5 6 6
        7 7 8 8 9 9
        7 7 8 8 9 9

        The following base pointers are determined:
        0 0 0 0 0 0
        0 0 0 0 0 0
        0 0 0 0 0 0
        0 0 0 0 0 0
        1 1 1 1 1 1
        1 1 1 1 1 1

        The distribution across clusters of the storage element table will not
        influence the computation of the base pointers, since they are used during
        inference to determine the physical location of the input data.

        Note: Eltwise operations with input storage element tables are not yet
        supported, so the pass will fail in case such an operation is encountered.
    }];

    let constructor = "vpux::VPUIP::createComputeSEBasePtrsPass()";
    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ConvertSETablesToConstants
//

def ConvertSETablesToConstants : PassBase<"convert-se-tables-to-constants", "vpux::FunctionPass"> {
    let summary = "Converts Storage Element Table operations to constants";

    let constructor = "vpux::VPUIP::createConvertSETablesToConstantsPass()";

    let description = [{
        Computes the Storage Element pointers for every Storage Element Table operation
        and stores them into a constant. The original operation is replaced by the new constant.

        The Storage Element pointers have the following format:
            31-29 28                            9 8         0
            -------------------------------------------------
            | xx |           DATA_PTR            | BASE_PTR |
            -------------------------------------------------
        The DATA_PTR represents the offset to a Storage Element in relation to the start of
        the input data. It is computed based on the information about the input data
        (e.g. shape, element byte-size) and, if present, using the value of the SEAttr from
        the Storage Element Table operation.

        BASE_PTR is used to decide what base address is added to DATA_PTR in order to find the
        location of the Storage Element in memory during inference. Its value is determined by
        the presence of the basePtrs attribute of the operation, which should have one value
        for every Storage Element pointer. The default value zero is used if the basePtrs
        attribute is missing.
    }];

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// AdjustInputDataForExplicitSETable
//

def AdjustInputDataForExplicitSETable : PassBase<"adjust-input-data-for-explicit-se-table", "vpux::FunctionPass"> {
    let summary = "Adjust input data type when a Storage Element table is present";

    let constructor = "vpux::VPUIP::createAdjustInputDataForExplicitSETablePass()";

    let description = [{
        For cases with explicit input Storage Element tables, the input data might have a different
        shape in memory compared to the way the IDU reads this data using the pointers inside the
        table. For the DPU to properly read the input, it has to interpret the data as having the
        shape represented by the table.

        This pass alters the view of the input data for the NCE operation with explicit Storage
        Element tables, so that it reflects the effective data represented by the table.
    }];

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ResolveDMAWithSwizzling
//

def ResolveDMAWithSwizzling : PassBase<"resolve-dma-with-swizzling", "vpux::FunctionPass"> {
    let summary = "Transform DMAs that operate on swizzled buffers";

    let description = [{
        AI-generated summary

        Transforms DMA operations on swizzled buffers whose real data size (shape times stride)
        does not match the total allocation size (which is aligned to 512 for swizzling).

        For each such DMA, the pass creates new flat tensor shapes that cover the full
        512-aligned allocation size. The source operand is replaced: if a constant, its shape
        is flattened in place; if a DeclareBufferOp, a new flat buffer declaration is created.
        The destination DeclareBufferOp is similarly replaced with a flat buffer of aligned size.
        A new NNDMAOp is then created to copy between the flat representations.

        Activation compression and decompression are handled as special cases: for compression
        (CMX source), only the input is required to be size-aligned; for decompression (DDR
        source), only the output must be aligned and the output buffer's allocation size is used
        for computing the flat tensor dimensions.
    }];

    let constructor = "vpux::VPUIP::createResolveDMAWithSwizzlingPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// TileActShaveKernelTask
//

def TileActShaveKernelTask : PassBase<"tile-act-shave-kernel-task", "vpux::FunctionPass"> {
    let summary = "Tile act shave kernel task on multi shaves";

    let description = [{
        AI-generated summary

        Splits SwKernelOp tasks across available SHAVE engines by tiling input/output tensors
        along a chosen dimension. Creates per-SHAVE SwKernel.Run entries so each SHAVE engine
        processes a subset of the data in parallel. Uses SwKernelRewriter for single-tile and
        ClusterSwKernelRewriter for distributed (multi-cluster) operations.
    }];

    let constructor = "vpux::VPUIP::createTileActShaveKernelTaskPass()";
}

//
// SetZeroOffsetWeightsTable
//

def SetZeroOffsetWeightsTable : PassBase<"set-zero-offset-weights-table", "vpux::FunctionPass"> {
    let summary = "Set NCEClusterTask isZeroOffsetWeightsTable attribute";

    let description = [{
        AI-generated summary

        Identifies NCEClusterTask operations whose weights table pointers can be relative to the start of the
        weights base address (i.e. the pointers start from zero) and adds the `isZeroOffsetWeightsTable`
        attribute to them. This will prevent future passes like PatchWeightsTable and
        PatchPopulateWeightTableWithShave to patch the weights table content, keeping the data offset starting from 0.

        The pass only applies to CONV operations (`NCETaskType::CONV`) where `isWeightsTableReuseEnabled`
        returns true.  Operations with sparse weights (non-null `weightsSparsityMap`) are skipped.

        This allows such operations to share the weights table buffer, if the pointer steps are the same.
        Compared to the operations targeted by this pass, the remaining NCEClusterTasks will have their
        weights table pointers relative to the start of the CMX address, which prevents the table from
        being shared.

        Currently, only operations that have sparse weights are not marked with `isZeroOffsetWeightsTable`,
        because we're forced to have a common base address for both the data pointer and sparsity pointer.
        Also the pointers inside the table are highly unlikely to be the same across multiple operations,
        meaning that the table contents are not similar.
    }];

    let constructor = "vpux::VPUIP::createSetZeroOffsetWeightsTablePass()";
}

//
// SegmentHalos
//

def SegmentHalos : PassBase<"segment-halos", "vpux::FunctionPass"> {
    let summary = "Segment halos according to DPUTask workloads";

    let description = [{
        Halos are a per-workload feature, meaning each workload will need to be programmed to send its halos separately.

        Initially, each halo region represents the entire halo slice sent/received by the current NCEClusterTask. However,
        the NCEClusterTask can be further segmented into workloads. To be able to lower the halo information from ITI Buffer
        to the workloads, we will need to split the halos so that each one of them that spans more than one workload is
        divided among them. That is the purpose of this pass.

        E.g.

        Full shape produced by cluster = [1, 16, 32, 17]
        Initial outward halo: shape = [1, 16, 32, 1], offset = [0, 0, 0, 16]

        Workload 0: outStart = [0, 0, 0] - outEnd = [16, 15, 15]
        Workload 1: outStart = [0, 16, 0] - outEnd = [16, 31, 15]

        After pass:
            Halo0: shape = [1, 16, 16, 1], offset = [0, 0, 0, 16]
            Halo1: shape = [1, 16, 16, 1], offset = [0, 0, 16, 16]
    }];

    let constructor = "vpux::VPUIP::createSegmentHalosPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// Adjust activation spilling size
//

def AdjustSpillSize : PassBase<"adjust-spill-size", "vpux::FunctionPass"> {
    let summary = "Adjusts activation spilling buffer size in DDR";

    let description = [{
        AI-generated summary

        This pass prepares DDR allocation for handling hardware activation spilling compression.
        In the worst case, compressed activation data can be larger than the original, so DDR
        spill buffers must be enlarged to prevent memory corruption.

        The pass walks all NNDMAOp operations that have a spillId attribute:
        - For spill-write (CMX -> DDR): If the source buffer size is supported for compression,
          the DDR allocation is enlarged using updateSizeForCompression (which accounts for
          worst-case compression overhead). For SEGMENTED or OVERLAPPED distributions, each
          per-cluster chunk is independently sized. The buffer type is updated with the new
          allocation size and marked as CompressionCandidate. The async result types are also
          updated.
        - For spill-read (DDR -> CMX): The input block argument type is updated to match the
          enlarged type from the corresponding spill-write, and the DMA is marked as a
          compress candidate.
    }];

    let constructor = "vpux::VPUIP::createAdjustSpillSizePass()";
}

//
// FuseDDRCopiesIntoConcats
//

def FuseDDRCopiesIntoConcats : PassBase<"fuse-ddr-copies-into-concats", "vpux::FunctionPass"> {
    let summary = "Fuse VPUIP.Copy into adjacent VPUIP.ConcatView";

    let description = [{
        AI-generated summary

        Fuses redundant DDR-to-DDR Copy operations into ConcatView operations.

        Rewriter: FuseCopies (greedy pattern on VPUIP.ConcatViewOp)
        Matches the pattern:
            CopyOp (CMX->DDR) -> memref::AllocOp -> CopyOp (DDR->DDR) -> ConcatViewOp (DDR)
        and fuses the two CopyOps into a single CMX-to-DDR Copy that writes directly into
        the ConcatView output buffer, eliminating the intermediate DDR allocation and DDR-to-DDR
        copy.

        Eligibility checks:
        - The ConcatView input must be produced by a DDR-to-DDR CopyOp
        - The DDR-to-DDR CopyOp input must come from a non-distributed CopyOp
        - The intermediate buffer must be allocated via memref::AllocOp (not a block argument)

        After fusion, unused intermediate CopyOps and their AllocOps are erased.
    }];

    let constructor = "vpux::VPUIP::createFuseDDRCopiesIntoConcats()";
}

//
// NNDMATiling
//

def NNDMATiling : PassBase<"nn-dma-tiling", "vpux::FunctionPass"> {
    let summary = "Legalizes NNDMA Ops which do not fit hardware capabilities";

    let description = [{
        AI-generated summary

        This pass splits NNDMAOp operations that exceed hardware DMA capabilities into multiple
        smaller tiles. Unlike the CopyOpTiling pass which handles CopyOp before unrolling,
        this pass operates on NNDMAOp after unroll-distributed-ops.

        Rewriter: SplitNNDMARewriter (greedy pattern on VPUIP.NNDMAOp)
        Splits an NNDMAOp when:
        - DMA data size exceeds the architecture-specific maximum transfer length
          (e.g. 16MB for some architectures)
        - Number of planes exceeds the maximum supported by the DMA engine
          (architecture-dependent, e.g. 255 planes)

        Tiling strategy:
        - Selects a tiling dimension using getCopyDMATilingDim (or
          getCopyDMATilingDimForLargePlaneNum for plane-count violations)
        - Computes balanced tile sizes to evenly distribute work
        - Creates per-tile VPURT::DeclareBufferOp declarations with adjusted byte offsets
        - Wraps each tile in a VPURT::TaskOp with the original barrier dependencies
        - Distributes tiles across DMA ports in round-robin fashion
        - Handles sparse buffer types by selecting the maximum component size for tiling
    }];

    let constructor = "vpux::VPUIP::createNNDMATilingPass()";
}

//
// UngroupBoundedBuffer
//

def UngroupBoundedBuffers : PassBase<"ungroup-bounded-buffers", "vpux::FunctionPass"> {
    let summary = "Ungroup all values (data and shape) of BoundedBufferType";

    let description = [{
        Feasible allocation does not support multiple root buffers per input, therefore separate processing of dynamic data
        and dynamic shape is required.

        This pass consists of two parts:
          * Ungroup CopyOp. All CopyOp working with BoundedBuffer are split in two CopyOp working with individual memrefs.
          * Ungroup SwKernelOp. All SwKernelOp working with BoundedBuffer are updated to work with data and dims individually.
    }];

    let constructor = "vpux::VPUIP::createUngroupBoundedBuffersPass()";
    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

def ConvertDynamicMemrefsToBoundedBuffers : PassBase<"convert-dynamic-memrefs-to-bounded-buffers", "vpux::ModulePass"> {
    let summary = "Convert dynamic memrefs to bounded buffers";

    let description = [{
        This pass converts dynamic memrefs to bounded buffers, which is, at the
        beginning of VPUIP, a more convenient buffer representation for the
        compiler. This pass is a compatibility layer to transition from
        MLIR-style dynamic memrefs to NPU-style type system.
    }];

    let constructor = "vpux::VPUIP::createConvertDynamicMemrefsToBoundedBuffersPass()";
    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// UngroupFuncArgs
//

def UngroupBoundedBuffersAsFuncArgs : PassBase<"ungroup-bounded-buffers-as-func-args", "vpux::ModulePass"> {
    let summary = "Ungroup func inputs/outputs if they are VPUIP.BoundedBuffers";

    let description = [{
        In this pass each !VPUIP.BoundedBuffer function input/output argument is replaced by two separate
        memref arguments representing data and dynamic_shape.
        Pass also introduces two new operations:
        - VPUIP.GroupBoundedBuffer that groups the input data and dynamic_shape memref arguments into !VPUIP.BoundedBuffer types at the begin of func
        - VPUIP.UngroupBoundedBuffer that ungroups the !VPUIP.BoundedBuffer at the end of func into separate memref types for data and dynamic_shape.
        Pass also makes changes in NetworkInfo accordinally
    }];

    let constructor = "vpux::VPUIP::createUngroupBoundedBuffersAsFuncArgsPass()";
    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

def UngroupHostBuffersAsFuncArgs : PassBase<"ungroup-host-buffers-as-func-args", "vpux::ModulePass"> {
    let summary = "Expose hidden bounded-buffer shapes at func boundaries";

    let description = [{
        This pass is similar to UngroupBoundedBuffersAsFuncArgs, but targets hidden boundaries.
        It detects hidden bounded-buffer boundaries introduced by one-shot bufferization,
        where dynamic memref boundaries are represented by Group/Ungroup and ReinterpretCast
        operations inside function bodies, and makes the shape explicit in function args/results.

        This pass also updates call sites and NetworkInfo with shape DataInfo entries.
    }];

    let constructor = "vpux::VPUIP::createUngroupHostBuffersAsFuncArgsPass()";
    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

def BatchMatMulToMatMul : PassBase<"batch-matmul-to-matmul", "vpux::FunctionPass"> {
    let summary = "Converts NCE MatMul tasks with batch > 1 to NCE MatMul tasks with batch = 1";

    let description = [{
        Transform 5-d workloads of an NCEClusterTask:
        ```
        %IN = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>
        %MATMUL = VPUIP.NCEClusterTask input(%5 : memref<2x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
        ```
        into serveral 4-d memref values:
        ```
        %IN_0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x1xf16, #NHWC, [@CMX_NN, 0]>
        %MATMUL_0 = VPUIP.NCEClusterTask input(%IN_0 : memref<1x16x4x1xf16, #NHWC, [@CMX_NN, 0]>)

        %IN_1 = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x16x4x1xf16, #NHWC, [@CMX_NN, 0]>
        %MATMUL_1 = VPUIP.NCEClusterTask input(%IN_1 : memref<1x16x4x1xf16, #NHWC, [@CMX_NN, 0]>)
        ```
    }];

    let constructor = "vpux::VPUIP::createBatchMatMulToMatMulPass()";
}

//
// AddCopyBetweenSWKernelsAndNetworkIO
//

def AddCopyBetweenSWKernelsAndNetworkIO : PassBase<"add-copy-between-swkernels-and-network-io", "vpux::ModulePass"> {
    let summary = "Adds NNDMA ops between SwKernls and NetworkInput or NetworkOutput";

    let description = [{
        Inserts NNDMA Ops before/after SWKernels that directly access NetworkInput/NetworkOutput,
        to avoid restricting NetworkInput and NetworkOutput section allocation to the ActSHAVE address range.
    }];

    let constructor = "vpux::VPUIP::createAddCopyBetweenSWKernelsAndNetworkIOPass()";
}

//
// ConvertVPUIPCopyToSWCopy
//

def ConvertVPUIPCopyToSWCopy : PassBase<"convert-vpuip-copy-to-sw-copy", "vpux::ModulePass"> {
    let summary = "Convert VPUIP.Copy to SW.Kernel Copy";

    let description = [{
        AI-generated summary

        Converts VPUIP.CopyOp operations to SwKernel Copy operations when the copy involves
        sub-byte element types (e.g. INT4/UINT4) with non-byte-aligned strides that cannot
        be executed as DMA operations.

        Rewriter: ConvertVPUIPCopyToSWCopy (partial conversion on VPUIP.CopyOp)
        A CopyOp is converted when:
        - The element type is sub-byte (SInt4 or UInt4), AND
        - Either strides are not byte-aligned, OR strides exceed dimension sizes with
          non-aligned innermost dimension

        The SW Copy kernel receives inBitOffset and outBitOffset attributes computed from
        any SubViewOp offsets on the input or output operands, indicating which nibble to
        start copying from. The offsets are transformed from logical dimension order to
        memory order.

        Restrictions:
        - Direct access to NetworkInput/NetworkOutput block arguments is not supported
        - The pass creates a runtime kernel definition and a built-in SW function for the
          copy operation on tile 0
    }];

    let constructor = "vpux::VPUIP::createConvertVPUIPCopyToSWCopyPass()";
}

//
// UnrollAnalysis
//

def UnrollDMAAnalysis : PassBase<"unroll-analysis", "vpux::FunctionPass"> {
    let summary = "Analysis pass to cache which unroll dma passes are needed";

    let description = [{
        AI-generated summary

        This pass creates and populates the UnrollDMAAnalysis cache. It traverses the IR and
        records which DMA operation types (ExpandDMA, GatherDMA, DepthToSpaceDMA,
        SpaceToDepthDMA, PermuteDMA, PerAxisTileDMA, UpsamplingDMA) are present and require
        unrolling.

        Downstream DMA unrolling passes (e.g. UnrollExpandDMA, UnrollPermuteDMA) query this
        analysis to skip execution entirely when no operations of their type exist, avoiding
        unnecessary IR traversal and reducing compile time for larger graphs.

        The analysis is marked as preserved so it can be reused by multiple subsequent passes
        without recomputation.
    }];

    let constructor = "vpux::VPUIP::createUnrollDMAAnalysisPass()";
}

//
// InvalidateUnrollAnalysis
//

def InvalidateUnrollDMAAnalysis : PassBase<"invalidate-unroll-dma-analysis", "vpux::FunctionPass"> {
    let summary = "Invalidates UnrollDMAAnalysis";

    let description = [{
        AI-generated summary

        Invalidates cached DMA unrolling analysis results when the IR has been modified by
        intervening passes. Marks the DMA split decision data as stale so that subsequent passes
        that query DMA unrolling will recompute the analysis from the current IR state.
    }];

    let constructor = "vpux::VPUIP::createInvalidateUnrollDMAAnalysisPass()";
}

//
// DispatchedInliner
//

def DispatchedInliner : PassBase<"dispatched-inliner", "vpux::ModulePass"> {
    let summary = "Inlines using VPUIP's FuncInlinerInterface";

    let description = [{
        AI-generated summary

        Inlines dispatched function calls by delegating to the MLIR built-in inliner.
        Sets the Core::InlinerDispatchAttr to VPUIPInlinerDispatchAttr (which provides
        VPUIP-specific inlining hooks via FuncInlinerInterface), then invokes
        mlir::createInlinerPass() to inline all CallOpInterface sites in the module.
    }];

    let constructor = "vpux::VPUIP::createDispatchedInlinerPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// UnrollDepthToSpaceDMA
//

def UnrollDepthToSpaceDMA : PassBase<"unroll-depth-to-space-dma", "vpux::FunctionPass"> {
    let summary = "Unroll DepthToSpaceDMA tasks";

    let description = [{
        AI-generated summary

        This pass unrolls DepthToSpaceDMA tasks to multiple new DepthToSpaceDMA tasks, where applicable.

        Main considerations when deciding to unroll a task to multiple new tasks are:
         - respecting HW restrictions, like maximum number of dimensions, maximum size of a dimension etc.
         - improving performance by distributing a task to all available HW ports.

        This pass makes use of per-architecture strategies.
        Currently, multi-cluster unrolling is common for all architectures and single-cluster unrolling
        is defined separately for NPU37XX and NPU40XX.

        For NPU37XX single-cluster unrolling:
        - if input/output layout is NHWC with model block_first, number of sub DepthToSpaceDMA is same as block_size.
        - if input/output layout is NHWC with model depth_first, number of sub DepthToSpaceDMA is OH * OW / block_size.
        block_size is the size as the spatial block. It is an attribute of DepthToSpace.
    }];

    let constructor = "vpux::VPUIP::createUnrollDepthToSpaceDMAPass()";
}

//
// UnrollSpaceToDepthDMA
//

def UnrollSpaceToDepthDMA : PassBase<"unroll-space-to-depth-dma", "vpux::FunctionPass"> {
    let summary = "Unroll SpaceToDepthDMA tasks";

    let description = [{
        This pass unrolls SpaceToDepthDMA tasks to multiple new SpaceToDepthDMA tasks, where applicable.

        Main considerations when deciding to unroll a task to multiple new tasks are:
         - respecting HW restrictions, like maximum number of dimensions, maximum size of a dimension etc.
         - improving performance by distributing a task to all available HW ports.

        This pass makes use of per-architecture strategies.
        Currently, multi-cluster unrolling is common for all architectures and single-cluster unrolling
        is defined separately for NPU37XX and NPU40XX.
    }];

    let constructor = "vpux::VPUIP::createUnrollSpaceToDepthDMAPass()";
}

//
// UnrollPermuteDMA
//

def UnrollPermuteDMA : PassBase<"unroll-permute-dma", "vpux::FunctionPass"> {
    let summary = "Unroll PermuteDMA tasks";

    let description = [{
        This pass unrolls PermuteDMA tasks to multiple new PermuteDMA tasks, where applicable.

        Main considerations when deciding to unroll a task to multiple new tasks are:
         - respecting HW restrictions, like maximum number of dimensions, maximum size of a dimension etc.
         - improving performance by distributing a task to all available HW ports.

        This pass makes use of per-architecture strategies.
        Currently, multi-cluster unrolling is common for all architectures and single-cluster unrolling
        is defined separately for NPU37XX and NPU40XX.
    }];

    let constructor = "vpux::VPUIP::createUnrollPermuteDMAPass()";
}

//
// AddSwKernelCacheHandlingOps
//

def AddSwKernelCacheHandlingOps : PassBase<"add-sw-kernel-cache-handling-ops", "vpux::FunctionPass"> {
    let summary = "Add cache-handling ops for SW Kernels";

    let description = [{
        This pass adds CACHE_FLUSH or CACHE_FLUSH_INVALIDATE when needed.

        CACHE_FLUSH will be added when at least one output buffer is located in DDR and at least one of its
        consumers is not an ActSHAVE such that the ActSHAVE cache is flushed after the ActSHAVE Kernel writes
        to DDR through the cache.

        CACHE_FLUSH_INVALIDATE will be added when at least one input buffer is located in DDR and it is not
        produced by an ActSHAVE or is a constant, such that the ActSHAVE cache is invalidated before the
        ActSHAVE Kernel reads from DDR through the cache.

        If there are two consecutive SW Kernels and a CACHE_FLUSH was added after the first one and a CACHE_INVALIDATE
        is needed before the second one, the resulting sequence of ops will be:
        [SW Kernel 1] - [CACHE_FLUSH] - [CACHE_FLUSH_INVALIDATE] - [SW Kernel 2].

        CACHE_INVALIDATE no longer used since it's possible that the SHAVE stack resides in DDR, in which case
        a CACHE_INVALIDATE on the Shave L2 cache causes the corruption of the SHAVE stack. The SHAVE jumps to a
        null address on completion of the L2 invalidate.
    }];

    let constructor = "vpux::VPUIP::createAddSwKernelCacheHandlingOpsPass()";
}

//
// UnrollDistributedOps
//

def UnrollDistributedOps : PassBase<"unroll-distributed-ops", "vpux::FunctionPass"> {
    let summary = "Extracts child operations from distributed operation";

    let description = [{
        AI-generated summary

        Unrolls distributed (multi-tile) operations into per-tile concrete operations. Replaces
        VPU.DistributedBuffer types with individual per-tile buffer declarations and generates
        explicit per-tile DMA copies, DPU tasks, and SHAVE invocations according to the
        distribution mode (SEGMENTED, DUPLICATED, OVERLAPPED).
    }];

    let constructor = "vpux::VPUIP::createUnrollDistributedOpsPass()";
}

//
// ComputeHaloRegionForDPUTaskOp
//

def ComputeHaloRegionForDPUTaskOp : PassBase<"compute-halo-region-for-dpu-task-op", "vpux::FunctionPass"> {
    let summary = "Add halo region information to each DPU Task Op.";

    let description = [{
        AI-generated summary

        This pass transfers halo region information from the ITIBuffer type of output and
        output_ITI operands of VPUIP.NCEClusterTask to each of its VPUIP.DPUTaskOps.

        For each NCEClusterTaskOp with an ITIBufferType output (multi-cluster with halos):
        1. Builds a mapping from each outward halo to the neighboring NCEClusterTaskOp that
           consumes it, by tracing through the output_ITI buffer operands.
        2. For each DPUTaskOp variant, checks whether its workload region [outStart, outEnd]
           intersects with each outward halo region. Non-intersecting halos are skipped.
        3. Computes per-workload DPUHaloRegionAttr containing: halo coordinate ranges,
           target byte offset (relative to the destination ITI buffer), target cluster IDs,
           sparsity offset (when sparse outputs are present), and target width for stride
           computation.
        4. Handles layout mismatches between input and output by applying dimension mapping.
        5. Replaces each DPUTaskOp with a new one carrying the computed haloRegions attribute.

        The pass is skipped for single-tile configurations.
    }];

    let constructor = "vpux::VPUIP::createComputeHaloRegionForDPUTaskOpPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// DMATaskProfilingHwDdr
//

def DMATaskProfilingHwDdr : PassBase<"dma-task-profiling-hw-ddr", "vpux::ModulePass"> {
    let summary = "DMA hardware task profiling";

    let description = [{
        AI-generated summary

        Assigns DMA HWP (Hardware Performance) IDs to profiled DMA tasks and allocates DDR
        profiling output buffers. Supports two modes: STATIC_HWP (assigns sequential IDs to
        all DMAs up to a limit) and DYNAMIC_HWP (manages profiling buffer copy-out DMAs
        after each group of measured tasks).
    }];

    let options = [
        Option<
            "enableDMAProfiling", "dma-profiling",
            "std::string", [{"false"}],
            "Enable DMA task profiling (true|static|false)"
        >
    ];

    let constructor = "vpux::VPUIP::createDMATaskProfilingHwDdrPass()";
}

//
// ConstantDpuProfHwpBase
//

def ConstantDpuProfHwpBase : PassBase<"constant-dpu-prof-hwp-base", "vpux::FunctionPass"> {
    let summary = "Sets single NCE HWP CMX base address";

    let description = [{
        In VPU4 NCE HWP CMX base address must remain the same for all DPU tasks since
        HW may capture the address from a subsequent descriptor before storing the profiling
        record for the task just completed. To prevent from such race condition NCE HWP CMX
        base address for each NCE task will be 0 and profiling slot addressing will be
        handled by workload_id.
    }];

    let constructor = "vpux::VPUIP::createConstantDpuProfHwpBasePass()";
}

//
// Compress DMA for activations
//

def CompressSpillDma : PassBase<"compress-spill-dma", "vpux::ModulePass"> {
    let summary = "CompressedDMA for activation spills";

    let description = [{
        AI-generated summary

        Converts eligible CMX-to-DDR spill NNDMAOps to CompressDMAOp and DDR-to-CMX spill
        reads to DecompressDMAOp. Manages a pool of reserved CMX slots for the act_comp_size
        buffer needed by hardware compression. Slot allocation tracks deallocation points
        to reuse slots across non-overlapping spills.
    }];

    let constructor = "vpux::VPUIP::createCompressSpillDmaPass()";
}

//
// OptimizeConvertDMAOp
//

def OptimizeConvertDMAOp : PassBase<"optimize-convert-dma-op", "vpux::FunctionPass"> {
    let summary = "This pass deals with optimizations around ConvertDMAOp";

    let description = [{
        AI-generated summary

        This pass applies optimizations to eliminate redundant Copy operations around
        ConvertDMAOp operations.

        Rewrite patterns:
        - ConvertDMAViewLikeCopy: Fuses ConvertDMA -> [ViewLikeOps] -> CopyOp into a single
          ConvertDMA by moving view-like operations (GenericReshape, PermuteCast) before the
          ConvertDMA and writing directly to the Copy output buffer. Validates stride
          compatibility and distribution attribute consistency.
        - CopyConvertDMA: Eliminates CopyOp in Copy -> ConvertDMA sequences by feeding the
          Copy source directly into the ConvertDMA. Checks that input and output distributed
          buffer types are compatible.
        - ConvertDMASubViewCopy: Handles ConvertDMA -> [ViewLikeOps] => SubView -> Copy
          patterns where the ConvertDMA has multiple SubView users. Moves view-like ops
          before ConvertDMA and replaces each SubView -> Copy pair with SubView -> ConvertDMA,
          eliminating the intermediate copies. Only applies to non-distributed, compact-stride
          inputs and excludes ReinterpretCast view ops. If there is no view-like operation, we
          don't need to check input has stride or not.
    }];

    let constructor = "vpux::VPUIP::createOptimizeConvertDMAOpPass()";
}

//
// DMAOutOfOrderOptimization
//

def DMAOutOfOrderOptimization : PassBase<"dma-out-of-order-optimization", "vpux::FunctionPass"> {
    let summary = "enable DMA out of order execution if possile";

    let description = [{
        AI-generated summary

        Enables out-of-order execution for consecutive DMA tasks that have no data dependency.

        The pass groups DMA operations by their hardware queue (port + channel) and tracks the
        last 1-2 DMA tasks per queue. For each new DMA task[N], the pass checks whether its
        input buffers overlap in memory with the output buffers of the previous task[N-1] and,
        if task[N-1] was also marked out-of-order, with task[N-2] on the same queue. When no
        overlap is detected, the ORD bit is cleared to allow hardware out-of-order execution.

        Profiling buffer management DMAs are excluded from the optimization and reset the
        per-queue tracking state to prevent concurrency issues in DMA HWP. After processing
        all DMA tasks, barrier slot counts are verified for correctness.
    }];

    let constructor = "vpux::VPUIP::createDMAOutOfOrderOptimizationPass()";
}

//
// ComputeTaskStripping
//

def ComputeTaskStripping : PassBase<"compute-task-stripping", "vpux::FunctionPass"> {
    let summary = "Strips compute tasks from the model";

    let description = [{
        AI-generated summary

        This pass strips DPU and/or SHAVE compute tasks from the model for dry-run or
        debugging purposes, controlled by the dpuDryRun and shaveDryRun options.

        The pass uses BarrierInfo to analyze the barrier dependency graph and performs:
        1. Producer legalization: For each barrier with illegal (stripped) producers, traces
           upward through the barrier graph to find the nearest legal (non-stripped) producers
           and redirects the barrier's producer set to them.
        2. Consumer legalization: For each barrier with illegal consumers, traces downward
           to find the nearest legal consumers and redirects the barrier's consumer set.
        3. Erases all stripped TaskOps from the IR.
        4. Removes orphaned barriers (those with no producers or no consumers) and performs
           post-processing cleanup of barrier operations.

        DPU stripping supports STRIP mode (removes DPU tasks entirely). SHAVE stripping
        removes all ActShave tasks when enabled.
    }];

    let constructor = "vpux::VPUIP::createComputeTaskStrippingPass()";

    let options = [
        Option<
            "dpuDryRun", "dpu-dry-run",
            "std::string", [{"none"}],
            "Patch DPU tasks to disable their functionality (none|stub|strip)."
        >,
        Option<
            "shaveDryRun", "shave-dry-run",
            "bool", "false",
            "Enable shave dry run stripping."
        >
    ];

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// AddStartBarrier
//

def AddStartBarrier: PassBase<"add-start-barrier", "vpux::FunctionPass"> {
    let summary = "Adds a start barrier at the begining of the model";


    let description = [{
        AI-generated summary

        Ensures a start barrier exists at the beginning of the network so that DPU/SHV task
        enqueueing can be deferred until the barrier is signaled.

        The pass analyzes all DMA queues and searches for a valid start barrier candidate among
        barriers updated by the first DMA on port 0, channel DDR. A valid candidate must satisfy:
        (1) it is produced by DMA P0 Channel DDR to allow WLM Fetch DMAs to be inserted before it,
        (2) it is consumed only by DMA tasks (not DPU or SHV),
        (3) none of its producers and consumers depend on DPU/SHV task execution, and
        (4) it is not produced by any task other than the first DMA (to avoid race conditions
            under compiler barrier programming).

        If no valid candidate is found, the pass creates a new virtual barrier and inserts a
        zero-copy sync DMA (transfer size 0) to produce it. Cross-queue dependencies are then
        added so that the first DMA on every other queue (port/channel combination) waits on the
        start barrier, guaranteeing that service DMAs (barrier programming, LUT programming)
        complete before compute tasks begin.

        Expected pattern after the pass:
        SyncDMA -> StartBarrier -> DMATask -> other tasks
    }];

    let constructor = "vpux::VPUIP::createAddStartBarrierPass()";
}

//
// DetectDMASplitCandidate
//

def DetectDMASplitCandidate : PassBase<"detect-dma-split-candidate", "vpux::FunctionPass"> {
    let summary = "Detect DMA split candidate";

    let description = [{
        AI-generated summary

        Analyzes DMA task cycle costs using an inference execution simulator. For each
        NNDMA, computes the idle interval on the other DMA port during the task's execution
        window. Marks the task as a split candidate if the idle interval can fit the split
        cost, enabling the subsequent SplitDMAToBalanceLoad pass to distribute the transfer
        across both DMA ports. Only applies when 2 DMA ports are available.
    }];

    let constructor = "vpux::VPUIP::createDetectDMASplitCandidatePass()";
}

//
// SplitDMAToBalanceLoad
//

def SplitDMAToBalanceLoad : PassBase<"split-dma-to-balance-load", "vpux::FunctionPass"> {
    let summary = "Legalize split candidates to balance load on DMA ports";

    let description = [{
        AI-generated summary

        Splits NNDMAOp and GatherDMAOp operations that were marked as split candidates by
        DetectDMASplitCandidate. Distributes the transfer across available DMA ports.
        Does not handle hardware descriptor size limit violations — that is done by NNDMATiling.
    }];

    let constructor = "vpux::VPUIP::createSplitDMAToBalanceLoadPass()";
}

//
// LegalizeNNDMADimSizes
//

def LegalizeNNDMADimSizes : PassBase<"legalize-nndma-dim-sizes", "vpux::FunctionPass"> {
    let summary = "Split NNDMA transfers to bound per-task transfer time for firmware preemption";

    let description = [{
        Splits NNDMA operations whose transfer time estimated by `MAX_TASK_BYTE_BUDGET`
        exceeds the target preemption window. Firmware preemption cannot interrupt a DMA
        task mid-transfer, so the preemption latency is bounded by the runtime of the
        longest in-flight task. When a single NNDMA takes too long to execute (for example
        a multi-MiB contiguous transfer, or a strided transfer with many bursts),
        preemption is delayed until it completes and the runtime misses its preemption KPI.

        Byte count and stride pattern are used here as a cheap proxy for transfer
        time. `MAX_TASK_BYTE_BUDGET` approximates the number of bytes a DMA engine
        can move within the target preemption window on the expected memory path.
        For strided transfers the budget is additionally scaled by
        `contigBytes / BURST_BUFFER_SIZE` on DDR-touching transfers,
        because short contiguous chunks lower effective bandwidth and would
        otherwise let a "small" transfer still exceed the time budget.

        Two paths are used depending on the transfer layout:

        1. Contiguous transfers: if the compact payload exceeds the budget, the pass
            picks the outermost logical dimension whose size is at least `numTiles =
            ceil(payload / budget)` and splits along it.

        2. Strided transfers: the pass reduces the DMA pattern to isolate the innermost
            contiguous chunk (see `DMAContigChunk`) and splits along the outer plane
            dimension that repeats it. The per-tile byte target for strided transfers is
            scaled by `contigBytes / BURST_BUFFER_SIZE` when the transfer touches DDR, to
            keep effective bandwidth predictable when contiguous chunks are small.

        The split is skipped when the payload already fits the budget, when the input
        and output shapes differ, when the DMA has dynamic strides, or when no logical
        dimension is large enough to produce the required number of tiles.

        Emitted parts execute sequentially on the original DMA port. Only the first
        part keeps the original wait barriers and only the last part keeps the original
        update barriers; intermediate parts run without extra synchronization.
    }];

    let constructor = "vpux::VPUIP::createLegalizeNNDMADimSizesPass()";
}

//
// FuseSegmentedDMA
//

def FuseSegmentedDMA : PassBase<"fuse-segmented-dma", "vpux::FunctionPass"> {
    let summary = "This pass fuse segmented DMAs, which has compatible parameters in one DMA to decrease configuration overhead";

    let description = [{
        AI-generated summary

        Fuses multiple small DMA operations that transfer contiguous memory segments into single
        larger DMA transfers. Combines per-tile DMA copies that were created during distributed
        operation unrolling when they can be coalesced into one descriptor, reducing DMA task
        scheduling overhead.
    }];

    let constructor = "vpux::VPUIP::createFuseSegmentedDmaPass()";
}

//
// AddPlaceholderFetchDMAs
//

def AddPlaceholderFetchDMAs : PassBase<"add-placeholder-fetch-dmas", "vpux::FunctionPass"> {
    let summary = "Pass inserts fetch dmas to facilitate add-fetch-ops pass";

    let description = [{
        [Group N-2]..[Group N-1]..[Group N]
        For the first 2 available groups it inserts FetchDMAs at beginning of the DMA list on port 0 ch:DDR

        For the rest of groups it inserts FetchDMA with barriers. It creates 2 barriers such that last task of Group N-2 updates
        barrier X and last task of Group N-1 waits for barrier Y. In such case it inserts FetchDMA which waits for barrier X and
        updates barrier Y ensuring fetching for Group N is finished before Group N-1 is finished execution

        These FetchDMAs are replaced with actual fetch DMAs by add-fetch-dma-ops-full-wlm pass
        To identify which DMA belongs to which group on which queue we store fetchDMAAttr which contains information like
        FetchDMAAttr{
            executorKind
            tileIdx
            listIdx
            executionGroupIdx
        }
        Any FetchDMA can be uniquely identified by above parameters and replaced
    }];

    let constructor = "vpux::VPUIP::createAddPlaceholderFetchDMAsPass()";
}


def LegalizeStridedDmas : PassBase<"legalize-strided-dmas", "vpux::ModulePass"> {
    let summary = "Legalizes the network so that it correctly supports strided IO";

    let description = [{
        Checks whether the DMAs reading from or writing to network I/Os are compatible with dynamic strides,
        and legalizes the network if they are not. A DMA is considered compatible with dynamic strides if the
        layout function of the DMA input/output has the same dimensionality and strides as the network IO or
        can be reliably transformed into such function by adding/removing terms that do not contribute to the function.
        Layout function in this context is a function which maps tensor indices to byte position in the memory. For instance
        consider a network IO mxnxpxq tensor with following layout:

        f(m, n, p, q) = m * s1 + n * s2 + p * s3 + q * s4

        Which is a typical strided tensor layout. Now let's say DMA reading that tensor has following layout:

        f'(m, n, p, q) = m * s1' + n * s2' + p * s3' + q * s4'

        DMA is compatible iff each sn == sn' or the terms with unequal strides is actually a vanishing term.
        Vanishing terms are terms for which the corresponding dimension is 1 so the term always evaluates to 0.

        An example of a vanishing term:

        argument shape - mxnxp

        DMA shape - mxnx1xp

        Layout function for argument

        f(m, n, p) = s1 * m + s2 * n + s3 * p

        Layout for DMA:

        f'(m, n, 0, p) = s1 * m + s2 * n + s3 * 0 + s3 * p

        Which trivially simplifies to:

        f'(m, n, p) = s1 * m + s2 * n + s3 * p

        Which is compatible under previous definition.

        If this pass detects any incompatible DMA it will allocate auxiliary DDR buffer and transfer
        data from network IO to it and replace all uses of the network IO with uses of auxiliary buffer.
        If the branch using that auxiliary buffer has writes legalization pass will additionally insert
        DMA to transfer data back from auxiliary buffer to network IO. Below is an example of the legalized IR:

        func(%arg) {
            ...
            %val = SomeChainOfOps(%arg)
            %incompatible_view = SomeIncompatibleView(%val)
            %compatible_view = SomeCompatibleView(%val)
            %write_compatible = DMA(%source, %compatible_view)
            %write_incompatible = DMA(%source2, %incompatible_view)
            ...
        }

        Is transformed into:

        func(%arg) {
            ...
            %val = SomeChainOfOps(%arg)

            // Legalization prologue
            %auxiliary = memref.alloc()
            %legalization = DMA {stridedInput} (%val, %auxiliary)

            %incompatible_view = SomeIncompatibleView(%legalization)
            %compatible_view = SomeCompatibleView(%legalization)
            %write_compatible = DMA(%source, %compatible_view)
            %write_incompatible = DMA(%source2, %incompatible_view)

            // Legalization epilogue
            %concat = ConcatView input(%write_compatible, %write_incompatible) output(%legalization)
            %legalization_write = DMA {stridedOutput} (%concat, %val)

            ...
        }
    }];

    let constructor = "vpux::VPUIP::createLegalizeStridedDMAsPass()";
}

//
// AddPlaceholderFetchDMAsPWLM
//

def AddPlaceholderFetchDMAsPWLM : PassBase<"add-placeholder-fetch-dmas-pwlm", "vpux::FunctionPass"> {
    let summary = "Insert placeholder fetch DMAs for PWLM";

    let description = [{
        This pass inserts placeholder Fetch DMA operations for PWLM scheduling.

        For execution groups:
        [Group N-2] .. [Group N-1] .. [Group N]

        For the first two available execution groups, Fetch DMAs are inserted at the
        beginning of the DMA list on port 0, channel DDR.

        For all subsequent execution groups, Fetch DMAs are inserted with explicit
        barrier dependencies to guarantee safe execution.

        Barrier semantics:

        SHV task lists:
        - The last two tasks of Group N-1 must directly or indirectly wait on barriers produced by the
            Fetch DMA.
        - The last two tasks of Group N-2 must update the barrier consumed by the
            Fetch DMA.
        - These dependencies guarantee correctness when SHV lists are shared
            across execution groups.

        DPU tasks:
        - The last task of Group N-2 updates a barrier consumed by the Fetch DMA.
        - The last task of Group N-1 directly/indirectly waits on a barrier produced by the Fetch DMA.

        An example of indirect dependency is below
        `SomeDMA->B1->DPU1->DPU2`
        In above case we don't explicitly need DPU2 to wait on B1 because by FIFO order we can guarantee that someDMA will
        finish before DPU2 will start execution

        The placeholder Fetch DMAs inserted by this pass are later replaced with concrete NNDMA DMA operations

        To enable deterministic replacement, each placeholder Fetch DMA is annotated
        with a FetchDMAAttr containing:
        - executorKind
        - tileIdx
        - listIdx
        - executionGroupIdx

        The combination of these fields uniquely identifies a Fetch DMA and allows it to be reliably matched and replaced in later passes.
    }];

    let constructor = "vpux::VPUIP::createAddPlaceholderFetchDMAsPWLMPass()";
}

//
// UngroupBufferSection
//

def UngroupBufferSectionRewriterExecutor : PassBase<"run-ungroup-buffer-section-rewriters", "vpux::FunctionPass"> {
    let summary = "Build ungroup sparse and bounded buffer rewriter collection";

    let description = [{
        AI-generated summary

        This pass executes a named rewriter from the ungroup-buffer-section strategy registry.
        The strategy is selected via the rewriterName option and the rewriters are registered
        by createUngroupBufferSectionStrategy. Typical rewriters include:
        - Hoisting SubViewOp through GroupSparseBufferOp with constant fusion
        - Decomposing sparse-buffer operations into per-buffer operations, inserting temporary
          group/ungroup ops that are later optimized away
        - Optionally reordering SubView operations (controlled by enableReorderSubViewOp flag)

        The pass uses the dynamic rewriter framework (RewriterExecutorInterface) to discover
        and apply registered rewriters by name.
    }];

    let options = [
        Option<
            "rewriterName", "rewriter",
            "std::string", "",
            "Name of the rewriter to run"
        >,
        Option<
            "enableReorderSubViewOp", "enable-reorder-sub-view-op",
            "bool", "false",
            "Flag to enable or disable reordering of the sub ops"
        >
    ];

    let constructor = "vpux::VPUIP::createUngroupBufferSectionRewriterExecutorPass()";
}

//
// UnrollShaveCacheOps
//

def UnrollShaveCacheOps : PassBase<"unroll-shave-cache-ops", "vpux::FunctionPass"> {
    let summary = "Replace once ShaveCacheOps with multiple ones according to number of SHAVEs";

    let description = [{
        We need to have dedicated ShaveCacheOp per each SHAVE in order to properly generate the cache handling code during codegen.
        This pass replaces single ShaveCacheOp with multiple ones according to number of SHAVEs which require cache handling.
    }];

    let constructor = "vpux::VPUIP::createUnrollShaveCacheOpsPass()";
}

//
// PrepareShaveSubmitDMAs
//

def PrepareShaveSubmitDMAs :
    PassBase<"prepare-shave-submit-dmas", "vpux::FunctionPass"> {
    let summary = "Insert FetchDMAs, SHV SyncDMAs, and Release SyncDMAs for SHV submitted DMA flows";

    let description = [{
        Split stage 1 for SHV submit DMA legalization.
        This pass performs the first half of legalizing DMA submissions originating from SHAVE
        kernels that use DMA operations. It scans the execution graph with BarrierInfo, identifies
        SHAVE tasks that submit DMA descriptors, and groups them by logical task index so the
        corresponding DMA queues can be prepared before SkipDMA materialization.

        For each logical SHAVE task group the pass plans and inserts:
            • Fetch DMAs that retrieve the submitted descriptors on the corresponding DMA queue.
            • A shared fetch barrier that orders descriptor fetch completion before sync execution.
            • One SHV SyncDMA per impacted DMA queue. Each SHV sync consumes the fetch barrier,
                then updates an intermediate sync-to-SHV barrier and carries the logical task marker
                used by the second split pass.
            • One Release SyncDMA per impacted DMA queue, inserted after the SHV sync anchor.
            • A shared barrier updated by SHV tasks and consumed by Release SyncDMAs.

        Fetch DMAs produce the fetch barrier, SHV SyncDMAs consume that barrier and produce
        the sync-to-SHV barrier, SHAVE tasks consume the sync-to-SHV barrier, then SHAVE tasks
        produce the SHV-to-release barrier and Release SyncDMAs consume it.
        This preserves the original ordering guarantees that descriptor preparation completes before
        submitted list execution, and that ReleaseDMAs run only after corresponding SHV work.

        The pass also legalizes control-block boundary dependencies. If an SHV sync becomes the
        last task on its DMA queue, the pass inserts an additional guard SyncDMA immediately after it.
        The guard sync exists only to preserve queue-tail legality across control-block splits and to
        keep a valid insertion region for later SkipDMA materialization. It is not treated as a second
        logical synchronization stage.

        A graph after split stage 1 looks like:

        -> Fetch(Skip_list0_DDR) -|
        -> Fetch(Skip_list1_DDR) -|       -> SyncDMA(shv_sync, DDR) -|                      |-> SyncDMA(shv_release, DDR)
                                  |-> BAR->|                          |-> BAR -> SHV -> BAR->|
        -> Fetch(Skip_list0_CMX) -|       -> SyncDMA(shv_sync, CMX) -|                      |-> SyncDMA(shv_release, CMX)
        -> Fetch(Skip_list1_CMX) -|


        Stage 1 intentionally stops before inserting SkipDMAs. Its output is a legal execution graph
        with SHV/Release sync insertion points and ordering metadata that can be consumed independently by split stage 2.
    }];

    let constructor = "vpux::VPUIP::createPrepareShaveSubmitDMAsPass()";

    let options = [
        Option<
            "numDmaEnginesOpt", "num-dma-engines",
            "int", "",
            "Number of DMA Engines to legalize for"
        >
    ];
}

//
// InsertShaveSubmitSkipDMAs
//

def InsertShaveSubmitSkipDMAs : PassBase<"insert-shave-submit-skip-dmas", "vpux::FunctionPass"> {
    let summary = "Insert barrier less SkipDMAs before release SyncDMAs created by split stage 1";

    let description = [{
        Split stage 2 for SHV submit DMA legalization.
        This pass performs the second half of SHV submit DMA legalization. It locates the per-queue
        release SyncDMA tasks created by prepare-shave-submit-dmas and materializes
        the SkipDMA operations for the corresponding logical SHAVE task immediately before those syncs.

        SkipDMAs are inserted using operation order only. They do not receive barrier dependencies,
        matching the original implementation where skip descriptors remained barrier-less and relied on
        queue ordering after the synchronization point. The pass preserves the planned per-queue skip
        ordering and copies the WLM page from the release SyncDMA anchor to every newly inserted SkipDMA.

        Before choosing the final insertion anchor, the pass scans the same DMA queue backwards
        from the release SyncDMA until it reaches the first preceding SyncDMA on that queue
        (the sync boundary for the SHV submit flow).
        If one or more EnqueueDMA ops are present in this interval, the earliest EnqueueDMA
        (closest to SHV SyncDMA) becomes the insertion anchor instead of the release SyncDMA.

        A graph after split stage 2 looks like:

        -> Fetch(Skip_list0_DDR) -|
        -> Fetch(Skip_list1_DDR) -|-> BAR -> |-> SyncDMA(shv_sync, DDR) -> SkipDMA(list0_DDR), SkipDMA(list1_DDR) -> Release SyncDMA
        -> Fetch(Skip_list0_CMX) -|          |-> SyncDMA(shv_sync, CMX) -> SkipDMA(list0_CMX), SkipDMA(list1_CMX) -> Release SyncDMA
        -> Fetch(Skip_list1_CMX) -|          |
                                             |-----------------------> SHV[list0](withDMA)
                                             |-----------------------> SHV[list1](withDMA)

        Overall, the two split passes preserve the original SHV submit DMA legalization semantics
        while allowing fetch/sync preparation and SkipDMA materialization to be invoked separately.
    }];

    let constructor = "vpux::VPUIP::createInsertShaveSubmitSkipDMAsPass()";

    let options = [
        Option<
            "numDmaEnginesOpt", "num-dma-engines",
            "int", "",
            "Number of DMA Engines to legalize for"
        >
    ];
}

//
// SplitLargeInvariants
//

def SplitLargeInvariants : PassBase<"split-large-invariants", "vpux::FunctionPass"> {
    let summary = "Split NCE tasks into chunks to cap variants per NCEClusterTask";

    let description = [{
        This pass splits VPURT.Task-wrapped VPUIP.NCEClusterTask operations when their number of
        DPU variants exceeds architecture constraints derived from NPU configuration.
        The main goal of this pass is to mitigate runtime problem of handling preemption in case descriptor group contains only single invariant.

        Each split task keeps the same inputs and
        outputs, while a chain of VPURT virtual barriers is inserted between consecutive chunks to
        preserve the original execution order.

        When DPU profiling is enabled, the pass also rewrites profiling_data and
        profilingMetadata.numVariants for each chunk to keep per-chunk profiling
        layout and metadata consistent.
    }];

    let constructor = "vpux::VPUIP::createSplitLargeInvariantsPass()";

    let options = [
        Option<
            "maxVariantCount", "max-variant-count",
            "int", "",
            "To control available number (a half of maxVariantCount) of DPU variants per NCEClusterTask"
        >
    ];
}

#endif
