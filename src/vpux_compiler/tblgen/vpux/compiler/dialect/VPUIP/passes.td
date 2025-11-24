//
// Copyright (C) 2022-2025 Intel Corporation.
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
        This pass adds a CACHE_PREFETCH SwKernel which loads specified kernel's instructions into L2 cache.
        Prefetch operations are added at the beginning of the inference if there is enough idle time on shave
        which can be configured via minimumShaveStartTimeForPrefetch option.

        Pass will first go over SW kernels present in the inference and gather unique kernels until it runs out of
        free cache space or until it encounters a cache handling operation. After gathering prefetch candidates it
        checks if first SW kernel start time is greater than minimumShaveStartTimeForPrefetch and inserts prefetch
        operations at inference start until it runs out of ActShaves. All kernels are prefetched in parallel and
        sequential prefetch is not supported right now. Cache is assumed to be shared between all ActShaves so it
        doesn't matter which ActShave runs prefetch.

        The reason for such a simplistic algorithm, which doesn't attempt to insert prefetch operations in the
        middle of the execution, is sensitivity to cost model scoring of different tasks, and regressions which
        appear when cost model has inaccurately predicted shave idle time. One way to mitigate the impact of those inaccuracies
        is to penalize prefetch task cost by some very large constant which eliminates the regressions but it also
        eliminates most of the beneficial prefetch operations except for those at the start of inference.

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
            "minimumShaveStartTimeForPrefetch", "minimum-shave-start-time-for-prefetch",
            "size_t", "250000",
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
        By moving pure view-like ops, this pass creates copy operation chains, that can be fused:
        Before: CopyOp -> PermuteCast -> GenericReshape -> CopyOp
        After: PermuteCast -> GenericReshape -> CopyOp -> CopyOp
    }];

    let constructor = "vpux::VPUIP::createMovePureViewOpBeforeCopyPass()";
}

//
// OptimizeCopies
//

def OptimizeCopies : PassBase<"optimize-copies", "vpux::FunctionPass"> {
    let summary = "Removes Copy Ops which are unnecessary";

    let description = [{
        This pass checks if Copy Op can be optimized out to reduce the amount of unnecessary DMAs and intermediate buffers.
        We need to check the workload management mode to determine if we can apply certain optimizations to the Copy Op sequence.
        This is necessary because the vpux::WorkloadManagementMode::PWLM_V0_LCA mode may fail with these optimizations, potentially causing performance regressions.
    }];

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

    let constructor = "vpux::VPUIP::createOptimizeCopiesPass()";
}

//
// UniquifyWeightsTableCopies
//

def UniquifyWeightsTableCopies : PassBase<"uniquify-weights-table-copies", "vpux::FunctionPass"> {
    let summary = "Removes Copy Ops of shared constant weights table";

    let description = [{
        This pass checks if Copy Op can be optimized out to reduce the amount of unnecessary DMAs
        and intermediate buffers when having weights table reuse enabled.
    }];

    let constructor = "vpux::VPUIP::createUniquifyWeightsTableCopiesPass()";
}

//
// MoveSubViewBeforeSparseBuffer
//

def MoveSubViewBeforeSparseBuffer : PassBase<"move-subview-before-sparse-buffer", "vpux::FunctionPass"> {
    let summary = "Moves child SubViewOp up through GroupSparseBufferOp";

    let description = [{
        Moves child SubViewOp up through GroupSparseBufferOp and fuses it with constants if possible.
        Reinfer output types of child operations since output type may change.
    }];

    let constructor = "vpux::VPUIP::createMoveSubViewBeforeSparseBufferPass()";
}

//
// OptimizeConcatViewCopies
//

def OptimizeConcatViewCopies : PassBase<"optimize-concat-view-copies", "vpux::FunctionPass"> {
    let summary = "Optimize ConcatView Op to remove unnecessary copies or reduce copies data size";

    let description = [{
        1. Propagate SubView Op before ConcatView Op to reduce data size of copies
        Target Pattern: NCE Task 16 channels output -> Copy 16 channels -> Concat -> Subview -> 3 channels
        Optimization:   NCE Task 16 channels output -> Copy 3 channels -> Concat -> Subview -> 3 channels

        2. Fuse ConcatView Ops to remove unnecessary copies, two conditions need to be satisfied:
        a) The Stride Level for each ConcatView input (after fusing) should be no more than 2;
           It's a runtime and HW limitation in order to get the right NNDMA descriptor, we support a maximum of 3D DMA transfers with 2 levels of striding.
        b) The number of inputs from the second ConcatView, which come from the output of the first ConcatView should no more than 1;
           For example, first ConcatView has M inputs, second ConcatView has N inputs, out of which P of them are the output of the first ConcatView
           After fusing, the number of input copies is: M * P + (N - P)
           Can't ensure we get benefit when P is of a large size. Limit optimization to P=1.
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
        This pass checks all the CopyOps consumed by tiles of one tiling subgraph.
        If the CopyOps operate on the same weight or activation, merge the parallel copies into one.
    }];

    let constructor = "vpux::VPUIP::createOptimizeParallelCopiesPass()";
}

//
// FuseLastCopy
//

def FuseLastCopy : PassBase<"fuse-last-copy", "vpux::FunctionPass"> {
    let summary = "Remove last Copy operations that output to DDR";

    let description = [{
        This pass checks if the Copy ops output to DDR can be optimized out to reduce the amount of unnecessary DMAs and intermediate buffers.
    }];

    let constructor = "vpux::VPUIP::createFuseLastCopyPass()";
}

//
// CopyOpTiling
//

def CopyOpTiling : PassBase<"tile-copies", "vpux::FunctionPass"> {
    let summary = "Legalizes Copy Ops which do not fit hardware capabilities";

    let description = [{
        This pass checks if Copy Op can be executed at target hardware and splits it into a few tiles if necessary.
        To fit hardware requirements it should copy less or equal than 16MB(2**24 bytes) and have less than 256 planes.
        The higher dimensional stride of the two-dimensional stride DMA is implemented through the plane.
        So if the copy has two dimensional stride, and the higher dimensional stride exceeds the number of planes,
        we need to split the copy on the higher stride dimension
    }];

    let constructor = "vpux::VPUIP::createCopyOpTilingPass()";
}

//
// ConvertDynamicReshapeToInPlace
//

def ConvertDynamicReshapeToInPlace : PassBase<"convert-dynamic-reshape-to-in-place", "vpux::FunctionPass"> {
    let summary = "Convert DynamicReshape operation to read and write to the same buffer in memory";

    let description = [{
        This pass will convert DynamicReshape to an inplace operation when the input rank is equal with the output rank.
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
        Extract pallet LUT from input quantile type and make it Const inputs.
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
        This pass updates all Types for internal memory buffers and function arguments and sets the specified memory space for them.
        Also updates the operand types for grouping operations, to cover scenarios where some operands are buffers and some are constants.
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
        This pass replaces all dynamic `alloc`/`dealloc` Operations with `VPUIP.StaticAlloc`.
        It uses simple LinearScan algorithm.
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
        This pass adds RelocateWeightsTable transformation to weights table constants. The transformation adds weights and sparsity base pointers
        to offset that are already filled in the weights table constants.
    }];

    let constructor = "vpux::VPUIP::createPatchWeightsTablePass()";
}

//
// PatchPopulateWeightTableWithShave
//

def PatchPopulateWeightTableWithShave : PassBase<"patch-populate-weight-table-with-shave", "vpux::FunctionPass"> {
    let summary = "Adjusts weights pointers after memory scheduling and before loosing data dependencies due to slices";

    let description = [{
        This pass updates weight base address and weight offset for Shave populating the weight table.
    }];

    let constructor = "vpux::VPUIP::createPatchPopulateWeightTableWithShavePass()";
}

//
// Linearization
//

def Linearization : PassBase<"linearization", "vpux::ModulePass"> {
    let summary = "Perform linearization of the IR";

    let description = [{
        Perform linearization of the IR with fully sequential execution.
    }];

    let constructor = "vpux::VPUIP::createLinearizationPass()";
}


//
// BreakDataFlow
//

def BreakDataFlow : PassBase<"break-data-flow", "vpux::FunctionPass"> {
    let summary = "Breaks the data flow in the graph";

    let description = [{
        This pass breaks the data flow in the graph. It is required for the VPURT dialect for correct task creation
        because all VPUIP dialect tasks will be inside body of the TaskOp and it is impossible to use operation results inside another body of TaskOp.
    }];

    let constructor = "vpux::VPUIP::createBreakDataFlowPass()";
}

//
// DMATaskProfilingAfterBarrierSched
//

def DMATaskProfilingAfterBarrierSched : PassBase<"dma-task-profiling-after-barrier", "vpux::ModulePass"> {
    let summary = "DMA task profiling handling after barrier scheduled";

    let description = [{
        This pass adds DMA profiling tasks after barrier scheduler.
    }];

    let constructor = "vpux::VPUIP::createDMATaskProfilingAfterBarrierSchedPass()";
}

//
// CaptureWorkpoint
//

def CaptureWorkpoint : PassBase<"capture-workpoint", "vpux::ModulePass"> {
    let summary = "Read value from WORKPOINT_CONFIG_MIRROR.FINAL_PLL_FREQ and store to profiling output";

    let description = [{
        Insert DMA transaction to capture PLL frequency multiplier needed for DPU profiling
    }];

    let constructor = "vpux::VPUIP::createCaptureWorkpointPass()";
}

//
// DPUProfiling
//

def DPUProfiling : PassBase<"dpu-profiling", "vpux::ModulePass"> {
    let summary = "DPU task profiling";

    let description = [{
        This pass allocate required memory for DPU profiling and perform buffer spilling
    }];

    let constructor = [{
        vpux::VPUIP::createDPUProfilingPass([](vpux::StringRef memSpaceName) {
            if (memSpaceName.empty()) {
                return std::optional<vpux::VPU::MemoryKind>{VPU::MemoryKind::CMX_NN};
            }

            return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpaceName);
        })
    }];

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
        This pass wraps each VPUIP layer operation into async region preserving linear execution.
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
    let summary = "Moves 'async.await' result usage from 'async.execute' body to it's operands";

    let constructor = "vpux::VPUIP::createMoveWaitResultToAsyncBlockArgsPass()";
}

//
// CalculateAsyncRegionCycleCost
//

def CalculateAsyncRegionCycleCost : PassBase<"calculate-async-region-cycle-cost", "vpux::FunctionPass"> {
    let summary = "Calculates cycle cost of 'async.execute'";

    let constructor = "vpux::VPUIP::createCalculateAsyncRegionCycleCostPass()";
}

//
// MoveViewOpsIntoAsyncRegions
//

def MoveViewOpsIntoAsyncRegions : PassBase<"move-view-ops-into-async-regions", "vpux::FunctionPass"> {
    let summary = "Moves view-like Operations inside the asynchronous regions which depends on them";

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
// GroupAsyncExecuteOps
//

def GroupAsyncExecuteOps : PassBase<"group-async-execute-ops", "vpux::FunctionPass"> {
    let summary = "Reduces number of async.execute operations";

    let description = [{
        Groups consecutive operations which utilizes the same executor and max resources into same async.execute region
    }];

    let constructor = "vpux::VPUIP::createGroupAsyncExecuteOpsPass()";

}

//
// FeasibleAllocation
//

def FeasibleAllocation : PassBase<"feasible-allocation", "vpux::FunctionPass"> {
    let summary = "Feasible Memory Scheduling Pass";

    let description = [{
        Schedule async.execute operations based on their dependencies and CMX memory availability
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
        This pass will replace specific op and permute as one DMA and fuse the unnecessary following copy ops
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
        This pass will replace specific tile op as DMA and fuse the unnecessary following copy ops.
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
        This pass will convert the Expand to copy and concat subgraph.
    }];

    let constructor = "vpux::VPUIP::createConvertExpandPass()";

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
        This pass will convert some SW operations (e.g. DepthToSpace, Permute) to DMA ops
        if it is possible to achieve better performance
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
        This pass allocate required memory for ActShaveProfiling profiling and perform buffer spilling
    }];

    let constructor = [{
        vpux::VPUIP::createActShaveProfilingPass([](vpux::StringRef memSpaceName) {
            if (memSpaceName.empty()) {
                return std::optional<vpux::VPU::MemoryKind>{VPU::MemoryKind::CMX_NN};
            }

            return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpaceName);
        })
    }];

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
        Group profiling buffers from different profiling engines into single profiling output buffer with name as
        [offset]_[profiling name]_[offset]_[profiling name] so postprocessing tool can parse it back
    }];

    let constructor = "vpux::VPUIP::createGroupProfilingBuffersPass()";
}

//
// UpdateSwKernelParams
//

def UpdateSwKernelParams : PassBase<"update-sw-kernel-params", "vpux::FunctionPass"> {
    let summary = "Update parameters sent to sw kernel for selected operations";

    let description = [{
        Part of shave kernel attributes depend by resolution. Pass allow update them
        based on final resolution information.
    }];

    let constructor = "vpux::VPUIP::createUpdateSwKernelParamsPass()";
}

//
// SyncShvDpu
//

def SyncShvDpu : PassBase<"sync-shv-dpu", "vpux::FunctionPass"> {
    let summary = "Add a sync dummy kernel after shave kernel that use DPU inside.";

    let description = [{
        Add a sync dummy kernel after shave kernel that use DPU inside.
        This is made automaticaly base on kernel description "isDpuShaveKernelType".
        It is necesarry for allow WLM to retrigger dpu fifo advance fill after filter finish.
        Previous, Dpu fifo was emptied, not allowing task fill advance.
    }];

    let constructor = "vpux::VPUIP::createSyncShvDpuPass()";
}

//
// DumpStatisticsOfTaskOpsPass
//

def DumpStatisticsOfTaskOpsPass : PassBase<"dump-statistics-of-task-ops", "vpux::FunctionPass"> {
    let summary = "Dump the statistics of operations (used Task operations and weights compression)";

    let description = [{
        This pass dumps the statistics for of used operations (e.g. tasks, weights compression) and makes a report as warning for operations not converted to DPU.
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

        This pass behaves differently for pre-NPU37XX and post-NPU37XX platforms. For former the compression
        is done using huffman encoding and applied only to quantized data types, for the latter the
        compression is done using bit-compactor library.
    }];

    let constructor = "vpux::VPUIP::createCompressWeightsBTCPass()";
}

//
// UnrollExpandDMA
//

def UnrollExpandDMA : PassBase<"unroll-expand-dma", "vpux::FunctionPass"> {
    let summary = "Unroll expand task with several NN DMA tasks";

    let description = [{
        This pass unroll ExpandDMA tasks with several NN DMA tasks, which are functionally equivalent.
        Each sub ExpandDMA will be converted to a NNDMA.
    }];

    let constructor = "vpux::VPUIP::createUnrollExpandDMAPass()";
}

//
// UnrollGatherDMA
//

def UnrollGatherDMA : PassBase<"unroll-gather-dma", "vpux::FunctionPass"> {
    let summary = "Unroll gather DMA task with several NN DMA tasks";

    let description = [{
        This pass unrolls GatherDMA tasks with several NN DMA tasks, which are functionally equivalent.
        Each sub GatherDMA will be converted to a NNDMA.
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
// UnrollSwKernel
//

def UnrollSwKernel : PassBase<"unroll-sw-kernel", "vpux::FunctionPass"> {
    let summary = "Unroll SwKernel task with several SwKernel.Run";

    let description = [{
        This pass unroll SwKernel task with several SwKernel.Run, which are functionally equivalent.
        Each SwkernelRun will be wrapped into a SwKernel task.
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
    let summary = "Convert data transfer operations to DMA";

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
    let summary = "Convert static allocations to declarations";

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
    let summary = "Convert view-like operations to declarations";

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
        On HW with swizzling support (VPUX37XX) enable activation swizzling for DPU to DPU
        buffers. This includes setting specific swizzling key and alignment as part of
        allocation operation.
        Swizzling requirement:
        - buffer needs to be properly aligned
        - swizzled buffers must be given in CMX space with size of multiple of 512
        - activation buffer must be one produced and consumed by DPU type task
        - buffer for weights can be swizzled and needs to have swizzling transformation performed on the content
        Device supported swizzling key
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

def DMABarrierOptimization : PassBase<"dma-barrier-optimization", "vpux::FunctionPass"> {
    let summary = "Optimize DMA related barriers after dma port has been assigned for VPUX37XX";
    let constructor = "vpux::VPUIP::createDMABarrierOptimizationPass()";
}

//
// OperationStubbing
//

def OperationStubbing : PassBase<"operation-stubbing", "vpux::FunctionPass"> {
    let summary = "Stub operations with StubOp";

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
        This pass checks if weights from a Convolution op were previously padded with zero, remove that pad, insert a ShapeCast op
        and set cm_sp_pattern value.
    }];

    let constructor = "vpux::VPUIP::createConvWeightsCompressionPass()";
}

//
// FuseConstants
//

def FuseConstants : PassBase<"fuse-constants", "vpux::FunctionPass"> {
    let summary = "Fuse constant inputs of NCEClusterOp";

    let description = [{
        Concatenates input constants into one in the following order:
            weight_table -> weights -> weights_sparsity_map
        For any NCEClusterTaskOp if the number of constants to fuse is 1 such layers are skipped
        Special Case with DWCONV, if the constants to fuse are == 2 such DWCONV are skipped as weights are not constants
        Special Case for Compressed Conv layer, if the weights are ShapeCast these such layers are skipped
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
// UngroupSparseBuffers
//

def UngroupSparseBuffers : PassBase<"ungroup-sparse-buffers", "vpux::FunctionPass"> {
    let summary = "Ungroups sparse buffers into individual buffers";

    let description = [{
        Splits operations that work with sparse buffers into multiple operations,
        each working with an individual buffer.

        These separate operations are then surrounded by UngroupSparseBuffer
        and / or GroupSparseBuffer operations, which are then optimized-out.
    }];

    let constructor = "vpux::VPUIP::createUngroupSparseBuffersPass()";

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
        Flattens the type of the sparse weights into a binary buffer of values.
        The weights operand of a NCE operation maintains the original type.
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
        This pass will transform DMAs that operate on swizzled buffers so that they
        copy buffers of size aligned to 512 to satisfy swizzling HW restrictions.
        Transformation will result in a flat buffer copy of required size
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
        This pass will tile act kernel run task on multi shaves.
    }];

    let constructor = "vpux::VPUIP::createTileActShaveKernelTaskPass()";
}

//
// SetZeroOffsetWeightsTable
//

def SetZeroOffsetWeightsTable : PassBase<"set-zero-offset-weights-table", "vpux::FunctionPass"> {
    let summary = "Set NCEClusterTask isZeroOffsetWeightsTable attribute";

    let description = [{
        Identifies NCEClusterTask operations whose weights table pointers can be relative to the start of the
        weights base address (i.e. the pointers start from zero) and adds the `isZeroOffsetWeightsTable`
        attribute to them. This will prevent future passes like PatchWeightsTable and
        PatchPopulateWeightTableWithShave to patch the weights table content, keeping the data offset starting from 0.

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
        This pass prepares DDR allocation for handling HW support for activation spilling compression.
        In worst case scenario DDR buffer needs to be enlarged in case size after compression is bigger
    }];

    let constructor = "vpux::VPUIP::createAdjustSpillSizePass()";
}

//
// FuseDDRCopiesIntoConcats
//

def FuseDDRCopiesIntoConcats : PassBase<"fuse-ddr-copies-into-concats", "vpux::FunctionPass"> {
    let summary = "Fuse VPUIP.Copy into adjacent VPUIP.ConcatView";

    let description = [{
        When there's NCEClusterTask (CMX2DDR) -> Copy (DDR2DDR) -> ConcatView (DDR) pattern in a graph,
        it is possible to get rid of the copy.
        In order to do that both NCEClusterTask and ConcatView must share output buffer.
    }];

    let constructor = "vpux::VPUIP::createFuseDDRCopiesIntoConcats()";
}

//
// NNDMATiling
//

def NNDMATiling : PassBase<"nn-dma-tiling", "vpux::FunctionPass"> {
    let summary = "Legalizes NNDMA Ops which do not fit hardware capabilities";

    let description = [{
        This pass splits NNDMAOp if DMA data size is greater than 16MB(2**24 bytes) or it has more than 256 planes after unroll-distributed-ops.
        CopyOpTiling pass has similar logic to tile CopyOp, but Distributed CopyOp can not be handled.
        This pass is added to legalize NNDMAOp so that each DMA job can fit hardware capabilities as a timely solution.
        The refactoring in E#84706 will provide a more maintainable long term solution.
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
           Traverses the IR and when found VPUIP.Copy it is converted as SW.Copy operator
           which is created in this pass on certain conditions.
           Need to cover the cases of memory transfers, with strides which are not byte aligned,
           and because of that can't be executed as DMA ops.

           The only solution for that is to have a dedicated SW Copy operator which can handle these sub byte accesses.
           Copy has the inBitOffset and outBitOffset attributes to suggest which nibble to start copying from.

           Current version has the base functionality: INPUT (strided) -> OUTPUT && INPUT -> OUTPUT (strided)
            The main conditions for using Copy operator are:
                sub_byte_type_used && strides_non_byte_aligned
                                    OR
                strides_aligned && strides_in_bit > dims_in_bit && innermost_dim_not_aligned
    }];

    let constructor = "vpux::VPUIP::createConvertVPUIPCopyToSWCopyPass()";
}

//
// UnrollAnalysis
//

def UnrollDMAAnalysis : PassBase<"unroll-analysis", "vpux::FunctionPass"> {
    let summary = "Analysis pass to cache which unroll dma passes are needed";

    let description = [{
        This pass creates and fill UnrollDMAAnalysis. It traverses the IR and looks for DMA ops that have to be unrolled.
        The DMA unrolling passes use UnrollDMAAnalysis analysis to return early if unrolling is not needed.
        Prevents slow IR traversal and checks and speedup compile time especially for bigger graphs.
    }];

    let constructor = "vpux::VPUIP::createUnrollDMAAnalysisPass()";
}

//
// InvalidateUnrollAnalysis
//

def InvalidateUnrollDMAAnalysis : PassBase<"invalidate-unroll-dma-analysis", "vpux::FunctionPass"> {
    let summary = "Invalidates UnrollDMAAnalysis";

    let description = [{
        Invalidates UnrollDMAAnalysis which is used to skipp unnecessary DMA unroll passes.
    }];

    let constructor = "vpux::VPUIP::createInvalidateUnrollDMAAnalysisPass()";
}

//
// DispatchedInliner
//

def DispatchedInliner : PassBase<"dispatched-inliner", "vpux::ModulePass"> {
    let summary = "Inlines using VPUIP's FuncInlinerInterface";

    let description = [{
        This pass executes MLIR's inliner pass which dispatches to VPUIP's FuncInlinerInterface.
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
        This pass unrolls DepthToSpaceDMA tasks to multiple new DepthToSpaceDMA tasks, where applicable.

        Main considerations when deciding to unroll a task to multiple new tasks are:
         - respecting HW restrictions, like maximum number of dimensions, maximum size of a dimension etc.
         - improving performance by distributing a task to all available HW ports.

        This pass makes use of per-architecture strategies.
        Currently, multi-cluster unrolling is common for all architectures and single-cluster unrolling
        is defined separately for NPU37XX and NPU40XX.

        For NPU37XX single-cluster unrolling:
        1. if input/output layout is NHWC with model block_first, number of sub DepthToSpaceDMA is same as block_size.
        2. if input/output layout is NHWC with model depth_first, number of sub DepthToSpaceDMA is OH * OW / block_size.
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
        Legalizes chain of distributed operations according to the tile schema
    }];

    let constructor = "vpux::VPUIP::createUnrollDistributedOpsPass()";
}

//
// ComputeHaloRegionForDPUTaskOp
//

def ComputeHaloRegionForDPUTaskOp : PassBase<"compute-halo-region-for-dpu-task-op", "vpux::FunctionPass"> {
    let summary = "Add halo region information to each DPU Task Op.";

    let description = [{
        This pass will transfer halo region information from the ITIBuffer type of output and output_ITI operands of
        VPUIP.NCEClusterTask to each of its VPUIP.DPUTaskOps.
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
        This pass enables hardware DMA profiling directly to DDR.
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
        Use Compress/Decompress DMA for DMA tasks responsible for activation spilling
    }];

    let constructor = "vpux::VPUIP::createCompressSpillDmaPass()";
}

//
// OptimizeConvertDMAOp
//

def OptimizeConvertDMAOp : PassBase<"optimize-convert-dma-op", "vpux::FunctionPass"> {
    let summary = "This pass deals with optimizations around ConvertDMAOp";

    let description = [{
        This pass deals with optimizations around ConvertDMAOp including rewriters like
        1. ConvertDMACopy - Eliminate CopyOp in ConvertDMA -> Copy sequence
        1. ConvertDMAClusterCopy - Eliminate CopyOp in ConvertDMA -> ClusteredCopy sequence
    }];

    let constructor = "vpux::VPUIP::createOptimizeConvertDMAOpPass()";
}

//
// DMAOutOfOrderOptimization
//

def DMAOutOfOrderOptimization : PassBase<"dma-out-of-order-optimization", "vpux::FunctionPass"> {
    let summary = "enable DMA out of order execution if possile";

    let description = [{
        Take advantage of DMA out of order execution if consequitive DMAs do not have any data dependency
    }];

    let constructor = "vpux::VPUIP::createDMAOutOfOrderOptimizationPass()";
}

//
// ComputeTaskStripping
//

def ComputeTaskStripping : PassBase<"compute-task-stripping", "vpux::FunctionPass"> {
    let summary = "Strips compute tasks from the model";

    let description = [{
        This pass strips compute tasks from the model.
        There are three options that can be selected to strip tasks : dpu, act-shave, all.
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
        Adds a start barrier to always ensure that there is a barrier consumption event. The barrier should
        be produced by a sync DMA which is supposed to be zero copy with transfer size 0. The pattern is
        expected after this pattern at the start of the nework:
        DMATask -> Barrier -> DMATask -> other tasks
    }];

    let constructor = "vpux::VPUIP::createAddStartBarrierPass()";
}

//
// DetectDMASplitCandidate
//

def DetectDMASplitCandidate : PassBase<"detect-dma-split-candidate", "vpux::FunctionPass"> {
    let summary = "Detect DMA split candidate";

    let description = [{
        This pass looks for DMAs need to be split and assign splitCandidate attribute
    }];

    let constructor = "vpux::VPUIP::createDetectDMASplitCandidatePass()";
}

//
// SplitDMAToBalanceLoad
//

def SplitDMAToBalanceLoad : PassBase<"split-dma-to-balance-load", "vpux::FunctionPass"> {
    let summary = "Legalize split candidates to balance load on DMA ports";

    let description = [{
        This pass looks for DMAs with splitCandidate attribute and split them
    }];

    let constructor = "vpux::VPUIP::createSplitDMAToBalanceLoadPass()";
}

//
// FuseSegmentedDMA
//

def FuseSegmentedDMA : PassBase<"fuse-segmented-dma", "vpux::FunctionPass"> {
    let summary = "This pass fuse segmented DMAs, which has compatible parameters in one DMA to decrease configuration overhead";

    let description = [{
        This pass combines DMA with help of inter-cluster stride to combine them
    }];

    let constructor = "vpux::VPUIP::createFuseSegmentedDmaPass()";
}

//
// LegalizeScheduleForPartialWlmFetchDmas
//

def LegalizeScheduleForPartialWlmFetchDmas : PassBase<"legalize-schedule-for-partial-wlm", "vpux::FunctionPass"> {
    let summary = "Pass inserts dummy dmas to facilitate add-fetch-ops pass";

    let description = [{
        Workload Management pass expects to always find a DMA on tile 0 and list 0 for attaching a
        fetch task which copies the descriptors from DDR to CMX Metadata. In cases when we have super optimized
        schedule we may not have any such DMAs which would cause the fetch task insertion algorithm to fail

        This pass understands how WLM creates execution groups and attaches fetch task. It emulates and tries to fit the
        dummy DMAs such we would always find DMAs during fetch task insertion while maintaining a valid schedule
    }];

    let constructor = "vpux::VPUIP::createLegalizeScheduleForPartialWlmFetchDmasPass()";
}
#endif
