//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_ARCH_40XX_PASSES
#define VPUX_COMPILER_DIALECT_VPUIP_ARCH_40XX_PASSES

include "mlir/Pass/PassBase.td"

//
// ComputeHaloRegionForDPUTaskOp
//

def ComputeHaloRegionForDPUTaskOp : PassBase<"compute-halo-region-for-dpu-task-op", "vpux::FunctionPass"> {
    let summary = "Add halo region information to each DPU Task Op.";

    let description = [{
        This pass will transfer halo region information from the ITIBuffer type of output and output_ITI operands of
        VPUIP.NCEClusterTask to each of its VPUIP.DPUTaskOps.
    }];

    let constructor = "vpux::VPUIP::arch40xx::createComputeHaloRegionForDPUTaskOpPass()";

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

    let constructor = "vpux::VPUIP::arch40xx::createDMATaskProfilingHwDdrPass()";
}

//
// ConstantDpuProfHwpBase
//

def ConstantDpuProfHwpBase : PassBase<"constant-dpu-prof-hwp-base", "vpux::FunctionPass"> {
    let summary = "Sets single NCE HWP CMX base address";

    let description = [{
        In NPU40XX NCE HWP CMX base address must remain the same for all DPU tasks since
        HW may capture the address from a subsequent descriptor before storing the profiling
        record for the task just completed. To prevent from such race condition NCE HWP CMX
        base address for each NCE task will be 0 and profiling slot addressing will be
        handled by workload_id.
    }];

    let constructor = "vpux::VPUIP::arch40xx::createConstantDpuProfHwpBasePass()";
}

//
// Compress DMA for activations
//

def CompressSpillDma : PassBase<"compress-spill-dma", "vpux::ModulePass"> {
    let summary = "CompressedDMA for activation spills";

    let description = [{
        Use Compress/Decompress DMA for DMA tasks responsible for activation spilling
    }];

    let constructor = "vpux::VPUIP::arch40xx::createCompressSpillDmaPass()";
}

//
// UnrollClusterTiling
//

def UnrollClusterTiling : PassBase<"unroll-cluster-tiling", "vpux::FunctionPass"> {
    let summary = "Extracts child operations from ClusterTiling operation";

    let description = [{
        Legalizes chain of ClusterTiling operations according to the tile schema
    }];

    let constructor = "vpux::VPUIP::arch40xx::createUnrollClusterTilingPass()";
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

    let constructor = "vpux::VPUIP::arch40xx::createOptimizeConvertDMAOpPass()";
}

//
// DMAOutOfOrderOptimization
//

def DMAOutOfOrderOptimization : PassBase<"dma-out-of-order-optimization", "vpux::FunctionPass"> {
    let summary = "enable DMA out of order execution if possile";

    let description = [{
        Take advantage of DMA out of order execution if consequitive DMAs do not have any data dependency
    }];

    let constructor = "vpux::VPUIP::arch40xx::createDMAOutOfOrderOptimizationPass()";
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

    let constructor = "vpux::VPUIP::arch40xx::createComputeTaskStrippingPass()";

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

    let constructor = "vpux::VPUIP::arch40xx::createAddStartBarrierPass()";
}

//
// DetectDMASplitCandidate
//

def DetectDMASplitCandidate : PassBase<"detect-dma-split-candidate", "vpux::FunctionPass"> {
    let summary = "Detect DMA split candidate";

    let description = [{
        This pass looks for DMAs need to be split and assign splitCandidate attribute
    }];

    let constructor = "vpux::VPUIP::arch40xx::createDetectDMASplitCandidatePass()";
}

//
// SplitDMAToBalanceLoad
//

def SplitDMAToBalanceLoad : PassBase<"split-dma-to-balance-load", "vpux::FunctionPass"> {
    let summary = "Legalize split candidates to balance load on DMA ports";

    let description = [{
        This pass looks for DMAs with splitCandidate attribute and split them
    }];

    let constructor = "vpux::VPUIP::arch40xx::createSplitDMAToBalanceLoadPass()";
}

//
// FuseSegmentedDMA
//

def FuseSegmentedDMA : PassBase<"fuse-segmented-dma", "vpux::FunctionPass"> {
    let summary = "This pass fuse segmented DMAs, which has compatible parameters in one DMA to decrease configuration overhead";

    let description = [{
        This pass combines DMA with help of inter-cluster stride to combine them
    }];

    let constructor = "vpux::VPUIP::arch40xx::createFuseSegmentedDmaPass()";
}

//
// LegalizeScheduleForWlmFetchDmas
//

def LegalizeScheduleForWlmFetchDmas : PassBase<"legalize-schedule-for-wlm", "vpux::FunctionPass"> {
    let summary = "Pass inserts dummy dmas to facilitate workload-management pass";

    let description = [{
        Workload Management pass expects to always find a DMA on tile 0 and list 0 for attaching a 
        fetch task which copies the descriptors from DDR to CMX Metadata. In cases when we have super optimized 
        schedule we may not have any such DMAs which would cause the fetch task insertion algorithm to fail

        This pass understands how WLM creates execution groups and attaches fetch task. It emulates and tries to fit the
        dummy DMAs such we would always find DMAs during fetch task insertion while maintaining a valid schedule
    }];

    let options = [
        Option<
            "workloadManagementModeOpt", "workload-management-mode",
            "vpux::WorkloadManagementMode", "",
            "Setting which controls mode of WLM",
            [{::llvm::cl::values(
            clEnumValN(vpux::WorkloadManagementMode::PWLM_V0_LCA, "PWLM_V0_LCA", "Legacy WLM"),
            clEnumValN(vpux::WorkloadManagementMode::PWLM_V1_BARRIER_FIFO, "PWLM_V1_BARRIER_FIFO", "WLM with Barrier FIFO enabled")
          )}]
        >
    ];

    let constructor = "vpux::VPUIP::arch40xx::createLegalizeScheduleForWlmFetchDmasPass()";
}

//
// UnrollDepthToSpaceDMA
//

def UnrollDepthToSpaceDMA : PassBase<"unroll-depth-to-space-dma", "vpux::FunctionPass"> {
    let summary = "Split DepthToSpaceDMA task with several NN DMA tasks";

    let description = [{
        This pass splits DepthToSpaceDMA tasks with several NN DMA tasks, which are functionally equivalent.
        Each sub DepthToSpaceDMA will be converted to a NNDMA.
        1. if input/output layout is NHWC with model block_first, number of sub DepthToSpaceDMA is same as block_size.
        2. if input/output layout is NHWC with model depth_first, number of sub DepthToSpaceDMA is OH * OW / block_size.
        block_size is the size of the spatial block. It is an attribution of DepthToSpace.
    }];

    let constructor = "vpux::VPUIP::arch40xx::createUnrollDepthToSpaceDMAPass()";
}

//
// UnrollSpaceToDepthDMA
//

def UnrollSpaceToDepthDMA : PassBase<"unroll-space-to-depth-dma", "vpux::FunctionPass"> {
    let summary = "Split SpaceToDepthDMA task with several NN DMA tasks";

    let description = [{
        This pass splits SpaceToDepthDMA tasks into several NN DMA tasks, which are functionally equivalent.
        Each sub SpaceToDepthDMA will be converted to a NNDMA.
    }];

    let constructor = "vpux::VPUIP::arch40xx::createUnrollSpaceToDepthDMAPass()";
}

#endif
