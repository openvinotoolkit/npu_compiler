//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI40XX_PASSES
#define VPUX_COMPILER_DIALECT_VPUMI40XX_PASSES

include "mlir/Pass/PassBase.td"

//
// SetupProfilingVPUMI40XX
//

def SetupProfilingVPUMI40XX : PassBase<"setup-profiling-VPUMI40XX", "vpux::ModulePass"> {
    let summary = "Apply HW profiling settings in MI";

    let description = [{
        AI-generated summary

        Configures DMA hardware profiling in the MappedInference operation. Based on the
        DMA profiling mode (STATIC_HWP, DYNAMIC_HWP, or SCRATCH), creates the DMA HWP
        base buffer pointing to the appropriate memory region (profiling output section,
        reserved DDR, or CMX scratch). Also creates a workpoint capture buffer from the
        profiling workpoint section for performance counter snapshots. Both buffers are
        attached to MappedInferenceOp for use by the runtime.
    }];

    let constructor = "vpux::VPUMI40XX::createSetupProfilingVPUMI40XXPass()";

    let options = [
        Option<
            "enableDMAProfiling", "dma-profiling",
            "std::string", [{"false"}],
            "Enable DMA task profiling (true|static|false)"
        >
    ];

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPUMI40XX::VPUMI40XXDialect"
    ];
}

//
// BarrierCompuation
//

def BarrierComputation : PassBase<"barrier-computation-VPUMI40XX", "vpux::FunctionPass"> {
    let summary = "Compute and setup scheduling related fields";

    let description = [{
        This pass computes and sets up the (barrier) scheduling related fields - next_same_id, previousSameId, start_after, clean_after.
        This logic was clasically part of the parser, but in ELF pipeline, it is included in the lowering.
    }];

    let constructor = "vpux::VPUMI40XX::createBarrierComputationPass()";

}


def ReorderMPIOps : PassBase<"reorder-mapped-inference-ops", "vpux::FunctionPass"> {
    let summary = [{Reorder Mapped Inference dialect ops to a *canonical* order.}];
    let description = [{
        AI-generated summary

        The mapped inference representation does not follow the *classical* rules of operation order inside the IR.
        In higher level dialects, OP order in the IR mostly followed the topological order of the execution, with
        Async ops and VPURT.TaskOps generally decorating extra information on the parallelism.

        In MappedInference representation execution flow is expressed differently, as it explicitly defines the list
        semantics of how the operations follow each other in execution flow, while parallelism is expressed at "list level".

        This pass re-orders the OPS inside the IR to follow these semantics. For DeclareTaskBuffer operations,
        the order must be aligned with firmware expectations. For other operations, the reorder primarily
        improves human readability of the IR without introducing positional semantics to the dialect.
    }];

    let constructor = "vpux::VPUMI40XX::reorderMappedInferenceOpsPass()";
}

def GroupExecutionOps : PassBase<"group-execution-ops", "vpux::FunctionPass"> {
    let summary = "Group DPU/ActKernel execution ops into ExecutionGroup containers";

    let description = [{
        AI-generated summary

        Groups execution tasks into ExecutionGroup operations for WLM scheduling.
        Pairs primary task types (DPU Invariant, ActKernel Range) with secondary types
        (Variant, Invocation). Respects metadata size constraints and barrier scheduling
        requirements so that WLM can schedule and enqueue primary+secondary task pairs
        atomically.
    }];

    let constructor = "vpux::VPUMI40XX::createGroupExecutionOpsPass()";
}

def UnGroupExecutionOps : PassBase<"ungroup-execution-ops", "vpux::FunctionPass">  {
    let summary = "Extract execution ops from ExecutionGroup containers";

    let description = [{
        AI-generated summary

        Reverses GroupExecutionOps transformation by extracting individual operations from
        ExecutionGroup blocks. Replaces block arguments with actual previous task values,
        replaces start/end indices with list heads/tails, and moves operations back to
        the parent function scope.
    }];

    let constructor = "vpux::VPUMI40XX::createUnGroupExecutionOpsPass()";
}

def ResolveWLMTaskLocation : PassBase<"resolve-wlm-task-location", "vpux::FunctionPass"> {
    let summary = [{ 40xx and WLM specific version of resolve task location pass }];
    let description = [{In case of workload management task locations assignment works by a different logic. In non-WLM
            scenario, the NNR feeders will bring tasks in local memory in a circular buffer manner, so the logic can
            iterate over all tasks and apply the round-robin logic.
            With WLM we bring tasks in groups and each group gets a ping-pong buffer. A group is materialized by an
            ExecutionGroup op, and every task inside an execution group will receive a task location from inside the PP
            buffer pool}];

    let constructor = "vpux::VPUMI40XX::createResolveWLMTaskLocationPass()";
}

def PropagateFinalBarrier : PassBase<"propagate-final-barrier", "vpux::FunctionPass"> {
    let summary = "Propagate final barrier ID to MappedInferenceOp";

    let description = [{
        AI-generated summary

        Identifies the final barrier operation in the compiled network and stores its
        index in MappedInferenceOp. The runtime reads this index from MappedInferenceOp
        to determine when all tasks have completed.
    }];

    let constructor = "vpux::VPUMI40XX::createPropagateFinalBarrierPass()";
}

def AddEnqueueOps : PassBase<"add-enqueue-ops", "vpux::FunctionPass"> {
    let summary = [{ Schedule Enqeue operations }];

    let description = [{
        Schedule Enqueu operations for DMA, DPU & Shave tasks.
        Even though DMAs are not fetched into CMX metadata space, they still need to be enqueued.

        It accounts for enqueues scheduled at the same barriers and fuses them into single enqueue ->
        enqueue can cover secondaries from different primaries -> enqueue can cover secondaries across
        Fetch task boundary.
    }];

    let constructor = "vpux::VPUMI40XX::createAddEnqueueOpsPass()";
}

def LinkEnqueueTargets : PassBase<"link-enqueue-targets", "vpux::FunctionPass"> {
    let summary = [{ Hard-Link Enqueue Ops targets}];

    let description = [{
        An enqueue op supports a range of tasks. It's semantics specify that all tasks inside that range can be
        enqueued(in order). As a performance optimization, we wouldn't want all tasks to be enqueued individually.
        So for the task types that support hard-linking, we would link all the tasks that are within an enqueue range,
        and set the enqueueOp to only enqueue the First one.
    }];

    let constructor = "vpux::VPUMI40XX::createLinkEnqueueTargetsPass()";

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

def UnrollFetchTaskOps : PassBase<"unroll-fetch-ops", "vpux::FunctionPass"> {
    let summary = [{ Unroll FetchTask Ops }];

    let description = [{
        Replaces single Fetch operation for a task group with 2 DMA operations: one for "primary" task type
        (e.g. invariant) range and one for "secondary" task type range.
    }];

    let constructor = "vpux::VPUMI40XX::createUnrollFetchTaskOpsPass()";
}

def SplitEnqueueOps : PassBase<"split-enqueue-ops", "vpux::FunctionPass"> {
    let summary = [{ Split enqueue Ops to avoid bug when we have last op in task buffer in LL }];

    let description = [{
        To account for preemption at a group boundary level scenario we have to leave last task in fetch group
        unlinked. The problem with preemption is due to concurrency between thread processing preemption and execution
        thread, upon restore of previous state we can't reliably check if last task in group was linked, since it could be
        replaced already with tasks from next group by execution thread. The solution is to leave last tasks in groups
        always unlinked.

        This pass splits single enqueue operation into multiple based on special attribute
        (lastSecondaryTaskInExecutionGroup) set by previous passes in the pipeline (group-execution-ops). Since single
        enqueue operation may cover a range of tasks spanning through multiple Fetch groups, it may be split into more
        than 2 enqueue operations.
    }];

    let constructor = "vpux::VPUMI40XX::createSplitEnqueueOpsPass()";
}

def UnrollEnqueueOps : PassBase<"unroll-enqueue-ops", "vpux::FunctionPass"> {
    let summary = "Expand range-based EnqueueOps into single-task EnqueueOps";

    let description = [{
        AI-generated summary

        Unrolls EnqueueOp operations that specify a range of tasks (start to end inclusive)
        into individual EnqueueOp operations targeting single tasks. Chains expanded
        operations together through result replacement: the first expanded op is used by
        MappedInferenceOp, the last is used by other consumers.
    }];

    let constructor = "vpux::VPUMI40XX::createUnrollEnqueueOpsPass()";
}

def LinkEnqueueOpsForSameBarrier : PassBase<"link-enqueue-ops-for-same-barrier", "vpux::FunctionPass"> {
    let summary = "Link consecutive EnqueueOps operating on the same barrier";

    let description = [{
        AI-generated summary

        Links EnqueueOps that share the same barrier index by chaining each to its
        predecessor. Without this chaining, multiple enqueues on the same barrier could
        race and produce undefined scheduling behavior in the firmware.
    }];

    let constructor = "vpux::VPUMI40XX::createLinkEnqueueOpsForSameBarrierPass()";
}

def AddBootstrapBarriers : PassBase<"add-bootstrap-barriers", "vpux::FunctionPass"> {
    let summary = [{ Add Bootstrap Barriers}];
    let description = [{
        AI-generated summary

        Creates BootstrapOp operations for each unique physical barrier ID encountered
        in ConfigureBarrier operations. Links the first BootstrapOp and the total count
        to MappedInferenceOp so the firmware can initialize barriers at inference start.
        Stops after processing up to the number of available physical barriers.
    }];

    let constructor = "vpux::VPUMI40XX::createAddBootstrapBarriersPass()";
}

def AddBootstrapWorkItems : PassBase<"add-bootstrap-work-items", "vpux::FunctionPass"> {
    let summary = [{ Add Bootstrap WorkItems}];
    let description = [{
        Analyze schedule and check presence of already present EnqueueOps (WorkItems) or enqueue DMAs. For initial range
        of tasks not handled by those enqueues create bootstrap EnqueueOps that will be scheduled directly by FW
        at the beginning of inference.
    }];

    let constructor = "vpux::VPUMI40XX::createAddBootstrapWorkItemsPass()";

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

def NextSameIdAssignment : PassBase<"next-same-id-assignment", "vpux::FunctionPass"> {
    let summary = "Compute and setup scheduling related fields";

    let description = [{
        AI-generated summary

        Assigns the nextSameId field for VPUMI40XX execution descriptor operations by linking
        operations that share the same task list ID. Builds linked lists of descriptors within
        each task queue so the firmware can traverse all descriptors belonging to the same
        logical task sequence.
    }];

    let constructor = "vpux::VPUMI40XX::createNextSameIdAssignmentPass()";

}

def AddPlatformInfo : PassBase<"add-platform-info", "vpux::FunctionPass"> {
    let summary = "Insert PlatformInfoOp with architecture and platform metadata";

    let description = [{
        AI-generated summary

        Creates a PlatformInfoOp with a trivial index to hold platform-level metadata
        for the compiled network.
    }];

    let constructor = "vpux::VPUMI40XX::createAddPlatformInfoPass()";

}

def SetupTaskDescriptorConstraints : PassBase<"setup-task-descriptor-constraints", "vpux::ModulePass"> {
    let summary = "Write task-descriptor binary-size constraints into PipelineOptionsOp";

    let description = [{
        Stores per-architecture task-descriptor binary sizes (DPU invariant/variant,
        ActKernel invocation/range, DMA, M2I) as config::OptionOp entries in
        PipelineOptionsOp. Called from each arch's VPUIP-to-VPUMI40XX conversion
        pipeline with arch-specific sizes computed from the platform register-descriptor
        layouts.
    }];

    let constructor = "vpux::VPUMI40XX::createSetupTaskDescriptorConstraintsPass()";

    let options = [
        Option<"dpuInvariantSize", "dpu-invariant-size", "int64_t", "0",
               "Binary size of DPU invariant descriptor (bytes)">,
        Option<"dpuVariantSize", "dpu-variant-size", "int64_t", "0",
               "Binary size of DPU variant descriptor (bytes)">,
        Option<"actKernelInvocationSize", "act-kernel-invocation-size", "int64_t", "0",
               "Binary size of ActKernel invocation descriptor (bytes)">,
        Option<"actKernelRangeSize", "act-kernel-range-size", "int64_t", "0",
               "Binary size of ActKernel range descriptor (bytes)">,
        Option<"dmaSize", "dma-size", "int64_t", "0",
               "Binary size of DMA descriptor (bytes)">,
        Option<"m2iSize", "m2i-size", "int64_t", "0",
               "Binary size of M2I descriptor (bytes)">
    ];

}

def DumpStatisticsOfWlmOps : PassBase<"dump-statistics-of-wlm-ops", "vpux::FunctionPass"> {
    let summary = "Print stats on WLM related ops";

    let description = [{
        AI-generated summary

        Dumps WLM operation counts to the debug log: fetch ops, barrier programming DMAs,
        enqueue ops per task type (DMA, DPU, SHAVE), and descriptor group statistics.
        Purely diagnostic — no IR modifications.
    }];

    let constructor = "vpux::VPUMI40XX::createDumpStatisticsOfWlmOpsPass()";

}

def AddBarrierConfigurationOps : PassBase<"add-barrier-configuration-ops", "vpux::FunctionPass"> {
    let summary = "Prepare information and ops for initial barrier configuration";

    let constructor = "vpux::VPUMI40XX::createAddBarrierConfigurationOps()";

    let description = [{
        AI-generated summary

        This pass creates the barrier configuration constants and DMAs to help RT program the barriers during inference
        WorkloadManagementBarrierProgrammingMode::LEGACY
            Not implemented. The pass throws an error for this mode
        WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED
            Pass creates stride constant and raw barrier configurations to help runtime program the barriers.
            Stride constant is a 1D constant/array of size nPhysBarr. Index of the array are considered to be pid and values at index represent usage of the pid
            e.g. [1,2,4,2,5] pid 0 is used once, pid 1 is used 2 times, pid 3 is used 4 times and so on

            RawBarrierConfiguration is a 1D constant/array of size nPhysBarr x (max(barrierWithMaximumUsage, BARRIER_FIFO_DEPTH) + BARRIER_FIFO_DEPTH - 1)
            TODO: Explain reason for (max(barrierWithMaximumUsage, BARRIER_FIFO_DEPTH) + BARRIER_FIFO_DEPTH - 1)

            Assuming 16 Physical Barriers such that barrierWithMaximumUsage = 1 and BARRIER_FIFO_DEPTH=4
            Shape of RawBarrierConfiguration is 16 x (max(1, 4) + 4 - 1) => 16 x 7 = 112
            Such shape means for every pid we will have 7 configurations stored in this constant

            e.g.
            [Pid0_Usage0, 0, 0, 0, 0 ,0 , 0, 0, Pid1_Usage0, 0, 0, 0, 0, 0, 0 ......]

            Additionally, in this mode the compiler creates a DMA at bootstrap which will push barrier configurations
            to FIFO_BARRIERS_NCE_FILL_BARRIER_FIFO_ADR register. In this case runtime doesn't need to program initial barriers
            "Initial barriers reprogramming" refer to programming BARRIER_FIFO_DEPTH usages of each physical barriers

            Take following barriers used in network as example compiling on single cluster(nPhysBarr = 16)
            VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}> pid0
            VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 2 : ui8}> pid1
            VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}> pid2
            VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}> pid3
            VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}> pid4

            For each barrier usage, DMA writes 32bits in total where each 8 bits represents the following

                7 6 5 4 3 2 1 0                15 14 13 12 11 10 9 8           23 22 21 20 19 18 17 16      31  30 29 28 27 26 25 24
                0 0 0 0 0 0 0 1                 0  0  0  0  0  0 0 0            0 0  0  0  0  0  0  1       0   0  0  0  0  0  0  1
            <----------8bit----------->     <----------8bit----------->     <----------8bit----------->     <----------8bit----------->
            [   producer_count,             producer_interrupt_enabled,            consumer_count,          consumer_interrupt_enabled]

            e.g for pid0 first usage in above example we will have the value 0x01010001 which translates to
            Producer Count: 1, Producer Interrupt: 0, Consumer Count: 1, Consumer Interrupt: 1
            Since pid0 has only single use the other 3 entries are all 0s. Similarly the whole constant for above barriers as example looks like this

                [
            pid0    0x01010001, 0x0, 0x0, 0x0,
            pid1    0x01010002, 0x0, 0x0, 0x0,
            pid2    0x01010001, 0x0, 0x0, 0x0,
            pid3    0x01010001, 0x0, 0x0, 0x0,
            pid4    0x00010101, 0x0, 0x0, 0x0,
            pid5         0x0, 0x0, 0x0, 0x0,
            pid6         0x0, 0x0, 0x0, 0x0,
            pid7         0x0, 0x0, 0x0, 0x0,
            pid8         0x0, 0x0, 0x0, 0x0,
            pid9         0x0, 0x0, 0x0, 0x0,
            pid10        0x0, 0x0, 0x0, 0x0,
            pid11        0x0, 0x0, 0x0, 0x0,
            pid12        0x0, 0x0, 0x0, 0x0,
            pid13        0x0, 0x0, 0x0, 0x0,
            pid14        0x0, 0x0, 0x0, 0x0,
            pid15        0x0, 0x0, 0x0, 0x0,
                ]

            Similarly other constant can be deduced to have below unpacked values(only explaining non zero)
            Producer Count: 1, Producer Interrupt: 0, Consumer Count: 1, Consumer Interrupt: 1
            Producer Count: 2, Producer Interrupt: 0, Consumer Count: 1, Consumer Interrupt: 1
            Producer Count: 1, Producer Interrupt: 0, Consumer Count: 1, Consumer Interrupt: 1
            Producer Count: 1, Producer Interrupt: 0, Consumer Count: 1, Consumer Interrupt: 1
            Producer Count: 1, Producer Interrupt: 1, Consumer Count: 1, Consumer Interrupt: 0

            Furthermore, the pass creates DMAs in the schedule at right place
            which will push barrier configurations to FIFO_BARRIERS_NCE_FILL_BARRIER_FIFO_ADR register. In this case runtime doesn't need to program any barriers

            In full Workload Management mode, it is recommended to enable the consumer interrupt on at least one barrier per page to maintain inference activity.
            This mechanism allows the firmware to send periodic heartbeat refresh to the host, indicating that the inference is still running. To reduce interrupt overhead,
            the consumer interrupt is typically set only on the first barrier within each page, though less frequent signaling may still be sufficient depending on system requirements.

        WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED_4K
            This mode does everything what ALL_BARRIER_DMAS_SCHEDULED but instead of utilizing the barrier FIFO it utilizes the 4K Barrier block

    }];

    let options = [
        Option<
            "workloadManagementBarrierProgrammingModeOpt", "workload-management-barrier-programming-mode",
            "vpux::WorkloadManagementBarrierProgrammingMode", "",
            "Setting which controls mode of barrier programming",
            [{::llvm::cl::values(
            clEnumValN(vpux::WorkloadManagementBarrierProgrammingMode::LEGACY, "LEGACY",
                "Runtime handles barrier programming independently"),
            clEnumValN(vpux::WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED, "ALL_BARRIER_DMAS_SCHEDULED",
                "Compiler creates DMAs to program all barriers"),
            clEnumValN(vpux::WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED_4K, "ALL_BARRIER_DMAS_SCHEDULED_4K",
                "Compiler creates DMAs to program all barriers leveraging 4K barrier block")
          )}]
        >
    ];
}

def AddMappedInferenceVersionOp : PassBase<"add-mapped-inference-version-op", "vpux::FunctionPass"> {
    let summary = "Add API version metadata to MappedInferenceOp";

    let description = [{
        AI-generated summary

        Creates MappedInferenceVersionOp with major, minor, and patch version numbers.
        The driver checks these version numbers at load time to reject blobs
        compiled with an incompatible API version.
    }];

    let constructor = "vpux::VPUMI40XX::createAddMappedInferenceVersionOpPass()";
}

def UpdateEnqueueDMAInputAndOutput : PassBase<"update-enqueue-dma-input-and-output", "vpux::FunctionPass"> {
    let summary = "Update input and output data for DMAOps that perform task enqueue";

    let constructor = "vpux::VPUMI40XX::createUpdateEnqueueDMAInputAndOutput()";

    let description = [{
        For each DMA that perform task enqueue (has EnqueueDmaAttr) update its input with a constant that contains
        the address of task descriptor in CMX metadata space and output with HW FIFO register for given task type
    }];
}

def ConvertFetchDmasToFetchTaskOps : PassBase<"convert-fetch-dmas-to-fetch-task-ops", "vpux::FunctionPass"> {
    let summary = [{ Insert Fetch operations using predefined placeholders }];

    let description = [{
        Insert FetchOps for DPU & Shv tasks
        These tasks are responsible to copy descriptors for Variant,Invariant,ShaveInvocation and ShaveRange
        to CMX Metadata space. Lower in the pipeline these tasks are lowered to NNDMAs
    }];

    let constructor = "vpux::VPUMI40XX::createConvertFetchDmasToFetchTaskOpsPass()";
}

def UpdateFetchDMAForSkipDMAs : PassBase<"update-fetch-dmas-for-skip-dmas", "vpux::FunctionPass"> {
    let summary = [{ Updates placeholder fetch operations }];
    let description = [{
        Updates placeholder NNDMA Fetch DMAs corresponding to Skip DMAs and rewires them to
        fetch DMA task descriptors via task ranges.

        The pass:
        - Matches Fetch and Skip DMAs using a common descriptor id
        - Allocates DMA task descriptors in CMX from a fixed metadata base offset
        - Enforces deterministic phase-based ordering (DDR -> CMX) per
          (tile, list, logicalTask)
        - Creates task buffers and uses ViewTaskRange to pass descriptor ranges
          as inputs to Fetch DMAs
        - Replaces placeholder Fetch DMAs with new ones consuming these task ranges
        - Updates task locations to enable correct linking and execution
    }];

    let constructor = "vpux::VPUMI40XX::createUpdateFetchDMAForSkipDMAsPass()";
}
#endif
