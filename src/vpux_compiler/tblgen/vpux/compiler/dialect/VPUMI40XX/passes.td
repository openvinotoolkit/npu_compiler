//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI40XX_PASSES
#define VPUX_COMPILER_DIALECT_VPUMI40XX_PASSES

include "mlir/Pass/PassBase.td"

//
// SetupProfilingVPUMI40XX
//

def SetupProfilingVPUMI40XX : PassBase<"setup-profiling-VPUMI40XX", "vpux::ModulePass"> {
    let summary = "Apply HW profiling settings in MI";

    let description = "This pass configures DmaHwpBase and HwpWorkpointCfg.";

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
        "vpux::VPURT::VPURTDialect",
        "vpux::VPUMI40XX::VPUMI40XXDialect",
        "vpux::VPURegMapped::VPURegMappedDialect",
        "vpux::ELFNPU37XX::ELFNPU37XXDialect"
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
    let summary = [{Reorder Mapped Inference dialect ops ot a *canonical* order.}];
    let description = [{
        The mapped inference representation does not follow the *classical* rules of operation order inside the IR.
        In higher level dialects, OP order in the IR mostly followed the topological order of the execution, with
        Async ops and VPURT.TaskOps generally decorating extra information on the parallelism.

        In MappedInference representation execution flow is expressed differently, as it explicitly defines the list
        semantics of how the operations follow each other in execution flow, while parralelism is expressed at "list level".

        This pass will just re-order the OPS inside the IR to folow this semantics. This reorder serves EXCLUSIVELY
        for HUMAN READABILITY of the IR but serves NO FUNCTIONAL purpose, and does not introduce positional semantics to the dialect.
    }];

    let constructor = "vpux::VPUMI40XX::reorderMappedInferenceOpsPass()";
}

def ResolveTaskLocation : PassBase<"resolve-task-location", "vpux::VPURegMapped::ResolveTaskLocationPass"> {
    let summary = [{ 40xx-specific version of resolve task location pass }];

    let constructor = "vpux::VPUMI40XX::createResolveTaskLocationPass()";
}

def BarrierTopologicalMapping : PassBase<"barrier-topological-mapping", "vpux::FunctionPass"> {
    let summary = [{Construct barrier topological dependencies}];

    let description = [{
        Created bar-to-bar dependencies represent a barrier only graph generated
        from original schedule with task nodes removed.
        Example schedule:
            BAR0 -> OP0 -> BAR1 -> OP1 -> BAR2
              \------------OP2---------->/
        Barrier dependencies:
            BAR0 --------> BAR1 --------> BAR2
              \------------------------->/

        Definition:
        BARx depends on BARy if BARx can get updated (BARx pcount decremented) when BARy was produced

        Those dependencies are to be used by add-enqueue-ops pass.
    }];

    let constructor = "vpux::VPUMI40XX::createBarrierTopologicalMappingPass()";
}

def GroupExecutionOps : PassBase<"group-execution-ops", "vpux::FunctionPass"> {
    let summary = [{Group execution ops into one view}];

    let constructor = "vpux::VPUMI40XX::createGroupExecutionOpsPass()";
}

def UnGroupExecutionOps : PassBase<"ungroup-execution-ops", "vpux::FunctionPass">  {
    let summary = [{UnGroup execution ops from one view}];

    let constructor = "vpux::VPUMI40XX::createUnGroupExecutionOpsPass()";
}

def AddFetchOps : PassBase<"add-fetch-ops", "vpux::FunctionPass"> {
    let summary = [{ Insert Fetch operations }];

    let description = [{
        Insert FetchOps for DPU & Shv tasks
        These tasks are responsible to copy descriptors for Variant,Invariant,ShaveInvocation and ShaveRange
        to CMX Metadata space. Lower in the pipeline these tasks are lowered to NNDMAs
    }];

    let constructor = "vpux::VPUMI40XX::createAddFetchOpsPass()";
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
    let summary = [{ Propagate info about final barrier to mapped inference}];

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

    let options = [
        Option<
            "enableWlmVpurtEnqueueOpt", "enable-wlm-vpurt-enqueue",
            "bool", "false",
            "Enable usage of predefined enqueue targets from VPURT"
        >
    ];

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
}

def LinkAllOps : PassBase<"link-all-ops", "vpux::FunctionPass"> {
    let summary = [{ Hard-Link all ops that support HardLink}];

    let description = [{
        For non-managed inferences, runtime will support and expect all tasks within a list to be HW link listed (hard-linked).
        This pass will hardLink all tasks that have support for it, with the exception of the first task.
    }];

    let constructor = "vpux::VPUMI40XX::createLinkAllOpsPass()";
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
    let summary = [{ Split enqueue Ops fot avoid bug when we have last op in task buffer in LL }];

    let description = [{
        To account for preemption at a group boundary level scenario we have to leave last task in fetch group
        unlinked. The problem with preemmption is due to concurrency between thread processing preemption and execution
        thread, upon restore of previous state we can't realiably check if last task in group was linked, since it could be
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
    let summary = [{ Unroll enqueue Ops so that each targets only one task}];

    let constructor = "vpux::VPUMI40XX::createUnrollEnqueueOpsPass()";
}

def LinkEnqueueOpsForSameBarrier : PassBase<"link-enqueue-ops-for-same-barrier", "vpux::FunctionPass"> {
    let summary = [{ Create a link to previous enqueue happening on same barrier}];

    let constructor = "vpux::VPUMI40XX::createLinkEnqueueOpsForSameBarrierPass()";
}

def AddBootstrapOps : PassBase<"add-bootstrap-ops", "vpux::FunctionPass"> {
    let summary = [{ Add Bootstrap Ops}];
    let description = [{Previously we initialize and used barriers on FW in fixed order
                        from 0 to MAX_BARRIER.
                        Bootstrap operation allow us start barriers in any order}];

    let constructor = "vpux::VPUMI40XX::createAddBootstrapOpsPass()";
}

def NextSameIdAssignment : PassBase<"next-same-id-assignment", "vpux::FunctionPass"> {
    let summary = "Compute and setup scheduling related fields";

    let description = [{
        This pass computes and sets up the (barrier) scheduling related fields - next_same_id, previousSameId
    }];

    let constructor = "vpux::VPUMI40XX::createNextSameIdAssignmentPass()";

}

def AddPlatformInfo : PassBase<"add-platform-info", "vpux::FunctionPass"> {
    let summary = "Adds platform info operation which provides info such as arch kind";

    let constructor = "vpux::VPUMI40XX::createAddPlatformInfoPass()";

}

def DumpStatisticsOfWlmOps : PassBase<"dump-statistics-of-wlm-ops", "vpux::FunctionPass"> {
    let summary = "Print stats on WLM related ops";

    let description = [{
        This pass dumps count of enqueue and fetch DMA ops inserted as part of WLM
    }];

    let constructor = "vpux::VPUMI40XX::createDumpStatisticsOfWlmOpsPass()";

}

def AddInitialBarrierConfigurationOps : PassBase<"add-initial-barrier-confuguration-ops", "vpux::FunctionPass"> {
    let summary = "Prepare information for initial barrier configuration";

    let constructor = "vpux::VPUMI40XX::createAddInitialBarrierConfigurationOps()";
}

def AddMappedInferenceVersionOp : PassBase<"add-mapped-inference-version-op", "vpux::FunctionPass"> {
    let summary = "Add explicit operation for Mapped Inference and Managed Mapped inference version representation";

    let constructor = "vpux::VPUMI40XX::createAddMappedInferenceVersionOpPass()";
}

#endif
