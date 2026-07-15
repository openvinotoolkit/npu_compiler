//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPURT_PASSES
#define VPUX_COMPILER_DIALECT_VPURT_PASSES

include "mlir/Pass/PassBase.td"

//
// AssignPhysicalBarriers
//

def AssignPhysicalBarriers : PassBase<"assign-physical-barriers", "vpux::FunctionPass"> {
    let summary = "Assign physical barriers";

    let description = [{
        AI-generated summary

        Maps virtual barriers to physical hardware barrier slots on the NPU. Uses a
        graph-coloring algorithm to assign physical barrier IDs to virtual barriers such that no
        two concurrently active barriers share the same physical slot, respecting the hardware
        limit on available physical barriers.
    }];

    let constructor = "vpux::VPURT::createAssignPhysicalBarriersPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >,
        Option<
            "workloadManagementModeOpt", "workload-management-mode",
            "vpux::WorkloadManagementMode", "",
            "Setting which controls mode of WLM",
            [{::llvm::cl::values(
                clEnumValN(vpux::WorkloadManagementMode::PWLM_V0_1_PAGES, "PWLM_V0_1_PAGES", "PWLM with split into subgraphs (pages)"),
                clEnumValN(vpux::WorkloadManagementMode::FWLM_V1_PAGES, "FWLM_V1_PAGES", "FWLM with split into subgraphs(pages)")
            )}]
        >
    ];
}

//
// SplitControlGraph
//

def SplitControlGraph : PassBase<"split-control-graph", "vpux::FunctionPass"> {
    let summary = "Split control graph";

    let description = [{
        AI-generated summary

        Splits the barrier control graph into smaller blocks of tasks to reduce memory
        usage and compilation time for barrier scheduling passes. If the total number of
        tasks is smaller than the configured block size, no split is performed. After
        splitting, execution tasks and barriers are reordered and the split is verified
        for correctness. Memory usage of scheduling passes is proportional to
        blockSize^2 / 8.
    }];

    let constructor = "vpux::VPURT::createSplitControlGraphPass()";

    let options = [
        Option<
            "blockSize", "block-size",
            "int", "",
            "Size of the block in number of tasks control graph will be split into. This impacts the amount of memory barrier scheduling passes will need to perform their legalization and optimization steps. Mem usage is proportional to blockSize^2 / 8"
        >
    ];
}

//
// SimplifySchedule
//

def SimplifySchedule : PassBase<"simplify-schedule", "vpux::FunctionPass"> {
    let summary = "Simplify schedule for runtime";

    let description = [{
        AI-generated summary

        Simplifies the barrier-based execution schedule through three optimizations:
            1. Resolves out-of-order dependencies for DMA tasks in the same FIFO by
               linking update barriers of earlier tasks to next tasks with earlier
               update barriers
            2. Shares wait and update barriers between tasks on the same FIFO so
               execution is fully controlled by barriers (WLM-dependent)
            3. Reduces parallel overlapping control flows per DMA FIFO by using cycle
               cost simulation to link tasks to intermediate barriers when performance
               is not affected
    }];

    let constructor = "vpux::VPURT::createSimplifySchedulePass()";
    let options = [
        Option<
            "shareWaitAndUpdateBarriersOpt", "share-wait-and-update-barriers",
            "bool", "",
            "Enforce sharing wait and update barriers between tasks on the same FIFO"
        >
    ];
}

//
// SplitExceedingBarrierSlotCountPass
//

def SplitExceedingBarrierSlotCount : PassBase<"split-exceeding-barrier-slot-count", "vpux::FunctionPass"> {
    let summary = "Split barriers with the number of producers and consumers exceeding the maximal barrier slot count";

    let description = [{
        This pass splits virtual barriers in the IR such that the number of producer slots <= MAX_VARIANT_COUNT / 2
        and the number of consumer slots <= MAX_VARIANT_COUNT / 2.
        Besides, there's a more strict limit named MAX_VARIANT_SUM from hw runtime. The value may be smaller than MAX_VARIANT_COUNT on same Arch.
        This limit will be enabled when MAX_VARIANT_SUM < MAX_VARIANT_COUNT and then check producers + consumers <= MAX_VARIANT_SUM. If not satisfied,
        It will split barriers to keep producers <= MAX_VARIANT_SUM / 2 and consumers <= MAX_VARIANT_SUM / 2.
    }];

    let constructor = "vpux::VPURT::createSplitExceedingBarrierSlotCountPass()";

    let options = [
        Option<
            "maxVariantCount", "max-variant-count",
            "int", "",
            "To control available number (a half of maxVariantCount) of producer and consumer slots per barrier"
        >,
        Option<
            "maxVariantSum", "max-variant-sum",
            "int", "",
            "Maximum sum of producer and consumer slots per barrier"
        >
    ];
}

//
// ReduceExceedingActiveCountBarriers
//

def ReduceExceedingActiveCountBarriers : PassBase<"reduce-exceeding-active-count-barriers", "vpux::FunctionPass"> {
    let summary = "Reduce exceeding active barrier count";

    let description = [{
        This pass linearizes virtual barriers in the IR such that the number of active barriers at any time
        does not exceed the physical number of available barriers.
        For VPU37XX and below: total producer variant count <= MAX_VARIANT_COUNT / 2 and total consumer variant count <= MAX_VARIANT_COUNT / 2
        For VPU40XX and above: total producer + consumer variant count <= MAX_VARIANT_COUNT.
        With exceeding active barrier count:
        - parallel barriers can be merged
        - parallel consumers to barrier can be linearized
        - parallel barriers can be linearized
        If nothing can be applied extreme linearization will be performed, where all tasks linked to active barriers
        will be linearized.
    }];

    let constructor = "vpux::VPURT::createReduceExceedingActiveCountBarriersPass()";

    let options = [
        Option<
            "numBarriers", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >,
        Option<
            "maxVariantCount", "max-variant-count",
            "int", "",
            "Number of producer and consumer slots per barrier, available for use"
        >,
        Option<
            "unevenVariantSplit", "enable-uneven-variant-split",
            "bool", "",
            "Enable uneven variant split"
        >
    ];
}

//
// BarrierSimulation
//

def BarrierSimulation : PassBase<"barrier-simulation", "vpux::FunctionPass"> {
    let summary = "Simulate barriers";

    let description = [{
        AI-generated summary

        Simulates barrier-based execution scheduling to verify correctness of the barrier
        assignment. Walks the execution graph and simulates producer/consumer barrier signaling,
        checking for deadlocks, barrier overflow, and ordering violations before generating
        final firmware descriptors.
    }];

    let constructor = "vpux::VPURT::createBarrierSimulationPass()";
}

//
// InferenceExecutionAnalysis
//

def InferenceExecutionAnalysis : PassBase<"inference-execution-analysis", "vpux::FunctionPass"> {
    let summary = "Perform inference execution simulation to visualize schedule";

    let description = [{
        AI-generated summary

        Runs cycle-accurate inference simulation (InferenceExecutionSimulator) over the
        barrier-resolved schedule. Computes total inference latency in cycles, DPU/SHAVE
        energy consumption, and activity factor. Stores the activity factor and timing
        in module attributes for downstream power estimation and schedule visualization.
    }];

    let constructor = "vpux::VPURT::createInferenceExecutionAnalysisPass()";
}

//
// IntermediateBufferOutput
//

def IntermediateBufferOutput : PassBase<"intermediate-buffer-output", "vpux::FunctionPass"> {
    let summary = "Accuracy debug utility which can dump a target buffer to output.";

    let description = [{
        Accuracy debug utility allowing to output intermediate buffer.
        The pass will insert a DMA to output for a target buffer, of a target operation, at a target index.
        All parameters need to be provided.
        Details about usage can be found in `guides/how-to-debug.md`
    }];

    let constructor = "vpux::VPURT::createIntermediateBufferOutputPass()";

    let options = [
        Option<
            "opIndexVal", "op-index",
            "int", "",
            "Target operation to use for buffer selection"
        >,
        Option<
            "bufferIndexVal", "buffer-index",
            "int", "",
            "Target operation operand / result index to output"
        >,
        Option<
            "insertionIndexVal", "insertion-index",
            "int", "",
            "Targed operation index after which DMA to output should be inserted"
        >
    ];
}

//
// InsertBarrierToMarkTheEndOfDescriptorGroup
//

def InsertBarrierToMarkTheEndOfDescriptorGroup : PassBase<"insert-barrier-to-mark-end-of-descriptor-group", "vpux::FunctionPass"> {
    let summary = "Pass inserts barriers to mark end of descriptor/execution group";

    let description = [{
        AI-generated summary

        Inserts barrier dependencies required for WLM descriptor group management.
        The pass is skipped for FWLM_V1_PAGES mode. If the number of virtual barriers
        exceeds a configurable threshold, the pass signals failure and disables WLM.
        It processes both DPU and SHV (ActShave) execution groups obtained from
        ExecutionGroupAnalysis, ensuring each group has a wait barrier on its first
        task and an update barrier on its last task. These barriers are required by
        subsequent passes (insert-fetch-ops, group-exec-ops) to determine descriptor
        fetch and eviction points for WLM scheduling.
    }];

    let constructor = "vpux::VPURT::createInsertBarrierToMarkTheEndOfDescriptorGroupPass()";
}

//
// AddFinalBarrier
//

def AddFinalBarrier : PassBase<"add-final-barrier", "vpux::FunctionPass"> {
    let summary = "Adds a dummy barrier to state that the task execution flow has been finished";

    let description = [{
        Adds a dummy update barrier state that the task execution flow has been finished.
        When the barrier's producer count is changed to zero, it means all tasks are finished.
    }];

    let constructor = "vpux::VPURT::createAddFinalBarrierPass()";
}

//
// InsertSyncTasks
//

def InsertSyncTasks : PassBase<"insert-sync-tasks", "vpux::FunctionPass"> {
    let summary = "Insert sync tasks";

    let description = [{
        Insert sync task that mark start and end of each FuncOp.
        Example:

               | op -> .. -> op |
        funcOp{| op -> .. -> op |}
               | op -> .. -> op |

                        =>

               |          /-> op -> .. -> op -\           |
        funcOp{| SyncTask --> op -> .. -> op --> SyncTask |}
               |          \-> op -> .. -> op -/           |

    }];

    let constructor = "vpux::VPURT::createInsertSyncTasksPass()";
}

//
// OptimizeSyncTasks
//

def OptimizeSyncTasks : PassBase<"optimize-sync-tasks", "vpux::FunctionPass"> {
    let summary = "Remove unnecessary sync tasks";

    let description = [{
        AI-generated summary

        Removes unnecessary sync tasks injected by InsertSyncTasks. The pass applies
        multiple optimization patterns:
        - MergeSyncTasks: merges consecutive sync tasks connected by a single barrier
           with no other users, removing the intermediate barrier and redundant task.
        - Redundant sync task removal: identifies sync tasks in regions
           "sync -> ... -> sync" whose synchronization responsibility can be taken
           over by a neighboring (parent or child) task, provided barrier variant
           count limits are not exceeded.
        - First/last sync task removal: unconditionally removes the very first and
           last sync tasks in the schedule, cleaning up associated barriers when safe.

        Example (MergeSyncTasks):

                 /-> op -> .. -> op -\                       /-> op -> .. -> op -\
        SyncTask --> op -> .. -> op --> SyncTask -> SyncTask --> op -> .. -> op --> SyncTask
                 \-> op -> .. -> op -/                       \-> op -> .. -> op -/

                        =>

        /-> op -> .. -> op -\           /-> op -> .. -> op -\
        --> op -> .. -> op --> SyncTask --> op -> .. -> op -->
        \-> op -> .. -> op -/           \-> op -> .. -> op -/
    }];

    let constructor = "vpux::VPURT::createOptimizeSyncTasksPass()";
}

//
// WlmSplitGraphToPages
//

def WlmSplitGraphToPages : PassBase<"wlm-split-graph-to-pages", "vpux::FunctionPass"> {
    let summary = "For WLM split model into subgraphs (pages)";

    let description = [{
        AI-generated summary

        Splits the WLM execution graph into subgraphs (pages) limited by the number of
        physical barriers, using BarrierPagesSplitHandler for assignment. Before splitting,
        the pass legalizes SHV-with-DPU scheduling by inserting dummy DMAs before sync DPU
        tasks that consume barriers produced by SHV-with-DPU on the same tile. After page
        assignment, dummy tasks are created for pages that end up with no tasks. When
        dedicated SHV FIFOs are not supported, the pass corrects SHV task page assignments
        to ensure consecutive tasks on the same cluster never have decreasing page numbers.
    }];

    let constructor = "vpux::VPURT::createWlmSplitGraphToPagesPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >
    ];
}

//
// WlmLegalizeSplitGraphToPages
//

def WlmLegalizeSplitGraphToPages : PassBase<"wlm-legalize-split-graph-to-pages", "vpux::FunctionPass"> {
    let summary = "For WLM legalize split model into subgraphs (pages)";

    let description = [{
        AI-generated summary

        Legalizes an existing Workload Manager barrier-page split so that the schedule
        conforms to firmware paging constraints:
        - Resolves dependencies that span beyond neighbor pages
        - Adds update barriers for last tasks on each page's FIFO when missing
        - Ensures boundary tasks between adjacent pages have proper dependencies
        - Verifies the final split is valid with no cyclic dependencies
    }];

    let constructor = "vpux::VPURT::createWlmLegalizeSplitGraphToPagesPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >
    ];
}

//
// WlmLegalizePagesForBarrierDmas
//

def WlmLegalizePagesForBarrierDmas : PassBase<"wlm-legalize-pages-for-barrier-dmas", "vpux::FunctionPass"> {
    let summary = "For WLM legalize split into pages for reprogramming Barrier FIFOs using compiler DMAs";

    let description = [{
        This pass is intended to legalize split into pages for WLM in case compiler will be responsible
        for programming virtual instances of physical barriers by writing to the Barrier FIFO using DMAs
        During this pass, we not only adjust the schedule but also insert reference DMAs
        These reference DMAs serve as placeholders and will later be replaced with actual DMAs that will
        write barrier descriptors to the Barrier FIFO registers
    }];

    let constructor = "vpux::VPURT::createWlmLegalizePagesForBarrierDmasPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >,
        Option<
            "barrierFifoDepthOpt", "barrier-fifo-depth",
            "int", "",
            "Depth of barrier fifo"
        >
    ];
}

//
// WlmInsertDummyDmasInPages
//

def WlmInsertDummyDmasInPages : PassBase<"wlm-insert-dummy-dmas-in-pages", "vpux::FunctionPass"> {
    let summary = "For WLM pages split insert dummy DMAs if given page doesn't have this DMA type";

    let description = [{
        AI-generated summary

        Ensures each WLM page has at least one DMA of each required type to enable
        bootstrap enqueuing of all DMAs as a single linked list. Before inserting
        type-specific dummy DMAs, pages with only one barrier are first legalized by
        inserting a dummy DMA and barrier to guarantee at least two barriers per page.
        Dummy DMAs are created with channel-type-aware buffers (DDR vs CMX) matching
        the target DMA queue type. After insertion, a post-legalization step verifies
        and adds update barriers for last tasks on each FIFO per page that lack them.
        Final verification ensures page splits, barrier page constraints, absence of
        cyclic dependencies, and control graph correctness.
    }];

    let constructor = "vpux::VPURT::createWlmInsertDummyDmasInPagesPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >
    ];
}

//
// WlmInsertDummyBarriersInPages
//

def WlmInsertDummyBarriersInPages : PassBase<"wlm-insert-dummy-barriers-in-pages", "vpux::FunctionPass"> {
    let summary = "For WLM pages split insert dummy barriers if given page doesn't use half of physical barriers";

    let description = [{
        This pass checks if in each page except last one number of barriers is equal to half of
        number of physical barriers. If not, then it inserts dummy barriers to fill the gap.
        Dummy barriers will be inserted parallel to some existing barrier with same consumer and producer.
    }];

    let constructor = "vpux::VPURT::createWlmInsertDummyBarriersInPagesPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >
    ];
}

//
// OptimizeBarriersSlotsUsage
//

def OptimizeBarriersSlotsUsage : PassBase<"optimize-barriers-slots-usage", "vpux::FunctionPass"> {
    let summary = "Perform final optimization of barriers slots usage";

    let description = [{
        This pass should be called at very end of scheduling before backend and perform final optimization
        of barriers slots usage so that barriers do not have redundant producers or consumers which
        are controlled by HW FIFO dependencies - no barrier should have more than 1 producer or 1 consumer
        from tasks of same queue type. This pass should not be used for nonWLM config as task barrier production
        or consumption events are used by runtime for task enqueue or descriptor loading and if task-barrier relation
        gets optimized this mechanism may be spoiled
    }];

    let constructor = "vpux::VPURT::createOptimizeBarriersSlotsUsagePass()";
}

//
// FindWlmEnqueueDmasBarrier
//

def FindWlmEnqueueDmasBarrier : PassBase<"find-wlm-enqueue-dmas-barrier", "vpux::FunctionPass"> {
    let summary = "For Full WLM with WLM pages split find barriers for enqueuing DMAs";

    let description = [{
        AI-generated summary

        Creates enqueue DMA operations for Full WLM scheduling. The pass is skipped
        when WLM is disabled or when the model contains only DMA tasks. For each
        execution group (DPU and ActShave), uses BarrierPagesSplitHandler and
        ExecutionGroupAnalysis to determine which barrier triggers the enqueue. Inserts
        NNDMA operations with EnqueueDMAAttr into the IR, optimizes redundant barrier
        dependencies introduced by the insertions, and legalizes page splits for any
        enqueue DMAs that end up as the last task on a FIFO.
    }];

    let constructor = "vpux::VPURT::createFindWlmEnqueueDmasBarrierPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >
    ];
}

//
// FindWlmEnqueueBarrierWithPages
//

def FindWlmEnqueueBarrierWithPages : PassBase<"find-wlm-enqueue-barrier-with-pages", "vpux::FunctionPass"> {
    let summary = "For Partial WLM with WLM pages split find barriers for task enqueue";

    let description = [{
        AI-generated summary

        Determines and assigns enqueue barriers for Partial WLM scheduling with page
        splits. Uses BarrierPagesSplitHandler to compute and legalize which barrier
        safely triggers enqueue for each task. Sets the enqueueBarrier attribute on
        task operations so the runtime knows when to schedule each task. Tasks with
        no assigned barrier are enqueued at bootstrap.
    }];

    let constructor = "vpux::VPURT::createFindWlmEnqueueBarrierWithPagesPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >
    ];
}

//
// CheckWlmPageSplitConstraints
//

def CheckWlmPageSplitConstraints : PassBase<"check-wlm-page-split-constraints", "vpux::FunctionPass"> {
    let summary = "For WLM verify pages split constraints are satisfied and schedule is valid";

    let description = [{
        AI-generated summary

        Verifies WLM page split constraints and schedule validity. Checks include:
        barrier page assignment validity, non-decreasing task page order on each queue,
        absence of cyclic dependencies, correct control graph split, and physical
        barrier dependencies. For Full WLM (FWLM_V1_PAGES), additionally verifies
        barrier reprogramming DMA dependencies, enqueue DMA correctness, and DMA
        enqueue ordering. For both Partial and Full WLM, verifies fetch DMA
        dependencies around DPU and SHV execution groups.
    }];

    let constructor = "vpux::VPURT::createCheckWlmPageSplitConstraintsPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >
    ];
}

#endif
