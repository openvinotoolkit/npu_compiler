//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
        This pass replaces virtual barriers with physical barriers and assign IDs to them.
    }];

    let constructor = "vpux::VPURT::createAssignPhysicalBarriersPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >,
        Option<
            "colorBinEnableOpt", "color-bin-enable",
            "bool", "",
            "Enable color bin physical barrier assignment algorithm"
        >,
        Option<
            "virtualBarrierThresholdForWlmOpt", "workload-management-barrier-count-threshold",
            "int", "",
            "Threshold for WLM optimization"
        >
    ];
}

//
// SplitControlGraph
//

def SplitControlGraph : PassBase<"split-control-graph", "vpux::FunctionPass"> {
    let summary = "Split control graph";

    let description = [{
        This optional pass will split control graph for large models
        to reduce overhead of barrier processing and optimizations
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
        This pass will simplify schedule by:
            1. sharing wait / update barriers for DMAs
            2. removing cycled dependencies where possible
    }];

    let constructor = "vpux::VPURT::createSimplifySchedulePass()";
}

//
// SplitExceedingVariantCountBarriers
//

def SplitExceedingVariantCountBarriers : PassBase<"split-exceeding-variant-count-barriers", "vpux::FunctionPass"> {
    let summary = "Split exceeding variant (producer and consumer) count barriers";

    let description = [{
        This pass splits virtual barriers in the IR such that the number of producer slots <= MAX_VARIANT_COUNT / 2
        and the number of consumer slots <= MAX_VARIANT_COUNT / 2.
        Besides, there's a more strict limit named MAX_VARIANT_SUM from hw runtime. The value may be smaller than MAX_VARIANT_COUNT on same Arch.
        This limit will be enabled when MAX_VARIANT_SUM < MAX_VARIANT_COUNT and then check producers + consumers <= MAX_VARIANT_SUM. If not satisfied,
        It will split barriers to keep producers <= MAX_VARIANT_SUM / 2 and consumers <= MAX_VARIANT_SUM / 2.
    }];

    let constructor = "vpux::VPURT::createSplitExceedingVariantCountBarriersPass()";

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
// SatisfyOneWaitBarrierPerTask
//

def SatisfyOneWaitBarrierPerTask : PassBase<"satisfy-one-wait-barrier-per-task", "vpux::FunctionPass"> {
    let summary = "Satisfy each task has one wait barrier";

    let description = [{
        This pass will satisfy barrier safety constraint
        -- single barrier drives each task
        Handling multi barrier input can be wasteful and considerably more difficult to manage
        barrier safety. Pass is manipulating tasks with parallel wait barriers by two modes:
        1. Merge parallel wait barriers with legal barrier variant count
        2. Linearize parallel wait barrier producers with exceeding variant count
    }];

    let constructor = "vpux::VPURT::createSatisfyOneWaitBarrierPerTaskPass()";

    let options = [
        Option<
            "maxVariantCount", "max-variant-count",
            "int", "",
            "Number of producer and consumer slots per barrier, available for use"
        >,
        Option<
            "mergeWaitBarriersIteratively", "merge-wait-barriers-iteratively",
            "bool", "",
            "Enable iterative merge of parallel wait barriers until one wait barrier condition is satisfied."
        >,
        Option<
            "unevenVariantSplit", "enable-uneven-variant-split",
            "bool", "",
            "Enable uneven variant split"
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
        For NPU37XX and below: total producer variant count <= MAX_VARIANT_COUNT / 2 and total consumer variant count <= MAX_VARIANT_COUNT / 2
        For NPU40XX and above: total producer + consumer variant count <= MAX_VARIANT_COUNT.
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
        >,
        Option<
            "shareWaitAndUpdateBarriersOpt", "share-wait-and-update-barriers",
            "bool", "",
            "Enforce sharing wait and update barriers between tasks on same FIFO"
        >
    ];
}

//
// BarrierSimulation
//

def BarrierSimulation : PassBase<"barrier-simulation", "vpux::FunctionPass"> {
    let summary = "Simulate barriers";

    let description = [{
        Simulates the physical barrier execution and stops compilation on fail.
    }];

    let constructor = "vpux::VPURT::createBarrierSimulationPass()";
}

//
// InferenceExecutionAnalysis
//

def InferenceExecutionAnalysis : PassBase<"inference-execution-analysis", "vpux::FunctionPass"> {
    let summary = "Perform inference execution simulation to visualize schedule";

    let description = [{
        Simulate the schedule generated by the compiler and using the cost model visualize inference execution.
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
        Pass inserts barrier dependencies for passes like insert-fetch-ops, group-exec-ops
        Group Exec ops pass creates ExecutionGroups of DPU/SW tasks, for group of 32 tasks it chooses wait barrier
        of first task as wait barrier for execution group and update barrier of last task as update barrier for execution group

        insert-fetch-ops looks at these wait and update barriers for the group to determine the latest point the descriptors for
        the group must be fetched and earliest point when they can be evicted from metadata thus having a update and wait barrier
        is a must. This pass inserts these barriers if not present already due to other compiler optimizations
    }];

    let constructor = "vpux::VPURT::createInsertBarrierToMarkTheEndOfDescriptorGroupPass()";

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

}

#endif
