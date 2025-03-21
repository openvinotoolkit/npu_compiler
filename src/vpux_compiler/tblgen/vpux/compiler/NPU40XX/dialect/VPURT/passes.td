//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPURT_ARCH_40XX_PASSES
#define VPUX_COMPILER_DIALECT_VPURT_ARCH_40XX_PASSES

include "mlir/Pass/PassBase.td"

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

    let constructor = "vpux::VPURT::arch40xx::createInsertSyncTasksPass()";
}

//
// OptimizeSyncTasks
//

def OptimizeSyncTasks : PassBase<"optimize-sync-tasks", "vpux::FunctionPass"> {
    let summary = "Remove unnecessary sync tasks";

    let description = [{
        Remove unnecessary sync task that were injected previously by InsertSyncTasks pass.
        Example:

                 /-> op -> .. -> op -\                       /-> op -> .. -> op -\
        SyncTask --> op -> .. -> op --> SyncTask -> SyncTask --> op -> .. -> op --> SyncTask
                 \-> op -> .. -> op -/                       \-> op -> .. -> op -/

                        =>

        /-> op -> .. -> op -\           /-> op -> .. -> op -\
        --> op -> .. -> op --> SyncTask --> op -> .. -> op -->
        \-> op -> .. -> op -/           \-> op -> .. -> op -/
    }];

    let constructor = "vpux::VPURT::arch40xx::createOptimizeSyncTasksPass()";
}

//
// FindWlmEnqueueBarrier
//

def FindWlmEnqueueBarrier : PassBase<"find-wlm-enqueue-barrier", "vpux::FunctionPass"> {
    let summary = "For WLM for each task find enqueue barrier";

    let description = [{
        For WLM each task needs to be enqueued at some point. This pass analyzes schedule
        and for each task finds a barrier at which it can be enqueued. Later this information
        will be lowered to dedicated enqueue ops (WorkItem task)
    }];

    let constructor = "vpux::VPURT::arch40xx::createFindWlmEnqueueBarrierPass()";
}

//
// OrderBarriersForWlm
//

def OrderBarriersForWlm : PassBase<"order-barriers-for-wlm", "vpux::FunctionPass"> {
    let summary = "Order barriers for WLM";

    let description = [{
        For WLM barriers are epxected to be ordered following consumption order
    }];

    let constructor = "vpux::VPURT::arch40xx::createOrderBarriersForWlmPass()";
}

#endif
