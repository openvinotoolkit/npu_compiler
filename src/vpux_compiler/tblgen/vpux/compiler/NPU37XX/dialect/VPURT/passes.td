//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPURT_ARCH_37XX_PASSES
#define VPUX_COMPILER_DIALECT_VPURT_ARCH_37XX_PASSES

include "mlir/Pass/PassBase.td"

//
// AddFinalBarrier
//

def AddFinalBarrier : PassBase<"add-final-barrier", "vpux::FunctionPass"> {
    let summary = "Adds a dummy barrier to state that the task execution flow has been finished";

    let description = [{
        Adds a dummy update barrier state that the task execution flow has been finished.
        When the barrier's producer count is changed to zero, it means all tasks are finished.
    }];

    let constructor = "vpux::VPURT::arch37xx::createAddFinalBarrierPass()";
}

#endif
