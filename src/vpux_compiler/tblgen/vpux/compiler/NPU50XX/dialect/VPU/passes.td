//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_ARCH_50XX_PASSES
#define VPUX_COMPILER_DIALECT_VPU_ARCH_50XX_PASSES

include "mlir/Pass/PassBase.td"

//
// AutopadChannels
//

def AutopadChannels : PassBase<"autopad-channels", "vpux::FunctionPass"> {
    let summary = "Autopad channels";

    let description = [{
        Enable autopad for NCE operations where this feature could improve performance.
        Autopadding allows NCE operations to consume and produce data where the number of channels is smaller than 16.

        At the moment, only ODU autopad is supported by this pass. It searches for NCE operations whose results have
        the channels padded to 16 and is consumed by Shave operations, then removes the padding.
    }];


    let constructor = "vpux::VPU::arch50xx::createAutopadChannelsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

#endif
