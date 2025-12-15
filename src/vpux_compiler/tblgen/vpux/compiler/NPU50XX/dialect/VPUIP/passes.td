//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_ARCH_50XX_PASSES
#define VPUX_COMPILER_DIALECT_VPUIP_ARCH_50XX_PASSES

include "mlir/Pass/PassBase.td"

//
// InsertDelayDPUVariant
//

def InsertDelayDPUVariant : PassBase<"insert-delay-dpu-variant", "vpux::FunctionPass"> {
    let summary = "Add dummy DPU variant in case of sprLUT, pallet LUT, DPU Dynamic task scaling, or ODU autopad feature usage";

    let description = [{
        As sprLUT, pallet LUT or SCALED_SEQ_LEN are prefetched before all the barriers for the DPU are produced
        (see DPU FSM diagram in HAS), we need to be sure that, once we read them, the DMA tasks that bring them into the CMX are completed.
        This is done by adding an additional dummy DPU variant, which exists purely for waiting
        until all the barriers are produced. It is important to do it before WLM legalization passes,
        because it is expected that no tasks will be inserted between legalization and main WLM pipeline.

        Similarly, in case an operation writes in other clusters via ITI, there is no guarantee that the write
        has been completed by the time the consumer tasks have started execution. This can result in incorrect data
        being accessed by the consumers. To prevent this, a small variant is added at the end of the list of variants
        which will produce a subset of the data as a previous variant, but without halo regions.
        This represents a delay in execution, which should give the ITI writes enough time to finish the transaction.
        Note: This is only done when ODU autopad is used by an operation, as these cases were discovered to lead to
        a race condition. The likely cause is the unaligned memory accesses when the channels are not aligned to 16 bytes.
    }];

    let constructor = "vpux::VPUIP::arch50xx::createInsertDelayDPUVariantPass()";

    let options = [
        Option<
            "dpuProfilingEnabled", "dpu-profiling-enabled",
            "bool", "false",
            "Specifies whether the DPU profiling feature is enabled"
        >
    ];
}

#endif
