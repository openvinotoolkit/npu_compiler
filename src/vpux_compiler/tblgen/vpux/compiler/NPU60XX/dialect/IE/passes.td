//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_ARCH_60XX_PASSES
#define VPUX_COMPILER_DIALECT_IE_ARCH_60XX_PASSES

include "mlir/Pass/PassBase.td"

//
// U16QuantizationFallback
//

def U16QuantizationFallback : PassBase<"u16-quantize-fallback", "vpux::FunctionPass"> {
    let summary = "Fallback for U16 quantization";

    let description = [{
        After the previous passes have been executed, U16QuantizationFallback checks whether an operation is still eligible for mapping to DCIM.
        If the operation is no longer eligible (e.g., the layout has changed), the pass dequantizes the operation input from U16 to FP16.
    }];

    let constructor = "vpux::IE::arch60xx::createU16QuantizationFallbackPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}
#endif
