//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_ARCH_50XX_PASSES
#define VPUX_COMPILER_DIALECT_IE_ARCH_50XX_PASSES

include "mlir/Pass/PassBase.td"

//
// ConvertFakeConvertToDynamicDequantize
//

def ConvertFakeConvertToDynamicDequantize : PassBase<"convert-fake-convert-to-dynamic-dequantize", "vpux::FunctionPass"> {
    let summary = "Convert weight FakeConvert to DynamicDequantize";

    let description = [{
        The pass is a part of the `LowPrecision` pipeline.

        This pass materializes constant float8 weights represented by FakeConvert and replaces the FakeConvert with
        DynamicDequantize using reciprocal scale and negated shift. FakeConvert ops outside a weights path and cases
        that cannot be represented exactly as DynamicDequantize remain unchanged.
    }];

    let constructor = "vpux::IE::arch50xx::createConvertFakeConvertToDynamicDequantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertFakeConvertToFakeQuantize
//

def ConvertFakeConvertToFakeQuantize : PassBase<"convert-fake-convert-to-fake-quantize", "vpux::FunctionPass"> {
    let summary = "Convert activation FakeConvert to FakeQuantize";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        This pass converts activation FakeConvert ops into FakeQuantize ops.
        Weights are computed into DynamicDequantize ops by ConvertFakeConvertToDynamicDequantize.
    }];

    let constructor = "vpux::IE::arch50xx::createConvertFakeConvertToFakeQuantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

#endif
