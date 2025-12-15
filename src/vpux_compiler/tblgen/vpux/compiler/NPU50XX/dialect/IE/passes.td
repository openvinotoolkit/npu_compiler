//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_ARCH_50XX_PASSES
#define VPUX_COMPILER_DIALECT_IE_ARCH_50XX_PASSES

include "mlir/Pass/PassBase.td"

//
// ConvertFakeConvertToFakeQuantize
//

def ConvertFakeConvertToFakeQuantize : PassBase<"convert-fake-convert-to-fake-quantize", "vpux::FunctionPass"> {
    let summary = "Convert FakeConvert to FakeQuantize";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        This pass converts FakeConvert ops into FakeQuantize ops, which are already implemented in the lower dialects.
        Thus, FakeConvert ops only needs to be supported by the IE dialect.
        FakeConvert's scale and shift attributes are converted to FakeQuantize's input/output intervals, and the
        float8 destination type is preserved by the resulting operation.
    }];

    let constructor = "vpux::IE::arch50xx::createConvertFakeConvertToFakeQuantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConsolidateActivationFP8Quantization
//

def ConsolidateActivationFP8Quantization : PassBase<"consolidate-activation-fp8-quantization", "vpux::FunctionPass"> {
    let summary = "Replace activation FP8 quantization with explicit Multiplies";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        This pass targets FP8 quantized activation with 4-bit weights.
        It identifies a MatMul / FullyConnected operation with FP8 input and INT4 weights, where both the weights and the quantization parameters are inputs.
        This pattern then decomposes FakeConvert to GroupConv->Quantize->Dequantize->Divide, then move divide post MatMul/FC.
    }];

    let constructor = "vpux::IE::arch50xx::createConsolidateActivationFP8QuantizationPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

#endif
