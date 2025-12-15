//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_LOGICAL
#define VPUX_COMPILER_DIALECT_IE_OPS_LOGICAL

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_AndOp :
        IE_LayerOp<
            "And",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine And layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, F64, SI32, SI64]>:$input1,
        RankedTensorOf<[I8, F16, F32, F64, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast

    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, F64, SI32, SI64, quant_QuantizedType]>:$output
    );
}

def IE_LogicalNotOp :
        IE_LayerOp<
            "LogicalNot",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Logical Not layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, F64, SI32, SI64]>:$input1
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, F64, SI32, SI64]>:$output
    );
}

def IE_LogicalOrOp :
        IE_LayerOp<
            "LogicalOr",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine LogicalOr layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, F64, SI32, SI64]>:$input1,
        RankedTensorOf<[I8, F16, F32, F64, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, F64, SI32, SI64]>:$output
    );
}

def IE_LogicalXorOp :
        IE_LayerOp<
            "LogicalXor",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine LogicalXor layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, F64, SI32, SI64]>:$input1,
        RankedTensorOf<[I8, F16, F32, F64, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, F64, SI32, SI64]>:$output
    );
}

#endif
