//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_COMPARISON
#define VPUX_COMPILER_DIALECT_IE_OPS_COMPARISON

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/core/types.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_EqualOp :
        IE_LayerOp<
            "Equal",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Equal layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[Bool8]>:$output
    );
}

def IE_GreaterOp :
        IE_LayerOp<
            "Greater",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Greater layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[Bool8]>:$output
    );
}

def IE_GreaterEqualOp :
        IE_LayerOp<
            "GreaterEqual",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine GreaterEqual layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$output
    );
}

def IE_LessOp :
        IE_LayerOp<
            "Less",
            [
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "InferenceEngine Less layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[Bool8]>:$output
    );
}

def IE_LessEqualOp :
        IE_LayerOp<
            "LessEqual",
            [
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "InferenceEngine LessEqual layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[Bool8]>:$output
    );
}

def IE_NotEqualOp :
        IE_LayerOp<
            "NotEqual",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine NotEqual layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[Bool8]>:$output
    );
}

#endif
