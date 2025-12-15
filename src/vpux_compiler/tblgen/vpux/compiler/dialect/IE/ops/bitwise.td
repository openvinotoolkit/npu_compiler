//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_BITWISE
#define VPUX_COMPILER_DIALECT_IE_OPS_BITWISE

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_BitwiseAndOp :
        IE_LayerOp<
            "BitwiseAnd",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine BitwiseAndOp layer";

    let arguments = (ins
        RankedTensorOf<[AnyInteger]>:$input1,
        RankedTensorOf<[AnyInteger]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[AnyInteger]>:$output
    );
}

def IE_BitwiseNotOp :
        IE_LayerOp<
            "BitwiseNot",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine BitwiseNotOp layer";

    let arguments = (ins
        RankedTensorOf<[AnyInteger]>:$input1
    );

    let results = (outs
        RankedTensorOf<[AnyInteger]>:$output
    );
}

def IE_BitwiseOrOp :
        IE_LayerOp<
            "BitwiseOr",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine BitwiseOrOp layer";

    let arguments = (ins
        RankedTensorOf<[AnyInteger]>:$input1,
        RankedTensorOf<[AnyInteger]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[AnyInteger]>:$output
    );
}

def IE_BitwiseXorOp :
        IE_LayerOp<
            "BitwiseXor",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine BitwiseXorOp layer";

    let arguments = (ins
        RankedTensorOf<[AnyInteger]>:$input1,
        RankedTensorOf<[AnyInteger]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[AnyInteger]>:$output
    );
}

#endif
