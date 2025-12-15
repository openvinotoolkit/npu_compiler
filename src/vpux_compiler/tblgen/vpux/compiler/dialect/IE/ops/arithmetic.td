//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_ARITHMETIC
#define VPUX_COMPILER_DIALECT_IE_OPS_ARITHMETIC

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_AbsOp :
        IE_LayerOp<
            "Abs",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Abs layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_AcosOp :
        IE_LayerOp<
            "Acos",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Acos layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_AcoshOp :
        IE_LayerOp<
            "Acosh",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Acosh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_AsinOp :
        IE_LayerOp<
            "Asin",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Asin layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_AsinhOp :
        IE_LayerOp<
            "Asinh",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Asinh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_AtanOp :
        IE_LayerOp<
            "Atan",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Atan layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_AtanhOp :
        IE_LayerOp<
            "Atanh",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Atanh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_CeilingOp :
        IE_LayerOp<
            "Ceiling",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Ceiling layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_ClampOp :
        IE_LayerOp<
            "Clamp",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Clamp layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        F64Attr:$min,
        F64Attr:$max
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def IE_CosOp :
        IE_LayerOp<
            "Cos",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Cos layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_CoshOp :
        IE_LayerOp<
            "Cosh",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Cosh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_CumSumOp :
        IE_LayerOp<
            "CumSum"
        > {
    let summary = "InferenceEngine CumSum layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axis,

        OptionalAttr<IntAttr>:$axis_value,
        UnitAttr:$exclusive,
        UnitAttr:$reverse
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def IE_ErfOp :
        IE_LayerOp<
            "Erf",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Erf layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_ExpOp :
        IE_LayerOp<
            "Exp",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Exp layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_FloorOp :
        IE_LayerOp<
            "Floor",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Floor layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_InverseOp :
        IE_LayerOp<
            "Inverse"
        > {
    let summary = "InferenceEngine Inverse layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,

        UnitAttr:$adjoint
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_LogOp :
        IE_LayerOp<
            "Log",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Log layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_NegativeOp :
        IE_LayerOp<
            "Negative",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Negative layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, SI32, SI64]>:$output
    );

    let hasCanonicalizer = 1;
}

def IE_RoundOp :
        IE_LayerOp<
            "Round",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Round layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,

        IE_RoundModeAttr:$mode
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_SignOp :
        IE_LayerOp<
            "Sign",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sign layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_SinOp :
        IE_LayerOp<
            "Sin",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sin layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_SinhOp :
        IE_LayerOp<
            "Sinh",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sinh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_SqrtOp :
        IE_LayerOp<
            "Sqrt",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sqrt layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_TanOp :
        IE_LayerOp<
            "Tan",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Tan layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_TanhOp :
        IE_LayerOp<
            "Tanh",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Tanh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

#endif
