//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_REDUCE
#define VPUX_COMPILER_DIALECT_IE_OPS_REDUCE

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_ReduceL1Op :
        IE_LayerOp<
            "ReduceL1",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>
            ]
        > {
    let summary = "InferenceEngine ReduceL1 layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

def IE_ReduceL2Op :
        IE_LayerOp<
            "ReduceL2",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>
            ]
        > {
    let summary = "InferenceEngine ReduceL2 layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

def IE_ReduceLogicalAndOp :
        IE_LayerOp<
            "ReduceLogicalAnd"
        > {
    let summary = "InferenceEngine  ReduceLogicalAnd layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

def IE_ReduceLogicalOrOp :
        IE_LayerOp<
            "ReduceLogicalOr"
        > {
    let summary = "InferenceEngine ReduceLogicalOr layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

def IE_ReduceMaxOp :
        IE_LayerOp<
            "ReduceMax",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>
            ]
        > {
    let summary = "InferenceEngine ReduceMax layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

def IE_ReduceMeanOp :
        IE_LayerOp<
            "ReduceMean"
        > {
    let summary = "InferenceEngine ReduceMean Layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let builders = [
        OpBuilder<(ins
            "mlir::Type":$outputType,
            "mlir::Value":$input,
            "mlir::ArrayAttr":$axes_value,
            "mlir::UnitAttr":$keep_dims
        )>,
        OpBuilder<(ins
            "mlir::Value":$input,
            "mlir::ArrayAttr":$axes_value,
            "mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;
    let hasVerifier = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

def IE_ReduceMinOp :
        IE_LayerOp<
            "ReduceMin",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>
            ]
        > {
    let summary = "InferenceEngine ReduceMin layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

def IE_ReduceProdOp :
        IE_LayerOp<
            "ReduceProd"
        > {
    let summary = "InferenceEngine ReduceProd layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

def IE_ReduceSumOp :
        IE_LayerOp<
            "ReduceSum",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>
            ]
        > {
    let summary = "InferenceEngine ReduceSum layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let builders = [
        OpBuilder<(ins
            "mlir::Type":$outputType,
            "mlir::Value":$input,
            "mlir::ArrayAttr":$axes_value,
            "mlir::UnitAttr":$keep_dims
        )>,
        OpBuilder<(ins
            "mlir::Value":$input,
            "mlir::ArrayAttr":$axes_value,
            "mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;
    let hasVerifier = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

def IE_ReduceSquareOp :
        IE_LayerOp<
            "ReduceSquare"
        > {
    let summary = "InferenceEngine ReduceSquare layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        OptionalAttr<F64Attr>:$epsilon,
        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
        OptionalAttr<I64Attr>:$scale
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

#endif
