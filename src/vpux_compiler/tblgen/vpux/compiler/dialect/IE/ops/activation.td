//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_ACTIVATION
#define VPUX_COMPILER_DIALECT_IE_OPS_ACTIVATION

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_EluOp :
        IE_LayerOp<
            "Elu",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Elu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,

        F64Attr:$x
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_GeluOp :
        IE_LayerOp<
            "Gelu",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Gelu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_HardSigmoidOp :
        IE_LayerOp<
            "HardSigmoid",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine HardSigmoid layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$alpha,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$beta,

        OptionalAttr<F64Attr>:$alpha_value,
        OptionalAttr<F64Attr>:$beta_value
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def IE_HSigmoidOp :
        IE_LayerOp<
            "HSigmoid",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine HSigmoid layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}


def IE_HSwishOp :
        IE_LayerOp<
            "HSwish",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine HSwish layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_LeakyReluOp :
        IE_LayerOp<
            "LeakyRelu",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine LeakyRelu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32, SI8, UI8]>:$input,

        F64Attr:$negative_slope
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32, SI8, UI8]>:$output
    );
    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

def IE_LogSoftmaxOp :
        IE_LayerOp<
            "LogSoftmax"
        > {
    let summary = "InferenceEngine LogSoftmax layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,

        IntAttr:$axisInd
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );

    let hasFolder = 1;
}

def IE_MishOp :
        IE_LayerOp<
            "Mish",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Mish layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_PReluOp :
        IE_LayerOp<
            "PRelu",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine PRelu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        RankedTensorOf<[F16, F32, F64]>:$negative_slope
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );

    let hasCanonicalizer = 1;
}

def IE_ReLUOp :
        IE_LayerOp<
            "ReLU",
            [
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                DeclareOpInterfaceMethods<StaticShapeOpInterface>
            ]
        > {
    let summary = "InferenceEngine ReLU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}


def IE_SeluOp :
        IE_LayerOp<
            "Selu",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine Selu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$data,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$alpha,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$lambda,

        OptionalAttr<F64Attr>:$alphaValue,
        OptionalAttr<F64Attr>:$lambdaValue
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );

    let hasCanonicalizer = 1;
}

def IE_SigmoidOp :
        IE_LayerOp<
            "Sigmoid",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sigmoid layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_SoftMaxOp :
        IE_LayerOp<
            "SoftMax",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "InferenceEngine SoftMax layer";

    let description = [{
        This operation is intended to run Softmax filter with extra capability
        to padding with 0 part of output on axis attribute direction.
        - padSize - attribute that allow to specify how much from axis dimension
        to not be taken in consideration from input and to be wrote with 0 on output.
        This extra functionality was added in order to cover,
        with minimum cost, align required from neighbor filters.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,

        IntAttr:$axisInd,
        OptionalAttr<IntAttr>:$padSize
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

def IE_SoftPlusOp :
        IE_LayerOp<
            "SoftPlus",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine SoftPlus layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_SwishOp :
        IE_LayerOp<
            "Swish",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Swish layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$beta,

        OptionalAttr<F64Attr>:$beta_value
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );

    let hasCanonicalizer = 1;
}

#endif
