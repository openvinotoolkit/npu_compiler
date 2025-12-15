//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_DATA_TYPE
#define VPUX_COMPILER_DIALECT_IE_OPS_DATA_TYPE

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/core/types.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_ConvertOp :
        IE_LayerOp<
            "Convert",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface, ["shouldJITCompile"]>,
                DeclareOpInterfaceMethods<CastOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Convert layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let hasVerifier = 1;
}

def IE_ConvertLikeOp :
        IE_LayerOp<
            "ConvertLike"
        > {
    let summary = "InferenceEngine ConvertLike layer";

    let description = [{
        This operator returns a tensor with the same shape as the data input
        and same type as the like input
    }];

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$like
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

def IE_DequantizeOp :
        IE_LayerOp<"Dequantize",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Dequantize layer";

    let arguments = (ins
        RankedTensorOf<[quant_QuantizedType]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

def IE_DynamicDequantizeOp :
        IE_LayerOp<"DynamicDequantize",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Dynamic Dequantize layer";

    let description = [{
        The operation dequantizes the input from like i8 i4 or i2 to f16 or f32.
        The input data type should be signed or unsigned, we can not support
        signless because we can not determine the data range.
    }];

    let arguments = (ins
        RankedTensorOf<[quant_QuantizedType]>:$input,
        RankedTensorOf<[F16, F32]>:$scale,
        Optional<RankedTensorOf<[I8, I4, I2, U2]>>:$zp,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasVerifier = 1;
}

def IE_DynamicFakeQuantizeOp :
        IE_LayerOp<
            "DynamicFakeQuantize",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine DynamicFakeQuantize layer";

    let description = [{
        The operation works in two modes:
        * integral quantization: specified by the 'levels' attribute
        * floating-point quantization: specified by the 'low_fp_type' attribute, [f8E4M3FN | f8E5M2]

        Only one of these attributes should be provided.

        Scale and zp attributes provided for weights quantization
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        // Used for dyn quant on weights path
        RankedTensorOf<[F16, F32]>:$scale,
        RankedTensorOf<[I8, I4]>:$zp,

        OptionalAttr<IntAttr>:$levels,
        OptionalAttr<TypeAttr>:$low_fp_type,
        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasVerifier = 1;
}

def IE_DynamicQuantizeOp :
        IE_LayerOp<
            "DynamicQuantize",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine Dynamic-Quantize layer";

    let arguments = (ins
        RankedTensorOf<[F32]>:$input,
        Optional<RankedTensorOf<[F32]>>:$min,
        Optional<RankedTensorOf<[F32]>>:$max
    );

    let results = (outs
        RankedTensorOf<[UI8]>:$output,
        RankedTensorOf<[F32]>:$scale,
        RankedTensorOf<[UI8]>:$zero_point
    );

    let hasVerifier = 1;
}

def IE_FakeConvertOp :
        IE_LayerOp<
            "FakeConvert",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine FakeConvert layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$scale,
        Optional<RankedTensorOf<[F16, F32]>>:$shift,

        TypeAttr:$dst_type
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasVerifier = 1;
}


def IE_FakeQuantizeOp :
        IE_LayerOp<
            "FakeQuantize",
            [
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                DeclareOpInterfaceMethods<StaticShapeOpInterface>
            ]
        > {
    let summary = "InferenceEngine FakeQuantize layer";

    let description = [{
        The operation works in two modes:
         * integral quantization: specified by the 'levels' attribute
         * floating-point quantization: specified by the 'low_fp_type' attribute, [f8E4M3FN | f8E5M2]

        Only one of these attributes should be provided.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$input_low,
        RankedTensorOf<[F16, F32]>:$input_high,
        RankedTensorOf<[F16, F32]>:$output_low,
        RankedTensorOf<[F16, F32]>:$output_high,

        OptionalAttr<IntAttr>:$levels,
        OptionalAttr<TypeAttr>:$low_fp_type,
        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
    let hasVerifier = 1;
}

def IE_QuantizeOp :
        IE_LayerOp<"Quantize",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Quantize layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[quant_QuantizedType]>:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def IE_QuantizeCastOp :
        IE_LayerOp<
            "QuantizeCast",
            [
                IE_ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine Quantize Cast layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
    let hasVerifier = 1;
}


#endif
