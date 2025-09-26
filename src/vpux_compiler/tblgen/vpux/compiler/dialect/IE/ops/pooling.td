//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_POOLING
#define VPUX_COMPILER_DIALECT_IE_OPS_POOLING

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_AdaptiveAvgPoolOp :
        IE_LayerOp<
            "AdaptiveAvgPool"
        > {
    let summary = "InferenceEngine AdaptiveAvgPool layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        1DTensorOf<[SI32, SI64]>:$pooled_spatial_shape
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_AdaptiveMaxPoolOp :
        IE_LayerOp<
            "AdaptiveMaxPool"
        > {
    let summary = "InferenceEngine AdaptiveMaxPool layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        1DTensorOf<[SI32, SI64]>:$pooled_spatial_shape,
        TypeAttr:$index_element_type
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output,
        RankedTensorOf<[SI32, SI64]>:$output_index

    );
}

def IE_AvgPoolOp :
        IE_LayerOp<
            "AvgPool"
        > {
    let summary = "InferenceEngine AvgPool layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,
        UnitAttr:$exclude_pads,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<F32Attr>:$static_scale,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$output
    );

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

def IE_AvgPool16Op :
        IE_LayerOp<
            "AvgPool16"
        > {
    let summary = "InferenceEngine AvgPool16 layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,
        UnitAttr:$exclude_pads,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<F32Attr>:$static_scale,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$output
    );
}

def IE_MaxPoolOp :
        IE_LayerOp<
            "MaxPool",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                DeclareOpInterfaceMethods<StaticShapeOpInterface>
            ]
        > {
    let summary = "InferenceEngine MaxPool layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$output
    );

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

def IE_MaxPool8Op :
        IE_LayerOp<
            "MaxPool8"
        > {
    let summary = "InferenceEngine MaxPool8 layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,
        TypeAttr:$index_element_type,

        IntAttr:$axis
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$output,
        RankedTensorOf<[SI32, SI64]>:$output_index
    );

}

#endif
