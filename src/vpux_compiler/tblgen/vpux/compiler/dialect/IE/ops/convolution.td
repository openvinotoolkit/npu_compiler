//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_CONV
#define VPUX_COMPILER_DIALECT_IE_OPS_CONV

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_ConvolutionOp :
        IE_LayerOp<
            "Convolution",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                DeclareOpInterfaceMethods<StaticShapeOpInterface>
            ]
        > {
    let summary = "InferenceEngine Convolution layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32]>:$input,
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32]>:$filter,
        Optional<RankedTensorOf<[F16, F32, F64, SI32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<F32Attr>:$static_scale, // constant applied to the weights table
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32]>:$output
    );

    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

def IE_ConvolutionBackpropDataOp :
        IE_LayerOp<
            "ConvolutionBackpropData"
        > {
    let summary = "InferenceEngine ConvolutionBackpropData layer";

    let description = [{
        Represents a transposed convolution, which consumes an input and filter tensor
        and generates an output larger than the input.

        Operands:
        - `input`: the input tensor of the operation; 4D layout [N, C_IN, Y, X]
        - `filter`: the convolutional kernel tensor; expected layout [C_IN, C_OUT, KY, KX]
        - (optional) `output_shape`: specifies the spatial shape of the output;
          expected values `[Y, X]`

        Attributes:
        - `strides`: represents the distance in pixels to slide the filter on the output
          tensor; expected values `[SY, SX]`
        - `pads_begin`: represents the number of pixels to remove from the beginning of
          each axis in the output; expected values `[PAD_TOP, PAD_LEFT]`
        - `pads_end`: represents the number of pixels to remove from the end of each axis
          in the output; expected values `[PAD_BOTTOM, PAD_RIGHT]`
        - `dilations`: has the same definition as dilations for a regular Convolution but
          applied in the backward way, for the output tensor; expected values `[DY, DX]`
        - `spatial_output_padding`: adds additional amount of paddings per each spatial axis in
          the output tensor; expected values `[PY; PX]`

        Results:
        - `output`: the output tensor of the operation; 4D layout [N, C_OUT, Y, X]
    }];

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$filter,
        Optional<1DTensorOf<[AnyInteger]>>:$output_shape,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$spatial_output_padding
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_DeformableConvolutionOp :
        IE_LayerOp<
            "DeformableConvolution"
        > {
    let summary = "InferenceEngine DeformableConvolution layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        RankedTensorOf<[F16, F32, F64]>:$offset,
        RankedTensorOf<[F16, F32, F64]>:$kernel,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$mask,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,

        IntAttr:$group,
        IntAttr:$deformable_group,
        UnitAttr:$bilinear_interpolate_pad
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_FullyConnectedOp :
        IE_LayerOp<
            "FullyConnected",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                DeclareOpInterfaceMethods<StaticShapeOpInterface>
            ]
        > {
    let summary = "InferenceEngine FullyConnected layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32]>:$input,
        RankedTensorOf<[F16, F32, F64, SI32]>:$weights,
        Optional<RankedTensorOf<[F16, F32, F64, SI32]>>:$bias
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, SI32]>:$output
    );

    let hasCanonicalizer = 1;
}

def IE_GroupConvolutionOp :
        IE_LayerOp<
            "GroupConvolution"
        > {
    let summary = "InferenceEngine GroupConvolution layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32]>:$input,
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32]>:$filter,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        OptionalAttr<IntAttr>:$groups,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32]>:$output
    );

    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

def IE_GroupConvolutionBackpropDataOp:
        IE_LayerOp<
            "GroupConvolutionBackpropData"
        > {
    let summary = "InferenceEngine GroupConvolutionBackpropData layer";

    let description = [{
        Represents a grouped transposed convolution, which consumes an input and filter tensor
        and generates an output larger than the input.

        Compared to the ConvolutionBackpropData operation, the input and filter tensors are
        split into multiple groups whose results are computed independently and then concatenated.
        The number of groups can be deduced from the shape of the filter.

        Operands:
        - `input`: the input tensor of the operation; 4D layout [N, GROUPS * C_IN, Y, X]
        - `filter`: the convolutional kernel tensor; expected layout [GROUPS, C_IN, C_OUT, KY, KX]
        - (optional) `output_shape`: specifies the spatial shape of the output;
          expected values `[Y, X]`

        Attributes:
        - `strides`: represents the distance in pixels to slide the filter on the output
          tensor; expected values `[SY, SX]`
        - `pads_begin`: represents the number of pixels to remove from the beginning of
          each axis in the output; expected values `[PAD_TOP, PAD_LEFT]`
        - `pads_end`: represents the number of pixels to remove from the end of each axis
          in the output; expected values `[PAD_BOTTOM, PAD_RIGHT]`
        - `dilations`: has the same definition as dilations for a regular Convolution but
          applied in the backward way, for the output tensor; expected values `[DY, DX]`
        - `spatial_output_padding`: adds additional amount of paddings per each spatial axis in
          the output tensor; expected values `[PAD_Y; PAD_X]`

        Results:
        - `output`: the output tensor of the operation; 4D layout [N, GROUPS * C_OUT, Y, X]
    }];

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$filter,
        Optional<1DTensorOf<[AnyInteger]>>:$output_shape,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$spatial_output_padding
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_GroupTransposedConvolutionOp :
        IE_LayerOp<
            "GroupTransposedConvolution"
        > {
    let summary = "GroupTransposedConvolution layer";

    let description = [{
        Represents a grouped transposed convolution, which consumes an input and filter tensor
        and generates an output larger than the input.

        Compared to the TransposedConvolution operation, the input and filter tensors are
        split into multiple groups whose results are computed independently and then concatenated.
        The number of groups can be deduced from the shape of the filter.

        Additionally, compared to the GroupConvolutionBackpropData operation, the filter expects
        the data in the [GROUPS, C_OUT, C_IN, KY, KX] format, similar to forward group convolutions.

        Operands:
        - `input`: the input tensor of the operation; 4D layout [N, GROUPS * C_IN, Y, X]
        - `filter`: the convolutional kernel tensor; expected layout [GROUPS, C_OUT, C_IN, KY, KX]
        - (optional) `output_shape`: specifies the spatial shape of the output;
          expected values `[Y, X]`

        Attributes:
        - `strides`: represents the distance in pixels to slide the filter on the output
          tensor; expected values `[SY, SX]`
        - `pads_begin`: represents the number of pixels to remove from the beginning of
          each axis in the output; expected values `[PAD_TOP, PAD_LEFT]`
        - `pads_end`: represents the number of pixels to remove from the end of each axis
          in the output; expected values `[PAD_BOTTOM, PAD_RIGHT]`
        - `dilations`: has the same definition as dilations for a regular Convolution but
          applied in the backward way, for the output tensor; expected values `[DY, DX]`
        - `spatial_output_padding`: adds additional amount of paddings per each spatial axis in
          the output tensor; expected values `[PAD_Y; PAD_X]`

        Results:
        - `output`: the output tensor of the operation; 4D layout [N, GROUPS * C_OUT, Y, X]
    }];

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$filter,
        Optional<1DTensorOf<[AnyInteger]>>:$output_shape,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$spatial_output_padding,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_MatMulOp :
        IE_LayerOp<
            "MatMul",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                DeclareOpInterfaceMethods<StaticShapeOpInterface>
            ]
        > {
    let summary = "InferenceEngine MatMul layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, quant_QuantizedType]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI32, quant_QuantizedType]>:$input2,

        UnitAttr:$transpose_a,
        UnitAttr:$transpose_b,

        OptionalAttr<IE_PostOpAttr>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, SI32, quant_QuantizedType]>:$output
    );

    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION];
}

def IE_TransposedConvolutionOp :
        IE_LayerOp<
            "TransposedConvolution",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "TransposedConvolution layer";

    let description = [{
        Represents a transposed convolution, which consumes an input and filter tensor
        and generates an output larger than the input.

        Compared to the ConvolutionBackpropData operation, the filter expects the data
        in the [C_OUT, C_IN, KY, KX] format, similar to forward convolutions.

        Operands:
        - `input`: the input tensor of the operation; 4D layout [N, C_IN, Y, X]
        - `filter`: the convolutional kernel tensor; expected layout [C_OUT, C_IN, KY, KX]
        - (optional) `output_shape`: specifies the spatial shape of the output;
          expected values `[Y, X]`
        - (optional) `bias`: the bias tensor; 4D layout [N, C, H, W]

        Attributes:
        - `strides`: represents the distance in pixels to slide the filter on the output
          tensor; expected values `[SY, SX]`
        - `pads_begin`: represents the number of pixels to remove from the beginning of
          each axis in the output; expected values `[PAD_TOP, PAD_LEFT]`
        - `pads_end`: represents the number of pixels to remove from the end of each axis
          in the output; expected values `[PAD_BOTTOM, PAD_RIGHT]`
        - `dilations`: has the same definition as dilations for a regular Convolution but
          applied in the backward way, for the output tensor; expected values `[DY, DX]`
        - `spatial_output_padding`: adds additional amount of paddings per each spatial axis in
          the output tensor; expected values `[PY; PX]`

        Results:
        - `output`: the output tensor of the operation; 4D layout [N, C_OUT, Y, X]
    }];

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$filter,
        Optional<1DTensorOf<[AnyInteger]>>:$output_shape,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$spatial_output_padding,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

#endif
