//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_CONV
#define VPUX_COMPILER_DIALECT_IE_OPS_CONV

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/core/types.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_ConvolutionOp :
        IE_LayerOp<
            "Convolution",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<IE_ShapeInfoOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
                DeclareOpInterfaceMethods<StaticShapeOpInterface>
            ]
        > {
    let summary = "InferenceEngine Convolution layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32]>:$input,
        RankedTensorOf<[AnyInteger, F16, F32, F64, vpux_QuantileType, quant_QuantizedType]>:$filter,
        Optional<RankedTensorOf<[F16, F32, F64, SI32]>>:$bias,
        Optional<RankedTensorOf<[F16, F32, F8E8M0FNU, F8E5M2, F8E4M3FN]>>:$scale,
        Optional<RankedTensorOf<[AnyInteger, F8E4M3FN, F8E5M2, F4E2M1FN]>>:$zero_points,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,

        OptionalAttr<IE_PostOpAttrInterface>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<F32Attr>:$static_scale, // constant applied to the weights table
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32]>:$output
    );

    let assemblyFormat = [{
        `(`
            $input `,`
            $filter
            (`,` $bias^)?
            (`,` $scale^)?
            (`,` $zero_points^)?
        `)`
        attr-dict `:` type(operands) `->` type(results)
    }];

    let hasCanonicalizer = 1;
    let hasVerifier = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$filter,
            "::mlir::ArrayAttr":$strides,
            "::mlir::ArrayAttr":$pads_begin,
            "::mlir::ArrayAttr":$pads_end,
            "::mlir::ArrayAttr":$dilations
        )>,
        OpBuilder<(ins
            "mlir::Type":$resultType,
            "::mlir::Value":$input,
            "::mlir::Value":$filter,
            "::mlir::ArrayAttr":$strides,
            "::mlir::ArrayAttr":$pads_begin,
            "::mlir::ArrayAttr":$pads_end,
            "::mlir::ArrayAttr":$dilations
        )>,
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$filter,
            "::mlir::Value":$bias,
            "::mlir::Value":$scale,
            "::mlir::Value":$zero_points,
            "::mlir::ArrayAttr":$strides,
            "::mlir::ArrayAttr":$pads_begin,
            "::mlir::ArrayAttr":$pads_end,
            "::mlir::ArrayAttr":$dilations
        )>,
        OpBuilder<(ins
            "mlir::Type":$resultType,
            "::mlir::Value":$input,
            "::mlir::Value":$filter,
            "::mlir::Value":$bias,
            "::mlir::Value":$scale,
            "::mlir::Value":$zero_points,
            "::mlir::ArrayAttr":$strides,
            "::mlir::ArrayAttr":$pads_begin,
            "::mlir::ArrayAttr":$pads_end,
            "::mlir::ArrayAttr":$dilations
        )>,
    ];
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
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
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

    // Needed for LayerWithPostOpInterface, however
    // FullyConnected will never reach the stage of the pipeline
    // where fusing post ops actually happens.
    // Should be removed once unification with MatMul op happens.
    // Ticket: E#234211
    let extraClassDeclaration = [{
        vpux::IE::PostOpAttr getPostOpAttr() const {
            return nullptr;
        }

        void setPostOpAttr(vpux::IE::PostOpAttr) const {
            return;
        }

        void removePostOpAttr() const {
            return;
        }

        void setClampAttr(mlir::DictionaryAttr) const {
            return;
        }

        mlir::DictionaryAttr getClampAttr() const {
            return nullptr;
        }

        std::optional<mlir::DictionaryAttr> getClamp() const {
            return std::nullopt;
        }

        void removeClampAttr() const {
            return;
        }
    }] # baseExtraClassDeclaration;
}

def IE_GroupConvolutionOp :
        IE_LayerOp<
            "GroupConvolution",
            [
                DeclareOpInterfaceMethods<IE_ShapeInfoOpInterface>
            ]
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

        OptionalAttr<IE_PostOpAttrInterface>:$post_op,
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

        OptionalAttr<IE_PostOpAttrInterface>:$post_op,
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
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
                DeclareOpInterfaceMethods<StaticShapeOpInterface>
            ]
        > {
    let summary = "InferenceEngine MatMul layer";

    let description = [{
        Computes `output = post_op(clamp(op(input1) x op(W)))`, where `op` is a transpose of the two innermost
        axes if `transpose_a` / `transpose_b` is set, and the leading axes are batch axes.

        By default `W = input2`. The optional `scale` and `zero_points` operands turn `input2` into quantized
        weights that are dequantized by the operation itself, which is equivalent to feeding the MatMul with the
        result of IE.DynamicDequantize:

            W = (input2 - zero_points) * scale

        Each of the two is optional on its own: without `scale` the multiplication is skipped (as if it were 1),
        without `zero_points` the subtraction is skipped (as if it were 0).

        - The dequantization is applied to `input2` as it is stored, i.e. before `transpose_b`. `scale` and
          `zero_points` hold one value per output channel of the MatMul (the last axis of the result, which is
          the second to last axis of `input2` if `transpose_b` is set and the last one otherwise) or a single
          value for all of them. The value of an output channel is shared by every batch / group of `input2`.
        - When both are provided, `scale` and `zero_points` must have the same shape. Canonical form is `[OC, 1, ..., 1]` for one value
          per output channel and `[1, ..., 1]` for a single value, also for whichever of the two is provided alone (for example, `[OC, 1]` for rank-2 and
          `[OC, 1, 1, 1]` for rank-4). Elements are indexed by output channel regardless of encoding; this is not
          a NumPy broadcast against `input2`.
        - `post_op` and `clamp` act on the dequantized result, after the accumulation.
        - `bias` is optional and independent of `scale` / `zero_points`.
        - The optional operands are positional and always in the order `bias`, `scale`, `zero_points`. As for
          IE.Convolution, an absent operand is skipped with an empty slot in the custom assembly format:
          `IE.MatMul(%in1, %in2, , %scale)` has a `scale` but no `bias`.

        The verifier checks the element type of `zero_points`, that `scale` and `zero_points` have the same shape
        when both are provided and, where dimensions are static, that this shape is `[OC, 1, ..., 1]` or
        `[1, ..., 1]`, and that `bias` is 4D. It does not check that the values are constant. Producers must
        respect the rules above, and every consumer that cannot honour them has to leave the operation untouched
        rather than ignore the operands: e.g. conversion to FullyConnected / Convolution and the generic
        IE -> VPU rewrite do not match a MatMul that carries `scale`, `zero_points` or `bias`.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, quant_QuantizedType]>:$input1,
        RankedTensorOf<[F16, F32, F64, SI32, quant_QuantizedType]>:$input2,
        Optional<RankedTensorOf<[F16, F32, F64, SI32]>>:$bias,
        Optional<RankedTensorOf<[F16, F32, F8E8M0FNU, F8E5M2, F8E4M3FN]>>:$scale,
        Optional<RankedTensorOf<[AnyInteger, F8E4M3FN, F8E5M2, F4E2M1FN]>>:$zero_points,

        UnitAttr:$transpose_a,
        UnitAttr:$transpose_b,

        OptionalAttr<IE_PostOpAttrInterface>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, SI32, quant_QuantizedType]>:$output
    );

    let assemblyFormat = [{
        `(`
            $input1 `,`
            $input2
            (`,` $bias^)?
            (`,` $scale^)?
            (`,` $zero_points^)?
        `)`
        attr-dict `:` type(operands) `->` type(results)
    }];

    let hasCanonicalizer = 1;
    let hasVerifier = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "bool":$transposeA,
            "bool":$transposeB
        )>,
        OpBuilder<(ins
            "::mlir::Type":$output,
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "bool":$transposeA,
            "bool":$transposeB
        )>
    ];
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

        OptionalAttr<IE_PostOpAttrInterface>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

#endif
