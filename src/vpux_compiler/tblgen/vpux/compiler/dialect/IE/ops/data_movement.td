//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_DATA_MOVEMENT
#define VPUX_COMPILER_DIALECT_IE_OPS_DATA_MOVEMENT

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_BatchToSpace :
        IE_LayerOp<
            "BatchToSpace",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine BatchToSpace layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<1DTensorOf<[AnyInteger]>>:$block_shape,
        Optional<1DTensorOf<[AnyInteger]>>:$crops_begin,
        Optional<1DTensorOf<[AnyInteger]>>:$crops_end,

        OptionalAttr<I64ArrayAttr>:$block_shape_value,
        OptionalAttr<I64ArrayAttr>:$crops_begin_value,
        OptionalAttr<I64ArrayAttr>:$crops_end_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
}

def IE_BroadcastOp :
        IE_LayerOp<
            "Broadcast"
        > {
    let summary = "InferenceEngine Broadcast layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        1DTensorOf<[AnyInteger]>:$target_shape,
        Optional<1DTensorOf<[AnyInteger]>>:$axes_mapping,

        OptionalAttr<IE_BroadcastTypeAttr>:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

def IE_ConcatOp :
        IE_LayerOp<
            "Concat",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "InferenceEngine Concat layer";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,

        OptionalAttr<IE_ConcatAttr>:$per_axis,
        OptionalAttr<I64ArrayOfArraysAttr>:$static_offsets
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "vpux::IE::ConcatAttr":$per_axis)
        >,
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "mlir::IntegerAttr":$axis,
                 CArg<"mlir::IntegerAttr", "{}">:$offset, CArg<"mlir::IntegerAttr", "{}">:$stride)
        >,
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "int64_t":$axis, CArg<"int64_t", "0">:$offset, CArg<"int64_t", "1">:$stride)
        >,
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "vpux::Dim":$axis, CArg<"int64_t", "0">:$offset, CArg<"int64_t", "1">:$stride)
        >,

        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "mlir::ArrayAttr":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::ArrayRef<vpux::Shape>":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::ArrayRef<vpux::ShapeRef>":$static_offsets)
        >,
    ];

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

def IE_DepthToSpaceOp :
        IE_LayerOp<
            "DepthToSpace"
        > {
    let summary = "InferenceEngine DepthToSpace layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        IntAttr:$block_size,
        IE_DepthToSpaceModeAttr:$mode,
        OptionalAttr<IE_ChannelPaddingAttr>:$padded_channels
    );

    let builders = [
    OpBuilder<(ins
            "mlir::Value":$input, "int64_t":$block_size, "IE::DepthToSpaceMode":$mode
        )>,
      OpBuilder<(ins
            "mlir::Value":$input, "mlir::IntegerAttr":$block_size, "IE::DepthToSpaceModeAttr":$mode
        )>,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::Value":$input, "mlir::IntegerAttr":$block_size, "IE::DepthToSpaceModeAttr":$mode)
        >,
    ];

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasVerifier = 1;
}

def IE_DynamicBroadcastOp :
        IE_LayerOp<
            "DynamicBroadcast"
        > {
    let summary = "InferenceEngine Broadcast layer for dynamic shapes";

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$target_shape,
        Optional<1DTensorOf<[AnyInteger]>>:$axes_mapping,

        OptionalAttr<IE_BroadcastTypeAttr>:$mode,

        I64ArrayAttr:$output_shape,
        I64ArrayAttr:$output_bounds
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_DynamicExpandOp :
        IE_LayerOp<
            "DynamicExpand"
        > {
    let summary = "DynamicExpand operation";

    let description = [{
        DynamicExpand operation isn't related to any InferenceEngine layer and is designed to take an input
        tensor with dynamic shapes and perform zero-padding to match the dimensions of an upper bound.
        This operation is specifically intended for converting a subgraph of a dynamic network into a static one
        Example:
        Input tensor<1x1x?x?xf16, {bounds = [1, 1, 5, 5]>
        Input dynamic shape: 1x1x3x3
        1, 1, 1
        1, 1, 1
        1, 1, 1
        In memory: 1, 1, 1, 1, 1, 1, 1, 1, 1
        Output tensor<1x1x5x5xf16>
        1, 1, 1, 0, 0
        1, 1, 1, 0, 0
        1, 1, 1, 0, 0
        0, 0, 0, 0, 0
        0, 0, 0, 0, 0
        In memory: 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    }];

    let arguments = (ins
        AnyRankedTensor:$input
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

def IE_DynamicTileOp :
        IE_LayerOp<
            "DynamicTile"
        > {
    let summary = "InferenceEngine Tile layer with dynamic operands";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$target_shape,

        Optional<RankedTensorOf<[AnyInteger]>>:$repeats,
        OptionalAttr<I64ArrayAttr>:$repeats_values,

        I64ArrayAttr:$output_shape,
        I64ArrayAttr:$output_bounds
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_ExpandOp :
        IE_LayerOp<
            "Expand"
        > {
    let summary = "Expand tensor with uninitialized values";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "std::optional<vpux::ShapeRef>":$pads_begin, "std::optional<vpux::ShapeRef>":$pads_end)
        >
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
    let hasVerifier = 1;
}

def IE_ExpandDilatedOp :
        IE_LayerOp<
            "ExpandDilated"
        > {
    let summary = "Expand tensor with uninitialized values according to dilations";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayAttr:$dilations
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

def IE_ExtractImagePatchesOp :
        IE_LayerOp<
            "ExtractImagePatches"
        > {
    let summary = "InferenceEngine ExtractImagePatches layer";

    let arguments = (ins
        4DTensorOf<[AnyType]>:$data,

        I64ArrayAttr:$sizes,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$rates,
        IE_PadTypeAttr:$autoPad
    );

    let results = (outs
        4DTensorOf<[AnyType]>:$output
    );
}

def IE_GatherOp :
        IE_LayerOp<
            "Gather"
        > {
    let summary = "InferenceEngine Gather layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        Optional<RankedTensorOf<[AnyInteger]>>:$axis,
        OptionalAttr<IntAttr>:$axis_value,
        IntAttr:$batch_dims,

        OptionalAttr<IntAttr>:$indices_rank
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
}

def IE_GatherElementsOp :
        IE_LayerOp<
              "GatherElements"
        > {
    let summary = "InferenceEngine GatherElements layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,

        IntAttr:$axis
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def IE_GatherNDOp :
        IE_LayerOp<
            "GatherND"
        > {
    let summary = "InferenceEngine GatherND layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        IntAttr:$batch_dims,

        OptionalAttr<I64ArrayAttr>:$original_shape
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$indices,
            "::mlir::IntegerAttr":$batch_dims
        )>
    ];

    let hasVerifier = 1;
}

def IE_GatherTreeOp :
        IE_LayerOp<
            "GatherTree"
        > {
    let summary = "InferenceEngine GatherTree layer";

    let arguments = (ins
        AnyRankedTensor:$stepIds,
        AnyRankedTensor:$parentIds,
        AnyRankedTensor:$maxSeqLen,
        AnyRankedTensor:$endToken
    );

    let results = (outs
        AnyRankedTensor:$finalIds
    );

    let hasVerifier = 1;
}

def IE_MemPermuteOp :
        IE_LayerOp<
            "MemPermute"
        > {
    let summary = "InferenceEngine MemPermute layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 0;
}

def IE_PadOp :
        IE_LayerOp<
            "Pad",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine Pad layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$pads_begin,
        Optional<RankedTensorOf<[AnyInteger]>>:$pads_end,
        Optional<RankedTensorOf<[AnyInteger, AnyFloat]>>:$pad_value,

        OptionalAttr<I64ArrayAttr>:$pads_begin_attr,
        OptionalAttr<I64ArrayAttr>:$pads_end_attr,
        OptionalAttr<F64Attr>:$pad_value_attr,

        IE_PadModeAttr:$mode,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let assemblyFormat = [{
        `(` $input `)` (`[` $pads_begin^ `,` $pads_end (`,` $pad_value^)? `]`)? attr-dict `:` type(operands) `->` type(results)
    }];
}

def IE_PerAxisTileOp :
        IE_LayerOp<
            "PerAxisTile"
        > {
    let summary = "InferenceEngine per axis Tile layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        IntAttr:$axis,
        IntAttr:$tiles
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_ReorderOp :
        IE_LayerOp<
            "Reorder"
        > {
    let summary = "InferenceEngine Reorder layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        AffineMapAttr:$dstOrder
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 0;
}

def IE_ReverseOp :
        IE_LayerOp<
            "Reverse"
        > {

    let summary = "Reverse operations reverse specified axis in an input tensor.";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<1DTensorOf<[AnyInteger]>>:$axis,

        OptionalAttr<I64ArrayAttr>:$axis_value,

        IE_ReverseModeAttr:$mode
    );

    let results = (outs
    AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
}

def IE_ReverseSequenceOp :
        IE_LayerOp<
            "ReverseSequence"
        > {
    let summary = "Reverse variable length sequence operation";

    let arguments = (ins
        AnyRankedTensor:$data,
        RankedTensorOf<[AnyInteger]>:$seq_length,

        IntAttr:$seq_axis,
        IntAttr:$batch_axis
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def IE_RollOp :
        IE_LayerOp<
            "Roll"
        > {
    let summary = "InferenceEngine Roll layer";

    let arguments = (ins
        AnyRankedTensor:$data,
        RankedTensorOf<[SI32, SI64]>:$shift,
        RankedTensorOf<[SI32, SI64]>:$axes
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_ScatterElementsUpdateOp :
        IE_LayerOp<
            "ScatterElementsUpdate"
        > {
    let summary = "InferenceEngine ScatterElementsUpdate layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        AnyRankedTensor:$updates,
        Optional<RankedTensorOf<[AnyInteger]>>:$axis,
        OptionalAttr<IntAttr>:$axis_value,
        IE_ScatterElementsUpdateReductionTypeAttr:$reduction,
        BoolAttr:$use_init_val
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def IE_ScatterNDUpdateOp :
        IE_LayerOp<
            "ScatterNDUpdate"
        > {
    let summary = "InferenceEngine ScatterNDUpdate layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        AnyRankedTensor:$updates
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_ScatterUpdateOp :
        IE_LayerOp<
            "ScatterUpdate"
        > {
    let summary = "InferenceEngine ScatterUpdate layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        AnyRankedTensor:$updates,
        Optional<AnyRankedTensor>:$axis,
        OptionalAttr<IntAttr>:$axis_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

     let hasCanonicalizer = 1;
}

def IE_ShuffleChannelsOp :
        IE_LayerOp<
            "ShuffleChannels",
            [
                SameOperandsAndResultType,
                SameOperandsAndResultShape
            ]
        > {
    let summary = "InferenceEngine ShuffleChannels layer";

    let arguments = (ins
        4DTensorOf<[F16, F32, F64]>:$input,

        IntAttr:$axis,
        IntAttr:$group
    );

    let results = (outs
        4DTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_SliceOp :
        IE_LayerOp<
            "Slice"
        > {
    let summary = "Extract single slice from tensor";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayAttr:$static_offsets,
        I64ArrayAttr:$static_sizes
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        $input $static_offsets $static_sizes
        attr-dict `:` type($input) `to` type(results)
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "vpux::ShapeRef":$static_offsets, "vpux::ShapeRef":$static_sizes)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "vpux::ArrayRef<int64_t>":$static_offsets, "vpux::ArrayRef<int64_t>":$static_sizes)
        >
    ];

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;

    let extraClassDeclaration = [{
        // Note: to be compatible with existing code
        mlir::Value getSource() { return getInput(); }
    }] # baseExtraClassDeclaration;
}

def IE_SpaceToBatch :
        IE_LayerOp<
            "SpaceToBatch",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine SpaceToBatch layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<1DTensorOf<[AnyInteger]>>:$block_shape,
        Optional<1DTensorOf<[AnyInteger]>>:$pads_begin,
        Optional<1DTensorOf<[AnyInteger]>>:$pads_end,

        OptionalAttr<I64ArrayAttr>:$block_shape_value,
        OptionalAttr<I64ArrayAttr>:$pads_begin_value,
        OptionalAttr<I64ArrayAttr>:$pads_end_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
}

def IE_SpaceToDepthOp :
        IE_LayerOp<
            "SpaceToDepthOp"
        > {
    let summary = "InferenceEngine SpaceToDepthOp layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        DefaultValuedAttr<IntAttr, "1">:$block_size,
        IE_SpaceToDepthModeAttr:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasVerifier = 1;
}

def IE_SplitOp :
        IE_LayerOp<
            "Split"
        > {
    let summary = "InferenceEngine Split layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<AnyRankedTensor>:$axis,

        IntAttr:$num_splits,
        OptionalAttr<IntAttr>:$axis_value
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );

    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;

    let hasVerifier = 1;
}

def IE_StridedSliceOp :
        IE_LayerOp<
            "StridedSlice",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<IE_ElemTypeInfoOpInterface>
            ]
        > {
    let summary = "InferenceEngine StridedSlice layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<1DTensorOf<[AnyInteger]>>:$begins,
        Optional<1DTensorOf<[AnyInteger]>>:$ends,
        Optional<1DTensorOf<[AnyInteger]>>:$strides,

        OptionalAttr<I64ArrayAttr>:$begins_attr,
        OptionalAttr<I64ArrayAttr>:$ends_attr,
        OptionalAttr<I64ArrayAttr>:$strides_attr,

        I64ArrayAttr:$begin_mask,
        I64ArrayAttr:$end_mask,
        I64ArrayAttr:$new_axis_mask,
        I64ArrayAttr:$shrink_axis_mask,
        I64ArrayAttr:$ellipsis_mask
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        bool isSimplified();
    }];

    let hasVerifier = 1;
}

def IE_TileOp :
        IE_LayerOp<
            "Tile"
        > {
    let summary = "InferenceEngine Tile layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$repeats,
        OptionalAttr<I64ArrayAttr>:$repeats_values
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

def IE_TransposeOp :
        IE_LayerOp<
            "Transpose",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                DeclareOpInterfaceMethods<StaticShapeOpInterface, ["requiresStaticShape"]>
            ]
        > {
    let summary = "InferenceEngine Transpose layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$order,

        OptionalAttr<AffineMapAttr>:$order_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 0;
}

def IE_UpsamplingOp :
        IE_LayerOp<
            "Upsampling"
        > {
    let summary = "InferenceEngine Upsampling layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType]>:$input,
        I64ArrayAttr:$upsampling_factor,
        OptionalAttr<IE_UpsamplingPadAttr>:$pad,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, quant_QuantizedType]>:$output
    );
}

def IE_VariadicSplitOp :
        IE_LayerOp<
            "VariadicSplit"
        > {
    let summary = "VariadicSplit operation";

    let description = [{
        This operation is analogous to the OpenVINO VariadicSplit operation. It splits an input tensor along a given axis into multiple tensors according to split lengths.

        'input' is the input tensor to split. It can have an arbitrary fixed shape. Its element type must not be quantized.

        'axis' is the target axis to split along. It can be in the range [-rank('input'); rank('input')-1]. Negative values are counted from the back, with -1 referring to the last axis, -2 to the second last, and so on.

        'split_lengths' specifies the sizes of the output tensor along the target axis. If all entries are positive, they must sum to the size of the target axis in 'input'.
        At most one entry can be -1. If that's the case then the size of the target axis of the corresponding output tensor is deduced from the sum of remaining 'split_lengths' values.

        To demonstrate the role of 'axis' and 'split_lengths', the following ops are all equal:
        ```MLIR
            %variadic_split_0 = IE.VariadicSplit(%input) {axis = 3, split_lengths = [1, 2, 3]} : tensor<3x3x3x6xf32> -> tensor<3x3x3x1xf32>, tensor<3x3x3x2xf32>, tensor<3x3x3x3xf32>
            %variadic_split_1 = IE.VariadicSplit(%input) {axis = 3, split_lengths = [1, 2, -1]} : tensor<3x3x3x6xf32> -> tensor<3x3x3x1xf32>, tensor<3x3x3x2xf32>, tensor<3x3x3x3xf32>
            %variadic_split_2 = IE.VariadicSplit(%input) {axis = -1, split_lengths = [1, 2, -1]} : tensor<3x3x3x6xf32> -> tensor<3x3x3x1xf32>, tensor<3x3x3x2xf32>, tensor<3x3x3x3xf32>
        ```
    }];

    let arguments = (ins
        RankedTensorOf<[AnyInteger, AnyFloat]>:$input,
        IntAttr:$axis,
        I64ArrayAttr:$split_lengths
    );

    let results = (outs
        Variadic<RankedTensorOf<[AnyInteger, AnyFloat]>>:$output
    );

    let extraClassDeclaration = [{
        // returns a copy of 'split_lengths' where the -1 value is inferred
        mlir::SmallVector<int64_t> getInferredSplitLengths();

        // returns the actual axis index which counts from the back if it's negative
        int64_t getInferredAxis();
    }];

    let hasVerifier = 1;
}

#endif
