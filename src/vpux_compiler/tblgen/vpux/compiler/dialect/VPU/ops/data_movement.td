//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_DATA_MOVEMENT
#define VPUX_COMPILER_DIALECT_VPU_OPS_DATA_MOVEMENT

include "vpux/compiler/dialect/core/types.td"
include "vpux/compiler/dialect/core/constraints.td"
include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"


def VPU_BatchToSpace :
        VPU_LayerOp<
            "BatchToSpace"
        > {
    let summary = "BatchToSpace VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        OptionalAttr<I64ArrayAttr>:$block_shape_value,
        OptionalAttr<I64ArrayAttr>:$crops_begin_value,
        OptionalAttr<I64ArrayAttr>:$crops_end_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_BroadcastOp :
        VPU_LayerOp<
            "Broadcast"
        > {
    let summary = "Broadcast VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        1DTensorOf<[AnyInteger]>:$target_shape,
        Optional<1DTensorOf<[AnyInteger]>>:$axes_mapping,

        OptionalAttr<IE_BroadcastTypeAttr>:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_ConcatOp :
        VPU_LayerOp<
            "Concat",
            [
                VPU_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "doesLayerFitIntoCMX"]>
            ]
        > {
    let summary = "VPU Concat layer";

    let arguments = (ins
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor, VPU_DistributedTensor]>>:$inputs,

        OptionalAttr<IE_ConcatAttr>:$per_axis,
        OptionalAttr<I64ArrayOfArraysAttr>:$static_offsets,
        OptionalAttr<I64ArrayOfArraysAttr>:$strides,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "vpux::IE::ConcatAttr":$per_axis)
        >,
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "vpux::IE::ConcatAttr":$per_axis, "mlir::ArrayAttr":$static_offsets)
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
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::IE::ConcatAttr":$per_axis, "mlir::ArrayAttr":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::ArrayRef<vpux::Shape>":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::ArrayRef<vpux::ShapeRef>":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "mlir::ArrayAttr":$static_offsets, "mlir::ArrayAttr":$strides)
        >
    ];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        ::mlir::LogicalResult customVerify();
        bool fitIntoCMX(vpux::NDTypeInterface output, Byte reservedMem);
        bool fitIntoCMX(vpux::NDTypeInterface output);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}



def VPU_CopyOp :
        VPU_LayerOp<
            "Copy"
        > {
    let summary = "Copy VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$input,

        OptionalAttr<IndexedSymbolAttr>:$out_mem_space
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_ConditionalCopyOp :
        VPU_LayerOp<
            "ConditionalCopyOp"
        > {
    let summary = "Conditional copy VPU layer";

    let arguments = (ins
        1DTensorOf<[Bool8, SI8]>:$cond,
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
}



def VPU_DepthToSpaceOp :
        VPU_LayerOp<
            "DepthToSpace",
            [
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "DepthToSpace VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        IntAttr:$block_size,
        IE_DepthToSpaceModeAttr:$mode,
        OptionalAttr<IE_ChannelPaddingAttr>:$padded_channels,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);

        bool isVFSupported();
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::IntegerAttr":$block_size,
            "vpux::IE::DepthToSpaceModeAttr":$mode,
            "vpux::IE::ChannelPaddingAttr":$padded_channels
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];

    let hasVerifier = 1;
}


def VPU_DynamicExpandOp :
        VPU_LayerOp<
            "DynamicExpand"
        > {
    let summary = "DynamicExpand operation";

    let description = [{
        DynamicExpand operation is designed to take an input tensor with dynamic shapes and perform zero-padding to
        match the dimensions of an upper bound.
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
}


def VPU_DynamicTileOp :
        VPU_LayerOp<
            "DynamicTile",
            [
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                VPU_BoundsRepresentationInterface
            ]
        > {
    let summary = "DynamicTile VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$target_shape,

        Optional<RankedTensorOf<[AnyInteger]>>:$repeats,
        OptionalAttr<I64ArrayAttr>:$repeats_values,

        I64ArrayAttr:$output_shape,
        I64ArrayAttr:$output_bounds,

        DefaultValuedAttr<VPU_BoundsRepresentationAttr, "vpux::VPU::BoundsRepresentation::BOUNDS">:$bounds_representation
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;
}


def VPU_ExpandOp :
        VPU_LayerOp<
            "Expand"
        > {
    let summary = "Expand tensor with uninitialized values";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,

        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "std::optional<vpux::ShapeRef>":$pads_begin, "std::optional<vpux::ShapeRef>":$pads_end)
        >
    ];

    let hasFolder = 1;

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}


def VPU_ExpandDilatedOp :
        VPU_LayerOp<
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
}


def VPU_ExtractImagePatchesOp :
        VPU_LayerOp<
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

def VPU_GatherOp :
        VPU_LayerOp<
            "Gather",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "Gather VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[AnyInteger]>, VPU_DistributedTensor]>:$indices,
        Optional<AnyRankedTensor>:$axis,
        OptionalAttr<IntAttr>:$axis_value,
        IntAttr:$batch_dims,

        OptionalAttr<IntAttr>:$indices_rank,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$indices,
            "::mlir::Value":$axis,
            "::mlir::IntegerAttr":$axis_value,
            "::mlir::IntegerAttr":$batch_dims,
            "::mlir::IntegerAttr":$indices_rank
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_GatherDMAOp :
        VPU_LayerOp<
            "GatherDMA",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["getDistributedTypeForOpOperand"]>
            ]
        > {
    let summary = "GatherDMA VPU layer which will be lowered to DMA, used for GatherOps which can be lowered to DMA";

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyTypeOf<[RankedTensorOf<[AnyInteger]>, VPU_DistributedTensor]>:$indices,
        Optional<AnyRankedTensor>:$axis,
        OptionalAttr<IntAttr>:$axis_value,
        IntAttr:$batch_dims,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface indices,vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface indices,vpux::NDTypeInterface output);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_GatherNDOp :
        VPU_LayerOp<
            "GatherND",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "GatherND VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[AnyInteger]>, VPU_DistributedTensor]>:$indices,
        IntAttr:$batch_dims,

        OptionalAttr<I64ArrayAttr>:$original_shape,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$indices,
            "::mlir::IntegerAttr":$batch_dims
        )>,
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$indices,
            "::mlir::IntegerAttr":$batch_dims,
            "::mlir::ArrayAttr":$original_shape
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];

    let hasVerifier = 1;
}


def VPU_GatherElementsOp :
        VPU_LayerOp<
            "GatherElements",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "GatherElements VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[AnyInteger]>, VPU_DistributedTensor]>:$indices,
        IntAttr:$axis,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$indices,
            "::mlir::IntegerAttr":$axis
        )>
    ];
}


def VPU_GatherTreeOp :
        VPU_LayerOp<
              "GatherTree"
        > {
    let summary = "GatherTree VPU layer";

    let arguments = (ins
        AnyRankedTensor:$stepIds,
        AnyRankedTensor:$parentIds,
        AnyRankedTensor:$maxSeqLen,
        AnyRankedTensor:$endToken
    );

    let results = (outs
        AnyRankedTensor:$finalIds
    );
}


def VPU_MemPermuteOp :
        VPU_LayerOp<
            "MemPermute",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "MemPermute VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
            bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

            bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
        }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::AffineMapAttr":$dst_order,
            "::mlir::AffineMapAttr":$mem_perm
        )>
    ];

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 0;
    let hasCanonicalizer = 1;
}


def VPU_PadOp :
        VPU_LayerOp<
            "Pad",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "Pad VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$pads_begin,
        Optional<RankedTensorOf<[AnyInteger]>>:$pads_end,
        Optional<RankedTensorOf<[AnyInteger, AnyFloat]>>:$pad_value,

        OptionalAttr<I64ArrayAttr>:$pads_begin_attr,
        OptionalAttr<I64ArrayAttr>:$pads_end_attr,
        OptionalAttr<F64Attr>:$pad_value_attr,

        IE_PadModeAttr:$mode,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)` (`[` $pads_begin^ `,` $pads_end (`,` $pad_value^)? `]`)? attr-dict `:` type(operands) `->` type(results)
    }];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$pads_begin,
            "::mlir::Value":$pads_end,
            "::mlir::Value":$pad_value,
            "::mlir::ArrayAttr":$pads_begin_attr,
            "::mlir::ArrayAttr":$pads_end_attr,
            "::mlir::FloatAttr":$pad_value_attr,
            "vpux::IE::PadModeAttr":$mode,
            "::mlir::ArrayAttr":$output_padding,
            "::mlir::ArrayAttr":$input_padding
        )>,
        OpBuilder<(ins
            "vpux::NDTypeInterface&":$input_type,
            "::mlir::Value":$input,
            "::mlir::Value":$pads_begin,
            "::mlir::Value":$pads_end,
            "::mlir::Value":$pad_value,
            "::mlir::ArrayAttr":$pads_begin_attr,
            "::mlir::ArrayAttr":$pads_end_attr,
            "::mlir::FloatAttr":$pad_value_attr,
            "vpux::IE::PadMode":$mode,
            "::mlir::ArrayAttr":$output_padding,
            "::mlir::ArrayAttr":$input_padding
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_PerAxisTileOp :
        VPU_LayerOp<
            "PerAxisTile"
        > {
    let summary = "Per axis Tile VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        IntAttr:$axis,
        IntAttr:$tiles
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_ReverseOp :
        VPU_LayerOp<
            "Reverse",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverBatchCompatible"]>
            ]
        > {
    let summary = "Reverse VPU operation";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axis_value,
        IE_ReverseModeAttr:$mode,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

     let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
        DimArr getTileableDims();
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axis_value,
            "vpux::IE::ReverseModeAttr":$mode
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_ReverseSequenceOp :
        VPU_LayerOp<
            "ReverseSequence"
        > {
    let summary = "Reverse variable length sequence  VPU operation";

    let arguments = (ins
        AnyRankedTensor:$data,
        1DTensorOf<[AnyInteger]>:$seq_length,

        IntAttr:$seq_axis,
        IntAttr:$batch_axis
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_RollOp :
        VPU_LayerOp<
            "Roll",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["getDistributedTypeForOpOperand",
                                                                     "getDistributedTypeForOpResult"]>
            ]
        > {
    let summary = "Roll VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$data,
        AnyTypeOf<[RankedTensorOf<[SI32, SI64]>, VPU_DistributedTensor]>:$shift,
        AnyTypeOf<[RankedTensorOf<[SI32, SI64]>, VPU_DistributedTensor]>:$axes,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$data,
            "::mlir::Value":$shift,
            "::mlir::Value":$axes
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_ScatterElementsUpdateOp :
        VPU_LayerOp<
            "ScatterElementsUpdate"
        > {
    let summary = "ScatterElementsUpdate VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        AnyRankedTensor:$updates,
        IntAttr:$axis,
        IE_ScatterElementsUpdateReductionTypeAttr:$reduction,
        BoolAttr:$use_init_val
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_ScatterNDUpdateOp :
        VPU_LayerOp<
            "ScatterNDUpdate"
        > {
    let summary = "ScatterNDUpdate VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        AnyRankedTensor:$updates
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_ScatterUpdateOp :
        VPU_LayerOp<
            "ScatterUpdate"
        > {
    let summary = "ScatterUpdate VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        AnyRankedTensor:$updates,
        OptionalAttr<IntAttr>:$axis_value

    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_SliceOp :
        VPU_LayerOp<
            "Slice",
            [
                VPU_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "Extract single slice from ranked tensor or distributed tensor";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor]>:$input,
        I64ArrayAttr:$static_offsets,
        I64ArrayAttr:$static_sizes
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor]>:$output
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
    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];

    let extraClassDeclaration = [{
                mlir::Value getSource() { return getInput(); }
    }] # baseExtraClassDeclaration;
}


def VPU_SpaceToDepthOp :
        VPU_LayerOp<
            "SpaceToDepthOp",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "SpaceToDepthOp VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        DefaultValuedAttr<IntAttr, "1">:$block_size,
        IE_SpaceToDepthModeAttr:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
}


def VPU_SpaceToBatch :
        VPU_LayerOp<
            "SpaceToBatch"
        > {
    let summary = "SpaceToBatch VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        OptionalAttr<I64ArrayAttr>:$block_shape_value,
        OptionalAttr<I64ArrayAttr>:$pads_begin_value,
        OptionalAttr<I64ArrayAttr>:$pads_end_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_SplitOp :
        VPU_LayerOp<
            "Split",
            [
            VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Split VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,
        Optional<AnyRankedTensor>:$axis,

        IntAttr:$num_splits,
        OptionalAttr<IntAttr>:$axis_value
    );

    let results = (outs
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>>:$outputs
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
    let hasVerifier = 1;
}


def VPU_StridedSliceOp :
        VPU_LayerOp<
            "StridedSlice",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_BoundsRepresentationInterface
            ]
        > {
    let summary = "StridedSlice VPU layer";

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
        I64ArrayAttr:$ellipsis_mask,

        DefaultValuedAttr<VPU_BoundsRepresentationAttr, "vpux::VPU::BoundsRepresentation::BOUNDS">:$bounds_representation
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let extraClassDeclaration = [{
        bool isSimplified();
    }];
}


def VPU_TileOp :
        VPU_LayerOp<
            "Tile",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "Tile VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$repeats_values
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}


def VPU_UpsamplingOp :
        VPU_LayerOp<
            "Upsampling"
        > {
    let summary = "Upsampling VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input,
        I64ArrayAttr:$upsampling_factor,
        OptionalAttr<IE_UpsamplingPadAttr>:$pad
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );
}

#endif
