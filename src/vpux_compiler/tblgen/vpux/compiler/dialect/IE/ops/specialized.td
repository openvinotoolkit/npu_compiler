//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_SPECIALIZED
#define VPUX_COMPILER_DIALECT_IE_OPS_SPECIALIZED

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/core/types.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_AssignOp :
        IE_Op<
            "Assign",
            [
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                DeclareOpInterfaceMethods<IE_LayerOpInterface>
            ]
        > {
    let summary = "InferenceEngine Assign layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        StrAttr:$name
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

def IE_BucketizeOp :
        IE_LayerOp<
            "Bucketize"
        > {
    let summary = "InferenceEngine Bucketize layer";

    let arguments = (ins
        AnyRankedTensor:$data,
        1DTensorOf<[AnyInteger, AnyFloat]>:$buckets,

        TypeAttr:$output_type,
        UnitAttr:$with_right_bound
    );

    let results = (outs
        RankedTensorOf<[SI32, SI64]>:$output
    );

    let hasVerifier = 1;
}

// CodeGen Ops - to be moved: E#166215

//
// CodeGenCapsule
//

def IE_CodeGenCapsuleOp :
        IE_Op<
            "CodeGenCapsule",
            [
                SingleBlockImplicitTerminator<"CGCYieldOp">,
                IsolatedFromAbove
            ]
        > {
    let summary = "Provides encapsulation for Shave-targeted code/IR entities";

    let arguments = (ins
        Variadic<AnyType>:$inputs
    );

    let results = (outs
        Variadic<AnyType>:$outputs
    );

    let regions = (region
        SizedRegion<1>:$content
    );

    let extraClassDeclaration = [{
        // Get the block arguments, as internal views of the kernel arguments
        mlir::SmallVector<mlir::Value> getInternalArgs();

        // Get the results, as viewed internally in the kernel
        mlir::SmallVector<mlir::Value> getInternalResults();

        mlir::DenseMap<mlir::Value, mlir::Value> getYieldToResultMapping();
        mlir::DenseMap<mlir::Value, mlir::Value> getResultToYieldMapping();
    }];

    let hasVerifier = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// CodeGenCapsuleYieldOp
//

def IE_CGCYieldOp :
        IE_Op<
            "CGCYield",
            [
                HasParent<"CodeGenCapsuleOp">,
                Pure,
                Terminator
            ]
        > {
    let summary = "Terminator & return value specifier for CodeGenCapsules";

    let arguments = (ins
        Variadic<AnyType>:$values
    );

    let builders = [
      OpBuilder<(ins)>
    ];

    let assemblyFormat = [{
        attr-dict ( $values^ `:` type($values) )?
    }];
}

//
// ExternalKernelOp
//

def IE_ExternalKernelOp :
        IE_Op<
            "ExternalKernel",
            []
        > {
    let summary = "Represents a kernel whose details/implementation are defined externally";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,
        DictionaryAttr:$attrDict,
        StrAttr:$unique_id,
        StrAttr:$kernel_path
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );

    let assemblyFormat = [{
        $unique_id
        `at_path` `(` $kernel_path `)`
        (`inputs` `(` $inputs^ `:` type($inputs) `)`)?
        `attrs` `(` $attrDict `)`
        attr-dict
        `->` type(results)
    }];
}

def IE_DetectionOutputOp :
        IE_LayerOp<
            "DetectionOutput",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine DetectionOutput layer";

    let arguments = (ins
        2DTensorOf<[AnyFloat]>:$in_box_logits,
        2DTensorOf<[AnyFloat]>:$in_class_preds,
        3DTensorOf<[AnyFloat]>:$in_proposals,
        Optional<2DTensorOf<[AnyFloat]>>:$in_additional_preds,
        Optional<2DTensorOf<[AnyFloat]>>:$in_additional_proposals,

        IE_DetectionOutputAttr:$attr
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_DFTOp :
        IE_LayerOp<
            "DFT",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine DFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,
        Optional<RankedTensorOf<[AnyInteger]>>:$signal_size,

        OptionalAttr<I64ArrayAttr>:$axes_attr,
        OptionalAttr<I64ArrayAttr>:$signal_size_attr
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
    let hasCanonicalizer = 1;
}

def IE_DynamicDataMaskOp :
        IE_LayerOp<
            "DynamicDataMask",
            [
                DeclareOpInterfaceMethods<StaticShapeOpInterface>
            ]
        > {
    let summary = "Create mask to clear garbage from pad area, added by processing dynamic tensors by upper bounds";

    let arguments = (ins
        AnyRankedTensor:$realShape,
        TypeAttr:$outputTensorType
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_EmbeddingBagOffsetsSumOp :
        IE_LayerOp<
            "EmbeddingBagOffsetsSum",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine EmbeddingBagOffsetsSum layer";

    let arguments = (ins
        AnyRankedTensor:$emb_table,

        Optional<1DTensorOf<[SI64, SI32]>>:$indices,
        Optional<1DTensorOf<[SI64, SI32]>>:$offsets,
        Optional<RankedTensorOf<[SI64, SI32]>>:$default_index,
        Optional<1DTensorOf<[AnyInteger, AnyFloat]>>:$per_sample_weights,

        OptionalAttr<I64ArrayAttr>:$indices_value,
        OptionalAttr<I64ArrayAttr>:$offsets_value,
        OptionalAttr<IntAttr>:$default_index_value,
        OptionalAttr<F64ArrayAttr>:$per_sample_weights_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def IE_EmbeddingBagPackedSumOp :
        IE_LayerOp<
            "EmbeddingBagPackedSum"
        > {
    let summary = "InferenceEngine EmbeddingBagPackedSum layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$emb_table,
        2DTensorOf<[SI32, SI64]>:$indices,
        Optional<2DTensorOf<[F16, F32, F64]>>:$per_sample_weights
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_EmbeddingSegmentsSumOp :
        IE_LayerOp<
            "EmbeddingSegmentsSum",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine EmbeddingSegmentsSum layer";

    let arguments = (ins
        AnyRankedTensor:$emb_table,
        Optional<1DTensorOf<[SI64, SI32]>>:$indices,
        Optional<1DTensorOf<[SI64, SI32]>>:$segment_ids,
        Optional<RankedTensorOf<[SI64, SI32]>>:$num_segments,
        Optional<RankedTensorOf<[SI64, SI32]>>:$default_index,
        Optional<1DTensorOf<[AnyInteger, AnyFloat]>>:$per_sample_weights,

        OptionalAttr<I64ArrayAttr>:$indices_value,
        OptionalAttr<I64ArrayAttr>:$segment_ids_value,
        OptionalAttr<IntAttr>:$num_segments_value,
        OptionalAttr<IntAttr>:$default_index_value,
        OptionalAttr<F64ArrayAttr>:$per_sample_weights_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def IE_EyeOp :
        IE_LayerOp<
            "Eye",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine Eye layer";

    let arguments = (ins
        Optional<1DTensorOf<[SI32, SI64]>>:$num_rows,
        Optional<1DTensorOf<[SI32, SI64]>>:$num_columns,
        1DTensorOf<[SI32, SI64]>:$diagonal_index,
        Optional<1DTensorOf<[SI32, SI64]>>:$batch_shape,

        OptionalAttr<IntAttr>:$num_rows_value,
        OptionalAttr<IntAttr>:$num_columns_value,
        OptionalAttr<I64ArrayAttr>:$batch_shape_value,

        TypeAttr:$outputType
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
}

def IE_IDFTOp :
        IE_LayerOp<
            "IDFT",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine IDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,
        Optional<RankedTensorOf<[AnyInteger]>>:$signal_size,

        OptionalAttr<I64ArrayAttr>:$axes_attr,
        OptionalAttr<I64ArrayAttr>:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
    let hasCanonicalizer = 1;
}

def IE_IRDFTOp :
        IE_LayerOp<
            "IRDFT",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine IRDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,
        Optional<RankedTensorOf<[AnyInteger]>>:$signal_size,

        OptionalAttr<I64ArrayAttr>:$axes_attr,
        OptionalAttr<I64ArrayAttr>:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
    let hasCanonicalizer = 1;
}

def IE_LayoutCastOp :
        IE_LayerOp<
            "LayoutCast",
            [
                IE_ViewLikeOpInterface
            ]
        > {
    let summary = "This layer overrides layout of a given tensor.";

    let arguments = (ins
        AnyRankedTensor:$input,
        AffineMapAttr:$dst_order
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

def IE_NonMaxSuppressionOp :
        IE_LayerOp<
            "NonMaxSuppression",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine NonMaxSuppression layer";

    let arguments = (ins
        3DTensorOf<[F16, F32, F64]>:$in_box_coords,
        3DTensorOf<[F16, F32, F64]>:$in_box_scores,
        Optional<1DTensorOf<[SI32, SI64]>>:$max_output_boxes_per_class,
        Optional<1DTensorOf<[F16, F32, F64]>>:$iou_threshold,
        Optional<1DTensorOf<[F16, F32, F64]>>:$score_threshold,
        Optional<1DTensorOf<[F16, F32, F64]>>:$soft_nms_sigma,

        IE_BoxEncodingTypeAttr:$box_encoding,
        UnitAttr:$sort_result_descending,

        OptionalAttr<IntAttr>:$max_output_boxes_per_class_value,
        OptionalAttr<F64Attr>:$iou_threshold_value,
        OptionalAttr<F64Attr>:$score_threshold_value,
        OptionalAttr<F64Attr>:$soft_nms_sigma_value
    );

    let results = (outs
        2DTensorOf<[SI32]>:$out_selected_indices,
        2DTensorOf<[F16, F32, F64]>:$out_selected_scores,
        1DTensorOf<[SI32]>:$out_valid_outputs
    );

    let hasCanonicalizer = 1;
}

def IE_NonZeroOp :
        IE_LayerOp<
            "NonZero"
        > {
    let summary = "InferenceEngine NonZero layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32, SI64, Bool8]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        2DTensorOf<[SI32, SI64]>:$output
    );
}

def IE_OneHotOp :
        IE_LayerOp<
            "OneHot",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine OneHot layer";

    let arguments = (ins
        RankedTensorOf<[SI32, SI64]>:$input,
        Optional<RankedTensorOf<[SI32, SI64]>>:$depth,
        Optional<RankedTensorOf<[AnyType]>>:$on_value,
        Optional<RankedTensorOf<[AnyType]>>:$off_value,

        OptionalAttr<IntAttr>:$depth_attr,
        OptionalAttr<F64Attr>:$on_value_attr,
        OptionalAttr<F64Attr>:$off_value_attr,

        IntAttr:$axis_attr,
        OptionalAttr<IE_OneHotModeAttr>:$mode,

        TypeAttr:$outputType
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//
// FlashSDPA
//

def IE_FlashSDPAOp :
        IE_LayerOp<
            "FlashSDPA"
        > {
    let summary = "Flash Attention SDPA layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$query,
        RankedTensorOf<[F16, F32, F64]>:$key,
        RankedTensorOf<[F16, F32, F64]>:$value,
        RankedTensorOf<[F16, F32, F64]>:$input_running_output,
        RankedTensorOf<[F16, F32, F64]>:$input_running_max,
        RankedTensorOf<[F16, F32, F64]>:$input_running_sum,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$attention_mask,

        BoolAttr:$is_head,
        BoolAttr:$is_tail,

        IntAttr:$source_seq_len_pad_size
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$result_running_output,
        RankedTensorOf<[F16, F32, F64]>:$result_running_max,
        RankedTensorOf<[F16, F32, F64]>:$result_running_sum
    );
}

def IE_PermuteCastOp :
        IE_LayerOp<
            "PermuteCast",
            [
                IE_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface, ["shouldJITCompile", "shouldJITCompileToEnableFusion"]>
            ]
        > {
    let summary = "InferenceEngine PermuteCast layer";

    let description = [{
        The op changes layout information in the following way:
            * dst_order: layout attribute of result is set to value of this arg
            * mem_perm: describes the permutation applied on the input value's memory shape
                        to obtain the memory shape of the output value.
    }];

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
    let checkInferredMemSpace = 1;
}

def IE_PermuteQuantizeOp :
        IE_LayerOp<
            "PermuteQuantize",
            [
                DeclareOpInterfaceMethods<StaticShapeOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "InferenceEngine PermuteQuantize layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm,
        TypeAttr:$dstElemType,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def IE_ProposalOp :
        IE_LayerOp<
            "Proposal"
        > {
    let summary = "InferenceEngine Proposal layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$class_probs,
        RankedTensorOf<[F16, F32, F64]>:$bbox_deltas,
        RankedTensorOf<[F16, F32, F64]>:$image_shape,

        IE_ProposalAttr:$proposal_attrs
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output,
        RankedTensorOf<[F16, F32, F64]>:$probs
    );
}

def IE_RandomUniformOp :
        IE_LayerOp<
            "RandomUniform"
        > {
    let summary = "InferenceEngine RandomUniform layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64, SI32]>:$min,
        RankedTensorOf<[F16, F32, F64, SI32]>:$max,

        I64ArrayAttr:$output_shape,
        TypeAttr:$outputType,
        IntAttr:$global_seed,
        IntAttr:$op_seed
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64, SI32]>:$output
    );

    let hasVerifier = 1;
}

def IE_RangeOp :
        IE_LayerOp<
            "Range"
        > {
    let summary = "InferenceEngine Range layer";

    let arguments = (ins
        AnyRankedTensor:$start,
        AnyRankedTensor:$stop,
        AnyRankedTensor:$step,
        TypeAttr:$dstElemType
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_RDFTOp :
        IE_LayerOp<
            "RDFT",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine RDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,
        Optional<RankedTensorOf<[AnyInteger]>>:$signal_size,

        OptionalAttr<I64ArrayAttr>:$axes_attr,
        OptionalAttr<I64ArrayAttr>:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
    let hasCanonicalizer = 1;
}

def IE_ReadValueOp :
        IE_Op<
            "ReadValue",
            [
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                DeclareOpInterfaceMethods<IE_LayerOpInterface>
            ]
        > {
    let summary = "InferenceEngine ReadValue layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        StrAttr:$name,
        OptionalAttr<TypeAttr>:$element_type,
        OptionalAttr<I64ArrayAttr>:$shape
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

def IE_RegionYoloOp :
        IE_LayerOp<
            "RegionYolo"
        > {
    let summary = "InferenceEngine RegionYolo layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$input,

        IntAttr:$coords,
        IntAttr:$classes,
        IntAttr:$num_regions,
        BoolAttr:$do_softmax,
        I64ArrayAttr:$mask,
        IntAttr:$axis,
        IntAttr:$end_axis,
        F64ArrayAttr:$anchors
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_ReorgYoloOp :
        IE_LayerOp<
            "ReorgYolo"
        > {
    let summary = "InferenceEngine ReorgYolo layer";

    let arguments = (ins
        4DTensorOf<[AnyInteger, AnyFloat]>:$input,

        IntAttr:$stride
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
}

def IE_RoPEOp :
        IE_LayerOp<
            "RoPE"
        > {
    let summary = "InferenceEngine RoPE layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        RankedTensorOf<[F16, F32, F64]>:$input_cos,
        RankedTensorOf<[F16, F32, F64]>:$input_sin,

        IE_RoPEModeAttr:$mode
     );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_SDPAOp :
        IE_LayerOp<
            "SDPA",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine SDPA layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$inputQ,
        RankedTensorOf<[F16, F32]>:$inputK,
        RankedTensorOf<[F16, F32]>:$inputV,
        Optional<RankedTensorOf<[F16, F32]>>:$inputMask,
        Optional<RankedTensorOf<[F16, F32]>>:$inputScale,

        UnitAttr:$causal
     );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );

    let hasCanonicalizer = 1;
}

def IE_SDPAExtendedOp :
        IE_LayerOp<
            "SDPAExtended",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine custom SDPA layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$inputQ,
        RankedTensorOf<[F16, F32]>:$inputK,
        RankedTensorOf<[F16, F32]>:$inputV,
        Optional<RankedTensorOf<[F16, F32]>>:$inputMask,
        Optional<RankedTensorOf<[F16, F32]>>:$inputScale,
        Optional<RankedTensorOf<[F16, F32]>>:$inputBias,

        OptionalAttr<IntAttr>:$padSizeS
     );
   let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_SelectOp :
        IE_LayerOp<
            "Select",
            [
                DeclareOpInterfaceMethods<IE_ShaveCodeGenSupportedOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Select layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        RankedTensorOf<[SI16, SI32, SI64, F16, F32, F64]>:$input2,
        RankedTensorOf<[SI16, SI32, SI64, F16, F32, F64]>:$input3,
        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_StubOp :
        IE_Op<
            "Stub",
            [
                Pure
            ]
        > {
    let summary = "Substitute operation for stubbing.";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

def IE_TopKOp :
        IE_LayerOp<
            "TopK"
        > {
    let summary = "InferenceEngine TopK layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$k,
        OptionalAttr<IntAttr>:$k_value,

        IntAttr:$axis,
        IE_TopKModeAttr:$mode,
        IE_TopKSortTypeAttr:$sort,
        TypeAttr:$element_type
    );

    let results = (outs
        AnyRankedTensor:$output_values,
        AnyRankedTensor:$target_shape
    );

    let hasCanonicalizer = 1;
    let hasVerifier = 1;
}

def IE_LogSoftmaxTopKOp :
        IE_LayerOp<
            "LogSoftmaxTopK"
        > {
    let summary = "InferenceEngine LogSoftmaxTopK layer";

    let description = [{
        This operator is fused from a LogSoftmaxOp and a TopKOp (K=1) with both on the inner axis
    }];

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        IntAttr:$axisInd,
        IntAttr:$padSize,
        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[F32]>:$output,
        RankedTensorOf<[SI64]>:$topKOutput
    );
}

def IE_LogSoftmaxPeakOp :
        IE_LayerOp<
            "LogSoftmaxPeak"
        > {
    let summary = "InferenceEngine LogSoftmaxPeak layer";

    let description = [{
        This operator is fused from a LogSoftmaxOp and a TopKOp (K=1) going into a GatherElements
    }];

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,

        IntAttr:$axisInd,
        IntAttr:$padSize,
        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[F32]>:$output,
        RankedTensorOf<[SI64]>:$topKOutput
    );
}

def IE_STFTOp :
        IE_LayerOp<
            "STFT",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine Short-Time Fourier Transform (STFT) layer";

    let description = [{
        STFT operation performs Short-Time Fourier Transform of real-valued input tensor.

        Based on OpenVINO STFT-15 specification:
        - Input signal: 1D [signal_size] or 2D [batch, signal_size]
        - Window: 1D [window_length] for windowing function
        - Frame size: scalar describing size of signal frame for FFT
        - Frame step: scalar describing distance between successive frames
        - Output: Complex result with real and imaginary parts in last dimension

        Output shapes:
        - transpose_frames=false: [frames, fft_results, 2] or [batch, frames, fft_results, 2]
        - transpose_frames=true: [fft_results, frames, 2] or [batch, fft_results, frames, 2]

        Where:
        - frames = (signal_shape[-1] - frame_size) / frame_step + 1
        - fft_results = (frame_size / 2) + 1
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$signal,
        RankedTensorOf<[F16, F32, F64]>:$window,
        Optional<RankedTensorOf<[AnyInteger]>>:$frame_size,
        Optional<RankedTensorOf<[AnyInteger]>>:$frame_step,

        OptionalAttr<IntAttr>:$frame_size_attr,
        OptionalAttr<IntAttr>:$frame_step_attr,
        UnitAttr:$transpose_frames
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );

    let hasVerifier = 1;
}

def IE_ISTFTOp :
        IE_LayerOp<
            "ISTFT"
        > {
    let summary = "InferenceEngine Inverse Short-Time Fourier Transform (ISTFT) layer";

    let description = [{
        ISTFT operation performs Inverse Short-Time Fourier Transform to reconstruct time-domain signals
        from frequency-domain representations.

        Based on OpenVINO ISTFT specification:
        - Input signal: 3D [fft_results, frames, 2] or 4D [batch, fft_results, frames, 2]
        - Window: Required 1D [window_length] tensor of type T, specifying the window   values applied to restore the signal. The window_length is required to be equal or smaller than frame_size, if smaller the window will be padded with zeros on the left and right sides.
        - Frame size: scalar describing the size of each time-domain frame
        - Frame step: scalar describing the hop length between frames
        - Signal length: optional scalar specifying the desired output signal length

        If the signal_length is not provided as an input value, it is calculated according to the following rules:

        - default_signal_length = (frames - 1) * frame_step for center == true
        - default_signal_length = (frames - 1) * frame_step + frame_size for center == false

        If the signal_length input is provided, the number of output values will be adjusted accordingly.
        - If signal_length > default_signal_length the output is padded with zeros at the end
        - If signal_length < default_signal_length any additional generated samples are cut to the signal_length size

        The operation reconstructs the time-domain signal using overlap-add method:
        1. Apply IRDFT to each frequency frame to get time-domain frames
        2. Apply windowing function (if provided) to each frame
        3. Overlap and add frames according to frame_step to reconstruct continuous signal

         Input to output shape mapping:
        - 3D input [fft_results, frames, 2] → 1D output [signal_length]
        - 4D input [batch, fft_results, frames, 2] → 2D output [batch, signal_length]


        Attributes:
        - center
        - normalized
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$signal,
        RankedTensorOf<[F16, F32, F64]>:$window,
        RankedTensorOf<[AnyInteger]>:$frame_size,
        RankedTensorOf<[AnyInteger]>:$frame_step,
        Optional<RankedTensorOf<[AnyInteger]>>:$signal_length,

        OptionalAttr<UnitAttr>:$center,
        OptionalAttr<UnitAttr>:$normalized
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );

    let hasVerifier = 1;
}

#endif
