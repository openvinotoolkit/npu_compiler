//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_SPECIALIZED
#define VPUX_COMPILER_DIALECT_IE_OPS_SPECIALIZED

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/core/types.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
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
    let hasCanonicalizer = 1;
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

//
// IncrementalSDPA
//

def IE_IncrementalSDPAOp :
        IE_LayerOp<
            "IncrementalSDPA",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "IncrementalSDPA VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$query,
        RankedTensorOf<[F16, F32, F64]>:$key,
        RankedTensorOf<[F16, F32, F64]>:$value,

        RankedTensorOf<[F16, F32, F64]>:$input_partial_output,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$input_running_max,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$input_running_sum,

        Optional<RankedTensorOf<[F16, F32, F64]>>:$attention_mask,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$scale,

        OptionalAttr<IntAttr>:$kv_num_blocks

    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$result_partial_output,
        RankedTensorOf<[F16, F32, F64]>:$result_running_max,
        RankedTensorOf<[F16, F32, F64]>:$result_running_sum
    );
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
        TypeAttr:$outputType
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//
// OnlineSDPA
//

def IE_OnlineSDPAOp :
        IE_LayerOp<
            "OnlineSDPA",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "OnlineSDPA VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$query,
        RankedTensorOf<[F16, F32, F64]>:$key,
        RankedTensorOf<[F16, F32, F64]>:$value,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$attention_mask,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$scale,

        OptionalAttr<IntAttr>:$kv_num_blocks
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_PermuteCastOp :
        IE_LayerOp<
            "PermuteCast",
            [
                IE_ViewLikeOpInterface
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
        RankedTensorOf<[F16, F32]>:$input,

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
        RankedTensorOf<[F16, F32, F64]>:$input_sin
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
        RankedTensorOf<[SI32, SI64, F16, F32, F64]>:$input2,
        RankedTensorOf<[SI32, SI64, F16, F32, F64]>:$input3,
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

#endif
