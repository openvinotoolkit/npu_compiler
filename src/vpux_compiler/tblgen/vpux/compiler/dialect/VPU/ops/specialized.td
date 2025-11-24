//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_SPECIALIZED
#define VPUX_COMPILER_DIALECT_VPU_OPS_SPECIALIZED

include "vpux/compiler/dialect/core/types.td"
include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"


def VPU_BucketizeOp :
        VPU_LayerOp<
            "Bucketize"
        > {
    let summary = "Bucketize VPU layer";

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


def VPU_DetectionOutputOp :
        VPU_LayerOp<
            "DetectionOutput",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "DetectionOutput VPU layer";

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


def VPU_DetectionOutputNormalizeOp:
        VPU_LayerOp<
            "DetectionOutputNormalize"
        > {
    let summary = "DetectionOutputNormalize VPU layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$prior_boxes,

        IntAttr:$input_width,
        IntAttr:$input_height
    );

    let results = (outs
        4DTensorOf<[AnyFloat]>:$out_prior_boxes
    );

    let hasVerifier = 1;
}


def VPU_DetectionOutputDecodeBoxesOp:
        VPU_LayerOp<
            "DetectionOutputDecodeBoxes",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "DetectionOutputDecodeBoxes VPU layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$box_logits,
        4DTensorOf<[AnyFloat]>:$prior_boxes,

        IE_DetectionOutputCodeTypeAttr:$code_type,
        BoolAttr:$clip_before_nms
    );

    let results = (outs
        4DTensorOf<[AnyFloat]>:$out_decoded_boxes
    );
}


def VPU_DetectionOutputSortOp:
        VPU_LayerOp<
            "DetectionOutputSort",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible"]>,
            ]
        > {
    let summary = "DetectionOutputSort VPU layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[AnyFloat]>, VPU_DistributedTensor]>:$confidence,
        AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>:$indicesBuffer,
        AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>:$sortingBuffer,

        F64Attr:$confidence_threshold,
        IntAttr:$top_k,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[AnyFloat]>, VPU_DistributedTensor]>:$out_confidence,
        AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>:$out_indices,
        AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>:$out_sizes
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$confidence,
            "::mlir::FloatAttr":$confidence_threshold,
            "::mlir::IntegerAttr":$top_k
        )>
    ];

    let extraClassDeclaration = [{
        OutputTiling getOutputTiling(const vpux::TileInfo& outputTile, vpux::Logger log);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_DetectionOutputNmsCaffeOp:
        VPU_LayerOp<
            "DetectionOutputNmsCaffe",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "DetectionOutputNmsCaffe VPU layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$confidence,
        4DTensorOf<[AnyFloat]>:$boxes,
        4DTensorOf<[SI32]>:$indices,
        4DTensorOf<[SI32]>:$sizes,

        IntAttr:$top_k,
        F64Attr:$nms_threshold,
        IntAttr:$background_id
    );

    let results = (outs
        4DTensorOf<[AnyFloat]>:$out_confidence,
        4DTensorOf<[AnyFloat]>:$out_boxes,
        4DTensorOf<[SI32]>:$out_sizes
    );

    let extraClassDeclaration = [{
        OutputTiling getOutputTiling(const vpux::TileInfo& outputTile, vpux::Logger log);
    }] # baseExtraClassDeclaration;
}


def VPU_DetectionOutputCollectResultsOp:
        VPU_LayerOp<
            "DetectionOutputCollectResults"
        > {
    let summary = "DetectionOutputCollectResults VPU layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$confidence,
        4DTensorOf<[AnyFloat]>:$boxes,
        4DTensorOf<[SI32]>:$sizes,

        IntAttr:$keep_top_k,
        BoolAttr:$clip_after_nms
    );

    let results = (outs
        4DTensorOf<[AnyFloat]>:$out_detections
    );
}


def VPU_DFTOp :
        VPU_LayerOp<
            "DFT",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "InferenceEngine DFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$twiddle_factors,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_RDFTOp :
        VPU_LayerOp<
            "RDFT"
        > {
    let summary = "InferenceEngine RDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$twiddle_factors,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_IDFTOp :
        VPU_LayerOp<
            "IDFT",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "InferenceEngine IDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$twiddle_factors,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_IRDFTOp :
        VPU_LayerOp<
            "IRDFT"
        > {
    let summary = "InferenceEngine IRDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$twiddle_factors,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_RDFTUncutOp :
        VPU_LayerOp<
            "RDFTUncut",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "RDFTUncut VPU layer";

    let description = [{
        Operation apply RDFT transformation but not cut symmetric part on last axis width value from axes_attr.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$twiddle_factors,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_IRDFTLastAxisOp :
        VPU_LayerOp<
            "IRDFTLastAxis",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "IRDFTLastAxis VPU layer";

    let description = [{
        Operation apply IRDFT transformation but just on last axis from standard IRDFT operation.
        Used to produce full IRDFT capability (in combination with IDFT) without computation
        and the data movement unnecessary for the direction of the last transform axis.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$twiddle_factors,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_DynamicDataMaskOp :
        VPU_LayerOp<
            "DynamicDataMask"
        > {
    let summary = "Create mask to clear garbage from pad area, added by processing dynamic tensors by upper bounds";

    let arguments = (ins
        Optional<1DTensorOf<[SI64, SI32]>>:$realShape,
        TypeAttr:$outputTensorType
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_EmbeddingBagOffsetsSumOp :
        VPU_LayerOp<
            "EmbeddingBagOffsetsSum",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine EmbeddingBagOffsetsSum layer";

    let arguments = (ins
        AnyRankedTensor:$emb_table,
        Optional<1DTensorOf<[SI32, SI64]>>:$indices,
        Optional<1DTensorOf<[SI32, SI64]>>:$offsets,
        Optional<1DTensorOf<[AnyInteger, AnyFloat]>>:$per_sample_weights,

        OptionalAttr<I64ArrayAttr>:$indices_value,
        OptionalAttr<I64ArrayAttr>:$offsets_value,
        IntAttr:$default_index_value,
        OptionalAttr<F64ArrayAttr>:$per_sample_weights_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_EmbeddingSegmentsSumOp :
        VPU_LayerOp<
            "EmbeddingSegmentsSum",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "EmbeddingSegmentsSum VPU layer";

    let arguments = (ins
        AnyRankedTensor:$emb_table,
        Optional<1DTensorOf<[SI64, SI32]>>:$indices,
        Optional<1DTensorOf<[SI64, SI32]>>:$segment_ids,
        Optional<1DTensorOf<[AnyInteger, AnyFloat]>>:$per_sample_weights,

        OptionalAttr<I64ArrayAttr>:$indices_value,
        OptionalAttr<I64ArrayAttr>:$segment_ids_value,
        IntAttr:$num_segments_value,
        IntAttr:$default_index_value,
        OptionalAttr<F64ArrayAttr>:$per_sample_weights_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_EmbeddingBagPackedSumOp :
        VPU_LayerOp<
            "EmbeddingBagPackedSum"
        > {
    let summary = "EmbeddingBagPackedSum VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$emb_table,
        2DTensorOf<[SI32, SI64]>:$indices,
        Optional<2DTensorOf<[F16, F32]>>:$per_sample_weights
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_EyeOp :
        VPU_LayerOp<
            "Eye"
        > {
    let summary = "Eye VPU layer";

    let arguments = (ins
        1DTensorOf<[SI32, SI64]>:$diagonal_index,

        IntAttr:$num_rows_value,
        IntAttr:$num_columns_value,
        I64ArrayAttr:$batch_shape_value,

        TypeAttr:$outputType
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_SDPAOp :
        VPU_LayerOp<
            "SDPA",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_AuxiliaryBufferOpInterface>
            ]
        > {
    let summary = "SDPA VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$inputQ,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$inputK,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$inputV,
        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>>:$inputMask,
        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>>:$inputScale,
        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>>:$inputBias,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$dataStorage,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$inputQ,
            "::mlir::Value":$inputK,
            "::mlir::Value":$inputV,
            "::mlir::Value":$inputMask,
            "::mlir::Value":$inputScale,
            "::mlir::Value":$inputBias,
            "::mlir::Value":$dataStorage
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_SDPAExtendedOp :
        VPU_LayerOp<
            "SDPAExtended",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_AuxiliaryBufferOpInterface>
            ]
        > {
    let summary = "SDPAExtended VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$inputQ,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$inputK,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$inputV,
        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>>:$inputMask,
        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>>:$inputScale,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$dataStorage,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$dpuStorage,
        OptionalAttr<IntAttr>:$padSizeS,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$inputQ,
            "::mlir::Value":$inputK,
            "::mlir::Value":$inputV,
            "::mlir::Value":$inputMask,
            "::mlir::Value":$inputScale,
            "::mlir::Value":$dataStorage,
            "::mlir::Value":$dpuStorage,
            "::mlir::IntegerAttr":$padSizeS
        )>,
        OpBuilder<(ins
            "::mlir::Value":$inputQ,
            "::mlir::Value":$inputK,
            "::mlir::Value":$inputV,
            "::mlir::Value":$inputMask,
            "::mlir::Value":$inputScale,
            "::mlir::Value":$dataStorage,
            "::mlir::IntegerAttr":$padSizeS
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_RoPEOp :
        VPU_LayerOp<
            "RoPE",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
            ]
        > {
    let summary = "RoPE VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input_cos,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input_sin,
        UnitAttr:$is_interleaved,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$input_cos,
            "::mlir::Value":$input_sin,
            "::mlir::UnitAttr":$is_interleaved
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_NonMaxSuppressionOp :
        VPU_LayerOp<
            "NonMaxSuppression",
            [
                DeclareOpInterfaceMethods<VPU_AuxiliaryBufferOpInterface>
            ]
        > {
    let summary = "NonMaxSuppression VPU layer";

    let arguments = (ins
        3DTensorOf<[F16, F32]>:$in_box_coords,
        3DTensorOf<[F16, F32]>:$in_box_scores,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$dataBuffer,

        IE_BoxEncodingTypeAttr:$box_encoding,
        UnitAttr:$sort_result_descending,

        OptionalAttr<IntAttr>:$max_output_boxes_per_class_value,
        OptionalAttr<F64Attr>:$iou_threshold_value,
        OptionalAttr<F64Attr>:$score_threshold_value,
        OptionalAttr<F64Attr>:$soft_nms_sigma_value
    );

    let results = (outs
        2DTensorOf<[SI32]>:$out_selected_indices,
        2DTensorOf<[F16, F32]>:$out_selected_scores,
        1DTensorOf<[SI32]>:$out_valid_outputs
    );
}


def VPU_NonZeroOp :
        VPU_LayerOp<
            "NonZero",
            [
                VPU_BoundsRepresentationInterface
            ]
        > {
    let summary = "NonZero VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, SI32, Bool8]>:$input,

        DefaultValuedAttr<VPU_BoundsRepresentationAttr, "vpux::VPU::BoundsRepresentation::BOUNDS">:$bounds_representation
    );

    let results = (outs
        2DTensorOf<[SI32]>:$output
    );

    let hasVerifier = 1;
}


def VPU_OneHotOp :
        VPU_LayerOp<
            "OneHot"
        > {
    let summary = "InferenceEngine OneHot layer";

    let arguments = (ins
        RankedTensorOf<[SI32, SI64]> :$input,

        IntAttr:$depth,
        F64Attr:$on_value,
        F64Attr:$off_value,
        IntAttr:$axis,

        TypeAttr:$outputType
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_ProposalOp :
        VPU_LayerOp<
            "Proposal",
            [
                DeclareOpInterfaceMethods<VPU_AuxiliaryBufferOpInterface>
            ]
        > {
    let summary = "Proposal VPU layer";

    let description = [{
        Proposal operation filters bounding boxes and outputs only those with the highest prediction confidence.
        The auxiliary buffer has the role of storing the intermediate results obtained inside the operation,
        then sorting them. Depending on some criteria, it recalculates the results and extracts the output from them.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$class_probs,
        RankedTensorOf<[F16, F32]>:$bbox_deltas,
        RankedTensorOf<[F16, F32]>:$image_shape,
        Optional<1DTensorOf<[UI8]>>:$auxiliary,

        IE_ProposalAttr:$proposal_attrs
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output,
        RankedTensorOf<[F16, F32]>:$probs
    );

}


def VPU_RandomUniformOp :
        VPU_LayerOp<
            "RandomUniform",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "RandomUniform VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$min,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$max,

        I64ArrayAttr:$output_shape,
        TypeAttr:$outputType,
        IntAttr:$global_seed,
        IntAttr:$op_seed,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$min,
            "::mlir::Value":$max,

            "::mlir::ArrayAttr":$output_shape,
            "::mlir::TypeAttr":$outputType,
            "::mlir::IntegerAttr":$global_seed,
            "::mlir::IntegerAttr":$op_seed
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_RangeOp :
        VPU_LayerOp<
            "Range",
            [
                VPU_BoundsRepresentationInterface
            ]
        > {
    let summary = "Range VPU layer";

    let arguments = (ins
        1DTensorOf<[AnyInteger, AnyFloat]>:$start,
        1DTensorOf<[AnyInteger, AnyFloat]>:$stop,
        1DTensorOf<[AnyInteger, AnyFloat]>:$step,
        TypeAttr:$dstElemType,

        DefaultValuedAttr<VPU_BoundsRepresentationAttr, "vpux::VPU::BoundsRepresentation::BOUNDS">:$bounds_representation
    );

    let results = (outs
        1DTensorOf<[AnyInteger, AnyFloat]>:$output
    );

    let hasVerifier = 1;
}




def VPU_RegionYoloOp :
        VPU_LayerOp<
            "RegionYolo"
        > {
    let summary = "RegionYolo VPU layer";

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


def VPU_ReorgYoloOp :
        VPU_LayerOp<
            "ReorgYolo"
        > {
    let summary = "ReorgYolo VPU layer";

    let arguments = (ins
        4DTensorOf<[AnyInteger, AnyFloat]>:$input,

        IntAttr:$stride
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
}

def VPU_SelectOp :
        VPU_LayerOp<
            "Select",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
            ]
        > {
    let summary = "Select VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[Bool8, SI32, F16]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[SI32, F16]>, VPU_DistributedTensor]>:$input2,
        AnyTypeOf<[RankedTensorOf<[SI32, F16]>, VPU_DistributedTensor]>:$input3,
        IE_AutoBroadcastTypeAttr:$auto_broadcast,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[SI32, F16]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "::mlir::Value":$input3,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_StubOp :
        VPU_Op<
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


def VPU_TopKOp :
        VPU_LayerOp<
            "TopK",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_AuxiliaryBufferOpInterface>,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "TopK VPU layer";

    let description = [{
        * lineBuffer - it is an auxiliary buffer, which has the role of storing
        some intermediate results in the software kernel of the operation.
    }];

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$k,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$lineBuffer,
        OptionalAttr<IntAttr>:$k_value,

        IntAttr:$axis,
        IE_TopKModeAttr:$mode,
        IE_TopKSortTypeAttr:$sort,
        TypeAttr:$element_type,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output_values,
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$target_shape
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
        OutputTiling getOutputTiling(const vpux::TileInfo& outputTile, vpux::Logger log);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_PermuteCastOp :
        VPU_LayerOp<
            "PermuteCast",
            [
                VPU_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_DistributedCastOpInterface>,
                DeclareOpInterfaceMethods<VPU_TilingViewLikeOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "PermuteCast VPU layer";

    let description = [{
        The op changes layout information in the following way:
            * dst_order: layout attribute of result is set to value of this arg
            * mem_perm: describes the permutation applied on the input value's memory shape
                        to obtain the memory shape of the output value.
    }];

    let extraClassDeclaration = [{
        bool isVFSupported();
    }] # baseExtraClassDeclaration;

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

def VPU_PermuteQuantizeOp :
        VPU_LayerOp<
            "PermuteQuantize",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "PermuteQuantize VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm,
        TypeAttr:$dstElemType,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end
    );

    let results = (outs
        RankedTensorOf<[quant_QuantizedType]>:$output
    );
}

def VPU_LayoutCastOp :
        VPU_LayerOp<
            "LayoutCast",
            [
                VPU_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_DistributedCastOpInterface>,
                DeclareOpInterfaceMethods<VPU_TilingViewLikeOpInterface>
            ]
        > {
    let summary = "This layer overrides layout of a given tensor.";

    let extraClassDeclaration = [{
        bool isVFSupported();
    }] # baseExtraClassDeclaration;

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        AffineMapAttr:$dst_order
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
    let hasVerifier = 1;
}

def VPU_FlashSDPAOp :
        VPU_LayerOp<
            "FlashSDPA",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "Implements SDPA computation using FlashAttention-2 algorithm";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, F64]>, VPU_DistributedTensor]>:$query,
        AnyTypeOf<[RankedTensorOf<[F16, F32, F64]>, VPU_DistributedTensor]>:$key,
        AnyTypeOf<[RankedTensorOf<[F16, F32, F64]>, VPU_DistributedTensor]>:$value,

        AnyTypeOf<[RankedTensorOf<[F16, F32, F64]>, VPU_DistributedTensor]>:$aux_buffer,

        AnyTypeOf<[RankedTensorOf<[F16, F32, F64]>, VPU_DistributedTensor]>:$input_running_output,
        AnyTypeOf<[RankedTensorOf<[F16, F32, F64]>, VPU_DistributedTensor]>:$input_running_max,
        AnyTypeOf<[RankedTensorOf<[F16, F32, F64]>, VPU_DistributedTensor]>:$input_running_sum,

        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32, F64]>, VPU_DistributedTensor]>>:$attention_mask,
        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32, F64]>, VPU_DistributedTensor]>>:$scale,

        BoolAttr:$is_head,
        BoolAttr:$is_tail,

        OptionalAttr<IntAttr>:$kv_num_blocks,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$query,
            "::mlir::Value":$key,
            "::mlir::Value":$value,
            "::mlir::Value":$input_running_output,
            "::mlir::Value":$input_running_max,
            "::mlir::Value":$input_running_sum,
            "::mlir::Value":$attention_mask,
            "::mlir::Value":$scale
        )>,

        OpBuilder<(ins
            "::mlir::Value":$query,
            "::mlir::Value":$key,
            "::mlir::Value":$value,
            "::mlir::Value":$input_running_output,
            "::mlir::Value":$input_running_max,
            "::mlir::Value":$input_running_sum,
            "::mlir::Value":$attention_mask,
            "::mlir::Value":$scale,
            "::mlir::BoolAttr":$is_head,
            "::mlir::BoolAttr":$is_tail,
            "::mlir::IntegerAttr":$kv_num_blocks,
            "vpux::VPU::MultiClusterStrategyAttr":$multi_cluster_strategy
        )>
    ];

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, F64]>, VPU_DistributedTensor]>:$result_running_output,
        AnyTypeOf<[RankedTensorOf<[F16, F32, F64]>, VPU_DistributedTensor]>:$result_running_max,
        AnyTypeOf<[RankedTensorOf<[F16, F32, F64]>, VPU_DistributedTensor]>:$result_running_sum,

        // This output shares the same buffer as the input query
        // and is used to preserve query tensor in CMX for multiple kernel invocations
        AnyTypeOf<[RankedTensorOf<[F16, F32, F64]>, VPU_DistributedTensor]>:$result_query
    );

    let extraClassDeclaration = [{
        OutputTiling getOutputTiling(const vpux::TileInfo& outputTile, vpux::Logger log);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);

        bool fitIntoCMXAfterKeyValueTiling(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem, int64_t kvNumBlocks);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

#endif
