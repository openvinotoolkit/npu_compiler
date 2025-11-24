//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_IE_ATTRIBUTES

include "mlir/IR/OpBase.td"
include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/IE/dialect.td"
include "vpux/compiler/dialect/IE/attr_interfaces.td"
include "vpux/compiler/dialect/core/type_interfaces.td"

//
// Base classes
//

class IE_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 0;
}

class IE_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<IE_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

class IE_Attr<string name, list<Trait> traits = []> :
        AttrDef<IE_Dialect, name, traits> {
    let mnemonic = name;
    let assemblyFormat = "`<` struct(params) `>`";
}

//
// TypeComparisonMode
//

defvar IE_STRICT_EQUAL = "STRICT_EQUAL";
defvar IE_ALLOW_MIXED_PRECISION = "ALLOW_MIXED_PRECISION";
defvar IE_ALLOW_DIFFERENT_QUANT = "ALLOW_DIFFERENT_QUANT";
defvar IE_ALLOW_GROUPED_OUTPUT = "ALLOW_GROUPED_OUTPUT";
defvar IE_ALLOW_DISTRIBUTED_OUTPUT = "ALLOW_DISTRIBUTED_OUTPUT";
defvar IE_TypeComparisonMode_STRICT_EQUAL = !strconcat("vpux::IE::TypeComparisonMode::", IE_STRICT_EQUAL);
defvar IE_TypeComparisonMode_ALLOW_MIXED_PRECISION = !strconcat("vpux::IE::TypeComparisonMode::", IE_ALLOW_MIXED_PRECISION);
defvar IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT = !strconcat("vpux::IE::TypeComparisonMode::", IE_ALLOW_DIFFERENT_QUANT);
defvar IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT = !strconcat("vpux::IE::TypeComparisonMode::", IE_ALLOW_GROUPED_OUTPUT);
defvar IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT = !strconcat("vpux::IE::TypeComparisonMode::", IE_ALLOW_DISTRIBUTED_OUTPUT);

def IE_TypeComparisonMode :
        I64BitEnumAttr<
            "TypeComparisonMode",
            "Tensor element type comparison modes",
            [
                I64BitEnumAttrCase<IE_STRICT_EQUAL, 0x1>,
                I64BitEnumAttrCase<IE_ALLOW_MIXED_PRECISION, 0x2>,
                I64BitEnumAttrCase<IE_ALLOW_DIFFERENT_QUANT, 0x4>,
                I64BitEnumAttrCase<IE_ALLOW_GROUPED_OUTPUT, 0x8>,
                I64BitEnumAttrCase<IE_ALLOW_DISTRIBUTED_OUTPUT, 0x10>
            ]
        > {
    let cppNamespace = "vpux::IE";
    let genSpecializedAttr = 1;
}

//
// EpsMode
//

def IE_EpsMode :
        IE_I64EnumAttr<
            "EpsMode",
            "EpsMode that the InferenceEngine supports",
            [
                I64EnumAttrCase<"ADD", 0>,
                I64EnumAttrCase<"MAX", 1>,
            ]
        > {
}

def IE_EpsModeAttr : IE_EnumAttr<IE_EpsMode, "eps_mode">;

//
// MvnEpsMode
//

def IE_MvnEpsMode :
        IE_I64EnumAttr<
            "MvnEpsMode",
            "MvnEpsMode that the InferenceEngine supports",
            [
                I64EnumAttrCase<"INSIDE_SQRT", 0>,
                I64EnumAttrCase<"OUTSIDE_SQRT", 1>,
            ]
        > {
}

def IE_MvnEpsModeAttr : IE_EnumAttr<IE_MvnEpsMode, "mvn_eps_mode">;

//
// TopKMode
//

def IE_TopKMode :
        IE_I64EnumAttr<
            "TopKMode",
            "TopKMode that the InferenceEngine supports",
            [
                I64EnumAttrCase<"MAX", 0>,
                I64EnumAttrCase<"MIN", 1>,
            ]
        > {
}

def IE_TopKModeAttr : IE_EnumAttr<IE_TopKMode, "topk_mode">;

//
// TopKSortType
//

def IE_TopKSortType :
        IE_I64EnumAttr<
            "TopKSortType",
            "TopKSortType that the InferenceEngine supports",
            [
                I64EnumAttrCase<"NONE", 0>,
                I64EnumAttrCase<"SORT_VALUES", 1>,
                I64EnumAttrCase<"SORT_INDICES", 2>,
            ]
        > {
}

def IE_TopKSortTypeAttr : IE_EnumAttr<IE_TopKSortType, "topk_sort_type">;

//
// GridSampleMode
//

def IE_GridSampleMode :
        IE_I64EnumAttr<
            "GridSampleMode",
            "GridSampleMode that the InferenceEngine supports",
            [
                I64EnumAttrCase<"BILINEAR", 0>,
                I64EnumAttrCase<"BICUBIC", 1>,
                I64EnumAttrCase<"NEAREST", 2>,
            ]
        > {
}

def IE_GridSampleModeAttr : IE_EnumAttr<IE_GridSampleMode, "grid_sample_mode">;

//
// GridSamplePaddingMode
//

def IE_GridSamplePaddingMode :
        IE_I64EnumAttr<
            "GridSamplePaddingMode",
            "GridSamplePaddingMode that the InferenceEngine supports",
            [
                I64EnumAttrCase<"ZEROS", 0>,
                I64EnumAttrCase<"BORDER", 1>,
                I64EnumAttrCase<"REFLECTION", 2>,
            ]
        > {
}

def IE_GridSamplePaddingModeAttr : IE_EnumAttr<IE_GridSamplePaddingMode, "grid_sample_padding_mode">;

//
// ROIPoolingMethod
//

def IE_ROIPoolingMethod :
        IE_I64EnumAttr<
            "ROIPoolingMethod",
            "ROIPoolingMethod that the InferenceEngine supports",
            [
                I64EnumAttrCase<"MAX", 0>,
                I64EnumAttrCase<"BILINEAR", 1>,
            ]
        > {
}

def IE_ROIPoolingMethodAttr : IE_EnumAttr<IE_ROIPoolingMethod, "roi_pooling_method">;

//
// ROIAlignMethod
//

def IE_ROIAlignMethod :
        IE_I64EnumAttr<
            "ROIAlignMethod",
            "ROIAlignMethod that the InferenceEngine supports",
            [
                I64EnumAttrCase<"AVG", 0>,
                I64EnumAttrCase<"MAX", 1>,
            ]
        > {
}

def IE_ROIAlignMethodAttr : IE_EnumAttr<IE_ROIAlignMethod, "roi_align_method">;

//
// ROIAlignAlignedMethod
//

def IE_ROIAlignAlignedMethod :
        IE_I64EnumAttr<
            "ROIAlignAlignedMethod",
            "ROIAlignAlignedMethod that the InferenceEngine supports",
            [
                I64EnumAttrCase<"ASYMMETRIC", 0>,
                I64EnumAttrCase<"HALF_PIXEL_FOR_NN", 1>,
                I64EnumAttrCase<"HALF_PIXEL", 2>,
            ]
        > {
}

def IE_ROIAlignAlignedMethodAttr : IE_EnumAttr<IE_ROIAlignAlignedMethod, "roi_align_aligned_method">;

//
// AutoBroadcastType
//

def IE_AutoBroadcastType :
        IE_I64EnumAttr<
            "AutoBroadcastType",
            "Specifies rules used for auto-broadcasting of input tensors",
            [
                I64EnumAttrCase<"NONE_OR_EXPLICIT", 0>,
                I64EnumAttrCase<"NUMPY", 1>,
                I64EnumAttrCase<"PDPD", 2>,
            ]
        > {
}

def IE_AutoBroadcastTypeAttr : IE_EnumAttr<IE_AutoBroadcastType, "auto_broadcast_type">;

//
// BroadcastType
//

def IE_BroadcastType :
        IE_I64EnumAttr<
            "BroadcastType",
            "Broadcast type that operations support",
            [
                I64EnumAttrCase<"NUMPY", 0>,
                I64EnumAttrCase<"EXPLICIT", 1>,
                I64EnumAttrCase<"BIDIRECTIONAL", 2>,
            ]
        > {
}

def IE_BroadcastTypeAttr : IE_EnumAttr<IE_BroadcastType, "broadcast_type">;

//
// RoundingType
//

def IE_RoundingType :
        IE_I64EnumAttr<
            "RoundingType",
            "Rounding type that operations support",
            [
                I64EnumAttrCase<"FLOOR", 0>,
                I64EnumAttrCase<"CEIL", 1>,
            ]
        > {
}

def IE_RoundingTypeAttr : IE_EnumAttr<IE_RoundingType, "rounding_type">;

//
// LRN_IERegion
//

def IE_LRN_IERegion :
        IE_I64EnumAttr<
            "LRN_IERegion",
            "LRN_IE region that operations support",
            [
                I64EnumAttrCase<"SAME", 0>,
                I64EnumAttrCase<"ACROSS", 1>
            ]
        > {
}

def IE_LRN_IERegionAttr : IE_EnumAttr<IE_LRN_IERegion, "lrn_ieregion">;

//
// Proposal
//

def IE_ProposalAttr : IE_Attr<"Proposal"> {
    let parameters = (ins
        "mlir::IntegerAttr":$baseSize,
        "mlir::IntegerAttr":$preNmsTopN,
        "mlir::IntegerAttr":$postNmsTopN,
        "mlir::FloatAttr":$nmsThresh,
        "mlir::IntegerAttr":$featStride,
        "mlir::IntegerAttr":$minSize,
        "mlir::ArrayAttr":$ratio,
        "mlir::ArrayAttr":$scale,
        "mlir::BoolAttr":$clipBeforeNms,
        "mlir::BoolAttr":$clipAfterNms,
        "mlir::BoolAttr":$normalize,
        "mlir::FloatAttr":$boxSizeScale,
        "mlir::FloatAttr":$boxCoordinateScale,
        "mlir::StringAttr":$framework,
        "mlir::BoolAttr":$inferProbs
    );
}

//
// ReverseMode
//

def IE_ReverseMode :
        IE_I64EnumAttr<
            "ReverseMode",
            "ReverseMode specifies how the second input tensor should be interpreted",
            [
                I64EnumAttrCase<"INDEX", 0>,
                I64EnumAttrCase<"MASK", 1>,
            ]
        > {
}

def IE_ReverseModeAttr : IE_EnumAttr<IE_ReverseMode, "reverse_mode">;

//
// Interpolate
//

def IE_InterpolateMode :
        IE_I64EnumAttr<
            "InterpolateMode",
            "Specifies type of interpolation",
            [
                I64EnumAttrCase<"NEAREST", 0>,
                I64EnumAttrCase<"LINEAR", 1>,
                I64EnumAttrCase<"LINEAR_ONNX", 2>,
                I64EnumAttrCase<"CUBIC", 3>,
            ]
        > {
}

def IE_InterpolateModeAttr : IE_EnumAttr<IE_InterpolateMode, "interpolate_mode">;

def IE_InterpolateCalcMode :
        IE_I64EnumAttr<
            "InterpolateCalcMode",
            "Specifies which input, sizes or scales, is used to calculate an output shape.",
            [
                I64EnumAttrCase<"SIZES", 0>,
                I64EnumAttrCase<"SCALES", 1>,
            ]
        > {
}

def IE_InterpolateCalcModeAttr : IE_EnumAttr<IE_InterpolateCalcMode, "interpolate_calc_mode">;

def IE_InterpolateCoordMode :
        IE_I64EnumAttr<
            "InterpolateCoordMode",
            "coordinate_transformation_mode specifies how to transform the coordinate.",
            [
                I64EnumAttrCase<"HALF_PIXEL", 0>,
                I64EnumAttrCase<"PYTORCH_HALF_PIXEL", 1>,
                I64EnumAttrCase<"ASYMMETRIC", 2>,
                I64EnumAttrCase<"TF_HALF_PIXEL_FOR_NN", 3>,
                I64EnumAttrCase<"ALIGN_CORNERS", 4>,
            ]
        > {
}

def IE_InterpolateCoordModeAttr : IE_EnumAttr<IE_InterpolateCoordMode, "interpolate_coord_mode">;

def IE_InterpolateNearestMode :
        IE_I64EnumAttr<
            "InterpolateNearestMode",
            "specifies round mode when mode == nearest",
            [
                I64EnumAttrCase<"ROUND_PREFER_FLOOR", 0>,
                I64EnumAttrCase<"ROUND_PREFER_CEIL", 1>,
                I64EnumAttrCase<"FLOOR", 2>,
                I64EnumAttrCase<"CEIL", 3>,
                I64EnumAttrCase<"SIMPLE", 4>,
            ]
        > {
}

def IE_InterpolateNearestModeAttr : IE_EnumAttr<IE_InterpolateNearestMode, "interpolate_nearest_mode">;

def IE_InterpolateAttr : IE_Attr<"Interpolate"> {
    let parameters = (ins
        "vpux::IE::InterpolateModeAttr":$mode,
        "vpux::IE::InterpolateCalcModeAttr":$shape_calc_mode,
        "vpux::IE::InterpolateCoordModeAttr":$coord_mode,
        "vpux::IE::InterpolateNearestModeAttr":$nearest_mode,
        "mlir::BoolAttr":$antialias,
        "mlir::ArrayAttr":$pads_begin,
        "mlir::ArrayAttr":$pads_end,
        "mlir::FloatAttr":$cube_coeff
    );
}

//
// DetectionOutputCodeType
//

def IE_DetectionOutputCodeType :
        IE_I64EnumAttr<
            "DetectionOutputCodeType",
            "DetectionOutput parameter that specifies bounding box decoding algorithm",
            [
                I64EnumAttrCase<"CENTER_SIZE", 0>,
                I64EnumAttrCase<"CORNER", 1>,
                I64EnumAttrCase<"CORNER_SIZE", 2>,
            ]
        > {
}

def IE_DetectionOutputCodeTypeAttr : IE_EnumAttr<IE_DetectionOutputCodeType, "detection_output_code_type">;

//
// DetectionOutputAttr
//

def IE_DetectionOutputAttr : IE_Attr<"DetectionOutput"> {
    let parameters = (ins
        "mlir::IntegerAttr":$num_classes,
        "mlir::IntegerAttr":$background_label_id,
        "mlir::IntegerAttr":$top_k,
        "mlir::BoolAttr":$variance_encoded_in_target,
        "mlir::ArrayAttr":$keep_top_k,
        "vpux::IE::DetectionOutputCodeTypeAttr":$code_type,
        "mlir::BoolAttr":$share_location,
        "mlir::FloatAttr":$nms_threshold,
        "mlir::FloatAttr":$confidence_threshold,
        "mlir::BoolAttr":$clip_after_nms,
        "mlir::BoolAttr":$clip_before_nms,
        "mlir::BoolAttr":$decrease_label_id,
        "mlir::BoolAttr":$normalized,
        "mlir::IntegerAttr":$input_height,
        "mlir::IntegerAttr":$input_width,
        "mlir::FloatAttr":$objectness_score
    );
}

//
// ExperimentalDetectronROIFeatureExtractorAttr
//

def IE_ExperimentalDetectronROIFeatureExtractorAttr : IE_Attr<"ExperimentalDetectronROIFeatureExtractor"> {
    let parameters = (ins
        "mlir::IntegerAttr":$output_size,
        "mlir::IntegerAttr":$sampling_ratio,
        "mlir::BoolAttr":$aligned,
        "mlir::ArrayAttr":$pyramid_scales
    );
}

//
// PadMode
//

def IE_PadMode :
        IE_I64EnumAttr<
            "PadMode",
            "TPadMode that the InferenceEngine supports",
            [
                I64EnumAttrCase<"CONSTANT", 0>,
                I64EnumAttrCase<"EDGE", 1>,
                I64EnumAttrCase<"REFLECT", 2>,
                I64EnumAttrCase<"SYMMETRIC", 3>,
            ]
        > {
}

def IE_PadModeAttr : IE_EnumAttr<IE_PadMode, "pad_mode">;

//
// RoundMode
//

def IE_RoundMode :
        IE_I64EnumAttr<
            "RoundMode",
            "RoundMode that the InferenceEngine supports",
            [
                I64EnumAttrCase<"HALF_TO_EVEN", 0>,
                I64EnumAttrCase<"HALF_AWAY_FROM_ZERO", 1>,
            ]
        > {
}

def IE_RoundModeAttr : IE_EnumAttr<IE_RoundMode, "round_mode">;

//
// RNNSequenceDirection
//

def IE_RNNSequenceDirection :
        IE_I64EnumAttr<
            "RNNSequenceDirection",
            "RNNSequenceDirection that the InferenceEngine supports",
            [
                I64EnumAttrCase<"FORWARD", 0>,
                I64EnumAttrCase<"REVERSE", 1>,
                I64EnumAttrCase<"BIDIRECTIONAL", 2>,
            ]
        > {
}

def IE_RNNSequenceDirectionAttr : IE_EnumAttr<IE_RNNSequenceDirection, "rnn_seq_direction">;

//
// PostOp
//

class IE_PostOpAttrBase<Pred condition, string summary> : Attr<condition, summary> {
    let storageType = [{ vpux::IE::PostOpAttr }];
    let returnType = [{ vpux::IE::PostOpAttr }];
    let convertFromStorage = "$_self";
}

def IE_PostOpAttr : IE_PostOpAttrBase<CPred<"mlir::isa<vpux::IE::PostOpAttr>($_self)">,
                                      "Post-Op Attribute"> {
    string cppType = "vpux::IE::PostOpAttr";
}

def IE_ReluAttr : IE_Attr<"Relu", [
        DeclareAttrInterfaceMethods<IE_PostOpAttrInterface>,
        IE_ChannelAgnosticAttr
    ] > {
}

def IE_ClampAttr : IE_Attr<"Clamp", [
        DeclareAttrInterfaceMethods<IE_PostOpAttrInterface>,
        IE_ChannelAgnosticAttr
    ] > {
    let parameters = (ins
        "mlir::FloatAttr":$min,
        "mlir::FloatAttr":$max
    );
}

def IE_LeakyReluAttr : IE_Attr<"LeakyRelu", [
        DeclareAttrInterfaceMethods<IE_PostOpAttrInterface>,
        IE_ChannelAgnosticAttr
    ] > {
    let parameters = (ins
        "mlir::FloatAttr":$negative_slope
    );
}

def IE_PReluAttr : IE_Attr<"PRelu", [
        DeclareAttrInterfaceMethods<IE_PostOpAttrInterface>
    ] > {
    let parameters = (ins
        "mlir::ArrayAttr":$negative_slope
    );
}

def IE_TanhAttr : IE_Attr<"Tanh", [
        DeclareAttrInterfaceMethods<IE_PostOpAttrInterface>,
        IE_ChannelAgnosticAttr
    ] > {
}

def IE_SigmoidAttr : IE_Attr<"Sigmoid", [
        DeclareAttrInterfaceMethods<IE_PostOpAttrInterface>,
        IE_ChannelAgnosticAttr
    ] > {
}

def IE_SwishAttr : IE_Attr<"Swish", [
        DeclareAttrInterfaceMethods<IE_PostOpAttrInterface>,
        IE_ChannelAgnosticAttr
    ] > {
    let parameters = (ins
        "mlir::FloatAttr":$beta
    );
}

def IE_GeluAttr : IE_Attr<"Gelu", [
        DeclareAttrInterfaceMethods<IE_PostOpAttrInterface>,
        IE_ChannelAgnosticAttr
    ] > {
}

def IE_ExpAttr : IE_Attr<"Exp", [
        DeclareAttrInterfaceMethods<IE_PostOpAttrInterface>,
        IE_ChannelAgnosticAttr
    ] > {
}

//
// ConcatAttr
//

def IE_ConcatAttr : IE_Attr<"Concat"> {
    let parameters = (ins
        "mlir::IntegerAttr":$axis,
        OptionalParameter<"mlir::IntegerAttr">:$offset,
        OptionalParameter<"mlir::IntegerAttr">:$stride
    );
}

//
// IE_DepthToSpaceMode
//

def IE_DepthToSpaceMode :
        IE_I64EnumAttr<
            "DepthToSpaceMode",
            "DepthToSpaceMode that the InferenceEngine supports",
            [
                I64EnumAttrCase<"BLOCKS_FIRST", 0>,
                I64EnumAttrCase<"DEPTH_FIRST", 1>,
            ]
        > {
}

def IE_DepthToSpaceModeAttr : IE_EnumAttr<IE_DepthToSpaceMode, "depth_to_space_mode">;

//
// IE_SpaceToDepthMode
//

def IE_SpaceToDepthMode :
        IE_I64EnumAttr<
            "SpaceToDepthMode",
            "SpaceToDepthMode that the InferenceEngine supports",
            [
                I64EnumAttrCase<"BLOCKS_FIRST", 0>,
                I64EnumAttrCase<"DEPTH_FIRST", 1>,
            ]
        > {
}

def IE_SpaceToDepthModeAttr : IE_EnumAttr<IE_SpaceToDepthMode, "space_to_depth_mode">;

//
// ChannelPaddingAttr
//

def IE_ChannelPaddingAttr : IE_Attr<"ChannelPadding"> {
    let parameters = (ins
        "mlir::IntegerAttr":$input,
        "mlir::IntegerAttr":$output
    );
}

//
// Color Formats
//

def IE_ColorFmt :
        IE_I64EnumAttr<
            "ColorFmt",
            "YUV, RGB color formats",
            [
                I64EnumAttrCase<"NV12", 0>,
                I64EnumAttrCase<"I420", 1>,
                I64EnumAttrCase<"RGB", 2>,
                I64EnumAttrCase<"BGR", 3>
            ]
        > {
    }

def IE_ColorFmtAttr : IE_EnumAttr<IE_ColorFmt, "color_fmt">;

//
// BoxEncodingType
//

def IE_BoxEncodingType :
        IE_I64EnumAttr<
            "BoxEncodingType",
            "BoxEncodingType that the InferenceEngine supports",
            [
                I64EnumAttrCase<"CORNER", 0>,
                I64EnumAttrCase<"CENTER", 1>,
            ]
        > {
}

def IE_BoxEncodingTypeAttr : IE_EnumAttr<IE_BoxEncodingType, "box_encoding_type">;

//
// PSROIPoolingMode
//

def IE_PSROIPoolingMode :
        IE_I64EnumAttr<
            "PSROIPoolingMode",
            "PSROIPoolingMode that the InferenceEngine supports",
            [
                I64EnumAttrCase<"AVERAGE", 0>,
                I64EnumAttrCase<"BILINEAR", 1>,
            ]
        > {
}

def IE_PSROIPoolingModeAttr : IE_EnumAttr<IE_PSROIPoolingMode, "psroi_pooling_mode">;

//
// PadType
//

def IE_PadType :
        IE_I64EnumAttr<
            "PadType",
            "PadType that the InferenceEngine supports",
            [
                I64EnumAttrCase<"SAME_LOWER", 0>,
                I64EnumAttrCase<"SAME_UPPER", 1>,
                I64EnumAttrCase<"VALID", 2>,
            ]
        > {
}

def IE_PadTypeAttr : IE_EnumAttr<IE_PadType, "pad_type">;

//
// DeformablePSROIPoolingMode
//

def IE_DeformablePSROIPoolingMode :
        IE_I64EnumAttr<
            "DeformablePSROIPoolingMode",
            "DeformablePSROIPoolingMode that the InferenceEngine supports",
            [
                I64EnumAttrCase<"AVERAGE", 0>,
                I64EnumAttrCase<"BILINEAR_DEFORMABLE", 1>,
            ]
        > {
}

def IE_DeformablePSROIPoolingModeAttr : IE_EnumAttr<IE_DeformablePSROIPoolingMode, "deformable_psroi_pooling_mode">;

//
// UpsamplingPadAttr
//

def IE_UpsamplingPadAttr : IE_Attr<"UpsamplingPad"> {
    let parameters = (ins
        "mlir::ArrayAttr":$pads_channel,
        "mlir::ArrayAttr":$pads_height,
        "mlir::ArrayAttr":$pads_width
    );
}

//
// ScatterElementsUpdateReductionType
//

def IE_ScatterElementsUpdateReductionType :
        IE_I64EnumAttr<
            "ScatterElementsUpdateReductionType",
            "ScatterElementsUpdateReductionType that the InferenceEngine supports",
            [
                I64EnumAttrCase<"NONE", 0>,
                I64EnumAttrCase<"SUM", 1>,
                I64EnumAttrCase<"PROD", 2>,
                I64EnumAttrCase<"MIN", 3>,
                I64EnumAttrCase<"MAX", 4>,
                I64EnumAttrCase<"MEAN", 5>
            ]
        > {
}

def IE_ScatterElementsUpdateReductionTypeAttr : IE_EnumAttr<IE_ScatterElementsUpdateReductionType, "scatter_elements_update_reduction_type">;

//
// PortMapAttr
//

def IE_SliceInputPortMapAttr : IE_Attr<"SliceInputPortMap"> {
    let description = [{
    This is the port map attribute for tensor iterator op.
        external_port_id represents I/O layers'id ousides of tensor iterator op.
        internal_layer_id represents I/O layers'id insides of tensor iterator body.
        axis, start, stride, end, part_size are valid only for cases where input needs slicing.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$external_port_id,
        "mlir::IntegerAttr":$internal_layer_id,
        "mlir::IntegerAttr":$axis,
        "mlir::IntegerAttr":$start,
        "mlir::IntegerAttr":$stride,
        "mlir::IntegerAttr":$part_size,
        "mlir::IntegerAttr":$end
    );
}

def IE_InvariantInputPortMapAttr : IE_Attr<"InvariantInputPortMap"> {
    let description = [{
    This is the port map attribute for tensor iterator op.
        external_port_id represents I/O layers'id ousides of tensor iterator op.
        internal_layer_id represents I/O layers'id insides of tensor iterator body.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$external_port_id,
        "mlir::IntegerAttr":$internal_layer_id
    );
}

def IE_MergedInputPortMapAttr : IE_Attr<"MergedInputPortMap"> {
    let description = [{
    This is the port map attribute for tensor iterator op.
        external_port_id represents I/O layers'id ousides of tensor iterator op.
        internal_layer_id represents I/O layers'id insides of tensor iterator body.
        body_input_index is valid only for cases where body output is the body param in the next iteration.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$external_port_id,
        "mlir::IntegerAttr":$internal_layer_id,
        "mlir::IntegerAttr":$body_input_index
    );
}

def IE_ConcatOutputPortMapAttr : IE_Attr<"ConcatOutputPortMap"> {
    let description = [{
    This is the port map attribute for tensor iterator op.
        external_port_id represents I/O layers'id ousides of tensor iterator op.
        internal_layer_id represents I/O layers'id insides of tensor iterator body.
        axis, start, stride, end, part_size are valid only for cases where output need concating.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$external_port_id,
        "mlir::IntegerAttr":$internal_layer_id,
        "mlir::IntegerAttr":$axis,
        "mlir::IntegerAttr":$start,
        "mlir::IntegerAttr":$stride,
        "mlir::IntegerAttr":$part_size,
        "mlir::IntegerAttr":$end
    );
}

def IE_InvariantOutputPortMapAttr : IE_Attr<"InvariantOutputPortMap"> {
    let description = [{
    This is the port map attribute for tensor iterator op.
        external_port_id represents I/O layers'id ousides of tensor iterator op.
        internal_layer_id represents I/O layers'id insides of tensor iterator body.
        iterations has default value -1 means pick the last iteration result, or iterations = i it will pick the i-th result.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$external_port_id,
        "mlir::IntegerAttr":$internal_layer_id,
        "mlir::IntegerAttr":$iterations
    );
}

#endif
