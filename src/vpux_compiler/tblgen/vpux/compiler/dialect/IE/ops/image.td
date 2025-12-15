//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_IMAGE
#define VPUX_COMPILER_DIALECT_IE_OPS_IMAGE

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_DeformablePSROIPoolingOp :
        IE_LayerOp<
            "DeformablePSROIPooling"
        > {
    let summary = "InferenceEngine DeformablePSROIPooling layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$input_score_maps,
        2DTensorOf<[AnyFloat]>:$input_rois,
        Optional<4DTensorOf<[AnyFloat]>>:$input_transformations,

        IntAttr:$output_dim,
        F64Attr:$spatial_scale,
        OptionalAttr<IntAttr>:$group_size,
        OptionalAttr<IntAttr>:$spatial_bins_x,
        OptionalAttr<IntAttr>:$spatial_bins_y,
        OptionalAttr<F64Attr>:$trans_std,
        OptionalAttr<IntAttr>:$part_size,
        OptionalAttr<IE_DeformablePSROIPoolingModeAttr>:$mode
    );

    let results = (outs
        4DTensorOf<[AnyFloat]>:$output
    );
}

def IE_ExperimentalDetectronROIFeatureExtractorOp :
        IE_LayerOp<
            "ExperimentalDetectronROIFeatureExtractor"
        > {
    let summary = "InferenceEngine ExperimentalDetectronROIFeatureExtractor layer";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,

        IE_ExperimentalDetectronROIFeatureExtractorAttr:$attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output,
        RankedTensorOf<[F16, F32, F64]>:$outputROIs
    );
}

def IE_GridSampleOp :
        IE_LayerOp<
            "GridSample"
        > {
    let summary = "InferenceEngine GridSample layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$grid,

        UnitAttr:$align_corners,
        OptionalAttr<IE_GridSampleModeAttr>:$mode,
        OptionalAttr<IE_GridSamplePaddingModeAttr>:$padding_mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_InterpolateOp :
        IE_LayerOp<
            "Interpolate",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine Interpolate layer";

    let arguments = (ins
        RankedTensorOf<[UI8, F16, F32, F64, quant_QuantizedType]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$sizes,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$scales,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$sizes_attr,
        OptionalAttr<F64ArrayAttr>:$scales_attr,
        OptionalAttr<I64ArrayAttr>:$axes_attr,
        OptionalAttr<F64ArrayAttr>:$tile_offset_attr,
        OptionalAttr<I64ArrayAttr>:$initial_input_dims_attr,
        OptionalAttr<I64ArrayAttr>:$initial_output_dims_attr,

        IE_InterpolateAttr:$attr,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        RankedTensorOf<[UI8, F16, F32, F64, quant_QuantizedType]>:$output
    );
    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

def IE_PSROIPoolingOp :
        IE_LayerOp<
            "PSROIPooling"
        > {
    let summary = "InferenceEngine PSROIPooling layer";

    let arguments = (ins
        4DTensorOf<[F16, F32, F64]>:$input,
        2DTensorOf<[F16, F32, F64]>:$coords,

        IntAttr:$output_dim,
        F64Attr:$spatial_scale,
        IntAttr:$group_size,
        OptionalAttr<IntAttr>:$spatial_bins_x,
        OptionalAttr<IntAttr>:$spatial_bins_y,
        OptionalAttr<IE_PSROIPoolingModeAttr>:$mode
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_ROIAlignOp :
        IE_LayerOp<
            "ROIAlign",
            [
               ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine ROIAlign layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        RankedTensorOf<[F16, F32, F64]>:$coords,
        1DTensorOf<[AnyInteger]>:$roisIdx,

        IntAttr:$pooled_h,
        IntAttr:$pooled_w,
        IntAttr:$sampling_ratio,
        F64Attr:$spatial_scale,
        IE_ROIAlignMethodAttr:$poolingMode,
        IE_ROIAlignAlignedMethodAttr:$alignedMode
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_ROIPoolingOp :
        IE_LayerOp<
            "ROIPooling"
        > {
    let summary = "InferenceEngine ROIPooling layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        RankedTensorOf<[F16, F32, F64]>:$coords,

        I64ArrayAttr:$output_size,
        F64Attr:$spatial_scale,
        IE_ROIPoolingMethodAttr:$method
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

//  Conversions:
//   NV12toRGB, NV12toBGR,
//   I420toRGB, I420toBGR
def IE_YuvToRgbOp :
        IE_LayerOp<
            "YuvToRgb",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine NV12/I420 to RGB/BGR layer";

    let arguments = (ins
                 4DTensorOf<[UI8, F16, F32, F64]> :$input1,
        Optional<4DTensorOf<[UI8, F16, F32, F64]>>:$input2,
        Optional<4DTensorOf<[UI8, F16, F32, F64]>>:$input3,

        IE_ColorFmtAttr:$inFmt,
        IE_ColorFmtAttr:$outFmt
    );

    let results = (outs
        4DTensorOf<[UI8, F16, F32, F64]>:$output
    );

    let hasCanonicalizer = 1;
}

#endif
