//
// Copyright (C) 2025-2026 Intel Corporation
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
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<IE_ShapeInfoOpInterface, ["verifyShapeInfo"]>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
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
    let hasVerifier = 1;

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
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "InferenceEngine NV12/I420 to RGB/BGR layer";

    let description = [{
        Converts an NV12 or I420 YUV image to RGB/BGR. Per NV12/I420 semantics, luma
        (`Y`) is a `[N, H, W, 1]` plane; chroma is either one interleaved
        `[N, H/2, W/2, 2]` `UV` plane (NV12) or two separate `[N, H/2, W/2, 1]` `U`
        and `V` planes (I420); the output is `[N, H, W, 3]`.

        Per OpenVINO's `NV12toRGB`/`I420toRGB` semantics, the input image can be
        represented in two ways:
        - Single plane, as it is stored in a file: only `input1` is set (`input2`/
          `input3` unset), and its height dimension is 1.5x the real image height
          (`C` = 1), for both NV12 and I420. E.g. for an image of `w=640, h=480`,
          `input1` has NHWC shape `[1, 720, 640, 1]`. `ConvertToMultiInputs`
          canonicalizes this into the split form below: 2-way (`Y`+`UV`) for NV12,
          3-way (`Y`+`U`+`V`) for I420.
        - Multiple planes (as commonly produced by physical video sources), each
          with real per-plane spatial dimensions: NV12 sets `input1`(`Y`)+`input2`
          (`UV`), `input3` unset; I420 sets all three, `input1`(`Y`)+`input2`(`U`)+
          `input3`(`V`). E.g. NV12 `input1`/`Y` has shape `[1, 480, 640, 1]`,
          `input2`/`UV` has shape `[1, 240, 320, 2]`.

            In this compiler, YuvToRgb is represented in one canonical semantic form:
            tensors carry `DimsOrder::NHWC`, while the shape array follows the compiler's
            internal logical `[N, C, H, W]` convention. This compiler-specific convention
            keeps semantic channel access stable at `Dims4D::Act::C` and semantic height at
            `Dims4D::Act::H` across importer, shape inference, and transformation passes.

            `input3` is only used for I420 (its dedicated 3-way split); NV12 never uses
            `input3`.

            Example (already-split 2-plane input, NV12, `input3` unset):
            `input1 : tensor<1x1x180x1280xf16, {order = #NHWC}>` (luma, `Y`),
            `input2 : tensor<1x2x90x640xf16, {order = #NHWC}>` (chroma, `UV`) ->
            `output : tensor<1x3x180x1280xf16, {order = #NHWC}>`.

        For a single-plane input (`input2`/`input3` unset), luma and chroma are
        stacked along the semantic height axis, so that axis is multiplied by 2/3
        to recover the real luma height.
    }];

    let arguments = (ins
                 4DTensorOf<[UI8, F16, F32, F64]> :$input1,
        Optional<4DTensorOf<[UI8, F16, F32, F64]>>:$input2,
        Optional<4DTensorOf<[UI8, F16, F32, F64]>>:$input3,

        IE_ColorFmtAttr:$inFmt,
        IE_ColorFmtAttr:$outFmt,
        OptionalAttr<F32Attr>:$scale
    );

    let results = (outs
        4DTensorOf<[UI8, F16, F32, F64]>:$output
    );

    let hasCanonicalizer = 1;
}

#endif
