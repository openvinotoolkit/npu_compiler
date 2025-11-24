//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_IMAGE
#define VPUX_COMPILER_DIALECT_VPU_OPS_IMAGE

include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"


def VPU_DeformablePSROIPoolingOp :
        VPU_LayerOp<
            "DeformablePSROIPooling"
        > {
    let summary = "DeformablePSROIPooling VPU layer";

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


def VPU_ExperimentalDetectronROIFeatureExtractorOp :
        VPU_LayerOp<
            "ExperimentalDetectronROIFeatureExtractor", [
               DeclareOpInterfaceMethods<VPU_AuxiliaryBufferOpInterface>,
               AttrSizedOperandSegments
            ]
        > {
    let summary = "ExperimentalDetectronROIFeatureExtractor VPU layer";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,
        Optional<1DTensorOf<[F16, F32]>>:$reorderedRois,
        Optional<1DTensorOf<[UI32]>>:$originalRoiMap,
        Optional<1DTensorOf<[F16, F32]>>:$outputRoisFeaturesTemp,
        Optional<1DTensorOf<[UI32]>>:$levels,

        IE_ExperimentalDetectronROIFeatureExtractorAttr:$attr

   );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output,
        RankedTensorOf<[F16, F32]>:$outputROIs
    );
}


def VPU_GridSampleOp :
        VPU_LayerOp<
            "GridSample",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverBatchCompatible"]>
            ]
        > {
    let summary = "GridSample VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$grid,

        UnitAttr:$align_corners,
        OptionalAttr<IE_GridSampleModeAttr>:$mode,
        OptionalAttr<IE_GridSamplePaddingModeAttr>:$padding_mode,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$grid,
            "::mlir::UnitAttr":$align_corners,
            "vpux::IE::GridSampleModeAttr":$mode,
            "vpux::IE::GridSamplePaddingModeAttr":$padding_mode
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_InterpolateOp :
        VPU_LayerOp<
            "Interpolate",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "Interpolate VPU layer";

    let description = [{
        The `coordinates` contain byte offsets for the current `input` tensor.
        The `lambdas` contain two interleaved values for each coordinate.
    }];

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[UI8, F16, F32, quant_QuantizedType]>, VPU_DistributedTensor]>:$input,
        Optional<AnyTypeOf<[RankedTensorOf<[AnyInteger]>, VPU_DistributedTensor]>>:$sizes,
        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>>:$scales,
        Optional<AnyTypeOf<[RankedTensorOf<[AnyInteger]>, VPU_DistributedTensor]>>:$axes,
        Optional<AnyTypeOf<[RankedTensorOf<[SI32]>, VPU_DistributedTensor]>>:$coordinates,
        Optional<AnyTypeOf<[RankedTensorOf<[F16]>, VPU_DistributedTensor]>>:$lambdas,

        OptionalAttr<I64ArrayAttr>:$sizes_attr,
        OptionalAttr<F64ArrayAttr>:$scales_attr,
        OptionalAttr<I64ArrayAttr>:$axes_attr,
        OptionalAttr<F64ArrayAttr>:$tile_offset_attr,
        OptionalAttr<I64ArrayAttr>:$initial_input_dims_attr,
        OptionalAttr<I64ArrayAttr>:$initial_output_dims_attr,
        OptionalAttr<I64ArrayAttr>:$initial_input_offset_attr,
        OptionalAttr<I64ArrayAttr>:$initial_output_offset_attr,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,

        IE_InterpolateAttr:$attr,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[UI8, F16, F32, quant_QuantizedType]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$sizes,
            "::mlir::Value":$scales,
            "::mlir::Value":$axes,
            "::mlir::Value":$coordinates,
            "::mlir::Value":$lambdas,
            "::mlir::ArrayAttr":$sizes_attr,
            "::mlir::ArrayAttr":$scales_attr,
            "::mlir::ArrayAttr":$axes_attr,
            "::mlir::ArrayAttr":$tile_offset_attr,
            "::mlir::ArrayAttr":$initial_input_dims_attr,
            "::mlir::ArrayAttr":$initial_output_dims_attr,
            "vpux::IE::InterpolateAttr":$attr,
            "::mlir::ArrayAttr":$output_padding,
            "::mlir::ArrayAttr":$input_padding
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_PSROIPoolingOp :
        VPU_LayerOp<
            "PSROIPooling"
        > {
    let summary = "PSROIPooling VPU layer";

    let arguments = (ins
        4DTensorOf<[F16, F32]>:$input,
        2DTensorOf<[F16, F32]>:$coords,

        IntAttr:$output_dim,
        F64Attr:$spatial_scale,
        IntAttr:$group_size,
        OptionalAttr<IntAttr>:$spatial_bins_x,
        OptionalAttr<IntAttr>:$spatial_bins_y,
        OptionalAttr<IE_PSROIPoolingModeAttr>:$mode
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_ROIAlignOp :
        VPU_LayerOp<
            "ROIAlign",
            [
               ResultsAreFloatLike
            ]
        > {
    let summary = "ROIAlign VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$coords,
        1DTensorOf<[AnyInteger]>:$roisIdx,

        IntAttr:$pooled_h,
        IntAttr:$pooled_w,
        IntAttr:$sampling_ratio,
        F64Attr:$spatial_scale,
        IE_ROIAlignMethodAttr:$poolingMode,
        IE_ROIAlignAlignedMethodAttr:$alignedMode
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_ROIPoolingOp :
        VPU_LayerOp<
            "ROIPooling"
        > {
    let summary = "ROIPooling VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$coords,

        I64ArrayAttr:$output_size,
        F64Attr:$spatial_scale,
        IE_ROIPoolingMethodAttr:$method
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_YuvToRgbOp :
        VPU_LayerOp<
            "YuvToRgb",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "InferenceEngine NV12/I420 to RGB/BGR layer";

    let arguments = (ins
                 AnyTypeOf<[RankedTensorOf<[UI8, F16, F32]>, VPU_DistributedTensor]> :$input1,
        Optional<AnyTypeOf<[RankedTensorOf<[UI8, F16, F32]>, VPU_DistributedTensor]>>:$input2,
        Optional<AnyTypeOf<[RankedTensorOf<[UI8, F16, F32]>, VPU_DistributedTensor]>>:$input3,

        IE_ColorFmtAttr:$inFmt,
        IE_ColorFmtAttr:$outFmt,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[UI8, F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "::mlir::Value":$input3,
            "vpux::IE::ColorFmtAttr":$inFmt,
            "vpux::IE::ColorFmtAttr":$outFmt
        )>,

        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::ColorFmtAttr":$inFmt,
            "vpux::IE::ColorFmtAttr":$outFmt
        )>,

        OpBuilder<(ins
            "::mlir::Value":$input1,
            "vpux::IE::ColorFmtAttr":$inFmt,
            "vpux::IE::ColorFmtAttr":$outFmt
        )>,
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

#endif
