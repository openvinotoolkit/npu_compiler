//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_CONVOLUTION
#define VPUX_COMPILER_DIALECT_VPU_OPS_CONVOLUTION

include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"


def VPU_ConvolutionOp :
        VPU_LayerOp<
            "Convolution",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "Convolution VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$filter,
        Optional<RankedTensorOf<[F16, F32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,

        OptionalAttr<IE_PostOpAttr>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_DeformableConvolutionOp :
        VPU_LayerOp<
            "DeformableConvolution",
             [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
             ]
        > {
    let summary = "DeformableConvolution VPU layer";

    let description = [{
        The 'initial_output_offset_attr' attribute is used in kernel for tiling, MC/MS logic.
    }];

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$offset,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$kernel,
        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>>:$mask,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,

        IntAttr:$group,
        IntAttr:$deformable_group,
        UnitAttr:$bilinear_interpolate_pad,

        OptionalAttr<I64ArrayAttr>:$initial_output_offset_attr,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$offset,
            "::mlir::Value":$kernel,
            "::mlir::Value":$mask,
            "::mlir::ArrayAttr":$strides,
            "::mlir::ArrayAttr":$pads_begin,
            "::mlir::ArrayAttr":$pads_end,
            "::mlir::ArrayAttr":$dilations,
            "::mlir::IntegerAttr":$group,
            "::mlir::IntegerAttr":$deformable_group,
            "::mlir::UnitAttr":$bilinear_interpolate_pad,
            "::mlir::ArrayAttr":$initial_output_offset_attr
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let checkInferredMemSpace = 0;
    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_FullyConnectedOp:
        VPU_LayerOp<
            "FullyConnected"
        > {
    let summary = "FullyConnected VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$weights,
        Optional<RankedTensorOf<[F16, F32]>>:$bias
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_GroupConvolutionOp :
        VPU_LayerOp<
            "GroupConvolution",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "GroupConvolution VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input,
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$filter,
        Optional<RankedTensorOf<[F16, F32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        OptionalAttr<IntAttr>:$groups,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );

    list<string> elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);
    }] # baseExtraClassDeclaration;
}


def VPU_MatMulOp:
        VPU_LayerOp<
            "MatMul",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "MatMul VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        UnitAttr:$transpose_a,
        UnitAttr:$transpose_b,
        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);

        static bool isSupported(vpux::IE::MatMulOp matmulOp);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "::mlir::UnitAttr":$transpose_a,
            "::mlir::UnitAttr":$transpose_b,
            "vpux::IE::PostOpAttr":$post_op
        )>
    ];

    let hasVerifier = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_TransposedConvolutionOp:
        VPU_LayerOp<
            "TransposedConvolution",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "TransposedConvolution VPU layer";

    let description = [{
        Represents a transposed convolution, which consumes an input and filter tensor
        and generates an output larger than the input.

        Operands:
        - `input`: the input tensor of the operation; 4D layout [N, C_IN, Y, X]
        - `filter`: the convolutional kernel tensor; expected layout [C_OUT, C_IN, KY, KX]
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
        Optional<RankedTensorOf<[F16, F32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$spatial_output_padding,

        OptionalAttr<IE_PostOpAttr>:$post_op,
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
