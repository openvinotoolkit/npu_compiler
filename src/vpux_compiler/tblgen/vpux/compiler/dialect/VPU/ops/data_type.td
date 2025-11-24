//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_DATA_TYPE
#define VPUX_COMPILER_DIALECT_VPU_OPS_DATA_TYPE

include "vpux/compiler/dialect/core/types.td"
include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/CastInterfaces.td"


def VPU_ConvertOp :
        VPU_LayerOp<
            "Convert",
            [
                DeclareOpInterfaceMethods<CastOpInterface>,
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "Convert VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        TypeAttr:$dstElemType,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::TypeAttr":$dstElemType
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_DequantizeOp :
        VPU_LayerOp<"Dequantize",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, [
                    "isOperationSplitOverKernelCompatible",]>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "Dequantize VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[quant_QuantizedType]>, VPU_DistributedTensor]>:$input,
        TypeAttr:$dstElemType,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32,]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::TypeAttr":$dstElemType
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
        bool isVFSupported();
    }] # baseExtraClassDeclaration;
}


def VPU_DynamicQuantizeOp :
        VPU_LayerOp<"DynamicQuantize",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getOutputTiling"]>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["getDistributedTypeForOpOperand",
                                                                     "getDistributedTypeForOpResult"]>
            ]
        > {
    let summary = "Dynamic-Quantize VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F32]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[F32]>, VPU_DistributedTensor]>:$min,
        AnyTypeOf<[RankedTensorOf<[F32]>, VPU_DistributedTensor]>:$max,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[UI8]>, VPU_DistributedTensor]>:$output,
        AnyTypeOf<[RankedTensorOf<[F32]>, VPU_DistributedTensor]>:$scale,
        AnyTypeOf<[RankedTensorOf<[UI8]>, VPU_DistributedTensor]>:$zero_point
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$min,
            "::mlir::Value":$max
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;
}


def VPU_SparsifyOp :
        VPU_LayerOp<"Sparsify",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Sparsify VPU layer";

    let arguments = (ins
        4DTensorOf<[quant_QuantizedType, F16, BF16]>:$input
    );

    let results = (outs
        VPU_SparseTensor:$output
    );
}


def VPU_DesparsifyOp :
        VPU_LayerOp<"Desparsify",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Desparsify VPU layer";

    let arguments = (ins
        VPU_SparseTensor:$input
    );

    let results = (outs
        4DTensorOf<[quant_QuantizedType, F16, BF16]>:$output
    );
}


def VPU_QuantizeOp :
        VPU_LayerOp<"Quantize",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Quantize VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[quant_QuantizedType]>:$output
    );
}




def VPU_DynamicDequantizeOp :
        VPU_LayerOp<"DynamicDequantize",
            [
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "InferenceEngine Dynamic Dequantize layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[quant_QuantizedType]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$scale,
        Optional<AnyTypeOf<[RankedTensorOf<[I8, I4, I2, U2]>, VPU_DistributedTensor]>>:$zp,

        TypeAttr:$dstElemType,
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
            "::mlir::Value":$scale,
            "::mlir::Value":$zp,
            "::mlir::TypeAttr":$dstElemType
        )>
    ];

    let hasVerifier = 1;
}


def VPU_FakeConvertOp :
        VPU_LayerOp<
            "FakeConvert"
        > {
    let summary = "FakeConvert VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$scale,
        Optional<RankedTensorOf<[F16, F32]>>:$shift,

        TypeAttr:$dst_type
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasVerifier = 1;
}


def VPU_FakeQuantizeOp :
        VPU_LayerOp<
            "FakeQuantize",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "FakeQuantize VPU layer";

    let description = [{
        The operation works in two modes:
         * integral quantization: specified by the 'levels' attribute
         * floating-point quantization: specified by the 'low_fp_type' attribute, [f8E4M3FN | f8E5M2]

        Only one of these attributes should be provided.
    }];

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input_low,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input_high,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output_low,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output_high,

        OptionalAttr<IntAttr>:$levels,
        OptionalAttr<TypeAttr>:$low_fp_type,
        IE_AutoBroadcastTypeAttr:$auto_broadcast,

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
            "::mlir::Value":$input_low,
            "::mlir::Value":$input_high,
            "::mlir::Value":$output_low,
            "::mlir::Value":$output_high,
            "::mlir::IntegerAttr":$levels,
            "::mlir::TypeAttr":$low_fp_type,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let hasVerifier = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_QuantizeCastOp :
        VPU_LayerOp<
            "QuantizeCast",
            [
                VPU_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_TilingViewLikeOpInterface>,
                DeclareOpInterfaceMethods<VPU_DistributedCastOpInterface>
            ]
        > {
    let summary = "Quantize Cast VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let hasVerifier = 1;
    let hasFolder = 1;
}

#endif
