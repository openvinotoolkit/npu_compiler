//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_ACTIVATION
#define VPUX_COMPILER_DIALECT_VPU_OPS_ACTIVATION

include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

def VPU_EluOp :
        VPU_LayerOp<
            "Elu",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "Elu VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        F64Attr:$x,
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
            "::mlir::FloatAttr":$x
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_GeluOp :
        VPU_LayerOp<
            "Gelu",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Gelu VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_HardSigmoidOp :
        VPU_LayerOp<
            "HardSigmoid",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "HardSigmoid VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        F64Attr:$alpha_value,
        F64Attr:$beta_value,

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
            "::mlir::FloatAttr":$alpha,
            "::mlir::FloatAttr":$beta
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_HSigmoidOp :
        VPU_LayerOp<
            "HSigmoid",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "HSigmoid VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

def VPU_HSwishOp :
        VPU_LayerOp<
            "HSwish",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "HSwish VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_LeakyReluOp :
        VPU_LayerOp<
            "LeakyRelu",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "LeakyRelu VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        F64Attr:$negative_slope,
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
            "::mlir::FloatAttr":$negative_slope
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_LogSoftmaxOp :
        VPU_LayerOp<
            "LogSoftmax",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "LogSoftmax VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        IntAttr:$axisInd,
        OptionalAttr<IntAttr>:$padSize,
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
            "::mlir::IntegerAttr":$axisInd,
            "::mlir::IntegerAttr":$padSize
        )>
    ];
}

def VPU_MishOp :
        VPU_LayerOp<
            "Mish",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "Mish VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
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
            "::mlir::Value":$input
        )>
    ];
}

def VPU_PReluOp :
        VPU_LayerOp<
            "PRelu",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "PRelu VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$negative_slope,

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
            "::mlir::Value":$input1,
            "::mlir::Value":$input2
        )>
    ];

    let hasVerifier = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_ReLUOp :
        VPU_LayerOp<
            "ReLU",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "ReLU VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI8, SI16, SI32]>, VPU_DistributedTensor]>:$input,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI8, SI16, SI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);

        bool isVFSupported();
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_SeluOp :
        VPU_LayerOp<
            "Selu",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "Selu VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        F64Attr:$alpha_value,
        F64Attr:$lambda_value,
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
            "::mlir::FloatAttr":$alpha_value,
            "::mlir::FloatAttr":$lambda_value
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_SigmoidOp :
        VPU_LayerOp<
            "Sigmoid",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "Sigmoid VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

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
            "::mlir::Value":$input
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_SoftMaxOp :
        VPU_LayerOp<
            "SoftMax",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "SoftMax VPU layer";

    let description = [{
        Computes the softmax function along the specified axis.

        The optional `max` operand allows passing a precomputed per-slice maximum,
        enabling the original Softmax to be decomposed into:
          1. A ReduceMax that produces `max` (shape = input with axis dim set to 1).
          2. This op consuming both `input` and `max`.

        This decomposition allows the ReduceMax to be fused into a parent DPU task.
        When `max` is absent the op behaves identically to the original single-input Softmax.

        Constraints on `max` (verified at construction time):
          - Same rank as `input`.
          - Size 1 on the softmax axis dimension.
          - Same size as `input` on all other dimensions.
    }];

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        // Optional precomputed maximum over the softmax axis (keep_dims=true).
        // Shape: same as $input with the axis dimension reduced to 1.
        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>>:$max,
        IntAttr:$axisInd,
        OptionalAttr<IntAttr>:$padSize,
        OptionalAttr<TypeAttr>:$dstElemType,
        UnitAttr:$maskAware,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);

        bool isVFSupported();

        DimArr restrictedFusionAxes();
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::std::optional<::mlir::Value>":$max,
            "::mlir::IntegerAttr":$axisInd,
            "::mlir::IntegerAttr":$padSize,
            "::mlir::TypeAttr":$dstElemType,
            CArg<"::mlir::UnitAttr", "{}">:$maskAware
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_SoftPlusOp :
        VPU_LayerOp<
            "SoftPlus",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "SoftPlus VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

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
            "::mlir::Value":$input
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_SoftSignOp :
        VPU_LayerOp<
            "SoftSign",
            [
                VPU_EltwiseOp
            ]
        > {
    let summary = "SoftSign VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );


    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

def VPU_SwishOp :
        VPU_LayerOp<
            "Swish",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "Swish VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>>:$beta,

        OptionalAttr<F64Attr>:$beta_value,
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
            "::mlir::Value":$beta,
            "::mlir::FloatAttr":$beta_value
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

#endif
