//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_ELTWISE
#define VPUX_COMPILER_DIALECT_VPU_OPS_ELTWISE

include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

def VPU_AddOp :
        VPU_LayerOp<
            "Add",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Add VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOpAttrInterface>:$post_op,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast,
            "vpux::IE::PostOpAttr":$post_op
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
    let hasVerifier = 1;
}


def VPU_DivideOp :
        VPU_LayerOp<
            "Divide",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Divide VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_FloorModOp :
        VPU_LayerOp<
            "FloorMod",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "FloorMod VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_MinimumOp :
        VPU_LayerOp<
            "Minimum",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "InferenceEngine Minimum layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_MaximumOp :
        VPU_LayerOp<
            "Maximum",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "Maximum VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input2,
        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_ModOp :
        VPU_LayerOp<
            "Mod",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Mod VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


def VPU_MultiplyOp :
        VPU_LayerOp<
            "Multiply",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "Multiply VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOpAttrInterface>:$post_op,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast,
            "vpux::IE::PostOpAttr":$post_op
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_PowerOp :
        VPU_LayerOp<
            "Power",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Power VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_ScaleShiftOp :
        VPU_LayerOp<
            "ScaleShift",
            [
                VPU_TilingBuilderOpInterface,
                AttrSizedOperandSegments,
                VPU_EltwiseOp
            ]
        > {
    let summary = "ScaleShift VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$weights,
        Optional<RankedTensorOf<[F16, F32]>>:$biases
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_SquaredDifferenceOp :
        VPU_LayerOp<
            "SquaredDiff",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "SquaredDiff VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_SubtractOp :
        VPU_LayerOp<
            "Subtract",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Subtract VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI8, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI8, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOpAttrInterface>:$post_op,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI8, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast,
            "vpux::IE::PostOpAttr":$post_op
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

#endif
