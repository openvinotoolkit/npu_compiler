//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_COMPARISON
#define VPUX_COMPILER_DIALECT_VPU_OPS_COMPARISON

include "vpux/compiler/dialect/core/types.td"
include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"


def VPU_EqualOp :
        VPU_LayerOp<
            "Equal",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Equal VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[Bool8]>, VPU_DistributedTensor]>:$output
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


def VPU_GreaterOp :
        VPU_LayerOp<
            "Greater",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Greater VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[Bool8]>, VPU_DistributedTensor]>:$output
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


def VPU_GreaterEqualOp :
        VPU_LayerOp<
            "GreaterEqual",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
            ]
        > {
    let summary = "GreaterEqual VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
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

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$output
    );
    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_LessOp :
        VPU_LayerOp<
            "Less",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Less VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[Bool8]>, VPU_DistributedTensor]>:$output
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


def VPU_LessEqualOp :
        VPU_LayerOp<
            "LessEqual",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "LessEqual VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[Bool8]>, VPU_DistributedTensor]>:$output
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


def VPU_NotEqualOp :
        VPU_LayerOp<
            "NotEqual",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
            ]
        > {
    let summary = "NotEqual VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[Bool8]>, VPU_DistributedTensor]>:$output
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

#endif
