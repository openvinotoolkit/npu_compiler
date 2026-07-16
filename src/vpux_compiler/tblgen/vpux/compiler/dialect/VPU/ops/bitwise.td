//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_BITWISE
#define VPUX_COMPILER_DIALECT_VPU_OPS_BITWISE

include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"


def VPU_BitwiseAndOp :
        VPU_LayerOp<
            "BitwiseAnd",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "BitwiseAnd VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$output
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


def VPU_BitwiseOrOp :
        VPU_LayerOp<
            "BitwiseOr",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "BitwiseOr VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$output
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


def VPU_BitwiseXorOp :
        VPU_LayerOp<
            "BitwiseXor",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "BitwiseXor VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[I32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[I32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[I32, SI32]>, VPU_DistributedTensor]>:$output
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


def VPU_BitwiseNotOp :
        VPU_LayerOp<
            "BitwiseNot",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "BitwiseNot VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$input1,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_BitwiseRightShiftOp :
        VPU_LayerOp<
            "BitwiseRightShift",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "BitwiseRightShift VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$output
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

def VPU_BitwiseLeftShiftOp :
        VPU_LayerOp<
            "BitwiseLeftShift",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "BitwiseLeftShift VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[I8, SI8, I32, SI32]>, VPU_DistributedTensor]>:$output
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
