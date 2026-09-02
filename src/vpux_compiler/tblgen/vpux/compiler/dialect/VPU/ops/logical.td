//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_LOGICAL
#define VPUX_COMPILER_DIALECT_VPU_OPS_LOGICAL

include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

def VPU_AndOp :
        VPU_LayerOp<
            "And",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible"]>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "And VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[I8, F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[I8, F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[I8, F16, F32, SI32]>, VPU_DistributedTensor]>:$output
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


def VPU_LogicalNotOp :
        VPU_LayerOp<
            "LogicalNot",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Logical Not VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[I8, F16, F32, SI32]>, VPU_DistributedTensor]>:$input,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[I8, F16, F32, SI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_LogicalOrOp :
        VPU_LayerOp<
            "LogicalOr",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible"]>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "LogicalOr VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[I8, F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[I8, F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[I8, F16, F32, SI32]>, VPU_DistributedTensor]>:$output
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


def VPU_LogicalXorOp :
        VPU_LayerOp<
            "LogicalXor",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "LogicalXor VPU layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, SI32]>:$input1,
        RankedTensorOf<[I8, F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, SI32]>:$output
    );
}

#endif
