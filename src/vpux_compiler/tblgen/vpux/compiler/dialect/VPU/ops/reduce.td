//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_REDUCE
#define VPUX_COMPILER_DIALECT_VPU_OPS_REDUCE

include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"


def VPU_ReduceL1Op :
        VPU_LayerOp<
            "ReduceL1",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceL1 VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
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
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_ReduceL2Op :
        VPU_LayerOp<
            "ReduceL2",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceL2 VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
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
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_ReduceLogicalAndOp :
        VPU_LayerOp<
            "ReduceLogicalAnd",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceLogicalAnd VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
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
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_ReduceLogicalOrOp :
        VPU_LayerOp<
            "ReduceLogicalOr",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceLogicalOr VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
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
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_ReduceMaxOp :
        VPU_LayerOp<
            "ReduceMax",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceMax VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
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
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_ReduceMeanOp :
        VPU_LayerOp<
            "ReduceMean",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceMean VPU Layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
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
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_ReduceMinOp :
        VPU_LayerOp<
            "ReduceMin",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
            ]
        > {
    let summary = "ReduceMin VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}


def VPU_ReduceProdOp :
        VPU_LayerOp<
            "ReduceProd",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceProd VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
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
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_ReduceSumOp :
        VPU_LayerOp<
            "ReduceSum",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceSum VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
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
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

def VPU_ReduceMeanSquareOp :
        VPU_LayerOp<
            "ReduceMeanSquare",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceMeanSquare VPU Layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
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
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

#endif
