//
// Copyright (C) 2025-2026 Intel Corporation
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
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getMaxNumTiles"]>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
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
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getMaxNumTiles"]>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
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
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getMaxNumTiles"]>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
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
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getMaxNumTiles"]>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
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
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getMaxNumTiles"]>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
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
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getMaxNumTiles"]>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
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


def VPU_ReduceMeanDMAOp : VPU_SwIoDmaOp<"ReduceMeanDMA", [
        DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
    ]> {
    let summary = "Spatial FP16 Mean using SHAVE-managed DMA";
    let description = [{
        Reduces the full runtime H/W domain of compact NCHW [1,C,H,W]
        input into static [1,C,1,1]. Compact input and positive runtime H/W
        are caller preconditions. Workers own disjoint whole channels and
        stream through private CMX scratch; spatial tiling is unsupported.
        The reduction axes are implicit, H and W: the op takes no axes operand.
    }];

    // ReduceMeanDMA is single-tile only.
    let checkStrategyCompatibilityBody = [{
            return false;
    }];

    let arguments = (ins
        RankedTensorOf<[F16]>:$input,
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$aux_buffer,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );
    let results = (outs RankedTensorOf<[F16]>:$output);
    let builders = [OpBuilder<(ins "::mlir::Value":$input)>];
    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
        static bool isSupportedInput(mlir::Type inputType, vpux::LogCb logCb);
    }] # baseExtraClassDeclaration;
    let hasVerifier = 1;
}


def VPU_ReduceMinOp :
        VPU_LayerOp<
            "ReduceMin",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getMaxNumTiles"]>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
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
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getMaxNumTiles"]>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
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
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getMaxNumTiles"]>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
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

def VPU_ReduceSquareOp :
        VPU_LayerOp<
            "ReduceSquare",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getMaxNumTiles"]>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "ReduceSquare VPU Layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
        OptionalAttr<F64Attr>:$epsilon,
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
            "::mlir::UnitAttr":$keep_dims,
            "::mlir::FloatAttr":$epsilon
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

#endif
