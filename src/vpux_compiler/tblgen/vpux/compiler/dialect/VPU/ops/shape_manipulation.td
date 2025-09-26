//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_SHAPE_MANIPULATION
#define VPUX_COMPILER_DIALECT_VPU_OPS_SHAPE_MANIPULATION

include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"


def VPU_AffineReshapeOp :
        VPU_LayerOp<
            "AffineReshape",
            [
                VPU_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_DistributedCastOpInterface>
            ]
        > {
    let summary = "AffineReshape VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$input,

        I64ArrayOfArraysAttr:$dim_mapping,
        I64ArrayAttr:$shape_value
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
    let hasFolder = 1;
}


def VPU_DynamicReshapeOp :
        VPU_LayerOp<
            "DynamicReshape",
            [
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                VPU_BoundsRepresentationInterface
            ]
        > {
    let summary = "DynamicReshape VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[SI32]>:$shape,

        I64ArrayAttr:$output_shape,
        I64ArrayAttr:$output_bounds,

        UnitAttr:$only_set_shape,

        DefaultValuedAttr<VPU_BoundsRepresentationAttr, "vpux::VPU::BoundsRepresentation::BOUNDS">:$bounds_representation
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;
}


def VPU_ReshapeOp :
        VPU_LayerOp<
            "Reshape",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Reshape VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$shape,

        UnitAttr:$special_zero,
        OptionalAttr<I64ArrayAttr>:$shape_value
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}


def VPU_SqueezeOp :
        VPU_LayerOp<
            "Squeeze",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Squeeze VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}


def VPU_UnsqueezeOp :
        VPU_LayerOp<
            "Unsqueeze",
            [
                VPU_ViewLikeOpInterface
            ]
        > {

    let summary = "Unsqueeze VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}


def VPU_ShapeCastOp :
        VPU_LayerOp<
            "ShapeCast",
            [
                VPU_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_TilingViewLikeOpInterface>,
                DeclareOpInterfaceMethods<VPU_DistributedCastOpInterface>
            ]
        > {
    let summary = "ShapeCast VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        I64ArrayAttr:$shape
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool isSupportedTilingDim(DimArrRef tilingDims);
        bool isSupportedTilingDimWithRestrictions(Dim tilingDim);
        bool isSupportedOutTile(const TileInfo& outTile);
        // Note: to be compatible with existing code
        mlir::Value getSource() { return getInput(); }
    }] # baseExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];

    let hasFolder = 1;

    let hasCanonicalizer = 1;
}


def VPU_ShapeOfOp :
        VPU_LayerOp<
            "ShapeOf",
            [
                DeclareOpInterfaceMethods<VPU_SWOpInterface>
            ]
        > {
    let summary = "ShapeOf VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor]>:$input
    );

    let results = (outs
        1DTensorOf<[SI32]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;
}

#endif
