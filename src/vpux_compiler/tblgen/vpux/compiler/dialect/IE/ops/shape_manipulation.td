//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_SHAPE_MANIPULATION
#define VPUX_COMPILER_DIALECT_IE_OPS_SHAPE_MANIPULATION

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_AffineReshapeOp :
        IE_LayerOp<
            "AffineReshape",
            [
                IE_ViewLikeOpInterface
            ]
        > {
    let summary = "AffineReshape layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayOfArraysAttr:$dim_mapping,
        I64ArrayAttr:$shape_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;

    let hasVerifier = 1;
}

def IE_DynamicReshapeOp :
        IE_LayerOp<
            "DynamicReshape"
        > {
    let summary = "InferenceEngine Reshape layer for dynamic shapes";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$shape,

        I64ArrayAttr:$output_shape,
        I64ArrayAttr:$output_bounds,

        UnitAttr:$only_set_shape
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def IE_ReshapeOp :
        IE_LayerOp<
            "Reshape",
            [
                IE_ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine Reshape layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$shape,

        UnitAttr:$special_zero,
        OptionalAttr<I64ArrayAttr>:$shape_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

def IE_ShapeCastOp :
        IE_LayerOp<
            "ShapeCast",
            [
                IE_ViewLikeOpInterface
            ]
        > {
    let summary = "ShapeCast layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$shape
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let extraClassDeclaration = [{
        // Note: to be compatible with existing code
        mlir::Value getSource() { return getInput(); }
    }] # baseExtraClassDeclaration;
}

def IE_ShapeOfOp :
        IE_LayerOp<
            "ShapeOf"
        > {
    let summary = "InferenceEngine ShapeOf layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        1DTensorOf<[SI32, SI64]>:$output
    );

    let hasFolder = 1;
}

def IE_SqueezeOp :
        IE_LayerOp<
            "Squeeze",
            [
                IE_ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine Squeeze layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

def IE_UnsqueezeOp :
        IE_LayerOp<
            "Unsqueeze",
            [
                IE_ViewLikeOpInterface
            ]
        > {

    let summary = "InferenceEngine Unsqueeze layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

#endif
