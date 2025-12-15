//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_NORMALIZATION
#define VPUX_COMPILER_DIALECT_IE_OPS_NORMALIZATION

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def IE_BatchNormInferenceOp :
        IE_LayerOp<
            "BatchNormInference",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine BatchNormInference layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,

        Optional<RankedTensorOf<[F16, F32, F64]>>:$gamma,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$beta,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$mean,
        Optional<RankedTensorOf<[F16, F32, F64]>>:$variance,

        OptionalAttr<F64ArrayAttr>:$gamma_value,
        OptionalAttr<F64ArrayAttr>:$beta_value,
        OptionalAttr<F64ArrayAttr>:$mean_value,
        OptionalAttr<F64ArrayAttr>:$variance_value,

        F64Attr:$eps
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );

    let hasCanonicalizer = 1;
}

def IE_GRNOp :
        IE_LayerOp<
            "GRN"
        > {
    let summary = "InferenceEngine GRN layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,

        F64Attr:$bias
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_GroupNormalizationOp :
        IE_LayerOp<
            "GroupNormalization"
        > {
    let summary = "InferenceEngine GroupNormalization layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        RankedTensorOf<[F16, F32, F64]>:$scale,
        RankedTensorOf<[F16, F32, F64]>:$bias,

        I32Attr:$num_groups,
        F32Attr:$epsilon

    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_LRNOp :
        IE_LayerOp<
            "LRN"
        > {
    let summary = "InferenceEngine LRN layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        Optional<1DTensorOf<[SI32, SI64]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$size
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );

    let hasCanonicalizer = 1;
}

def IE_LRN_IEOp :
        IE_LayerOp<
            "LRN_IE"
        > {
    let summary = "InferenceEngine LRN_IE layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$size,
        IE_LRN_IERegionAttr:$region
    );

    let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );
}

def IE_MVNOp :
        IE_LayerOp<
            "MVN"
        > {
    let summary = "InferenceEngine MVN1 layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        BoolAttr:$across_channels,
        BoolAttr:$normalize_variance,
        F64Attr:$eps,
        OptionalAttr<I64ArrayAttr>:$internal_reshape,
        DefaultValuedOptionalAttr<BoolAttr, "false">:$high_precision_normalize
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::BoolAttr":$across_channels,
            "::mlir::BoolAttr":$normalize_variance,
            "::mlir::FloatAttr":$eps
        )>,
    ];

    let hasCanonicalizer = 1;
}

def IE_MVN6Op :
        IE_LayerOp<
            "MVN6",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine MVN6 layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<AnyRankedTensor>:$scale,
        Optional<AnyRankedTensor>:$bias,
        Optional<1DTensorOf<[SI32, SI64]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value,
        BoolAttr:$normalize_variance,
        F64Attr:$eps,
        IE_MvnEpsModeAttr:$eps_mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
}

def IE_NormalizeIEOp :
        IE_LayerOp<
            "NormalizeIE"
        > {
    let summary = "InferenceEngine NormalizeIE layer";

    let arguments = (ins
        AnyRankedTensor:$data,
        AnyRankedTensor:$weights,

        F64Attr:$eps,
        BoolAttr:$across_spatial,
        BoolAttr:$channel_shared
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

def IE_NormalizeL2Op :
        IE_LayerOp<
            "NormalizeL2"
        > {
    let summary = "InferenceEngine NormalizeL2 layer";

    let arguments = (ins
        AnyRankedTensor:$data,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,
        OptionalAttr<ArrayAttr>:$axes_value,

        F64Attr:$eps,
        IE_EpsModeAttr:$eps_mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def IE_RMSOp :
        IE_LayerOp<
            "RMS"
        > {
    let summary = "InferenceEngine RMS layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, F64]>:$input,
        RankedTensorOf<[F16, F32, F64]>:$gamma,
        F64Attr:$eps
     );
   let results = (outs
        RankedTensorOf<[F16, F32, F64]>:$output
    );

    let hasCanonicalizer = 1;
}

#endif
