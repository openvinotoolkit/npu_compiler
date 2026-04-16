//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_NORMALIZATION
#define VPUX_COMPILER_DIALECT_VPU_OPS_NORMALIZATION

include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"



def VPU_GRNOp :
        VPU_LayerOp<
            "GRN"
        > {
    let summary = "GRN VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$bias
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_GroupNormalizationOp :
        VPU_LayerOp<
            "GroupNormalization",
            [
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
            ]
        > {
    let summary = "GroupNormalization VPU layer";

    let arguments = (ins
       RankedTensorOf<[F16, F32]>:$input,
       RankedTensorOf<[F16, F32]>:$scale,
       RankedTensorOf<[F16, F32]>:$bias,

       I32Attr:$num_groups,
       F32Attr: $epsilon
    );

    let results = (outs
       RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_LRNOp :
        VPU_LayerOp<
            "LRN"
        > {
    let summary = "LRN VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        I64ArrayAttr:$axes,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$size
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_LRN_IEOp :
        VPU_LayerOp<
            "LRN_IE"
        > {
    let summary = "LRN_IE VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$size,
        IE_LRN_IERegionAttr:$region
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_MVNOp :
        VPU_LayerOp<
            "MVN",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                VPU_EltwiseOp
            ]

        > {
    let summary = "MVN1 VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        BoolAttr:$across_channels,
        BoolAttr:$normalize_variance,
        F64Attr:$eps,
        OptionalAttr<I64ArrayAttr>:$internal_reshape,
        DefaultValuedOptionalAttr<BoolAttr, "false">:$high_precision_normalize,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);

        DimArr getNonNormDims();
    }] # baseExtraClassDeclaration;


    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::BoolAttr":$across_channels,
            "::mlir::BoolAttr":$normalize_variance,
            "::mlir::FloatAttr":$eps
        )>,
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::BoolAttr":$across_channels,
            "::mlir::BoolAttr":$normalize_variance,
            "::mlir::FloatAttr":$eps,
            "::mlir::ArrayAttr":$internal_reshape
        )>,
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::BoolAttr":$across_channels,
            "::mlir::BoolAttr":$normalize_variance,
            "::mlir::FloatAttr":$eps,
            "::mlir::ArrayAttr":$internal_reshape,
            "::mlir::BoolAttr":$high_precision_normalize
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_MVN1SumOp :
        VPU_LayerOp<
            "MVN1SumOp",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "MVN1SumOp VPU layer (step 1/3 in MVN1 decomposition)";

    let description = [{
        Overview:

        Large MVN1 tensors that cannot be tiled, are decomposed into 3 tileable sub-ops:
        1. **MVN1SumOp** : computes partial sums on input tiles
        2. **MVN1MeanVarOp** : sum-reduces concatenated partial sums from previous step and computes _mean_, _variance_
        3. **MVN1Normalize**: applies normalization on input tiles

        Details:
        - **input** - tile of original _MVN1Op_ input tensor
        - **sum** - output tensor of shape **[N, C, H, W]**, with (0,1,2,3)->(0,2,3,1) layout (irrespective of input layout)
            - N = input N
            - C = input C if _across_channels_ = false, or 1 if _across_channels_ = true
            - H = number of clusters (output_height)
            - W = 2 if _normalize_variance_ = true (compute _sum_ and _sumOfSquares_ terms), else 1 (compute just _sum_ term)
    }];

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        BoolAttr:$across_channels,
        BoolAttr:$normalize_variance,
        IntAttr:$output_height,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
        static bool buffsFitIntoCMX(mlir::ModuleOp module, vpux::NDTypeInterface in, vpux::NDTypeInterface out);
    }] # baseExtraClassDeclaration;

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F32]>, VPU_DistributedTensor]>:$sum
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "bool":$across_channels,
            "bool":$normalize_variance,
            "int64_t":$output_height
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_MVN1MeanVarOp :
        VPU_LayerOp<
            "MVN1MeanVar",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "MVN1MeanVar VPU layer (step 2/3 in MVN1 decomposition)";

    let description = [{
        Background: see _MVN1SumOp_ description

        Details:
        Accumulates partial _sum_ (and optionally _sumOfSquares_) of concatenated input and computes _mean_ (and optionally _1/variance_ ) required for normalization.

        - **sum** - input is a concatenation (over W) of _MVN1SumOp_ outputs, shape = [N,C,W x num_parts]
            - W = 2 if _normalize_variance_ = true (input _sum_ and _sumOfSquares_), else 1 (input _sum_)
        - **meanVar** - output shape = [N,C,W], where
            - W = 2 if _normalize_variance_ = true (output _mean_ and _1/variance_ terms), else 1 (output just _mean_)
    }];

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F32]>, VPU_DistributedTensor]>:$sum,

        I64ArrayAttr:$orig_shape,
        BoolAttr:$across_channels,
        BoolAttr:$normalize_variance,
        F64Attr:$eps,
        TypeAttr:$output_type,
        OptionalAttr<I64ArrayAttr>:$internal_reshape,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$meanVar
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$sum,
            "::mlir::ArrayAttr":$orig_shape,
            "bool":$across_channels,
            "bool":$normalize_variance,
            "::mlir::APFloat":$eps,
            "::mlir::Type":$output_type
        )>,
        OpBuilder<(ins
            "::mlir::Value":$sum,
            "::mlir::ArrayAttr":$orig_shape,
            "bool":$across_channels,
            "bool":$normalize_variance,
            "::mlir::APFloat":$eps,
            "::mlir::Type":$output_type,
            "::mlir::ArrayAttr":$internal_reshape
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_MVN1NormalizeOp :
        VPU_LayerOp<
            "MVN1Normalize",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "MVN1Normalize VPU layer (step 3/3 in MVN1 decomposition)";

    let description = [{
        Background: see _MVN1SumOp_ description

        Applies normalization on a tile of input tensor.

        Details:
        - **input** - input tile of original _MVN1Op_ input tensor
        - **meanVar** - this input is the output of _MVN1MeanVarOp_
        - **output** - output tile of final result
    }];

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$meanVar,

        BoolAttr:$across_channels,
        BoolAttr:$normalize_variance,
        DefaultValuedOptionalAttr<BoolAttr, "false">:$high_precision_normalize,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$meanVar,
            "::mlir::BoolAttr":$across_channels,
            "::mlir::BoolAttr":$normalize_variance
        )>,
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$meanVar,
            "::mlir::BoolAttr":$across_channels,
            "::mlir::BoolAttr":$normalize_variance,
            "::mlir::BoolAttr":$high_precision_normalize
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_MVN6Op :
        VPU_LayerOp<
            "MVN6",
            [
                AttrSizedOperandSegments,
                VPU_EltwiseOp,
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "MVN6 VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$scale,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$bias,
        I64ArrayAttr:$axes,
        BoolAttr:$normalize_variance,
        F64Attr:$eps,
        IE_MvnEpsModeAttr:$eps_mode,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
        DimArr getNonNormDims();
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes,
            "::mlir::BoolAttr":$normalize_variance,
            "::mlir::FloatAttr":$eps,
            "vpux::IE::MvnEpsModeAttr":$eps_mode
        )>,
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$scale,
            "::mlir::Value":$bias,
            "::mlir::ArrayAttr":$axes,
            "::mlir::BoolAttr":$normalize_variance,
            "::mlir::FloatAttr":$eps,
            "vpux::IE::MvnEpsModeAttr":$eps_mode
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_NormalizeL2Op :
        VPU_LayerOp<
            "NormalizeL2",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "NormalizeL2 VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$data,
        ArrayAttr:$axes_value,

        F64Attr:$eps,
        IE_EpsModeAttr:$eps_mode,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let hasVerifier = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_NormalizeIEOp :
        VPU_LayerOp<
            "NormalizeIE"
        > {
    let summary = "NormalizeIE VPU layer";

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


def VPU_RMSOp :
        VPU_LayerOp<
            "RMS",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getMaxNumTiles"]>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
            ]
        > {
    let summary = "RMS VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$gamma,

        F64Attr:$eps,
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
            "::mlir::Value":$gamma,
            "::mlir::FloatAttr":$eps
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

#endif
