//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_POOLING
#define VPUX_COMPILER_DIALECT_VPU_OPS_POOLING

include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"


def VPU_AdaptiveAvgPoolOp :
        VPU_LayerOp<
            "AdaptiveAvgPool"
        > {
    let summary = "AdaptiveAvgPool VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        1DTensorOf<[SI32, SI64]>:$pooled_spatial_shape
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}


def VPU_AdaptiveMaxPoolOp :
        VPU_LayerOp<
            "AdaptiveMaxPool"
        > {
    let summary = "AdaptiveMaxPool VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        1DTensorOf<[SI32, SI64]>:$pooled_spatial_shape,
        TypeAttr:$index_element_type
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output,
        RankedTensorOf<[SI32, SI64]>:$output_index
    );
}


def VPU_AvgPoolOp :
        VPU_LayerOp<
            "AvgPool",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "AvgPool VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,
        UnitAttr:$exclude_pads
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$output
    );
}

def VPU_AvgPool16Op :
        VPU_LayerOp<
            "AvgPool16",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "AvgPool16 VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,
        UnitAttr:$exclude_pads
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$output
    );
}


def VPU_MaxPoolOp :
        VPU_LayerOp<
            "MaxPool",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "MaxPool VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,

        OptionalAttr<IE_PostOpAttr>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$output
    );
}


def VPU_MaxPool8Op :
        VPU_LayerOp<
            "MaxPool8"
        > {
    let summary = "MaxPool8 VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,
        TypeAttr:$index_element_type,

        IntAttr:$axis
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$output,
        RankedTensorOf<[SI32, SI64]>:$output_index
    );
}

#endif
