//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --optimize-sparsify-desparsify-pairs="sparsity-profile=S1" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @OptimizeMultipleConsumers(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %0 = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %2 = VPU.Sparsify(%1) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %3 = VPU.Desparsify(%2) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %4 = VPU.Sparsify(%3) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %5 = VPU.Sparsify(%3) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %6 = VPU.NCE.Eltwise(%4, %5) {op_type = "ADD", ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 0 : i64, lrelu_shift = 0 : i64, mode = "ADD"}} -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %7 = VPU.Sparsify(%6) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %8 = VPU.Desparsify(%7) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %9 = VPU.Sparsify(%3) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %10 = VPU.NCE.Convolution(%9, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %11 = VPU.Sparsify(%10) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %12 = VPU.Desparsify(%11) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %8, %12 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL2:%.+]] = VPU.Sparsify([[VAL1]])

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Eltwise([[VAL2]], [[VAL2]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.Sparsify([[VAL3]])
    // CHECK:       [[VAL5:%.+]] = VPU.Desparsify([[VAL4]]

    // CHECK:       [[VAL6:%.+]] = VPU.NCE.Convolution([[VAL2]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL7:%.+]] = VPU.Sparsify([[VAL6]])
    // CHECK:       [[VAL8:%.+]] = VPU.Desparsify([[VAL7]]

    // CHECK:       return [[VAL5]], [[VAL8]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @OptimizeMultipleMixedConsumers(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %0 = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %2 = VPU.Sparsify(%1) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %3 = VPU.Desparsify(%2) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %4 = VPU.Sparsify(%3) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %5 = VPU.Sparsify(%3) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %6 = VPU.NCE.Eltwise(%4, %5) {op_type = "ADD", ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 0 : i64, lrelu_shift = 0 : i64, mode = "ADD"}} -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %7 = VPU.Sparsify(%6) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %8 = VPU.Desparsify(%7) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %9 = VPU.MaxPool(%3) {
            kernel_size = [3, 3],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            rounding_type = "FLOOR",
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %8, %9 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL2:%.+]] = VPU.Sparsify([[VAL1]])
    // CHECK:       [[VAL3:%.+]] = VPU.Desparsify([[VAL2]]

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.Eltwise([[VAL2]], [[VAL2]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL5:%.+]] = VPU.Sparsify([[VAL4]])
    // CHECK:       [[VAL6:%.+]] = VPU.Desparsify([[VAL5]]

    // CHECK:       [[VAL7:%.+]] = VPU.MaxPool([[VAL3]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL6]], [[VAL7]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!PreConcatType = type tensor<1x8x16x16xf16, {order = #NHWC}>
!PostConcatType = type tensor<1x16x16x16xf16, {order = #NHWC}>
!DefaultType = type tensor<1x16x16x16xf16, {order = #NHWC}>

func @OptimizeConcat(%arg0: !PreConcatType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (!DefaultType, !DefaultType) {
    %0 = VPU.Sparsify(%arg0) : !PreConcatType -> !VPU.SparseTensor<data=!PreConcatType>
    %1 = VPU.Desparsify(%0) : !VPU.SparseTensor<data=!PreConcatType> -> !PreConcatType
    %2 = VPU.Desparsify(%0) : !VPU.SparseTensor<data=!PreConcatType> -> !PreConcatType
    %3 = VPU.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 8, 0, 0]]} : !PreConcatType, !PreConcatType -> !PostConcatType
    %4 = VPU.Sparsify(%3) : !PostConcatType -> !VPU.SparseTensor<data=!PostConcatType>
    %5 = VPU.Sparsify(%3) : !PostConcatType -> !VPU.SparseTensor<data=!PostConcatType>

    %6 = VPU.NCE.Convolution(%4, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> !DefaultType
    %7 = VPU.NCE.Convolution(%5, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> !DefaultType


    return %6, %7 : !DefaultType, !DefaultType

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
    // CHECK-NOT:   VPU.Desparsify
    // CHECK-NOT:   VPU.Desparsify
    // CHECK:       [[VAL1:%.+]] = VPU.Concat([[VAL0]], [[VAL0]])
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    // CHECK-NOT:   VPU.Sparsify
    // CHECK-NOT:   VPU.Sparsify
    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], %arg2, %arg1)
    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL1]], %arg2, %arg1)
    // CHECK:       return [[VAL2]], [[VAL3]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!PreConcatType = type tensor<1x8x16x16xf16, {order = #NHWC}>
!PostConcatType = type tensor<1x16x16x16xf16, {order = #NHWC}>
!DefaultType = type tensor<1x16x16x16xf16, {order = #NHWC}>

func @OptimizeConcatMixedConsumers(%arg0: !PreConcatType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (!DefaultType, !PostConcatType) {
    %0 = VPU.Sparsify(%arg0) : !PreConcatType -> !VPU.SparseTensor<data=!PreConcatType>
    %1 = VPU.Desparsify(%0) : !VPU.SparseTensor<data=!PreConcatType> -> !PreConcatType
    %2 = VPU.Desparsify(%0) : !VPU.SparseTensor<data=!PreConcatType> -> !PreConcatType
    %3 = VPU.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 8, 0, 0]]} : !PreConcatType, !PreConcatType -> !PostConcatType
    %4 = VPU.Sparsify(%3) : !PostConcatType -> !VPU.SparseTensor<data=!PostConcatType>

    %5 = VPU.NCE.Convolution(%4, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> !DefaultType
    %6 = VPU.MaxPool(%3) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : !PostConcatType -> !PostConcatType


    return %5, %6 : !DefaultType, !PostConcatType

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
    // CHECK-NOT:   VPU.Desparsify
    // CHECK-NOT:   VPU.Desparsify

    // CHECK:       [[VAL1:%.+]] = VPU.Concat([[VAL0]], [[VAL0]])
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>

    // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]])

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL1]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.MaxPool([[VAL2]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       return [[VAL3]], [[VAL4]]
}
