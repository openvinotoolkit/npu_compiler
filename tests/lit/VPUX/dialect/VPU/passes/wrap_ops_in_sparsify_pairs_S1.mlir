//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --wrap-ops-in-sparsify-pairs="sparsity-profile=S1" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @WrapSingleOp(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
    // CHECK:       [[VAL1:%.+]] = VPU.Desparsify([[VAL0]]

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.Sparsify([[VAL2]])
    // CHECK:       [[VAL4:%.+]] = VPU.Desparsify([[VAL3]]
    // CHECK:       return [[VAL4]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @WrapChainedMixedOps(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %1 = VPU.NCE.Convolution(%0, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
    } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[POOL:%.+]] = VPU.MaxPool(%arg0)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>


    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify([[POOL]])
    // CHECK:       [[VAL1:%.+]] = VPU.Desparsify([[VAL0]]

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.Sparsify([[VAL2]])
    // CHECK:       [[VAL4:%.+]] = VPU.Desparsify([[VAL3]]

    // CHECK:       return [[VAL4]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @WrapMultipleConsumers(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%1, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %2, %3 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
    // CHECK:       [[VAL1:%.+]] = VPU.Desparsify([[VAL0]]

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.Sparsify([[VAL2]])
    // CHECK:       [[VAL4:%.+]] = VPU.Desparsify([[VAL3]]

    // CHECK:       [[VAL5:%.+]] = VPU.Sparsify([[VAL4]])
    // CHECK:       [[VAL6:%.+]] = VPU.Desparsify([[VAL5]]

    // CHECK:       [[VAL9:%.+]] = VPU.NCE.Convolution([[VAL6]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL10:%.+]] = VPU.Sparsify([[VAL9]])
    // CHECK:       [[VAL11:%.+]] = VPU.Desparsify([[VAL10]]

    // CHECK:       [[VAL12:%.+]] = VPU.Sparsify([[VAL4]])
    // CHECK:       [[VAL13:%.+]] = VPU.Desparsify([[VAL12]]

    // CHECK:       [[VAL14:%.+]] = VPU.NCE.Convolution([[VAL13]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL15:%.+]] = VPU.Sparsify([[VAL14]])
    // CHECK:       [[VAL16:%.+]] = VPU.Desparsify([[VAL15]]

    // CHECK:       return [[VAL11]], [[VAL16]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @WrapMultipleMixedConsumers(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%1, %1) {
                op_type = "ADD",
                ppe = {clamp_high = 2147483647, clamp_low = -2147483648, lrelu_mult = 1, lrelu_shift = 0, mode = "ADD"}
            } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %3 = VPU.MaxPool(%1) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %2, %3 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
    // CHECK:       [[VAL1:%.+]] = VPU.Desparsify([[VAL0]]

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.Sparsify([[VAL2]])
    // CHECK:       [[VAL4:%.+]] = VPU.Desparsify([[VAL3]]

    // CHECK:       [[VAL5:%.+]] = VPU.NCE.Eltwise([[VAL4]], [[VAL4]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL6:%.+]] = VPU.Sparsify([[VAL5]])
    // CHECK:       [[VAL7:%.+]] = VPU.Desparsify([[VAL6]]

    // CHECK:       [[VAL8:%.+]] = VPU.MaxPool([[VAL4]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL7]], [[VAL8]]
}
