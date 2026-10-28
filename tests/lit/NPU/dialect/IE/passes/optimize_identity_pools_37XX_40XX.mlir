//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-identity-pools %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @NotFuseAddIdentityAvgPoolAsUnsupportedPostOp([[ARG0:%.+]]: tensor<1x16x320x320x!qElemType, {order = #NHWC}>) -> tensor<1x16x320x320xf16, {order = #NHWC}> {
func.func @NotFuseAddIdentityAvgPoolAsUnsupportedPostOp(%arg0 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>) -> tensor<1x16x320x320xf16, {order = #NHWC}> {
    %add = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320x!qElemType, {order = #NHWC}>, tensor<1x16x320x320x!qElemType, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>

    %ave_pool = IE.AvgPool(%add) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}  :
                  tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>

    return %ave_pool : tensor<1x16x320x320xf16, {order = #NHWC}>
    // CHECK:       IE.Add([[ARG0]], [[ARG0]])
    // CHECK:       IE.AvgPool
    // CHECK-SAME:       post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>
    // CHECK-SAME:       tensor<1x16x320x320xf16, {order = #NHWC}>

}

// -----

// CHECK-LABEL: @DoNotDropNonTransferablePostOpOfIdentityAvgPool
func.func @DoNotDropNonTransferablePostOpOfIdentityAvgPool(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64},
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Tanh<>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // sprLUT is HW feature not supported by MTL/LNL, so using monotonically increasing activation non-linear function
    // to fuse post_op into Convolution is not possible.

    // CHECK:       [[CONV:%.+]] = IE.Convolution
    // CHECK-NOT:      clamp
    // CHECK-NOT:      post_op
    // CHECK:       IE.AvgPool([[CONV]])
    // CHECK-SAME:     clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64},
    // CHECK-SAME:     kernel_size = [1, 1],
    // CHECK-SAME:     post_op = #IE.Tanh<>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @FuseIdentityAvgPoolIntoQuantizedAvgPoolAsClamp([[ARG0:%.+]]: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
func.func @FuseIdentityAvgPoolIntoQuantizedAvgPoolAsClamp(%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) {clamp = {max = 5.000000e+00 : f64, min = -1.000000e+00 : f64}, exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    // Clamp with a non-zero minimum for quantized operations is supported

    // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[ARG0]])
    // CHECK-SAME:      clamp = {max = 5.000000e+00 : f64, min = -1.000000e+00 : f64}
    // CHECK-SAME:      tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    // CHECK:      return [[AVG_POOL]] : tensor<1x16x320x320x!qElemType, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @FuseIdentityAvgPoolWithQuantizedAddIntersectNegativeClamp([[ARG0:%.+]]: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
func.func @FuseIdentityAvgPoolWithQuantizedAddIntersectNegativeClamp(%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) {clamp = {max = 5.000000e+00 : f64, min = -1.000000e+00 : f64}, exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, clamp = {max = 2.000000e+00 : f64, min = -2.000000e+00 : f64} } :
              tensor<1x16x320x320xf16, {order = #NHWC}>, tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    // Clamp with a non-zero minimum for quantized operations is supported
    // The clamps are intersected: clamp1 = [-1, 5], clamp2 = [-2, 2] becomes clamp = [-1, 2]

    // CHECK:       [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK-SAME:      clamp = {max = 2.000000e+00 : f64, min = -1.000000e+00 : f64}
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>

    // CHECK:      return [[ADD]] : tensor<1x16x320x320x!qElemType, {order = #NHWC}>
}
