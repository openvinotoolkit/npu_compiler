//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --handle-eltwise-with-small-height %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @HandleEltwiseWithSmallHeight
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x1920x3x1080xf16, {order = #NHWC}>
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x1920x3x1080xf16, {order = #NHWC}>)

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

func.func @HandleEltwiseWithSmallHeight(%arg0: tensor<1x1920x3x1080xf16, {order = #NHWC}>, %arg1: tensor<1x1920x3x1080xf16, {order = #NHWC}>) -> tensor<1x1920x3x1080xf32, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1920x3x1080xf16, {order = #NHWC}>, tensor<1x1920x3x1080xf16, {order = #NHWC}> -> tensor<1x1920x3x1080xf16, {order = #NHWC}>
    %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x1920x3x1080xf16, {order = #NHWC}> -> tensor<1x1920x3x1080xf32, {order = #NHWC}>
    return %1 : tensor<1x1920x3x1080xf32, {order = #NHWC}>

    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 1920, 60, 54]} : tensor<1x1920x3x1080xf16, {order = #NHWC}> -> tensor<1x1920x60x54xf16, {order = #NHWC}>

    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 1920, 60, 54]} : tensor<1x1920x3x1080xf16, {order = #NHWC}> -> tensor<1x1920x60x54xf16, {order = #NHWC}>

    // CHECK: [[ADD:%.+]] = IE.Add([[RESHAPE0]], [[RESHAPE1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1920x60x54xf16, {order = #NHWC}>, tensor<1x1920x60x54xf16, {order = #NHWC}> -> tensor<1x1920x60x54xf16, {order = #NHWC}>

    // CHECK: [[RESHAPE2:%.+]] = IE.AffineReshape([[ADD]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 1920, 3, 1080]} : tensor<1x1920x60x54xf16, {order = #NHWC}> -> tensor<1x1920x3x1080xf16, {order = #NHWC}>

    // CHECK: [[CONVERT:%.+]] = IE.Convert([[RESHAPE2]]) {dstElemType = f32} : tensor<1x1920x3x1080xf16, {order = #NHWC}> -> tensor<1x1920x3x1080xf32, {order = #NHWC}>
    // CHECK: return [[CONVERT]]  : tensor<1x1920x3x1080xf32, {order = #NHWC}>
}
}
