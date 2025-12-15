//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --handle-eltwise-with-small-height %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @HandleEltwiseWithSmallHeightByReshapingW
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x1920x3x1080xf16, {order = #NHWC}>
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x1920x3x1080xf16, {order = #NHWC}>)

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz{
    config.MemoryResource 1000000 bytes of @CMX_NN
}

func.func @HandleEltwiseWithSmallHeightByReshapingW(%arg0: tensor<1x1920x3x1080xf16, {order = #NHWC}>, %arg1: tensor<1x1920x3x1080xf16, {order = #NHWC}>) -> tensor<1x1920x3x1080xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1920x3x1080xf16, {order = #NHWC}>, tensor<1x1920x3x1080xf16, {order = #NHWC}> -> tensor<1x1920x3x1080xf16, {order = #NHWC}>
    return %0 : tensor<1x1920x3x1080xf16, {order = #NHWC}>

    // CHECK: [[RESHAPE0:%.+]] = IE.ShapeCast {shape = [1, 1920, 810, 4]}
    // CHECK-SAME: inputs([[INPUT0]] : tensor<1x1920x3x1080xf16, {order = #NHWC}>) -> tensor<1x1920x810x4xf16, {order = #NHWC}>

    // CHECK: [[RESHAPE1:%.+]] = IE.ShapeCast {shape = [1, 1920, 810, 4]}
    // CHECK-SAME: inputs([[INPUT1]] : tensor<1x1920x3x1080xf16, {order = #NHWC}>) -> tensor<1x1920x810x4xf16, {order = #NHWC}>

    // CHECK: [[ADD:%.+]] = IE.Add([[RESHAPE0]], [[RESHAPE1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1920x810x4xf16, {order = #NHWC}>, tensor<1x1920x810x4xf16, {order = #NHWC}> -> tensor<1x1920x810x4xf16, {order = #NHWC}>

    // CHECK: [[RESHAPE2:%.+]] = IE.ShapeCast {shape = [1, 1920, 3, 1080]}
    // CHECK-SAME: inputs([[ADD]] : tensor<1x1920x810x4xf16, {order = #NHWC}>) -> tensor<1x1920x3x1080xf16, {order = #NHWC}>

    // CHECK: return [[RESHAPE2]]  : tensor<1x1920x3x1080xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NotHandleEltwiseWithSmallHeightByReshapingW
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x16x3x1080xf16, {order = #NHWC}>
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x16x3x1080xf16, {order = #NHWC}>)

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz{
    config.MemoryResource 1000000 bytes of @CMX_NN
}

func.func @NotHandleEltwiseWithSmallHeightByReshapingW(%arg0: tensor<1x16x3x1080xf16, {order = #NHWC}>, %arg1: tensor<1x16x3x1080xf16, {order = #NHWC}>) -> tensor<1x16x3x1080xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x1080xf16, {order = #NHWC}>, tensor<1x16x3x1080xf16, {order = #NHWC}> -> tensor<1x16x3x1080xf16, {order = #NHWC}>
    return %0 : tensor<1x16x3x1080xf16, {order = #NHWC}>

    // CHECK-NOT: IE.ShapeCast
}
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @HandleEltwiseWithSmallHeightByReshapingC
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x200000x1x1xf16, {order = #NHWC}>
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x200000x1x1xf16, {order = #NHWC}>)

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz{
    config.MemoryResource 1000000 bytes of @CMX_NN
}

func.func @HandleEltwiseWithSmallHeightByReshapingC(%arg0: tensor<1x200000x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x200000x1x1xf16, {order = #NHWC}>) -> tensor<1x200000x1x1xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x200000x1x1xf16, {order = #NHWC}>, tensor<1x200000x1x1xf16, {order = #NHWC}> -> tensor<1x200000x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x200000x1x1xf16, {order = #NHWC}>

    // CHECK: [[RESHAPE0:%.+]] = IE.ShapeCast {shape = [1, 16, 3125, 4]}
    // CHECK-SAME: inputs([[INPUT0]] : tensor<1x200000x1x1xf16, {order = #NHWC}>) -> tensor<1x16x3125x4xf16, {order = #NHWC}>

    // CHECK: [[RESHAPE1:%.+]] = IE.ShapeCast {shape = [1, 16, 3125, 4]}
    // CHECK-SAME: inputs([[INPUT1]] : tensor<1x200000x1x1xf16, {order = #NHWC}>) -> tensor<1x16x3125x4xf16, {order = #NHWC}>

    // CHECK: [[ADD:%.+]] = IE.Add([[RESHAPE0]], [[RESHAPE1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3125x4xf16, {order = #NHWC}>, tensor<1x16x3125x4xf16, {order = #NHWC}> -> tensor<1x16x3125x4xf16, {order = #NHWC}>

    // CHECK: [[RESHAPE2:%.+]] = IE.ShapeCast {shape = [1, 200000, 1, 1]}
    // CHECK-SAME: inputs([[ADD]] : tensor<1x16x3125x4xf16, {order = #NHWC}>) -> tensor<1x200000x1x1xf16, {order = #NHWC}>

    // CHECK: return [[RESHAPE2]]  : tensor<1x200000x1x1xf16, {order = #NHWC}>
}
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @NotHandleEltwiseWithSmallHeightByReshapingC
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x200000x1x1xf16, {order = #NHWC}>
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x200000x1x1xf16, {order = #NHWC}>)

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz{
    config.MemoryResource 1000000 bytes of @CMX_NN
}

func.func @NotHandleEltwiseWithSmallHeightByReshapingC(%arg0: tensor<1x200000x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x200000x1x1xf16, {order = #NHWC}>) -> tensor<1x200000x1x1xf16, {order = #NCHW}> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x200000x1x1xf16, {order = #NHWC}>, tensor<1x200000x1x1xf16, {order = #NHWC}> -> tensor<1x200000x1x1xf16, {order = #NCHW}>
    return %0 : tensor<1x200000x1x1xf16, {order = #NCHW}>

    // CHECK-NOT: IE.ShapeCast
}
}
