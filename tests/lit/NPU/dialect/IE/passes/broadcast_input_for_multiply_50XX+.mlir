//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --broadcast-input-for-multiply  %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BroadcastInputForMultiplyNCHW
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x24x1x64xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x1x64xf16>
func.func @BroadcastInputForMultiplyNCHW(%arg0: tensor<1x24x1x64xf16>, %arg1: tensor<1x1x1x64xf16>) -> tensor<1x24x1x64xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x24x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x24x1x64xf16>

    return %0 : tensor<1x24x1x64xf16>

    // CHECK-DAG:   [[TARGET_SHAPE:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 24, 1, 64]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[LHS:%.+]] = IE.PermuteCast([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x24x1x64xf16> -> tensor<1x64x24x1xf16, {order = #NHWC}>
    // CHECK:       [[BROADCAST:%.+]] = IE.Broadcast([[INPUT_1]], [[TARGET_SHAPE]]) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x1x1x64xf16>, tensor<4xsi32> -> tensor<1x24x1x64xf16>

    // CHECK:       [[RHS:%.+]] = IE.PermuteCast([[BROADCAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x24x1x64xf16> -> tensor<1x64x24x1xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[LHS]], [[RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x24x1xf16, {order = #NHWC}>, tensor<1x64x24x1xf16, {order = #NHWC}> -> tensor<1x64x24x1xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x24x1xf16, {order = #NHWC}> -> tensor<1x24x1x64xf16>

    // CHECK:       return [[OUTPUT_CAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @BroadcastInputForMultiplyNHCW
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x32x16x128xf16, {order = #NHCW}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x32x1x128xf16, {order = #NHCW}>
func.func @BroadcastInputForMultiplyNHCW(%arg0: tensor<1x32x16x128xf16, {order = #NHCW}>, %arg1: tensor<1x32x1x128xf16, {order = #NHCW}>) -> tensor<1x32x16x128xf16, {order = #NHCW}> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
            tensor<1x32x16x128xf16, {order = #NHCW}>,
            tensor<1x32x1x128xf16, {order = #NHCW}>
            -> tensor<1x32x16x128xf16, {order = #NHCW}>

    return %0 : tensor<1x32x16x128xf16, {order = #NHCW}>

    // CHECK-DAG:   [[TARGET_SHAPE:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 16, 32, 128]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[LHS:%.+]] = IE.PermuteCast([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x16x128xf16, {order = #NHCW}> -> tensor<1x128x16x32xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[INPUT_1]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x1x128xf16, {order = #NHCW}> -> tensor<1x1x32x128xf16>
    // CHECK:       [[BROADCAST:%.+]] = IE.Broadcast([[PERMUTECAST]], [[TARGET_SHAPE]]) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x1x32x128xf16>, tensor<4xsi32> -> tensor<1x16x32x128xf16>

    // CHECK:       [[RHS:%.+]] = IE.PermuteCast([[BROADCAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x32x128xf16> -> tensor<1x128x16x32xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[LHS]], [[RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x16x32xf16, {order = #NHWC}>, tensor<1x128x16x32xf16, {order = #NHWC}> -> tensor<1x128x16x32xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x128x16x32xf16, {order = #NHWC}> -> tensor<1x32x16x128xf16, {order = #NHCW}>

    // CHECK:       return [[OUTPUT_CAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotBroadcastInnermostDimInput
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x40x64xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x101x40x64xf16, {order = #NHWC}>
func.func @NotBroadcastInnermostDimInput(%arg0: tensor<1x1x40x64xf16, {order = #NHWC}>, %arg1: tensor<1x101x40x64xf16, {order = #NHWC}>) -> tensor<1x101x40x64xf16, {order = #NHWC}> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf16, {order = #NHWC}>, tensor<1x101x40x64xf16, {order = #NHWC}> -> tensor<1x101x40x64xf16, {order = #NHWC}>

    return %0 : tensor<1x101x40x64xf16, {order = #NHWC}>

    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf16, {order = #NHWC}>, tensor<1x101x40x64xf16, {order = #NHWC}> -> tensor<1x101x40x64xf16, {order = #NHWC}>
    // CHECK:       return [[MULTIPLY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotBroadcastTheNonHighestDimInput
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x16x77x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x16x77x77xf16, {order = #NHWC}>
func.func @NotBroadcastTheNonHighestDimInput(%arg0: tensor<1x16x77x1xf16, {order = #NHWC}>, %arg1: tensor<1x16x77x77xf16, {order = #NHWC}>) -> tensor<1x16x77x77xf16, {order = #NHWC}> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x77x1xf16, {order = #NHWC}>, tensor<1x16x77x77xf16, {order = #NHWC}> -> tensor<1x16x77x77xf16, {order = #NHWC}>

    return %0 : tensor<1x16x77x77xf16, {order = #NHWC}>

    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x77x1xf16, {order = #NHWC}>, tensor<1x16x77x77xf16, {order = #NHWC}> -> tensor<1x16x77x77xf16, {order = #NHWC}>
    // CHECK:       return [[MULTIPLY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotBroadcastUnalignedInnermostDim
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x24x1x3xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x1x3xf16>
func.func @NotBroadcastUnalignedInnermostDim(%arg0: tensor<1x24x1x3xf16>, %arg1: tensor<1x1x1x3xf16>) -> tensor<1x24x1x3xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x24x1x3xf16>, tensor<1x1x1x3xf16> -> tensor<1x24x1x3xf16>

    return %0 : tensor<1x24x1x3xf16>

    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x24x1x3xf16>, tensor<1x1x1x3xf16> -> tensor<1x24x1x3xf16>
    // CHECK:       return [[MULTIPLY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @BroadcastInputForSplatMultiplyNCHW
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x1x1xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x512x1x1xf16>
func.func @BroadcastInputForSplatMultiplyNCHW(%arg0: tensor<1x1x1x1xf16>, %arg1: tensor<1x512x1x1xf16>) -> tensor<1x512x1x1xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x1x1xf16>

    return %0 : tensor<1x512x1x1xf16>

    // CHECK-DAG:   [[TARGET_SHAPE:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 512, 1, 1]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[BROADCAST:%.+]] = IE.Broadcast([[INPUT_0]], [[TARGET_SHAPE]]) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<4xsi32> -> tensor<1x512x1x1xf16>

    // CHECK:       [[LHS:%.+]] = IE.PermuteCast([[BROADCAST]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x512x1x1xf16> -> tensor<1x512x1x1xf16, {order = #NHWC}>
    // CHECK:       [[RHS:%.+]] = IE.PermuteCast([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x512x1x1xf16> -> tensor<1x512x1x1xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[LHS]], [[RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x1x1xf16, {order = #NHWC}>, tensor<1x512x1x1xf16, {order = #NHWC}> -> tensor<1x512x1x1xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x512x1x1xf16, {order = #NHWC}> -> tensor<1x512x1x1xf16>

    // CHECK:       return [[OUTPUT_CAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BroadcastInputForSplatMultiplyNHWC
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x512x1x1xf16, {order = #NHWC}>
func.func @BroadcastInputForSplatMultiplyNHWC(%arg0: tensor<1x1x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x512x1x1xf16, {order = #NHWC}>) -> tensor<1x512x1x1xf16, {order = #NHWC}> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16, {order = #NHWC}>, tensor<1x512x1x1xf16, {order = #NHWC}> -> tensor<1x512x1x1xf16, {order = #NHWC}>

    return %0 : tensor<1x512x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[TARGET_SHAPE:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 1, 1, 512]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT_0]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16>
    // CHECK:       [[BROADCAST:%.+]] = IE.Broadcast([[PERMUTE_CAST]], [[TARGET_SHAPE]]) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<4xsi32> -> tensor<1x1x1x512xf16>
    // CHECK:       [[LHS:%.+]] = IE.PermuteCast([[BROADCAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x1x512xf16> -> tensor<1x512x1x1xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[LHS]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x1x1xf16, {order = #NHWC}>, tensor<1x512x1x1xf16, {order = #NHWC}> -> tensor<1x512x1x1xf16, {order = #NHWC}>

    // CHECK:       return [[MULTIPLY]]
}
