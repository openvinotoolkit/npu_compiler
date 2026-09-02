//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --broadcast-input-for-multiply="broadcast-input-for-multiply=true"  %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BroadcastInputForMultiplyNCHW
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x50x1x32xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x1x32xf16>
func.func @BroadcastInputForMultiplyNCHW(%arg0: tensor<1x50x1x32xf16>, %arg1: tensor<1x1x1x32xf16>) -> tensor<1x50x1x32xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x50x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x50x1x32xf16>

    return %0 : tensor<1x50x1x32xf16>

    // CHECK:       [[LHS:%.+]] = IE.PermuteCast([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x50x1x32xf16> -> tensor<1x32x50x1xf16, {order = #NHWC}>
    // CHECK:       [[BROADCAST:%.+]] = IE.Tile([[INPUT_1]]) {repeats_values = [1, 50, 1, 1]} : tensor<1x1x1x32xf16> -> tensor<1x50x1x32xf16>

    // CHECK:       [[RHS:%.+]] = IE.PermuteCast([[BROADCAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x50x1x32xf16> -> tensor<1x32x50x1xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[LHS]], [[RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x50x1xf16, {order = #NHWC}>, tensor<1x32x50x1xf16, {order = #NHWC}> -> tensor<1x32x50x1xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x50x1xf16, {order = #NHWC}> -> tensor<1x50x1x32xf16>

    // CHECK:       return [[OUTPUT_CAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @BroadcastInputForMultiplyNHCW
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x113x15x32xf16, {order = #NHCW}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x113x1x32xf16, {order = #NHCW}>
func.func @BroadcastInputForMultiplyNHCW(%arg0: tensor<1x113x15x32xf16, {order = #NHCW}>, %arg1: tensor<1x113x1x32xf16, {order = #NHCW}>) -> tensor<1x113x15x32xf16, {order = #NHCW}> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
            tensor<1x113x15x32xf16, {order = #NHCW}>,
            tensor<1x113x1x32xf16, {order = #NHCW}>
            -> tensor<1x113x15x32xf16, {order = #NHCW}>

    return %0 : tensor<1x113x15x32xf16, {order = #NHCW}>

    // CHECK:       [[LHS:%.+]] = IE.PermuteCast([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x113x15x32xf16, {order = #NHCW}> -> tensor<1x32x15x113xf16, {order = #NHWC}>
    // CHECK:       [[BROADCAST:%.+]] = IE.Tile([[INPUT_1]]) {repeats_values = [1, 1, 15, 1]} : tensor<1x113x1x32xf16, {order = #NHCW}> -> tensor<1x113x15x32xf16, {order = #NHCW}>

    // CHECK:       [[RHS:%.+]] = IE.PermuteCast([[BROADCAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x113x15x32xf16, {order = #NHCW}> -> tensor<1x32x15x113xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[LHS]], [[RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x15x113xf16, {order = #NHWC}>, tensor<1x32x15x113xf16, {order = #NHWC}> -> tensor<1x32x15x113xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x32x15x113xf16, {order = #NHWC}> -> tensor<1x113x15x32xf16, {order = #NHCW}>

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

// CHECK-LABEL: @BroadcastWhenDiffDimIsOneRhs15
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x113x15x32xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x113x1x32xf16>
func.func @BroadcastWhenDiffDimIsOneRhs15(%arg0: tensor<1x113x15x32xf16>, %arg1: tensor<1x113x1x32xf16>) -> tensor<1x113x15x32xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x113x15x32xf16>, tensor<1x113x1x32xf16> -> tensor<1x113x15x32xf16>

    return %0 : tensor<1x113x15x32xf16>

    // CHECK:       [[LHS:%.+]] = IE.PermuteCast([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x113x15x32xf16> -> tensor<1x32x113x15xf16, {order = #NHWC}>
    // CHECK:       [[BROADCAST:%.+]] = IE.Tile([[INPUT_1]]) {repeats_values = [1, 1, 15, 1]} : tensor<1x113x1x32xf16> -> tensor<1x113x15x32xf16>
    // CHECK:       [[RHS:%.+]] = IE.PermuteCast([[BROADCAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x113x15x32xf16> -> tensor<1x32x113x15xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[LHS]], [[RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x113x15xf16, {order = #NHWC}>, tensor<1x32x113x15xf16, {order = #NHWC}> -> tensor<1x32x113x15xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x113x15xf16, {order = #NHWC}> -> tensor<1x113x15x32xf16>
    // CHECK:       return [[OUTPUT_CAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BroadcastWhenDiffDimIsOneRhs5
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x113x5x32xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x113x1x32xf16>
func.func @BroadcastWhenDiffDimIsOneRhs5(%arg0: tensor<1x113x5x32xf16>, %arg1: tensor<1x113x1x32xf16>) -> tensor<1x113x5x32xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x113x5x32xf16>, tensor<1x113x1x32xf16> -> tensor<1x113x5x32xf16>

    return %0 : tensor<1x113x5x32xf16>

    // CHECK:       [[LHS:%.+]] = IE.PermuteCast([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x113x5x32xf16> -> tensor<1x32x113x5xf16, {order = #NHWC}>
    // CHECK:       [[BROADCAST:%.+]] = IE.Tile([[INPUT_1]]) {repeats_values = [1, 1, 5, 1]} : tensor<1x113x1x32xf16> -> tensor<1x113x5x32xf16>
    // CHECK:       [[RHS:%.+]] = IE.PermuteCast([[BROADCAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x113x5x32xf16> -> tensor<1x32x113x5xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[LHS]], [[RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x113x5xf16, {order = #NHWC}>, tensor<1x32x113x5xf16, {order = #NHWC}> -> tensor<1x32x113x5xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x113x5xf16, {order = #NHWC}> -> tensor<1x113x5x32xf16>
    // CHECK:       return [[OUTPUT_CAST]]
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

    // CHECK:       [[BROADCAST:%.+]] = IE.Tile([[INPUT_0]]) {repeats_values = [1, 512, 1, 1]} : tensor<1x1x1x1xf16> -> tensor<1x512x1x1xf16>

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

    // CHECK:       [[BROADCAST:%.+]] = IE.Tile([[INPUT_0]]) {repeats_values = [1, 512, 1, 1]} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x512x1x1xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[BROADCAST]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x1x1xf16, {order = #NHWC}>, tensor<1x512x1x1xf16, {order = #NHWC}> -> tensor<1x512x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[MULTIPLY]]
}

// -----

// CHECK-LABEL: @FuseMultiplyBroadcastRightInput
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x64x64x128xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x64x1xf16>
func.func @FuseMultiplyBroadcastRightInput(%arg0: tensor<1x64x64x128xf16>, %arg1: tensor<1x64x64x1xf16>) -> tensor<1x64x64x128xf16> {
    %0 = IE.Tile(%arg1) {repeats_values = [1, 1, 1, 128]} : tensor<1x64x64x1xf16> -> tensor<1x64x64x128xf16>
    %1 = IE.Multiply(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x128xf16>, tensor<1x64x64x128xf16> -> tensor<1x64x64x128xf16>
    return %1 : tensor<1x64x64x128xf16>

    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x128xf16>, tensor<1x64x64x1xf16> -> tensor<1x64x64x128xf16>
    // CHECK:       return [[MULTIPLY]]
}

// -----

// CHECK-LABEL: @FuseMultiplyBroadcastLeftInput
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x64x64x1xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x64x128xf16>
func.func @FuseMultiplyBroadcastLeftInput(%arg0: tensor<1x64x64x1xf16>, %arg1: tensor<1x64x64x128xf16>) -> tensor<1x64x64x128xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 128]} : tensor<1x64x64x1xf16> -> tensor<1x64x64x128xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x128xf16>, tensor<1x64x64x128xf16> -> tensor<1x64x64x128xf16>
    return %1 : tensor<1x64x64x128xf16>

    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x1xf16>, tensor<1x64x64x128xf16> -> tensor<1x64x64x128xf16>
    // CHECK:       return [[MULTIPLY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BroadcastWhenOutputCJustBelowRegressionThreshold
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1023x1x32xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x1x32xf16>
func.func @BroadcastWhenOutputCJustBelowRegressionThreshold(%arg0: tensor<1x1023x1x32xf16>, %arg1: tensor<1x1x1x32xf16>) -> tensor<1x1023x1x32xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1023x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x1023x1x32xf16>

    return %0 : tensor<1x1023x1x32xf16>

    // CHECK:       [[LHS:%.+]] = IE.PermuteCast([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1023x1x32xf16> -> tensor<1x32x1023x1xf16, {order = #NHWC}>
    // CHECK:       [[BROADCAST:%.+]] = IE.Tile([[INPUT_1]]) {repeats_values = [1, 1023, 1, 1]} : tensor<1x1x1x32xf16> -> tensor<1x1023x1x32xf16>

    // CHECK:       [[RHS:%.+]] = IE.PermuteCast([[BROADCAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1023x1x32xf16> -> tensor<1x32x1023x1xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[LHS]], [[RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1023x1xf16, {order = #NHWC}>, tensor<1x32x1023x1xf16, {order = #NHWC}> -> tensor<1x32x1023x1xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x1023x1xf16, {order = #NHWC}> -> tensor<1x1023x1x32xf16>

    // CHECK:       return [[OUTPUT_CAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotBroadcastWhenOutputCAtRegressionThreshold
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1024x1x32xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x1x32xf16>
func.func @NotBroadcastWhenOutputCAtRegressionThreshold(%arg0: tensor<1x1024x1x32xf16>, %arg1: tensor<1x1x1x32xf16>) -> tensor<1x1024x1x32xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x1024x1x32xf16>

    return %0 : tensor<1x1024x1x32xf16>

    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x1024x1x32xf16>
    // CHECK:       return [[MULTIPLY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BroadcastChannelWhenSpatialSizeAtThreshold
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x512x64x32xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x64x32xf16>
func.func @BroadcastChannelWhenSpatialSizeAtThreshold(%arg0: tensor<1x512x64x32xf16>, %arg1: tensor<1x1x64x32xf16>) -> tensor<1x512x64x32xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x64x32xf16>, tensor<1x1x64x32xf16> -> tensor<1x512x64x32xf16>

    return %0 : tensor<1x512x64x32xf16>

    // CHECK:       [[LHS:%.+]] = IE.PermuteCast([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x512x64x32xf16> -> tensor<1x32x512x64xf16, {order = #NHWC}>
    // CHECK:       [[BROADCAST:%.+]] = IE.Tile([[INPUT_1]]) {repeats_values = [1, 512, 1, 1]} : tensor<1x1x64x32xf16> -> tensor<1x512x64x32xf16>
    // CHECK:       [[RHS:%.+]] = IE.PermuteCast([[BROADCAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x512x64x32xf16> -> tensor<1x32x512x64xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[LHS]], [[RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x512x64xf16, {order = #NHWC}>, tensor<1x32x512x64xf16, {order = #NHWC}> -> tensor<1x32x512x64xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x512x64xf16, {order = #NHWC}> -> tensor<1x512x64x32xf16>

    // CHECK:       return [[OUTPUT_CAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotBroadcastChannelWhenSpatialSizeAboveThreshold
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x512x64x64xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x64x64xf16>
func.func @NotBroadcastChannelWhenSpatialSizeAboveThreshold(%arg0: tensor<1x512x64x64xf16>, %arg1: tensor<1x1x64x64xf16>) -> tensor<1x512x64x64xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x64x64xf16>, tensor<1x1x64x64xf16> -> tensor<1x512x64x64xf16>

    return %0 : tensor<1x512x64x64xf16>

    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x64x64xf16>, tensor<1x1x64x64xf16> -> tensor<1x512x64x64xf16>
    // CHECK:       return [[MULTIPLY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotBroadcastWhenShapesMatch
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x35840x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x35840x1x1xf16, {order = #NHWC}>
func.func @NotBroadcastWhenShapesMatch(%arg0: tensor<1x35840x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x35840x1x1xf16, {order = #NHWC}>) -> tensor<1x35840x1x1xf16, {order = #NHWC}> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x35840x1x1xf16, {order = #NHWC}>, tensor<1x35840x1x1xf16, {order = #NHWC}> -> tensor<1x35840x1x1xf16, {order = #NHWC}>

    return %0 : tensor<1x35840x1x1xf16, {order = #NHWC}>

    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x35840x1x1xf16, {order = #NHWC}>, tensor<1x35840x1x1xf16, {order = #NHWC}> -> tensor<1x35840x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[MULTIPLY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NotBroadcastDynamicMultiply
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 10]> : tensor<4xsi64>, order = #NCHW}>
func.func @NotBroadcastDynamicMultiply(%arg0: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 10]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 10]> : tensor<4xsi64>, order = #NCHW}> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 10]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x1xf32> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 10]> : tensor<4xsi64>, order = #NCHW}>

    return %0 : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 10]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:       [[CONST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 10]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x1xf32> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 10]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       return [[MULTIPLY]]
}
