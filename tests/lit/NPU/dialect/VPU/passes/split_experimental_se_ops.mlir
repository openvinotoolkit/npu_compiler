//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --split-se-ops="se-experimental-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @RollSplitWithLargeSize([[INPUT_DATA:%.+]]: tensor<1x128x160x160xf16, {order = #NHWC}>) -> tensor<1x128x160x160xf16, {order = #NHWC}> {
func.func @RollSplitWithLargeSize(%arg0: tensor<1x128x160x160xf16, {order = #NHWC}>) -> tensor<1x128x160x160xf16, {order = #NHWC}> {
    %shift = const.Declare tensor<2xsi32> = dense<[6, 5]> : tensor<2xsi32>
    %axes = const.Declare tensor<2xsi32> = dense<[2, 3]> : tensor<2xsi32>
    %roll = VPU.Roll(%arg0, %shift, %axes) : tensor<1x128x160x160xf16, {order = #NHWC}>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x128x160x160xf16, {order = #NHWC}>
    return %roll : tensor<1x128x160x160xf16, {order = #NHWC}>

    // CHECK-DAG: [[AXES:%.+]] = const.Declare tensor<2xsi32> = dense<[2, 3]> : tensor<2xsi32>
    // CHECK-DAG: [[SHIFT_0:%.+]] = const.Declare tensor<2xsi32> = dense<[0, 5]> : tensor<2xsi32>
    // CHECK-DAG: [[SHIFT_1:%.+]] = const.Declare tensor<2xsi32> = dense<[6, 0]> : tensor<2xsi32>
    // CHECK: [[ROLL_0:%.+]] = VPU.Roll([[INPUT_DATA]], [[SHIFT_0]], [[AXES]]) : tensor<1x128x160x160xf16, {order = #NHWC}>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x128x160x160xf16, {order = #NHWC}>
    // CHECK: [[ROLL_1:%.+]] = VPU.Roll([[ROLL_0]], [[SHIFT_1]], [[AXES]]) : tensor<1x128x160x160xf16, {order = #NHWC}>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x128x160x160xf16, {order = #NHWC}>
    // CHECK: return [[ROLL_1]] : tensor<1x128x160x160xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @RollSplitWithLargeSizeAndSingleShift([[INPUT_DATA:%.+]]: tensor<1x128x160x160xf16, {order = #NHWC}>) -> tensor<1x128x160x160xf16, {order = #NHWC}> {
func.func @RollSplitWithLargeSizeAndSingleShift(%arg0: tensor<1x128x160x160xf16, {order = #NHWC}>) -> tensor<1x128x160x160xf16, {order = #NHWC}> {
    %shift = const.Declare tensor<1xsi32> = dense<[6]> : tensor<1xsi32>
    %axes = const.Declare tensor<2xsi32> = dense<[2, 3]> : tensor<2xsi32>
    %roll = VPU.Roll(%arg0, %shift, %axes) : tensor<1x128x160x160xf16, {order = #NHWC}>, tensor<1xsi32>, tensor<2xsi32> -> tensor<1x128x160x160xf16, {order = #NHWC}>
    return %roll : tensor<1x128x160x160xf16, {order = #NHWC}>

    // CHECK-DAG: [[AXES:%.+]] = const.Declare tensor<2xsi32> = dense<[2, 3]> : tensor<2xsi32>
    // CHECK-DAG: [[SHIFT_0:%.+]] = const.Declare tensor<2xsi32> = dense<[0, 6]> : tensor<2xsi32>
    // CHECK-DAG: [[SHIFT_1:%.+]] = const.Declare tensor<2xsi32> = dense<[6, 0]> : tensor<2xsi32>
    // CHECK: [[ROLL_0:%.+]] = VPU.Roll([[INPUT_DATA]], [[SHIFT_0]], [[AXES]]) : tensor<1x128x160x160xf16, {order = #NHWC}>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x128x160x160xf16, {order = #NHWC}>
    // CHECK: [[ROLL_1:%.+]] = VPU.Roll([[ROLL_0]], [[SHIFT_1]], [[AXES]]) : tensor<1x128x160x160xf16, {order = #NHWC}>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x128x160x160xf16, {order = #NHWC}>
    // CHECK: return [[ROLL_1]] : tensor<1x128x160x160xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @NotRollSplitWithSmallSize([[INPUT_DATA:%.+]]: tensor<1x16x16x160xf16, {order = #NHWC}>) -> tensor<1x16x16x160xf16, {order = #NHWC}> {
func.func @NotRollSplitWithSmallSize(%arg0: tensor<1x16x16x160xf16, {order = #NHWC}>) -> tensor<1x16x16x160xf16, {order = #NHWC}> {
    %shift = const.Declare tensor<2xsi32> = dense<[6, 5]> : tensor<2xsi32>
    %axes = const.Declare tensor<2xsi32> = dense<[2, 3]> : tensor<2xsi32>
    %roll = VPU.Roll(%arg0, %shift, %axes) : tensor<1x16x16x160xf16, {order = #NHWC}>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x16x16x160xf16, {order = #NHWC}>
    return %roll : tensor<1x16x16x160xf16, {order = #NHWC}>

    // CHECK-DAG: [[AXES:%.+]] = const.Declare tensor<2xsi32> = dense<[2, 3]> : tensor<2xsi32>
    // CHECK-DAG: [[SHIFT:%.+]] = const.Declare tensor<2xsi32> = dense<[6, 5]> : tensor<2xsi32>
    // CHECK: [[ROLL:%.+]] = VPU.Roll([[INPUT_DATA]], [[SHIFT]], [[AXES]]) : tensor<1x16x16x160xf16, {order = #NHWC}>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x16x16x160xf16, {order = #NHWC}>
    // CHECK: return [[ROLL]] : tensor<1x16x16x160xf16, {order = #NHWC}>
}
