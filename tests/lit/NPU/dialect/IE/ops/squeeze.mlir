//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @Eliminate
func.func @Eliminate(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = IE.Squeeze(%arg0) { axes_value = [] } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: IE.Squeeze
    // CHECK:     return %arg0
}

// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<4x4xf32> {
    %0 = const.Declare tensor<1x1x4x4xf32> = dense<1.0> : tensor<1x1x4x4xf32>
    %1 = IE.Squeeze(%0) { axes_value = [0, 1] } : tensor<1x1x4x4xf32> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK-DAG:       [[VAL0:%.+]] = const.Declare tensor<4x4xf32> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x1x4x4xf32>, [#const.Reshape<[4, 4]>]
    // CHECK-NOT:   IE.Squeeze
    // CHECK:       return [[VAL0]]
}

// CHECK-LABEL: @FuseWithReshape
func.func @FuseWithReshape(%arg0: tensor<16x1xf32>) -> tensor<4x4xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 1, 4, 4] } : tensor<16x1xf32> -> tensor<1x1x4x4xf32>
    %1 = IE.Squeeze(%0) { axes_value = [] } : tensor<1x1x4x4xf32> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK: [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [4, 4]} : tensor<16x1xf32> -> tensor<4x4xf32>
    // CHECK: return [[VAL0]] : tensor<4x4xf32>
}

// CHECK-LABEL: @ConvertConstToAttr
func.func @ConvertConstToAttr(%arg0: tensor<1x1x4x4xf32>) -> tensor<4x4xf32> {
    %0 = const.Declare tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
    %1 = IE.Squeeze(%arg0, %0) : tensor<1x1x4x4xf32>, tensor<2xsi64> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK: [[VAL0:%.+]] = IE.Squeeze(%arg0) {axes_value = [0, 1]} : tensor<1x1x4x4xf32> -> tensor<4x4xf32>
    // CHECK: return [[VAL0]] : tensor<4x4xf32>
}


#CN = affine_map<(d0, d1) -> (d1, d0)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
// CHECK-LABEL: @FuseWithAffineReshapeAndAddPermuteCast
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x1x16x4xf16, {order = #map}>
func.func @FuseWithAffineReshapeAndAddPermuteCast(%arg0: tensor<1x1x16x4xf16, {order = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>}>) -> tensor<1x64xf16, {order = affine_map<(d0, d1) -> (d1, d0)>}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [2]], shape_value = [1, 1, 64]} : tensor<1x1x16x4xf16, {order = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>}> -> tensor<1x1x64xf16, {order = affine_map<(d0, d1, d2) -> (d2, d0, d1)>}>
    %1 = IE.Squeeze(%0) {axes_value = [1]} : tensor<1x1x64xf16, {order = affine_map<(d0, d1, d2) -> (d2, d0, d1)>}> -> tensor<1x64xf16, {order = affine_map<(d0, d1) -> (d1, d0)>}>
    return %1 : tensor<1x64xf16, {order = affine_map<(d0, d1) -> (d1, d0)>}>

    // CHECK: [[VAL0:%.*]] = IE.Reshape([[INPUT0]]) {shape_value = [1, 64]} : tensor<1x1x16x4xf16, {order = #map}> -> tensor<1x64xf16>
    // CHECK: [[VAL1:%.*]] = IE.PermuteCast([[VAL0]]) {dst_order = #CN, mem_perm = #CN} : tensor<1x64xf16> -> tensor<1x64xf16, {order = #CN}>
    // CHECK: return [[VAL1]] : tensor<1x64xf16, {order = #CN}>
}
