//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @Eliminate
func @Eliminate(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [4, 4] } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: IE.Reshape
    // CHECK:     return %arg0
}

// CHECK-LABEL: @ConstFold
func @ConstFold() -> tensor<4x4xf32> {
    %0 = const.Declare tensor<16xf32> = dense<1.0> : tensor<16xf32>
    %1 = IE.Reshape(%0) { shape_value = [4, 4] } : tensor<16xf32> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK:       [[VAL0:%.+]] = const.Declare tensor<4x4xf32> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<16xf32>, [#const.Reshape<[4, 4]>]
    // CHECK-NOT:   IE.Reshape
    // CHECK:       return [[VAL0]]
}

// CHECK-LABEL: @FuseReshapes
func @FuseReshapes(%arg0: tensor<16x1xf32>) -> tensor<4x4xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 1, 4, 4] } : tensor<16x1xf32> -> tensor<1x1x4x4xf32>
    %1 = IE.Reshape(%0) { shape_value = [4, 4] } : tensor<1x1x4x4xf32> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK: [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [4, 4]} : tensor<16x1xf32> -> tensor<4x4xf32>
    // CHECK: return [[VAL0]] : tensor<4x4xf32>
}

// CHECK-LABEL: @ConvertConstToAttr
func @ConvertConstToAttr(%arg0: tensor<16x1xf32>) -> tensor<4x4xf32> {
    %0 = const.Declare tensor<2xsi64> = dense<[4, 4]> : tensor<2xsi64>
    %1 = IE.Reshape(%arg0, %0) : tensor<16x1xf32>, tensor<2xsi64> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK: [[VAL0:%.+]] = IE.Reshape(%arg0) {shape_value = [4, 4]} : tensor<16x1xf32> -> tensor<4x4xf32>
    // CHECK: return [[VAL0]] : tensor<4x4xf32>
}

// CHECK-LABEL: @ConvertToAffineReshape
func @ConvertToAffineReshape(%arg0: tensor<1x1x9x16x2xf32>) -> tensor<1x3x3x32xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 3, 3, 32] } : tensor<1x1x9x16x2xf32> -> tensor<1x3x3x32xf32>
    return %0 : tensor<1x3x3x32xf32>

    // CHECK: [[VAL0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2], [3], [3]], shape_value = [1, 3, 3, 32]} : tensor<1x1x9x16x2xf32> -> tensor<1x3x3x32xf32>
    // CHECK: return [[VAL0]] : tensor<1x3x3x32xf32>
}
