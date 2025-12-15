//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @Eliminate
func.func @Eliminate(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [4, 4] } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: IE.Reshape
    // CHECK:     return %arg0
}

// -----

// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<4x4xf32> {
    %0 = const.Declare tensor<16xf32> = dense<1.0> : tensor<16xf32>
    %1 = IE.Reshape(%0) { shape_value = [4, 4] } : tensor<16xf32> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK-DAG:       [[VAL0:%.+]] = const.Declare tensor<4x4xf32> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<16xf32>, [#const.Reshape<[4, 4]>]
    // CHECK-NOT:   IE.Reshape
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @FuseReshapes
func.func @FuseReshapes(%arg0: tensor<16x1xf32>) -> tensor<4x4xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 1, 4, 4] } : tensor<16x1xf32> -> tensor<1x1x4x4xf32>
    %1 = IE.Reshape(%0) { shape_value = [4, 4] } : tensor<1x1x4x4xf32> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK: [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [4, 4]} : tensor<16x1xf32> -> tensor<4x4xf32>
    // CHECK: return [[VAL0]] : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @ConvertConstToAttr
func.func @ConvertConstToAttr(%arg0: tensor<16x1xf32>) -> tensor<4x4xf32> {
    %0 = const.Declare tensor<2xsi64> = dense<[4, 4]> : tensor<2xsi64>
    %1 = IE.Reshape(%arg0, %0) : tensor<16x1xf32>, tensor<2xsi64> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK: [[VAL0:%.+]] = IE.Reshape(%arg0) {shape_value = [4, 4]} : tensor<16x1xf32> -> tensor<4x4xf32>
    // CHECK: return [[VAL0]] : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @ConvertToAffineReshape
func.func @ConvertToAffineReshape(%arg0: tensor<1x1x9x16x2xf32>) -> tensor<1x3x3x32xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 3, 3, 32] } : tensor<1x1x9x16x2xf32> -> tensor<1x3x3x32xf32>
    return %0 : tensor<1x3x3x32xf32>

    // CHECK: [[VAL0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2], [3], [3]], shape_value = [1, 3, 3, 32]} : tensor<1x1x9x16x2xf32> -> tensor<1x3x3x32xf32>
    // CHECK: return [[VAL0]] : tensor<1x3x3x32xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DontConvertToAffineReshape
// Because there's no explicit valid layout infered based on the input layout and dim mapping
func.func @DontConvertToAffineReshape(%arg0: tensor<1x1x1x160000xf16, {order = #NHWC}>) -> tensor<1x2500x1x64xf16> {
    %0 = IE.Reshape(%arg0) {shape_value = [1, 2500, 1, 64]} : tensor<1x1x1x160000xf16, {order = #NHWC}> -> tensor<1x2500x1x64xf16>
    return %0 : tensor<1x2500x1x64xf16>

    // CHECK: [[VAL0:%.*]] = IE.Reshape(%arg0)
    // CHECK-SAME{LITERAL}: {shape_value = [1, 2500, 1, 64]} : tensor<1x1x1x160000xf16, {order = #NHWC}> -> tensor<1x2500x1x64xf16>
    // CHECK: return [[VAL0]] : tensor<1x2500x1x64xf16>
}
