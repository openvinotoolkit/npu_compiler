//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @ConvertToShapeCast
func.func @ConvertToShapeCast(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16> {
    %0 = VPU.Reshape(%arg0) {shape_value = [1, 3, 2, 4]} : tensor<1x2x3x4xf16> -> tensor<1x3x2x4xf16>
    return %0 : tensor<1x3x2x4xf16>

    // CHECK:    [[SHAPE_CAST:%.*]] = VPU.ShapeCast {shape = [1, 3, 2, 4]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16>
    // CHECK:    return [[SHAPE_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotConvertToShapeCastWithNotIdentityLayout
func.func @NotConvertToShapeCastWithNotIdentityLayout(%arg0 : tensor<1x2x3x4xf16, {order = #NHWC}>) -> tensor<1x3x2x4xf16> {
    %0 = VPU.Reshape(%arg0) {shape_value = [1, 3, 2, 4]} : tensor<1x2x3x4xf16, {order = #NHWC}> -> tensor<1x3x2x4xf16>
    return %0 : tensor<1x3x2x4xf16>

    // CHECK:    [[RESHAPE:%.*]] = VPU.Reshape(%arg0) {shape_value = [1, 3, 2, 4]} : tensor<1x2x3x4xf16, {order = #NHWC}> -> tensor<1x3x2x4xf16>
    // CHECK:    return [[RESHAPE]]
}

// -----

// CHECK-LABEL: @ConvertToAffineReshapeWithDifferentRank
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x2x3x4xf16>
func.func @ConvertToAffineReshapeWithDifferentRank(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x6x4xf16> {
    %0 = VPU.Reshape(%arg0) {shape_value = [1, 6, 4]} : tensor<1x2x3x4xf16> -> tensor<1x6x4xf16>
    return %0 : tensor<1x6x4xf16>

    // CHECK:    [[AFFINERESHAPE:%.+]] = VPU.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 6, 4]} : tensor<1x2x3x4xf16> -> tensor<1x6x4xf16>
    // CHECK:    return [[AFFINERESHAPE]]
}

// -----

// CHECK-LABEL: @FuseReshapes
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x8xsi32>
func.func @FuseReshapes(%arg0: tensor<1x1x1x8xsi32>) -> tensor<4x2xsi32> {
    %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [0]], shape_value = [8]} : tensor<1x1x1x8xsi32> -> tensor<8xsi32>
    %1 = VPU.Reshape(%0) {shape_value = [4, 2]} : tensor<8xsi32> -> tensor<4x2xsi32>

    return %1 : tensor<4x2xsi32>

    // CHECK:    [[RESHAPE:%.+]] = VPU.Reshape([[INPUT]]) {shape_value = [4, 2]} : tensor<1x1x1x8xsi32> -> tensor<4x2xsi32>
    // CHECK:    return [[RESHAPE]] : tensor<4x2xsi32>
}
