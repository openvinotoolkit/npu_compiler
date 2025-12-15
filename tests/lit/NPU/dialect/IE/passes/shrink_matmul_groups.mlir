//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --shrink-matmul-groups %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ShrinkMatmulGroups
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x24x1x64xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x8x1x1024x64xf32>
func.func @ShrinkMatmulGroups(%arg0: tensor<1x24x1x64xf32>, %arg1: tensor<1x8x1x1024x64xf32>) -> tensor<1x24x1x1024xf32> {
    %cst = const.Declare tensor<5xsi64> = dense<[1, 8, 3, 1024, 64]> : tensor<5xsi64>

    %0 = IE.Broadcast(%arg1, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x8x1x1024x64xf32>, tensor<5xsi64> -> tensor<1x8x3x1024x64xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 24, 1024, 64]} : tensor<1x8x3x1024x64xf32> -> tensor<1x24x1024x64xf32>
    %2 = IE.MatMul(%arg0, %1) {transpose_b} : tensor<1x24x1x64xf32>, tensor<1x24x1024x64xf32> -> tensor<1x24x1x1024xf32>

    return %2 : tensor<1x24x1x1024xf32>

    // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [1, 8, 3, 64]} : tensor<1x24x1x64xf32> -> tensor<1x8x3x64xf32>
    // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT2]]) {shape_value = [1, 8, 1024, 64]} : tensor<1x8x1x1024x64xf32> -> tensor<1x8x1024x64xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b} : tensor<1x8x3x64xf32>, tensor<1x8x1024x64xf32> -> tensor<1x8x3x1024xf32>
    // CHECK:       [[RESULT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 24, 1, 1024]} : tensor<1x8x3x1024xf32> -> tensor<1x24x1x1024xf32>

    // CHECK:       return  [[RESULT]] : tensor<1x24x1x1024xf32>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @ShrinkMatmulGroupsWithTranspose
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x24x1x1024xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x8x1x1024x64xf32>
func.func @ShrinkMatmulGroupsWithTranspose(%arg0: tensor<1x24x1x1024xf32>, %arg1: tensor<1x8x1x1024x64xf32>) -> tensor<1x24x1x64xf32> {
    %cst = const.Declare tensor<5xsi64> = dense<[1, 8, 3, 1024, 64]> : tensor<5xsi64>

    %0 = IE.Broadcast(%arg1, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x8x1x1024x64xf32>, tensor<5xsi64> -> tensor<1x8x3x1024x64xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 24, 1024, 64]} : tensor<1x8x3x1024x64xf32> -> tensor<1x24x1024x64xf32>
    %2 = IE.Transpose(%1) {order_value = #NCWH} : tensor<1x24x1024x64xf32> -> tensor<1x24x64x1024xf32>
    %3 = IE.MatMul(%arg0, %2) {transpose_b} : tensor<1x24x1x1024xf32>, tensor<1x24x64x1024xf32> -> tensor<1x24x1x64xf32>

    return %3 : tensor<1x24x1x64xf32>

    // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [1, 8, 3, 1024]} : tensor<1x24x1x1024xf32> -> tensor<1x8x3x1024xf32>
    // CHECK:       [[RHS_RESHAPE:%.+]] = IE.Reshape([[INPUT2]]) {shape_value = [1, 8, 1024, 64]} : tensor<1x8x1x1024x64xf32> -> tensor<1x8x1024x64xf32>
    // CHECK:       [[RHS_TRANSPOSE:%.+]] = IE.Transpose([[RHS_RESHAPE]]) {order_value = #NCWH} : tensor<1x8x1024x64xf32> -> tensor<1x8x64x1024xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS_TRANSPOSE]]) {transpose_b} : tensor<1x8x3x1024xf32>, tensor<1x8x64x1024xf32> -> tensor<1x8x3x64xf32>
    // CHECK:       [[RESULT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 24, 1, 64]} : tensor<1x8x3x64xf32> -> tensor<1x24x1x64xf32>

    // CHECK:       return  [[RESULT]] : tensor<1x24x1x64xf32>
}

// -----

// CHECK-LABEL: @ShrinkMatmulGroupsWithTrivialD1
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x8x1x64xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x1x1x1024x64xf32>
func.func @ShrinkMatmulGroupsWithTrivialD1(%arg0: tensor<1x8x1x64xf32>, %arg1: tensor<1x1x1x1024x64xf32>) -> tensor<1x8x1x1024xf32> {
    %cst = const.Declare tensor<5xsi64> = dense<[1, 1, 8, 1024, 64]> : tensor<5xsi64>

    %0 = IE.Broadcast(%arg1, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1x1024x64xf32>, tensor<5xsi64> -> tensor<1x1x8x1024x64xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 8, 1024, 64]} : tensor<1x1x8x1024x64xf32> -> tensor<1x8x1024x64xf32>
    %2 = IE.MatMul(%arg0, %1) {transpose_b} : tensor<1x8x1x64xf32>, tensor<1x8x1024x64xf32> -> tensor<1x8x1x1024xf32>

    return %2 : tensor<1x8x1x1024xf32>

    // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [1, 1, 8, 64]} : tensor<1x8x1x64xf32> -> tensor<1x1x8x64xf32>
    // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT2]]) {shape_value = [1, 1, 1024, 64]} : tensor<1x1x1x1024x64xf32> -> tensor<1x1x1024x64xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b} : tensor<1x1x8x64xf32>, tensor<1x1x1024x64xf32> -> tensor<1x1x8x1024xf32>
    // CHECK:       [[RESULT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 8, 1, 1024]} : tensor<1x1x8x1024xf32> -> tensor<1x8x1x1024xf32>

    // CHECK:       return  [[RESULT]] : tensor<1x8x1x1024xf32>
}

// -----

// CHECK-LABEL: @ShrinkForBeneficialGroupMatMul
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x24x16x64xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x8x1x1024x64xf32>
func.func @ShrinkForBeneficialGroupMatMul(%arg0: tensor<1x24x16x64xf32>, %arg1: tensor<1x8x1x1024x64xf32>) -> tensor<1x24x16x1024xf32> {
    %cst = const.Declare tensor<5xsi64> = dense<[1, 8, 3, 1024, 64]> : tensor<5xsi64>

    %0 = IE.Broadcast(%arg1, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x8x1x1024x64xf32>, tensor<5xsi64> -> tensor<1x8x3x1024x64xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 24, 1024, 64]} : tensor<1x8x3x1024x64xf32> -> tensor<1x24x1024x64xf32>
    %2 = IE.MatMul(%arg0, %1) {transpose_b} : tensor<1x24x16x64xf32>, tensor<1x24x1024x64xf32> -> tensor<1x24x16x1024xf32>

    return %2 : tensor<1x24x16x1024xf32>

    // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [1, 8, 48, 64]} : tensor<1x24x16x64xf32> -> tensor<1x8x48x64xf32>
    // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT2]]) {shape_value = [1, 8, 1024, 64]} : tensor<1x8x1x1024x64xf32> -> tensor<1x8x1024x64xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b} : tensor<1x8x48x64xf32>, tensor<1x8x1024x64xf32> -> tensor<1x8x48x1024xf32>
    // CHECK:       [[RESULT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 24, 16, 1024]} : tensor<1x8x48x1024xf32> -> tensor<1x24x16x1024xf32>

    // CHECK:       return  [[RESULT]] : tensor<1x24x16x1024xf32>
}


// -----

// CHECK-LABEL: @NotShrinkForUnbeneficialGroupMatMul
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x24x1024x64xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x8x1x1024x64xf32>
func.func @NotShrinkForUnbeneficialGroupMatMul(%arg0: tensor<1x24x1024x64xf32>, %arg1: tensor<1x8x1x1024x64xf32>) -> tensor<1x24x1024x1024xf32> {
    %cst = const.Declare tensor<5xsi64> = dense<[1, 8, 3, 1024, 64]> : tensor<5xsi64>

    %0 = IE.Broadcast(%arg1, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x8x1x1024x64xf32>, tensor<5xsi64> -> tensor<1x8x3x1024x64xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 24, 1024, 64]} : tensor<1x8x3x1024x64xf32> -> tensor<1x24x1024x64xf32>
    %2 = IE.MatMul(%arg0, %1) {transpose_b} : tensor<1x24x1024x64xf32>, tensor<1x24x1024x64xf32> -> tensor<1x24x1024x1024xf32>

    return %2 : tensor<1x24x1024x1024xf32>

    // CHECK:       IE.Broadcast
}
