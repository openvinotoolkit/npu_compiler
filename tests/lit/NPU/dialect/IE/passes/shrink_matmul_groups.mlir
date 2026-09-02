//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --shrink-matmul-groups %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

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

// -----

// Case 3: Broadcast -> Transpose(5D swap last 2 dims) -> AffineReshape -> MatMul
// Models the KV-cache V-matmul pattern in Qwen2 decode:
//   RHS: 1x2x1x1152x64 -> Broadcast -> 1x2x7x1152x64
//         -> Transpose(d0,d1,d2,d4,d3) -> 1x2x7x64x1152
//         -> AffineReshape -> 1x14x64x1152
//   LHS: 1x14x1x1152
// Shrinks to a 2-group MatMul.

#NDHWC_SWAP = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @ShrinkMatmulGroupsCase3VMatMul
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x14x1x1152xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x2x1x1152x64xf32>
func.func @ShrinkMatmulGroupsCase3VMatMul(%arg0: tensor<1x14x1x1152xf32>, %arg1: tensor<1x2x1x1152x64xf32>) -> tensor<1x14x1x64xf32> {
    %cst = const.Declare tensor<5xsi64> = dense<[1, 2, 7, 1152, 64]> : tensor<5xsi64>

    %0 = IE.Broadcast(%arg1, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x2x1x1152x64xf32>, tensor<5xsi64> -> tensor<1x2x7x1152x64xf32>
    %1 = IE.Transpose(%0) {order_value = #NDHWC_SWAP} : tensor<1x2x7x1152x64xf32> -> tensor<1x2x7x64x1152xf32>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 14, 64, 1152]} : tensor<1x2x7x64x1152xf32> -> tensor<1x14x64x1152xf32>
    %3 = IE.MatMul(%arg0, %2) {transpose_b} : tensor<1x14x1x1152xf32>, tensor<1x14x64x1152xf32> -> tensor<1x14x1x64xf32>

    return %3 : tensor<1x14x1x64xf32>

    // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [1, 2, 7, 1152]} : tensor<1x14x1x1152xf32> -> tensor<1x2x7x1152xf32>
    // CHECK:       [[RHS_RESHAPE:%.+]] = IE.Reshape([[INPUT2]]) {shape_value = [1, 2, 1152, 64]} : tensor<1x2x1x1152x64xf32> -> tensor<1x2x1152x64xf32>
    // CHECK:       [[RHS_TRANSPOSE:%.+]] = IE.Transpose([[RHS_RESHAPE]]) {order_value = #NCWH} : tensor<1x2x1152x64xf32> -> tensor<1x2x64x1152xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS_TRANSPOSE]]) {transpose_b} : tensor<1x2x7x1152xf32>, tensor<1x2x64x1152xf32> -> tensor<1x2x7x64xf32>
    // CHECK:       [[RESULT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 14, 1, 64]} : tensor<1x2x7x64xf32> -> tensor<1x14x1x64xf32>
    // CHECK:       return [[RESULT]] : tensor<1x14x1x64xf32>
}

// -----

// Case 3 variant: inner 5D Transpose swaps last 2 dims of the Broadcast output
// then the AffineReshape collapses (d1,d2) into C, matching the K-matmul pattern
// in Qwen2 decode where RHS is 1x2x1x64x1152 (transposed KV-cache).

#NDHWC_SWAP = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @ShrinkMatmulGroupsCase3KMatMul
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x14x1x64xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x2x1x64x1152xf32>
func.func @ShrinkMatmulGroupsCase3KMatMul(%arg0: tensor<1x14x1x64xf32>, %arg1: tensor<1x2x1x64x1152xf32>) -> tensor<1x14x1x1152xf32> {
    %cst = const.Declare tensor<5xsi64> = dense<[1, 2, 7, 64, 1152]> : tensor<5xsi64>

    %0 = IE.Broadcast(%arg1, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x2x1x64x1152xf32>, tensor<5xsi64> -> tensor<1x2x7x64x1152xf32>
    %1 = IE.Transpose(%0) {order_value = #NDHWC_SWAP} : tensor<1x2x7x64x1152xf32> -> tensor<1x2x7x1152x64xf32>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 14, 1152, 64]} : tensor<1x2x7x1152x64xf32> -> tensor<1x14x1152x64xf32>
    %3 = IE.MatMul(%arg0, %2) {transpose_b} : tensor<1x14x1x64xf32>, tensor<1x14x1152x64xf32> -> tensor<1x14x1x1152xf32>

    return %3 : tensor<1x14x1x1152xf32>

    // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [1, 2, 7, 64]} : tensor<1x14x1x64xf32> -> tensor<1x2x7x64xf32>
    // CHECK:       [[RHS_RESHAPE:%.+]] = IE.Reshape([[INPUT2]]) {shape_value = [1, 2, 64, 1152]} : tensor<1x2x1x64x1152xf32> -> tensor<1x2x64x1152xf32>
    // CHECK:       [[RHS_TRANSPOSE:%.+]] = IE.Transpose([[RHS_RESHAPE]]) {order_value = #NCWH} : tensor<1x2x64x1152xf32> -> tensor<1x2x1152x64xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS_TRANSPOSE]]) {transpose_b} : tensor<1x2x7x64xf32>, tensor<1x2x1152x64xf32> -> tensor<1x2x7x1152xf32>
    // CHECK:       [[RESULT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 14, 1, 1152]} : tensor<1x2x7x1152xf32> -> tensor<1x14x1x1152xf32>
    // CHECK:       return [[RESULT]] : tensor<1x14x1x1152xf32>
}

// -----

// Case 3 with dim_mapping = [[0],[0],[1],[2],[3]]: the AffineReshape collapses d0+d1
// (trivial N=1) rather than d1+d2, mirroring ShrinkMatmulGroupsWithTrivialD1 but with
// an inner 5D Transpose present. This exercises the [[0],[0],[1],[2],[3]] path in the
// unified RHS reshape logic and ensures the rewrite correctly drops the unit d2 from
// the broadcast input regardless of which leading dims are folded.
// newGroupNum = broadcastOutputShape[C] = 1.

#NDHWC_SWAP = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @ShrinkMatmulGroupsCase3TrivialC
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x7x1x1152xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x1x1x1152x64xf32>
func.func @ShrinkMatmulGroupsCase3TrivialC(%arg0: tensor<1x7x1x1152xf32>, %arg1: tensor<1x1x1x1152x64xf32>) -> tensor<1x7x1x64xf32> {
    %cst = const.Declare tensor<5xsi64> = dense<[1, 1, 7, 1152, 64]> : tensor<5xsi64>

    %0 = IE.Broadcast(%arg1, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1x1152x64xf32>, tensor<5xsi64> -> tensor<1x1x7x1152x64xf32>
    %1 = IE.Transpose(%0) {order_value = #NDHWC_SWAP} : tensor<1x1x7x1152x64xf32> -> tensor<1x1x7x64x1152xf32>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 7, 64, 1152]} : tensor<1x1x7x64x1152xf32> -> tensor<1x7x64x1152xf32>
    %3 = IE.MatMul(%arg0, %2) {transpose_b} : tensor<1x7x1x1152xf32>, tensor<1x7x64x1152xf32> -> tensor<1x7x1x64xf32>

    return %3 : tensor<1x7x1x64xf32>

    // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [1, 1, 7, 1152]} : tensor<1x7x1x1152xf32> -> tensor<1x1x7x1152xf32>
    // CHECK:       [[RHS_RESHAPE:%.+]] = IE.Reshape([[INPUT2]]) {shape_value = [1, 1, 1152, 64]} : tensor<1x1x1x1152x64xf32> -> tensor<1x1x1152x64xf32>
    // CHECK:       [[RHS_TRANSPOSE:%.+]] = IE.Transpose([[RHS_RESHAPE]]) {order_value = #NCWH} : tensor<1x1x1152x64xf32> -> tensor<1x1x64x1152xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS_TRANSPOSE]]) {transpose_b} : tensor<1x1x7x1152xf32>, tensor<1x1x64x1152xf32> -> tensor<1x1x7x64xf32>
    // CHECK:       [[RESULT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 7, 1, 64]} : tensor<1x1x7x64xf32> -> tensor<1x7x1x64xf32>
    // CHECK:       return [[RESULT]] : tensor<1x7x1x64xf32>
}

// -----

// Negative test for Case 3: inner Transpose is NOT a last-2-dims swap
// (swaps dims 1 and 2 instead of dims 3 and 4), so
// checkSwapLast2DimsTranspose() rejects it and the pass must not fire.
// Shapes are chosen so checkMatMul and checkAffineReshape both pass, ensuring
// the rejection happens inside the checkSwapLast2DimsTranspose() guard.

#SWAP_D1_D2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3, d4)>

// CHECK-LABEL: @NotShrinkCase3WrongInnerTranspose
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x14x1x64xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x2x1x1152x64xf32>
func.func @NotShrinkCase3WrongInnerTranspose(%arg0: tensor<1x14x1x64xf32>, %arg1: tensor<1x2x1x1152x64xf32>) -> tensor<1x14x1x1152xf32> {
    %cst = const.Declare tensor<5xsi64> = dense<[1, 2, 7, 1152, 64]> : tensor<5xsi64>

    %0 = IE.Broadcast(%arg1, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x2x1x1152x64xf32>, tensor<5xsi64> -> tensor<1x2x7x1152x64xf32>
    %1 = IE.Transpose(%0) {order_value = #SWAP_D1_D2} : tensor<1x2x7x1152x64xf32> -> tensor<1x7x2x1152x64xf32>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 14, 1152, 64]} : tensor<1x7x2x1152x64xf32> -> tensor<1x14x1152x64xf32>
    %3 = IE.MatMul(%arg0, %2) {transpose_b} : tensor<1x14x1x64xf32>, tensor<1x14x1152x64xf32> -> tensor<1x14x1x1152xf32>

    return %3 : tensor<1x14x1x1152xf32>

    // CHECK:       IE.Broadcast
    // CHECK:       IE.Transpose
    // CHECK:       IE.AffineReshape
    // CHECK:       IE.MatMul
}

// -----

// Negative test for Case 3: inner Transpose IS a valid last-2-dims swap, but the
// AffineReshape uses dim_mapping = [[0], [1], [2], [2], [3]] (merges d2+d3 into output d2),
// so the last two dims are NOT preserved (outputShape[H]=448 != inputShape[H]=64).
// Shapes are chosen so checkMatMul passes (LHS C=2 matches RHS C=2, LHS W=1152 matches
// RHS W=1152), ensuring the rejection happens at checkAffineReshape.

#NDHWC_SWAP = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>

// CHECK-LABEL: @NotShrinkCase3WrongDimMapping
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x2x1x1152xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x2x1x1152x64xf32>
func.func @NotShrinkCase3WrongDimMapping(%arg0: tensor<1x2x1x1152xf32>, %arg1: tensor<1x2x1x1152x64xf32>) -> tensor<1x2x1x448xf32> {
    %cst = const.Declare tensor<5xsi64> = dense<[1, 2, 7, 1152, 64]> : tensor<5xsi64>

    %0 = IE.Broadcast(%arg1, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x2x1x1152x64xf32>, tensor<5xsi64> -> tensor<1x2x7x1152x64xf32>
    %1 = IE.Transpose(%0) {order_value = #NDHWC_SWAP} : tensor<1x2x7x1152x64xf32> -> tensor<1x2x7x64x1152xf32>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 2, 448, 1152]} : tensor<1x2x7x64x1152xf32> -> tensor<1x2x448x1152xf32>
    %3 = IE.MatMul(%arg0, %2) {transpose_b} : tensor<1x2x1x1152xf32>, tensor<1x2x448x1152xf32> -> tensor<1x2x1x448xf32>

    return %3 : tensor<1x2x1x448xf32>

    // CHECK:       IE.Broadcast
    // CHECK:       IE.Transpose
    // CHECK:       IE.AffineReshape
    // CHECK:       IE.MatMul
}

// -----

// LHS has H > 1 (H=8, token length > 1), matching the Q*K^T pattern in LLM
// attention where the sequence length is small (e.g. speculative decoding validation stage).

// CHECK-LABEL: @ShrinkMatmulGroupsMultiTokenLHS
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x8x128xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x8x1x8704x128xf32>
func.func @ShrinkMatmulGroupsMultiTokenLHS(%arg0: tensor<1x32x8x128xf32>, %arg1: tensor<1x8x1x8704x128xf32>) -> tensor<1x32x8x8704xf32> {
    %cst = const.Declare tensor<5xsi64> = dense<[1, 8, 4, 8704, 128]> : tensor<5xsi64>

    %0 = IE.Broadcast(%arg1, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x8x1x8704x128xf32>, tensor<5xsi64> -> tensor<1x8x4x8704x128xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 32, 8704, 128]} : tensor<1x8x4x8704x128xf32> -> tensor<1x32x8704x128xf32>
    %2 = IE.MatMul(%arg0, %1) {transpose_b} : tensor<1x32x8x128xf32>, tensor<1x32x8704x128xf32> -> tensor<1x32x8x8704xf32>

    return %2 : tensor<1x32x8x8704xf32>

    // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [1, 8, 32, 128]} : tensor<1x32x8x128xf32> -> tensor<1x8x32x128xf32>
    // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT2]]) {shape_value = [1, 8, 8704, 128]} : tensor<1x8x1x8704x128xf32> -> tensor<1x8x8704x128xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b} : tensor<1x8x32x128xf32>, tensor<1x8x8704x128xf32> -> tensor<1x8x32x8704xf32>
    // CHECK:       [[RESULT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 32, 8, 8704]} : tensor<1x8x32x8704xf32> -> tensor<1x32x8x8704xf32>

    // CHECK:       return  [[RESULT]] : tensor<1x32x8x8704xf32>
}

// -----

// Negative test: LHS H=64 exceeds the MAX_LHS_H_FOR_SHRINK=32 threshold,
// so the pass must not fire regardless of other conditions.

// CHECK-LABEL: @NotShrinkMatmulGroupsLargeH
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x64x128xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x8x1x8704x128xf32>
func.func @NotShrinkMatmulGroupsLargeH(%arg0: tensor<1x32x64x128xf32>, %arg1: tensor<1x8x1x8704x128xf32>) -> tensor<1x32x64x8704xf32> {
    %cst = const.Declare tensor<5xsi64> = dense<[1, 8, 4, 8704, 128]> : tensor<5xsi64>

    %0 = IE.Broadcast(%arg1, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x8x1x8704x128xf32>, tensor<5xsi64> -> tensor<1x8x4x8704x128xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 32, 8704, 128]} : tensor<1x8x4x8704x128xf32> -> tensor<1x32x8704x128xf32>
    %2 = IE.MatMul(%arg0, %1) {transpose_b} : tensor<1x32x64x128xf32>, tensor<1x32x8704x128xf32> -> tensor<1x32x64x8704xf32>

    return %2 : tensor<1x32x64x8704xf32>

    // CHECK:       IE.Broadcast
    // CHECK:       IE.AffineReshape
    // CHECK:       IE.MatMul
}
