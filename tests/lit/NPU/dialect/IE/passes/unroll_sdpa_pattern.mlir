//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --unroll-sdpa-pattern %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @UnrollSDPAPatternWithMatmulsHaveDifferentShrinkingBehavior
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x128x128xf32>, [[ARG1:%.+]]: tensor<1x8x128x128xf32>, [[MASK:%.+]]: tensor<1x1x128x128xf32>, [[V_IN:%.+]]: tensor<1x1x1x128x128xf32>)
func.func @UnrollSDPAPatternWithMatmulsHaveDifferentShrinkingBehavior(%arg0: tensor<1x8x128x128xf32>, %arg1: tensor<1x8x128x128xf32>, %mask: tensor<1x1x128x128xf32>, %v_in: tensor<1x1x1x128x128xf32>) -> tensor<1x8x128x128xf32> {
  %target_shape = const.Declare tensor<5xsi64> = dense<[1, 1, 8, 128, 128]> : tensor<5xsi64>
  %broadcast = IE.Broadcast(%v_in, %target_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1x128x128xf32>, tensor<5xsi64> -> tensor<1x1x8x128x128xf32>
  %reshape = IE.AffineReshape(%broadcast) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 8, 128, 128]} : tensor<1x1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %transpose = IE.Transpose(%reshape) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %1 = IE.Add(%0, %mask) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x128x128xf32>, tensor<1x1x128x128xf32> -> tensor<1x8x128x128xf32>
  %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %3 = IE.MatMul(%2, %transpose) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  return %3 : tensor<1x8x128x128xf32>

  // CHECK: [[TARGET_SHAPE:%.+]] = const.Declare
  // CHECK: [[BROADCAST:%.+]] = IE.Broadcast([[V_IN]], [[TARGET_SHAPE]])
  // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[BROADCAST]])
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])

  // CHECK: [[SLICE_Q_0:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_1:%.+]] = IE.Slice [[ARG0]] [0, 1, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_2:%.+]] = IE.Slice [[ARG0]] [0, 2, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_3:%.+]] = IE.Slice [[ARG0]] [0, 3, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_4:%.+]] = IE.Slice [[ARG0]] [0, 4, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_5:%.+]] = IE.Slice [[ARG0]] [0, 5, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_6:%.+]] = IE.Slice [[ARG0]] [0, 6, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_7:%.+]] = IE.Slice [[ARG0]] [0, 7, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_0:%.+]] = IE.Slice [[ARG1]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_1:%.+]] = IE.Slice [[ARG1]] [0, 1, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_2:%.+]] = IE.Slice [[ARG1]] [0, 2, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_3:%.+]] = IE.Slice [[ARG1]] [0, 3, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_4:%.+]] = IE.Slice [[ARG1]] [0, 4, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_5:%.+]] = IE.Slice [[ARG1]] [0, 5, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_6:%.+]] = IE.Slice [[ARG1]] [0, 6, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_7:%.+]] = IE.Slice [[ARG1]] [0, 7, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_0:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_1:%.+]] = IE.Slice [[TRANSPOSE]] [0, 1, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_2:%.+]] = IE.Slice [[TRANSPOSE]] [0, 2, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_3:%.+]] = IE.Slice [[TRANSPOSE]] [0, 3, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_4:%.+]] = IE.Slice [[TRANSPOSE]] [0, 4, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_5:%.+]] = IE.Slice [[TRANSPOSE]] [0, 5, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_6:%.+]] = IE.Slice [[TRANSPOSE]] [0, 6, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_7:%.+]] = IE.Slice [[TRANSPOSE]] [0, 7, 0, 0] [1, 1, 128, 128]

  // CHECK: [[MATMUL_1_0:%.+]] = IE.MatMul([[SLICE_Q_0]], [[SLICE_K_0]]) {transpose_b}
  // CHECK: [[ADD_0:%.+]] = IE.Add([[MATMUL_1_0]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_0:%.+]] = IE.SoftMax([[ADD_0]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_0:%.+]] = IE.MatMul([[SOFTMAX_0]], [[SLICE_V_0]]) {transpose_b}

  // CHECK: [[MATMUL_1_1:%.+]] = IE.MatMul([[SLICE_Q_1]], [[SLICE_K_1]]) {transpose_b}
  // CHECK: [[ADD_1:%.+]] = IE.Add([[MATMUL_1_1]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_1:%.+]] = IE.SoftMax([[ADD_1]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_1:%.+]] = IE.MatMul([[SOFTMAX_1]], [[SLICE_V_1]]) {transpose_b}

  // CHECK: [[MATMUL_1_2:%.+]] = IE.MatMul([[SLICE_Q_2]], [[SLICE_K_2]]) {transpose_b}
  // CHECK: [[ADD_2:%.+]] = IE.Add([[MATMUL_1_2]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_2:%.+]] = IE.SoftMax([[ADD_2]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_2:%.+]] = IE.MatMul([[SOFTMAX_2]], [[SLICE_V_2]]) {transpose_b}

  // CHECK: [[MATMUL_1_3:%.+]] = IE.MatMul([[SLICE_Q_3]], [[SLICE_K_3]]) {transpose_b}
  // CHECK: [[ADD_3:%.+]] = IE.Add([[MATMUL_1_3]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_3:%.+]] = IE.SoftMax([[ADD_3]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_3:%.+]] = IE.MatMul([[SOFTMAX_3]], [[SLICE_V_3]]) {transpose_b}

  // CHECK: [[MATMUL_1_4:%.+]] = IE.MatMul([[SLICE_Q_4]], [[SLICE_K_4]]) {transpose_b}
  // CHECK: [[ADD_4:%.+]] = IE.Add([[MATMUL_1_4]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_4:%.+]] = IE.SoftMax([[ADD_4]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_4:%.+]] = IE.MatMul([[SOFTMAX_4]], [[SLICE_V_4]]) {transpose_b}

  // CHECK: [[MATMUL_1_5:%.+]] = IE.MatMul([[SLICE_Q_5]], [[SLICE_K_5]]) {transpose_b}
  // CHECK: [[ADD_5:%.+]] = IE.Add([[MATMUL_1_5]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_5:%.+]] = IE.SoftMax([[ADD_5]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_5:%.+]] = IE.MatMul([[SOFTMAX_5]], [[SLICE_V_5]]) {transpose_b}

  // CHECK: [[MATMUL_1_6:%.+]] = IE.MatMul([[SLICE_Q_6]], [[SLICE_K_6]]) {transpose_b}
  // CHECK: [[ADD_6:%.+]] = IE.Add([[MATMUL_1_6]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_6:%.+]] = IE.SoftMax([[ADD_6]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_6:%.+]] = IE.MatMul([[SOFTMAX_6]], [[SLICE_V_6]]) {transpose_b}

  // CHECK: [[MATMUL_1_7:%.+]] = IE.MatMul([[SLICE_Q_7]], [[SLICE_K_7]]) {transpose_b}
  // CHECK: [[ADD_7:%.+]] = IE.Add([[MATMUL_1_7]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX_7:%.+]] = IE.SoftMax([[ADD_7]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_7:%.+]] = IE.MatMul([[SOFTMAX_7]], [[SLICE_V_7]]) {transpose_b}

  // CHECK: [[CONCAT:%.+]] = IE.Concat([[MATMUL_V_0]], [[MATMUL_V_1]], [[MATMUL_V_2]], [[MATMUL_V_3]], [[MATMUL_V_4]], [[MATMUL_V_5]], [[MATMUL_V_6]], [[MATMUL_V_7]])
  // CHECK: return [[CONCAT]]
}

// -----

// CHECK-LABEL: @DontUnrollSDPAPatternWithMatmulsHaveSameShrinkingBehavior
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x128x128xf32>, [[ARG1:%.+]]: tensor<1x8x128x128xf32>, [[MASK:%.+]]: tensor<1x1x128x128xf32>, [[ARG2:%.+]]: tensor<1x8x128x128xf32>)
func.func @DontUnrollSDPAPatternWithMatmulsHaveSameShrinkingBehavior(%arg0: tensor<1x8x128x128xf32>, %arg1: tensor<1x8x128x128xf32>, %mask: tensor<1x1x128x128xf32>, %arg2: tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %1 = IE.Add(%0, %mask) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x128x128xf32>, tensor<1x1x128x128xf32> -> tensor<1x8x128x128xf32>
  %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %3 = IE.MatMul(%2, %arg2) : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  return %3 : tensor<1x8x128x128xf32>

  // CHECK: [[MATMUL_1:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
  // CHECK: [[ADD:%.+]] = IE.Add([[MATMUL_1]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]])
  // CHECK: return [[MATMUL_V]]
}

// -----

// CHECK-LABEL: @DontUnrollSDPAPatternBatch1
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<8x1x128x128xf32>, [[ARG1:%.+]]: tensor<8x1x128x128xf32>, [[MASK:%.+]]: tensor<1x1x128x128xf32>, [[V_IN:%.+]]: tensor<8x1x1x128x128xf32>)
func.func @DontUnrollSDPAPatternBatch1(%arg0: tensor<8x1x128x128xf32>, %arg1: tensor<8x1x128x128xf32>, %mask: tensor<1x1x128x128xf32>, %v_in: tensor<8x1x1x128x128xf32>) -> tensor<8x1x128x128xf32> {
  %target_shape = const.Declare tensor<5xsi64> = dense<[8, 1, 1, 128, 128]> : tensor<5xsi64>
  %broadcast = IE.Broadcast(%v_in, %target_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<8x1x1x128x128xf32>, tensor<5xsi64> -> tensor<8x1x1x128x128xf32>
  %reshape = IE.AffineReshape(%broadcast) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [8, 1, 128, 128]} : tensor<8x1x1x128x128xf32> -> tensor<8x1x128x128xf32>
  %transpose = IE.Transpose(%reshape) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<8x1x128x128xf32> -> tensor<8x1x128x128xf32>
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<8x1x128x128xf32>, tensor<8x1x128x128xf32> -> tensor<8x1x128x128xf32>
  %1 = IE.Add(%0, %mask) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x1x128x128xf32>, tensor<1x1x128x128xf32> -> tensor<8x1x128x128xf32>
  %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<8x1x128x128xf32> -> tensor<8x1x128x128xf32>
  %3 = IE.MatMul(%2, %transpose) {transpose_b} : tensor<8x1x128x128xf32>, tensor<8x1x128x128xf32> -> tensor<8x1x128x128xf32>
  return %3 : tensor<8x1x128x128xf32>

  // CHECK: [[TARGET_SHAPE:%.+]] = const.Declare
  // CHECK: [[BROADCAST:%.+]] = IE.Broadcast([[V_IN]], [[TARGET_SHAPE]])
  // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[BROADCAST]])
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])

	// CHECK: [[MATMUL_1:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
	// CHECK: [[ADD:%.+]] = IE.Add([[MATMUL_1]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
	// CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 3 : i64}
	// CHECK: [[MATMUL_V:%.+]] = IE.MatMul([[SOFTMAX]], [[TRANSPOSE]])
	// CHECK: return [[MATMUL_V]]
}

// -----

// CHECK-LABEL: @DontUnrollSDPAPatternNoAdd
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x128x128xf32>, [[ARG1:%.+]]: tensor<1x8x128x128xf32>, [[V_IN:%.+]]: tensor<1x1x1x128x128xf32>)
func.func @DontUnrollSDPAPatternNoAdd(%arg0: tensor<1x8x128x128xf32>, %arg1: tensor<1x8x128x128xf32>, %v_in: tensor<1x1x1x128x128xf32>) -> tensor<1x8x128x128xf32> {
  %target_shape = const.Declare tensor<5xsi64> = dense<[1, 1, 8, 128, 128]> : tensor<5xsi64>
  %broadcast = IE.Broadcast(%v_in, %target_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1x128x128xf32>, tensor<5xsi64> -> tensor<1x1x8x128x128xf32>
  %reshape = IE.AffineReshape(%broadcast) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 8, 128, 128]} : tensor<1x1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %transpose = IE.Transpose(%reshape) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %2 = IE.MatMul(%1, %transpose) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  return %2 : tensor<1x8x128x128xf32>

  // CHECK: [[TARGET_SHAPE:%.+]] = const.Declare
  // CHECK: [[BROADCAST:%.+]] = IE.Broadcast([[V_IN]], [[TARGET_SHAPE]])
  // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[BROADCAST]])
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])

  // CHECK: [[MATMUL_1:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
	// CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MATMUL_1]]) {axisInd = 3 : i64}
	// CHECK: [[MATMUL_V:%.+]] = IE.MatMul([[SOFTMAX]], [[TRANSPOSE]]) {transpose_b}
  // CHECK: return [[MATMUL_V]]
}

// -----

// CHECK-LABEL: @UnrollSDPAPatternWithConcatAndSlice
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x128x128xf32>, [[ARG1:%.+]]: tensor<1x8x128x128xf32>, [[MASK:%.+]]: tensor<1x1x128x128xf32>, [[CONCAT_INPUT:%.+]]: tensor<1x8x128x1xf32>, [[V_IN:%.+]]: tensor<1x1x1x128x128xf32>)
func.func @UnrollSDPAPatternWithConcatAndSlice(%arg0: tensor<1x8x128x128xf32>, %arg1: tensor<1x8x128x128xf32>, %mask: tensor<1x1x128x128xf32>, %concat_input: tensor<1x8x128x1xf32>, %v_in: tensor<1x1x1x128x128xf32>) -> tensor<1x8x128x128xf32> {
  %target_shape = const.Declare tensor<5xsi64> = dense<[1, 1, 8, 128, 128]> : tensor<5xsi64>
  %broadcast = IE.Broadcast(%v_in, %target_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1x128x128xf32>, tensor<5xsi64> -> tensor<1x1x8x128x128xf32>
  %reshape = IE.AffineReshape(%broadcast) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 8, 128, 128]} : tensor<1x1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %transpose = IE.Transpose(%reshape) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %1 = IE.Add(%0, %mask) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x128x128xf32>, tensor<1x1x128x128xf32> -> tensor<1x8x128x128xf32>
  %2 = IE.Concat(%1, %concat_input) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x8x128x128xf32>, tensor<1x8x128x1xf32> -> tensor<1x8x128x129xf32>
  %3 = IE.SoftMax(%2) {axisInd = 3 : i64} : tensor<1x8x128x129xf32> -> tensor<1x8x128x129xf32>
  %4 = IE.Slice %3 [0, 0, 0, 0] [1, 8, 128, 128] : tensor<1x8x128x129xf32> to tensor<1x8x128x128xf32>
  %5 = IE.MatMul(%4, %transpose) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  return %5 : tensor<1x8x128x128xf32>

  // CHECK: [[TARGET_SHAPE:%.+]] = const.Declare
  // CHECK: [[BROADCAST:%.+]] = IE.Broadcast([[V_IN]], [[TARGET_SHAPE]])
  // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[BROADCAST]])
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])

  // CHECK: [[SLICE_Q_0:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_1:%.+]] = IE.Slice [[ARG0]] [0, 1, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_2:%.+]] = IE.Slice [[ARG0]] [0, 2, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_3:%.+]] = IE.Slice [[ARG0]] [0, 3, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_4:%.+]] = IE.Slice [[ARG0]] [0, 4, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_5:%.+]] = IE.Slice [[ARG0]] [0, 5, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_6:%.+]] = IE.Slice [[ARG0]] [0, 6, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_7:%.+]] = IE.Slice [[ARG0]] [0, 7, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_0:%.+]] = IE.Slice [[ARG1]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_1:%.+]] = IE.Slice [[ARG1]] [0, 1, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_2:%.+]] = IE.Slice [[ARG1]] [0, 2, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_3:%.+]] = IE.Slice [[ARG1]] [0, 3, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_4:%.+]] = IE.Slice [[ARG1]] [0, 4, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_5:%.+]] = IE.Slice [[ARG1]] [0, 5, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_6:%.+]] = IE.Slice [[ARG1]] [0, 6, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_7:%.+]] = IE.Slice [[ARG1]] [0, 7, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_0:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_1:%.+]] = IE.Slice [[TRANSPOSE]] [0, 1, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_2:%.+]] = IE.Slice [[TRANSPOSE]] [0, 2, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_3:%.+]] = IE.Slice [[TRANSPOSE]] [0, 3, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_4:%.+]] = IE.Slice [[TRANSPOSE]] [0, 4, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_5:%.+]] = IE.Slice [[TRANSPOSE]] [0, 5, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_6:%.+]] = IE.Slice [[TRANSPOSE]] [0, 6, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_7:%.+]] = IE.Slice [[TRANSPOSE]] [0, 7, 0, 0] [1, 1, 128, 128]

  // CHECK: [[MATMUL_1_0:%.+]] = IE.MatMul([[SLICE_Q_0]], [[SLICE_K_0]]) {transpose_b}
  // CHECK: [[ADD_0:%.+]] = IE.Add([[MATMUL_1_0]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SLICE_CONCAT_0:%.+]] = IE.Slice [[CONCAT_INPUT]] [0, 0, 0, 0] [1, 1, 128, 1]
  // CHECK: [[CONCAT_0:%.+]] = IE.Concat([[ADD_0]], [[SLICE_CONCAT_0]]) {per_axis = #IE.Concat<axis = 3 : i64>}
  // CHECK: [[SOFTMAX_0:%.+]] = IE.SoftMax([[CONCAT_0]]) {axisInd = 3 : i64}
  // CHECK: [[SLICE_0:%.+]] = IE.Slice [[SOFTMAX_0]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[MATMUL_V_0:%.+]] = IE.MatMul([[SLICE_0]], [[SLICE_V_0]]) {transpose_b}

  // CHECK: [[MATMUL_1_1:%.+]] = IE.MatMul([[SLICE_Q_1]], [[SLICE_K_1]]) {transpose_b}
  // CHECK: [[ADD_1:%.+]] = IE.Add([[MATMUL_1_1]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SLICE_CONCAT_1:%.+]] = IE.Slice [[CONCAT_INPUT]] [0, 1, 0, 0] [1, 1, 128, 1]
  // CHECK: [[CONCAT_1:%.+]] = IE.Concat([[ADD_1]], [[SLICE_CONCAT_1]]) {per_axis = #IE.Concat<axis = 3 : i64>}
  // CHECK: [[SOFTMAX_1:%.+]] = IE.SoftMax([[CONCAT_1]]) {axisInd = 3 : i64}
  // CHECK: [[SLICE_1:%.+]] = IE.Slice [[SOFTMAX_1]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[MATMUL_V_1:%.+]] = IE.MatMul([[SLICE_1]], [[SLICE_V_1]]) {transpose_b}

  // CHECK: [[MATMUL_1_2:%.+]] = IE.MatMul([[SLICE_Q_2]], [[SLICE_K_2]]) {transpose_b}
  // CHECK: [[ADD_2:%.+]] = IE.Add([[MATMUL_1_2]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SLICE_CONCAT_2:%.+]] = IE.Slice [[CONCAT_INPUT]] [0, 2, 0, 0] [1, 1, 128, 1]
  // CHECK: [[CONCAT_2:%.+]] = IE.Concat([[ADD_2]], [[SLICE_CONCAT_2]]) {per_axis = #IE.Concat<axis = 3 : i64>}
  // CHECK: [[SOFTMAX_2:%.+]] = IE.SoftMax([[CONCAT_2]]) {axisInd = 3 : i64}
  // CHECK: [[SLICE_2:%.+]] = IE.Slice [[SOFTMAX_2]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[MATMUL_V_2:%.+]] = IE.MatMul([[SLICE_2]], [[SLICE_V_2]]) {transpose_b}

  // CHECK: [[MATMUL_1_3:%.+]] = IE.MatMul([[SLICE_Q_3]], [[SLICE_K_3]]) {transpose_b}
  // CHECK: [[ADD_3:%.+]] = IE.Add([[MATMUL_1_3]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SLICE_CONCAT_3:%.+]] = IE.Slice [[CONCAT_INPUT]] [0, 3, 0, 0] [1, 1, 128, 1]
  // CHECK: [[CONCAT_3:%.+]] = IE.Concat([[ADD_3]], [[SLICE_CONCAT_3]]) {per_axis = #IE.Concat<axis = 3 : i64>}
  // CHECK: [[SOFTMAX_3:%.+]] = IE.SoftMax([[CONCAT_3]]) {axisInd = 3 : i64}
  // CHECK: [[SLICE_3:%.+]] = IE.Slice [[SOFTMAX_3]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[MATMUL_V_3:%.+]] = IE.MatMul([[SLICE_3]], [[SLICE_V_3]]) {transpose_b}

  // CHECK: [[MATMUL_1_4:%.+]] = IE.MatMul([[SLICE_Q_4]], [[SLICE_K_4]]) {transpose_b}
  // CHECK: [[ADD_4:%.+]] = IE.Add([[MATMUL_1_4]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SLICE_CONCAT_4:%.+]] = IE.Slice [[CONCAT_INPUT]] [0, 4, 0, 0] [1, 1, 128, 1]
  // CHECK: [[CONCAT_4:%.+]] = IE.Concat([[ADD_4]], [[SLICE_CONCAT_4]]) {per_axis = #IE.Concat<axis = 3 : i64>}
  // CHECK: [[SOFTMAX_4:%.+]] = IE.SoftMax([[CONCAT_4]]) {axisInd = 3 : i64}
  // CHECK: [[SLICE_4:%.+]] = IE.Slice [[SOFTMAX_4]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[MATMUL_V_4:%.+]] = IE.MatMul([[SLICE_4]], [[SLICE_V_4]]) {transpose_b}

  // CHECK: [[MATMUL_1_5:%.+]] = IE.MatMul([[SLICE_Q_5]], [[SLICE_K_5]]) {transpose_b}
  // CHECK: [[ADD_5:%.+]] = IE.Add([[MATMUL_1_5]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SLICE_CONCAT_5:%.+]] = IE.Slice [[CONCAT_INPUT]] [0, 5, 0, 0] [1, 1, 128, 1]
  // CHECK: [[CONCAT_5:%.+]] = IE.Concat([[ADD_5]], [[SLICE_CONCAT_5]]) {per_axis = #IE.Concat<axis = 3 : i64>}
  // CHECK: [[SOFTMAX_5:%.+]] = IE.SoftMax([[CONCAT_5]]) {axisInd = 3 : i64}
  // CHECK: [[SLICE_5:%.+]] = IE.Slice [[SOFTMAX_5]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[MATMUL_V_5:%.+]] = IE.MatMul([[SLICE_5]], [[SLICE_V_5]]) {transpose_b}

  // CHECK: [[MATMUL_1_6:%.+]] = IE.MatMul([[SLICE_Q_6]], [[SLICE_K_6]]) {transpose_b}
  // CHECK: [[ADD_6:%.+]] = IE.Add([[MATMUL_1_6]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SLICE_CONCAT_6:%.+]] = IE.Slice [[CONCAT_INPUT]] [0, 6, 0, 0] [1, 1, 128, 1]
  // CHECK: [[CONCAT_6:%.+]] = IE.Concat([[ADD_6]], [[SLICE_CONCAT_6]]) {per_axis = #IE.Concat<axis = 3 : i64>}
  // CHECK: [[SOFTMAX_6:%.+]] = IE.SoftMax([[CONCAT_6]]) {axisInd = 3 : i64}
  // CHECK: [[SLICE_6:%.+]] = IE.Slice [[SOFTMAX_6]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[MATMUL_V_6:%.+]] = IE.MatMul([[SLICE_6]], [[SLICE_V_6]]) {transpose_b}

  // CHECK: [[MATMUL_1_7:%.+]] = IE.MatMul([[SLICE_Q_7]], [[SLICE_K_7]]) {transpose_b}
  // CHECK: [[ADD_7:%.+]] = IE.Add([[MATMUL_1_7]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SLICE_CONCAT_7:%.+]] = IE.Slice [[CONCAT_INPUT]] [0, 7, 0, 0] [1, 1, 128, 1]
  // CHECK: [[CONCAT_7:%.+]] = IE.Concat([[ADD_7]], [[SLICE_CONCAT_7]]) {per_axis = #IE.Concat<axis = 3 : i64>}
  // CHECK: [[SOFTMAX_7:%.+]] = IE.SoftMax([[CONCAT_7]]) {axisInd = 3 : i64}
  // CHECK: [[SLICE_7:%.+]] = IE.Slice [[SOFTMAX_7]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[MATMUL_V_7:%.+]] = IE.MatMul([[SLICE_7]], [[SLICE_V_7]]) {transpose_b}

  // CHECK: [[CONCAT:%.+]] = IE.Concat([[MATMUL_V_0]], [[MATMUL_V_1]], [[MATMUL_V_2]], [[MATMUL_V_3]], [[MATMUL_V_4]], [[MATMUL_V_5]], [[MATMUL_V_6]], [[MATMUL_V_7]])
  // CHECK: return [[CONCAT]]
}

// -----

// CHECK-LABEL: @DontUnrollSDPAPatternDecodingCase
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x1x128xf32>, [[ARG1:%.+]]: tensor<1x8x128x128xf32>, [[MASK:%.+]]: tensor<1x1x1x128xf32>, [[CONCAT_INPUT:%.+]]: tensor<1x8x1x1xf32>, [[V_IN:%.+]]: tensor<1x1x1x128x128xf32>)
func.func @DontUnrollSDPAPatternDecodingCase(%arg0: tensor<1x8x1x128xf32>, %arg1: tensor<1x8x128x128xf32>, %mask: tensor<1x1x1x128xf32>, %concat_input: tensor<1x8x1x1xf32>, %v_in: tensor<1x1x1x128x128xf32>) -> tensor<1x8x1x128xf32> {
  %target_shape = const.Declare tensor<5xsi64> = dense<[1, 1, 8, 128, 128]> : tensor<5xsi64>
  %broadcast = IE.Broadcast(%v_in, %target_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1x128x128xf32>, tensor<5xsi64> -> tensor<1x1x8x128x128xf32>
  %reshape = IE.AffineReshape(%broadcast) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 8, 128, 128]} : tensor<1x1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %transpose = IE.Transpose(%reshape) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x1x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x1x128xf32>
  %1 = IE.Add(%0, %mask) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x1x128xf32>, tensor<1x1x1x128xf32> -> tensor<1x8x1x128xf32>
  %2 = IE.Concat(%1, %concat_input) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x8x1x128xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x1x129xf32>
  %3 = IE.SoftMax(%2) {axisInd = 3 : i64} : tensor<1x8x1x129xf32> -> tensor<1x8x1x129xf32>
  %4 = IE.Slice %3 [0, 0, 0, 0] [1, 8, 1, 128] : tensor<1x8x1x129xf32> to tensor<1x8x1x128xf32>
  %5 = IE.MatMul(%4, %transpose) {transpose_b} : tensor<1x8x1x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x1x128xf32>
  return %5 : tensor<1x8x1x128xf32>

  // CHECK: [[TARGET_SHAPE:%.+]] = const.Declare
  // CHECK: [[BROADCAST:%.+]] = IE.Broadcast([[V_IN]], [[TARGET_SHAPE]])
  // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[BROADCAST]])
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])

  // CHECK: [[MATMUL_1:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
  // CHECK: [[ADD:%.+]] = IE.Add([[MATMUL_1]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[CONCAT:%.+]] = IE.Concat([[ADD]], [[CONCAT_INPUT]]) {per_axis = #IE.Concat<axis = 3 : i64>}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[CONCAT]]) {axisInd = 3 : i64}
  // CHECK: [[SLICE:%.+]] = IE.Slice [[SOFTMAX]] [0, 0, 0, 0] [1, 8, 1, 128]
  // CHECK: [[MATMUL_V:%.+]] = IE.MatMul([[SLICE]], [[TRANSPOSE]]) {transpose_b}
  // CHECK: return [[MATMUL_V]]
}

// -----

module @executors {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

// CHECK-LABEL: @UnrollInefficientSDPAPatternWithTransposeQKV
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x1504x768xf32>)
func.func @UnrollInefficientSDPAPatternWithTransposeQKV(%input: tensor<1x1504x768xf32>) -> tensor<1x12x1504x64xf32> {
  %q_weight = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<768x768xf32>
  %q_scale = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
  %q_bias = const.Declare tensor<1x1x768xf32> = dense<0.000000e+00> : tensor<1x1x768xf32>
  %k_weight = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<768x768xf32>
  %v_weight = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<768x768xf32>
  %v_bias = const.Declare tensor<1x1x768xf32> = dense<0.000000e+00> : tensor<1x1x768xf32>
  %0 = IE.AffineReshape(%input) {dim_mapping = [[0], [0], [1]], shape_value = [1504, 768]} : tensor<1x1504x768xf32> -> tensor<1504x768xf32>
  %1 = IE.FullyConnected(%0, %q_weight) : tensor<1504x768xf32>, tensor<768x768xf32> -> tensor<1504x768xf32>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1504, 768]} : tensor<1504x768xf32> -> tensor<1x1504x768xf32>
  %3 = IE.Multiply(%2, %q_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1504x768xf32>, tensor<1x1x1xf32> -> tensor<1x1504x768xf32>
  %4 = IE.Add(%3, %q_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1504x768xf32>, tensor<1x1x768xf32> -> tensor<1x1504x768xf32>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1504, 12, 64]} : tensor<1x1504x768xf32> -> tensor<1x1504x12x64xf32>
  %6 = IE.Transpose(%5) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1504x12x64xf32> -> tensor<1x12x1504x64xf32>
  %7 = IE.FullyConnected(%0, %k_weight) : tensor<1504x768xf32>, tensor<768x768xf32> -> tensor<1504x768xf32>
  %8 = IE.AffineReshape(%7) {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 1504, 12, 64]} : tensor<1504x768xf32> -> tensor<1x1504x12x64xf32>
  %9 = IE.Transpose(%8) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1504x12x64xf32> -> tensor<1x12x1504x64xf32>
  %10 = IE.MatMul(%6, %9) {transpose_b} : tensor<1x12x1504x64xf32>, tensor<1x12x1504x64xf32> -> tensor<1x12x1504x1504xf32>
  %11 = IE.SoftMax(%10) {axisInd = 3 : i64} : tensor<1x12x1504x1504xf32> -> tensor<1x12x1504x1504xf32>
  %12 = IE.FullyConnected(%0, %v_weight) : tensor<1504x768xf32>, tensor<768x768xf32> -> tensor<1504x768xf32>
  %13 = IE.AffineReshape(%12) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1504, 768]} : tensor<1504x768xf32> -> tensor<1x1504x768xf32>
  %14 = IE.Add(%13, %v_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1504x768xf32>, tensor<1x1x768xf32> -> tensor<1x1504x768xf32>
  %15 = IE.AffineReshape(%14) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1504, 12, 64]} : tensor<1x1504x768xf32> -> tensor<1x1504x12x64xf32>
  %16 = IE.Transpose(%15) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x1504x12x64xf32> -> tensor<1x12x64x1504xf32>
  %17 = IE.MatMul(%11, %16) {transpose_b} : tensor<1x12x1504x1504xf32>, tensor<1x12x64x1504xf32> -> tensor<1x12x1504x64xf32>
  return %17 : tensor<1x12x1504x64xf32>

  // CHECK: [[Q_SCALE:%.+]] = const.Declare tensor<1x1x1xf32>
  // CHECK: [[INPUT_RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])

  // CHECK: [[Q_WEIGHT_0:%.+]] = const.Declare tensor<64x768xf32> = dense<1.000000e+00> : tensor<768x768xf32>, [#const.SubView<[0, 0], [64, 768]>]
  // CHECK: [[Q_FC_0:%.+]] = IE.FullyConnected([[INPUT_RESHAPE]], [[Q_WEIGHT_0]])
  // CHECK: [[Q_RESHAPE_0:%.+]] = IE.AffineReshape([[Q_FC_0]])
  // CHECK: [[Q_MUL_0:%.+]] = IE.Multiply([[Q_RESHAPE_0]], [[Q_SCALE]])
  // CHECK: [[Q_BIAS_0:%.+]] = const.Declare tensor<1x1x64xf32> = dense<0.000000e+00> : tensor<1x1x768xf32>, [#const.SubView<[0, 0, 0], [1, 1, 64]>]
  // CHECK: [[Q_ADD_0:%.+]] = IE.Add([[Q_MUL_0]], [[Q_BIAS_0]])
  // CHECK: [[Q_RESHAPE_HEAD_0:%.+]] = IE.AffineReshape([[Q_ADD_0]])
  // CHECK: [[Q_TRANSPOSE_0:%.+]] = IE.Transpose([[Q_RESHAPE_HEAD_0]])

  // CHECK: [[Q_WEIGHT_11:%.+]] = const.Declare tensor<64x768xf32> = dense<1.000000e+00> : tensor<768x768xf32>, [#const.SubView<[704, 0], [64, 768]>]
  // CHECK: [[Q_FC_11:%.+]] = IE.FullyConnected([[INPUT_RESHAPE]], [[Q_WEIGHT_11]])
  // CHECK: [[Q_RESHAPE_11:%.+]] = IE.AffineReshape([[Q_FC_11]])
  // CHECK: [[Q_MUL_11:%.+]] = IE.Multiply([[Q_RESHAPE_11]], [[Q_SCALE]])
  // CHECK: [[Q_BIAS_11:%.+]] = const.Declare tensor<1x1x64xf32> = dense<0.000000e+00> : tensor<1x1x768xf32>, [#const.SubView<[0, 0, 704], [1, 1, 64]>]
  // CHECK: [[Q_ADD_11:%.+]] = IE.Add([[Q_MUL_11]], [[Q_BIAS_11]])
  // CHECK: [[Q_RESHAPE_HEAD_11:%.+]] = IE.AffineReshape([[Q_ADD_11]])
  // CHECK: [[Q_TRANSPOSE_11:%.+]] = IE.Transpose([[Q_RESHAPE_HEAD_11]])

  // CHECK: [[K_WEIGHT_0:%.+]] = const.Declare tensor<64x768xf32> = dense<1.000000e+00> : tensor<768x768xf32>, [#const.SubView<[0, 0], [64, 768]>]
  // CHECK: [[K_FC_0:%.+]] = IE.FullyConnected([[INPUT_RESHAPE]], [[K_WEIGHT_0]])
  // CHECK: [[K_RESHAPE_0:%.+]] = IE.AffineReshape([[K_FC_0]])
  // CHECK: [[K_TRANSPOSE_0:%.+]] = IE.Transpose([[K_RESHAPE_0]])

  // CHECK: [[K_WEIGHT_11:%.+]] = const.Declare tensor<64x768xf32> = dense<1.000000e+00> : tensor<768x768xf32>, [#const.SubView<[704, 0], [64, 768]>]
  // CHECK: [[K_FC_11:%.+]] = IE.FullyConnected([[INPUT_RESHAPE]], [[K_WEIGHT_11]])
  // CHECK: [[K_RESHAPE_11:%.+]] = IE.AffineReshape([[K_FC_11]])
  // CHECK: [[K_TRANSPOSE_11:%.+]] = IE.Transpose([[K_RESHAPE_11]])

  // CHECK: [[V_WEIGHT_0:%.+]] = const.Declare tensor<64x768xf32> = dense<1.000000e+00> : tensor<768x768xf32>, [#const.SubView<[0, 0], [64, 768]>]
  // CHECK: [[V_FC_0:%.+]] = IE.FullyConnected([[INPUT_RESHAPE]], [[V_WEIGHT_0]])
  // CHECK: [[V_RESHAPE_0:%.+]] = IE.AffineReshape([[V_FC_0]])
  // CHECK: [[V_BIAS_0:%.+]] = const.Declare tensor<1x1x64xf32> = dense<0.000000e+00> : tensor<1x1x768xf32>, [#const.SubView<[0, 0, 0], [1, 1, 64]>]
  // CHECK: [[V_ADD_0:%.+]] = IE.Add([[V_RESHAPE_0]], [[V_BIAS_0]])
  // CHECK: [[V_RESHAPE_HEAD_0:%.+]] = IE.AffineReshape([[V_ADD_0]])
  // CHECK: [[V_TRANSPOSE_0:%.+]] = IE.Transpose([[V_RESHAPE_HEAD_0]])

  // CHECK: [[V_WEIGHT_11:%.+]] = const.Declare tensor<64x768xf32> = dense<1.000000e+00> : tensor<768x768xf32>, [#const.SubView<[704, 0], [64, 768]>]
  // CHECK: [[V_FC_11:%.+]] = IE.FullyConnected([[INPUT_RESHAPE]], [[V_WEIGHT_11]])
  // CHECK: [[V_RESHAPE_11:%.+]] = IE.AffineReshape([[V_FC_11]])
  // CHECK: [[V_BIAS_11:%.+]] = const.Declare tensor<1x1x64xf32> = dense<0.000000e+00> : tensor<1x1x768xf32>, [#const.SubView<[0, 0, 704], [1, 1, 64]>]
  // CHECK: [[V_ADD_11:%.+]] = IE.Add([[V_RESHAPE_11]], [[V_BIAS_11]])
  // CHECK: [[V_RESHAPE_HEAD_11:%.+]] = IE.AffineReshape([[V_ADD_11]])
  // CHECK: [[V_TRANSPOSE_11:%.+]] = IE.Transpose([[V_RESHAPE_HEAD_11]])

  // CHECK: [[MATMUL_1_0:%.+]] = IE.MatMul([[Q_TRANSPOSE_0]], [[K_TRANSPOSE_0]]) {transpose_b}
  // CHECK: [[SOFTMAX_0:%.+]] = IE.SoftMax([[MATMUL_1_0]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_0:%.+]] = IE.MatMul([[SOFTMAX_0]], [[V_TRANSPOSE_0]]) {transpose_b}
  // CHECK: [[MATMUL_1_11:%.+]] = IE.MatMul([[Q_TRANSPOSE_11]], [[K_TRANSPOSE_11]]) {transpose_b}
  // CHECK: [[SOFTMAX_11:%.+]] = IE.SoftMax([[MATMUL_1_11]]) {axisInd = 3 : i64}
  // CHECK: [[MATMUL_V_11:%.+]] = IE.MatMul([[SOFTMAX_11]], [[V_TRANSPOSE_11]]) {transpose_b}
  // CHECK: [[CONCAT:%.+]] = IE.Concat
  // CHECK: return [[CONCAT]]
}
}

// -----

module @executors {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

// CHECK-LABEL: @UnrollSDPAPatternWithFakeQuantize
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x577x768xf32>) -> tensor<1x12x577x64xf32>
func.func @UnrollSDPAPatternWithFakeQuantize(%arg0: tensor<1x577x768xf32>) -> tensor<1x12x577x64xf32> {
  %cst = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 0], [1, 1, 768]>]
  %cst_0 = const.Declare tensor<768x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[0, 0], [768, 1]>]
  %cst_1 = const.Declare tensor<768x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[0, 0], [768, 1]>]
  %cst_2 = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[0, 0], [768, 768]>]
  %cst_3 = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1]>]
  %cst_4 = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1]>]
  %cst_5 = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 768], [1, 1, 768]>]
  %cst_6 = const.Declare tensor<768x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[768, 0], [768, 1]>]
  %cst_7 = const.Declare tensor<768x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[768, 0], [768, 1]>]
  %cst_8 = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[768, 0], [768, 768]>]
  %cst_9 = const.Declare tensor<1x1x768xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 1536], [1, 1, 768]>]
  %cst_11 = const.Declare tensor<768x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[1536, 0], [768, 768]>]
  %cst_12 = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1]>]
  %cst_13 = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1]>]
  %cst_14 = const.Declare tensor<768x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[1536, 0], [768, 1]>]
  %cst_15 = const.Declare tensor<768x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[1536, 0], [768, 1]>]
  %cst_16 = const.Declare tensor<1x15x768xf32> = dense<0.000000e+00> : tensor<1x15x768xf32>
  %0 = IE.Concat(%arg0, %cst_16) {static_offsets = [[0, 0, 0], [0, 577, 0]]} : tensor<1x577x768xf32>, tensor<1x15x768xf32> -> tensor<1x592x768xf32>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1]], shape_value = [592, 768]} : tensor<1x592x768xf32> -> tensor<592x768xf32>
  %2 = IE.FakeQuantize(%cst_11, %cst_12, %cst_13, %cst_14, %cst_15) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<768x768xf32>, tensor<1xf32>, tensor<1xf32>, tensor<768x1xf32>, tensor<768x1xf32> -> tensor<768x768xf32>
  %3 = IE.FullyConnected(%1, %2) : tensor<592x768xf32>, tensor<768x768xf32> -> tensor<592x768xf32>
  %4 = IE.AffineReshape(%3) {dim_mapping = [[0, 1], [2]], shape_value = [1, 592, 768]} : tensor<592x768xf32> -> tensor<1x592x768xf32>
  %5 = IE.Add(%4, %cst_9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x592x768xf32>, tensor<1x1x768xf32> -> tensor<1x592x768xf32>
  %6 = IE.AffineReshape(%5) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 592, 12, 64]} : tensor<1x592x768xf32> -> tensor<1x592x12x64xf32>
  %7 = IE.Transpose(%6) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x592x12x64xf32> -> tensor<1x12x64x592xf32>
  %8 = IE.FakeQuantize(%cst_8, %cst_12, %cst_13, %cst_7, %cst_6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<768x768xf32>, tensor<1xf32>, tensor<1xf32>, tensor<768x1xf32>, tensor<768x1xf32> -> tensor<768x768xf32>
  %9 = IE.FullyConnected(%1, %8) : tensor<592x768xf32>, tensor<768x768xf32> -> tensor<592x768xf32>
  %10 = IE.AffineReshape(%9) {dim_mapping = [[0, 1], [2]], shape_value = [1, 592, 768]} : tensor<592x768xf32> -> tensor<1x592x768xf32>
  %11 = IE.Add(%10, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x592x768xf32>, tensor<1x1x768xf32> -> tensor<1x592x768xf32>
  %12 = IE.AffineReshape(%11) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 592, 12, 64]} : tensor<1x592x768xf32> -> tensor<1x592x12x64xf32>
  %13 = IE.FakeQuantize(%12, %cst_4, %cst_3, %cst_4, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x592x12x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x592x12x64xf32>
  %14 = IE.Transpose(%13) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x592x12x64xf32> -> tensor<1x12x592x64xf32>
  %15 = IE.FakeQuantize(%cst_2, %cst_12, %cst_13, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<768x768xf32>, tensor<1xf32>, tensor<1xf32>, tensor<768x1xf32>, tensor<768x1xf32> -> tensor<768x768xf32>
  %16 = IE.FullyConnected(%1, %15) : tensor<592x768xf32>, tensor<768x768xf32> -> tensor<592x768xf32>
  %17 = IE.AffineReshape(%16) {dim_mapping = [[0, 1], [2]], shape_value = [1, 592, 768]} : tensor<592x768xf32> -> tensor<1x592x768xf32>
  %18 = IE.Add(%17, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x592x768xf32>, tensor<1x1x768xf32> -> tensor<1x592x768xf32>
  %19 = IE.AffineReshape(%18) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 592, 12, 64]} : tensor<1x592x768xf32> -> tensor<1x592x12x64xf32>
  %20 = IE.FakeQuantize(%19, %cst_4, %cst_3, %cst_4, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x592x12x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x592x12x64xf32>
  %21 = IE.Transpose(%20) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x592x12x64xf32> -> tensor<1x12x592x64xf32>
  %22 = IE.MatMul(%21, %14) {transpose_b} : tensor<1x12x592x64xf32>, tensor<1x12x592x64xf32> -> tensor<1x12x592x592xf32>
  %23 = IE.SoftMax(%22) {axisInd = 3 : i64, padSize = 15 : i64} : tensor<1x12x592x592xf32> -> tensor<1x12x592x592xf32>
  %24 = IE.MatMul(%23, %7) {transpose_b} : tensor<1x12x592x592xf32>, tensor<1x12x64x592xf32> -> tensor<1x12x592x64xf32>
  %25 = IE.Slice %24 [0, 0, 0, 0] [1, 12, 577, 64] : tensor<1x12x592x64xf32> to tensor<1x12x577x64xf32>
  return %25 : tensor<1x12x577x64xf32>

  // CHECK: [[PAD:%.+]] = const.Declare tensor<1x15x768xf32> = dense<0.000000e+00> : tensor<1x15x768xf32>
  // CHECK: [[PADDED_INPUT:%.+]] = IE.Concat([[INPUT]], [[PAD]])
  // CHECK-SAME: tensor<1x577x768xf32>, tensor<1x15x768xf32> -> tensor<1x592x768xf32>
  // CHECK: [[ROOT_RESHAPE:%.+]] = IE.AffineReshape([[PADDED_INPUT]])
  // CHECK-SAME: shape_value = [592, 768]

  // CHECK: [[Q0_WEIGHT:%.+]] = const.Declare tensor<64x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[0, 0], [64, 768]>]
  // CHECK: [[Q0_INPUT_LOW:%.+]] = const.Declare tensor<1xf32>
  // CHECK: [[Q0_INPUT_HIGH:%.+]] = const.Declare tensor<1xf32>
  // CHECK: [[Q0_OUTPUT_LOW:%.+]] = const.Declare tensor<64x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[0, 0], [64, 1]>]
  // CHECK: [[Q0_OUTPUT_HIGH:%.+]] = const.Declare tensor<64x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[0, 0], [64, 1]>]
  // CHECK: [[Q0_WEIGHT_FQ:%.+]] = IE.FakeQuantize([[Q0_WEIGHT]], [[Q0_INPUT_LOW]], [[Q0_INPUT_HIGH]], [[Q0_OUTPUT_LOW]], [[Q0_OUTPUT_HIGH]])
  // CHECK-SAME: tensor<64x768xf32>, tensor<1xf32>, tensor<1xf32>, tensor<64x1xf32>, tensor<64x1xf32> -> tensor<64x768xf32>
  // CHECK: [[Q0_FC:%.+]] = IE.FullyConnected([[ROOT_RESHAPE]], [[Q0_WEIGHT_FQ]])
  // CHECK: [[Q0_RESHAPE:%.+]] = IE.AffineReshape([[Q0_FC]])
  // CHECK: [[Q0_BIAS:%.+]] = const.Declare tensor<1x1x64xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 0], [1, 1, 64]>]
  // CHECK: [[Q0_ADD:%.+]] = IE.Add([[Q0_RESHAPE]], [[Q0_BIAS]])
  // CHECK: [[Q0_HEAD_RESHAPE:%.+]] = IE.AffineReshape([[Q0_ADD]])
  // CHECK: [[Q0_ACT_FQ:%.+]] = IE.FakeQuantize([[Q0_HEAD_RESHAPE]]
  // CHECK: [[Q0_TRANSPOSE:%.+]] = IE.Transpose([[Q0_ACT_FQ]])

  // CHECK: [[Q11_WEIGHT:%.+]] = const.Declare tensor<64x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[704, 0], [64, 768]>]
  // CHECK: [[Q11_INPUT_LOW:%.+]] = const.Declare tensor<1xf32>
  // CHECK: [[Q11_INPUT_HIGH:%.+]] = const.Declare tensor<1xf32>
  // CHECK: [[Q11_OUTPUT_LOW:%.+]] = const.Declare tensor<64x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[704, 0], [64, 1]>]
  // CHECK: [[Q11_OUTPUT_HIGH:%.+]] = const.Declare tensor<64x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[704, 0], [64, 1]>]
  // CHECK: [[Q11_WEIGHT_FQ:%.+]] = IE.FakeQuantize([[Q11_WEIGHT]], [[Q11_INPUT_LOW]], [[Q11_INPUT_HIGH]], [[Q11_OUTPUT_LOW]], [[Q11_OUTPUT_HIGH]])
  // CHECK-SAME: tensor<64x768xf32>, tensor<1xf32>, tensor<1xf32>, tensor<64x1xf32>, tensor<64x1xf32> -> tensor<64x768xf32>
  // CHECK: [[Q11_FC:%.+]] = IE.FullyConnected([[ROOT_RESHAPE]], [[Q11_WEIGHT_FQ]])
  // CHECK: [[Q11_RESHAPE:%.+]] = IE.AffineReshape([[Q11_FC]])
  // CHECK: [[Q11_BIAS:%.+]] = const.Declare tensor<1x1x64xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 704], [1, 1, 64]>]
  // CHECK: [[Q11_ADD:%.+]] = IE.Add([[Q11_RESHAPE]], [[Q11_BIAS]])
  // CHECK: [[Q11_HEAD_RESHAPE:%.+]] = IE.AffineReshape([[Q11_ADD]])
  // CHECK: [[Q11_ACT_FQ:%.+]] = IE.FakeQuantize([[Q11_HEAD_RESHAPE]]
  // CHECK: [[Q11_TRANSPOSE:%.+]] = IE.Transpose([[Q11_ACT_FQ]])

  // CHECK: [[K0_WEIGHT:%.+]] = const.Declare tensor<64x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[768, 0], [64, 768]>]
  // CHECK: [[K0_INPUT_LOW:%.+]] = const.Declare tensor<1xf32>
  // CHECK: [[K0_INPUT_HIGH:%.+]] = const.Declare tensor<1xf32>
  // CHECK: [[K0_OUTPUT_LOW:%.+]] = const.Declare tensor<64x1xf32> = dense<0.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[768, 0], [64, 1]>]
  // CHECK: [[K0_OUTPUT_HIGH:%.+]] = const.Declare tensor<64x1xf32> = dense<1.000000e+00> : tensor<2304x1xf32>, [#const.SubView<[768, 0], [64, 1]>]
  // CHECK: [[K0_WEIGHT_FQ:%.+]] = IE.FakeQuantize([[K0_WEIGHT]], [[K0_INPUT_LOW]], [[K0_INPUT_HIGH]], [[K0_OUTPUT_LOW]], [[K0_OUTPUT_HIGH]])
  // CHECK-SAME: tensor<64x768xf32>, tensor<1xf32>, tensor<1xf32>, tensor<64x1xf32>, tensor<64x1xf32> -> tensor<64x768xf32>
  // CHECK: [[K0_FC:%.+]] = IE.FullyConnected([[ROOT_RESHAPE]], [[K0_WEIGHT_FQ]])
  // CHECK: [[K0_RESHAPE:%.+]] = IE.AffineReshape([[K0_FC]])
  // CHECK: [[K0_BIAS:%.+]] = const.Declare tensor<1x1x64xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 768], [1, 1, 64]>]
  // CHECK: [[K0_ADD:%.+]] = IE.Add([[K0_RESHAPE]], [[K0_BIAS]])
  // CHECK: [[K0_HEAD_RESHAPE:%.+]] = IE.AffineReshape([[K0_ADD]])
  // CHECK: [[K0_ACT_FQ:%.+]] = IE.FakeQuantize([[K0_HEAD_RESHAPE]]
  // CHECK: [[K0_TRANSPOSE:%.+]] = IE.Transpose([[K0_ACT_FQ]])

  // CHECK: [[K11_WEIGHT:%.+]] = const.Declare tensor<64x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[1472, 0], [64, 768]>]
  // CHECK: [[K11_WEIGHT_FQ:%.+]] = IE.FakeQuantize([[K11_WEIGHT]]
  // CHECK-SAME: tensor<64x768xf32>, tensor<1xf32>, tensor<1xf32>, tensor<64x1xf32>, tensor<64x1xf32> -> tensor<64x768xf32>
  // CHECK: [[K11_FC:%.+]] = IE.FullyConnected([[ROOT_RESHAPE]], [[K11_WEIGHT_FQ]])
  // CHECK: [[K11_RESHAPE:%.+]] = IE.AffineReshape([[K11_FC]])
  // CHECK: [[K11_BIAS:%.+]] = const.Declare tensor<1x1x64xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 1472], [1, 1, 64]>]
  // CHECK: [[K11_ADD:%.+]] = IE.Add([[K11_RESHAPE]], [[K11_BIAS]])
  // CHECK: [[K11_HEAD_RESHAPE:%.+]] = IE.AffineReshape([[K11_ADD]])
  // CHECK: [[K11_ACT_FQ:%.+]] = IE.FakeQuantize([[K11_HEAD_RESHAPE]]
  // CHECK: [[K11_TRANSPOSE:%.+]] = IE.Transpose([[K11_ACT_FQ]])

  // CHECK: [[V0_WEIGHT:%.+]] = const.Declare tensor<64x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[1536, 0], [64, 768]>]
  // CHECK: [[V0_WEIGHT_FQ:%.+]] = IE.FakeQuantize([[V0_WEIGHT]]
  // CHECK-SAME: tensor<64x768xf32>, tensor<1xf32>, tensor<1xf32>, tensor<64x1xf32>, tensor<64x1xf32> -> tensor<64x768xf32>
  // CHECK: [[V0_FC:%.+]] = IE.FullyConnected([[ROOT_RESHAPE]], [[V0_WEIGHT_FQ]])
  // CHECK: [[V0_RESHAPE:%.+]] = IE.AffineReshape([[V0_FC]])
  // CHECK: [[V0_BIAS:%.+]] = const.Declare tensor<1x1x64xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 1536], [1, 1, 64]>]
  // CHECK: [[V0_ADD:%.+]] = IE.Add([[V0_RESHAPE]], [[V0_BIAS]])
  // CHECK: [[V0_HEAD_RESHAPE:%.+]] = IE.AffineReshape([[V0_ADD]])
  // CHECK: [[V0_TRANSPOSE:%.+]] = IE.Transpose([[V0_HEAD_RESHAPE]])

  // CHECK: [[V11_WEIGHT:%.+]] = const.Declare tensor<64x768xf32> = dense<1.000000e+00> : tensor<2304x768xf32>, [#const.SubView<[2240, 0], [64, 768]>]
  // CHECK: [[V11_WEIGHT_FQ:%.+]] = IE.FakeQuantize([[V11_WEIGHT]]
  // CHECK-SAME: tensor<64x768xf32>, tensor<1xf32>, tensor<1xf32>, tensor<64x1xf32>, tensor<64x1xf32> -> tensor<64x768xf32>
  // CHECK: [[V11_FC:%.+]] = IE.FullyConnected([[ROOT_RESHAPE]], [[V11_WEIGHT_FQ]])
  // CHECK: [[V11_RESHAPE:%.+]] = IE.AffineReshape([[V11_FC]])
  // CHECK: [[V11_BIAS:%.+]] = const.Declare tensor<1x1x64xf32> = dense<2.000000e+00> : tensor<1x1x2304xf32>, [#const.SubView<[0, 0, 2240], [1, 1, 64]>]
  // CHECK: [[V11_ADD:%.+]] = IE.Add([[V11_RESHAPE]], [[V11_BIAS]])
  // CHECK: [[V11_HEAD_RESHAPE:%.+]] = IE.AffineReshape([[V11_ADD]])
  // CHECK: [[V11_TRANSPOSE:%.+]] = IE.Transpose([[V11_HEAD_RESHAPE]])

  // CHECK: [[Q0_MATMUL:%.+]] = IE.MatMul([[Q0_TRANSPOSE]], [[K0_TRANSPOSE]]) {transpose_b}
  // CHECK-SAME: tensor<1x1x592x64xf32>, tensor<1x1x592x64xf32> -> tensor<1x1x592x592xf32>
  // CHECK: [[Q0_SOFTMAX:%.+]] = IE.SoftMax([[Q0_MATMUL]]) {axisInd = 3 : i64, padSize = 15 : i64}
  // CHECK: [[V0_MATMUL:%.+]] = IE.MatMul([[Q0_SOFTMAX]], [[V0_TRANSPOSE]]) {transpose_b}
  // CHECK: [[Q11_MATMUL:%.+]] = IE.MatMul([[Q11_TRANSPOSE]], [[K11_TRANSPOSE]]) {transpose_b}
  // CHECK-SAME: tensor<1x1x592x64xf32>, tensor<1x1x592x64xf32> -> tensor<1x1x592x592xf32>
  // CHECK: [[Q11_SOFTMAX:%.+]] = IE.SoftMax([[Q11_MATMUL]]) {axisInd = 3 : i64, padSize = 15 : i64}
  // CHECK: [[V11_MATMUL:%.+]] = IE.MatMul([[Q11_SOFTMAX]], [[V11_TRANSPOSE]]) {transpose_b}
  // CHECK-SAME: tensor<1x1x592x592xf32>, tensor<1x1x64x592xf32> -> tensor<1x1x592x64xf32>
  // CHECK: [[CONCAT_OUT:%.+]] = IE.Concat
  // CHECK-SAME: tensor<1x12x592x64xf32>
  // CHECK: [[SLICE_OUT:%.+]] = IE.Slice [[CONCAT_OUT]] [0, 0, 0, 0] [1, 12, 577, 64] : tensor<1x12x592x64xf32> to tensor<1x12x577x64xf32>
  // CHECK: return [[SLICE_OUT]] : tensor<1x12x577x64xf32>
}
}

// -----

module @executors {
  config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }

// CHECK-LABEL: @DontUnrollInefficientSDPAWithUnalignedSeqLen
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x129x512xf32>)
func.func @DontUnrollInefficientSDPAWithUnalignedSeqLen(%input: tensor<1x129x512xf32>) -> tensor<1x8x129x64xf32> {
  %q_weight = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %q_scale = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
  %q_bias = const.Declare tensor<1x1x512xf32> = dense<0.000000e+00> : tensor<1x1x512xf32>
  %k_weight = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %v_weight = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %v_bias = const.Declare tensor<1x1x512xf32> = dense<0.000000e+00> : tensor<1x1x512xf32>
  %0 = IE.AffineReshape(%input) {dim_mapping = [[0], [0], [1]], shape_value = [129, 512]} : tensor<1x129x512xf32> -> tensor<129x512xf32>
  %1 = IE.FullyConnected(%0, %q_weight) : tensor<129x512xf32>, tensor<512x512xf32> -> tensor<129x512xf32>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1], [2]], shape_value = [1, 129, 512]} : tensor<129x512xf32> -> tensor<1x129x512xf32>
  %3 = IE.Multiply(%2, %q_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x129x512xf32>, tensor<1x1x1xf32> -> tensor<1x129x512xf32>
  %4 = IE.Add(%3, %q_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x129x512xf32>, tensor<1x1x512xf32> -> tensor<1x129x512xf32>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 129, 8, 64]} : tensor<1x129x512xf32> -> tensor<1x129x8x64xf32>
  %6 = IE.Transpose(%5) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x129x8x64xf32> -> tensor<1x8x129x64xf32>
  %7 = IE.FullyConnected(%0, %k_weight) : tensor<129x512xf32>, tensor<512x512xf32> -> tensor<129x512xf32>
  %8 = IE.AffineReshape(%7) {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 129, 8, 64]} : tensor<129x512xf32> -> tensor<1x129x8x64xf32>
  %9 = IE.Transpose(%8) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x129x8x64xf32> -> tensor<1x8x129x64xf32>
  %10 = IE.MatMul(%6, %9) {transpose_b} : tensor<1x8x129x64xf32>, tensor<1x8x129x64xf32> -> tensor<1x8x129x129xf32>
  %11 = IE.SoftMax(%10) {axisInd = 3 : i64} : tensor<1x8x129x129xf32> -> tensor<1x8x129x129xf32>
  %12 = IE.FullyConnected(%0, %v_weight) : tensor<129x512xf32>, tensor<512x512xf32> -> tensor<129x512xf32>
  %13 = IE.AffineReshape(%12) {dim_mapping = [[0, 1], [2]], shape_value = [1, 129, 512]} : tensor<129x512xf32> -> tensor<1x129x512xf32>
  %14 = IE.Add(%13, %v_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x129x512xf32>, tensor<1x1x512xf32> -> tensor<1x129x512xf32>
  %15 = IE.AffineReshape(%14) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 129, 8, 64]} : tensor<1x129x512xf32> -> tensor<1x129x8x64xf32>
  %16 = IE.Transpose(%15) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x129x8x64xf32> -> tensor<1x8x64x129xf32>
  %17 = IE.MatMul(%11, %16) {transpose_b} : tensor<1x8x129x129xf32>, tensor<1x8x64x129xf32> -> tensor<1x8x129x64xf32>
  return %17 : tensor<1x8x129x64xf32>

  // CHECK-NOT: IE.Slice
  // CHECK: [[INPUT_RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[INPUT_RESHAPE]], {{%.+}})
  // CHECK: [[Q_RESHAPE:%.+]] = IE.AffineReshape([[Q_FC]])
  // CHECK: [[Q_MUL:%.+]] = IE.Multiply([[Q_RESHAPE]], {{%.+}})
  // CHECK: [[Q_ADD:%.+]] = IE.Add([[Q_MUL]], {{%.+}})
  // CHECK: [[Q_RESHAPE_HEAD:%.+]] = IE.AffineReshape([[Q_ADD]])
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_RESHAPE_HEAD]])
  // CHECK: [[K_FC:%.+]] = IE.FullyConnected([[INPUT_RESHAPE]], {{%.+}})
  // CHECK: [[K_RESHAPE:%.+]] = IE.AffineReshape([[K_FC]])
  // CHECK: [[K_TRANSPOSE:%.+]] = IE.Transpose([[K_RESHAPE]])
  // CHECK: [[MATMUL:%.+]] = IE.MatMul([[Q_TRANSPOSE]], [[K_TRANSPOSE]]) {transpose_b}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MATMUL]]) {axisInd = 3 : i64}
  // CHECK: [[V_FC:%.+]] = IE.FullyConnected([[INPUT_RESHAPE]], {{%.+}})
  // CHECK: [[V_RESHAPE:%.+]] = IE.AffineReshape([[V_FC]])
  // CHECK: [[V_ADD:%.+]] = IE.Add([[V_RESHAPE]], {{%.+}})
  // CHECK: [[V_RESHAPE_HEAD:%.+]] = IE.AffineReshape([[V_ADD]])
  // CHECK: [[V_TRANSPOSE:%.+]] = IE.Transpose([[V_RESHAPE_HEAD]])
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[V_TRANSPOSE]]) {transpose_b}
  // CHECK: return [[OUT]]
}
}

// -----

module @executors {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

// CHECK-LABEL: @DontUnrollInefficientSDPAWithLargeOp
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1504x1280xf32>)
func.func @DontUnrollInefficientSDPAWithLargeOp(%input: tensor<1504x1280xf32>) -> tensor<1x20x1504x64xf32> {
  %root_bias = const.Declare tensor<1x1280xf32> = dense<0.000000e+00> : tensor<1x1280xf32>
  %q_weight = const.Declare tensor<1280x1280xf32> = dense<1.000000e+00> : tensor<1280x1280xf32>
  %q_scale = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
  %q_bias = const.Declare tensor<1x1x1280xf32> = dense<0.000000e+00> : tensor<1x1x1280xf32>
  %k_weight = const.Declare tensor<1280x1280xf32> = dense<1.000000e+00> : tensor<1280x1280xf32>
  %v_weight = const.Declare tensor<1280x1280xf32> = dense<1.000000e+00> : tensor<1280x1280xf32>
  %v_bias = const.Declare tensor<1x1x1280xf32> = dense<0.000000e+00> : tensor<1x1x1280xf32>
  %root = IE.Add(%input, %root_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1504x1280xf32>, tensor<1x1280xf32> -> tensor<1504x1280xf32>
  %0 = IE.FullyConnected(%root, %q_weight) : tensor<1504x1280xf32>, tensor<1280x1280xf32> -> tensor<1504x1280xf32>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1504, 1280]} : tensor<1504x1280xf32> -> tensor<1x1504x1280xf32>
  %2 = IE.Multiply(%1, %q_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1504x1280xf32>, tensor<1x1x1xf32> -> tensor<1x1504x1280xf32>
  %3 = IE.Add(%2, %q_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1504x1280xf32>, tensor<1x1x1280xf32> -> tensor<1x1504x1280xf32>
  %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1504, 20, 64]} : tensor<1x1504x1280xf32> -> tensor<1x1504x20x64xf32>
  %5 = IE.Transpose(%4) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1504x20x64xf32> -> tensor<1x20x1504x64xf32>
  %6 = IE.FullyConnected(%root, %k_weight) : tensor<1504x1280xf32>, tensor<1280x1280xf32> -> tensor<1504x1280xf32>
  %7 = IE.AffineReshape(%6) {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 1504, 20, 64]} : tensor<1504x1280xf32> -> tensor<1x1504x20x64xf32>
  %8 = IE.Transpose(%7) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1504x20x64xf32> -> tensor<1x20x1504x64xf32>
  %9 = IE.MatMul(%5, %8) {transpose_b} : tensor<1x20x1504x64xf32>, tensor<1x20x1504x64xf32> -> tensor<1x20x1504x1504xf32>
  %10 = IE.SoftMax(%9) {axisInd = 3 : i64} : tensor<1x20x1504x1504xf32> -> tensor<1x20x1504x1504xf32>
  %11 = IE.FullyConnected(%root, %v_weight) : tensor<1504x1280xf32>, tensor<1280x1280xf32> -> tensor<1504x1280xf32>
  %12 = IE.AffineReshape(%11) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1504, 1280]} : tensor<1504x1280xf32> -> tensor<1x1504x1280xf32>
  %13 = IE.Add(%12, %v_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1504x1280xf32>, tensor<1x1x1280xf32> -> tensor<1x1504x1280xf32>
  %14 = IE.AffineReshape(%13) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1504, 20, 64]} : tensor<1x1504x1280xf32> -> tensor<1x1504x20x64xf32>
  %15 = IE.Transpose(%14) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x1504x20x64xf32> -> tensor<1x20x64x1504xf32>
  %16 = IE.MatMul(%10, %15) {transpose_b} : tensor<1x20x1504x1504xf32>, tensor<1x20x64x1504xf32> -> tensor<1x20x1504x64xf32>
  return %16 : tensor<1x20x1504x64xf32>

  // CHECK-NOT: IE.Slice
  // CHECK: [[ROOT_BIAS:%.+]] = const.Declare tensor<1x1280xf32>
  // CHECK: [[ROOT:%.+]] = IE.Add([[INPUT]], [[ROOT_BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[Q_FC:%.+]] = IE.FullyConnected([[ROOT]], {{%.+}})
  // CHECK: [[Q_RESHAPE:%.+]] = IE.AffineReshape([[Q_FC]])
  // CHECK: [[Q_MUL:%.+]] = IE.Multiply([[Q_RESHAPE]], {{%.+}})
  // CHECK: [[Q_ADD:%.+]] = IE.Add([[Q_MUL]], {{%.+}})
  // CHECK: [[Q_RESHAPE_HEAD:%.+]] = IE.AffineReshape([[Q_ADD]])
  // CHECK: [[Q_TRANSPOSE:%.+]] = IE.Transpose([[Q_RESHAPE_HEAD]])
  // CHECK: [[K_FC:%.+]] = IE.FullyConnected([[ROOT]], {{%.+}})
  // CHECK: [[K_RESHAPE:%.+]] = IE.AffineReshape([[K_FC]])
  // CHECK: [[K_TRANSPOSE:%.+]] = IE.Transpose([[K_RESHAPE]])
  // CHECK: [[MATMUL:%.+]] = IE.MatMul([[Q_TRANSPOSE]], [[K_TRANSPOSE]]) {transpose_b}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MATMUL]]) {axisInd = 3 : i64}
  // CHECK: [[V_FC:%.+]] = IE.FullyConnected([[ROOT]], {{%.+}})
  // CHECK: [[V_RESHAPE:%.+]] = IE.AffineReshape([[V_FC]])
  // CHECK: [[V_ADD:%.+]] = IE.Add([[V_RESHAPE]], {{%.+}})
  // CHECK: [[V_RESHAPE_HEAD:%.+]] = IE.AffineReshape([[V_ADD]])
  // CHECK: [[V_TRANSPOSE:%.+]] = IE.Transpose([[V_RESHAPE_HEAD]])
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[V_TRANSPOSE]]) {transpose_b}
  // CHECK: return [[OUT]]
}
}

// -----

module @executors {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

// CHECK-LABEL: @DontUnrollInefficientSDPAPatternWithAttnMask
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x1504x512xf32>, [[MASK:%.+]]: tensor<1x1x1504x1504xf32>)
func.func @DontUnrollInefficientSDPAPatternWithAttnMask(%input: tensor<1x1504x512xf32>, %mask: tensor<1x1x1504x1504xf32>) -> tensor<1x8x1504x64xf32> {
  %q_weight = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %k_weight = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %v_weight = const.Declare tensor<512x512xf32> = dense<1.000000e+00> : tensor<512x512xf32>
  %0 = IE.AffineReshape(%input) {dim_mapping = [[0], [0], [1]], shape_value = [1504, 512]} : tensor<1x1504x512xf32> -> tensor<1504x512xf32>
  %1 = IE.FullyConnected(%0, %q_weight) : tensor<1504x512xf32>, tensor<512x512xf32> -> tensor<1504x512xf32>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1504, 8, 64]} : tensor<1504x512xf32> -> tensor<1x1504x8x64xf32>
  %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1504x8x64xf32> -> tensor<1x8x1504x64xf32>
  %4 = IE.FullyConnected(%0, %k_weight) : tensor<1504x512xf32>, tensor<512x512xf32> -> tensor<1504x512xf32>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1504, 8, 64]} : tensor<1504x512xf32> -> tensor<1x1504x8x64xf32>
  %6 = IE.Transpose(%5) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1504x8x64xf32> -> tensor<1x8x1504x64xf32>
  %7 = IE.MatMul(%3, %6) {transpose_b} : tensor<1x8x1504x64xf32>, tensor<1x8x1504x64xf32> -> tensor<1x8x1504x1504xf32>
  %8 = IE.Add(%7, %mask) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x1504x1504xf32>, tensor<1x1x1504x1504xf32> -> tensor<1x8x1504x1504xf32>
  %9 = IE.SoftMax(%8) {axisInd = 3 : i64} : tensor<1x8x1504x1504xf32> -> tensor<1x8x1504x1504xf32>
  %10 = IE.FullyConnected(%0, %v_weight) : tensor<1504x512xf32>, tensor<512x512xf32> -> tensor<1504x512xf32>
  %11 = IE.AffineReshape(%10) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1504, 8, 64]} : tensor<1504x512xf32> -> tensor<1x1504x8x64xf32>
  %12 = IE.Transpose(%11) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x1504x8x64xf32> -> tensor<1x8x64x1504xf32>
  %13 = IE.MatMul(%9, %12) {transpose_b} : tensor<1x8x1504x1504xf32>, tensor<1x8x64x1504xf32> -> tensor<1x8x1504x64xf32>
  return %13 : tensor<1x8x1504x64xf32>

  // CHECK-NOT: IE.Slice
  // CHECK: [[MATMUL:%.+]] = IE.MatMul
  // CHECK: [[ADD:%.+]] = IE.Add([[MATMUL]], [[MASK]])
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], {{%.+}}) {transpose_b}
  // CHECK: return [[OUT]]
}
}

// -----

module @executors {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

// CHECK-LABEL: @DontUnrollInefficientSDPAPatternProduceSliceCopy
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x1504x8x64xf32>)
func.func @DontUnrollInefficientSDPAPatternProduceSliceCopy(%input: tensor<1x1504x8x64xf32>) -> tensor<1x8x1504x64xf32> {
  %q_scale = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
  %k_scale = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
  %v_scale = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
  %0 = IE.Multiply(%input, %q_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1504x8x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1504x8x64xf32>
  %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1504x8x64xf32> -> tensor<1x8x1504x64xf32>
  %2 = IE.Multiply(%input, %k_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1504x8x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1504x8x64xf32>
  %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1504x8x64xf32> -> tensor<1x8x1504x64xf32>
  %4 = IE.MatMul(%1, %3) {transpose_b} : tensor<1x8x1504x64xf32>, tensor<1x8x1504x64xf32> -> tensor<1x8x1504x1504xf32>
  %5 = IE.SoftMax(%4) {axisInd = 3 : i64} : tensor<1x8x1504x1504xf32> -> tensor<1x8x1504x1504xf32>
  %6 = IE.Multiply(%input, %v_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1504x8x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1504x8x64xf32>
  %7 = IE.Transpose(%6) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x1504x8x64xf32> -> tensor<1x8x64x1504xf32>
  %8 = IE.MatMul(%5, %7) {transpose_b} : tensor<1x8x1504x1504xf32>, tensor<1x8x64x1504xf32> -> tensor<1x8x1504x64xf32>
  return %8 : tensor<1x8x1504x64xf32>

  // CHECK-NOT: IE.Slice
  // CHECK: [[MATMUL:%.+]] = IE.MatMul
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MATMUL]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], {{%.+}}) {transpose_b}
  // CHECK: return [[OUT]]
}
}
