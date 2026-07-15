//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --fuse-ops-to-matmul="enable-grouped-matmul=false" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @PropagateTransposeThroughMulAndReduce
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x4x256x64x64xf16>, [[ARG1:%.+]]: tensor<1x4x256x64x128xf16>
func.func @PropagateTransposeThroughMulAndReduce(%arg0: tensor<1x4x256x64x64xf16>, %arg1: tensor<1x4x256x64x128xf16>) -> tensor<1x4x64x128x64xf16> {
  %t0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>} : tensor<1x4x256x64x64xf16> -> tensor<1x4x64x256x64xf16>
  %r0 = IE.AffineReshape(%t0) {dim_mapping = [[0], [1], [2], [3, 4], [5]], shape_value = [1, 4, 64, 256, 1, 64]} : tensor<1x4x64x256x64xf16> -> tensor<1x4x64x256x1x64xf16>

  %t1 = IE.Transpose(%arg1) {order_value = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>} : tensor<1x4x256x64x128xf16> -> tensor<1x4x64x256x128xf16>
  %r1 = IE.AffineReshape(%t1) {dim_mapping = [[0], [1], [2], [3], [4, 5]], shape_value = [1, 4, 64, 256, 128, 1]} : tensor<1x4x64x256x128xf16> -> tensor<1x4x64x256x128x1xf16>

  %mul = IE.Multiply(%r0, %r1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x64x256x1x64xf16>, tensor<1x4x64x256x128x1xf16> -> tensor<1x4x64x256x128x64xf16>
  %reduce = IE.ReduceSum(%mul) {axes_value = [3]} : tensor<1x4x64x256x128x64xf16> -> tensor<1x4x64x128x64xf16>
  return %reduce : tensor<1x4x64x128x64xf16>
  
  // CHECK: [[RESH0:%.+]] = IE.AffineReshape([[ARG0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3, 4], [5]], shape_value = [1, 4, 256, 64, 1, 64]} : tensor<1x4x256x64x64xf16> -> tensor<1x4x256x64x1x64xf16>
  // CHECK: [[RESH1:%.+]] = IE.AffineReshape([[ARG1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3], [4, 5]], shape_value = [1, 4, 256, 64, 128, 1]} : tensor<1x4x256x64x128xf16> -> tensor<1x4x256x64x128x1xf16>
  // CHECK: [[MUL:%.+]] = IE.Multiply([[RESH0]], [[RESH1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x256x64x1x64xf16>, tensor<1x4x256x64x128x1xf16> -> tensor<1x4x256x64x128x64xf16>
  // CHECK: [[RS:%.+]] = IE.ReduceSum([[MUL]]) {axes_value = [2]} : tensor<1x4x256x64x128x64xf16> -> tensor<1x4x64x128x64xf16>
  // CHECK: return [[RS]] : tensor<1x4x64x128x64xf16>
}

// -----

// CHECK-LABEL: @PropagateTransposeThroughMulAndReducWithDimsPosChanged
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x4x64x256x64xf16>, [[ARG1:%.+]]: tensor<1x4x128x256x64xf16>
func.func @PropagateTransposeThroughMulAndReducWithDimsPosChanged(%arg0: tensor<1x4x64x256x64xf16>, %arg1: tensor<1x4x128x256x64xf16>) -> tensor<1x4x128x64x64xf16> {
  %t0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>} : tensor<1x4x64x256x64xf16> -> tensor<1x4x64x64x256xf16>
  %r0 = IE.AffineReshape(%t0) {dim_mapping = [[0], [1, 2], [3], [4], [5]], shape_value = [1, 4, 1, 64, 64, 256]} : tensor<1x4x64x64x256xf16> -> tensor<1x4x1x64x64x256xf16>

  %t1 = IE.Transpose(%arg1) {order_value = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>} : tensor<1x4x128x256x64xf16> -> tensor<1x4x128x64x256xf16>
  %r1 = IE.AffineReshape(%t1) {dim_mapping = [[0], [1], [2, 3], [4], [5]], shape_value = [1, 4, 128, 1, 64, 256]} : tensor<1x4x128x64x256xf16> -> tensor<1x4x128x1x64x256xf16>

  %mul = IE.Multiply(%r0, %r1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x1x64x64x256xf16>, tensor<1x4x128x1x64x256xf16> -> tensor<1x4x128x64x64x256xf16>
  %reduce = IE.ReduceSum(%mul) {axes_value = [5]} : tensor<1x4x128x64x64x256xf16> -> tensor<1x4x128x64x64xf16>
  return %reduce : tensor<1x4x128x64x64xf16>

  // CHECK: [[RESH0:%.+]] = IE.AffineReshape([[ARG0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3], [4], [5]], shape_value = [1, 4, 1, 64, 256, 64]} : tensor<1x4x64x256x64xf16> -> tensor<1x4x1x64x256x64xf16>
  // CHECK: [[RESH1:%.+]] = IE.AffineReshape([[ARG1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [4], [5]], shape_value = [1, 4, 128, 1, 256, 64]} : tensor<1x4x128x256x64xf16> -> tensor<1x4x128x1x256x64xf16>
  // CHECK: [[MUL:%.+]] = IE.Multiply([[RESH0]], [[RESH1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1x64x256x64xf16>, tensor<1x4x128x1x256x64xf16> -> tensor<1x4x128x64x256x64xf16>
  // CHECK: [[RS:%.+]] = IE.ReduceSum([[MUL]]) {axes_value = [4]} : tensor<1x4x128x64x256x64xf16> -> tensor<1x4x128x64x64xf16>
  // CHECK: return [[RS]] : tensor<1x4x128x64x64xf16>
}

// -----

// CHECK-LABEL: @NotPropagateTransposeThroughMulAndReduceDueToInputShape
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x4x256x64x64xf16>, [[ARG1:%.+]]: tensor<1x4x256x64x128xf16>
func.func @NotPropagateTransposeThroughMulAndReduceDueToInputShape(%arg0: tensor<1x4x256x64x64xf16>, %arg1: tensor<1x4x256x64x128xf16>) -> tensor<1x4x64x128x64xf16> {
  %t0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>} : tensor<1x4x256x64x64xf16> -> tensor<1x4x64x256x64xf16>
  %r0 = IE.AffineReshape(%t0) {dim_mapping = [[0], [1], [2], [3, 4], [5]], shape_value = [1, 4, 64, 256, 1, 64]} : tensor<1x4x64x256x64xf16> -> tensor<1x4x64x256x1x64xf16>

  %t1 = IE.Transpose(%arg1) {order_value = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>} : tensor<1x4x256x64x128xf16> -> tensor<1x4x64x256x128xf16>
  %r1 = IE.AffineReshape(%t1) {dim_mapping = [[0], [0], [1], [2], [3, 4]], shape_value = [4, 64, 256, 128, 1]} : tensor<1x4x64x256x128xf16> -> tensor<4x64x256x128x1xf16>

  // input1 shape: 6D, input2 shape: 5D
  %mul = IE.Multiply(%r0, %r1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x64x256x1x64xf16>, tensor<4x64x256x128x1xf16> -> tensor<1x4x64x256x128x64xf16>
  %reduce = IE.ReduceSum(%mul) {axes_value = [3]} : tensor<1x4x64x256x128x64xf16> -> tensor<1x4x64x128x64xf16>
  return %reduce : tensor<1x4x64x128x64xf16>

  // CHECK: [[TRANS0:%.+]] = IE.Transpose([[ARG0]])
  // CHECK: [[RESH0:%.+]] = IE.AffineReshape([[TRANS0]])
  // CHECK: [[TRANS1:%.+]] = IE.Transpose([[ARG1]])
  // CHECK: [[RESH1:%.+]] = IE.AffineReshape([[TRANS1]])
  // CHECK: [[MUL:%.+]] = IE.Multiply([[RESH0]], [[RESH1]])
  // CHECK: [[RS:%.+]] = IE.ReduceSum([[MUL]])
  // CHECK: return [[RS]] : tensor<1x4x64x128x64xf16>
}

// -----

// CHECK-LABEL: @NotPropagateTransposeThroughMulAndReduceDueToDimsChanged
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x4x256x64x64xf16>, [[ARG1:%.+]]: tensor<1x4x256x64x128xf16>
func.func @NotPropagateTransposeThroughMulAndReduceDueToDimsChanged(%arg0: tensor<1x4x256x64x64xf16>, %arg1: tensor<1x4x256x64x128xf16>) -> tensor<1x4x64x256x128x64xf16> {
  %t0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>} : tensor<1x4x256x64x64xf16> -> tensor<1x4x64x256x64xf16>
  %r0 = IE.AffineReshape(%t0) {dim_mapping = [[0], [1], [2, 3], [4, 5], [6]], shape_value = [1, 4, 64, 1, 256, 1, 64]} : tensor<1x4x64x256x64xf16> -> tensor<1x4x64x1x256x1x64xf16>

  %t1 = IE.Transpose(%arg1) {order_value = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>} : tensor<1x4x256x64x128xf16> -> tensor<1x4x64x256x128xf16>
  %r1 = IE.AffineReshape(%t1) {dim_mapping = [[0], [1], [2], [3, 4], [5, 6]], shape_value = [1, 4, 64, 256, 1, 128, 1]} : tensor<1x4x64x256x128xf16> -> tensor<1x4x64x256x1x128x1xf16>

  %mul = IE.Multiply(%r0, %r1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x64x1x256x1x64xf16>, tensor<1x4x64x256x1x128x1xf16> -> tensor<1x4x64x256x256x128x64xf16>
  %reduce = IE.ReduceSum(%mul) {axes_value = [3]} : tensor<1x4x64x256x256x128x64xf16> -> tensor<1x4x64x256x128x64xf16>
  return %reduce : tensor<1x4x64x256x128x64xf16>
  
  // CHECK: [[TRANS0:%.+]] = IE.Transpose([[ARG0]])
  // CHECK: [[RESH0:%.+]] = IE.AffineReshape([[TRANS0]])
  // CHECK: [[TRANS1:%.+]] = IE.Transpose([[ARG1]])
  // CHECK: [[RESH1:%.+]] = IE.AffineReshape([[TRANS1]])
  // CHECK: [[MUL:%.+]] = IE.Multiply([[RESH0]], [[RESH1]])
  // CHECK: [[RS:%.+]] = IE.ReduceSum([[MUL]])
  // CHECK: return [[RS]] : tensor<1x4x64x256x128x64xf16>
}
