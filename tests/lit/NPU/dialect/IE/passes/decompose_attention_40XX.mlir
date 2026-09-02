//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --decompose-attention="force-attention-decomposition=true" %s | FileCheck %s
// REQUIRES: platform-NPU4000


// CHECK-LABEL: @ForceDecomposeLegalConfigOnNPU4000
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x16x577x32xf16>, [[ARG1:%.+]]: tensor<1x16x577x32xf16>, [[ARG2:%.+]]: tensor<1x16x32x577xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @ForceDecomposeLegalConfigOnNPU4000(%arg0: tensor<1x16x577x32xf16>, %arg1: tensor<1x16x577x32xf16>, %arg2: tensor<1x16x32x577xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<1x16x577x32xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x16x577x32xf16>, tensor<1x16x577x32xf16>, tensor<1x16x32x577xf16>, tensor<1x1x1x1xf32> -> tensor<1x16x577x32xf16>
  return %0 : tensor<1x16x577x32xf16>

  // CHECK-NOT: IE.Attention
  // CHECK:     [[QK:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
  // CHECK:     [[SCALED:%.+]] = IE.Multiply([[QK]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:     [[SOFTMAX:%.+]] = IE.SoftMax([[SCALED]]) {axisInd = 3 : i64}
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK:     return [[OUT]]
}

// -----

// CHECK-LABEL: @ForceDecomposeLegalHighHeadCountOnNPU4000
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x192x225x16xf16>, [[ARG1:%.+]]: tensor<1x192x225x16xf16>, [[ARG2:%.+]]: tensor<1x192x16x225xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @ForceDecomposeLegalHighHeadCountOnNPU4000(%arg0: tensor<1x192x225x16xf16>, %arg1: tensor<1x192x225x16xf16>, %arg2: tensor<1x192x16x225xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<1x192x225x16xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x192x225x16xf16>, tensor<1x192x225x16xf16>, tensor<1x192x16x225xf16>, tensor<1x1x1x1xf32> -> tensor<1x192x225x16xf16>
  return %0 : tensor<1x192x225x16xf16>

  // CHECK-NOT: IE.Attention
  // CHECK:     [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:     [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK:     [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK:     return [[OUT]]
}

// -----

// CHECK-LABEL: @NPU4ScaleOnQueryOperandAligned
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @NPU4ScaleOnQueryOperandAligned(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<1x8x64x32xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1x1x1x1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK:     [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:     [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK-NOT: IE.Multiply
  // CHECK:     [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK:     return [[OUT]]
}

// -----

// CHECK-LABEL: @NPU4NoScaleAligned
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>)
func.func @NPU4NoScaleAligned(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>) -> tensor<1x8x64x32xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK-NOT: IE.Multiply
  // CHECK:     [[QK:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
  // CHECK:     [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK:     return [[OUT]]
}
