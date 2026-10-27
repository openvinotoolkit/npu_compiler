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

// -----

// Decode phase (L == 1), batched GQA-structured layout (qBatch > 1) with a channel-aligned head count
// (>= VPU_CHANNEL_ALIGNMENT): the grouped QK MatMul accepts a static scale, so the 1/sqrt(E) scalar is
// placed on the result to fold in via backward PPE fusion.
// CHECK-LABEL: @NPU4DecodeBatchedAlignedScaleOnResult
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<4x8x1x128xf16>, [[ARG1:%.+]]: tensor<4x8x1024x128xf16>, [[ARG2:%.+]]: tensor<4x8x128x1024xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @NPU4DecodeBatchedAlignedScaleOnResult(%arg0: tensor<4x8x1x128xf16>, %arg1: tensor<4x8x1024x128xf16>, %arg2: tensor<4x8x128x1024xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<4x8x1x128xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<4x8x1x128xf16>, tensor<4x8x1024x128xf16>, tensor<4x8x128x1024xf16>, tensor<1x1x1x1xf32> -> tensor<4x8x1x128xf16>
  return %0 : tensor<4x8x1x128xf16>

  // CHECK-NOT: IE.Attention
  // CHECK:     [[QK:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
  // CHECK:     [[SCALED:%.+]] = IE.Multiply([[QK]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:     [[SOFTMAX:%.+]] = IE.SoftMax([[SCALED]]) {axisInd = 3 : i64}
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK:     return [[OUT]]
}

// -----

// Decode phase (L == 1), batched layout but sub-aligned head count (< VPU_CHANNEL_ALIGNMENT): the QK MatMul
// does not accept a static scale, so the cheaper OnQuery placement is kept.
// CHECK-LABEL: @NPU4DecodeSubAlignedScaleOnQuery
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<2x6x1x128xf16>, [[ARG1:%.+]]: tensor<2x6x1024x128xf16>, [[ARG2:%.+]]: tensor<2x6x128x1024xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @NPU4DecodeSubAlignedScaleOnQuery(%arg0: tensor<2x6x1x128xf16>, %arg1: tensor<2x6x1024x128xf16>, %arg2: tensor<2x6x128x1024xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<2x6x1x128xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<2x6x1x128xf16>, tensor<2x6x1024x128xf16>, tensor<2x6x128x1024xf16>, tensor<1x1x1x1xf32> -> tensor<2x6x1x128xf16>
  return %0 : tensor<2x6x1x128xf16>

  // CHECK-NOT: IE.Attention
  // CHECK:     [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:     [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK-NOT: IE.Multiply
  // CHECK:     [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK:     return [[OUT]]
}

// -----

// Decode phase (L == 1), flat single-batch layout (qBatch == 1) even with an aligned head count: the flat
// head layout does not benefit from OnResult, so the cheaper OnQuery placement is kept.
// CHECK-LABEL: @NPU4DecodeFlatHeadsScaleOnQuery
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x28x1x128xf16>, [[ARG1:%.+]]: tensor<1x28x1024x128xf16>, [[ARG2:%.+]]: tensor<1x28x128x1024xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @NPU4DecodeFlatHeadsScaleOnQuery(%arg0: tensor<1x28x1x128xf16>, %arg1: tensor<1x28x1024x128xf16>, %arg2: tensor<1x28x128x1024xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<1x28x1x128xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x28x1x128xf16>, tensor<1x28x1024x128xf16>, tensor<1x28x128x1024xf16>, tensor<1x1x1x1xf32> -> tensor<1x28x1x128xf16>
  return %0 : tensor<1x28x1x128xf16>

  // CHECK-NOT: IE.Attention
  // CHECK:     [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:     [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK-NOT: IE.Multiply
  // CHECK:     [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK:     return [[OUT]]
}

// -----

// Decode phase (L == 1) batched GQA/MQA (qBatch > 1, qHeads > kHeads): Q's batch is folded into channels
// ([4, 7, ...] -> [1, 28, ...]), K/V heads are broadcast to match, the aligned grouped QK MatMul takes the
// scale on its result, and the output is reshaped back to the original [4, 7, ...] layout.
// CHECK-LABEL: @NPU4DecodeBatchedGQAFold
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<4x7x1x128xf16>, [[ARG1:%.+]]: tensor<4x1x1024x128xf16>, [[ARG2:%.+]]: tensor<4x1x128x1024xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @NPU4DecodeBatchedGQAFold(%arg0: tensor<4x7x1x128xf16>, %arg1: tensor<4x1x1024x128xf16>, %arg2: tensor<4x1x128x1024xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<4x7x1x128xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<4x7x1x128xf16>, tensor<4x1x1024x128xf16>, tensor<4x1x128x1024xf16>, tensor<1x1x1x1xf32> -> tensor<4x7x1x128xf16>
  return %0 : tensor<4x7x1x128xf16>

  // CHECK-NOT: IE.Attention
  // CHECK:     [[QFOLD:%.+]] = IE.Reshape([[ARG0]]) {{.*}} -> tensor<1x28x1x128xf16>
  // CHECK:     IE.Broadcast
  // CHECK:     [[QK:%.+]] = IE.MatMul([[QFOLD]], %{{.+}}) {transpose_b} : tensor<1x28x1x128xf16>, tensor<1x28x1024x128xf16> -> tensor<1x28x1x1024xf16>
  // CHECK:     [[SCALED:%.+]] = IE.Multiply([[QK]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:     [[SOFTMAX:%.+]] = IE.SoftMax([[SCALED]]) {axisInd = 3 : i64}
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], %{{.+}}) {transpose_b} : tensor<1x28x1x1024xf16>, tensor<1x28x128x1024xf16> -> tensor<1x28x1x128xf16>
  // CHECK:     [[RES:%.+]] = IE.Reshape([[OUT]]) {{.*}} -> tensor<4x7x1x128xf16>
  // CHECK:     return [[RES]]
}
