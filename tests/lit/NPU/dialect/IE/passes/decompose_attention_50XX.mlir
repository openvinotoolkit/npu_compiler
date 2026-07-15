//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --decompose-attention %s | FileCheck %s
// REQUIRES: platform-NPU5010

// CHECK-LABEL: @DecomposeBasicAttentionExtended
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @DecomposeBasicAttentionExtended(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<1x8x64x32xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1x1x1x1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeAttentionWithMask
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1x8x64x64xf16>, [[ARG4:%.+]]: tensor<1x1x1x1xf32>)
func.func @DecomposeAttentionWithMask(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1x8x64x64xf16>, %arg4: tensor<1x1x1x1xf32>) -> tensor<1x8x64x32xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1x8x64x64xf16>, tensor<1x1x1x1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[MASKED:%.+]] = IE.Add([[QK]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MASKED]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeAttentionWithCustomScale
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1xf32>)
func.func @DecomposeAttentionWithCustomScale(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1xf32>) -> tensor<1x8x64x32xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeAttentionWithBooleanMask
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x12x512x64xf16>, [[ARG1:%.+]]: tensor<1x12x512x64xf16>, [[ARG2:%.+]]: tensor<1x12x64x512xf16>, [[ARG3:%.+]]: tensor<1x1x512x512xi8>, [[ARG4:%.+]]: tensor<1xf16>)
func.func @DecomposeAttentionWithBooleanMask(%arg0: tensor<1x12x512x64xf16>, %arg1: tensor<1x12x512x64xf16>, %arg2: tensor<1x12x64x512xf16>, %arg3: tensor<1x1x512x512xi8>, %arg4: tensor<1xf16>) -> tensor<1x12x512x64xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>} : tensor<1x12x512x64xf16>, tensor<1x12x512x64xf16>, tensor<1x12x64x512xf16>, tensor<1x1x512x512xi8>, tensor<1xf16> -> tensor<1x12x512x64xf16>
  return %0 : tensor<1x12x512x64xf16>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK-DAG: [[KEEP_VAL:%.+]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
  // CHECK-DAG: [[MASK_VAL:%.+]] = const.Declare tensor<f16> = dense<0xFC00> : tensor<f16>
  // CHECK: [[CONVERTED_MASK:%.+]] = IE.Select([[ARG3]], [[KEEP_VAL]], [[MASK_VAL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[MASKED:%.+]] = IE.Add([[QK]], [[CONVERTED_MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MASKED]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeAttentionWithMaskAware
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1x8x64x64xi8>, [[ARG4:%.+]]: tensor<1x1x1x1xf32>)
func.func @DecomposeAttentionWithMaskAware(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1x8x64x64xi8>, %arg4: tensor<1x1x1x1xf32>) -> tensor<1x8x64x32xf16> {
  %keep_val = const.Declare tensor<f16> = dense<0.0> : tensor<f16>
  %fp16_min_val = const.Declare tensor<f16> = dense<0xFBFF> : tensor<f16>
  %mask = IE.Select(%arg3, %keep_val, %fp16_min_val) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x64x64xi8>, tensor<f16>, tensor<f16> -> tensor<1x8x64x64xf16>
  %0 = IE.Attention(%arg0, %arg1, %arg2, %mask, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1x8x64x64xf16>, tensor<1x1x1x1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // FP16_MIN in the SelectOp false branch is replaced with FP16 -inf
  // CHECK-DAG: [[KEEP_VAL:%.+]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
  // CHECK-DAG: [[MINUS_INF:%.+]] = const.Declare tensor<f16> = dense<0xFC00> : tensor<f16>
  // CHECK: [[MASK:%.+]] = IE.Select([[ARG3]], [[KEEP_VAL]], [[MINUS_INF]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[MASKED:%.+]] = IE.Add([[QK]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MASKED]]) {axisInd = 3 : i64, maskAware}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeAttentionWithBias
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>, [[ARG4:%.+]]: tensor<1x8x64x64xf16>)
func.func @DecomposeAttentionWithBias(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1x1x1x1xf32>, %arg4: tensor<1x8x64x64xf16>) -> tensor<1x8x64x32xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 1>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1x1x1x1xf32>, tensor<1x8x64x64xf16> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[BIASED:%.+]] = IE.Add([[QK]], [[ARG4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[BIASED]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeAttentionWithMaskAndScale
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1x8x64x64xf16>, [[ARG4:%.+]]: tensor<1x1x1x1xf32>)
func.func @DecomposeAttentionWithMaskAndScale(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1x8x64x64xf16>, %arg4: tensor<1x1x1x1xf32>) -> tensor<1x8x64x32xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1x8x64x64xf16>, tensor<1x1x1x1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[MASKED:%.+]] = IE.Add([[QK]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MASKED]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @NotDecomposeLegalAttention
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x192x225x16xf16>, [[ARG1:%.+]]: tensor<1x192x225x16xf16>, [[ARG2:%.+]]: tensor<1x192x16x225xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @NotDecomposeLegalAttention(%arg0: tensor<1x192x225x16xf16>, %arg1: tensor<1x192x225x16xf16>, %arg2: tensor<1x192x16x225xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<1x192x225x16xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x192x225x16xf16>, tensor<1x192x225x16xf16>, tensor<1x192x16x225xf16>, tensor<1x1x1x1xf32> -> tensor<1x192x225x16xf16>
  return %0 : tensor<1x192x225x16xf16>

  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]])
  // CHECK: return [[ATTENTION]]
}

// -----

// CHECK-LABEL: @DecomposeAttentionRank3
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<8x64x32xf16>, [[ARG1:%.+]]: tensor<8x64x32xf16>, [[ARG2:%.+]]: tensor<8x32x64xf16>, [[ARG3:%.+]]: tensor<1x1x1xf32>)
func.func @DecomposeAttentionRank3(%arg0: tensor<8x64x32xf16>, %arg1: tensor<8x64x32xf16>, %arg2: tensor<8x32x64xf16>, %arg3: tensor<1x1x1xf32>) -> tensor<8x64x32xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<8x64x32xf16>, tensor<8x64x32xf16>, tensor<8x32x64xf16>, tensor<1x1x1xf32> -> tensor<8x64x32xf16>
  return %0 : tensor<8x64x32xf16>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 2 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeAttentionWithAllZeroMask
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @DecomposeAttentionWithAllZeroMask(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<1x8x64x32xf16> {
  %mask = const.Declare tensor<1x8x64x64xf16> = dense<0.0> : tensor<1x8x64x64xf16>
  %0 = IE.Attention(%arg0, %arg1, %arg2, %mask, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1x8x64x64xf16>, tensor<1x1x1x1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK-DAG: [[MASK:%.+]] = const.Declare tensor<1x8x64x64xf16> = dense<0.000000e+00>
  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[ARG3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK-NOT: IE.Add
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeAttentionWithAllOneScale
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x8x64x32xf16>, [[ARG2:%.+]]: tensor<1x8x32x64xf16>)
func.func @DecomposeAttentionWithAllOneScale(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x8x64x32xf16>, %arg2: tensor<1x8x32x64xf16>) -> tensor<1x8x64x32xf16> {
  %scale = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
  %0 = IE.Attention(%arg0, %arg1, %arg2, %scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16>, tensor<1x8x32x64xf16>, tensor<1xf32> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00>
  // CHECK: [[QK:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
  // CHECK-NOT: IE.Multiply
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeAttentionWithRecursiveVPreprocessing
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<28x64x128xf32>, [[ARG1:%.+]]: tensor<28x1024x128xf32>, [[ARG2:%.+]]: tensor<1024x1x3584xf32>, [[ARG3:%.+]]: tensor<3584x3584xf32>, [[ARG4:%.+]]: tensor<28x1x1024xf32>)
func.func @DecomposeAttentionWithRecursiveVPreprocessing(%arg0: tensor<28x64x128xf32>, %arg1: tensor<28x1024x128xf32>, %arg2: tensor<1024x1x3584xf32>, %arg3: tensor<3584x3584xf32>, %arg4: tensor<28x1x1024xf32>) -> tensor<28x64x128xf32> {
  %cst_bias = const.Declare tensor<1x1x3584xf32> = dense<1.0> : tensor<1x1x3584xf32>
  %cst_scale = const.Declare tensor<1xf32> = dense<0.088388346> : tensor<1xf32>

  // V preprocessing chain: MatMul -> Add -> AffineReshape -> Transpose
  %0 = IE.MatMul(%arg2, %arg3) {transpose_b} : tensor<1024x1x3584xf32>, tensor<3584x3584xf32> -> tensor<1024x1x3584xf32>
  %1 = IE.Add(%0, %cst_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1x3584xf32>, tensor<1x1x3584xf32> -> tensor<1024x1x3584xf32>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [1, 2]], shape_value = [1024, 28, 128]} : tensor<1024x1x3584xf32> -> tensor<1024x28x128xf32>
  %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : tensor<1024x28x128xf32> -> tensor<28x128x1024xf32>

  %4 = IE.Attention(%arg0, %arg1, %3, %arg4, %cst_scale) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>} : tensor<28x64x128xf32>, tensor<28x1024x128xf32>, tensor<28x128x1024xf32>, tensor<28x1x1024xf32>, tensor<1xf32> -> tensor<28x64x128xf32>
  return %4 : tensor<28x64x128xf32>

  // CHECK: [[CST_BIAS:%.+]] = const.Declare tensor<1x1x3584xf32> = dense<1.000000e+00>
  // CHECK: [[CST_SCALE:%.+]] = const.Declare tensor<1xf32> = dense<{{.*}}>

  // CHECK: [[SCALED_Q:%.+]] = IE.Multiply([[ARG0]], [[CST_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[QK:%.+]] = IE.MatMul([[SCALED_Q]], [[ARG1]]) {transpose_b}
  // CHECK: [[MASKED:%.+]] = IE.Add([[QK]], [[ARG4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MASKED]]) {axisInd = 2 : i64}

  // V preprocessing moved after softmax - the operations are cloned
  // CHECK: [[V_MATMUL:%.+]] = IE.MatMul([[ARG2]], [[ARG3]]) {transpose_b}
  // CHECK: [[V_ADD:%.+]] = IE.Add([[V_MATMUL]], [[CST_BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[V_RESHAPE:%.+]] = IE.AffineReshape([[V_ADD]]) {dim_mapping = {{\[\[}}0], [0], [1, 2]], shape_value = [1024, 28, 128]}
  // CHECK: [[V_TRANSPOSE:%.+]] = IE.Transpose([[V_RESHAPE]])

  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[V_TRANSPOSE]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @NotLegalizeMQAIfDoesNotFitIntoCMX
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x1024x8192xf16>, [[ARG1:%.+]]: tensor<1x1x2048x8192xf16>, [[ARG2:%.+]]: tensor<1x1x8192x2048xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @NotLegalizeMQAIfDoesNotFitIntoCMX(%arg0: tensor<1x32x1024x8192xf16>, %arg1: tensor<1x1x2048x8192xf16>, %arg2: tensor<1x1x8192x2048xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<1x32x1024x8192xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x32x1024x8192xf16>, tensor<1x1x2048x8192xf16>, tensor<1x1x8192x2048xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x8192xf16>
  return %0 : tensor<1x32x1024x8192xf16>

  // CHECK-NOT: IE.Attention
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @DecomposeAttentionWithTransposedVNonSquare
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x1x512xf32>, [[ARG1:%.+]]: tensor<1x8x384x512xf32>, [[ARG2:%.+]]: tensor<1x8x384x512xf32>)
func.func @DecomposeAttentionWithTransposedVNonSquare(
    %arg0: tensor<1x8x1x512xf32>,
    %arg1: tensor<1x8x384x512xf32>,
    %arg2: tensor<1x8x384x512xf32>) -> tensor<1x8x1x512xf32> {
  %transposed_v = IE.Transpose(%arg2) {order_value = #NCWH} : tensor<1x8x384x512xf32> -> tensor<1x8x512x384xf32>
  %transposed_v_anchor = IE.Transpose(%transposed_v) {order_value = #NCWH} : tensor<1x8x512x384xf32> -> tensor<1x8x384x512xf32>
  %0 = IE.Attention(%arg0, %arg1, %transposed_v) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x8x1x512xf32>, tensor<1x8x384x512xf32>, tensor<1x8x512x384xf32> -> tensor<1x8x1x512xf32>
  return %0 : tensor<1x8x1x512xf32>

  // CHECK: [[V_T:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH}
  // CHECK: [[KEEP:%.+]] = IE.Transpose([[V_T]]) {order_value = #NCWH}
  // CHECK: [[QK:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[ARG2]]) :
  // CHECK: return [[OUT]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @DecomposeAttentionWithTransposedVAndPreprocessing
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x1x512xf32>, [[ARG1:%.+]]: tensor<1x8x384x512xf32>, [[ARG2:%.+]]: tensor<1x8x384x512xf32>)
func.func @DecomposeAttentionWithTransposedVAndPreprocessing(
    %arg0: tensor<1x8x1x512xf32>,
    %arg1: tensor<1x8x384x512xf32>,
    %arg2: tensor<1x8x384x512xf32>) -> tensor<1x8x1x512xf32> {
  %cst_scale_v = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
  %scaled_v = IE.Multiply(%arg2, %cst_scale_v) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x384x512xf32>, tensor<1x1x1x1xf32> -> tensor<1x8x384x512xf32>
  %transposed_v = IE.Transpose(%scaled_v) {order_value = #NCWH} : tensor<1x8x384x512xf32> -> tensor<1x8x512x384xf32>
  %transposed_v_anchor = IE.Transpose(%transposed_v) {order_value = #NCWH} : tensor<1x8x512x384xf32> -> tensor<1x8x384x512xf32>
  %0 = IE.Attention(%arg0, %arg1, %transposed_v) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x8x1x512xf32>, tensor<1x8x384x512xf32>, tensor<1x8x512x384xf32> -> tensor<1x8x1x512xf32>
  return %0 : tensor<1x8x1x512xf32>

  // CHECK-DAG: [[CST:%.+]] = const.Declare
  // CHECK: [[V_SCALED:%.+]] = IE.Multiply([[ARG2]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[V_T:%.+]] = IE.Transpose([[V_SCALED]]) {order_value = #NCWH}
  // CHECK: [[KEEP:%.+]] = IE.Transpose([[V_T]]) {order_value = #NCWH}
  // CHECK: [[QK:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[V_SCALED]]) :
  // CHECK: return [[OUT]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @DecomposeAttentionWithTransposedVSquare
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x1x64xf32>, [[ARG1:%.+]]: tensor<1x8x64x64xf32>, [[ARG2:%.+]]: tensor<1x8x64x64xf32>)
func.func @DecomposeAttentionWithTransposedVSquare(
    %arg0: tensor<1x8x1x64xf32>,
    %arg1: tensor<1x8x64x64xf32>,
    %arg2: tensor<1x8x64x64xf32>) -> tensor<1x8x1x64xf32> {
  %transposed_v = IE.Transpose(%arg2) {order_value = #NCWH} : tensor<1x8x64x64xf32> -> tensor<1x8x64x64xf32>
  %0 = IE.Attention(%arg0, %arg1, %transposed_v) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x8x1x64xf32>, tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32> -> tensor<1x8x1x64xf32>
  return %0 : tensor<1x8x1x64xf32>

  // CHECK: [[V_T0:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH}
  // CHECK: [[QK:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64}
  // CHECK: [[REBUILT_T:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH}
  // CHECK: [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[REBUILT_T]]) {transpose_b}
  // CHECK: return [[OUT]]
}

// -----

// {qHeadSize=1, tSL=80, sSL=826} is listed in LEGAL_ATTENTION_CONFIGS, so the
// AttentionOp must be kept intact (no decomposition into MatMul/SoftMax/MatMul).

// CHECK-LABEL: @KeepLegalAttentionMqaFoldedShape
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x80x256xf16>, [[ARG1:%.+]]: tensor<1x1x826x256xf16>, [[ARG2:%.+]]: tensor<1x1x256x826xf16>)
func.func @KeepLegalAttentionMqaFoldedShape(%arg0: tensor<1x1x80x256xf16>, %arg1: tensor<1x1x826x256xf16>, %arg2: tensor<1x1x256x826xf16>) -> tensor<1x1x80x256xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x1x80x256xf16>, tensor<1x1x826x256xf16>, tensor<1x1x256x826xf16> -> tensor<1x1x80x256xf16>
  return %0 : tensor<1x1x80x256xf16>

  // CHECK: [[OUT:%.+]] = IE.Attention([[ARG0]], [[ARG1]], [[ARG2]])
  // CHECK-NOT: IE.MatMul
  // CHECK-NOT: IE.SoftMax
  // CHECK: return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeGQABatch1
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x4x64x32xf16>, [[ARG2:%.+]]: tensor<1x4x32x64xf16>)
func.func @DecomposeGQABatch1(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x4x64x32xf16>, %arg2: tensor<1x4x32x64xf16>) -> tensor<1x8x64x32xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x8x64x32xf16>, tensor<1x4x64x32xf16>, tensor<1x4x32x64xf16> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK:       [[K_5D:%.+]] = IE.AffineReshape([[ARG1]]) {dim_mapping = {{\[\[}}0], [1, 2], [3], [4]], shape_value = [1, 4, 1, 64, 32]} : tensor<1x4x64x32xf16> -> tensor<1x4x1x64x32xf16>
  // CHECK:       [[K_SHAPE:%.+]] = const.Declare tensor<5xsi32> = dense<[1, 4, 2, 64, 32]> : tensor<5xsi64>, [#const.CastElemType<si32>]
  // CHECK:       [[K_BROADCAST:%.+]] = IE.Broadcast([[K_5D]], [[K_SHAPE]]) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x4x1x64x32xf16>, tensor<5xsi32> -> tensor<1x4x2x64x32xf16>
  // CHECK:       [[K_4D:%.+]] = IE.AffineReshape([[K_BROADCAST]]) {dim_mapping = {{\[\[}}0], [1], [1], [2], [3]], shape_value = [1, 8, 64, 32]} : tensor<1x4x2x64x32xf16> -> tensor<1x8x64x32xf16>
  // CHECK:       [[V_SHAPE:%.+]] = const.Declare tensor<5xsi32> = dense<[1, 4, 2, 32, 64]> : tensor<5xsi64>, [#const.CastElemType<si32>]
  // CHECK:       [[QK:%.+]] = IE.MatMul([[ARG0]], [[K_4D]]) {transpose_b} : tensor<1x8x64x32xf16>, tensor<1x8x64x32xf16> -> tensor<1x8x64x64xf16>
  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64} : tensor<1x8x64x64xf16> -> tensor<1x8x64x64xf16>
  // CHECK:       [[V_5D:%.+]] = IE.AffineReshape([[ARG2]]) {dim_mapping = {{\[\[}}0], [1, 2], [3], [4]], shape_value = [1, 4, 1, 32, 64]} : tensor<1x4x32x64xf16> -> tensor<1x4x1x32x64xf16>
  // CHECK:       [[V_BROADCAST:%.+]] = IE.Broadcast([[V_5D]], [[V_SHAPE]]) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x4x1x32x64xf16>, tensor<5xsi32> -> tensor<1x4x2x32x64xf16>
  // CHECK:       [[V_4D:%.+]] = IE.AffineReshape([[V_BROADCAST]]) {dim_mapping = {{\[\[}}0], [1], [1], [2], [3]], shape_value = [1, 8, 32, 64]} : tensor<1x4x2x32x64xf16> -> tensor<1x8x32x64xf16>
  // CHECK:       [[OUT:%.+]] = IE.MatMul([[SOFTMAX]], [[V_4D]]) {transpose_b} : tensor<1x8x64x64xf16>, tensor<1x8x32x64xf16> -> tensor<1x8x64x32xf16>
  // CHECK:       return [[OUT]]
}

// -----

// CHECK-LABEL: @LegalMQABatch1
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x64x32xf16>, [[ARG1:%.+]]: tensor<1x1x64x32xf16>, [[ARG2:%.+]]: tensor<1x1x32x64xf16>)
func.func @LegalMQABatch1(%arg0: tensor<1x8x64x32xf16>, %arg1: tensor<1x1x64x32xf16>, %arg2: tensor<1x1x32x64xf16>) -> tensor<1x8x64x32xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x8x64x32xf16>, tensor<1x1x64x32xf16>, tensor<1x1x32x64xf16> -> tensor<1x8x64x32xf16>
  return %0 : tensor<1x8x64x32xf16>

  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[ARG0]], [[ARG1]], [[ARG2]])
  // CHECK: return [[ATTENTION]]
}

// -----

// CHECK-LABEL: @DecomposeAttentionWithSink
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x1x64xf16>, [[ARG1:%.+]]: tensor<1x8x128x64xf16>, [[ARG2:%.+]]: tensor<1x8x128x64xf16>, [[SINK:%.+]]: tensor<1x8x1x1xf16>)
func.func @DecomposeAttentionWithSink(%arg0: tensor<1x8x1x64xf16>, %arg1: tensor<1x8x128x64xf16>, %arg2: tensor<1x8x128x64xf16>, %sink: tensor<1x8x1x1xf16>) -> tensor<1x8x1x64xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %sink) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1, 0>} : tensor<1x8x1x64xf16>, tensor<1x8x128x64xf16>, tensor<1x8x128x64xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x1x64xf16>
  return %0 : tensor<1x8x1x64xf16>

  // CHECK-NOT: IE.Attention
  // CHECK:     [[QK:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
  // CHECK:     [[CONCAT:%.+]] = IE.Concat([[QK]], [[SINK]]) {{.*}}axis = 3{{.*}} -> tensor<1x8x1x129xf16>
  // CHECK:     [[SM:%.+]] = IE.SoftMax([[CONCAT]]) {axisInd = 3 : i64} : tensor<1x8x1x129xf16> -> tensor<1x8x1x129xf16>
  // CHECK:     [[SLICE:%.+]] = IE.Slice [[SM]] [0, 0, 0, 0] [1, 8, 1, 128] : tensor<1x8x1x129xf16> to tensor<1x8x1x128xf16>
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[SLICE]], [[ARG2]])
  // CHECK:     return [[OUT]]
}

// -----

// CHECK-LABEL: @DecomposeAttentionWithSinkBroadcast
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x1x64xf16>, [[ARG1:%.+]]: tensor<1x8x128x64xf16>, [[ARG2:%.+]]: tensor<1x8x128x64xf16>, [[SINK:%.+]]: tensor<1x1x1x1xf16>)
func.func @DecomposeAttentionWithSinkBroadcast(%arg0: tensor<1x8x1x64xf16>, %arg1: tensor<1x8x128x64xf16>, %arg2: tensor<1x8x128x64xf16>, %sink: tensor<1x1x1x1xf16>) -> tensor<1x8x1x64xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %sink) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1, 0>} : tensor<1x8x1x64xf16>, tensor<1x8x128x64xf16>, tensor<1x8x128x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x8x1x64xf16>
  return %0 : tensor<1x8x1x64xf16>

  // CHECK-NOT: IE.Attention
  // CHECK:     [[QK:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b}
  // CHECK:     [[SINK_BC:%.+]] = IE.Broadcast([[SINK]], {{.*}}) {mode = #IE.broadcast_type<BIDIRECTIONAL>}
  // CHECK:     [[CONCAT:%.+]] = IE.Concat([[QK]], [[SINK_BC]]) {{.*}}axis = 3{{.*}} -> tensor<1x8x1x129xf16>
  // CHECK:     [[SM:%.+]] = IE.SoftMax([[CONCAT]]) {axisInd = 3 : i64} : tensor<1x8x1x129xf16> -> tensor<1x8x1x129xf16>
  // CHECK:     [[SLICE:%.+]] = IE.Slice [[SM]] [0, 0, 0, 0] [1, 8, 1, 128] : tensor<1x8x1x129xf16> to tensor<1x8x1x128xf16>
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[SLICE]], [[ARG2]])
  // CHECK:     return [[OUT]]
}

// -----

// CHECK-LABEL: @KeepGQAAttentionWithSink
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<8x8x1024x64xf32>, [[ARG1:%.+]]: tensor<8x1x1024x64xf32>, [[ARG2:%.+]]: tensor<8x1x64x1024xf32>, [[ARG3:%.+]]: tensor<1x1x1024x1024xf32>, [[ARG4:%.+]]: tensor<1x1x1x1xf32>, [[ARG5:%.+]]: tensor<1x64x1x1xf32>)
func.func @KeepGQAAttentionWithSink(
    %arg0: tensor<8x8x1024x64xf32>,
    %arg1: tensor<8x1x1024x64xf32>,
    %arg2: tensor<8x1x64x1024xf32>,
    %arg3: tensor<1x1x1024x1024xf32>,
    %arg4: tensor<1x1x1x1xf32>,
    %arg5: tensor<1x64x1x1xf32>) -> tensor<8x8x1024x64xf32> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0>} : tensor<8x8x1024x64xf32>, tensor<8x1x1024x64xf32>, tensor<8x1x64x1024xf32>, tensor<1x1x1024x1024xf32>, tensor<1x1x1x1xf32>, tensor<1x64x1x1xf32> -> tensor<8x8x1024x64xf32>
  return %0 : tensor<8x8x1024x64xf32>

  // CHECK:     [[OUT:%.+]] = IE.Attention([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]])
  // CHECK-NOT: IE.MatMul
  // CHECK-NOT: IE.SoftMax
  // CHECK:     return [[OUT]]
}
