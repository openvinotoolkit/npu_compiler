//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-attention --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

// CHECK-LABEL: @Fuse_MM_SM_MM
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x24x225x16xf16>, [[ARG1:%.+]]: tensor<1x24x225x16xf16>, [[ARG2:%.+]]: tensor<1x24x225x16xf16>)
func.func @Fuse_MM_SM_MM(%arg0: tensor<1x24x225x16xf16>, %arg1: tensor<1x24x225x16xf16>, %arg2: tensor<1x24x225x16xf16>) -> tensor<1x24x225x16xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x24x225x16xf16>, tensor<1x24x225x16xf16> -> tensor<1x24x225x225xf16>
  %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x24x225x225xf16> -> tensor<1x24x225x225xf16>
  %2 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x24x225x16xf16> -> tensor<1x24x16x225xf16>
  %3 = IE.MatMul(%1, %2) {transpose_b} : tensor<1x24x225x225xf16>, tensor<1x24x16x225xf16> -> tensor<1x24x225x16xf16>
  return %3 : tensor<1x24x225x16xf16>

  //CHECK: [[TRANS:%.+]] = IE.Transpose
  //CHECK: IE.Attention([[ARG0]], [[ARG1]], [[TRANS]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x24x225x16xf16>, tensor<1x24x225x16xf16>, tensor<1x24x16x225xf16> -> tensor<1x24x225x16xf16>
}

// -----

// CHECK-LABEL: @Fuse_Attention
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x55x128xf32>, [[ARG1:%.+]]: tensor<1x1x80x128xf32>, [[ARG2:%.+]]: tensor<1x1x80x128xf32>)
func.func @Fuse_Attention(%arg0: tensor<1x1x55x128xf32>, %arg1: tensor<1x1x80x128xf32>, %arg2: tensor<1x1x80x128xf32>) -> tensor<1x1x55x128xf32> {
  %cst = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
  %0 = IE.SDPA(%arg0, %arg1, %arg2, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x1x55x128xf32>, tensor<1x1x80x128xf32>, tensor<1x1x80x128xf32>, tensor<1xf32> -> tensor<1x1x55x128xf32>
  return %0 : tensor<1x1x55x128xf32>

  //CHECK: [[SCALE:%.+]] = const.Declare
  //CHECK: [[TRANS:%.+]] = IE.Transpose
  //CHECK: IE.Attention([[ARG0]], [[ARG1]], [[TRANS]],  [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x1x55x128xf32>, tensor<1x1x80x128xf32>, tensor<1x1x128x80xf32>, tensor<1xf32> -> tensor<1x1x55x128xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Fuse_MM_SM_MM_WithConsecutiveTransposes
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x24x225x16xf16>, [[ARG1:%.+]]: tensor<1x24x225x16xf16>, [[ARG2:%.+]]: tensor<1x225x24x16xf16>)
func.func @Fuse_MM_SM_MM_WithConsecutiveTransposes(%arg0: tensor<1x24x225x16xf16>, %arg1: tensor<1x24x225x16xf16>, %arg2: tensor<1x225x24x16xf16>) -> tensor<1x24x225x16xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x24x225x16xf16>, tensor<1x24x225x16xf16> -> tensor<1x24x225x225xf16>
  %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x24x225x225xf16> -> tensor<1x24x225x225xf16>
  %2 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x225x24x16xf16> -> tensor<1x24x225x16xf16>
  %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x24x225x16xf16> -> tensor<1x24x16x225xf16>
  %4 = IE.MatMul(%1, %3) {transpose_b} : tensor<1x24x225x225xf16>, tensor<1x24x16x225xf16> -> tensor<1x24x225x16xf16>
  return %4 : tensor<1x24x225x16xf16>

  // CHECK: [[COMPOSED_TRANS:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NHWC} : tensor<1x225x24x16xf16> -> tensor<1x24x16x225xf16>
  // CHECK: IE.Attention([[ARG0]], [[ARG1]], [[COMPOSED_TRANS]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x24x225x16xf16>, tensor<1x24x225x16xf16>, tensor<1x24x16x225xf16> -> tensor<1x24x225x16xf16>
}

// -----

// CHECK-LABEL: @Fuse_MM_SM_MM_WithBatchToChannels
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<10x11x128x64xf16>, [[ARG1:%.+]]: tensor<10x11x128x64xf16>, [[ARG2:%.+]]: tensor<10x11x128x64xf16>)
func.func @Fuse_MM_SM_MM_WithBatchToChannels(%arg0: tensor<10x11x128x64xf16>, %arg1: tensor<10x11x128x64xf16>, %arg2: tensor<10x11x128x64xf16>) -> tensor<10x11x128x64xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<10x11x128x64xf16>, tensor<10x11x128x64xf16> -> tensor<10x11x128x128xf16>
  %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<10x11x128x128xf16> -> tensor<10x11x128x128xf16>
  %2 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<10x11x128x64xf16> -> tensor<10x11x64x128xf16>
  %3 = IE.MatMul(%1, %2) {transpose_b} : tensor<10x11x128x128xf16>, tensor<10x11x64x128xf16> -> tensor<10x11x128x64xf16>
  return %3 : tensor<10x11x128x64xf16>

  // CHECK: [[TRANS:%.+]] = IE.Transpose([[ARG2]])
  // CHECK: [[RESHAPE_Q:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [1, 110, 128, 64]} : tensor<10x11x128x64xf16> -> tensor<1x110x128x64xf16>
  // CHECK: [[RESHAPE_K:%.+]] = IE.Reshape([[ARG1]]) {shape_value = [1, 110, 128, 64]} : tensor<10x11x128x64xf16> -> tensor<1x110x128x64xf16>
  // CHECK: [[RESHAPE_V:%.+]] = IE.Reshape([[TRANS]]) {shape_value = [1, 110, 64, 128]} : tensor<10x11x64x128xf16> -> tensor<1x110x64x128xf16>
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[RESHAPE_Q]], [[RESHAPE_K]], [[RESHAPE_V]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x110x128x64xf16>, tensor<1x110x128x64xf16>, tensor<1x110x64x128xf16> -> tensor<1x110x128x64xf16>
  // CHECK: [[RESHAPE_BACK:%.+]] = IE.Reshape([[ATTENTION]]) {shape_value = [10, 11, 128, 64]} : tensor<1x110x128x64xf16> -> tensor<10x11x128x64xf16>
  // CHECK: return [[RESHAPE_BACK]]
}

// -----

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>

// CHECK-LABEL: @Fuse_Attention_WithConsecutiveTransposes
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<28x64x128xf32>, [[ARG1:%.+]]: tensor<28x1024x128xf32>, [[ARG2:%.+]]: tensor<1024x28x128xf32>)
func.func @Fuse_Attention_WithConsecutiveTransposes(%arg0: tensor<28x64x128xf32>, %arg1: tensor<28x1024x128xf32>, %arg2: tensor<1024x28x128xf32>) -> tensor<28x64x128xf32> {
  %cst = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
  %0 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2) -> (d1, d0, d2)>} : tensor<1024x28x128xf32> -> tensor<28x1024x128xf32>
  %1 = IE.SDPA(%arg0, %arg1, %0, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<28x64x128xf32>, tensor<28x1024x128xf32>, tensor<28x1024x128xf32>, tensor<1xf32> -> tensor<28x64x128xf32>
  return %1 : tensor<28x64x128xf32>

  // CHECK: [[SCALE:%.+]] = const.Declare
  // CHECK: [[COMPOSED_TRANS:%.+]] = IE.Transpose([[ARG2]]) {order_value = #HWC} : tensor<1024x28x128xf32> -> tensor<28x128x1024xf32>
  // CHECK: IE.Attention([[ARG0]], [[ARG1]], [[COMPOSED_TRANS]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<28x64x128xf32>, tensor<28x1024x128xf32>, tensor<28x128x1024xf32>, tensor<1xf32> -> tensor<28x64x128xf32>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @Fuse_Attention_WithBatchToChannels
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<10x11x128x64xf16>, [[ARG1:%.+]]: tensor<10x11x128x64xf16>, [[ARG2:%.+]]: tensor<10x11x128x64xf16>)
func.func @Fuse_Attention_WithBatchToChannels(%arg0: tensor<10x11x128x64xf16>, %arg1: tensor<10x11x128x64xf16>, %arg2: tensor<10x11x128x64xf16>) -> tensor<10x11x128x64xf16> {
  %cst = const.Declare tensor<1xf32> = dense<8.000000e+00> : tensor<1xf32>
  %1 = IE.SDPA(%arg0, %arg1, %arg2, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<10x11x128x64xf16>, tensor<10x11x128x64xf16>, tensor<10x11x128x64xf16>, tensor<1xf32> -> tensor<10x11x128x64xf16>
  return %1 : tensor<10x11x128x64xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare
  // CHECK: [[TRANS:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH} : tensor<10x11x128x64xf16> -> tensor<10x11x64x128xf16>
  // CHECK: [[RESHAPE_Q:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [1, 110, 128, 64]} : tensor<10x11x128x64xf16> -> tensor<1x110x128x64xf16>
  // CHECK: [[RESHAPE_K:%.+]] = IE.Reshape([[ARG1]]) {shape_value = [1, 110, 128, 64]} : tensor<10x11x128x64xf16> -> tensor<1x110x128x64xf16>
  // CHECK: [[RESHAPE_V:%.+]] = IE.Reshape([[TRANS]]) {shape_value = [1, 110, 64, 128]} : tensor<10x11x64x128xf16> -> tensor<1x110x64x128xf16>
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[RESHAPE_Q]], [[RESHAPE_K]], [[RESHAPE_V]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x110x128x64xf16>, tensor<1x110x128x64xf16>, tensor<1x110x64x128xf16>, tensor<1xf32> -> tensor<1x110x128x64xf16>
  // CHECK: [[RESHAPE_BACK:%.+]] = IE.Reshape([[ATTENTION]]) {shape_value = [10, 11, 128, 64]} : tensor<1x110x128x64xf16> -> tensor<10x11x128x64xf16>
  // CHECK: return [[RESHAPE_BACK]]
}

// -----

// CHECK-LABEL: @Fuse_Attention_WithIllegalBatchToChannels
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<10x11x128x64xf16>, [[ARG1:%.+]]: tensor<10x11x128x64xf16>, [[ARG2:%.+]]: tensor<10x11x128x64xf16>, [[ARG3:%.+]]: tensor<10x1x128x64xf16>)
func.func @Fuse_Attention_WithIllegalBatchToChannels(%arg0: tensor<10x11x128x64xf16>, %arg1: tensor<10x11x128x64xf16>, %arg2: tensor<10x11x128x64xf16>, %arg3: tensor<10x1x128x64xf16>) -> tensor<10x11x128x64xf16> {
  %cst = const.Declare tensor<1xf32> = dense<8.000000e+00> : tensor<1xf32>
  %1 = IE.SDPA(%arg0, %arg1, %arg2, %arg3, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : tensor<10x11x128x64xf16>, tensor<10x11x128x64xf16>, tensor<10x11x128x64xf16>, tensor<10x1x128x64xf16>, tensor<1xf32> -> tensor<10x11x128x64xf16>
  return %1 : tensor<10x11x128x64xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare
  // CHECK: [[TRANS:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH} : tensor<10x11x128x64xf16> -> tensor<10x11x64x128xf16>
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[ARG0]], [[ARG1]], [[TRANS]], [[ARG3]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>} : tensor<10x11x128x64xf16>, tensor<10x11x128x64xf16>, tensor<10x11x64x128xf16>, tensor<10x1x128x64xf16>, tensor<1xf32> -> tensor<10x11x128x64xf16>
  // CHECK: return [[ATTENTION]]
}

// -----

// CHECK-LABEL: @Fuse_Attention_With3DMaskNotBroadcastable
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<64x3x128x64xf16>, [[ARG1:%.+]]: tensor<64x3x128x64xf16>, [[ARG2:%.+]]: tensor<64x3x128x64xf16>, [[ARG3:%.+]]: tensor<3x128x128xf16>)
func.func @Fuse_Attention_With3DMaskNotBroadcastable(%arg0: tensor<64x3x128x64xf16>, %arg1: tensor<64x3x128x64xf16>, %arg2: tensor<64x3x128x64xf16>, %arg3: tensor<3x128x128xf16>) -> tensor<64x3x128x64xf16> {
  %cst = const.Declare tensor<1xf32> = dense<8.000000e+00> : tensor<1xf32>
  %1 = IE.SDPA(%arg0, %arg1, %arg2, %arg3, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : tensor<64x3x128x64xf16>, tensor<64x3x128x64xf16>, tensor<64x3x128x64xf16>, tensor<3x128x128xf16>, tensor<1xf32> -> tensor<64x3x128x64xf16>
  return %1 : tensor<64x3x128x64xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare
  // CHECK: [[TRANS:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH} : tensor<64x3x128x64xf16> -> tensor<64x3x64x128xf16>
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[ARG0]], [[ARG1]], [[TRANS]], [[ARG3]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>} : tensor<64x3x128x64xf16>, tensor<64x3x128x64xf16>, tensor<64x3x64x128xf16>, tensor<3x128x128xf16>, tensor<1xf32> -> tensor<64x3x128x64xf16>
  // CHECK: return [[ATTENTION]]
}

// -----

// CHECK-LABEL: @NotFuse_MM_SM_MM_With5DMask
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x24x225x16xf16>, [[ARG1:%.+]]: tensor<1x24x225x16xf16>, [[ARG2:%.+]]: tensor<1x24x225x16xf16>, [[ARG3:%.+]]: tensor<1x1x1x225x225xf16>)
func.func @NotFuse_MM_SM_MM_With5DMask(%arg0: tensor<1x24x225x16xf16>, %arg1: tensor<1x24x225x16xf16>, %arg2: tensor<1x24x225x16xf16>, %arg3: tensor<1x1x1x225x225xf16>) -> tensor<1x24x225x16xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x24x225x16xf16>, tensor<1x24x225x16xf16> -> tensor<1x24x225x225xf16>
  %1 = IE.Reshape(%0) {shape_value = [1, 1, 24, 225, 225]} : tensor<1x24x225x225xf16> -> tensor<1x1x24x225x225xf16>
  %2 = IE.Add(%1, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x24x225x225xf16>, tensor<1x1x1x225x225xf16> -> tensor<1x1x24x225x225xf16>
  %3 = IE.Reshape(%2) {shape_value = [1, 24, 225, 225]} : tensor<1x1x24x225x225xf16> -> tensor<1x24x225x225xf16>
  %4 = IE.SoftMax(%3) {axisInd = 3 : i64} : tensor<1x24x225x225xf16> -> tensor<1x24x225x225xf16>
  %5 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x24x225x16xf16> -> tensor<1x24x16x225xf16>
  %6 = IE.MatMul(%4, %5) {transpose_b} : tensor<1x24x225x225xf16>, tensor<1x24x16x225xf16> -> tensor<1x24x225x16xf16>
  return %6 : tensor<1x24x225x16xf16>

  // CHECK-NOT: IE.Attention
  // CHECK: IE.MatMul
}

// -----

// CHECK-LABEL: @NoFuseAttention_ReshapeSwapsLastTwoDimsBetweenMatMulAndSoftmax
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x4x320x32xf16>, [[ARG1:%.+]]: tensor<1x4x1x32xf16>, [[ARG2:%.+]]: tensor<1x4x32x320xf16>)
func.func @NoFuseAttention_ReshapeSwapsLastTwoDimsBetweenMatMulAndSoftmax(%arg0: tensor<1x4x320x32xf16>, %arg1: tensor<1x4x1x32xf16>, %arg2: tensor<1x4x32x320xf16>) -> tensor<1x4x1x32xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x4x320x32xf16>, tensor<1x4x1x32xf16> -> tensor<1x4x320x1xf16>
  %1 = IE.Reshape(%0) {shape_value = [1, 4, 1, 320]} : tensor<1x4x320x1xf16> -> tensor<1x4x1x320xf16>
  %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<1x4x1x320xf16> -> tensor<1x4x1x320xf16>
  %3 = IE.MatMul(%2, %arg2) {transpose_b} : tensor<1x4x1x320xf16>, tensor<1x4x32x320xf16> -> tensor<1x4x1x32xf16>
  return %3 : tensor<1x4x1x32xf16>

  // CHECK-NOT: IE.Attention
}

// -----

// CHECK-LABEL: @NoFuseAttention_TransposeSwapsLastTwoDimsBetweenMatMulAndSoftmax
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x4x320x32xf16>, [[ARG1:%.+]]: tensor<1x4x320x32xf16>, [[ARG2:%.+]]: tensor<1x4x32x320xf16>)
func.func @NoFuseAttention_TransposeSwapsLastTwoDimsBetweenMatMulAndSoftmax(%arg0: tensor<1x4x320x32xf16>, %arg1: tensor<1x4x320x32xf16>, %arg2: tensor<1x4x32x320xf16>) -> tensor<1x4x320x32xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x4x320x32xf16>, tensor<1x4x320x32xf16> -> tensor<1x4x320x320xf16>
  %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x4x320x320xf16> -> tensor<1x4x320x320xf16>
  %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<1x4x320x320xf16> -> tensor<1x4x320x320xf16>
  %3 = IE.MatMul(%2, %arg2) {transpose_b} : tensor<1x4x320x320xf16>, tensor<1x4x32x320xf16> -> tensor<1x4x320x32xf16>
  return %3 : tensor<1x4x320x32xf16>

  // CHECK-NOT: IE.Attention
}

// -----

// CHECK-LABEL: @Fuse_Attention_MQA
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x1024x64xf16>, [[ARG1:%.+]]: tensor<1x1x1024x64xf16>, [[ARG2:%.+]]: tensor<1x1x1024x64xf16>)
func.func @Fuse_Attention_MQA(%arg0: tensor<1x32x1024x64xf16>, %arg1: tensor<1x1x1024x64xf16>, %arg2: tensor<1x1x1024x64xf16>) -> tensor<1x32x1024x64xf16> {
  %cst = const.Declare tensor<1xf32> = dense<0.125> : tensor<1xf32>
  %0 = IE.SDPA(%arg0, %arg1, %arg2, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x32x1024x64xf16>, tensor<1x1x1024x64xf16>, tensor<1x1x1024x64xf16>, tensor<1xf32> -> tensor<1x32x1024x64xf16>
  return %0 : tensor<1x32x1024x64xf16>

  // CHECK: [[SCALE:%.+]] = const.Declare
  // CHECK: [[TRANS:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH} : tensor<1x1x1024x64xf16> -> tensor<1x1x64x1024xf16>
  // CHECK: IE.Attention([[ARG0]], [[ARG1]], [[TRANS]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x32x1024x64xf16>, tensor<1x1x1024x64xf16>, tensor<1x1x64x1024xf16>, tensor<1xf32> -> tensor<1x32x1024x64xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @Fuse_MM_SM_MM_MQA
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x40x2048x128xf16>, [[ARG1:%.+]]: tensor<1x1x2048x128xf16>, [[ARG2:%.+]]: tensor<1x1x2048x128xf16>)
func.func @Fuse_MM_SM_MM_MQA(%arg0: tensor<1x40x2048x128xf16>, %arg1: tensor<1x1x2048x128xf16>, %arg2: tensor<1x1x2048x128xf16>) -> tensor<1x40x2048x128xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x40x2048x128xf16>, tensor<1x1x2048x128xf16> -> tensor<1x40x2048x2048xf16>
  %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x40x2048x2048xf16> -> tensor<1x40x2048x2048xf16>
  %2 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x2048x128xf16> -> tensor<1x1x128x2048xf16>
  %3 = IE.MatMul(%1, %2) {transpose_b} : tensor<1x40x2048x2048xf16>, tensor<1x1x128x2048xf16> -> tensor<1x40x2048x128xf16>
  return %3 : tensor<1x40x2048x128xf16>

  // CHECK: [[TRANS:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH} : tensor<1x1x2048x128xf16> -> tensor<1x1x128x2048xf16>
  // CHECK: IE.Attention([[ARG0]], [[ARG1]], [[TRANS]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x40x2048x128xf16>, tensor<1x1x2048x128xf16>, tensor<1x1x128x2048xf16> -> tensor<1x40x2048x128xf16>
}

// -----

// CHECK-LABEL: @NoFuse_Attention_WithComplexBroadcastPattern
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x16x1024x128xf32>, [[ARG1:%.+]]: tensor<1024x1x128xf32>, [[ARG2:%.+]]: tensor<1024x1x128xf32>)
func.func @NoFuse_Attention_WithComplexBroadcastPattern(%arg0: tensor<1x16x1024x128xf32>, %arg1: tensor<1024x1x128xf32>, %arg2: tensor<1024x1x128xf32>) -> tensor<1x16x1024x128xf32> {
  %cst_broadcast = const.Declare tensor<5xsi64> = dense<[1024, 1, 1, 16, 128]> : tensor<5xsi64>
  %cst_scale = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>

  // Key
  %0 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1, 2, 3], [4]], shape_value = [1024, 1, 1, 1, 128]} : tensor<1024x1x128xf32> -> tensor<1024x1x1x1x128xf32>
  %1 = IE.Broadcast(%0, %cst_broadcast) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1024x1x1x1x128xf32>, tensor<5xsi64> -> tensor<1024x1x1x16x128xf32>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1024, 1, 16, 128]} : tensor<1024x1x1x16x128xf32> -> tensor<1024x1x16x128xf32>
  %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>} : tensor<1024x1x16x128xf32> -> tensor<1x16x1024x128xf32>

  // Value
  %4 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1, 2, 3], [4]], shape_value = [1024, 1, 1, 1, 128]} : tensor<1024x1x128xf32> -> tensor<1024x1x1x1x128xf32>
  %5 = IE.Broadcast(%4, %cst_broadcast) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1024x1x1x1x128xf32>, tensor<5xsi64> -> tensor<1024x1x1x16x128xf32>
  %6 = IE.AffineReshape(%5) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1024, 1, 16, 128]} : tensor<1024x1x1x16x128xf32> -> tensor<1024x1x16x128xf32>
  %7 = IE.Transpose(%6) {order_value = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>} : tensor<1024x1x16x128xf32> -> tensor<1x16x1024x128xf32>

  %8 = IE.SDPA(%arg0, %3, %7, %cst_scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x16x1024x128xf32>, tensor<1x16x1024x128xf32>, tensor<1x16x1024x128xf32>, tensor<1xf32> -> tensor<1x16x1024x128xf32>
  return %8 : tensor<1x16x1024x128xf32>

  // CHECK: IE.Attention({{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x16x1024x128xf32>, tensor<1x16x1024x128xf32>, tensor<1x16x128x1024xf32>, tensor<1xf32> -> tensor<1x16x1024x128xf32>
}

// -----

// CHECK-LABEL: @Fuse_Attention_WithSink
// CHECK-SAME:  ([[Q:%.+]]: tensor<1x2x16x8xf16>, [[K:%.+]]: tensor<1x2x16x8xf16>, [[V:%.+]]: tensor<1x2x16x8xf16>, [[MASK:%.+]]: tensor<1x1x16x16xf16>, [[SINK:%.+]]: tensor<1x2x1x1xf16>)
func.func @Fuse_Attention_WithSink(%arg0: tensor<1x2x16x8xf16>, %arg1: tensor<1x2x16x8xf16>, %arg2: tensor<1x2x16x8xf16>, %arg3: tensor<1x1x16x16xf16>, %arg4: tensor<1x2x1x1xf16>) -> tensor<1x2x16x8xf16> {
  %cst = const.Declare tensor<4xsi32> = dense<[1, 2, 16, 1]> : tensor<4xsi32>
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x2x16x8xf16>, tensor<1x2x16x8xf16> -> tensor<1x2x16x16xf16>
  %1 = IE.Add(%0, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16>, tensor<1x1x16x16xf16> -> tensor<1x2x16x16xf16>
  %2 = IE.Broadcast(%arg4, %cst) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x2x1x1xf16>, tensor<4xsi32> -> tensor<1x2x16x1xf16>
  %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]]} : tensor<1x2x16x16xf16>, tensor<1x2x16x1xf16> -> tensor<1x2x16x17xf16>
  %4 = IE.SoftMax(%3) {axisInd = 3 : i64} : tensor<1x2x16x17xf16> -> tensor<1x2x16x17xf16>
  %5 = IE.Slice %4 [0, 0, 0, 0] [1, 2, 16, 16] : tensor<1x2x16x17xf16> to tensor<1x2x16x16xf16>
  %6 = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x2x16x8xf16> -> tensor<1x2x8x16xf16>
  %7 = IE.MatMul(%5, %6) {transpose_b} : tensor<1x2x16x16xf16>, tensor<1x2x8x16xf16> -> tensor<1x2x16x8xf16>
  return %7 : tensor<1x2x16x8xf16>

  // CHECK: [[TRANS:%.+]] = IE.Transpose([[V]]) {order_value = #NCWH} : tensor<1x2x16x8xf16> -> tensor<1x2x8x16xf16>
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q]], [[K]], [[TRANS]], [[MASK]], [[SINK]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>} : tensor<1x2x16x8xf16>, tensor<1x2x16x8xf16>, tensor<1x2x8x16xf16>, tensor<1x1x16x16xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x16x8xf16>
  // CHECK: return [[ATTENTION]]
}

// -----

// CHECK-LABEL: @Fuse_AttentionOp_WithSink
// CHECK-SAME:  ([[Q:%.+]]: tensor<1x2x16x32xf16>, [[K:%.+]]: tensor<1x2x16x32xf16>, [[V:%.+]]: tensor<1x2x16x32xf16>, [[MASK:%.+]]: tensor<1x2x16x16xf16>, [[SINK:%.+]]: tensor<1x2x16x1xf16>)
func.func @Fuse_AttentionOp_WithSink(%Q: tensor<1x2x16x32xf16>, %K: tensor<1x2x16x32xf16>, %V: tensor<1x2x16x32xf16>, %Mask: tensor<1x2x16x16xf16>, %Sink: tensor<1x2x16x1xf16>) -> tensor<1x2x16x32xf16> {
  %cst_scale = const.Declare tensor<1xf16> = dense<1.767580e-01> : tensor<1xf16>
  %sdpa = IE.SDPA(%Q, %K, %V, %Mask, %cst_scale, %Sink) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : tensor<1x2x16x32xf16>, tensor<1x2x16x32xf16>, tensor<1x2x16x32xf16>, tensor<1x2x16x16xf16>, tensor<1xf16>, tensor<1x2x16x1xf16> -> tensor<1x2x16x32xf16>
  return %sdpa : tensor<1x2x16x32xf16>

  // CHECK: [[SCALE:%.+]] = const.Declare tensor<1xf16> = dense<1.767580e-01> : tensor<1xf16>
  // CHECK: [[TRANS:%.+]] = IE.Transpose([[V]]) {order_value = #NCWH} : tensor<1x2x16x32xf16> -> tensor<1x2x32x16xf16>
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[Q]], [[K]], [[TRANS]], [[MASK]], [[SCALE]], [[SINK]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0>} : tensor<1x2x16x32xf16>, tensor<1x2x16x32xf16>, tensor<1x2x32x16xf16>, tensor<1x2x16x16xf16>, tensor<1xf16>, tensor<1x2x16x1xf16> -> tensor<1x2x16x32xf16>
  // CHECK: return [[ATTENTION]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @Fuse_Attention_MQA_WithRank2BroadcastInput
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x16x1024x64xf16>, [[ARG1:%.+]]: tensor<1024x64xf16>, [[ARG2:%.+]]: tensor<1024x64xf16>)
func.func @Fuse_Attention_MQA_WithRank2BroadcastInput(%arg0: tensor<1x16x1024x64xf16>, %arg1: tensor<1024x64xf16>, %arg2: tensor<1024x64xf16>) -> tensor<1x16x1024x64xf16> {
  %cst = const.Declare tensor<1xf32> = dense<1.250000e-01> : tensor<1xf32>
  %cst_0 = const.Declare tensor<5xsi64> = dense<[1, 1, 16, 1024, 64]> : tensor<5xsi64>
  %0 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1024, 64]} : tensor<1024x64xf16> -> tensor<1x1x1x1024x64xf16>
  %1 = IE.Broadcast(%0, %cst_0) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1x1024x64xf16>, tensor<5xsi64> -> tensor<1x1x16x1024x64xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 16, 1024, 64]} : tensor<1x1x16x1024x64xf16> -> tensor<1x16x1024x64xf16>
  %3 = IE.AffineReshape(%arg2) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1024, 64]} : tensor<1024x64xf16> -> tensor<1x1x1x1024x64xf16>
  %4 = IE.Broadcast(%3, %cst_0) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1x1024x64xf16>, tensor<5xsi64> -> tensor<1x1x16x1024x64xf16>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 16, 1024, 64]} : tensor<1x1x16x1024x64xf16> -> tensor<1x16x1024x64xf16>
  %6 = IE.SDPA(%arg0, %2, %5, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x16x1024x64xf16>, tensor<1x16x1024x64xf16>, tensor<1x16x1024x64xf16>, tensor<1xf32> -> tensor<1x16x1024x64xf16>
  return %6 : tensor<1x16x1024x64xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<1.250000e-01> : tensor<1xf32>
  // CHECK: [[RESHAPE_K:%.+]] = IE.AffineReshape([[ARG1]]) {dim_mapping = {{\[}}[0, 1, 2], [3]{{\]}}, shape_value = [1, 1, 1024, 64]} : tensor<1024x64xf16> -> tensor<1x1x1024x64xf16>
  // CHECK: [[RESHAPE_V:%.+]] = IE.AffineReshape([[ARG2]]) {dim_mapping = {{\[}}[0, 1, 2], [3]{{\]}}, shape_value = [1, 1, 1024, 64]} : tensor<1024x64xf16> -> tensor<1x1x1024x64xf16>
  // CHECK: [[TRANS_V:%.+]] = IE.Transpose([[RESHAPE_V]]) {order_value = #NCWH} : tensor<1x1x1024x64xf16> -> tensor<1x1x64x1024xf16>
  // CHECK: IE.Attention([[ARG0]], [[RESHAPE_K]], [[TRANS_V]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x16x1024x64xf16>, tensor<1x1x1024x64xf16>, tensor<1x1x64x1024xf16>, tensor<1xf32> -> tensor<1x16x1024x64xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @Fuse_Attention_MQA_WithRank3BroadcastInput
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x16x1024x64xf16>, [[ARG1:%.+]]: tensor<1x1024x64xf16>, [[ARG2:%.+]]: tensor<1x1024x64xf16>)
func.func @Fuse_Attention_MQA_WithRank3BroadcastInput(%arg0: tensor<1x16x1024x64xf16>, %arg1: tensor<1x1024x64xf16>, %arg2: tensor<1x1024x64xf16>) -> tensor<1x16x1024x64xf16> {
  %cst = const.Declare tensor<1xf32> = dense<1.250000e-01> : tensor<1xf32>
  %cst_0 = const.Declare tensor<5xsi64> = dense<[1, 1, 16, 1024, 64]> : tensor<5xsi64>
  %0 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 1024, 64]} : tensor<1x1024x64xf16> -> tensor<1x1x1024x64xf16>
  %1 = IE.Broadcast(%0, %cst_0) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1024x64xf16>, tensor<5xsi64> -> tensor<1x1x16x1024x64xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 16, 1024, 64]} : tensor<1x1x16x1024x64xf16> -> tensor<1x16x1024x64xf16>
  %3 = IE.AffineReshape(%arg2) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 1024, 64]} : tensor<1x1024x64xf16> -> tensor<1x1x1024x64xf16>
  %4 = IE.Broadcast(%3, %cst_0) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1024x64xf16>, tensor<5xsi64> -> tensor<1x1x16x1024x64xf16>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 16, 1024, 64]} : tensor<1x1x16x1024x64xf16> -> tensor<1x16x1024x64xf16>
  %6 = IE.SDPA(%arg0, %2, %5, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x16x1024x64xf16>, tensor<1x16x1024x64xf16>, tensor<1x16x1024x64xf16>, tensor<1xf32> -> tensor<1x16x1024x64xf16>
  return %6 : tensor<1x16x1024x64xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<1.250000e-01> : tensor<1xf32>
  // CHECK: [[RESHAPE_K:%.+]] = IE.AffineReshape([[ARG1]]) {dim_mapping = {{\[}}[0, 1], [2], [3]{{\]}}, shape_value = [1, 1, 1024, 64]} : tensor<1x1024x64xf16> -> tensor<1x1x1024x64xf16>
  // CHECK: [[RESHAPE_V:%.+]] = IE.AffineReshape([[ARG2]]) {dim_mapping = {{\[}}[0, 1], [2], [3]{{\]}}, shape_value = [1, 1, 1024, 64]} : tensor<1x1024x64xf16> -> tensor<1x1x1024x64xf16>
  // CHECK: [[TRANS_V:%.+]] = IE.Transpose([[RESHAPE_V]]) {order_value = #NCWH} : tensor<1x1x1024x64xf16> -> tensor<1x1x64x1024xf16>
  // CHECK: IE.Attention([[ARG0]], [[RESHAPE_K]], [[TRANS_V]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x16x1024x64xf16>, tensor<1x1x1024x64xf16>, tensor<1x1x64x1024xf16>, tensor<1xf32> -> tensor<1x16x1024x64xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @Fuse_Attention_MQA_WithRank4BatchBroadcastInput
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<2x16x1024x64xf16>, [[ARG1:%.+]]: tensor<2x1x1024x64xf16>, [[ARG2:%.+]]: tensor<2x1x1024x64xf16>)
func.func @Fuse_Attention_MQA_WithRank4BatchBroadcastInput(%arg0: tensor<2x16x1024x64xf16>, %arg1: tensor<2x1x1024x64xf16>, %arg2: tensor<2x1x1024x64xf16>) -> tensor<2x16x1024x64xf16> {
  %cst = const.Declare tensor<1xf32> = dense<1.250000e-01> : tensor<1xf32>
  %cst_0 = const.Declare tensor<5xsi64> = dense<[2, 1, 16, 1024, 64]> : tensor<5xsi64>
  %0 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [2, 1, 1, 1024, 64]} : tensor<2x1x1024x64xf16> -> tensor<2x1x1x1024x64xf16>
  %1 = IE.Broadcast(%0, %cst_0) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<2x1x1x1024x64xf16>, tensor<5xsi64> -> tensor<2x1x16x1024x64xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [2, 16, 1024, 64]} : tensor<2x1x16x1024x64xf16> -> tensor<2x16x1024x64xf16>
  %3 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [2, 1, 1, 1024, 64]} : tensor<2x1x1024x64xf16> -> tensor<2x1x1x1024x64xf16>
  %4 = IE.Broadcast(%3, %cst_0) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<2x1x1x1024x64xf16>, tensor<5xsi64> -> tensor<2x1x16x1024x64xf16>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [2, 16, 1024, 64]} : tensor<2x1x16x1024x64xf16> -> tensor<2x16x1024x64xf16>
  %6 = IE.SDPA(%arg0, %2, %5, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<2x16x1024x64xf16>, tensor<2x16x1024x64xf16>, tensor<2x16x1024x64xf16>, tensor<1xf32> -> tensor<2x16x1024x64xf16>
  return %6 : tensor<2x16x1024x64xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<1.250000e-01> : tensor<1xf32>
  // CHECK: [[TRANS_V:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH} : tensor<2x1x1024x64xf16> -> tensor<2x1x64x1024xf16>
  // CHECK: IE.Attention([[ARG0]], [[ARG1]], [[TRANS_V]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<2x16x1024x64xf16>, tensor<2x1x1024x64xf16>, tensor<2x1x64x1024xf16>, tensor<1xf32> -> tensor<2x16x1024x64xf16>
}

// -----

// CHECK-LABEL: @Fuse_Attention_MQA_WithRank4BatchReshapeBroadcastInput
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<2x16x1024x64xf16>, [[ARG1:%.+]]: tensor<2x1x1024x64xf16>, [[ARG2:%.+]]: tensor<1x1x1024x64xf16>)
func.func @Fuse_Attention_MQA_WithRank4BatchReshapeBroadcastInput(%arg0: tensor<2x16x1024x64xf16>, %arg1: tensor<2x1x1024x64xf16>, %arg2: tensor<1x1x1024x64xf16>) -> tensor<2x16x1024x64xf16> {
  %cst = const.Declare tensor<1xf32> = dense<1.250000e-01> : tensor<1xf32>
  %cst_0 = const.Declare tensor<5xsi64> = dense<[2, 1, 16, 1024, 64]> : tensor<5xsi64>
  %0 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [2, 1, 1, 1024, 64]} : tensor<2x1x1024x64xf16> -> tensor<2x1x1x1024x64xf16>
  %1 = IE.Broadcast(%0, %cst_0) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<2x1x1x1024x64xf16>, tensor<5xsi64> -> tensor<2x1x16x1024x64xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [2, 16, 1024, 64]} : tensor<2x1x16x1024x64xf16> -> tensor<2x16x1024x64xf16>
  %3 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 1, 1, 1024, 64]} : tensor<1x1x1024x64xf16> -> tensor<1x1x1x1024x64xf16>
  %4 = IE.Broadcast(%3, %cst_0) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1x1024x64xf16>, tensor<5xsi64> -> tensor<2x1x16x1024x64xf16>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [2, 16, 1024, 64]} : tensor<2x1x16x1024x64xf16> -> tensor<2x16x1024x64xf16>
  %6 = IE.SDPA(%arg0, %2, %5, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<2x16x1024x64xf16>, tensor<2x16x1024x64xf16>, tensor<2x16x1024x64xf16>, tensor<1xf32> -> tensor<2x16x1024x64xf16>
  return %6 : tensor<2x16x1024x64xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<1.250000e-01> : tensor<1xf32>
  // CHECK: [[RESHAPE_V:%.+]] = IE.Reshape([[ARG2]]) {shape_value = [2, 1, 1024, 64]} : tensor<1x1x1024x64xf16> -> tensor<2x1x1024x64xf16>
  // CHECK: [[TRANS_V:%.+]] = IE.Transpose([[RESHAPE_V]]) {order_value = #NCWH} : tensor<2x1x1024x64xf16> -> tensor<2x1x64x1024xf16>
  // CHECK: IE.Attention([[ARG0]], [[ARG1]], [[TRANS_V]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<2x16x1024x64xf16>, tensor<2x1x1024x64xf16>, tensor<2x1x64x1024xf16>, tensor<1xf32> -> tensor<2x16x1024x64xf16>
}

// -----

// CHECK-LABEL: @MqaFold_Attention_AllowedShape
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x10x256xf16>, [[ARG1:%.+]]: tensor<1x1x826x256xf16>, [[ARG2:%.+]]: tensor<1x1x256x826xf16>)
func.func @MqaFold_Attention_AllowedShape(%arg0: tensor<1x8x10x256xf16>, %arg1: tensor<1x1x826x256xf16>, %arg2: tensor<1x1x256x826xf16>) -> tensor<1x8x10x256xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x8x10x256xf16>, tensor<1x1x826x256xf16>, tensor<1x1x256x826xf16> -> tensor<1x8x10x256xf16>
  return %0 : tensor<1x8x10x256xf16>

  // CHECK: [[RESHAPE_Q:%.+]] = IE.AffineReshape([[ARG0]]) {{.*}}shape_value = [1, 1, 80, 256]{{.*}} : tensor<1x8x10x256xf16> -> tensor<1x1x80x256xf16>
  // CHECK: [[ATT:%.+]] = IE.Attention([[RESHAPE_Q]], [[ARG1]], [[ARG2]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x1x80x256xf16>, tensor<1x1x826x256xf16>, tensor<1x1x256x826xf16> -> tensor<1x1x80x256xf16>
  // CHECK: [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[ATT]]) {{.*}}shape_value = [1, 8, 10, 256]{{.*}} : tensor<1x1x80x256xf16> -> tensor<1x8x10x256xf16>
  // CHECK: return [[RESHAPE_OUT]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @Fuse_Attention_GQA
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x1024x64xf16>, [[ARG1:%.+]]: tensor<1x8x1024x64xf16>, [[ARG2:%.+]]: tensor<1x8x1024x64xf16>)
func.func @Fuse_Attention_GQA(%arg0: tensor<1x32x1024x64xf16>, %arg1: tensor<1x8x1024x64xf16>, %arg2: tensor<1x8x1024x64xf16>) -> tensor<1x32x1024x64xf16> {
  %cst = const.Declare tensor<1xf32> = dense<0.125> : tensor<1xf32>
  %0 = IE.SDPA(%arg0, %arg1, %arg2, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x32x1024x64xf16>, tensor<1x8x1024x64xf16>, tensor<1x8x1024x64xf16>, tensor<1xf32> -> tensor<1x32x1024x64xf16>
  return %0 : tensor<1x32x1024x64xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32>
  // CHECK: [[TRANS_V:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH} : tensor<1x8x1024x64xf16> -> tensor<1x8x64x1024xf16>
  // CHECK: [[RESHAPE_Q:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [8, 4, 1024, 64]} : tensor<1x32x1024x64xf16> -> tensor<8x4x1024x64xf16>
  // CHECK: [[RESHAPE_K:%.+]] = IE.AffineReshape([[ARG1]]) {dim_mapping = {{\[}}[0], [0], [1, 2], [3]{{\]}}, shape_value = [8, 1, 1024, 64]} : tensor<1x8x1024x64xf16> -> tensor<8x1x1024x64xf16>
  // CHECK: [[RESHAPE_V:%.+]] = IE.AffineReshape([[TRANS_V]]) {dim_mapping = {{\[}}[0], [0], [1, 2], [3]{{\]}}, shape_value = [8, 1, 64, 1024]} : tensor<1x8x64x1024xf16> -> tensor<8x1x64x1024xf16>
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[RESHAPE_Q]], [[RESHAPE_K]], [[RESHAPE_V]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<8x4x1024x64xf16>, tensor<8x1x1024x64xf16>, tensor<8x1x64x1024xf16>, tensor<1xf32> -> tensor<8x4x1024x64xf16>
  // CHECK: [[RESHAPE_BACK:%.+]] = IE.Reshape([[ATTENTION]]) {shape_value = [1, 32, 1024, 64]} : tensor<8x4x1024x64xf16> -> tensor<1x32x1024x64xf16>
  // CHECK: return [[RESHAPE_BACK]]
}

// -----

// Negative case: an MQA shape that is NOT in ALLOWED_MQA_FOLD_SHAPES must be
// left unchanged (no Reshape on Q, AttentionOp keeps its qHeads dimension).

// CHECK-LABEL: @MqaFold_Attention_DisallowedShape
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x4x16x128xf16>, [[ARG1:%.+]]: tensor<1x1x512x128xf16>, [[ARG2:%.+]]: tensor<1x1x128x512xf16>)
func.func @MqaFold_Attention_DisallowedShape(%arg0: tensor<1x4x16x128xf16>, %arg1: tensor<1x1x512x128xf16>, %arg2: tensor<1x1x128x512xf16>) -> tensor<1x4x16x128xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x4x16x128xf16>, tensor<1x1x512x128xf16>, tensor<1x1x128x512xf16> -> tensor<1x4x16x128xf16>
  return %0 : tensor<1x4x16x128xf16>

  // CHECK-NOT: IE.Reshape
  // CHECK-NOT: IE.AffineReshape
  // CHECK: [[OUT:%.+]] = IE.Attention([[ARG0]], [[ARG1]], [[ARG2]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x4x16x128xf16>, tensor<1x1x512x128xf16>, tensor<1x1x128x512xf16> -> tensor<1x4x16x128xf16>
  // CHECK: return [[OUT]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @Fuse_Attention_GQA_WithBroadcastInput
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x1024x64xf16>, [[ARG1:%.+]]: tensor<1x8x1024x64xf16>, [[ARG2:%.+]]: tensor<1x8x1024x64xf16>)
func.func @Fuse_Attention_GQA_WithBroadcastInput(%arg0: tensor<1x32x1024x64xf16>, %arg1: tensor<1x8x1024x64xf16>, %arg2: tensor<1x8x1024x64xf16>) -> tensor<1x32x1024x64xf16> {
  %cst = const.Declare tensor<1xf32> = dense<0.125> : tensor<1xf32>
  %cst_0 = const.Declare tensor<5xsi64> = dense<[1, 8, 4, 1024, 64]> : tensor<5xsi64>
  // Key: [1, 8, 1024, 64] -> [1, 8, 1, 1024, 64] -> broadcast -> [1, 8, 4, 1024, 64] -> [1, 32, 1024, 64]
  %0 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 8, 1, 1024, 64]} : tensor<1x8x1024x64xf16> -> tensor<1x8x1x1024x64xf16>
  %1 = IE.Broadcast(%0, %cst_0) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x8x1x1024x64xf16>, tensor<5xsi64> -> tensor<1x8x4x1024x64xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 32, 1024, 64]} : tensor<1x8x4x1024x64xf16> -> tensor<1x32x1024x64xf16>
  // Value: same broadcast pattern
  %3 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 8, 1, 1024, 64]} : tensor<1x8x1024x64xf16> -> tensor<1x8x1x1024x64xf16>
  %4 = IE.Broadcast(%3, %cst_0) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x8x1x1024x64xf16>, tensor<5xsi64> -> tensor<1x8x4x1024x64xf16>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 32, 1024, 64]} : tensor<1x8x4x1024x64xf16> -> tensor<1x32x1024x64xf16>
  %6 = IE.SDPA(%arg0, %2, %5, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x32x1024x64xf16>, tensor<1x32x1024x64xf16>, tensor<1x32x1024x64xf16>, tensor<1xf32> -> tensor<1x32x1024x64xf16>
  return %6 : tensor<1x32x1024x64xf16>

  // broadcast is stripped; ARG1 and ARG2 are used directly as GQA K/V inputs
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32>
  // CHECK: [[TRANS_V:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH} : tensor<1x8x1024x64xf16> -> tensor<1x8x64x1024xf16>
  // CHECK: [[RESHAPE_Q:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [8, 4, 1024, 64]} : tensor<1x32x1024x64xf16> -> tensor<8x4x1024x64xf16>
  // CHECK: [[RESHAPE_K:%.+]] = IE.AffineReshape([[ARG1]]) {dim_mapping = {{\[}}[0], [0], [1, 2], [3]{{\]}}, shape_value = [8, 1, 1024, 64]} : tensor<1x8x1024x64xf16> -> tensor<8x1x1024x64xf16>
  // CHECK: [[RESHAPE_V:%.+]] = IE.AffineReshape([[TRANS_V]]) {dim_mapping = {{\[}}[0], [0], [1, 2], [3]{{\]}}, shape_value = [8, 1, 64, 1024]} : tensor<1x8x64x1024xf16> -> tensor<8x1x64x1024xf16>
  // CHECK: [[ATTENTION:%.+]] = IE.Attention([[RESHAPE_Q]], [[RESHAPE_K]], [[RESHAPE_V]], [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<8x4x1024x64xf16>, tensor<8x1x1024x64xf16>, tensor<8x1x64x1024xf16>, tensor<1xf32> -> tensor<8x4x1024x64xf16>
  // CHECK: [[RESHAPE_BACK:%.+]] = IE.Reshape([[ATTENTION]]) {shape_value = [1, 32, 1024, 64]} : tensor<8x4x1024x64xf16> -> tensor<1x32x1024x64xf16>
  // CHECK: return [[RESHAPE_BACK]]
}

// -----

// CHECK-LABEL: @Fuse_Attention_From5D_LeadingSingletonGroupDim
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x14x1x130x64xf16>, [[ARG1:%.+]]: tensor<1x14x1x130x64xf16>, [[ARG2:%.+]]: tensor<1x14x1x130x64xf16>)
func.func @Fuse_Attention_From5D_LeadingSingletonGroupDim(%arg0: tensor<1x14x1x130x64xf16>, %arg1: tensor<1x14x1x130x64xf16>, %arg2: tensor<1x14x1x130x64xf16>) -> tensor<1x14x1x130x64xf16> {
  %cst = const.Declare tensor<1xf16> = dense<1.250000e-01> : tensor<1xf16>
  %0 = IE.SDPA(%arg0, %arg1, %arg2, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x14x1x130x64xf16>, tensor<1x14x1x130x64xf16>, tensor<1x14x1x130x64xf16>, tensor<1xf16> -> tensor<1x14x1x130x64xf16>
  return %0 : tensor<1x14x1x130x64xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare
  // CHECK-DAG: IE.AffineReshape([[ARG0]]) {{.*}}shape_value = [1, 14, 130, 64]{{.*}} : tensor<1x14x1x130x64xf16> -> tensor<1x14x130x64xf16>
  // CHECK-DAG: IE.AffineReshape([[ARG1]]) {{.*}}shape_value = [1, 14, 130, 64]{{.*}} : tensor<1x14x1x130x64xf16> -> tensor<1x14x130x64xf16>
  // CHECK: [[ATTENTION:%.+]] = IE.Attention({{%.+}}, {{%.+}}, {{%.+}}, [[SCALE]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x14x130x64xf16>, tensor<1x14x130x64xf16>, tensor<1x14x64x130xf16>, tensor<1xf16> -> tensor<1x14x130x64xf16>
  // CHECK: IE.AffineReshape([[ATTENTION]]) {{.*}}shape_value = [1, 14, 1, 130, 64]{{.*}} : tensor<1x14x130x64xf16> -> tensor<1x14x1x130x64xf16>
  // CHECK-NOT: IE.SDPA
}

// -----

// CHECK-LABEL: @Fuse_Attention_From6D_TwoSingletonDims
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x14x1x130x64xf16>, [[ARG1:%.+]]: tensor<1x1x14x1x130x64xf16>, [[ARG2:%.+]]: tensor<1x1x14x1x130x64xf16>)
func.func @Fuse_Attention_From6D_TwoSingletonDims(%arg0: tensor<1x1x14x1x130x64xf16>, %arg1: tensor<1x1x14x1x130x64xf16>, %arg2: tensor<1x1x14x1x130x64xf16>) -> tensor<1x1x14x1x130x64xf16> {
  %cst = const.Declare tensor<1xf16> = dense<1.250000e-01> : tensor<1xf16>
  %0 = IE.SDPA(%arg0, %arg1, %arg2, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x1x14x1x130x64xf16>, tensor<1x1x14x1x130x64xf16>, tensor<1x1x14x1x130x64xf16>, tensor<1xf16> -> tensor<1x1x14x1x130x64xf16>
  return %0 : tensor<1x1x14x1x130x64xf16>

  // CHECK: IE.Attention({{.*}}) {{.*}} : tensor<1x14x130x64xf16>, tensor<1x14x130x64xf16>, tensor<1x14x64x130xf16>, tensor<1xf16> -> tensor<1x14x130x64xf16>
  // CHECK-NOT: IE.SDPA
}

// -----

// CHECK-LABEL: @NoFuse_Attention_From5D_NonSqueezableDims
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<2x14x4x130x64xf16>, [[ARG1:%.+]]: tensor<2x14x4x130x64xf16>, [[ARG2:%.+]]: tensor<2x14x4x130x64xf16>)
func.func @NoFuse_Attention_From5D_NonSqueezableDims(%arg0: tensor<2x14x4x130x64xf16>, %arg1: tensor<2x14x4x130x64xf16>, %arg2: tensor<2x14x4x130x64xf16>) -> tensor<2x14x4x130x64xf16> {
  %cst = const.Declare tensor<1xf16> = dense<1.250000e-01> : tensor<1xf16>
  %0 = IE.SDPA(%arg0, %arg1, %arg2, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<2x14x4x130x64xf16>, tensor<2x14x4x130x64xf16>, tensor<2x14x4x130x64xf16>, tensor<1xf16> -> tensor<2x14x4x130x64xf16>
  return %0 : tensor<2x14x4x130x64xf16>

  // CHECK-NOT: IE.Attention
  // CHECK: IE.SDPA
}
