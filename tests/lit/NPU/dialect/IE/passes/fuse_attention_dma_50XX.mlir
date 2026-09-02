//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-attention --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

// CHECK-LABEL: @Fuse_AttentionDMA_LegalConfig
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x12x3600x16xf16>, [[ARG1:%.+]]: tensor<1x12x3600x16xf16>, [[ARG2:%.+]]: tensor<1x12x16x3600xf16>)
func.func @Fuse_AttentionDMA_LegalConfig(%arg0: tensor<1x12x3600x16xf16>, %arg1: tensor<1x12x3600x16xf16>, %arg2: tensor<1x12x16x3600xf16>) -> tensor<1x12x3600x16xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x12x3600x16xf16>, tensor<1x12x3600x16xf16> -> tensor<1x12x3600x3600xf16>
  %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x12x3600x3600xf16> -> tensor<1x12x3600x3600xf16>
  %2 = IE.MatMul(%1, %arg2) {transpose_b} : tensor<1x12x3600x3600xf16>, tensor<1x12x16x3600xf16> -> tensor<1x12x3600x16xf16>
  return %2 : tensor<1x12x3600x16xf16>

  // CHECK-NOT: IE.Attention(
  // CHECK: IE.AttentionDMA([[ARG0]], [[ARG1]], [[ARG2]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0>}
  // CHECK-SAME: tensor<1x12x3600x16xf16>, tensor<1x12x3600x16xf16>, tensor<1x12x16x3600xf16> -> tensor<1x12x3600x16xf16>
  // CHECK-NOT: IE.SDPA
}

// -----

// The mask encodes seq_len_k as a runtime triangle (-inf on right):
//   seqLenK -> Convert -> Add(c) -> AffineReshape -> Add(c) -> Add(c) -> Greater(c) -> Select.
// FuseAttention detects the left-aligned pattern (IE.Greater) and extracts seqLenK,
// wiring it as the last AttentionDMA operand (mask is set to null).

// CHECK-LABEL: @Fuse_AttentionDMA_SeqLenK_LeftAligned_Prefill
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x40x1024x128xf16>, [[ARG1:%.+]]: tensor<1x40x8192x128xf16>, [[ARG2:%.+]]: tensor<1x40x128x8192xf16>, [[SEQLENK:%.+]]: tensor<1x1xsi32>)
func.func @Fuse_AttentionDMA_SeqLenK_LeftAligned_Prefill(%arg0: tensor<1x40x1024x128xf16>, %arg1: tensor<1x40x8192x128xf16>, %arg2: tensor<1x40x128x8192xf16>, %seqLenK: tensor<1x1xsi32>) -> tensor<1x40x1024x128xf16> {
  %cst0 = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
  %cstA = const.Declare tensor<1x1x1024x1xsi64> = dense<1> : tensor<1x1x1024x1xsi64>
  %cstB = const.Declare tensor<1x1x1x1xsi64> = dense<2> : tensor<1x1x1x1xsi64>
  %cstC = const.Declare tensor<1x1x1x1xsi64> = dense<3> : tensor<1x1x1x1xsi64>
  %cstT = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
  %cstF = const.Declare tensor<1x1x1x1xf16> = dense<-6.550400e+04> : tensor<1x1x1x1xf16>

  %conv = IE.Convert(%seqLenK) {dstElemType = si64} : tensor<1x1xsi32> -> tensor<1x1xsi64>
  %add0 = IE.Add(%conv, %cst0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xsi64>, tensor<1xsi64> -> tensor<1x1xsi64>
  %ar = IE.AffineReshape(%add0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1]} : tensor<1x1xsi64> -> tensor<1x1x1x1xsi64>
  %add1 = IE.Add(%ar, %cstA) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xsi64>, tensor<1x1x1024x1xsi64> -> tensor<1x1x1024x1xsi64>
  %add2 = IE.Add(%add1, %cstB) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1024x1xsi64>, tensor<1x1x1x1xsi64> -> tensor<1x1x1024x1xsi64>
  %greater = IE.Greater(%add2, %cstC) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1024x1xsi64>, tensor<1x1x1x1xsi64> -> tensor<1x1x1024x1xi8>
  %mask = IE.Select(%greater, %cstT, %cstF) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1024x1xi8>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1024x1xf16>

  %att = IE.Attention(%arg0, %arg1, %arg2, %mask) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0>} : tensor<1x40x1024x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x128x8192xf16>, tensor<1x1x1024x1xf16> -> tensor<1x40x1024x128xf16>
  return %att : tensor<1x40x1024x128xf16>

  // CHECK-NOT: IE.Greater
  // CHECK-NOT: IE.Select
  // CHECK-NOT: IE.Attention(
  // CHECK: IE.AttentionDMA([[ARG0]], [[ARG1]], [[ARG2]], [[SEQLENK]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 1>}
  // CHECK-SAME: tensor<1x40x1024x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x128x8192xf16>, tensor<1x1xsi32> -> tensor<1x40x1024x128xf16>
  // CHECK-NOT: IE.SDPA
}

// -----

// CHECK-LABEL: @Fuse_AttentionDMA_SeqLenK_LeftAligned_Kvcache
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x40x1x128xf16>, [[ARG1:%.+]]: tensor<1x40x8192x128xf16>, [[ARG2:%.+]]: tensor<1x40x128x8192xf16>, [[SEQLENK:%.+]]: tensor<1x1xsi32>)
func.func @Fuse_AttentionDMA_SeqLenK_LeftAligned_Kvcache(%arg0: tensor<1x40x1x128xf16>, %arg1: tensor<1x40x8192x128xf16>, %arg2: tensor<1x40x128x8192xf16>, %seqLenK: tensor<1x1xsi32>) -> tensor<1x40x1x128xf16> {
  %cst0 = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
  %cstA = const.Declare tensor<1x1x1x1xsi64> = dense<1> : tensor<1x1x1x1xsi64>
  %cstB = const.Declare tensor<1x1x1x1xsi64> = dense<2> : tensor<1x1x1x1xsi64>
  %cstC = const.Declare tensor<1x1x1x1xsi64> = dense<3> : tensor<1x1x1x1xsi64>
  %cstT = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
  %cstF = const.Declare tensor<1x1x1x1xf16> = dense<-6.550400e+04> : tensor<1x1x1x1xf16>

  %conv = IE.Convert(%seqLenK) {dstElemType = si64} : tensor<1x1xsi32> -> tensor<1x1xsi64>
  %add0 = IE.Add(%conv, %cst0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xsi64>, tensor<1xsi64> -> tensor<1x1xsi64>
  %ar = IE.AffineReshape(%add0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1]} : tensor<1x1xsi64> -> tensor<1x1x1x1xsi64>
  %add1 = IE.Add(%ar, %cstA) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xsi64>, tensor<1x1x1x1xsi64> -> tensor<1x1x1x1xsi64>
  %add2 = IE.Add(%add1, %cstB) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xsi64>, tensor<1x1x1x1xsi64> -> tensor<1x1x1x1xsi64>
  %greater = IE.Greater(%add2, %cstC) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xsi64>, tensor<1x1x1x1xsi64> -> tensor<1x1x1x1xi8>
  %mask = IE.Select(%greater, %cstT, %cstF) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xi8>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

  %att = IE.Attention(%arg0, %arg1, %arg2, %mask) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0>} : tensor<1x40x1x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x128x8192xf16>, tensor<1x1x1x1xf16> -> tensor<1x40x1x128xf16>
  return %att : tensor<1x40x1x128xf16>

  // CHECK-NOT: IE.Greater
  // CHECK-NOT: IE.Select
  // CHECK-NOT: IE.Attention(
  // CHECK: IE.AttentionDMA([[ARG0]], [[ARG1]], [[ARG2]], [[SEQLENK]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 1>}
  // CHECK-SAME: tensor<1x40x1x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x128x8192xf16>, tensor<1x1xsi32> -> tensor<1x40x1x128xf16>
  // CHECK-NOT: IE.SDPA
}

// -----

// The mask encodes seq_len_k as a baked causal triangle (-inf on the left):
//   seqLenK -> Convert -> Add -> AffineReshape -> Subtract -> Broadcast -> GreaterEqual -> Select.
// FuseAttention detects the right-aligned pattern (IE.GreaterEqual) but does NOT extract seqLenK

// CHECK-LABEL: @Fuse_AttentionDMA_SeqLenK_RightAligned
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x40x1024x128xf16>, [[ARG1:%.+]]: tensor<1x40x8192x128xf16>, [[ARG2:%.+]]: tensor<1x40x128x8192xf16>, [[SEQLENK:%.+]]: tensor<1x1xsi32>)
func.func @Fuse_AttentionDMA_SeqLenK_RightAligned(%arg0: tensor<1x40x1024x128xf16>, %arg1: tensor<1x40x8192x128xf16>, %arg2: tensor<1x40x128x8192xf16>, %seqLenK: tensor<1x1xsi32>) -> tensor<1x40x1024x128xf16> {
  %cst0 = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
  %cstSub = const.Declare tensor<1x1x1x1xsi64> = dense<1> : tensor<1x1x1x1xsi64>
  %cstShape = const.Declare tensor<4xsi64> = dense<[1, 1, 1024, 8192]> : tensor<4xsi64>
  %cstGE = const.Declare tensor<1x1x1x1xsi64> = dense<3> : tensor<1x1x1x1xsi64>
  %cstT = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
  %cstF = const.Declare tensor<1x1x1x1xf16> = dense<-6.550400e+04> : tensor<1x1x1x1xf16>

  %conv = IE.Convert(%seqLenK) {dstElemType = si64} : tensor<1x1xsi32> -> tensor<1x1xsi64>
  %add0 = IE.Add(%conv, %cst0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xsi64>, tensor<1xsi64> -> tensor<1x1xsi64>
  %ar = IE.AffineReshape(%add0) {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1x1xsi64> -> tensor<1x1x1x1xsi64>
  %sub = IE.Subtract(%ar, %cstSub) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xsi64>, tensor<1x1x1x1xsi64> -> tensor<1x1x1x1xsi64>
  %bcast = IE.Broadcast(%sub, %cstShape) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x1x1x1xsi64>, tensor<4xsi64> -> tensor<1x1x1024x8192xsi64>
  %ge = IE.GreaterEqual(%bcast, %cstGE) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1024x8192xsi64>, tensor<1x1x1x1xsi64> -> tensor<1x1x1024x8192xsi64>
  %mask = IE.Select(%ge, %cstT, %cstF) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1024x8192xsi64>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1024x8192xf16>

  %att = IE.Attention(%arg0, %arg1, %arg2, %mask) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0>} : tensor<1x40x1024x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x128x8192xf16>, tensor<1x1x1024x8192xf16> -> tensor<1x40x1024x128xf16>
  return %att : tensor<1x40x1024x128xf16>

  // CHECK: [[GE:%.+]] = IE.GreaterEqual
  // CHECK: [[MASK:%.+]] = IE.Select([[GE]]
  // CHECK: IE.Attention(
  // CHECK-NOT: IE.AttentionDMA([[ARG0]], [[ARG1]], [[ARG2]], [[MASK]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 0>}
  // CHECK-NOT: IE.SDPA
}
