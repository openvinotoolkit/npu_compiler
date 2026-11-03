//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-attention="fuse-only-attention-dma-patterns=true" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000

// Aligned sequence, scale on the scores: decomposition would move it onto Q (512x96 vs 512x512), which is shared
// with another consumer, so it no longer folds. Kept.

// CHECK-LABEL: @KeepAlignedNonDMAAttentionPattern
// CHECK-SAME:  ([[ARG_Q:%.+]]: tensor<1x12x512x96xf16>, [[ARG_K:%.+]]: tensor<1x12x512x96xf16>, [[ARG_V:%.+]]: tensor<1x12x512x64xf16>, [[ARG_MASK:%.+]]: tensor<1x1x1x512xf16>, [[ARG_G:%.+]]: tensor<1x12x8x96xf16>)
func.func @KeepAlignedNonDMAAttentionPattern(
        %arg0: tensor<1x12x512x96xf16>,
        %arg1: tensor<1x12x512x96xf16>,
        %arg2: tensor<1x12x512x64xf16>,
        %arg3: tensor<1x1x1x512xf16>,
        %arg4: tensor<1x12x8x96xf16>) -> (tensor<1x12x512x64xf16>, tensor<1x12x512x8xf16>) {
  %scale = const.Declare tensor<1x1x1x1xf16> = dense<1.020000e-01> : tensor<1x1x1x1xf16>
  %qk = IE.MatMul(%arg0, %arg1) {transpose_b} :
      tensor<1x12x512x96xf16>, tensor<1x12x512x96xf16> -> tensor<1x12x512x512xf16>
  %scaled = IE.Multiply(%qk, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x12x512x512xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x512x512xf16>
  %masked = IE.Add(%scaled, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x12x512x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x12x512x512xf16>
  %attn = IE.SoftMax(%masked) {axisInd = 3 : i64} :
      tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
  %out = IE.MatMul(%attn, %arg2) :
      tensor<1x12x512x512xf16>, tensor<1x12x512x64xf16> -> tensor<1x12x512x64xf16>
  %gate = IE.MatMul(%arg0, %arg4) {transpose_b} :
      tensor<1x12x512x96xf16>, tensor<1x12x8x96xf16> -> tensor<1x12x512x8xf16>
  return %out, %gate : tensor<1x12x512x64xf16>, tensor<1x12x512x8xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16>
  // CHECK:     [[QK:%.+]] = IE.MatMul([[ARG_Q]], [[ARG_K]]) {transpose_b}
  // CHECK:     [[SCALED:%.+]] = IE.Multiply([[QK]], [[SCALE]])
  // CHECK:     [[MASKED:%.+]] = IE.Add([[SCALED]], [[ARG_MASK]])
  // CHECK:     [[SM:%.+]] = IE.SoftMax([[MASKED]]) {axisInd = 3 : i64}
  // CHECK:     [[VT:%.+]] = IE.Transpose([[ARG_V]])
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[SM]], [[VT]]) {transpose_b}
  // CHECK:     [[GATE:%.+]] = IE.MatMul([[ARG_Q]], [[ARG_G]]) {transpose_b}
  // CHECK:     return [[OUT]], [[GATE]]
  // CHECK-NOT: IE.Attention
}

// -----

// Misaligned sequence: decomposition keeps the scale on the scores and rebuilds an equivalent subgraph. Fused.

// CHECK-LABEL: @FuseMisalignedNonDMAAttentionPattern
func.func @FuseMisalignedNonDMAAttentionPattern(
        %arg0: tensor<1x12x77x96xf16>,
        %arg1: tensor<1x12x77x96xf16>,
        %arg2: tensor<1x12x77x64xf16>,
        %arg3: tensor<1x1x1x77xf16>) -> tensor<1x12x77x64xf16> {
  %scale = const.Declare tensor<1x1x1x1xf16> = dense<1.020000e-01> : tensor<1x1x1x1xf16>
  %qk = IE.MatMul(%arg0, %arg1) {transpose_b} :
      tensor<1x12x77x96xf16>, tensor<1x12x77x96xf16> -> tensor<1x12x77x77xf16>
  %scaled = IE.Multiply(%qk, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x12x77x77xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x77x77xf16>
  %masked = IE.Add(%scaled, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x12x77x77xf16>, tensor<1x1x1x77xf16> -> tensor<1x12x77x77xf16>
  %attn = IE.SoftMax(%masked) {axisInd = 3 : i64} :
      tensor<1x12x77x77xf16> -> tensor<1x12x77x77xf16>
  %out = IE.MatMul(%attn, %arg2) :
      tensor<1x12x77x77xf16>, tensor<1x12x77x64xf16> -> tensor<1x12x77x64xf16>
  return %out : tensor<1x12x77x64xf16>

  // CHECK:     IE.Attention
  // CHECK-NOT: IE.SoftMax
}

// -----

// Batch folded into channels: decomposition would restore the batch and unroll per batch. Kept.

// CHECK-LABEL: @KeepBatchFoldedNonDMAAttentionPattern
func.func @KeepBatchFoldedNonDMAAttentionPattern(
        %arg0: tensor<2x8x77x64xf16>,
        %arg1: tensor<2x8x77x64xf16>,
        %arg2: tensor<2x8x77x64xf16>) -> tensor<2x8x77x64xf16> {
  %scale = const.Declare tensor<1x1x1x1xf16> = dense<1.250000e-01> : tensor<1x1x1x1xf16>
  %q = IE.Reshape(%arg0) {shape_value = [1, 16, 77, 64]} : tensor<2x8x77x64xf16> -> tensor<1x16x77x64xf16>
  %k = IE.Reshape(%arg1) {shape_value = [1, 16, 77, 64]} : tensor<2x8x77x64xf16> -> tensor<1x16x77x64xf16>
  %v = IE.Reshape(%arg2) {shape_value = [1, 16, 77, 64]} : tensor<2x8x77x64xf16> -> tensor<1x16x77x64xf16>
  %qk = IE.MatMul(%q, %k) {transpose_b} :
      tensor<1x16x77x64xf16>, tensor<1x16x77x64xf16> -> tensor<1x16x77x77xf16>
  %scaled = IE.Multiply(%qk, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x16x77x77xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x77x77xf16>
  %attn = IE.SoftMax(%scaled) {axisInd = 3 : i64} :
      tensor<1x16x77x77xf16> -> tensor<1x16x77x77xf16>
  %out = IE.MatMul(%attn, %v) :
      tensor<1x16x77x77xf16>, tensor<1x16x77x64xf16> -> tensor<1x16x77x64xf16>
  %res = IE.Reshape(%out) {shape_value = [2, 8, 77, 64]} : tensor<1x16x77x64xf16> -> tensor<2x8x77x64xf16>
  return %res : tensor<2x8x77x64xf16>

  // CHECK:     IE.SoftMax
  // CHECK-NOT: IE.Attention
}

// -----

// Cross-attention (L != S): decomposition would split it into extra ops. Kept.

// CHECK-LABEL: @KeepCrossAttentionNonDMAPattern
func.func @KeepCrossAttentionNonDMAPattern(
        %arg0: tensor<1x6x4x64xf16>,
        %arg1: tensor<1x6x1500x64xf16>,
        %arg2: tensor<1x6x1500x64xf16>) -> tensor<1x6x4x64xf16> {
  %scale = const.Declare tensor<1x1x1x1xf16> = dense<1.250000e-01> : tensor<1x1x1x1xf16>
  %qk = IE.MatMul(%arg0, %arg1) {transpose_b} :
      tensor<1x6x4x64xf16>, tensor<1x6x1500x64xf16> -> tensor<1x6x4x1500xf16>
  %scaled = IE.Multiply(%qk, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x6x4x1500xf16>, tensor<1x1x1x1xf16> -> tensor<1x6x4x1500xf16>
  %attn = IE.SoftMax(%scaled) {axisInd = 3 : i64} :
      tensor<1x6x4x1500xf16> -> tensor<1x6x4x1500xf16>
  %out = IE.MatMul(%attn, %arg2) :
      tensor<1x6x4x1500xf16>, tensor<1x6x1500x64xf16> -> tensor<1x6x4x64xf16>
  return %out : tensor<1x6x4x64xf16>

  // CHECK:     IE.SoftMax
  // CHECK-NOT: IE.Attention
}

// -----

// Masked single token decode, scale on K: rebuilt as is with the scale on Q, the smallest operand. Fused.

// CHECK-LABEL: @FuseDecodeNonDMAAttentionPattern
func.func @FuseDecodeNonDMAAttentionPattern(
        %arg0: tensor<1x32x1x96xf16>,
        %arg1: tensor<1x32x1024x96xf16>,
        %arg2: tensor<1x32x96x1024xf16>,
        %arg3: tensor<1x1x1x1024xf16>) -> tensor<1x32x1x96xf16> {
  %scale = const.Declare tensor<1x1x1x1xf16> = dense<1.020000e-01> : tensor<1x1x1x1xf16>
  %k = IE.Multiply(%arg1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x32x1024x96xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1024x96xf16>
  %qk = IE.MatMul(%arg0, %k) {transpose_b} :
      tensor<1x32x1x96xf16>, tensor<1x32x1024x96xf16> -> tensor<1x32x1x1024xf16>
  %masked = IE.Add(%qk, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x32x1x1024xf16>, tensor<1x1x1x1024xf16> -> tensor<1x32x1x1024xf16>
  %attn = IE.SoftMax(%masked) {axisInd = 3 : i64} :
      tensor<1x32x1x1024xf16> -> tensor<1x32x1x1024xf16>
  %out = IE.MatMul(%attn, %arg2) {transpose_b} :
      tensor<1x32x1x1024xf16>, tensor<1x32x96x1024xf16> -> tensor<1x32x1x96xf16>
  return %out : tensor<1x32x1x96xf16>

  // CHECK:     IE.Attention
  // CHECK-NOT: IE.SoftMax
}

// -----

// Unmasked single token decode (encoder-decoder): decomposition would split it into extra ops. Kept.

// CHECK-LABEL: @KeepUnmaskedDecodeNonDMAAttentionPattern
func.func @KeepUnmaskedDecodeNonDMAAttentionPattern(
        %arg0: tensor<1x6x1x64xf16>,
        %arg1: tensor<1x6x448x64xf16>,
        %arg2: tensor<1x6x448x64xf16>) -> tensor<1x6x1x64xf16> {
  %scale = const.Declare tensor<1x1x1x1xf16> = dense<1.250000e-01> : tensor<1x1x1x1xf16>
  %q = IE.Multiply(%arg0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x6x1x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x6x1x64xf16>
  %qk = IE.MatMul(%q, %arg1) {transpose_b} :
      tensor<1x6x1x64xf16>, tensor<1x6x448x64xf16> -> tensor<1x6x1x448xf16>
  %attn = IE.SoftMax(%qk) {axisInd = 3 : i64} :
      tensor<1x6x1x448xf16> -> tensor<1x6x1x448xf16>
  %out = IE.MatMul(%attn, %arg2) :
      tensor<1x6x1x448xf16>, tensor<1x6x448x64xf16> -> tensor<1x6x1x64xf16>
  return %out : tensor<1x6x1x64xf16>

  // CHECK:     IE.SoftMax
  // CHECK-NOT: IE.Attention
}

// -----

// Aligned sequence, scale on Q: already the cheapest operand, same ops with shorter live ranges. Fused.

// CHECK-LABEL: @FuseAlignedQueryScaledNonDMAAttentionPattern
func.func @FuseAlignedQueryScaledNonDMAAttentionPattern(
        %arg0: tensor<1x12x512x64xf16>,
        %arg1: tensor<1x12x512x64xf16>,
        %arg2: tensor<1x12x64x512xf16>) -> tensor<1x12x512x64xf16> {
  %scale = const.Declare tensor<1x1x1x1xf16> = dense<1.250000e-01> : tensor<1x1x1x1xf16>
  %q = IE.Multiply(%arg0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x12x512x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x512x64xf16>
  %qk = IE.MatMul(%q, %arg1) {transpose_b} :
      tensor<1x12x512x64xf16>, tensor<1x12x512x64xf16> -> tensor<1x12x512x512xf16>
  %attn = IE.SoftMax(%qk) {axisInd = 3 : i64} :
      tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
  %out = IE.MatMul(%attn, %arg2) {transpose_b} :
      tensor<1x12x512x512xf16>, tensor<1x12x64x512xf16> -> tensor<1x12x512x64xf16>
  return %out : tensor<1x12x512x64xf16>

  // CHECK:     IE.Attention
  // CHECK-NOT: IE.SoftMax
}

// -----

// Aligned sequence, masked, scale on K: moved onto Q at the same cost. Fused.

// CHECK-LABEL: @FuseAlignedKeyScaledNonDMAAttentionPattern
func.func @FuseAlignedKeyScaledNonDMAAttentionPattern(
        %arg0: tensor<1x12x512x64xf16>,
        %arg1: tensor<1x12x512x64xf16>,
        %arg2: tensor<1x12x64x512xf16>,
        %arg3: tensor<1x1x1x512xf16>) -> tensor<1x12x512x64xf16> {
  %scale = const.Declare tensor<1x1x1x1xf16> = dense<1.250000e-01> : tensor<1x1x1x1xf16>
  %k = IE.Multiply(%arg1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x12x512x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x512x64xf16>
  %qk = IE.MatMul(%arg0, %k) {transpose_b} :
      tensor<1x12x512x64xf16>, tensor<1x12x512x64xf16> -> tensor<1x12x512x512xf16>
  %masked = IE.Add(%qk, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x12x512x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x12x512x512xf16>
  %attn = IE.SoftMax(%masked) {axisInd = 3 : i64} :
      tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
  %out = IE.MatMul(%attn, %arg2) {transpose_b} :
      tensor<1x12x512x512xf16>, tensor<1x12x64x512xf16> -> tensor<1x12x512x64xf16>
  return %out : tensor<1x12x512x64xf16>

  // CHECK:     IE.Attention
  // CHECK-NOT: IE.SoftMax
}

// -----

// Aligned sequence, scale on the scores, Q only feeds the attention: decomposition moves the scale onto Q, where it
// folds into the Q producer. Fused.

// CHECK-LABEL: @FuseAlignedScoresScaledUnsharedQueryNonDMAAttentionPattern
func.func @FuseAlignedScoresScaledUnsharedQueryNonDMAAttentionPattern(
        %arg0: tensor<1x8x256x128xf16>,
        %arg1: tensor<1x8x256x128xf16>,
        %arg2: tensor<1x8x128x256xf16>,
        %arg3: tensor<1x1x1x256xf16>) -> tensor<1x8x256x128xf16> {
  %scale = const.Declare tensor<1x1x1x1xf16> = dense<8.837890e-02> : tensor<1x1x1x1xf16>
  %qk = IE.MatMul(%arg0, %arg1) {transpose_b} :
      tensor<1x8x256x128xf16>, tensor<1x8x256x128xf16> -> tensor<1x8x256x256xf16>
  %scaled = IE.Multiply(%qk, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x8x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x8x256x256xf16>
  %masked = IE.Add(%scaled, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x8x256x256xf16>, tensor<1x1x1x256xf16> -> tensor<1x8x256x256xf16>
  %attn = IE.SoftMax(%masked) {axisInd = 3 : i64} :
      tensor<1x8x256x256xf16> -> tensor<1x8x256x256xf16>
  %out = IE.MatMul(%attn, %arg2) {transpose_b} :
      tensor<1x8x256x256xf16>, tensor<1x8x128x256xf16> -> tensor<1x8x256x128xf16>
  return %out : tensor<1x8x256x128xf16>

  // CHECK:     IE.Attention
  // CHECK-NOT: IE.SoftMax
}
