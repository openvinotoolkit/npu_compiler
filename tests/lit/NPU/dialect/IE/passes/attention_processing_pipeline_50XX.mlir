//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --attention-processing="convert-to-attention=true" %s | FileCheck %s
// REQUIRES: platform-NPU5010

// -----

// Prefill mode (tSL=1024): 32 Q heads, 8 KV heads (group_size=4), 3 KV blocks, with causal mask.
// sSL = 3 x 1024 = 3072 >= 2048 (prefill threshold).
//
// Input pattern:  MatMul(Q, K^T) -> Add(mask) -> SoftMax -> MatMul(attn, V^T)
// Expected pipeline effect:
//   1. FuseAttention:                  MatMul+Add+SoftMax+MatMul -> IE.Attention (mask captured)
//   2. ConvertBlockCacheAttentionToFlashSDPA: IE.Attention              -> 3x IE.FlashSDPA cascade
//      (mask sliced per tile along seq dim)

// CHECK-LABEL: @GQAPrefillAttentionPipeline
// CHECK-SAME:  ([[ARG_Q:%.+]]: tensor<1x32x1024x128xf16>,
// CHECK-SAME:  [[ARG_K0:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K1:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K2:%.+]]: tensor<1x8x1024x128xf16>,
// CHECK-SAME:  [[ARG_V0:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V1:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V2:%.+]]: tensor<1x8x128x1024xf16>,
// CHECK-SAME:  [[ARG_MASK:%.+]]: tensor<1x1x1024x3072xf16>)
func.func @GQAPrefillAttentionPipeline(
        %arg0: tensor<1x32x1024x128xf16>,
        %arg1: tensor<1x8x1024x128xf16>,
        %arg2: tensor<1x8x1024x128xf16>,
        %arg3: tensor<1x8x1024x128xf16>,
        %arg4: tensor<1x8x128x1024xf16>,
        %arg5: tensor<1x8x128x1024xf16>,
        %arg6: tensor<1x8x128x1024xf16>,
        %arg7: tensor<1x1x1024x3072xf16>) -> tensor<1x32x1024x128xf16> {
  %cst_k_shape = const.Declare tensor<5xsi64> = dense<[1, 8, 4, 3072, 128]> : tensor<5xsi64>
  %cst_v_shape = const.Declare tensor<5xsi64> = dense<[1, 8, 4, 128, 3072]> : tensor<5xsi64>

  // GQA K expansion: Concat(k0,k1,k2) -> [1,8,3072,128] -> broadcast -> [1,32,3072,128]
  %k_cat = IE.Concat(%arg1, %arg2, %arg3) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 1024, 0], [0, 0, 2048, 0]]} :
      tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>
      -> tensor<1x8x3072x128xf16>
  %k_r1 = IE.AffineReshape(%k_cat) {
      dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 8, 1, 3072, 128]} :
      tensor<1x8x3072x128xf16> -> tensor<1x8x1x3072x128xf16>
  %k_bc = IE.Broadcast(%k_r1, %cst_k_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} :
      tensor<1x8x1x3072x128xf16>, tensor<5xsi64> -> tensor<1x8x4x3072x128xf16>
  %k_expanded = IE.AffineReshape(%k_bc) {
      dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 32, 3072, 128]} :
      tensor<1x8x4x3072x128xf16> -> tensor<1x32x3072x128xf16>

  // GQA V expansion: V tiles in [N,H,E,S] layout, Concat -> broadcast -> [1,32,128,3072]
  %v_cat = IE.Concat(%arg4, %arg5, %arg6) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1024], [0, 0, 0, 2048]]} :
      tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>
      -> tensor<1x8x128x3072xf16>
  %v_r1 = IE.AffineReshape(%v_cat) {
      dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 8, 1, 128, 3072]} :
      tensor<1x8x128x3072xf16> -> tensor<1x8x1x128x3072xf16>
  %v_bc = IE.Broadcast(%v_r1, %cst_v_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} :
      tensor<1x8x1x128x3072xf16>, tensor<5xsi64> -> tensor<1x8x4x128x3072xf16>
  %v_expanded = IE.AffineReshape(%v_bc) {
      dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 32, 128, 3072]} :
      tensor<1x8x4x128x3072xf16> -> tensor<1x32x128x3072xf16>

  // Attention computation: MatMul(Q, K^T) -> Add(mask) -> SoftMax -> MatMul(attn, V^T)
  %qk = IE.MatMul(%arg0, %k_expanded) {transpose_b} :
      tensor<1x32x1024x128xf16>, tensor<1x32x3072x128xf16> -> tensor<1x32x1024x3072xf16>
  %masked_qk = IE.Add(%qk, %arg7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x32x1024x3072xf16>, tensor<1x1x1024x3072xf16> -> tensor<1x32x1024x3072xf16>
  %attn = IE.SoftMax(%masked_qk) {axisInd = 3 : i64} :
      tensor<1x32x1024x3072xf16> -> tensor<1x32x1024x3072xf16>
  %out = IE.MatMul(%attn, %v_expanded) {transpose_b} :
      tensor<1x32x1024x3072xf16>, tensor<1x32x128x3072xf16> -> tensor<1x32x1024x128xf16>
  return %out : tensor<1x32x1024x128xf16>

  // CHECK-DAG: [[CST_SUM:%.+]] = const.Declare tensor<1x32x1024xf32> = dense<0.000000e+00>
  // CHECK-DAG: [[CST_MAX:%.+]] = const.Declare tensor<1x32x1024xf16> = dense<0xFC00>
  // CHECK-DAG: [[CST_OUT:%.+]] = const.Declare tensor<1x32x1024x128xf16> = dense<0.000000e+00>
  // Q is [1,32,1024,128] (batch=1, 32 Q heads); K/V tiles are [1,8,...] (8 KV heads).
  // FlashSDPA handles GQA natively (qH=32 > kvH=8). The FuseAttention GQA fold
  // (unfold_q_gqa composed with the original reshape_q_gqa) is folded away by MLIR,
  // leaving %arg0 passed directly, and the output reshape pair is similarly eliminated.
  // Tile 0: V transposed, mask sliced [0:1024], head FlashSDPA.
  // CHECK:     [[V0_T:%.+]] = IE.Transpose([[ARG_V0]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK:     [[MASK0:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 0] [1, 1, 1024, 1024]
  // CHECK:     [[OUT0:%.+]], [[MAX0:%.+]], [[SUM0:%.+]] = IE.FlashSDPA([[ARG_Q]], [[ARG_K0]], [[V0_T]], [[CST_OUT]], [[CST_MAX]], [[CST_SUM]], [[MASK0]]) {is_head = true, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // Tile 1: mask sliced [1024:2048].
  // CHECK:     [[V1_T:%.+]] = IE.Transpose([[ARG_V1]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK:     [[MASK1:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 1024] [1, 1, 1024, 1024]
  // CHECK:     [[OUT1:%.+]], [[MAX1:%.+]], [[SUM1:%.+]] = IE.FlashSDPA([[ARG_Q]], [[ARG_K1]], [[V1_T]], [[OUT0]], [[MAX0]], [[SUM0]], [[MASK1]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // Tile 2: mask sliced [2048:3072], tail.
  // CHECK:     [[V2_T:%.+]] = IE.Transpose([[ARG_V2]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK:     [[MASK2:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 2048] [1, 1, 1024, 1024]
  // CHECK:     [[OUT2:%.+]], {{%.+}}, {{%.+}} = IE.FlashSDPA([[ARG_Q]], [[ARG_K2]], [[V2_T]], [[OUT1]], [[MAX1]], [[SUM1]], [[MASK2]]) {is_head = false, is_tail = true, source_seq_len_pad_size = 0 : i64}
  // CHECK:     return [[OUT2]]
  // CHECK-NOT: IE.Attention
  // CHECK-NOT: IE.MatMul
  // CHECK-NOT: IE.SoftMax
}

// -----

// Decode mode (tSL=1): 32 Q heads, 8 KV heads (group_size=4), 8 full blocks of
// 1024 tokens + 127-token partial block + 1 present token, with causal mask.
// sSL = 8x1024+127+1 = 8320 >= 8192 (decode threshold), 8320 % 16 == 0.
//
// Input pattern:  MatMul(Q, K^T) -> Add(mask) -> SoftMax -> MatMul(attn, V^T)
// Expected pipeline effect:
//   1. FuseAttention:                  MatMul+Add+SoftMax+MatMul -> IE.Attention (mask captured)
//   2. ConvertBlockCacheAttentionToFlashSDPA: IE.Attention              -> 9x IE.FlashSDPA cascade
//      (tiles [127, 1] greedily merged into one 128-token tile; mask sliced per tile)

// CHECK-LABEL: @GQADecodeAttentionPipeline
// CHECK-SAME:  ([[ARG_Q:%.+]]: tensor<1x32x1x128xf16>,
// CHECK-SAME:  [[ARG_K0:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K1:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K2:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K3:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K4:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K5:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K6:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K7:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_KT:%.+]]: tensor<1x8x127x128xf16>, [[ARG_KP:%.+]]: tensor<1x8x1x128xf16>,
// CHECK-SAME:  [[ARG_V0:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V1:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V2:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V3:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V4:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V5:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V6:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V7:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_VT:%.+]]: tensor<1x8x128x127xf16>, [[ARG_VP:%.+]]: tensor<1x8x128x1xf16>,
// CHECK-SAME:  [[ARG_MASK:%.+]]: tensor<1x1x1x8320xf16>)
func.func @GQADecodeAttentionPipeline(
        %arg0:  tensor<1x32x1x128xf16>,
        %arg1:  tensor<1x8x1024x128xf16>,
        %arg2:  tensor<1x8x1024x128xf16>,
        %arg3:  tensor<1x8x1024x128xf16>,
        %arg4:  tensor<1x8x1024x128xf16>,
        %arg5:  tensor<1x8x1024x128xf16>,
        %arg6:  tensor<1x8x1024x128xf16>,
        %arg7:  tensor<1x8x1024x128xf16>,
        %arg8:  tensor<1x8x1024x128xf16>,
        %arg9:  tensor<1x8x127x128xf16>,
        %arg10: tensor<1x8x1x128xf16>,
        %arg11: tensor<1x8x128x1024xf16>,
        %arg12: tensor<1x8x128x1024xf16>,
        %arg13: tensor<1x8x128x1024xf16>,
        %arg14: tensor<1x8x128x1024xf16>,
        %arg15: tensor<1x8x128x1024xf16>,
        %arg16: tensor<1x8x128x1024xf16>,
        %arg17: tensor<1x8x128x1024xf16>,
        %arg18: tensor<1x8x128x1024xf16>,
        %arg19: tensor<1x8x128x127xf16>,
        %arg20: tensor<1x8x128x1xf16>,
        %arg21: tensor<1x1x1x8320xf16>) -> tensor<1x32x1x128xf16> {
  %cst_k_shape = const.Declare tensor<5xsi64> = dense<[1, 8, 4, 8320, 128]> : tensor<5xsi64>
  %cst_v_shape = const.Declare tensor<5xsi64> = dense<[1, 8, 4, 128, 8320]> : tensor<5xsi64>

  // GQA K expansion
  %k_cat = IE.Concat(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 1024, 0], [0, 0, 2048, 0], [0, 0, 3072, 0],
                        [0, 0, 4096, 0], [0, 0, 5120, 0], [0, 0, 6144, 0], [0, 0, 7168, 0],
                        [0, 0, 8192, 0], [0, 0, 8319, 0]]} :
      tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>,
      tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>,
      tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x127x128xf16>,
      tensor<1x8x1x128xf16>
      -> tensor<1x8x8320x128xf16>
  %k_r1 = IE.AffineReshape(%k_cat) {
      dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 8, 1, 8320, 128]} :
      tensor<1x8x8320x128xf16> -> tensor<1x8x1x8320x128xf16>
  %k_bc = IE.Broadcast(%k_r1, %cst_k_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} :
      tensor<1x8x1x8320x128xf16>, tensor<5xsi64> -> tensor<1x8x4x8320x128xf16>
  %k_expanded = IE.AffineReshape(%k_bc) {
      dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 32, 8320, 128]} :
      tensor<1x8x4x8320x128xf16> -> tensor<1x32x8320x128xf16>

  // GQA V expansion
  %v_cat = IE.Concat(%arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1024], [0, 0, 0, 2048], [0, 0, 0, 3072],
                        [0, 0, 0, 4096], [0, 0, 0, 5120], [0, 0, 0, 6144], [0, 0, 0, 7168],
                        [0, 0, 0, 8192], [0, 0, 0, 8319]]} :
      tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>,
      tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>,
      tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>, tensor<1x8x128x127xf16>,
      tensor<1x8x128x1xf16>
      -> tensor<1x8x128x8320xf16>
  %v_r1 = IE.AffineReshape(%v_cat) {
      dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 8, 1, 128, 8320]} :
      tensor<1x8x128x8320xf16> -> tensor<1x8x1x128x8320xf16>
  %v_bc = IE.Broadcast(%v_r1, %cst_v_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} :
      tensor<1x8x1x128x8320xf16>, tensor<5xsi64> -> tensor<1x8x4x128x8320xf16>
  %v_expanded = IE.AffineReshape(%v_bc) {
      dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 32, 128, 8320]} :
      tensor<1x8x4x128x8320xf16> -> tensor<1x32x128x8320xf16>

  // Attention computation: MatMul(Q, K^T) -> Add(mask) -> SoftMax -> MatMul(attn, V^T)
  %qk = IE.MatMul(%arg0, %k_expanded) {transpose_b} :
      tensor<1x32x1x128xf16>, tensor<1x32x8320x128xf16> -> tensor<1x32x1x8320xf16>
  %masked_qk = IE.Add(%qk, %arg21) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x32x1x8320xf16>, tensor<1x1x1x8320xf16> -> tensor<1x32x1x8320xf16>
  %attn = IE.SoftMax(%masked_qk) {axisInd = 3 : i64} :
      tensor<1x32x1x8320xf16> -> tensor<1x32x1x8320xf16>
  %out = IE.MatMul(%attn, %v_expanded) {transpose_b} :
      tensor<1x32x1x8320xf16>, tensor<1x32x128x8320xf16> -> tensor<1x32x1x128xf16>
  return %out : tensor<1x32x1x128xf16>

  // CHECK-DAG: [[CST_SUM:%.+]] = const.Declare tensor<1x32x1xf32> = dense<0.000000e+00>
  // CHECK-DAG: [[CST_MAX:%.+]] = const.Declare tensor<1x32x1xf16> = dense<0xFC00>
  // CHECK-DAG: [[CST_OUT:%.+]] = const.Declare tensor<1x32x1x128xf16> = dense<0.000000e+00>
  // Q is [1,32,1,128] (batch=1, 32 Q heads); K/V tiles are [1,8,...] (8 KV heads).
  // FlashSDPA handles GQA natively (qH=32 > kvH=8). The FuseAttention GQA fold
  // (unfold_q_gqa composed with the original reshape_q_gqa) is folded away by MLIR.
  // Tiles 0-7: full 1024-token blocks, mask sliced per tile.
  // CHECK:     [[V0_T:%.+]] = IE.Transpose([[ARG_V0]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK:     [[MASK0:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 0] [1, 1, 1, 1024]
  // CHECK:     [[OUT0:%.+]], [[MAX0:%.+]], [[SUM0:%.+]] = IE.FlashSDPA([[ARG_Q]], [[ARG_K0]], [[V0_T]], [[CST_OUT]], [[CST_MAX]], [[CST_SUM]], [[MASK0]]) {is_head = true, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK:     [[V1_T:%.+]] = IE.Transpose([[ARG_V1]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK:     [[MASK1:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 1024] [1, 1, 1, 1024]
  // CHECK:     [[OUT1:%.+]], [[MAX1:%.+]], [[SUM1:%.+]] = IE.FlashSDPA([[ARG_Q]], [[ARG_K1]], [[V1_T]], [[OUT0]], [[MAX0]], [[SUM0]], [[MASK1]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK:     [[V2_T:%.+]] = IE.Transpose([[ARG_V2]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK:     [[MASK2:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 2048] [1, 1, 1, 1024]
  // CHECK:     [[OUT2:%.+]], [[MAX2:%.+]], [[SUM2:%.+]] = IE.FlashSDPA([[ARG_Q]], [[ARG_K2]], [[V2_T]], [[OUT1]], [[MAX1]], [[SUM1]], [[MASK2]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK:     [[V3_T:%.+]] = IE.Transpose([[ARG_V3]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK:     [[MASK3:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 3072] [1, 1, 1, 1024]
  // CHECK:     [[OUT3:%.+]], [[MAX3:%.+]], [[SUM3:%.+]] = IE.FlashSDPA([[ARG_Q]], [[ARG_K3]], [[V3_T]], [[OUT2]], [[MAX2]], [[SUM2]], [[MASK3]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK:     [[V4_T:%.+]] = IE.Transpose([[ARG_V4]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK:     [[MASK4:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 4096] [1, 1, 1, 1024]
  // CHECK:     [[OUT4:%.+]], [[MAX4:%.+]], [[SUM4:%.+]] = IE.FlashSDPA([[ARG_Q]], [[ARG_K4]], [[V4_T]], [[OUT3]], [[MAX3]], [[SUM3]], [[MASK4]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK:     [[V5_T:%.+]] = IE.Transpose([[ARG_V5]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK:     [[MASK5:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 5120] [1, 1, 1, 1024]
  // CHECK:     [[OUT5:%.+]], [[MAX5:%.+]], [[SUM5:%.+]] = IE.FlashSDPA([[ARG_Q]], [[ARG_K5]], [[V5_T]], [[OUT4]], [[MAX4]], [[SUM4]], [[MASK5]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK:     [[V6_T:%.+]] = IE.Transpose([[ARG_V6]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK:     [[MASK6:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 6144] [1, 1, 1, 1024]
  // CHECK:     [[OUT6:%.+]], [[MAX6:%.+]], [[SUM6:%.+]] = IE.FlashSDPA([[ARG_Q]], [[ARG_K6]], [[V6_T]], [[OUT5]], [[MAX5]], [[SUM5]], [[MASK6]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK:     [[V7_T:%.+]] = IE.Transpose([[ARG_V7]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK:     [[MASK7:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 7168] [1, 1, 1, 1024]
  // CHECK:     [[OUT7:%.+]], [[MAX7:%.+]], [[SUM7:%.+]] = IE.FlashSDPA([[ARG_Q]], [[ARG_K7]], [[V7_T]], [[OUT6]], [[MAX6]], [[SUM6]], [[MASK7]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // Tile 8: [127-token tail, 1 present token] greedily merged (127+1=128 is 16-aligned); mask sliced [8192:8320].
  // CHECK:     [[K_TAIL:%.+]] = IE.Concat([[ARG_KT]], [[ARG_KP]]) {{.*}} : tensor<1x8x127x128xf16>, tensor<1x8x1x128xf16> -> tensor<1x8x128x128xf16>
  // CHECK:     [[V_TAIL:%.+]] = IE.Concat([[ARG_VT]], [[ARG_VP]]) {{.*}} : tensor<1x8x128x127xf16>, tensor<1x8x128x1xf16> -> tensor<1x8x128x128xf16>
  // CHECK:     [[VTT:%.+]] = IE.Transpose([[V_TAIL]]) {order_value = #NCWH} : tensor<1x8x128x128xf16> -> tensor<1x8x128x128xf16>
  // CHECK:     [[MASK8:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 8192] [1, 1, 1, 128]
  // CHECK:     [[OUT8:%.+]], {{%.+}}, {{%.+}} = IE.FlashSDPA([[ARG_Q]], [[K_TAIL]], [[VTT]], [[OUT7]], [[MAX7]], [[SUM7]], [[MASK8]]) {is_head = false, is_tail = true, source_seq_len_pad_size = 0 : i64}
  // CHECK:     return [[OUT8]]
  // CHECK-NOT: IE.Attention
  // CHECK-NOT: IE.MatMul
  // CHECK-NOT: IE.SoftMax
}
