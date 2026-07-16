//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --attention-processing="convert-to-attention=true" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// -----

// Prefill mode (tSL=1024): 32 Q heads, 8 KV heads (group_size=4), 3 KV blocks.
// sSL = 3 x 1024 = 3072 >= 2048 (prefill threshold).
//
// CHECK-LABEL: @GQAPrefillAttentionPipeline
// CHECK-SAME:  ([[ARG_Q:%.+]]: tensor<1x32x1024x128xf16>, [[ARG_K0:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K1:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K2:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_V0:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V1:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V2:%.+]]: tensor<1x8x128x1024xf16>)
func.func @GQAPrefillAttentionPipeline(
        %arg0: tensor<1x32x1024x128xf16>,
        %arg1: tensor<1x8x1024x128xf16>,
        %arg2: tensor<1x8x1024x128xf16>,
        %arg3: tensor<1x8x1024x128xf16>,
        %arg4: tensor<1x8x128x1024xf16>,
        %arg5: tensor<1x8x128x1024xf16>,
        %arg6: tensor<1x8x128x1024xf16>) -> tensor<1x32x1024x128xf16> {
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

  // Pre-FuseAttention attention computation: MatMul(Q, K^T) -> SoftMax -> MatMul(attn, V^T)
  %qk  = IE.MatMul(%arg0, %k_expanded) {transpose_b} :
      tensor<1x32x1024x128xf16>, tensor<1x32x3072x128xf16> -> tensor<1x32x1024x3072xf16>
  %attn = IE.SoftMax(%qk) {axisInd = 3 : i64} :
      tensor<1x32x1024x3072xf16> -> tensor<1x32x1024x3072xf16>
  %out = IE.MatMul(%attn, %v_expanded) {transpose_b} :
      tensor<1x32x1024x3072xf16>, tensor<1x32x128x3072xf16> -> tensor<1x32x1024x128xf16>
  return %out : tensor<1x32x1024x128xf16>

  // FuseAttention+DecomposeAttention round-trip: net result is MatMul+SoftMax+MatMul.
  // CHECK:     [[K_CAT:%.+]] = IE.Concat([[ARG_K0]], [[ARG_K1]], [[ARG_K2]])
  // CHECK:     [[V_CAT:%.+]] = IE.Concat([[ARG_V0]], [[ARG_V1]], [[ARG_V2]])
  // CHECK:     [[CST_K:%.+]] = const.Declare tensor<5xsi32> = dense<[1, 8, 4, 3072, 128]>
  // CHECK:     [[K_BC:%.+]] = IE.Broadcast
  // CHECK:     [[K_EXP:%.+]] = IE.AffineReshape([[K_BC]])
  // CHECK:     [[CST_V:%.+]] = const.Declare tensor<5xsi32> = dense<[1, 8, 4, 128, 3072]>
  // CHECK:     [[QK:%.+]] = IE.MatMul([[ARG_Q]], [[K_EXP]])
  // CHECK:     [[ATTN:%.+]] = IE.SoftMax([[QK]])
  // CHECK:     [[V_BC:%.+]] = IE.Broadcast
  // CHECK:     [[V_EXP:%.+]] = IE.AffineReshape([[V_BC]])
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[ATTN]], [[V_EXP]])
  // CHECK-NOT: IE.Attention
  // CHECK-NOT: IE.FlashSDPA
}

// -----

// Decode mode (tSL=1): 32 Q heads, 8 KV heads (group_size=4), 8 full blocks of
// 1024 tokens + 127-token partial block + 1 present token
// (sSL = 8x1024+127+1 = 8320 >= 8192, decode threshold).
//
// CHECK-LABEL: @GQADecodeAttentionPipeline
// CHECK-SAME:  ([[ARG_Q:%.+]]: tensor<1x32x1x128xf16>, [[ARG_K0:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K1:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K2:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K3:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K4:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K5:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K6:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K7:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_KT:%.+]]: tensor<1x8x127x128xf16>, [[ARG_KP:%.+]]: tensor<1x8x1x128xf16>, [[ARG_V0:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V1:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V2:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V3:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V4:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V5:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V6:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V7:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_VT:%.+]]: tensor<1x8x128x127xf16>, [[ARG_VP:%.+]]: tensor<1x8x128x1xf16>)
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
        %arg20: tensor<1x8x128x1xf16>) -> tensor<1x32x1x128xf16> {
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

  // Pre-FuseAttention attention computation
  %qk  = IE.MatMul(%arg0, %k_expanded) {transpose_b} :
      tensor<1x32x1x128xf16>, tensor<1x32x8320x128xf16> -> tensor<1x32x1x8320xf16>
  %attn = IE.SoftMax(%qk) {axisInd = 3 : i64} :
      tensor<1x32x1x8320xf16> -> tensor<1x32x1x8320xf16>
  %out = IE.MatMul(%attn, %v_expanded) {transpose_b} :
      tensor<1x32x1x8320xf16>, tensor<1x32x128x8320xf16> -> tensor<1x32x1x128xf16>
  return %out : tensor<1x32x1x128xf16>

  // FuseAttention+DecomposeAttention round-trip: net result is MatMul+SoftMax+MatMul.
  // CHECK:     [[K_CAT:%.+]] = IE.Concat([[ARG_K0]], [[ARG_K1]], [[ARG_K2]], [[ARG_K3]], [[ARG_K4]], [[ARG_K5]], [[ARG_K6]], [[ARG_K7]], [[ARG_KT]], [[ARG_KP]])
  // CHECK:     [[V_CAT:%.+]] = IE.Concat([[ARG_V0]], [[ARG_V1]], [[ARG_V2]], [[ARG_V3]], [[ARG_V4]], [[ARG_V5]], [[ARG_V6]], [[ARG_V7]], [[ARG_VT]], [[ARG_VP]])
  // CHECK:     [[CST_K:%.+]] = const.Declare tensor<5xsi32> = dense<[1, 8, 4, 8320, 128]>
  // CHECK:     [[K_BC:%.+]] = IE.Broadcast
  // CHECK:     [[K_EXP:%.+]] = IE.AffineReshape([[K_BC]])
  // CHECK:     [[CST_V:%.+]] = const.Declare tensor<5xsi32> = dense<[1, 8, 4, 128, 8320]>
  // CHECK:     [[QK:%.+]] = IE.MatMul([[ARG_Q]], [[K_EXP]])
  // CHECK:     [[ATTN:%.+]] = IE.SoftMax([[QK]])
  // CHECK:     [[V_BC:%.+]] = IE.Broadcast
  // CHECK:     [[V_EXP:%.+]] = IE.AffineReshape([[V_BC]])
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[ATTN]], [[V_EXP]])
  // CHECK-NOT: IE.Attention
  // CHECK-NOT: IE.FlashSDPA
}
