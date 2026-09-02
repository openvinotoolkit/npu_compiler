//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-block-cache-attention-to-flash-sdpa --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

// -----

// GQA Prefill, 3 equal 1024-token KV blocks, no scale, no mask.
// Input is the post-FuseAttention state:
//   - Q is GQA head-folded: [1,32,1024,128] -> [8,4,1024,128]  (kv_heads=8, grp=4)
//   - K is Concat([1,8,1024,128] x 3) wrapped by Reshape -> [8,1,3072,128]
//   - V is Concat([1,8,128,1024] x 3) wrapped by Reshape -> [8,1,128,3072]
// sSL=3072 >= 2048 (prefill threshold), 3072 % 16 == 0.
// The pass un-folds Q to [1,32,1024,128], feeds raw tiles to FlashSDPA (GQA handled
// natively by the kernel), and re-folds the output to [8,4,1024,128].

// CHECK-LABEL: @GQAPrefillThreeBlocks
// CHECK-SAME:  ([[ARG_Q:%.+]]: tensor<8x4x1024x128xf16>,
// CHECK-SAME:  [[ARG_K0:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K1:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K2:%.+]]: tensor<1x8x1024x128xf16>,
// CHECK-SAME:  [[ARG_V0:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V1:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V2:%.+]]: tensor<1x8x128x1024xf16>)
func.func @GQAPrefillThreeBlocks(
        %arg0: tensor<8x4x1024x128xf16>,
        %arg1: tensor<1x8x1024x128xf16>,
        %arg2: tensor<1x8x1024x128xf16>,
        %arg3: tensor<1x8x1024x128xf16>,
        %arg4: tensor<1x8x128x1024xf16>,
        %arg5: tensor<1x8x128x1024xf16>,
        %arg6: tensor<1x8x128x1024xf16>) -> tensor<8x4x1024x128xf16> {
  %k_cat = IE.Concat(%arg1, %arg2, %arg3) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 1024, 0], [0, 0, 2048, 0]]} :
      tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>
      -> tensor<1x8x3072x128xf16>
  %k_rs = IE.Reshape(%k_cat) {shape_value = [8, 1, 3072, 128]} :
      tensor<1x8x3072x128xf16> -> tensor<8x1x3072x128xf16>
  %v_cat = IE.Concat(%arg4, %arg5, %arg6) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1024], [0, 0, 0, 2048]]} :
      tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>
      -> tensor<1x8x128x3072xf16>
  %v_rs = IE.Reshape(%v_cat) {shape_value = [8, 1, 128, 3072]} :
      tensor<1x8x128x3072xf16> -> tensor<8x1x128x3072xf16>
  %out = IE.Attention(%arg0, %k_rs, %v_rs) {
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} :
      tensor<8x4x1024x128xf16>, tensor<8x1x3072x128xf16>, tensor<8x1x128x3072xf16>
      -> tensor<8x4x1024x128xf16>
  return %out : tensor<8x4x1024x128xf16>

  // Running-state constants are sized from the un-folded Q shape [1,32,1024,128].
  // CHECK-DAG: [[INIT_OUT:%.+]] = const.Declare tensor<1x32x1024x128xf16> = dense<0.000000e+00>
  // CHECK-DAG: [[INIT_MAX:%.+]] = const.Declare tensor<1x32x1024xf16> = dense<0xFC00>
  // CHECK-DAG: [[INIT_SUM:%.+]] = const.Declare tensor<1x32x1024xf32> = dense<0.000000e+00>
  // Q is un-folded from [8,4,1024,128] to [1,32,1024,128].
  // CHECK: [[Q_UF:%.+]] = IE.Reshape([[ARG_Q]]) {shape_value = [1, 32, 1024, 128]} : tensor<8x4x1024x128xf16> -> tensor<1x32x1024x128xf16>
  // K/V tiles are used directly (batch=1); V is transposed to [...,S,E] layout for FlashSDPA.
  // Tile 0 (head).
  // CHECK: [[V0_T:%.+]] = IE.Transpose([[ARG_V0]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK: [[OUT0:%.+]], [[MAX0:%.+]], [[SUM0:%.+]] = IE.FlashSDPA([[Q_UF]], [[ARG_K0]], [[V0_T]], [[INIT_OUT]], [[INIT_MAX]], [[INIT_SUM]]) {is_head = true, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // Tile 1 (middle).
  // CHECK: [[V1_T:%.+]] = IE.Transpose([[ARG_V1]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK: [[OUT1:%.+]], [[MAX1:%.+]], [[SUM1:%.+]] = IE.FlashSDPA([[Q_UF]], [[ARG_K1]], [[V1_T]], [[OUT0]], [[MAX0]], [[SUM0]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // Tile 2 (tail).
  // CHECK: [[V2_T:%.+]] = IE.Transpose([[ARG_V2]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK: [[OUT2:%.+]], {{%.+}}, {{%.+}} = IE.FlashSDPA([[Q_UF]], [[ARG_K2]], [[V2_T]], [[OUT1]], [[MAX1]], [[SUM1]]) {is_head = false, is_tail = true, source_seq_len_pad_size = 0 : i64}
  // Output is re-folded to the original AttentionOp result type [8,4,1024,128].
  // CHECK: [[OUT_FOLD:%.+]] = IE.Reshape([[OUT2]]) {shape_value = [8, 4, 1024, 128]} : tensor<1x32x1024x128xf16> -> tensor<8x4x1024x128xf16>
  // CHECK: return [[OUT_FOLD]]
  // CHECK-NOT: IE.Attention
}

// -----

// GQA Decode, 10 KV tiles (8×1024 + 127 + 1 = 8320), scale and causal mask.
// This is the closest case to real model IR produced by FuseAttentionPass:
//   - Q: [1,32,1,128] -> [8,4,1,128]     (kv_heads=8, grp=4)
//   - K: Concat(8×[1,8,1024] + [1,8,127] + [1,8,1]) -> Reshape [8,1,8320,128]
//   - V: Concat(8×[1,8,128,1024] + [1,8,128,127] + [1,8,128,1]) -> Reshape [8,1,128,8320]
//   - Scale: const [1,1,1,1] f16; Mask: [1,1,1,8320] f16
// sSL=8320 >= 8192 (decode threshold), 8320 % 16 == 0.
// Greedy merge: tiles 0-7 (1024, aligned) flush individually;
//              tiles 8-9 (127+1=128, aligned) merge into one group.
// Result: 9 FlashSDPA tiles.

// CHECK-LABEL: @GQADecodeTenBlocks
// CHECK-SAME:  ([[ARG_Q:%.+]]: tensor<8x4x1x128xf16>,
// CHECK-SAME:  [[ARG_K0:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K1:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K2:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K3:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K4:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K5:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K6:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_K7:%.+]]: tensor<1x8x1024x128xf16>, [[ARG_KT:%.+]]: tensor<1x8x127x128xf16>, [[ARG_KP:%.+]]: tensor<1x8x1x128xf16>,
// CHECK-SAME:  [[ARG_V0:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V1:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V2:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V3:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V4:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V5:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V6:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_V7:%.+]]: tensor<1x8x128x1024xf16>, [[ARG_VT:%.+]]: tensor<1x8x128x127xf16>, [[ARG_VP:%.+]]: tensor<1x8x128x1xf16>,
// CHECK-SAME:  [[ARG_MASK:%.+]]: tensor<1x1x1x8320xf16>)
func.func @GQADecodeTenBlocks(
        %arg0:  tensor<8x4x1x128xf16>,
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
        %arg21: tensor<1x1x1x8320xf16>) -> tensor<8x4x1x128xf16> {
  %cst_scale = const.Declare tensor<1x1x1x1xf16> = dense<8.83883476e-02> : tensor<1x1x1x1xf16>
  %k_cat = IE.Concat(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 1024, 0], [0, 0, 2048, 0], [0, 0, 3072, 0],
                        [0, 0, 4096, 0], [0, 0, 5120, 0], [0, 0, 6144, 0], [0, 0, 7168, 0],
                        [0, 0, 8192, 0], [0, 0, 8319, 0]]} :
      tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>,
      tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>,
      tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x127x128xf16>,
      tensor<1x8x1x128xf16>
      -> tensor<1x8x8320x128xf16>
  %k_rs = IE.Reshape(%k_cat) {shape_value = [8, 1, 8320, 128]} :
      tensor<1x8x8320x128xf16> -> tensor<8x1x8320x128xf16>
  %v_cat = IE.Concat(%arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1024], [0, 0, 0, 2048], [0, 0, 0, 3072],
                        [0, 0, 0, 4096], [0, 0, 0, 5120], [0, 0, 0, 6144], [0, 0, 0, 7168],
                        [0, 0, 0, 8192], [0, 0, 0, 8319]]} :
      tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>,
      tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>,
      tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>, tensor<1x8x128x127xf16>,
      tensor<1x8x128x1xf16>
      -> tensor<1x8x128x8320xf16>
  %v_rs = IE.Reshape(%v_cat) {shape_value = [8, 1, 128, 8320]} :
      tensor<1x8x128x8320xf16> -> tensor<8x1x128x8320xf16>
  %out = IE.Attention(%arg0, %k_rs, %v_rs, %arg21, %cst_scale) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>} :
      tensor<8x4x1x128xf16>, tensor<8x1x8320x128xf16>, tensor<8x1x128x8320xf16>,
      tensor<1x1x1x8320xf16>, tensor<1x1x1x1xf16>
      -> tensor<8x4x1x128xf16>
  return %out : tensor<8x4x1x128xf16>

  // Q un-folded [8,4,1,128] -> [1,32,1,128]; running states sized from un-folded Q.
  // CHECK-DAG: [[CST_SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16>
  // CHECK-DAG: [[INIT_OUT:%.+]] = const.Declare tensor<1x32x1x128xf16> = dense<0.000000e+00>
  // CHECK-DAG: [[INIT_MAX:%.+]] = const.Declare tensor<1x32x1xf16> = dense<0xFC00>
  // CHECK-DAG: [[INIT_SUM:%.+]] = const.Declare tensor<1x32x1xf32> = dense<0.000000e+00>
  // CHECK: [[Q_UF:%.+]] = IE.Reshape([[ARG_Q]]) {shape_value = [1, 32, 1, 128]} : tensor<8x4x1x128xf16> -> tensor<1x32x1x128xf16>
  // CHECK: [[Q_SC:%.+]] = IE.Multiply([[Q_UF]], [[CST_SCALE]])
  // Tiles 0-7: 1024 tokens each (16-aligned), flushed individually.
  // CHECK: [[V0_T:%.+]] = IE.Transpose([[ARG_V0]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK: [[MASK0:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 0] [1, 1, 1, 1024]
  // CHECK: [[O0:%.+]], [[M0:%.+]], [[S0:%.+]] = IE.FlashSDPA([[Q_SC]], [[ARG_K0]], [[V0_T]], [[INIT_OUT]], [[INIT_MAX]], [[INIT_SUM]], [[MASK0]]) {is_head = true, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK: [[V1_T:%.+]] = IE.Transpose([[ARG_V1]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK: [[MASK1:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 1024] [1, 1, 1, 1024]
  // CHECK: [[O1:%.+]], [[M1:%.+]], [[S1:%.+]] = IE.FlashSDPA([[Q_SC]], [[ARG_K1]], [[V1_T]], [[O0]], [[M0]], [[S0]], [[MASK1]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK: [[V2_T:%.+]] = IE.Transpose([[ARG_V2]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK: [[MASK2:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 2048] [1, 1, 1, 1024]
  // CHECK: [[O2:%.+]], [[M2:%.+]], [[S2:%.+]] = IE.FlashSDPA([[Q_SC]], [[ARG_K2]], [[V2_T]], [[O1]], [[M1]], [[S1]], [[MASK2]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK: [[V3_T:%.+]] = IE.Transpose([[ARG_V3]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK: [[MASK3:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 3072] [1, 1, 1, 1024]
  // CHECK: [[O3:%.+]], [[M3:%.+]], [[S3:%.+]] = IE.FlashSDPA([[Q_SC]], [[ARG_K3]], [[V3_T]], [[O2]], [[M2]], [[S2]], [[MASK3]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK: [[V4_T:%.+]] = IE.Transpose([[ARG_V4]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK: [[MASK4:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 4096] [1, 1, 1, 1024]
  // CHECK: [[O4:%.+]], [[M4:%.+]], [[S4:%.+]] = IE.FlashSDPA([[Q_SC]], [[ARG_K4]], [[V4_T]], [[O3]], [[M3]], [[S3]], [[MASK4]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK: [[V5_T:%.+]] = IE.Transpose([[ARG_V5]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK: [[MASK5:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 5120] [1, 1, 1, 1024]
  // CHECK: [[O5:%.+]], [[M5:%.+]], [[S5:%.+]] = IE.FlashSDPA([[Q_SC]], [[ARG_K5]], [[V5_T]], [[O4]], [[M4]], [[S4]], [[MASK5]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK: [[V6_T:%.+]] = IE.Transpose([[ARG_V6]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK: [[MASK6:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 6144] [1, 1, 1, 1024]
  // CHECK: [[O6:%.+]], [[M6:%.+]], [[S6:%.+]] = IE.FlashSDPA([[Q_SC]], [[ARG_K6]], [[V6_T]], [[O5]], [[M5]], [[S5]], [[MASK6]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // CHECK: [[V7_T:%.+]] = IE.Transpose([[ARG_V7]]) {order_value = #NCWH} : tensor<1x8x128x1024xf16> -> tensor<1x8x1024x128xf16>
  // CHECK: [[MASK7:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 7168] [1, 1, 1, 1024]
  // CHECK: [[O7:%.+]], [[M7:%.+]], [[S7:%.+]] = IE.FlashSDPA([[Q_SC]], [[ARG_K7]], [[V7_T]], [[O6]], [[M6]], [[S6]], [[MASK7]]) {is_head = false, is_tail = false, source_seq_len_pad_size = 0 : i64}
  // Tile 8 (tail): [127-token, 1-token] merged to 128 tokens (16-aligned).
  // CHECK: [[K_TAIL:%.+]] = IE.Concat([[ARG_KT]], [[ARG_KP]]) {{.*}} : tensor<1x8x127x128xf16>, tensor<1x8x1x128xf16> -> tensor<1x8x128x128xf16>
  // CHECK: [[V_TAIL:%.+]] = IE.Concat([[ARG_VT]], [[ARG_VP]]) {{.*}} : tensor<1x8x128x127xf16>, tensor<1x8x128x1xf16> -> tensor<1x8x128x128xf16>
  // CHECK: [[VTT:%.+]] = IE.Transpose([[V_TAIL]]) {order_value = #NCWH} : tensor<1x8x128x128xf16> -> tensor<1x8x128x128xf16>
  // CHECK: [[MASK8:%.+]] = IE.Slice [[ARG_MASK]] [0, 0, 0, 8192] [1, 1, 1, 128]
  // CHECK: [[O8:%.+]], {{%.+}}, {{%.+}} = IE.FlashSDPA([[Q_SC]], [[K_TAIL]], [[VTT]], [[O7]], [[M7]], [[S7]], [[MASK8]]) {is_head = false, is_tail = true, source_seq_len_pad_size = 0 : i64}
  // Output re-folded to the original AttentionOp result type [8,4,1,128].
  // CHECK: [[OUT_FOLD:%.+]] = IE.Reshape([[O8]]) {shape_value = [8, 4, 1, 128]} : tensor<1x32x1x128xf16> -> tensor<8x4x1x128xf16>
  // CHECK: return [[OUT_FOLD]]
  // CHECK-NOT: IE.Attention
}

// -----

// sSL = 1024 < 2048 (prefill threshold). Conversion skipped.

// CHECK-LABEL: @GQASmallSSL_NotConverted
func.func @GQASmallSSL_NotConverted(
        %arg0: tensor<8x4x1024x128xf16>,
        %arg1: tensor<1x8x512x128xf16>,
        %arg2: tensor<1x8x512x128xf16>,
        %arg3: tensor<1x8x128x512xf16>,
        %arg4: tensor<1x8x128x512xf16>) -> tensor<8x4x1024x128xf16> {
  %k_cat = IE.Concat(%arg1, %arg2) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 512, 0]]} :
      tensor<1x8x512x128xf16>, tensor<1x8x512x128xf16> -> tensor<1x8x1024x128xf16>
  %k_rs = IE.Reshape(%k_cat) {shape_value = [8, 1, 1024, 128]} :
      tensor<1x8x1024x128xf16> -> tensor<8x1x1024x128xf16>
  %v_cat = IE.Concat(%arg3, %arg4) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 0, 512]]} :
      tensor<1x8x128x512xf16>, tensor<1x8x128x512xf16> -> tensor<1x8x128x1024xf16>
  %v_rs = IE.Reshape(%v_cat) {shape_value = [8, 1, 128, 1024]} :
      tensor<1x8x128x1024xf16> -> tensor<8x1x128x1024xf16>
  %out = IE.Attention(%arg0, %k_rs, %v_rs) {
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} :
      tensor<8x4x1024x128xf16>, tensor<8x1x1024x128xf16>, tensor<8x1x128x1024xf16>
      -> tensor<8x4x1024x128xf16>
  return %out : tensor<8x4x1024x128xf16>
  // CHECK-NOT: IE.FlashSDPA
  // CHECK: IE.Attention
}

// -----

// Decode, sSL = 8x1024+127 = 8319, 8319 % 16 != 0.
// Conversion skipped because no tile grouping can produce all 16-aligned FlashSDPA inputs.

// CHECK-LABEL: @GQAUnalignedSSL_NotConverted
func.func @GQAUnalignedSSL_NotConverted(
        %arg0: tensor<8x4x1x128xf16>,
        %arg1: tensor<1x8x1024x128xf16>,
        %arg2: tensor<1x8x1024x128xf16>,
        %arg3: tensor<1x8x1024x128xf16>,
        %arg4: tensor<1x8x1024x128xf16>,
        %arg5: tensor<1x8x1024x128xf16>,
        %arg6: tensor<1x8x1024x128xf16>,
        %arg7: tensor<1x8x1024x128xf16>,
        %arg8: tensor<1x8x1024x128xf16>,
        %arg9: tensor<1x8x127x128xf16>,
        %arg10: tensor<1x8x128x1024xf16>,
        %arg11: tensor<1x8x128x1024xf16>,
        %arg12: tensor<1x8x128x1024xf16>,
        %arg13: tensor<1x8x128x1024xf16>,
        %arg14: tensor<1x8x128x1024xf16>,
        %arg15: tensor<1x8x128x1024xf16>,
        %arg16: tensor<1x8x128x1024xf16>,
        %arg17: tensor<1x8x128x1024xf16>,
        %arg18: tensor<1x8x128x127xf16>) -> tensor<8x4x1x128xf16> {
  %k_cat = IE.Concat(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 1024, 0], [0, 0, 2048, 0], [0, 0, 3072, 0],
                        [0, 0, 4096, 0], [0, 0, 5120, 0], [0, 0, 6144, 0], [0, 0, 7168, 0],
                        [0, 0, 8192, 0]]} :
      tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>,
      tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>,
      tensor<1x8x1024x128xf16>, tensor<1x8x1024x128xf16>, tensor<1x8x127x128xf16>
      -> tensor<1x8x8319x128xf16>
  %k_rs = IE.Reshape(%k_cat) {shape_value = [8, 1, 8319, 128]} :
      tensor<1x8x8319x128xf16> -> tensor<8x1x8319x128xf16>
  %v_cat = IE.Concat(%arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1024], [0, 0, 0, 2048], [0, 0, 0, 3072],
                        [0, 0, 0, 4096], [0, 0, 0, 5120], [0, 0, 0, 6144], [0, 0, 0, 7168],
                        [0, 0, 0, 8192]]} :
      tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>,
      tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>,
      tensor<1x8x128x1024xf16>, tensor<1x8x128x1024xf16>, tensor<1x8x128x127xf16>
      -> tensor<1x8x128x8319xf16>
  %v_rs = IE.Reshape(%v_cat) {shape_value = [8, 1, 128, 8319]} :
      tensor<1x8x128x8319xf16> -> tensor<8x1x128x8319xf16>
  %out = IE.Attention(%arg0, %k_rs, %v_rs) {
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} :
      tensor<8x4x1x128xf16>, tensor<8x1x8319x128xf16>, tensor<8x1x128x8319xf16>
      -> tensor<8x4x1x128xf16>
  return %out : tensor<8x4x1x128xf16>
  // CHECK-NOT: IE.FlashSDPA
  // CHECK: IE.Attention
}

// -----

// Prefill, sSL = 7168 + 1024 = 8192, 8192 % 16 == 0, >= 2048 (prefill threshold).
// Conversion skipped because the Concat has only 2 inputs: this is not a paged/block
// attention pattern (which requires >= 3 inputs).

// CHECK-LABEL: @GQATwoTiles_NotConverted
func.func @GQATwoTiles_NotConverted(
        %arg0: tensor<8x4x1024x128xf16>,
        %arg1: tensor<1x8x7168x128xf16>,
        %arg2: tensor<1x8x1024x128xf16>,
        %arg3: tensor<1x8x128x7168xf16>,
        %arg4: tensor<1x8x128x1024xf16>) -> tensor<8x4x1024x128xf16> {
  %k_cat = IE.Concat(%arg1, %arg2) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 7168, 0]]} :
      tensor<1x8x7168x128xf16>, tensor<1x8x1024x128xf16> -> tensor<1x8x8192x128xf16>
  %k_rs = IE.Reshape(%k_cat) {shape_value = [8, 1, 8192, 128]} :
      tensor<1x8x8192x128xf16> -> tensor<8x1x8192x128xf16>
  %v_cat = IE.Concat(%arg3, %arg4) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 0, 7168]]} :
      tensor<1x8x128x7168xf16>, tensor<1x8x128x1024xf16> -> tensor<1x8x128x8192xf16>
  %v_rs = IE.Reshape(%v_cat) {shape_value = [8, 1, 128, 8192]} :
      tensor<1x8x128x8192xf16> -> tensor<8x1x128x8192xf16>
  %out = IE.Attention(%arg0, %k_rs, %v_rs) {
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} :
      tensor<8x4x1024x128xf16>, tensor<8x1x8192x128xf16>, tensor<8x1x128x8192xf16>
      -> tensor<8x4x1024x128xf16>
  return %out : tensor<8x4x1024x128xf16>
  // CHECK-NOT: IE.FlashSDPA
  // CHECK: IE.Attention
}
