//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% workload-management-mode=FWLM_V1_PAGES" --fuse-attention="fuse-only-attention-dma-patterns=true" --canonicalize %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-attention="fuse-only-attention-dma-patterns=true" --canonicalize %s | FileCheck %s --check-prefix=NO-FWLM
// REQUIRES: platform-NPU4000

// Legal AttentionDMA config with the scale on the scores and a shared Q: not beneficial for the forced
// decomposition, but fused because it becomes AttentionDMA under FWLM. Kept without FWLM.

// CHECK-LABEL: @FuseLegalConfigAttentionDMAPattern
// CHECK-SAME:  ([[ARG_Q:%.+]]: tensor<1x12x3600x16xf16>, [[ARG_K:%.+]]: tensor<1x12x3600x16xf16>, [[ARG_V:%.+]]: tensor<1x12x16x3600xf16>, [[ARG_G:%.+]]: tensor<1x12x8x16xf16>)
// NO-FWLM-LABEL: @FuseLegalConfigAttentionDMAPattern
func.func @FuseLegalConfigAttentionDMAPattern(
        %arg0: tensor<1x12x3600x16xf16>,
        %arg1: tensor<1x12x3600x16xf16>,
        %arg2: tensor<1x12x16x3600xf16>,
        %arg3: tensor<1x12x8x16xf16>) -> (tensor<1x12x3600x16xf16>, tensor<1x12x3600x8xf16>) {
  %scale = const.Declare tensor<1x1x1x1xf16> = dense<2.500000e-01> : tensor<1x1x1x1xf16>
  %qk = IE.MatMul(%arg0, %arg1) {transpose_b} :
      tensor<1x12x3600x16xf16>, tensor<1x12x3600x16xf16> -> tensor<1x12x3600x3600xf16>
  %scaled = IE.Multiply(%qk, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
      tensor<1x12x3600x3600xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x3600x3600xf16>
  %attn = IE.SoftMax(%scaled) {axisInd = 3 : i64} :
      tensor<1x12x3600x3600xf16> -> tensor<1x12x3600x3600xf16>
  %out = IE.MatMul(%attn, %arg2) {transpose_b} :
      tensor<1x12x3600x3600xf16>, tensor<1x12x16x3600xf16> -> tensor<1x12x3600x16xf16>
  %gate = IE.MatMul(%arg0, %arg3) {transpose_b} :
      tensor<1x12x3600x16xf16>, tensor<1x12x8x16xf16> -> tensor<1x12x3600x8xf16>
  return %out, %gate : tensor<1x12x3600x16xf16>, tensor<1x12x3600x8xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16>
  // CHECK-NOT: IE.SoftMax
  // CHECK-NOT: IE.Attention(
  // CHECK:     [[OUT:%.+]] = IE.AttentionDMA([[ARG_Q]], [[ARG_K]], [[ARG_V]], [[SCALE]])
  // CHECK:     [[GATE:%.+]] = IE.MatMul([[ARG_Q]], [[ARG_G]]) {transpose_b}
  // CHECK:     return [[OUT]], [[GATE]]

  // NO-FWLM:     IE.SoftMax
  // NO-FWLM-NOT: IE.Attention
}

// -----

// A dynamic (bounded) K/V shape is always legal for AttentionDMA inputs, but AttentionDMA itself is
// only available under FWLM (see isAttentionDMAAvailable), so without FWLM the SDPA still becomes
// IE.Attention but is not further converted to IE.AttentionDMA.

// CHECK-LABEL: @FuseDynamicShapeAttentionDMAPattern
// CHECK-SAME:  ([[ARG_Q:%.+]]: tensor<1x6x64x64xf16>, [[ARG_K:%.+]]: tensor<1x6x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 6, 256, 64]> : tensor<4xsi64>, order = #NCHW}>, [[ARG_V:%.+]]: tensor<1x6x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 6, 256, 64]> : tensor<4xsi64>, order = #NCHW}>)
// NO-FWLM-LABEL: @FuseDynamicShapeAttentionDMAPattern
func.func @FuseDynamicShapeAttentionDMAPattern(
        %arg0: tensor<1x6x64x64xf16>,
        %arg1: tensor<1x6x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 6, 256, 64]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>,
        %arg2: tensor<1x6x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 6, 256, 64]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> tensor<1x6x64x64xf16> {
  %cst = const.Declare tensor<1xf16> = dense<1.250000e-01> : tensor<1xf16>
  %0 = IE.SDPA(%arg0, %arg1, %arg2, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} :
      tensor<1x6x64x64xf16>,
      tensor<1x6x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 6, 256, 64]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>,
      tensor<1x6x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 6, 256, 64]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>,
      tensor<1xf16> -> tensor<1x6x64x64xf16>
  return %0 : tensor<1x6x64x64xf16>

  // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[ARG_V]])
  // CHECK-NOT:   IE.SDPA
  // CHECK-NOT:   IE.Attention(
  // CHECK:       [[OUT:%.+]] = IE.AttentionDMA([[ARG_Q]], [[ARG_K]], [[TRANSPOSE]], [[SCALE:%.+]])
  // CHECK:       return [[OUT]]

  // NO-FWLM-NOT: IE.SDPA
  // NO-FWLM:     IE.Attention(
  // NO-FWLM-NOT: IE.AttentionDMA
}
