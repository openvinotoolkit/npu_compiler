//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --attention-processing="convert-to-attention=true decompose-attention=false" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// -----

// Standard MHA subgraph: MatMul(Q, K^T) -> SoftMax -> MatMul(attn, V^T)
// FuseAttention should fold it into IE.AttentionOp.
//
// CHECK-LABEL: @SubgraphFusionToAttention
// CHECK-SAME:  ([[ARG_Q:%.+]]: tensor<1x16x128x64xf16>, [[ARG_K:%.+]]: tensor<1x16x128x64xf16>, [[ARG_V:%.+]]: tensor<1x16x128x64xf16>)
func.func @SubgraphFusionToAttention(
        %arg0: tensor<1x16x128x64xf16>,
        %arg1: tensor<1x16x128x64xf16>,
        %arg2: tensor<1x16x128x64xf16>) -> tensor<1x16x128x64xf16> {
  %qk = IE.MatMul(%arg0, %arg1) {transpose_b} :
      tensor<1x16x128x64xf16>, tensor<1x16x128x64xf16> -> tensor<1x16x128x128xf16>
  %attn = IE.SoftMax(%qk) {axisInd = 3 : i64} :
      tensor<1x16x128x128xf16> -> tensor<1x16x128x128xf16>
  %v_t = IE.Transpose(%arg2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} :
      tensor<1x16x128x64xf16> -> tensor<1x16x64x128xf16>
  %out = IE.MatMul(%attn, %v_t) {transpose_b} :
      tensor<1x16x128x128xf16>, tensor<1x16x64x128xf16> -> tensor<1x16x128x64xf16>
  return %out : tensor<1x16x128x64xf16>

  // CHECK:     [[VT:%.+]] = IE.Transpose([[ARG_V]])
  // CHECK:     [[OUT:%.+]] = IE.Attention([[ARG_Q]], [[ARG_K]], [[VT]]) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : tensor<1x16x128x64xf16>, tensor<1x16x128x64xf16>, tensor<1x16x64x128xf16> -> tensor<1x16x128x64xf16>
  // CHECK-NOT: IE.MatMul
  // CHECK-NOT: IE.SoftMax
  // CHECK:     return [[OUT]]
}
