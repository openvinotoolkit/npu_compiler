//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --attention-processing="convert-to-attention=true force-attention-decomposition=true" %s | FileCheck %s
// REQUIRES: platform-NPU4000

// CHECK-LABEL: @DecomposeMisalignedAttentionWithoutExpand
func.func @DecomposeMisalignedAttentionWithoutExpand(
        %q: tensor<1x8x1500x64xf16>,
        %k: tensor<1x8x1500x64xf16>,
        %v: tensor<1x8x1500x64xf16>) -> tensor<1x8x1500x64xf16> {
  %scale = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
  %attention = IE.Attention(%q, %k, %v, %scale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x8x1500x64xf16>, tensor<1x8x1500x64xf16>, tensor<1x8x1500x64xf16>, tensor<1xf16> -> tensor<1x8x1500x64xf16>
  return %attention : tensor<1x8x1500x64xf16>

  // CHECK-NOT: padSize
  // CHECK-NOT: 1504
  // CHECK:     [[QK:%.+]] = IE.MatMul([[ARG0:%.+]], [[ARG1:%.+]]) {transpose_b} : tensor<1x8x1500x64xf16>, tensor<1x8x1500x64xf16> -> tensor<1x8x1500x1500xf16>
  // CHECK:     [[SM:%.+]] = IE.SoftMax([[QK]]) {axisInd = 3 : i64} : tensor<1x8x1500x1500xf16> -> tensor<1x8x1500x1500xf16>
  // CHECK:     [[OUT:%.+]] = IE.MatMul([[SM]], [[ARG2:%.+]]) : tensor<1x8x1500x1500xf16>, tensor<1x8x1500x64xf16> -> tensor<1x8x1500x64xf16>
  // CHECK:     return [[OUT]]
}
