//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --run-adjust-for-vpu-rewriters="rewriter=convert-shuffle-channels" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @ConvertShuffleChannels
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x4x3x2xf16>
func.func @ConvertShuffleChannels(%arg0: tensor<1x4x3x2xf16>) -> tensor<1x4x3x2xf16> {

  %prob = IE.ShuffleChannels(%arg0) {axis = 1, group = 2} : tensor<1x4x3x2xf16> -> tensor<1x4x3x2xf16>

  return %prob : tensor<1x4x3x2xf16>

  //CHECK:              [[VAL0:%.+]] = IE.Reshape([[ARG_0]])
  //CHECK-SAME{LITERAL}:                  {shape_value = [2, 2, 3, 2]} : tensor<1x4x3x2xf16> -> tensor<2x2x3x2xf16>
  //CHECK:              [[VAL1:%.+]] = IE.Transpose([[VAL0]]) {order_value = #map} : tensor<2x2x3x2xf16> -> tensor<2x2x3x2xf16>
  //CHECK:              [[VAL2:%.+]] = IE.Reshape([[VAL1]])
  //CHECK-SAME{LITERAL}:                  {shape_value = [1, 4, 3, 2]} : tensor<2x2x3x2xf16> -> tensor<1x4x3x2xf16>
  //CHECK:              return [[VAL2]]
}
