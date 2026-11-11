//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @AddAttentionMaskWithCausalMask
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x64x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x32x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x32x128xf16>
func.func @AddAttentionMaskWithCausalMask(%arg0: tensor<8x64x64xf16>, %arg1: tensor<8x32x64xf16>, %arg2: tensor<8x32x128xf16>) -> tensor<8x64x128xf16> {
  %0 = IE.SDPA(%arg0, %arg1, %arg2) {causal, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>} : tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16> -> tensor<8x64x128xf16>
  return %0 : tensor<8x64x128xf16>

  // CHECK:     [[CAUSAL_MASK:%.+]] = const.Declare tensor<64x32xf16>
  // CHECK:     [[RESULT:%.+]] = IE.SDPA([[QUERY]], [[KEY]], [[VALUE]], [[CAUSAL_MASK]])
  // CHECK-NOT:     causal
  // CHECK-SAME:    tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16>, tensor<64x32xf16>
  // CHECK-SAME:    -> tensor<8x64x128xf16>
  // CHECK:     return [[RESULT]]
}

// -----

// CHECK-LABEL: @ReplaceAttentionMaskWithCausalMask
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x64x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x32x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x32x128xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<8x64x32xf16>
func.func @ReplaceAttentionMaskWithCausalMask(%arg0: tensor<8x64x64xf16>, %arg1: tensor<8x32x64xf16>, %arg2: tensor<8x32x128xf16>, %arg3: tensor<8x64x32xf16>) -> tensor<8x64x128xf16> {
  %0 = IE.SDPA(%arg0, %arg1, %arg2, %arg3) {causal, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>} : tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16>, tensor<8x64x32xf16> -> tensor<8x64x128xf16>
  return %0 : tensor<8x64x128xf16>

  // CHECK:     [[CAUSAL_MASK:%.+]] = const.Declare tensor<64x32xf16>
  // CHECK:     [[RESULT:%.+]] = IE.SDPA([[QUERY]], [[KEY]], [[VALUE]], [[CAUSAL_MASK]])
  // CHECK-NOT:     causal
  // CHECK-SAME:    tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16>, tensor<64x32xf16>
  // CHECK-SAME:    -> tensor<8x64x128xf16>
  // CHECK:     return [[RESULT]]
}
