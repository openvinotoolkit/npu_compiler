//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --decompose-attention %s | FileCheck %s
// REQUIRES: platform-NPU4000

// CHECK-LABEL: @ForceDecomposeNonLegalNPU4000Config
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x16x577x32xf16>, [[ARG1:%.+]]: tensor<1x16x577x32xf16>, [[ARG2:%.+]]: tensor<1x16x32x577xf16>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>)
func.func @ForceDecomposeNonLegalNPU4000Config(%arg0: tensor<1x16x577x32xf16>, %arg1: tensor<1x16x577x32xf16>, %arg2: tensor<1x16x32x577xf16>, %arg3: tensor<1x1x1x1xf32>) -> tensor<1x16x577x32xf16> {
  %0 = IE.Attention(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>} : tensor<1x16x577x32xf16>, tensor<1x16x577x32xf16>, tensor<1x16x32x577xf16>, tensor<1x1x1x1xf32> -> tensor<1x16x577x32xf16>
  return %0 : tensor<1x16x577x32xf16>

  // CHECK-NOT: IE.Attention
}
