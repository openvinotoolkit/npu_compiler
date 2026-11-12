//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @LegalizeAxisInd
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x8x4x4xf32>
func.func @LegalizeAxisInd(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
    %softmax = IE.SoftMax(%arg0) {axisInd = -1} : tensor<1x8x4x4xf32> -> tensor<1x8x4x4xf32>
    return %softmax : tensor<1x8x4x4xf32>

    // CHECK:       [[VAL0:%.+]] = IE.SoftMax([[ARG_0]]) {axisInd = 3 : i64} : tensor<1x8x4x4xf32> -> tensor<1x8x4x4xf32>
    // CHECK-NOT:   IE.SoftMax
    // CHECK:       return [[VAL0]] : tensor<1x8x4x4xf32>
}
