//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @BatchToSpaceAttr
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<24x16x7x8xf32>
  func.func @BatchToSpaceAttr(%arg0: tensor<24x16x7x8xf32>) -> tensor<1x32x20x31xf32> {
    %0 = IE.BatchToSpace(%arg0) {block_shape_value = [1, 2, 3, 4], crops_begin_value = [0, 0, 0, 1], crops_end_value = [0, 0, 1, 0]} : tensor<24x16x7x8xf32> -> tensor<1x32x20x31xf32>

    return %0 : tensor<1x32x20x31xf32>

    // CHECK-NOT:   const.Declare
    // CHECK: [[VAL0:%.+]] = IE.BatchToSpace([[ARG_0]]) {block_shape_value = [1, 2, 3, 4], crops_begin_value = [0, 0, 0, 1], crops_end_value = [0, 0, 1, 0]} : tensor<24x16x7x8xf32> -> tensor<1x32x20x31xf32>
    // CHECK: return [[VAL0]] : tensor<1x32x20x31xf32>
}
