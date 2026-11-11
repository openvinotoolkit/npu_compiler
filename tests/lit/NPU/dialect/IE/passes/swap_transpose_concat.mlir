//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --swap-transpose-concat %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapTransposeConcatWithOffsets
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x76x4x1xf16>, [[ARG_1:%[^:]+]]: tensor<1x76x4x1xf16>)
func.func @SwapTransposeConcatWithOffsets(%arg0: tensor<1x76x4x1xf16>, %arg1: tensor<1x76x4x1xf16> ) -> tensor<1x8x76x1xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHCW} : tensor<1x76x4x1xf16> -> tensor<1x4x76x1xf16>
    %1 = IE.Transpose(%arg1) {order_value = #NHCW} : tensor<1x76x4x1xf16> -> tensor<1x4x76x1xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]]} : tensor<1x4x76x1xf16>, tensor<1x4x76x1xf16> -> tensor<1x8x76x1xf16>
    return %2: tensor<1x8x76x1xf16>

    // CHECK:                IE.Concat([[ARG_0]], [[ARG_1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x76x4x1xf16>, tensor<1x76x4x1xf16> -> tensor<1x76x8x1xf16>
    // CHECK:                IE.Transpose
    // CHECK-SAME:           {order_value = #NHCW} : tensor<1x76x8x1xf16> -> tensor<1x8x76x1xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapTransposeConcatWithAxis
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x76x4x1xf16>, [[ARG_1:%[^:]+]]: tensor<1x76x4x1xf16>)
func.func @SwapTransposeConcatWithAxis(%arg0: tensor<1x76x4x1xf16>, %arg1: tensor<1x76x4x1xf16> ) -> tensor<1x8x76x1xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHCW} : tensor<1x76x4x1xf16> -> tensor<1x4x76x1xf16>
    %1 = IE.Transpose(%arg1) {order_value = #NHCW} : tensor<1x76x4x1xf16> -> tensor<1x4x76x1xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x4x76x1xf16>, tensor<1x4x76x1xf16> -> tensor<1x8x76x1xf16>
    return %2: tensor<1x8x76x1xf16>

    // CHECK:                IE.Concat([[ARG_0]], [[ARG_1]])
    // CHECK-SAME{LITERAL}:     {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x76x4x1xf16>, tensor<1x76x4x1xf16> -> tensor<1x76x8x1xf16>
    // CHECK:                IE.Transpose
    // CHECK-SAME:              {order_value = #NHCW} : tensor<1x76x8x1xf16> -> tensor<1x8x76x1xf16>
}
