//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-l2-ops %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @DecomposeNormalizeL2
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x24x64xf16>)
func.func @DecomposeNormalizeL2(%arg0: tensor<1x8x24x64xf16>) -> tensor<1x8x24x64xf16> {
    %0 = IE.NormalizeL2(%arg0) {axes_value = [0, 1, 2, 3], eps = 9.9999999392252903E-9 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x8x24x64xf16> -> tensor<1x8x24x64xf16>
    return %0 : tensor<1x8x24x64xf16>

// CHECK:   [[MULTIPLY:%.+]]   = IE.Multiply([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x24x64xf16>, tensor<1x8x24x64xf16> -> tensor<1x8x24x64xf16>
// CHECK:   [[REDUCE_SUM:%.+]] = IE.ReduceSum([[MULTIPLY]]) {axes_value = [0, 1, 2, 3]} : tensor<1x8x24x64xf16> -> tensor<1xf16>
// CHECK:   [[SQRT:%.+]]       = IE.Sqrt([[REDUCE_SUM]]) : tensor<1xf16> -> tensor<1xf16>
// CHECK:   [[DIVIDE:%.+]]     = IE.Divide([[ARG0]], [[SQRT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x24x64xf16>, tensor<1xf16> -> tensor<1x8x24x64xf16>
// CHECK:   return [[DIVIDE]] : tensor<1x8x24x64xf16>
}

// -----

// CHECK-LABEL: @DecomposeReduceL2
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x24x64xf16>)
func.func @DecomposeReduceL2(%arg0: tensor<1x8x24x64xf16>) -> tensor<1x8x24x1xf16> {
    %0 = IE.ReduceL2(%arg0) {axes_value = [3], keep_dims} : tensor<1x8x24x64xf16> -> tensor<1x8x24x1xf16>
    return %0 : tensor<1x8x24x1xf16>

// CHECK:   [[MULTIPLY:%.+]]   = IE.Multiply([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x24x64xf16>, tensor<1x8x24x64xf16> -> tensor<1x8x24x64xf16>
// CHECK:   [[REDUCE_SUM:%.+]] = IE.ReduceSum([[MULTIPLY]]) {axes_value = [3], keep_dims} : tensor<1x8x24x64xf16> -> tensor<1x8x24x1xf16>
// CHECK:   [[SQRT:%.+]]       = IE.Sqrt([[REDUCE_SUM]]) : tensor<1x8x24x1xf16> -> tensor<1x8x24x1xf16>
// CHECK:   return [[SQRT]] : tensor<1x8x24x1xf16>
}

// -----

// CHECK-LABEL: @NotDecomposeReduceL2SmallInput
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x3xf16>)
func.func @NotDecomposeReduceL2SmallInput(%arg0: tensor<1x3xf16>) -> tensor<1x1xf16> {
    %0 = IE.ReduceL2(%arg0) {axes_value = [1], keep_dims} : tensor<1x3xf16> -> tensor<1x1xf16>
    return %0 : tensor<1x1xf16>

// CHECK: IE.ReduceL2
}

// -----

// CHECK-LABEL: @NotDecomposeReduceL2TooMuchReducedElements
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x24x64xf16>)
func.func @NotDecomposeReduceL2TooMuchReducedElements(%arg0: tensor<1x8x24x64xf16>) -> tensor<1x8x1x1xf16> {
    %0 = IE.ReduceL2(%arg0) {axes_value = [2,3], keep_dims} : tensor<1x8x24x64xf16> -> tensor<1x8x1x1xf16>
    return %0 : tensor<1x8x1x1xf16>

// CHECK: IE.ReduceL2
}
