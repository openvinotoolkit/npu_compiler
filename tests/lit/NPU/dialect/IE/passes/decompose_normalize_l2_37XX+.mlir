//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-normalize-l2 %s | FileCheck %s
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
