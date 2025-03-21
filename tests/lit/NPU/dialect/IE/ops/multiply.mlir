//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<1x8x4x4xf32> {
    %0 = const.Declare tensor<1x8x4x4xf32> = dense<5.0> : tensor<1x8x4x4xf32>
    %1 = const.Declare tensor<1x8x4x4xf32> = dense<1.0> : tensor<1x8x4x4xf32>
    %2 = IE.Multiply(%0, %1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x8x4x4xf32>, tensor<1x8x4x4xf32> -> tensor<1x8x4x4xf32>
    return %2 : tensor<1x8x4x4xf32>

    // CHECK-DAG:   [[VAL0:%.+]] = const.Declare tensor<1x8x4x4xf32> = dense<5.000000e+00> : tensor<1x8x4x4xf32>
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   IE.Multiply
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: func.func @NotFoldConst(
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1xf32>
func.func @NotFoldConst(%arg0: tensor<1xf32>) -> tensor<1x10x1xf32> {
  %0 = const.Declare tensor<1x10x1xf32> = dense<1.0> : tensor<1x10x1xf32>
  %1 = IE.Multiply(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x10x1xf32> -> tensor<1x10x1xf32>
  return %1 : tensor<1x10x1xf32>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x10x1xf32> = dense<1.000000e+00> : tensor<1x10x1xf32>
  // CHECK:       [[MUL:%.+]] = IE.Multiply([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x10x1xf32> -> tensor<1x10x1xf32>
  // CHECK:       return [[MUL]]
}

// -----

// CHECK-LABEL: func.func @MultiplyBroadcastable(
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x1x1x128xf16>
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x25x36x1xf16>
func.func @MultiplyBroadcastable(%arg0: tensor<1x1x1x128xf16>, %arg1: tensor<1x25x36x1xf16>) -> tensor<1x25x36x128xf16> {
  %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x128xf16>, tensor<1x25x36x1xf16> -> tensor<1x25x36x128xf16>
  return %0 : tensor<1x25x36x128xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x128xf16>, tensor<1x25x36x1xf16> -> tensor<1x25x36x128xf16>
  // CHECK-NOT:   IE.Multiply
  // CHECK:       return [[VAL0]]
}
