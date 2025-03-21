//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<1x8x4x4xf32> {
    %0 = const.Declare tensor<1x8x4x4xf32> = dense<5.0> : tensor<1x8x4x4xf32>
    %1 = const.Declare tensor<1x8x4x4xf32> = dense<0.0> : tensor<1x8x4x4xf32>
    %2 = IE.Add(%0, %1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x8x4x4xf32>, tensor<1x8x4x4xf32> -> tensor<1x8x4x4xf32>
    return %2 : tensor<1x8x4x4xf32>

    // CHECK-DAG:   [[VAL0:%.+]] = const.Declare tensor<1x8x4x4xf32> = dense<5.000000e+00> : tensor<1x8x4x4xf32>
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   IE.Add
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: func.func @NotFoldConst(
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1xf32>
func.func @NotFoldConst(%arg0: tensor<1xf32>) -> tensor<1x10x1xf32> {
  %0 = const.Declare tensor<1x10x1xf32> = dense<0.0> : tensor<1x10x1xf32>
  %1 = IE.Add(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x10x1xf32> -> tensor<1x10x1xf32>
  return %1 : tensor<1x10x1xf32>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x10x1xf32> = dense<0.000000e+00> : tensor<1x10x1xf32>
  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x10x1xf32> -> tensor<1x10x1xf32>
  // CHECK:       return [[ADD]]
}
