//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FoldGatherElementsWithConstInputs
func.func @FoldGatherElementsWithConstInputs() -> tensor<2x2xf32> {
    %cst_input = const.Declare tensor<2x3xf32> = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [-1.000000e+00, -2.000000e+00, -3.000000e+00]]> : tensor<2x3xf32>
    %cst_indices = const.Declare tensor<2x2xsi64> = dense<[[2, 0], [1, 2]]> : tensor<2x2xsi64>
    %0 = IE.GatherElements(%cst_input, %cst_indices) {axis = 1 : i64} : tensor<2x3xf32>, tensor<2x2xsi64> -> tensor<2x2xf32>

    return %0 : tensor<2x2xf32>

    // CHECK: [[RESULT:%.+]] = const.Declare tensor<2x2xf32> =
    // CHECK-SAME{LITERAL}:        dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [-1.000000e+00, -2.000000e+00, -3.000000e+00]]> : tensor<2x3xf32>,
    // CHECK-SAME{LITERAL}:        [#const.GatherElements<1 : i64, dense<[[2, 0], [1, 2]]> : tensor<2x2xsi64>>]
    // CHECK: return [[RESULT]] : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @FoldGatherElementsWithI32Indices
func.func @FoldGatherElementsWithI32Indices() -> tensor<2x2xf32> {
    // Regression test: indices narrower than i64 (e.g. i32) must be sign-extended before folding,
    // otherwise GatherElementsAttr::verify() rejects the resulting constant.
    %cst_input = const.Declare tensor<2x3xf32> = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [-1.000000e+00, -2.000000e+00, -3.000000e+00]]> : tensor<2x3xf32>
    %cst_indices = const.Declare tensor<2x2xsi32> = dense<[[2, -1], [1, 2]]> : tensor<2x2xsi32>
    %0 = IE.GatherElements(%cst_input, %cst_indices) {axis = 1 : i64} : tensor<2x3xf32>, tensor<2x2xsi32> -> tensor<2x2xf32>

    return %0 : tensor<2x2xf32>

    // CHECK: [[RESULT:%.+]] = const.Declare tensor<2x2xf32> =
    // CHECK-SAME{LITERAL}:        dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [-1.000000e+00, -2.000000e+00, -3.000000e+00]]> : tensor<2x3xf32>,
    // CHECK-SAME{LITERAL}:        [#const.GatherElements<1 : i64, dense<[[2, -1], [1, 2]]> : tensor<2x2xsi64>>]
    // CHECK: return [[RESULT]] : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @FoldGatherElementsWithNegativeAxis
func.func @FoldGatherElementsWithNegativeAxis() -> tensor<2x2xf32> {
    %cst_input = const.Declare tensor<2x3xf32> = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [-1.000000e+00, -2.000000e+00, -3.000000e+00]]> : tensor<2x3xf32>
    %cst_indices = const.Declare tensor<2x2xsi64> = dense<[[2, 0], [1, 2]]> : tensor<2x2xsi64>
    %0 = IE.GatherElements(%cst_input, %cst_indices) {axis = -1 : i64} : tensor<2x3xf32>, tensor<2x2xsi64> -> tensor<2x2xf32>

    return %0 : tensor<2x2xf32>

    // CHECK: [[RESULT:%.+]] = const.Declare tensor<2x2xf32> =
    // CHECK-SAME{LITERAL}:        dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [-1.000000e+00, -2.000000e+00, -3.000000e+00]]> : tensor<2x3xf32>,
    // CHECK-SAME{LITERAL}:        [#const.GatherElements<1 : i64, dense<[[2, 0], [1, 2]]> : tensor<2x2xsi64>>]
    // CHECK: return [[RESULT]] : tensor<2x2xf32>
}
