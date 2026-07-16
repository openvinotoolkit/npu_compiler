//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Inclusive cumsum along axis 1 for f32 constants: folded to const attribute.
// CHECK-LABEL: @CumSumConstFoldF32
func.func @CumSumConstFoldF32() -> tensor<1x4xf32> {
    %cst = const.Declare tensor<1x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>
    %result = IE.CumSum(%cst) {axis_value = 1 : i64} : tensor<1x4xf32> -> tensor<1x4xf32>
    return %result : tensor<1x4xf32>

    // CHECK:       [[FOLDED:%.+]] = const.Declare tensor<1x4xf32> =
    // CHECK-SAME:      dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf32>,
    // CHECK-SAME:      [#const.CumSum<1 : i64, false, false>]
    // CHECK-NOT:   IE.CumSum
    // CHECK:       return [[FOLDED]]
}

// -----

// Inclusive cumsum for f16 constants.
// CHECK-LABEL: @CumSumConstFoldF16
func.func @CumSumConstFoldF16() -> tensor<1x4xf16> {
    %cst = const.Declare tensor<1x4xf16> = dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf16>
    %result = IE.CumSum(%cst) {axis_value = 1 : i64} : tensor<1x4xf16> -> tensor<1x4xf16>
    return %result : tensor<1x4xf16>

    // CHECK:       [[FOLDED:%.+]] = const.Declare tensor<1x4xf16> =
    // CHECK-SAME:      dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf16>,
    // CHECK-SAME:      [#const.CumSum<1 : i64, false, false>]
    // CHECK-NOT:   IE.CumSum
    // CHECK:       return [[FOLDED]]
}

// -----

// Exclusive cumsum: current element not included.
// CHECK-LABEL: @CumSumConstFoldExclusive
func.func @CumSumConstFoldExclusive() -> tensor<1x4xf32> {
    %cst = const.Declare tensor<1x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>
    %result = IE.CumSum(%cst) {axis_value = 1 : i64, exclusive} : tensor<1x4xf32> -> tensor<1x4xf32>
    return %result : tensor<1x4xf32>

    // CHECK:       [[FOLDED:%.+]] = const.Declare tensor<1x4xf32> =
    // CHECK-SAME:      dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf32>,
    // CHECK-SAME:      [#const.CumSum<1 : i64, true, false>]
    // CHECK-NOT:   IE.CumSum
    // CHECK:       return [[FOLDED]]
}

// -----

// Reverse cumsum: accumulate from the end.
// CHECK-LABEL: @CumSumConstFoldReverse
func.func @CumSumConstFoldReverse() -> tensor<1x4xf32> {
    %cst = const.Declare tensor<1x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>
    %result = IE.CumSum(%cst) {axis_value = 1 : i64, reverse} : tensor<1x4xf32> -> tensor<1x4xf32>
    return %result : tensor<1x4xf32>

    // CHECK:       [[FOLDED:%.+]] = const.Declare tensor<1x4xf32> =
    // CHECK-SAME:      dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf32>,
    // CHECK-SAME:      [#const.CumSum<1 : i64, false, true>]
    // CHECK-NOT:   IE.CumSum
    // CHECK:       return [[FOLDED]]
}

// -----

// Exclusive + reverse cumsum.
// CHECK-LABEL: @CumSumConstFoldExclusiveReverse
func.func @CumSumConstFoldExclusiveReverse() -> tensor<1x4xf32> {
    %cst = const.Declare tensor<1x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>
    %result = IE.CumSum(%cst) {axis_value = 1 : i64, exclusive, reverse} : tensor<1x4xf32> -> tensor<1x4xf32>
    return %result : tensor<1x4xf32>

    // CHECK:       [[FOLDED:%.+]] = const.Declare tensor<1x4xf32> =
    // CHECK-SAME:      dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf32>,
    // CHECK-SAME:      [#const.CumSum<1 : i64, true, true>]
    // CHECK-NOT:   IE.CumSum
    // CHECK:       return [[FOLDED]]
}

// -----

// Not folded: dynamic (non-const) input.
// CHECK-LABEL: func.func @CumSumNotFoldDynamic(
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4xf32>
func.func @CumSumNotFoldDynamic(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %result = IE.CumSum(%arg0) {axis_value = 1 : i64} : tensor<1x4xf32> -> tensor<1x4xf32>
    return %result : tensor<1x4xf32>

    // CHECK: [[OUT:%.+]] = IE.CumSum([[INPUT]]) {axis_value = 1 : i64} : tensor<1x4xf32> -> tensor<1x4xf32>
    // CHECK: return [[OUT]]
}

// -----

// End-to-end: axis provided as const operand (canonicalizer converts to axis_value, then fold applies).
// CHECK-LABEL: @CumSumConstAxisOperand
func.func @CumSumConstAxisOperand() -> tensor<1x4xf32> {
    %cst = const.Declare tensor<1x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>
    %axis = const.Declare tensor<si64> = dense<1> : tensor<si64>
    %result = IE.CumSum(%cst, %axis) : tensor<1x4xf32>, tensor<si64> -> tensor<1x4xf32>
    return %result : tensor<1x4xf32>

    // CHECK:       [[FOLDED:%.+]] = const.Declare tensor<1x4xf32> =
    // CHECK-SAME:      dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf32>,
    // CHECK-SAME:      [#const.CumSum<1 : i64, false, false>]
    // CHECK-NOT:   IE.CumSum
    // CHECK:       return [[FOLDED]]
}

// -----

// Inclusive cumsum for i32 constants.
// CHECK-LABEL: @CumSumConstFoldI32
func.func @CumSumConstFoldI32() -> tensor<1x4xi32> {
    %cst = const.Declare tensor<1x4xi32> = dense<[[1, 2, 3, 4]]> : tensor<1x4xi32>
    %result = IE.CumSum(%cst) {axis_value = 1 : i64} : tensor<1x4xi32> -> tensor<1x4xi32>
    return %result : tensor<1x4xi32>

    // CHECK:       [[FOLDED:%.+]] = const.Declare tensor<1x4xi32> =
    // CHECK-SAME:      dense<{{\[\[}}1, 2, 3, 4]]> : tensor<1x4xi32>,
    // CHECK-SAME:      [#const.CumSum<1 : i64, false, false>]
    // CHECK-NOT:   IE.CumSum
    // CHECK:       return [[FOLDED]]
}

// -----

// Inclusive cumsum for i64 constants.
// CHECK-LABEL: @CumSumConstFoldI64
func.func @CumSumConstFoldI64() -> tensor<1x4xi64> {
    %cst = const.Declare tensor<1x4xi64> = dense<[[1, 2, 3, 4]]> : tensor<1x4xi64>
    %result = IE.CumSum(%cst) {axis_value = 1 : i64} : tensor<1x4xi64> -> tensor<1x4xi64>
    return %result : tensor<1x4xi64>

    // CHECK:       [[FOLDED:%.+]] = const.Declare tensor<1x4xi64> =
    // CHECK-SAME:      dense<{{\[\[}}1, 2, 3, 4]]> : tensor<1x4xi64>,
    // CHECK-SAME:      [#const.CumSum<1 : i64, false, false>]
    // CHECK-NOT:   IE.CumSum
    // CHECK:       return [[FOLDED]]
}
