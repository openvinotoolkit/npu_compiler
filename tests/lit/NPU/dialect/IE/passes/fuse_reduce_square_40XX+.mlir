//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-reduce-square --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FuseReduceSquare
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x32x96xf32>)
func.func @FuseReduceSquare(%arg0: tensor<1x32x32x96xf32>) -> tensor<1x32x32x1xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x32x96xf32>
    %1 = IE.ReduceMean(%0) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf32> -> tensor<1x32x32x1xf32>
    %2 = IE.Sqrt(%1) : tensor<1x32x32x1xf32> -> tensor<1x32x32x1xf32>
    return %2 : tensor<1x32x32x1xf32>

    // CHECK: [[ReduceSquare:%.+]] = IE.ReduceSquare([[ARG0]]) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf32> -> tensor<1x32x32x1xf32>
    // CHECK: return [[ReduceSquare]] : tensor<1x32x32x1xf32>
}

// -----

// CHECK-LABEL: @FuseReduceSquareMultiply
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x32x96xf32>)
func.func @FuseReduceSquareMultiply(%arg0: tensor<1x32x32x96xf32>) -> tensor<1x32x32x1xf32> {
    %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf32>, tensor<1x32x32x96xf32> -> tensor<1x32x32x96xf32>
    %1 = IE.ReduceMean(%0) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf32> -> tensor<1x32x32x1xf32>
    %2 = IE.Sqrt(%1) : tensor<1x32x32x1xf32> -> tensor<1x32x32x1xf32>
    return %2 : tensor<1x32x32x1xf32>

    // CHECK: [[ReduceSquare:%.+]] = IE.ReduceSquare([[ARG0]]) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf32> -> tensor<1x32x32x1xf32>
    // CHECK: return [[ReduceSquare]] : tensor<1x32x32x1xf32>
}

// -----

// CHECK-LABEL: @FuseReduceSquareWithAdd
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x32x96xf16>)
func.func @FuseReduceSquareWithAdd(%arg0: tensor<1x32x32x96xf16>) -> tensor<1x32x32x1xf16> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<3.0> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x96xf16>
    %1 = IE.ReduceMean(%0) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf16> -> tensor<1x32x32x1xf16>
    %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x1xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x1xf16>
    %3 = IE.Sqrt(%2) : tensor<1x32x32x1xf16> -> tensor<1x32x32x1xf16>
    return %3 : tensor<1x32x32x1xf16>

    // CHECK: [[ReduceSquare:%.+]] = IE.ReduceSquare([[ARG0]]) {axes_value = [3], epsilon = 3.000000e+00 : f64, keep_dims} : tensor<1x32x32x96xf16> -> tensor<1x32x32x1xf16>
    // CHECK: return [[ReduceSquare]] : tensor<1x32x32x1xf16>
}

// -----

// CHECK-LABEL: @FuseReduceSquareWithPowerHalf
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x32x96xf16>)
func.func @FuseReduceSquareWithPowerHalf(%arg0: tensor<1x32x32x96xf16>) -> tensor<1x32x32x1xf16> {
    %cst_pow2 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %cst_pow_half = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%arg0, %cst_pow2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x96xf16>
    %1 = IE.ReduceMean(%0) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf16> -> tensor<1x32x32x1xf16>
    %2 = IE.Power(%1, %cst_pow_half) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x1xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x1xf16>
    return %2 : tensor<1x32x32x1xf16>

    // CHECK: [[ReduceSquare:%.+]] = IE.ReduceSquare([[ARG0]]) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf16> -> tensor<1x32x32x1xf16>
    // CHECK: return [[ReduceSquare]] : tensor<1x32x32x1xf16>
}

// -----

// CHECK-LABEL: @FuseReduceSquareWithAddAndPowerHalf
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x32x96xf16>)
func.func @FuseReduceSquareWithAddAndPowerHalf(%arg0: tensor<1x32x32x96xf16>) -> tensor<1x32x32x1xf16> {
    %cst_eps = const.Declare tensor<1x1x1x1xf32> = dense<3.0> : tensor<1x1x1x1xf32>
    %cst_pow2 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %cst_pow_half = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%arg0, %cst_pow2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x96xf16>
    %1 = IE.ReduceMean(%0) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf16> -> tensor<1x32x32x1xf16>
    %2 = IE.Add(%1, %cst_eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x1xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x1xf16>
    %3 = IE.Power(%2, %cst_pow_half) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x1xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x1xf16>
    return %3 : tensor<1x32x32x1xf16>

    // CHECK: [[ReduceSquare:%.+]] = IE.ReduceSquare([[ARG0]]) {axes_value = [3], epsilon = 3.000000e+00 : f64, keep_dims} : tensor<1x32x32x96xf16> -> tensor<1x32x32x1xf16>
    // CHECK: return [[ReduceSquare]] : tensor<1x32x32x1xf16>
}

// -----

// CHECK-LABEL: @NoFuseReduceSquareEpsilonNonInnermost
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x32x96xf16>)
func.func @NoFuseReduceSquareEpsilonNonInnermost(%arg0: tensor<1x32x32x96xf16>) -> tensor<1x1x32x96xf16> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<3.0> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x96xf16>
    %1 = IE.ReduceMean(%0) {axes_value = [1], keep_dims} : tensor<1x32x32x96xf16> -> tensor<1x1x32x96xf16>
    %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x32x96xf16>, tensor<1x1x1x1xf32> -> tensor<1x1x32x96xf16>
    %3 = IE.Sqrt(%2) : tensor<1x1x32x96xf16> -> tensor<1x1x32x96xf16>
    return %3 : tensor<1x1x32x96xf16>

    // CHECK-NOT: IE.ReduceSquare
    // CHECK: return
}

// -----

// CHECK-LABEL: @NoFuseReduceSquareNonConstEpsilon
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x32x96xf16>, [[ARG1:%.+]]: tensor<1x32x32x1xf16>)
func.func @NoFuseReduceSquareNonConstEpsilon(%arg0: tensor<1x32x32x96xf16>, %arg1: tensor<1x32x32x1xf16>) -> tensor<1x32x32x1xf16> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf16>, tensor<1x1x1x1xf32> -> tensor<1x32x32x96xf16>
    %1 = IE.ReduceMean(%0) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf16> -> tensor<1x32x32x1xf16>
    %2 = IE.Add(%1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x1xf16>, tensor<1x32x32x1xf16> -> tensor<1x32x32x1xf16>
    %3 = IE.Sqrt(%2) : tensor<1x32x32x1xf16> -> tensor<1x32x32x1xf16>
    return %3 : tensor<1x32x32x1xf16>

    // CHECK-NOT: IE.ReduceSquare
    // CHECK: return
}

// -----

// CHECK-LABEL: @FuseReduceSumSquare
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x512x18x80xf32>)
func.func @FuseReduceSumSquare(%arg0: tensor<1x512x18x80xf32>) -> tensor<1x512x18x1xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x18x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x18x80xf32>
    %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x512x18x80xf32> -> tensor<1x512x18x1xf32>
    %2 = IE.Sqrt(%1) : tensor<1x512x18x1xf32> -> tensor<1x512x18x1xf32>
    return %2 : tensor<1x512x18x1xf32>

    // CHECK: [[ReduceL2:%.+]] = IE.ReduceL2([[ARG0]]) {axes_value = [3], keep_dims} : tensor<1x512x18x80xf32> -> tensor<1x512x18x1xf32>
    // CHECK: return [[ReduceL2]] : tensor<1x512x18x1xf32>
}

// -----

// CHECK-LABEL: @FuseReduceSumSquareFP16
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x512x18x80xf16>)
func.func @FuseReduceSumSquareFP16(%arg0: tensor<1x512x18x80xf16>) -> tensor<1x512x18x1xf16> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x18x80xf16>, tensor<1x1x1x1xf32> -> tensor<1x512x18x80xf16>
    %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x512x18x80xf16> -> tensor<1x512x18x1xf16>
    %2 = IE.Sqrt(%1) : tensor<1x512x18x1xf16> -> tensor<1x512x18x1xf16>
    return %2 : tensor<1x512x18x1xf16>

    // CHECK: [[ReduceL2:%.+]] = IE.ReduceL2([[ARG0]]) {axes_value = [3], keep_dims} : tensor<1x512x18x80xf16> -> tensor<1x512x18x1xf16>
    // CHECK: return [[ReduceL2]] : tensor<1x512x18x1xf16>
}

// -----

// CHECK-LABEL: @NoFuseReduceSumSquareWithAdd
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x512x18x80xf32>)
func.func @NoFuseReduceSumSquareWithAdd(%arg0: tensor<1x512x18x80xf32>) -> tensor<1x512x18x1xf32> {
    %eps = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
    %pow2 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%arg0, %pow2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x18x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x18x80xf32>
    %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x512x18x80xf32> -> tensor<1x512x18x1xf32>
    %2 = IE.Add(%1, %eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x18x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x18x1xf32>
    %3 = IE.Sqrt(%2) : tensor<1x512x18x1xf32> -> tensor<1x512x18x1xf32>
    return %3 : tensor<1x512x18x1xf32>

    // CHECK-NOT: IE.ReduceSquare
    // CHECK: return
}

// -----

// CHECK-LABEL: @FuseReduceSumSquareF16WithPowerHalf
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x512x18x80xf16>)
func.func @FuseReduceSumSquareF16WithPowerHalf(%arg0: tensor<1x512x18x80xf16>) -> tensor<1x512x18x1xf16> {
    %cst_pow2 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %cst_pow_half = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%arg0, %cst_pow2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x18x80xf16>, tensor<1x1x1x1xf32> -> tensor<1x512x18x80xf16>
    %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x512x18x80xf16> -> tensor<1x512x18x1xf16>
    %2 = IE.Power(%1, %cst_pow_half) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x18x1xf16>, tensor<1x1x1x1xf32> -> tensor<1x512x18x1xf16>
    return %2 : tensor<1x512x18x1xf16>

    // CHECK: [[ReduceL2:%.+]] = IE.ReduceL2([[ARG0]]) {axes_value = [3], keep_dims} : tensor<1x512x18x80xf16> -> tensor<1x512x18x1xf16>
    // CHECK: return [[ReduceL2]] : tensor<1x512x18x1xf16>
}
