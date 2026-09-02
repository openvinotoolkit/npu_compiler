//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FoldReduceL1
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceL1(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceL1(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceL1
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceL2
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceL2(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceL2(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceL2
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceLogicalAnd
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceLogicalAnd(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceLogicalAnd(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceLogicalAnd
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceLogicalOr
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceLogicalOr(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceLogicalOr(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceLogicalOr
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceMax
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceMax(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceMax(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceMax
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceMean
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceMean(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceMean
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @DoNotFoldPaddedReduceMean
func.func @DoNotFoldPaddedReduceMean(%arg0: tensor<1x16x4x2xf16>) -> tensor<1x16x4x2xf16> {
    %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims, input_padding = [0, 4, 0, 0], output_padding = [0, 15, 0, 0]} : tensor<1x16x4x2xf16> -> tensor<1x16x4x2xf16>
    return %0 : tensor<1x16x4x2xf16>

    // CHECK:   [[REDUCE:%.+]] = IE.ReduceMean
    // CHECK:   return [[REDUCE]]
}

// -----

// CHECK-LABEL: @FoldReduceMin
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceMin(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceMin(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceMin
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceProd
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceProd(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceProd(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceProd
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @FoldReduceSum
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x2xf16>)
func.func @FoldReduceSum(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceSum
    // CHECK:       return [[ARG_0]]
}

// -----

// CHECK-LABEL: @DoNotFoldPaddedReduceSum
func.func @DoNotFoldPaddedReduceSum(%arg0: tensor<1x16x4x2xf16>) -> tensor<1x16x4x2xf16> {
    %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims, input_padding = [0, 4, 0, 0], output_padding = [0, 15, 0, 0]} : tensor<1x16x4x2xf16> -> tensor<1x16x4x2xf16>
    return %0 : tensor<1x16x4x2xf16>

    // CHECK:   [[REDUCE:%.+]] = IE.ReduceSum
    // CHECK:   return [[REDUCE]]
}
