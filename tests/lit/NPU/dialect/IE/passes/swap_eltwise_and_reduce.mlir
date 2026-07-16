//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --swap-eltwise-and-reduce %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @SwapMulReduceSum
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x64x5x5x1x1xf16>, [[ARG1:%.+]]: tensor<1x64x1x5x64x128xf16>
func.func @SwapMulReduceSum(%arg0: tensor<1x64x5x5x1x1xf16>, %arg1: tensor<1x64x1x5x64x128xf16>) -> tensor<1x64x1x5x64x128xf16> {
    %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x64x5x5x1x1xf16>, tensor<1x64x1x5x64x128xf16> -> tensor<1x64x5x5x64x128xf16>
    %reduce = IE.ReduceSum(%mul) {axes_value = [2], keep_dims} :
        tensor<1x64x5x5x64x128xf16> -> tensor<1x64x1x5x64x128xf16>
    return %reduce : tensor<1x64x1x5x64x128xf16>

    // CHECK:     [[REDUCE:%.+]] = IE.ReduceSum([[ARG0]]) {axes_value = [2], keep_dims}
    // CHECK-SAME:    tensor<1x64x5x5x1x1xf16> -> tensor<1x64x1x5x1x1xf16>
    // CHECK:     [[MUL:%.+]] = IE.Multiply([[REDUCE]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x64x1x5x1x1xf16>, tensor<1x64x1x5x64x128xf16> -> tensor<1x64x1x5x64x128xf16>
    // CHECK:     return [[MUL]] : tensor<1x64x1x5x64x128xf16>
}

// -----

// CHECK-LABEL: @SwapMulReduceSumKeepDimsFalse
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x64x5x5x1x1xf16>, [[ARG1:%.+]]: tensor<1x64x1x5x64x128xf16>
func.func @SwapMulReduceSumKeepDimsFalse(%arg0: tensor<1x64x5x5x1x1xf16>, %arg1: tensor<1x64x1x5x64x128xf16>) -> tensor<1x64x5x64x128xf16> {
    %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x64x5x5x1x1xf16>, tensor<1x64x1x5x64x128xf16> -> tensor<1x64x5x5x64x128xf16>
    %reduce = IE.ReduceSum(%mul) {axes_value = [2]} :
        tensor<1x64x5x5x64x128xf16> -> tensor<1x64x5x64x128xf16>
    return %reduce : tensor<1x64x5x64x128xf16>

    // CHECK:     [[RS:%.+]] = IE.ReduceSum([[ARG0]]) {axes_value = [2], keep_dims}
    // CHECK-SAME:    tensor<1x64x5x5x1x1xf16> -> tensor<1x64x1x5x1x1xf16>
    // CHECK:     [[MUL:%.+]] = IE.Multiply([[RS]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x64x1x5x1x1xf16>, tensor<1x64x1x5x64x128xf16> -> tensor<1x64x1x5x64x128xf16>
    // CHECK:     [[SQUEEZE:%.+]] = IE.Reshape([[MUL]]) {shape_value = [1, 64, 5, 64, 128]}
    // CHECK-SAME:    tensor<1x64x1x5x64x128xf16> -> tensor<1x64x5x64x128xf16>
    // CHECK:     return [[SQUEEZE]] : tensor<1x64x5x64x128xf16>
}

// -----

// CHECK-LABEL: @NoSwapMulReduceSumBothNonBroadcast
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x64x5x5xf16>, [[ARG1:%.+]]: tensor<1x64x5x5xf16>
func.func @NoSwapMulReduceSumBothNonBroadcast(%arg0: tensor<1x64x5x5xf16>, %arg1: tensor<1x64x5x5xf16>) -> tensor<1x64x1x5xf16> {
    %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x64x5x5xf16>, tensor<1x64x5x5xf16> -> tensor<1x64x5x5xf16>
    %reduce = IE.ReduceSum(%mul) {axes_value = [2], keep_dims} :
        tensor<1x64x5x5xf16> -> tensor<1x64x1x5xf16>
    return %reduce : tensor<1x64x1x5xf16>

    // CHECK:     [[MUL:%.+]] = IE.Multiply([[ARG0]], [[ARG1]])
    // CHECK-SAME:    tensor<1x64x5x5xf16>, tensor<1x64x5x5xf16> -> tensor<1x64x5x5xf16>
    // CHECK:     [[REDUCE:%.+]] = IE.ReduceSum([[MUL]])
    // CHECK-SAME:    {axes_value = [2], keep_dims}
    // CHECK:     return [[REDUCE]] : tensor<1x64x1x5xf16>
}

// -----

// CHECK-LABEL: @SwapMulReduceMean
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x64x5x5x1x1xf16>, [[ARG1:%.+]]: tensor<1x64x1x5x64x128xf16>
func.func @SwapMulReduceMean(%arg0: tensor<1x64x5x5x1x1xf16>, %arg1: tensor<1x64x1x5x64x128xf16>) -> tensor<1x64x1x5x64x128xf16> {
    %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x64x5x5x1x1xf16>, tensor<1x64x1x5x64x128xf16> -> tensor<1x64x5x5x64x128xf16>
    %reduce = IE.ReduceMean(%mul) {axes_value = [2], keep_dims} :
        tensor<1x64x5x5x64x128xf16> -> tensor<1x64x1x5x64x128xf16>
    return %reduce : tensor<1x64x1x5x64x128xf16>

    // CHECK:     [[REDUCE:%.+]] = IE.ReduceMean([[ARG0]]) {axes_value = [2], keep_dims}
    // CHECK-SAME:    tensor<1x64x5x5x1x1xf16> -> tensor<1x64x1x5x1x1xf16>
    // CHECK:     [[MUL:%.+]] = IE.Multiply([[REDUCE]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x64x1x5x1x1xf16>, tensor<1x64x1x5x64x128xf16> -> tensor<1x64x1x5x64x128xf16>
    // CHECK:     return [[MUL]] : tensor<1x64x1x5x64x128xf16>
}

// -----

// CHECK-LABEL: @SwapMulReduceSumInput1Broadcast
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x64x1x5x64x128xf16>, [[ARG1:%.+]]: tensor<1x64x5x5x1x1xf16>
func.func @SwapMulReduceSumInput1Broadcast(%arg0: tensor<1x64x1x5x64x128xf16>, %arg1: tensor<1x64x5x5x1x1xf16>) -> tensor<1x64x1x5x64x128xf16> {
    %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x64x1x5x64x128xf16>, tensor<1x64x5x5x1x1xf16> -> tensor<1x64x5x5x64x128xf16>
    %reduce = IE.ReduceSum(%mul) {axes_value = [2], keep_dims} :
        tensor<1x64x5x5x64x128xf16> -> tensor<1x64x1x5x64x128xf16>
    return %reduce : tensor<1x64x1x5x64x128xf16>

    // CHECK:     [[REDUCE:%.+]] = IE.ReduceSum([[ARG1]]) {axes_value = [2], keep_dims}
    // CHECK-SAME:    tensor<1x64x5x5x1x1xf16> -> tensor<1x64x1x5x1x1xf16>
    // CHECK:     [[MUL:%.+]] = IE.Multiply([[ARG0]], [[REDUCE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x64x1x5x64x128xf16>, tensor<1x64x1x5x1x1xf16> -> tensor<1x64x1x5x64x128xf16>
    // CHECK:     return [[MUL]] : tensor<1x64x1x5x64x128xf16>
}

// -----

// CHECK-LABEL: @SwapAddReduceMean
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x64x5x5x1x1xf16>, [[ARG1:%.+]]: tensor<1x64x1x5x64x128xf16>
func.func @SwapAddReduceMean(%arg0: tensor<1x64x5x5x1x1xf16>, %arg1: tensor<1x64x1x5x64x128xf16>) -> tensor<1x64x1x5x64x128xf16> {
    %add = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x64x5x5x1x1xf16>, tensor<1x64x1x5x64x128xf16> -> tensor<1x64x5x5x64x128xf16>
    %reduce = IE.ReduceMean(%add) {axes_value = [2], keep_dims} :
        tensor<1x64x5x5x64x128xf16> -> tensor<1x64x1x5x64x128xf16>
    return %reduce : tensor<1x64x1x5x64x128xf16>

    // CHECK:     [[REDUCE:%.+]] = IE.ReduceMean([[ARG0]]) {axes_value = [2], keep_dims}
    // CHECK-SAME:    tensor<1x64x5x5x1x1xf16> -> tensor<1x64x1x5x1x1xf16>
    // CHECK:     [[ADD:%.+]] = IE.Add([[REDUCE]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x64x1x5x1x1xf16>, tensor<1x64x1x5x64x128xf16> -> tensor<1x64x1x5x64x128xf16>
    // CHECK:     return [[ADD]] : tensor<1x64x1x5x64x128xf16>
}
