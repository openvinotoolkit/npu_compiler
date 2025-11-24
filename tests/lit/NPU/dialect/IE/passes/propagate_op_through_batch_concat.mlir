//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-op-through-batch-concat %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @PropagateSoftmaxThroughBatchUnrolledMatmul
func.func @PropagateSoftmaxThroughBatchUnrolledMatmul(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.Reshape(%1) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %3 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %5 = IE.Concat(%2, %4) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    %6 = IE.SoftMax(%5) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    return %6 : tensor<2x16x2xf32>

    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MATMUL_1]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[MATMUL_2]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[SOFTMAX_1:%.+]] = IE.SoftMax([[RESHAPE_1]]) {axisInd = -1 : i64} : tensor<1x16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[SOFTMAX_2:%.+]] = IE.SoftMax([[RESHAPE_2]]) {axisInd = -1 : i64} : tensor<1x16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SOFTMAX_1]], [[SOFTMAX_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       return [[CONCAT]] : tensor<2x16x2xf32>
}

// -----

// CHECK-LABEL: @NoPropagateSoftmaxForConcatHasNonMatmulInput
func.func @NoPropagateSoftmaxForConcatHasNonMatmulInput(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<32x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %cst_1 = const.Declare tensor<16x2xf32> = dense<1.000000e+00> : tensor<16x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.Add(%arg1, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<16x2xf32>
    %3 = IE.Concat(%1, %2) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    %4 = IE.SoftMax(%3) {axisInd = -1 : i64} : tensor<32x2xf32> -> tensor<32x2xf32>
    return %4 : tensor<32x2xf32>

    // CHECK:       [[MATMUL:%.+]] = IE.MatMul
    // CHECK:       [[ADD:%.+]] = IE.Add
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[MATMUL]], [[ADD]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[CONCAT]]) {axisInd = -1 : i64} : tensor<32x2xf32> -> tensor<32x2xf32>
    // CHECK:       return [[SOFTMAX]] : tensor<32x2xf32>
}

// -----

// CHECK-LABEL: @NoPropagateForConcatWithoutPerAxisAttr
func.func @NoPropagateForConcatWithoutPerAxisAttr(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.Reshape(%1) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %3 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %5 = IE.Concat(%2, %4) {static_offsets = [[0, 0, 0], [1, 0, 0]]} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    %6 = IE.SoftMax(%5) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    return %6 : tensor<2x16x2xf32>

    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MATMUL_1]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[MATMUL_2]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_1]], [[RESHAPE_2]]) {static_offsets = {{\[\[}}0, 0, 0], [1, 0, 0]]} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[CONCAT]]) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       return [[SOFTMAX]] : tensor<2x16x2xf32>
}

// -----

// CHECK-LABEL: @NoPropagateWhenAxisConflict
func.func @NoPropagateWhenAxisConflict(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.Reshape(%1) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %3 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %5 = IE.Concat(%2, %4) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    %6 = IE.SoftMax(%5) {axisInd = 0 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    return %6 : tensor<2x16x2xf32>

    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MATMUL_1]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[MATMUL_2]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_1]], [[RESHAPE_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[CONCAT]]) {axisInd = 0 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       return [[SOFTMAX]] : tensor<2x16x2xf32>
}

// -----

// CHECK-LABEL: @PropagateConstantInputAddSoftmaxThroughBatchUnrolledMatmul
func.func @PropagateConstantInputAddSoftmaxThroughBatchUnrolledMatmul(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %cst_1 = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.Reshape(%1) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %3 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %5 = IE.Concat(%2, %4) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    %6 = IE.Add(%5, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x16x2xf32>, tensor<1x1x1xf32> -> tensor<2x16x2xf32>
    %7 = IE.SoftMax(%6) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    return %7 : tensor<2x16x2xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MATMUL_1]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[MATMUL_2]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[ADD_1:%.+]] = IE.Add([[RESHAPE_1]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x2xf32>, tensor<1x1x1xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[SOFTMAX_1:%.+]] = IE.SoftMax([[ADD_1]]) {axisInd = -1 : i64} : tensor<1x16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[ADD_2:%.+]] = IE.Add([[RESHAPE_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x2xf32>, tensor<1x1x1xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[SOFTMAX_2:%.+]] = IE.SoftMax([[ADD_2]]) {axisInd = -1 : i64} : tensor<1x16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SOFTMAX_1]], [[SOFTMAX_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       return [[CONCAT]] : tensor<2x16x2xf32>
}

// -----

// CHECK-LABEL: @PropagateNonConstantInputAddSoftmaxThroughBatchUnrolledMatmul
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x1024x1024xf32>
func.func @PropagateNonConstantInputAddSoftmaxThroughBatchUnrolledMatmul(
            %arg0: tensor<1024x128xf32>,
            %arg1: tensor<1024x128xf32>,
            %arg2: tensor<1x1x1024x1024xf32>) -> tensor<1x2x1024x1024xf32> {
    %cst = const.Declare tensor<1024x128xf32> = dense<1.000000e+00> : tensor<1024x128xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<1024x128xf32>, tensor<1024x128xf32> -> tensor<1024x1024xf32>
    %1 = IE.MatMul(%arg1, %cst) {transpose_b} : tensor<1024x128xf32>, tensor<1024x128xf32> -> tensor<1024x1024xf32>
    %2 = IE.Reshape(%0) {shape_value = [1, 1, 1024, 1024]} : tensor<1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    %3 = IE.Reshape(%1) {shape_value = [1, 1, 1024, 1024]} : tensor<1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    %4 = IE.Concat(%2, %3) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32> -> tensor<1x2x1024x1024xf32>
    %5 = IE.Add(%4, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1024x1024xf32>, tensor<1x1x1024x1024xf32> -> tensor<1x2x1024x1024xf32>
    %6 = IE.SoftMax(%5) {axisInd = 3 : i64} : tensor<1x2x1024x1024xf32> -> tensor<1x2x1024x1024xf32>

    return %6 : tensor<1x2x1024x1024xf32>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1024x128xf32> = dense<1.000000e+00> : tensor<1024x128xf32>
    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MATMUL_1]]) {shape_value = [1, 1, 1024, 1024]} : tensor<1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[MATMUL_2]]) {shape_value = [1, 1, 1024, 1024]} : tensor<1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    // CHECK:       [[ADD_1:%.]] = IE.Add([[RESHAPE_1]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    // CHECK:       [[SOFTMAX_1:%.+]] = IE.SoftMax([[ADD_1]]) {axisInd = 3 : i64} : tensor<1x1x1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    // CHECK:       [[ADD_2:%.+]] = IE.Add([[RESHAPE_2]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    // CHECK:       [[SOFTMAX_2:%.+]] = IE.SoftMax([[ADD_2]]) {axisInd = 3 : i64} : tensor<1x1x1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SOFTMAX_1]], [[SOFTMAX_2]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32> -> tensor<1x2x1024x1024xf32>

    // CHECK:       return [[CONCAT]] : tensor<1x2x1024x1024xf32>
}

// -----

// CHECK-LABEL: @PropagateNonConstantInputSqueezeAddSoftmaxThroughBatchUnrolledMatmul
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x1024x1024xf32>
func.func @PropagateNonConstantInputSqueezeAddSoftmaxThroughBatchUnrolledMatmul(
            %arg0: tensor<1024x128xf32>,
            %arg1: tensor<1024x128xf32>,
            %arg2: tensor<1x1x1024x1024xf32>) -> tensor<1x2x1024x1024xf32> {
    %cst = const.Declare tensor<1024x128xf32> = dense<1.000000e+00> : tensor<1024x128xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<1024x128xf32>, tensor<1024x128xf32> -> tensor<1024x1024xf32>
    %1 = IE.MatMul(%arg1, %cst) {transpose_b} : tensor<1024x128xf32>, tensor<1024x128xf32> -> tensor<1024x1024xf32>
    %2 = IE.Reshape(%0) {shape_value = [1, 1, 1024, 1024]} : tensor<1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    %3 = IE.Reshape(%1) {shape_value = [1, 1, 1024, 1024]} : tensor<1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    %4 = IE.Concat(%2, %3) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32> -> tensor<1x2x1024x1024xf32>
    %5 = IE.Squeeze(%arg2) {axes_value = [0, 1]} : tensor<1x1x1024x1024xf32> -> tensor<1024x1024xf32>
    %6 = IE.Add(%4, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1024x1024xf32>, tensor<1024x1024xf32> -> tensor<1x2x1024x1024xf32>
    %7 = IE.SoftMax(%6) {axisInd = 3 : i64} : tensor<1x2x1024x1024xf32> -> tensor<1x2x1024x1024xf32>

    return %7 : tensor<1x2x1024x1024xf32>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1024x128xf32> = dense<1.000000e+00> : tensor<1024x128xf32>
    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MATMUL_1]]) {shape_value = [1, 1, 1024, 1024]} : tensor<1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[MATMUL_2]]) {shape_value = [1, 1, 1024, 1024]} : tensor<1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    // CHECK:       [[SQUEEZE:%.+]] = IE.Squeeze([[INPUT]]) {axes_value = [0, 1]} : tensor<1x1x1024x1024xf32> -> tensor<1024x1024xf32>
    // CHECK:       [[ADD_1:%.+]] = IE.Add([[RESHAPE_1]], [[SQUEEZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1024x1024xf32>, tensor<1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    // CHECK:       [[SOFTMAX_1:%.+]] = IE.SoftMax([[ADD_1]]) {axisInd = 3 : i64} : tensor<1x1x1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    // CHECK:       [[ADD_2:%.+]] = IE.Add([[RESHAPE_2]], [[SQUEEZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1024x1024xf32>, tensor<1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    // CHECK:       [[SOFTMAX_2:%.+]] = IE.SoftMax([[ADD_2]]) {axisInd = 3 : i64} : tensor<1x1x1024x1024xf32> -> tensor<1x1x1024x1024xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SOFTMAX_1]], [[SOFTMAX_2]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32> -> tensor<1x2x1024x1024xf32>

    // CHECK:       return [[CONCAT]] : tensor<1x2x1024x1024xf32>
}

// -----

// CHECK-LABEL: @NotPropagateAddSoftmaxThroughBatchUnrolledMatmulDueToNoDimLeftForSplit
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x1x1024xf32>
func.func @NotPropagateAddSoftmaxThroughBatchUnrolledMatmulDueToNoDimLeftForSplit(
            %arg0: tensor<1x128xf32>,
            %arg1: tensor<1x128xf32>,
            %arg2: tensor<1x1x1x1024xf32>) -> tensor<1x2x1x1024xf32> {
    %cst = const.Declare tensor<1024x128xf32> = dense<1.000000e+00> : tensor<1024x128xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<1x128xf32>, tensor<1024x128xf32> -> tensor<1x1024xf32>
    %1 = IE.MatMul(%arg1, %cst) {transpose_b} : tensor<1x128xf32>, tensor<1024x128xf32> -> tensor<1x1024xf32>
    %2 = IE.Reshape(%0) {shape_value = [1, 1, 1, 1024]} : tensor<1x1024xf32> -> tensor<1x1x1x1024xf32>
    %3 = IE.Reshape(%1) {shape_value = [1, 1, 1, 1024]} : tensor<1x1024xf32> -> tensor<1x1x1x1024xf32>
    %4 = IE.Concat(%2, %3) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32> -> tensor<1x2x1x1024xf32>
    %5 = IE.Add(%4, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x1024xf32>, tensor<1x1x1x1024xf32> -> tensor<1x2x1x1024xf32>
    %6 = IE.SoftMax(%5) {axisInd = 3 : i64} : tensor<1x2x1x1024xf32> -> tensor<1x2x1x1024xf32>

    return %6 : tensor<1x2x1x1024xf32>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1024x128xf32> = dense<1.000000e+00> : tensor<1024x128xf32>
    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MATMUL_1]]) {shape_value = [1, 1, 1, 1024]} : tensor<1x1024xf32> -> tensor<1x1x1x1024xf32>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[MATMUL_2]]) {shape_value = [1, 1, 1, 1024]} : tensor<1x1024xf32> -> tensor<1x1x1x1024xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_1]], [[RESHAPE_2]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32> -> tensor<1x2x1x1024xf32>
    // CHECK:       [[ADD:%.+]] = IE.Add([[CONCAT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x1024xf32>, tensor<1x1x1x1024xf32> -> tensor<1x2x1x1024xf32>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 3 : i64} : tensor<1x2x1x1024xf32> -> tensor<1x2x1x1024xf32>

    // CHECK:       return [[SOFTMAX]] : tensor<1x2x1x1024xf32>
}

// -----

// CHECK-LABEL: @NoPropagateAddSoftmaxWithInvalidAddShape
func.func @NoPropagateAddSoftmaxWithInvalidAddShape(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %cst1 = const.Declare tensor<2x16x2xf32> = dense<1.000000e+00> : tensor<2x16x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.Reshape(%1) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %3 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %5 = IE.Concat(%2, %4) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    %6 = IE.Add(%5, %cst1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x16x2xf32>, tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    %7 = IE.SoftMax(%6) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    return %7 : tensor<2x16x2xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<2x16x2xf32> = dense<1.000000e+00> : tensor<2x16x2xf32>
    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MATMUL_1]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[MATMUL_2]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_1]], [[RESHAPE_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       [[ADD:%.+]] = IE.Add([[CONCAT]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x16x2xf32>, tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       return [[SOFTMAX]] : tensor<2x16x2xf32>
}

// -----

// CHECK-LABEL: @NoPropagateAddSoftmaxWithInvalidAddSource
func.func @NoPropagateAddSoftmaxWithInvalidAddSource(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>, %arg2: tensor<1x1x1xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.Reshape(%1) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %3 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %5 = IE.Concat(%2, %4) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    %6 = IE.Add(%5, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x16x2xf32>, tensor<1x1x1xf32> -> tensor<2x16x2xf32>
    %7 = IE.SoftMax(%6) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    return %7 : tensor<2x16x2xf32>

    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MATMUL_1]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[MATMUL_2]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_1]], [[RESHAPE_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       [[ADD:%.+]] = IE.Add([[CONCAT]], %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x16x2xf32>, tensor<1x1x1xf32> -> tensor<2x16x2xf32>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       return [[SOFTMAX]] : tensor<2x16x2xf32>
}

// -----

// CHECK-LABEL: @NotPropagateNonConstantSmallSizeInputAddSoftmaxThroughBatchUnrolledMatmul
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x77x77xf32>
func.func @NotPropagateNonConstantSmallSizeInputAddSoftmaxThroughBatchUnrolledMatmul(
            %arg0: tensor<77x128xf32>,
            %arg1: tensor<77x128xf32>,
            %arg2: tensor<1x1x77x77xf32>) -> tensor<1x2x77x77xf32> {
    %cst = const.Declare tensor<77x128xf32> = dense<1.000000e+00> : tensor<77x128xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<77x128xf32>, tensor<77x128xf32> -> tensor<77x77xf32>
    %1 = IE.MatMul(%arg1, %cst) {transpose_b} : tensor<77x128xf32>, tensor<77x128xf32> -> tensor<77x77xf32>
    %2 = IE.Reshape(%0) {shape_value = [1, 1, 77, 77]} : tensor<77x77xf32> -> tensor<1x1x77x77xf32>
    %3 = IE.Reshape(%1) {shape_value = [1, 1, 77, 77]} : tensor<77x77xf32> -> tensor<1x1x77x77xf32>
    %4 = IE.Concat(%2, %3) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x77x77xf32>, tensor<1x1x77x77xf32> -> tensor<1x2x77x77xf32>
    %5 = IE.Add(%4, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x77x77xf32>, tensor<1x1x77x77xf32> -> tensor<1x2x77x77xf32>
    %6 = IE.SoftMax(%5) {axisInd = 3 : i64} : tensor<1x2x77x77xf32> -> tensor<1x2x77x77xf32>

    return %6 : tensor<1x2x77x77xf32>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<77x128xf32> = dense<1.000000e+00> : tensor<77x128xf32>
    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MATMUL_1]]) {shape_value = [1, 1, 77, 77]} : tensor<77x77xf32> -> tensor<1x1x77x77xf32>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[MATMUL_2]]) {shape_value = [1, 1, 77, 77]} : tensor<77x77xf32> -> tensor<1x1x77x77xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_1]], [[RESHAPE_2]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x77x77xf32>, tensor<1x1x77x77xf32> -> tensor<1x2x77x77xf32>
    // CHECK:       [[ADD:%.+]] = IE.Add([[CONCAT]], %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x77x77xf32>, tensor<1x1x77x77xf32> -> tensor<1x2x77x77xf32>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 3 : i64} : tensor<1x2x77x77xf32> -> tensor<1x2x77x77xf32>

    // CHECK:       return [[SOFTMAX]] : tensor<1x2x77x77xf32>
}

// -----

// CHECK-LABEL: @PropagateReshapeThroughBatchUnrolledMatmul
func.func @PropagateReshapeThroughBatchUnrolledMatmul(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %3 = IE.Concat(%1, %2) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [2, 16, 2]} : tensor<32x2xf32> -> tensor<2x16x2xf32>
    return %4 : tensor<2x16x2xf32>

    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MATMUL_1]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[MATMUL_2]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_1]], [[RESHAPE_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       return [[CONCAT]] : tensor<2x16x2xf32>
}

// -----

// CHECK-LABEL: @NoPropagateReshapeWithInvalidConcatInputs
func.func @NoPropagateReshapeWithInvalidConcatInputs(%arg0: tensor<16x2xf32>, %arg1: tensor<18x2xf32>) -> tensor<2x17x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.MatMul(%arg1, %cst) : tensor<18x2xf32>, tensor<2x2xf32> -> tensor<18x2xf32>
    %3 = IE.Concat(%1, %2) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<16x2xf32>, tensor<18x2xf32> -> tensor<34x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [2, 17, 2]} : tensor<34x2xf32> -> tensor<2x17x2xf32>
    return %4 : tensor<2x17x2xf32>

    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[MATMUL_1]], [[MATMUL_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<16x2xf32>, tensor<18x2xf32> -> tensor<34x2xf32>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [2, 17, 2]} : tensor<34x2xf32> -> tensor<2x17x2xf32>
    // CHECK:       return [[RESHAPE]] : tensor<2x17x2xf32>
}

// -----

// CHECK-LABEL: @Propagate4DReshapeSoftMaxThroughBatchUnrolledMatmul
func.func @Propagate4DReshapeSoftMaxThroughBatchUnrolledMatmul(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<1x2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %3 = IE.Concat(%1, %2) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [1, 2, 16, 2]} : tensor<32x2xf32> -> tensor<1x2x16x2xf32>
    %5 = IE.SoftMax(%4) {axisInd = -1 : i64} : tensor<1x2x16x2xf32> -> tensor<1x2x16x2xf32>
    return %5 : tensor<1x2x16x2xf32>

    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MATMUL_1]]) {shape_value = [1, 1, 16, 2]} : tensor<16x2xf32> -> tensor<1x1x16x2xf32>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[MATMUL_2]]) {shape_value = [1, 1, 16, 2]} : tensor<16x2xf32> -> tensor<1x1x16x2xf32>
    // CHECK:       [[SOFTMAX_1:%.+]] = IE.SoftMax([[RESHAPE_1]]) {axisInd = -1 : i64} : tensor<1x1x16x2xf32> -> tensor<1x1x16x2xf32>
    // CHECK:       [[SOFTMAX_2:%.+]] = IE.SoftMax([[RESHAPE_2]]) {axisInd = -1 : i64} : tensor<1x1x16x2xf32> -> tensor<1x1x16x2xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SOFTMAX_1]], [[SOFTMAX_2]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x16x2xf32>, tensor<1x1x16x2xf32> -> tensor<1x2x16x2xf32>
    // CHECK:       return [[CONCAT]] : tensor<1x2x16x2xf32>
}

// -----

// CHECK-LABEL: @PropagateSoftMaxReshapeThroughBatchUnrolledMatmul
func.func @PropagateSoftMaxReshapeThroughBatchUnrolledMatmul(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %3 = IE.Concat(%1, %2) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    %4 = IE.SoftMax(%3) {axisInd = 1 : i64} : tensor<32x2xf32> -> tensor<32x2xf32>
    %5 = IE.Reshape(%4) {shape_value = [2, 16, 2]} : tensor<32x2xf32> -> tensor<2x16x2xf32>
    return %5 : tensor<2x16x2xf32>

    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[SOFTMAX_1:%.+]] = IE.SoftMax([[MATMUL_1]]) {axisInd = 1 : i64} : tensor<16x2xf32> -> tensor<16x2xf32>
    // CHECK:       [[SOFTMAX_2:%.+]] = IE.SoftMax([[MATMUL_2]]) {axisInd = 1 : i64} : tensor<16x2xf32> -> tensor<16x2xf32>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SOFTMAX_1]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SOFTMAX_2]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_1]], [[RESHAPE_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       return [[CONCAT]] : tensor<2x16x2xf32>
}

// -----

// CHECK-LABEL: @PropagateSoftmaxFakeQuantizeThroughBatchUnrolledMatmul
func.func @PropagateSoftmaxFakeQuantizeThroughBatchUnrolledMatmul(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %input_low = const.Declare tensor<1x1x1xf32> = dense<0.0> : tensor<1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1xf32> = dense<1.0> : tensor<1x1x1xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.Reshape(%1) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %3 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %5 = IE.Concat(%2, %4) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    %6 = IE.SoftMax(%5) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    %7 = IE.FakeQuantize(%6, %input_low, %input_high, %input_low, %input_high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                    levels = 255 : i64
                } : tensor<2x16x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<2x16x2xf32>
    return %7 : tensor<2x16x2xf32>

    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    // CHECK:       [[MATMUL_1:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MATMUL_1]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[MATMUL_2:%.+]] = IE.MatMul
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[MATMUL_2]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[SOFTMAX_1:%.+]] = IE.SoftMax([[RESHAPE_1]]) {axisInd = -1 : i64} : tensor<1x16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[SOFTMAX_2:%.+]] = IE.SoftMax([[RESHAPE_2]]) {axisInd = -1 : i64} : tensor<1x16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[FAKE_QUANTIZE_1:%.+]] = IE.FakeQuantize([[SOFTMAX_1]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64}
    // CHECK-SAME:      : tensor<1x16x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[FAKE_QUANTIZE_2:%.+]] = IE.FakeQuantize([[SOFTMAX_2]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64}
    // CHECK-SAME:      : tensor<1x16x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x16x2xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[FAKE_QUANTIZE_1]], [[FAKE_QUANTIZE_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       return [[CONCAT]] : tensor<2x16x2xf32>
}

// -----

// CHECK-LABEL: @PropagateMultiply
// CHECK-SAME:  ([[INPUT_0:%.+]]: tensor<1x16x1x128xf16>, [[INPUT_1:%.+]]: tensor<1x16x1x128xf16>)
func.func @PropagateMultiply(%arg0: tensor<1x16x1x128xf16>, %arg1: tensor<1x16x1x128xf16>) -> tensor<2x16x1x128xf16> {
    %gamma1 = const.Declare tensor<128xf16> = dense<1.000000e+00> : tensor<128xf16>
    %gamma2 = const.Declare tensor<128xf16> = dense<2.000000e+00> : tensor<128xf16>
    %cst_mul = const.Declare tensor<f16> = dense<2.000000e+00> : tensor<f16>

    %0 = IE.RMS(%arg0, %gamma1) {epsilon = 9.9999999747524271E-7 : f64} : tensor<1x16x1x128xf16>, tensor<128xf16> -> tensor<1x16x1x128xf16>
    %1 = IE.RMS(%arg1, %gamma2) {epsilon = 9.9999999747524271E-7 : f64} : tensor<1x16x1x128xf16>, tensor<128xf16> -> tensor<1x16x1x128xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 0>} : tensor<1x16x1x128xf16>, tensor<1x16x1x128xf16> -> tensor<2x16x1x128xf16>
    %3 = IE.Multiply(%2, %cst_mul) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x16x1x128xf16>, tensor<f16> -> tensor<2x16x1x128xf16>

    return %3 : tensor<2x16x1x128xf16>

    // CHECK-DAG:   [[GAMMA1:%.+]] = const.Declare tensor<128xf16> = dense<1.000000e+00>
    // CHECK-DAG:   [[GAMMA2:%.+]] = const.Declare tensor<128xf16> = dense<2.000000e+00>
    // CHECK-DAG:   [[CST_MUL:%.+]] = const.Declare tensor<f16> = dense<2.000000e+00>
    // CHECK:       [[RMS0:%.+]] = IE.RMS([[INPUT_0]], [[GAMMA1]]) {epsilon = 9.9999999747524271E-7 : f64} : tensor<1x16x1x128xf16>, tensor<128xf16> -> tensor<1x16x1x128xf16>
    // CHECK:       [[RMS1:%.+]] = IE.RMS([[INPUT_1]], [[GAMMA2]]) {epsilon = 9.9999999747524271E-7 : f64} : tensor<1x16x1x128xf16>, tensor<128xf16> -> tensor<1x16x1x128xf16>
    // CHECK:       [[MUL0:%.+]] = IE.Multiply([[RMS0]], [[CST_MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x128xf16>, tensor<f16> -> tensor<1x16x1x128xf16>
    // CHECK:       [[MUL1:%.+]] = IE.Multiply([[RMS1]], [[CST_MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x128xf16>, tensor<f16> -> tensor<1x16x1x128xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[MUL0]], [[MUL1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x1x128xf16>, tensor<1x16x1x128xf16> -> tensor<2x16x1x128xf16>
    // CHECK:       return [[CONCAT]] : tensor<2x16x1x128xf16>
}

// -----

// CHECK-LABEL: @NotPropagateMultiply
// CHECK-SAME:  ([[INPUT_0:%.+]]: tensor<1x16x1x128xf16>, [[INPUT_1:%.+]]: tensor<1x16x1x128xf16>, [[INPUT_2:%.+]]: tensor<128xf16>)
func.func @NotPropagateMultiply(%arg0: tensor<1x16x1x128xf16>, %arg1: tensor<1x16x1x128xf16>, %arg2: tensor<128xf16>) -> tensor<2x16x1x128xf16> {
    %gamma1 = const.Declare tensor<128xf16> = dense<1.000000e+00> : tensor<128xf16>
    %cst_mul = const.Declare tensor<f16> = dense<2.000000e+00> : tensor<f16>

    %0 = IE.RMS(%arg0, %gamma1) {epsilon = 9.9999999747524271E-7 : f64} : tensor<1x16x1x128xf16>, tensor<128xf16> -> tensor<1x16x1x128xf16>
    %1 = IE.RMS(%arg1, %arg2) {epsilon = 9.9999999747524271E-7 : f64} : tensor<1x16x1x128xf16>, tensor<128xf16> -> tensor<1x16x1x128xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 0>} : tensor<1x16x1x128xf16>, tensor<1x16x1x128xf16> -> tensor<2x16x1x128xf16>
    %3 = IE.Multiply(%2, %cst_mul) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x16x1x128xf16>, tensor<f16> -> tensor<2x16x1x128xf16>

    return %3 : tensor<2x16x1x128xf16>

    // CHECK-DAG:   [[GAMMA1:%.+]] = const.Declare tensor<128xf16> = dense<1.000000e+00>
    // CHECK-DAG:   [[CST_MUL:%.+]] = const.Declare tensor<f16> = dense<2.000000e+00>
    // CHECK:       [[RMS0:%.+]] = IE.RMS([[INPUT_0]], [[GAMMA1]]) {epsilon = 9.9999999747524271E-7 : f64} : tensor<1x16x1x128xf16>, tensor<128xf16> -> tensor<1x16x1x128xf16>
    // CHECK:       [[RMS1:%.+]] = IE.RMS([[INPUT_1]], [[INPUT_2]]) {epsilon = 9.9999999747524271E-7 : f64} : tensor<1x16x1x128xf16>, tensor<128xf16> -> tensor<1x16x1x128xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RMS0]], [[RMS1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x1x128xf16>, tensor<1x16x1x128xf16> -> tensor<2x16x1x128xf16>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[CONCAT]], [[CST_MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x16x1x128xf16>, tensor<f16> -> tensor<2x16x1x128xf16>

    // CHECK:       return [[MUL]] : tensor<2x16x1x128xf16>
}
