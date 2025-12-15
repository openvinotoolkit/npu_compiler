//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-op-through-batch-concat %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @PropagateSoftmaxFakeQuantizeThroughBatchUnrolledMatmulF8E4M3FN
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<16x2xf32>, [[INPUT_1:%.+]]: tensor<16x2xf32>
func.func @PropagateSoftmaxFakeQuantizeThroughBatchUnrolledMatmulF8E4M3FN(%input0: tensor<16x2xf32>, %input1: tensor<16x2xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %low = const.Declare tensor<1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1xf32>
    %high = const.Declare tensor<1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1xf32>

    %1 = IE.MatMul(%input0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.Reshape(%1) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %3 = IE.MatMul(%input1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    %5 = IE.Concat(%2, %4) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>
    %6 = IE.SoftMax(%5) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>

    %7 = IE.FakeQuantize(%6, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<2x16x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<2x16x2xf32>

    return %7 : tensor<2x16x2xf32>

    // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1xf32>

    // CHECK:    [[MUL_0:%.+]] = IE.MatMul([[INPUT_0]], [[CST]]) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    // CHECK:    [[RESHAPE_0:%.+]] = IE.Reshape([[MUL_0]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:    [[MUL_1:%.+]] = IE.MatMul([[INPUT_1]], [[CST]]) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    // CHECK:    [[RESHAPE_1:%.+]] = IE.Reshape([[MUL_1]]) {shape_value = [1, 16, 2]} : tensor<16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:    [[MAX_0:%.+]] = IE.SoftMax([[RESHAPE_0]]) {axisInd = -1 : i64} : tensor<1x16x2xf32> -> tensor<1x16x2xf32>
    // CHECK:    [[MAX_1:%.+]] = IE.SoftMax([[RESHAPE_1]]) {axisInd = -1 : i64} : tensor<1x16x2xf32> -> tensor<1x16x2xf32>

    // CHECK:    [[FQ_0:%.+]] = IE.FakeQuantize([[MAX_0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x16x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x16x2xf32>
    // CHECK:    [[FQ_1:%.+]] = IE.FakeQuantize([[MAX_1]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x16x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x16x2xf32>

    // CHECK:    [[CONCAT:%.+]] = IE.Concat([[FQ_0]], [[FQ_1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x16x2xf32>, tensor<1x16x2xf32> -> tensor<2x16x2xf32>

    // CHECK:    return [[CONCAT]] : tensor<2x16x2xf32>
}
