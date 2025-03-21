//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @FuseScaleAndBias
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x300x300xf32>)
func.func @FuseScaleAndBias(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %0 = IE.ScaleShift(%arg0, %weights)
        {operandSegmentSizes = array<i32: 1, 1, 0>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    %bias = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %1 : tensor<1x3x300x300xf32>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       [[SCALE_SHIFT:%.+]] = IE.ScaleShift([[ARG0]], [[WEIGHTS]], [[BIAS]])
    // CHECK:       return [[SCALE_SHIFT]]
}

// -----

// Fuse ScaleShift and Bias should fail
// CHECK-LABEL: @FuseScaleAndBias
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x300x300xf32>)
func.func @FuseScaleAndBias(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %bias0 = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %0 = IE.ScaleShift(%arg0, %weights, %bias0)
        {operandSegmentSizes = array<i32: 1, 1, 1>}:
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    %bias1 = const.Declare tensor<1x3x1x1xf32> = dense<4.0> : tensor<1x3x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias1)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %1 : tensor<1x3x300x300xf32>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   [[BIAS_0:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   [[BIAS_1:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<4.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       [[SCALE_SHIFT_0:%.+]] = IE.ScaleShift([[ARG0]], [[WEIGHTS]], [[BIAS_0]])
    // CHECK:       [[SCALE_SHIFT_1:%.+]] = IE.ScaleShift([[SCALE_SHIFT_0]], [[BIAS_1]])
    // CHECK:       return [[SCALE_SHIFT_1]]
}

// -----

// Fuse Scale and ScaleShift should fail
// CHECK-LABEL: @FuseScaleAndBias
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x300x300xf32>)
func.func @FuseScaleAndBias(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %0 = IE.ScaleShift(%arg0, %weights)
        {operandSegmentSizes = array<i32: 1, 1, 0>}:
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    %weights1 = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %bias = const.Declare tensor<1x3x1x1xf32> = dense<4.0> : tensor<1x3x1x1xf32>
    %1 = IE.ScaleShift(%0, %weights1, %bias)
        {operandSegmentSizes = array<i32: 1, 1, 1>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %1 : tensor<1x3x300x300xf32>

    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<4.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       [[SCALE_SHIFT_0:%.+]] = IE.ScaleShift([[ARG0]], [[WEIGHTS_0]])
    // CHECK:       [[SCALE_SHIFT_1:%.+]] = IE.ScaleShift([[SCALE_SHIFT_0]], [[WEIGHTS_1]], [[BIAS]])
    // CHECK:       return [[SCALE_SHIFT_1]]
}

// -----

// CHECK-LABEL: @FuseScaleShifts
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x300x300xf32>)
func.func @FuseScaleShifts(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights_0 = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %bias_0 = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %0 = IE.ScaleShift(%arg0, %weights_0, %bias_0)
        {operandSegmentSizes = array<i32: 1, 1, 1>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    %weights_1 = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %bias_1 = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %1 = IE.ScaleShift(%0, %weights_1, %bias_1)
        {operandSegmentSizes = array<i32: 1, 1, 1>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %1 : tensor<1x3x300x300xf32>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<4.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<9.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       [[SCALE_SHIFT:%.+]] = IE.ScaleShift([[ARG0]], [[WEIGHTS]], [[BIAS]])
    // CHECK:       return [[SCALE_SHIFT]]
}
