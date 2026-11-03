//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @MatMulWithoutScaleAndZeroPoints
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:      [[ARG1:%.+]]: tensor<1x16x16x16x!qElemType>
func.func @MatMulWithoutScaleAndZeroPoints(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<1x16x16x16x!qElemType>) -> tensor<1x16x16x16xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16xf16>
    return %0 : tensor<1x16x16x16xf16>

    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16xf16>
    // CHECK:       return [[MATMUL]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @MatMulWithScale
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:      [[ARG1:%.+]]: tensor<1x16x16x16x!qElemType>,
// CHECK-SAME:      [[ARG2:%.+]]: tensor<16x1x1x1xf32>
func.func @MatMulWithScale(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<1x16x16x16x!qElemType>, %arg2: tensor<16x1x1x1xf32>) -> tensor<1x16x16x16xf16> {
    %0 = IE.MatMul(%arg0, %arg1, , %arg2) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<16x1x1x1xf32> -> tensor<1x16x16x16xf16>
    return %0 : tensor<1x16x16x16xf16>

    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[ARG1]], [[ARG2]]) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<16x1x1x1xf32> -> tensor<1x16x16x16xf16>
    // CHECK:       return [[MATMUL]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @MatMulWithScaleAndZeroPoints
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:      [[ARG1:%.+]]: tensor<1x16x16x16x!qElemType>,
// CHECK-SAME:      [[ARG2:%.+]]: tensor<16x1x1x1xf32>,
// CHECK-SAME:      [[ARG3:%.+]]: tensor<16x1x1x1xsi8>
func.func @MatMulWithScaleAndZeroPoints(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<1x16x16x16x!qElemType>, %arg2: tensor<16x1x1x1xf32>, %arg3: tensor<16x1x1x1xsi8>) -> tensor<1x16x16x16xf16> {
    %0 = IE.MatMul(%arg0, %arg1, , %arg2, %arg3) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xsi8> -> tensor<1x16x16x16xf16>
    return %0 : tensor<1x16x16x16xf16>

    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]]) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xsi8> -> tensor<1x16x16x16xf16>
    // CHECK:       return [[MATMUL]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @MatMulWithBias
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:      [[ARG1:%.+]]: tensor<1x16x16x16x!qElemType>,
// CHECK-SAME:      [[ARG2:%.+]]: tensor<1x1x1x16xf16>
func.func @MatMulWithBias(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<1x16x16x16x!qElemType>, %arg2: tensor<1x1x1x16xf16>) -> tensor<1x16x16x16xf16> {
    %0 = IE.MatMul(%arg0, %arg1, %arg2) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<1x1x1x16xf16> -> tensor<1x16x16x16xf16>
    return %0 : tensor<1x16x16x16xf16>

    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[ARG1]], [[ARG2]]) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<1x1x1x16xf16> -> tensor<1x16x16x16xf16>
    // CHECK:       return [[MATMUL]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @MatMulWithScaleAndBias
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:      [[ARG1:%.+]]: tensor<1x16x16x16x!qElemType>,
// CHECK-SAME:      [[ARG2:%.+]]: tensor<16x1x1x1xf32>,
// CHECK-SAME:      [[ARG3:%.+]]: tensor<1x1x1x16xf16>
func.func @MatMulWithScaleAndBias(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<1x16x16x16x!qElemType>, %arg2: tensor<16x1x1x1xf32>, %arg3: tensor<1x1x1x16xf16>) -> tensor<1x16x16x16xf16> {
    %0 = IE.MatMul(%arg0, %arg1, %arg3, %arg2) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<1x1x1x16xf16>, tensor<16x1x1x1xf32> -> tensor<1x16x16x16xf16>
    return %0 : tensor<1x16x16x16xf16>

    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[ARG1]], [[ARG3]], [[ARG2]]) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<1x1x1x16xf16>, tensor<16x1x1x1xf32> -> tensor<1x16x16x16xf16>
    // CHECK:       return [[MATMUL]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @MatMulWithScaleZeroPointsAndBias
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:      [[ARG1:%.+]]: tensor<1x16x16x16x!qElemType>,
// CHECK-SAME:      [[ARG2:%.+]]: tensor<16x1x1x1xf32>,
// CHECK-SAME:      [[ARG3:%.+]]: tensor<16x1x1x1xsi8>,
// CHECK-SAME:      [[ARG4:%.+]]: tensor<1x1x1x1xf16>
func.func @MatMulWithScaleZeroPointsAndBias(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<1x16x16x16x!qElemType>, %arg2: tensor<16x1x1x1xf32>, %arg3: tensor<16x1x1x1xsi8>, %arg4: tensor<1x1x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %0 = IE.MatMul(%arg0, %arg1, %arg4, %arg2, %arg3) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<1x1x1x1xf16>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xsi8> -> tensor<1x16x16x16xf16>
    return %0 : tensor<1x16x16x16xf16>

    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[ARG1]], [[ARG4]], [[ARG2]], [[ARG3]]) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<1x1x1x1xf16>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xsi8> -> tensor<1x16x16x16xf16>
    // CHECK:       return [[MATMUL]]
}


// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @MatMulWithZeroPointsOnly
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:      [[ARG1:%.+]]: tensor<1x16x16x16x!qElemType>,
// CHECK-SAME:      [[ARG2:%.+]]: tensor<16x1x1x1xsi8>
func.func @MatMulWithZeroPointsOnly(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<1x16x16x16x!qElemType>, %arg2: tensor<16x1x1x1xsi8>) -> tensor<1x16x16x16xf16> {
    %0 = IE.MatMul(%arg0, %arg1, , , %arg2) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<16x1x1x1xsi8> -> tensor<1x16x16x16xf16>
    return %0 : tensor<1x16x16x16xf16>

    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[ARG1]], [[ARG2]]) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<16x1x1x1xsi8> -> tensor<1x16x16x16xf16>
    // CHECK:       return [[MATMUL]]
}
