//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @Rank2MatMulIsConvertedToFullyConnected
// CHECK-SAME:      [[ARG0:%.+]]: tensor<16x16xf16>,
// CHECK-SAME:      [[ARG1:%.+]]: tensor<16x16xf16>
func.func @Rank2MatMulIsConvertedToFullyConnected(%arg0: tensor<16x16xf16>, %arg1: tensor<16x16xf16>) -> tensor<16x16xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<16x16xf16>, tensor<16x16xf16> -> tensor<16x16xf16>
    return %0 : tensor<16x16xf16>

    // CHECK-NOT:   IE.MatMul
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[ARG0]], [[ARG1]])
    // CHECK:       return [[FC]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// A rank-2 MatMul carrying scale must stay a MatMul: FullyConnected would drop the scale and cannot take quantized weights
// CHECK-LABEL: @Rank2MatMulWithScaleIsNotConvertedToFullyConnected
// CHECK-SAME:      [[ARG0:%.+]]: tensor<16x16xf16>,
// CHECK-SAME:      [[ARG1:%.+]]: tensor<16x16x!qElemType>,
// CHECK-SAME:      [[ARG2:%.+]]: tensor<16x1xf32>
func.func @Rank2MatMulWithScaleIsNotConvertedToFullyConnected(%arg0: tensor<16x16xf16>, %arg1: tensor<16x16x!qElemType>, %arg2: tensor<16x1xf32>) -> tensor<16x16xf16> {
    %0 = IE.MatMul(%arg0, %arg1, , %arg2) {transpose_b} : tensor<16x16xf16>, tensor<16x16x!qElemType>, tensor<16x1xf32> -> tensor<16x16xf16>
    return %0 : tensor<16x16xf16>

    // CHECK-NOT:   IE.FullyConnected
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[ARG1]], [[ARG2]]) {transpose_b}
    // CHECK:       return [[MATMUL]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @Rank2MatMulWithScaleAndZeroPointsIsNotConvertedToFullyConnected
// CHECK-SAME:      [[ARG0:%.+]]: tensor<16x16xf16>,
// CHECK-SAME:      [[ARG1:%.+]]: tensor<16x16x!qElemType>,
// CHECK-SAME:      [[ARG2:%.+]]: tensor<16x1xf32>,
// CHECK-SAME:      [[ARG3:%.+]]: tensor<16x1xsi8>
func.func @Rank2MatMulWithScaleAndZeroPointsIsNotConvertedToFullyConnected(%arg0: tensor<16x16xf16>, %arg1: tensor<16x16x!qElemType>, %arg2: tensor<16x1xf32>, %arg3: tensor<16x1xsi8>) -> tensor<16x16xf16> {
    %0 = IE.MatMul(%arg0, %arg1, , %arg2, %arg3) {transpose_b} : tensor<16x16xf16>, tensor<16x16x!qElemType>, tensor<16x1xf32>, tensor<16x1xsi8> -> tensor<16x16xf16>
    return %0 : tensor<16x16xf16>

    // CHECK-NOT:   IE.FullyConnected
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]]) {transpose_b}
    // CHECK:       return [[MATMUL]]
}
