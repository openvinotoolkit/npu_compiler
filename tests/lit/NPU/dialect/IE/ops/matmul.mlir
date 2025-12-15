//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @MatMulTransposeInputsTransposeAllTrue
// CHECK-SAME:      [[INPUT:%.+]]: tensor<256x166xf32>
func.func @MatMulTransposeInputsTransposeAllTrue(%arg0: tensor<256x166xf32>, %arg1: tensor<256x256xf32>) -> tensor<166x256xf32> {
    %cst = const.Declare tensor<256x256xf32> = dense<1.0> : tensor<256x256xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_a, transpose_b} : tensor<256x166xf32>, tensor<256x256xf32> -> tensor<166x256xf32>

    return %0 : tensor<166x256xf32>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<256x256xf32> = dense<1.000000e+00> : tensor<256x256xf32>
    // CHECK:   IE.Transpose([[INPUT]]) {order_value = #CN} : tensor<256x166xf32> -> tensor<166x256xf32>
    // CHECK:   IE.FullyConnected(%0, [[CST]]) : tensor<166x256xf32>, tensor<256x256xf32> -> tensor<166x256xf32>
    // CHECK:   return %1 : tensor<166x256xf32>
}

// -----

// CHECK-LABEL: @MatMulTransposeInputsTransposeAFalse
// CHECK-SAME:      [[INPUT:%.+]]: tensor<196x128xf32>
func.func @MatMulTransposeInputsTransposeAFalse(%arg0: tensor<196x128xf32>, %arg1: tensor<640x128xf32>) -> tensor<196x640xf32> {
    %cst = const.Declare tensor<640x128xf32> = dense<1.0> : tensor<640x128xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<196x128xf32>, tensor<640x128xf32> -> tensor<196x640xf32>

    return %0 : tensor<196x640xf32>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<640x128xf32> = dense<1.000000e+00> : tensor<640x128xf32>
    // CHECK:   %0 = IE.FullyConnected([[INPUT]], [[CST]]) : tensor<196x128xf32>, tensor<640x128xf32> -> tensor<196x640xf32>
    // CHECK:   return %0 : tensor<196x640xf32>
}

// // -----

// CHECK-LABEL: @MatMulTransposeInputsTransposeBFalse
// CHECK-SAME:      [[INPUT:%.+]]: tensor<40x131072xf32>
func.func @MatMulTransposeInputsTransposeBFalse(%arg0: tensor<40x131072xf32>, %arg1: tensor<40x19xf32>) -> tensor<131072x19xf32> {
    %cst = const.Declare tensor<40x19xf32> = dense<1.0> : tensor<40x19xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_a} : tensor<40x131072xf32>, tensor<40x19xf32> -> tensor<131072x19xf32>

    return %0 : tensor<131072x19xf32>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<19x40xf32> = dense<1.000000e+00> : tensor<40x19xf32>, [#const.Transpose<#CN>]
    // CHECK:   %0 = IE.Transpose([[INPUT]]) {order_value = #CN} : tensor<40x131072xf32> -> tensor<131072x40xf32>
    // CHECK:   %1 = IE.FullyConnected(%0, [[CST]]) : tensor<131072x40xf32>, tensor<19x40xf32> -> tensor<131072x19xf32>
    // CHECK:   return %1 : tensor<131072x19xf32>
}

// -----

// CHECK-LABEL: @MatMulTransposeInputsTransposeAllFalse
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x784xf32>
func.func @MatMulTransposeInputsTransposeAllFalse(%arg0: tensor<1x784xf32>, %arg1: tensor<784x256xf32>) -> tensor<1x256xf32> {
    %cst = const.Declare tensor<784x256xf32> = dense<1.0> : tensor<784x256xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x784xf32>, tensor<784x256xf32> -> tensor<1x256xf32>

    return %0 : tensor<1x256xf32>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<256x784xf32> = dense<1.000000e+00> : tensor<784x256xf32>, [#const.Transpose<#CN>]
    // CHECK:   %0 = IE.FullyConnected([[INPUT]], [[CST]]) : tensor<1x784xf32>, tensor<256x784xf32> -> tensor<1x256xf32>
    // CHECK:   return %0 : tensor<1x256xf32>
}
