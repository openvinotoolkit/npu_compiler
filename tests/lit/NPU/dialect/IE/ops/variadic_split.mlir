//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% %s --verify-diagnostics | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @ParseAndPrint(%arg: tensor<2x3x4x5xf32, {order = #NCWH}>) -> (tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>) {
    %variadic:3 = IE.VariadicSplit(%arg) {axis=3, split_lengths=[2, 2, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>
    return %variadic#0, %variadic#1, %variadic#2 : tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>
    // CHECK: @ParseAndPrint([[ARG:%.+]]: tensor<2x3x4x5xf32, {order = #NCWH}>)
    // CHECK:     [[VARIADIC:%[0-9]+]]:3 = IE.VariadicSplit([[ARG]]) {axis = 3 : i64, split_lengths = [2, 2, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>
    // CHECK:     return [[VARIADIC]]
}

// -----

func.func @AxisOutOfRangeTooSmall(%arg: tensor<2x3x4x5xf32>) -> (tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>) {
    // expected-error@+1 {{'IE.VariadicSplit' op 'axis' must be in the interval [-4, 3] but got -5}}
    %variadic:3 = IE.VariadicSplit(%arg) {axis=-5, split_lengths=[2, 2, 1]} : tensor<2x3x4x5xf32> -> tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>
    return %variadic#0, %variadic#1, %variadic#2 : tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>
}

// -----

func.func @AxisOutOfRangeTooLarge(%arg: tensor<2x3x4x5xf32>) -> (tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>) {
    // expected-error@+1 {{'IE.VariadicSplit' op 'axis' must be in the interval [-4, 3] but got 5}}
    %variadic:3 = IE.VariadicSplit(%arg) {axis=5, split_lengths=[2, 2, 1]} : tensor<2x3x4x5xf32> -> tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>
    return %variadic#0, %variadic#1, %variadic#2 : tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>
}

// -----

func.func @SplitLengthsTooNegative(%arg: tensor<2x3x4x5xf32>) -> (tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>) {
    // expected-error@+1 {{'IE.VariadicSplit' op all values in 'split_lengths' must be -1 or greater}}
    %variadic:3 = IE.VariadicSplit(%arg) {axis=3, split_lengths=[2, 2, -2]} : tensor<2x3x4x5xf32> -> tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>
    return %variadic#0, %variadic#1, %variadic#2 : tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>
}

// -----

func.func @SplitLengthsTooManyMinusOne(%arg: tensor<2x3x4x5xf32>) -> (tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>) {
    // expected-error@+1 {{'IE.VariadicSplit' op 'split_lengths' can contain at most one -1 value}}
    %variadic:3 = IE.VariadicSplit(%arg) {axis=3, split_lengths=[2, -1, -1]} : tensor<2x3x4x5xf32> -> tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>
    return %variadic#0, %variadic#1, %variadic#2 : tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>
}

// -----

func.func @InvalidSplitLengthsSum(%arg: tensor<2x3x4x5xf32>) -> (tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>) {
    // expected-error@+1 {{'IE.VariadicSplit' op entries in 'split_lengths' are expected to sum up to axis dimension but got 6}}
    %variadic:3 = IE.VariadicSplit(%arg) {axis=3, split_lengths=[2, 2, 2]} : tensor<2x3x4x5xf32> -> tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>
    return %variadic#0, %variadic#1, %variadic#2 : tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>
}

// -----

func.func @CannotInferSplitLengths(%arg: tensor<2x3x4x5xf32>) -> (tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>) {
    // expected-error@+1 {{'IE.VariadicSplit' op cannot infer a positive value for the -1 value in 'split_lengths'}}
    %variadic:3 = IE.VariadicSplit(%arg) {axis=3, split_lengths=[9, 9, -1]} : tensor<2x3x4x5xf32> -> tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>
    return %variadic#0, %variadic#1, %variadic#2 : tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>
}

// -----

func.func @MismatchNumberOfOutputs(%arg: tensor<2x3x4x5xf32>) -> (tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>) {
    // expected-error@+1 {{'IE.VariadicSplit' op number of outputs 2 does not match length of 'split_lengths' 3}}
    %variadic:2 = IE.VariadicSplit(%arg) {axis=3, split_lengths=[2, 2, 1]} : tensor<2x3x4x5xf32> -> tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>
    return %variadic#0, %variadic#1 : tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>
}

// -----

func.func @WrongDimensionSize(%arg: tensor<2x3x4x5xf32>) -> (tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x9xf32>) {
    // expected-error@+1 {{'IE.VariadicSplit' op output 2 is expected to have size 1 for axis 3 but got 9}}
    %variadic:3 = IE.VariadicSplit(%arg) {axis=3, split_lengths=[2, 2, 1]} : tensor<2x3x4x5xf32> -> tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x9xf32>
    return %variadic#0, %variadic#1, %variadic#2 : tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x9xf32>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.078431372549019607>

func.func @QuantizedInputs(%arg: tensor<2x3x4x5x!qElemType>) -> (tensor<2x3x4x2x!qElemType>, tensor<2x3x4x2x!qElemType>, tensor<2x3x4x1x!qElemType>) {
    // expected-error@+1 {{'IE.VariadicSplit' op operand #0 must be ranked tensor of integer or floating-point values, but got 'tensor<2x3x4x5x!quant.uniform<u8:f16, 0.078431372549019607>>'}}
    %variadic:3 = IE.VariadicSplit(%arg) {axis=3, split_lengths=[2, 2, 1]} : tensor<2x3x4x5x!qElemType> -> tensor<2x3x4x2x!qElemType>, tensor<2x3x4x2x!qElemType>, tensor<2x3x4x1x!qElemType>
    return %variadic#0, %variadic#1, %variadic#2 : tensor<2x3x4x2x!qElemType>, tensor<2x3x4x2x!qElemType>, tensor<2x3x4x1x!qElemType>
}
