//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --move-multiply-divide-post-op %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010


// -----

// CHECK-LABEL: @SwapMultiplyWithMatmul
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @SwapMultiplyWithMatmul(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[INPUT1]]) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[MATMUL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1x1024xf32>

    // CHECK:       return  [[MULTIPLY]] : tensor<1x32x1x1024xf32>
}


// -----

// CHECK-LABEL: @SwapTwoMultiplyWithMatmul
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x80x1024xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x80x1024xf32>
func.func @SwapTwoMultiplyWithMatmul(%arg0: tensor<1x32x80x1024xf32>, %arg1: tensor<1x32x80x1024xf32>) -> tensor<1x32x80x80xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x1024xf32>
    %cst0 = const.Declare tensor<1x1x1x1xf32> = dense<0.2> : tensor<1x1x1x1xf32>
    %1 = IE.Multiply(%arg1, %cst0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x1024xf32>

    %2 = IE.MatMul(%0, %1) {transpose_b} : tensor<1x32x80x1024xf32>, tensor<1x32x80x1024xf32> -> tensor<1x32x80x80xf32>

    return %2 : tensor<1x32x80x80xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK:       [[MATMUL:%.+]]  = IE.MatMul([[INPUT1]], [[INPUT2]]) {transpose_b} : tensor<1x32x80x1024xf32>, tensor<1x32x80x1024xf32> -> tensor<1x32x80x80xf32>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[MATMUL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x80xf32>
    // CHECK:       [[MULTIPLY_0:%.+]] = IE.Multiply([[MULTIPLY]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x80xf3

    // CHECK:       return  [[MULTIPLY_0]] : tensor<1x32x80x80xf32>
}


// -----

// CHECK-LABEL: @NotSwapMultiplyWithPostOp
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @NotSwapMultiplyWithPostOp(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.Relu<>} : tensor<1x1x1x1xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[CST]], [[INPUT1]])
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1x1024xf32>
}


// -----

// CHECK-LABEL: @NotSwapMultiplyWithClamp
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @NotSwapMultiplyWithClamp(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%cst, %arg0) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        clamp = {min = 0.000000e+00 : f64, max = 1.000000e+00 : f64}
    } : tensor<1x1x1x1xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[CST]], [[INPUT1]])
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1x1024xf32>
}

// -----

// CHECK-LABEL: @NotSwapMultiplyForNotSplatConst
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x2xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x2xf32>
func.func @NotSwapMultiplyForNotSplatConst(%arg0: tensor<1x32x1024x2xf32>, %arg1: tensor<1x32x1x2xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x2xf32> = dense<[[[[1.0,1.6]]]]> : tensor<1x1x1x2xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x2xf32>, tensor<1x1x1x2xf32> -> tensor<1x32x1024x2xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x2xf32>, tensor<1x32x1024x2xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1x1024xf32>
}

// -----

// CHECK-LABEL: @NoSwapMultiplyForMultiUser
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @NoSwapMultiplyForMultiUser(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> (tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>) {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %0, %1 : tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MULTIPLY:%.+]], [[MATMUL]] : tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>
}


// -----

// CHECK-LABEL: @NotBeneficialForSwapMultiply
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1025x80xf32>
func.func @NotBeneficialForSwapMultiply(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1025x80xf32>) -> tensor<1x32x1025x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.Relu<>} : tensor<1x1x1x1xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1025x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1025x1024xf32>

    return %1 : tensor<1x32x1025x1024xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[CST]], [[INPUT1]])
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1025x1024xf32>
}


// -----

// CHECK-LABEL: @NotBeneficialForSwapMultiplyWithClamp
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1025x80xf32>
func.func @NotBeneficialForSwapMultiplyWithClamp(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1025x80xf32>) -> tensor<1x32x1025x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%cst, %arg0) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        clamp = {min = 0.000000e+00 : f64, max = 1.000000e+00 : f64}
    } : tensor<1x1x1x1xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1025x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1025x1024xf32>

    return %1 : tensor<1x32x1025x1024xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[CST]], [[INPUT1]])
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1025x1024xf32>
}

// -----

// CHECK-LABEL: @MoveMultiplyPostConcat
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x3584xf16>, [[INPUT2:%.+]]: tensor<1x3584xf16>, [[INPUT3:%.+]]: tensor<1x3584xf16>, [[INPUT4:%.+]]: tensor<1x3584xf16>
func.func @MoveMultiplyPostConcat(%arg0: tensor<1x3584xf16>, %arg1: tensor<1x3584xf16>, %arg2: tensor<1x3584xf16>, %arg3: tensor<1x3584xf16>) -> tensor<2x3584xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3584xf16>, tensor<1x3584xf16> -> tensor<1x3584xf16>
    %1 = IE.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3584xf16>, tensor<1x3584xf16> -> tensor<1x3584xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x3584xf16>, tensor<1x3584xf16> -> tensor<2x3584xf16>
    return %2 : tensor<2x3584xf16>

    // CHECK:       [[CONCAT_RIGHT:%.+]]  = IE.Concat([[INPUT2]], [[INPUT4]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x3584xf16>, tensor<1x3584xf16> -> tensor<2x3584xf16>
    // CHECK:       [[CONCAT_LEFT:%.+]] = IE.Concat([[INPUT1]], [[INPUT3]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x3584xf16>, tensor<1x3584xf16> -> tensor<2x3584xf16>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[CONCAT_LEFT]], [[CONCAT_RIGHT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3584xf16>, tensor<2x3584xf16> -> tensor<2x3584xf16>

    // CHECK:       return [[MULTIPLY]] : tensor<2x3584xf16>
}

// -----

// CHECK-LABEL: @MoveMultiplyPostConcatWithReshape
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x3584xf16>, [[INPUT2:%.+]]: tensor<1x3584xf16>, [[INPUT3:%.+]]: tensor<1x3584xf16>, [[INPUT4:%.+]]: tensor<1x3584xf16>
func.func @MoveMultiplyPostConcatWithReshape(%arg0: tensor<1x3584xf16>, %arg1: tensor<1x3584xf16>, %arg2: tensor<1x3584xf16>, %arg3: tensor<1x3584xf16>) -> tensor<1x2x1x3584xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3584xf16>, tensor<1x3584xf16> -> tensor<1x3584xf16>
    %1 = IE.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3584xf16>, tensor<1x3584xf16> -> tensor<1x3584xf16>
    %2 = IE.Reshape(%0) {shape_value = [1, 1, 1, 3584]} : tensor<1x3584xf16> -> tensor<1x1x1x3584xf16>
    %3 = IE.Reshape(%1) {shape_value = [1, 1, 1, 3584]} : tensor<1x3584xf16> -> tensor<1x1x1x3584xf16>
    %4 = IE.Concat(%2, %3) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x3584xf16>, tensor<1x1x1x3584xf16> -> tensor<1x2x1x3584xf16>
    return %4 : tensor<1x2x1x3584xf16>

    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [1, 1, 1, 3584]} : tensor<1x3584xf16> -> tensor<1x1x1x3584xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[INPUT2]]) {shape_value = [1, 1, 1, 3584]} : tensor<1x3584xf16> -> tensor<1x1x1x3584xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[INPUT3]]) {shape_value = [1, 1, 1, 3584]} : tensor<1x3584xf16> -> tensor<1x1x1x3584xf16>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.Reshape([[INPUT4]]) {shape_value = [1, 1, 1, 3584]} : tensor<1x3584xf16> -> tensor<1x1x1x3584xf16>
    // CHECK:       [[CONCAT_LEFT:%.+]] = IE.Concat([[RESHAPE_2]], [[RESHAPE_4]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x3584xf16>, tensor<1x1x1x3584xf16> -> tensor<1x2x1x3584xf16>
    // CHECK:       [[CONCAT_RIGHT:%.+]] = IE.Concat([[RESHAPE_1]], [[RESHAPE_3]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x3584xf16>, tensor<1x1x1x3584xf16> -> tensor<1x2x1x3584xf16>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[CONCAT_RIGHT]], [[CONCAT_LEFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x3584xf16>, tensor<1x2x1x3584xf16> -> tensor<1x2x1x3584xf16>

    // CHECK:       return [[MULTIPLY]] : tensor<1x2x1x3584xf16>
}


// -----

// CHECK-LABEL: @NotMovePostConcatForNonMultiplyParent
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x3584xf16>, [[INPUT2:%.+]]: tensor<1x3584xf16>, [[INPUT3:%.+]]: tensor<1x3584xf16>, [[INPUT4:%.+]]: tensor<1x3584xf16>
func.func @NotMovePostConcatForNonMultiplyParent(%arg0: tensor<1x3584xf16>, %arg1: tensor<1x3584xf16>, %arg2: tensor<1x3584xf16>, %arg3: tensor<1x3584xf16>) -> tensor<3x3584xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3584xf16>, tensor<1x3584xf16> -> tensor<1x3584xf16>
    %1 = IE.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3584xf16>, tensor<1x3584xf16> -> tensor<1x3584xf16>
    %2 = IE.Concat(%0, %1, %arg0) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x3584xf16>, tensor<1x3584xf16>, tensor<1x3584xf16> -> tensor<3x3584xf16>
    return %2 : tensor<3x3584xf16>

    // CHECK:       [[MULTIPLY_1:%.+]] = IE.Multiply([[INPUT1]], [[INPUT2]])
    // CHECK:       [[MULTIPLY_2:%.+]] = IE.Multiply([[INPUT3]], [[INPUT4]])
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[MULTIPLY_1]], [[MULTIPLY_2]], [[INPUT1]])

    // CHECK:       return  [[CONCAT]] : tensor<3x3584xf16>
}


// -----

// CHECK-LABEL: @NotMovePostConcatForShapeMismatch
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<2x3584xf16>, [[INPUT2:%.+]]: tensor<1x3584xf16>, [[INPUT3:%.+]]: tensor<1x3584xf16>, [[INPUT4:%.+]]: tensor<1x3584xf16>
func.func @NotMovePostConcatForShapeMismatch(%arg0: tensor<2x3584xf16>, %arg1: tensor<1x3584xf16>, %arg2: tensor<1x3584xf16>, %arg3: tensor<1x3584xf16>) -> tensor<3x3584xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3584xf16>, tensor<1x3584xf16> -> tensor<2x3584xf16>
    %1 = IE.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3584xf16>, tensor<1x3584xf16> -> tensor<1x3584xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<2x3584xf16>, tensor<1x3584xf16>-> tensor<3x3584xf16>
    return %2 : tensor<3x3584xf16>

    // CHECK:       [[MULTIPLY_1:%.+]] = IE.Multiply([[INPUT1]], [[INPUT2]])
    // CHECK:       [[MULTIPLY_2:%.+]] = IE.Multiply([[INPUT3]], [[INPUT4]])
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[MULTIPLY_1]], [[MULTIPLY_2]])

    // CHECK:       return  [[CONCAT]] : tensor<3x3584xf16>
}

// -----

// CHECK-LABEL: @SwapMultiplyWithFullyConnected
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<2048x12288xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x12288xf32>
func.func @SwapMultiplyWithFullyConnected(%arg0: tensor<2048x12288xf32>, %arg1: tensor<1x12288xf32>) -> tensor<1x2048xf32> {
    %cst = const.Declare tensor<1x1xf32> = dense<0.1> : tensor<1x1xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2048x12288xf32>, tensor<1x1xf32> -> tensor<2048x12288xf32>
    %1 = IE.FullyConnected(%arg1, %0) : tensor<1x12288xf32>, tensor<2048x12288xf32> -> tensor<1x2048xf32>

    return %1 : tensor<1x2048xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1xf32> = dense<1.000000e-01> : tensor<1x1xf32>
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[INPUT2]], [[INPUT1]]) : tensor<1x12288xf32>, tensor<2048x12288xf32> -> tensor<1x2048xf32>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[FC]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2048xf32>, tensor<1x1xf32> -> tensor<1x2048xf32>

    // CHECK:       return  [[MULTIPLY]] : tensor<1x2048xf32>
}

// -----

// CHECK-LABEL: @NotSwapMultiplyWithFullyConnectedWithBias
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<2048x12288xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x12288xf32>
func.func @NotSwapMultiplyWithFullyConnectedWithBias(%arg0: tensor<2048x12288xf32>, %arg1: tensor<1x12288xf32>) -> tensor<1x2048xf32> {
    %cst = const.Declare tensor<1x1xf32> = dense<0.1> : tensor<1x1xf32>
    %bias = const.Declare tensor<1x2048xf32> = dense<0.1> : tensor<1x2048xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2048x12288xf32>, tensor<1x1xf32> -> tensor<2048x12288xf32>
    %1 = IE.FullyConnected(%arg1, %0, %bias) : tensor<1x12288xf32>, tensor<2048x12288xf32>, tensor<1x2048xf32> -> tensor<1x2048xf32>

    return %1 : tensor<1x2048xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1xf32> = dense<1.000000e-01> : tensor<1x1xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x2048xf32> = dense<1.000000e-01> : tensor<1x2048xf32>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2048x12288xf32>, tensor<1x1xf32> -> tensor<2048x12288xf32>
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[INPUT2]], [[MULTIPLY]], [[BIAS]]) : tensor<1x12288xf32>, tensor<2048x12288xf32>, tensor<1x2048xf32> -> tensor<1x2048xf32>

    // CHECK:       return  [[FC]] : tensor<1x2048xf32>
}

// -----

// CHECK-LABEL: @SwapDivideWithMatmul
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @SwapDivideWithMatmul(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Divide(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[INPUT1]]) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>
    // CHECK:       [[DIVIDE:%.+]] = IE.Divide([[MATMUL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1x1024xf32>

    // CHECK:       return  [[DIVIDE]] : tensor<1x32x1x1024xf32>
}


// -----

// CHECK-LABEL: @SwapTwoDivideWithMatmul
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x80x1024xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x80x1024xf32>
func.func @SwapTwoDivideWithMatmul(%arg0: tensor<1x32x80x1024xf32>, %arg1: tensor<1x32x80x1024xf32>) -> tensor<1x32x80x80xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Divide(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x1024xf32>
    %cst0 = const.Declare tensor<1x1x1x1xf32> = dense<0.2> : tensor<1x1x1x1xf32>
    %1 = IE.Divide(%arg1, %cst0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x1024xf32>

    %2 = IE.MatMul(%0, %1) {transpose_b} : tensor<1x32x80x1024xf32>, tensor<1x32x80x1024xf32> -> tensor<1x32x80x80xf32>

    return %2 : tensor<1x32x80x80xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK:       [[MATMUL:%.+]]  = IE.MatMul([[INPUT1]], [[INPUT2]]) {transpose_b} : tensor<1x32x80x1024xf32>, tensor<1x32x80x1024xf32> -> tensor<1x32x80x80xf32>
    // CHECK:       [[DIVIDE:%.+]] = IE.Divide([[MATMUL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x80xf32>
    // CHECK:       [[DIVIDE_0:%.+]] = IE.Divide([[DIVIDE]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x80xf3

    // CHECK:       return  [[DIVIDE_0]] : tensor<1x32x80x80xf32>
}

// -----

// CHECK-LABEL: @NotSwapDivideForNotSplatConst
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x2xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x2xf32>
func.func @NotSwapDivideForNotSplatConst(%arg0: tensor<1x32x1024x2xf32>, %arg1: tensor<1x32x1x2xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x2xf32> = dense<[[[[1.0,1.6]]]]> : tensor<1x1x1x2xf32>
    %0 = IE.Divide(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x2xf32>, tensor<1x1x1x2xf32> -> tensor<1x32x1024x2xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x2xf32>, tensor<1x32x1024x2xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare
    // CHECK:       [[DIVIDE:%.+]] = IE.Divide([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[DIVIDE]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1x1024xf32>
}

// -----

// CHECK-LABEL: @NoSwapDivideForMultiUser
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @NoSwapDivideForMultiUser(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> (tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>) {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Divide(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %0, %1 : tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare
    // CHECK:       [[DIVIDE:%.+]] = IE.Divide([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[DIVIDE]]) {transpose_b}

    // CHECK:       return  [[MULTIPLY:%.+]], [[MATMUL]] : tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>
}


// -----

// CHECK-LABEL: @NotBeneficialForSwapDivide
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1025x80xf32>
func.func @NotBeneficialForSwapDivide(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1025x80xf32>) -> tensor<1x32x1025x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Divide(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1025x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1025x1024xf32>

    return %1 : tensor<1x32x1025x1024xf32>

    // CHECK:   [[CST:%.+]] = const.Declare
    // CHECK:       [[DIVIDE:%.+]] = IE.Divide([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[DIVIDE]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1025x1024xf32>
}

// -----

// CHECK-LABEL: @SwapDivideWithFullyConnected
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<2048x12288xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x12288xf32>
func.func @SwapDivideWithFullyConnected(%arg0: tensor<2048x12288xf32>, %arg1: tensor<1x12288xf32>) -> tensor<1x2048xf32> {
    %cst = const.Declare tensor<1x1xf32> = dense<0.1> : tensor<1x1xf32>
    %0 = IE.Divide(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2048x12288xf32>, tensor<1x1xf32> -> tensor<2048x12288xf32>
    %1 = IE.FullyConnected(%arg1, %0) : tensor<1x12288xf32>, tensor<2048x12288xf32> -> tensor<1x2048xf32>

    return %1 : tensor<1x2048xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1xf32> = dense<1.000000e-01> : tensor<1x1xf32>
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[INPUT2]], [[INPUT1]]) : tensor<1x12288xf32>, tensor<2048x12288xf32> -> tensor<1x2048xf32>
    // CHECK:       [[DIVIDE:%.+]] = IE.Divide([[FC]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2048xf32>, tensor<1x1xf32> -> tensor<1x2048xf32>

    // CHECK:       return  [[DIVIDE]] : tensor<1x2048xf32>
}

// -----

// CHECK-LABEL: @NotSwapDivideWithFullyConnectedWithBias
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<2048x12288xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x12288xf32>
func.func @NotSwapDivideWithFullyConnectedWithBias(%arg0: tensor<2048x12288xf32>, %arg1: tensor<1x12288xf32>) -> tensor<1x2048xf32> {
    %cst = const.Declare tensor<1x1xf32> = dense<0.1> : tensor<1x1xf32>
    %bias = const.Declare tensor<1x2048xf32> = dense<0.1> : tensor<1x2048xf32>
    %0 = IE.Divide(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2048x12288xf32>, tensor<1x1xf32> -> tensor<2048x12288xf32>
    %1 = IE.FullyConnected(%arg1, %0, %bias) : tensor<1x12288xf32>, tensor<2048x12288xf32>, tensor<1x2048xf32> -> tensor<1x2048xf32>

    return %1 : tensor<1x2048xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1xf32> = dense<1.000000e-01> : tensor<1x1xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x2048xf32> = dense<1.000000e-01> : tensor<1x2048xf32>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Divide([[INPUT1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2048x12288xf32>, tensor<1x1xf32> -> tensor<2048x12288xf32>
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[INPUT2]], [[MULTIPLY]], [[BIAS]]) : tensor<1x12288xf32>, tensor<2048x12288xf32>, tensor<1x2048xf32> -> tensor<1x2048xf32>

    // CHECK:       return  [[FC]] : tensor<1x2048xf32>
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @MultiplyDynamic
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-SAME:  [[INPUT_1:%.+]]: tensor<1x16x1x1xf16>

func.func @MultiplyDynamic(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x16x1x1xf16> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[RESULT:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x16x1x1xf16> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: return [[RESULT]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

// CHECK-LABEL: @SwapMultiplyWithMatmulSplatBroadcastConst
// When a splat const has been broadcast via IE.Tile to a multi-element shape
// (e.g. 1x4x32x128), moving it past MatMul would produce a shape mismatch.
// The pass must create a new scalar [1] const with the same splat value.
//
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x4x32x128xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x4x16x128xf32>
func.func @SwapMultiplyWithMatmulSplatBroadcastConst(
        %arg0: tensor<1x4x32x128xf32>,
        %arg1: tensor<1x4x16x128xf32>) -> tensor<1x4x16x32xf32> {
    // Splat const with same shape as arg0 — simulates the IE.Tile-broadcast pattern.
    // All elements are identical (splat), so it qualifies as "single data input".
    %cst = const.Declare tensor<1x4x32x128xf32> = dense<0.5> : tensor<1x4x32x128xf32>
    // Multiply input: 1x4x32x128.  After MatMul(arg1, ...) {transpose_b}:
    //   arg1=1x4x16x128, cst^T=1x4x128x32 → output 1x4x16x32 (incompatible with cst shape)
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x32x128xf32>, tensor<1x4x32x128xf32> -> tensor<1x4x32x128xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x4x16x128xf32>, tensor<1x4x32x128xf32> -> tensor<1x4x16x32xf32>

    return %1 : tensor<1x4x16x32xf32>

    // The original splat-broadcast const should NOT appear in the output.
    // CHECK-NOT: const.Declare tensor<1x4x32x128xf32>

    // A new scalar [1xf32] const with the same splat value must be created.
    // CHECK-DAG: [[SCALAR_CST:%.+]] = const.Declare tensor<1xf32> = dense<5.000000e-01> : tensor<1xf32>

    // MatMul is moved before Multiply.
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[INPUT1]]) {transpose_b} : tensor<1x4x16x128xf32>, tensor<1x4x32x128xf32> -> tensor<1x4x16x32xf32>

    // Multiply uses the scalar const (NUMPY broadcast).
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[MATMUL]], [[SCALAR_CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x16x32xf32>, tensor<1xf32> -> tensor<1x4x16x32xf32>
    // CHECK: return [[MULTIPLY]] : tensor<1x4x16x32xf32>
}

// -----

// CHECK-LABEL: @SwapMultiplyWithMatmulSplatBroadcastConst_AutoBroadcastRewrite
// When the original Multiply uses NONE (shapes are identical), creating a scalar const
// requires rewriting the cloned op's auto_broadcast to NUMPY for verification to pass.
//
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x4x32x128xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x4x16x128xf32>
func.func @SwapMultiplyWithMatmulSplatBroadcastConst_AutoBroadcastRewrite(
        %arg0: tensor<1x4x32x128xf32>,
        %arg1: tensor<1x4x16x128xf32>) -> tensor<1x4x16x32xf32> {
    %cst = const.Declare tensor<1x4x32x128xf32> = dense<0.5> : tensor<1x4x32x128xf32>
    // Original Multiply uses NONE_OR_EXPLICIT because shapes are identical (no actual broadcast needed).
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x32x128xf32>, tensor<1x4x32x128xf32> -> tensor<1x4x32x128xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x4x16x128xf32>, tensor<1x4x32x128xf32> -> tensor<1x4x16x32xf32>

    return %1 : tensor<1x4x16x32xf32>

    // CHECK-NOT: const.Declare tensor<1x4x32x128xf32>
    // CHECK-DAG: [[SCALAR_CST:%.+]] = const.Declare tensor<1xf32> = dense<5.000000e-01> : tensor<1xf32>
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[INPUT1]]) {transpose_b} : tensor<1x4x16x128xf32>, tensor<1x4x32x128xf32> -> tensor<1x4x16x32xf32>

    // The rewritten Multiply MUST use NUMPY (not NONE_OR_EXPLICIT) to broadcast the scalar const.
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[MATMUL]], [[SCALAR_CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x16x32xf32>, tensor<1xf32> -> tensor<1x4x16x32xf32>
    // CHECK: return [[MULTIPLY]] : tensor<1x4x16x32xf32>
}

// -----

// CHECK-LABEL: @SwapDivideWithMatmulSplatBroadcastConst_AutoBroadcastRewrite
// Same test for IE.Divide to ensure both Multiply and Divide handle the auto_broadcast rewrite.
//
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x4x32x128xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x4x16x128xf32>
func.func @SwapDivideWithMatmulSplatBroadcastConst_AutoBroadcastRewrite(
        %arg0: tensor<1x4x32x128xf32>,
        %arg1: tensor<1x4x16x128xf32>) -> tensor<1x4x16x32xf32> {
    %cst = const.Declare tensor<1x4x32x128xf32> = dense<2.0> : tensor<1x4x32x128xf32>
    // Original Divide uses NONE_OR_EXPLICIT because shapes are identical.
    %0 = IE.Divide(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x32x128xf32>, tensor<1x4x32x128xf32> -> tensor<1x4x32x128xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x4x16x128xf32>, tensor<1x4x32x128xf32> -> tensor<1x4x16x32xf32>

    return %1 : tensor<1x4x16x32xf32>

    // CHECK-NOT: const.Declare tensor<1x4x32x128xf32>
    // CHECK-DAG: [[SCALAR_CST:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[INPUT2]], [[INPUT1]]) {transpose_b} : tensor<1x4x16x128xf32>, tensor<1x4x32x128xf32> -> tensor<1x4x16x32xf32>

    // The rewritten Divide MUST use NUMPY (not NONE_OR_EXPLICIT) to broadcast the scalar const.
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[MATMUL]], [[SCALAR_CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x16x32xf32>, tensor<1xf32> -> tensor<1x4x16x32xf32>
    // CHECK: return [[DIVIDE]] : tensor<1x4x16x32xf32>
}
