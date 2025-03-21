//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --move-multiply-post-op %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// -----

// CHECK-LABEL: @SwapMultiplyWithMatmul
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @SwapMultiplyWithMatmul(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[INPUT1]]) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[MATMUL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1x1024xf32>

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

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK:       [[MATMUL:%.*]]  = IE.MatMul([[INPUT1]], [[INPUT2]]) {transpose_b} : tensor<1x32x80x1024xf32>, tensor<1x32x80x1024xf32> -> tensor<1x32x80x80xf32>
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[MATMUL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x80xf32>
    // CHECK:       [[MULTIPLY_0:%.*]] = IE.Multiply([[MULTIPLY]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x80xf3

    // CHECK:       return  [[MULTIPLY_0]] : tensor<1x32x80x80xf32>
}


// -----

// CHECK-LABEL: @NotSwapWithPostOp
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @NotSwapWithPostOp(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>} : tensor<1x1x1x1xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1x1024xf32>
}

// -----

// CHECK-LABEL: @NotSwapForNotSplatConst
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x2xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x2xf32>
func.func @NotSwapForNotSplatConst(%arg0: tensor<1x32x1024x2xf32>, %arg1: tensor<1x32x1x2xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x2xf32> = dense<[[[[1.0,1.6]]]]> : tensor<1x1x1x2xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x2xf32>, tensor<1x1x1x2xf32> -> tensor<1x32x1024x2xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x2xf32>, tensor<1x32x1024x2xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1x1024xf32>
}

// -----

// CHECK-LABEL: @NoSwapForMultiUser
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @NoSwapForMultiUser(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> (tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>) {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %0, %1 : tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MULTIPLY:%.*]], [[MATMUL]] : tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>
}


// -----

// CHECK-LABEL: @NotBeneficialForSwap
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1025x80xf32>
func.func @NotBeneficialForSwap(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1025x80xf32>) -> tensor<1x32x1025x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>} : tensor<1x1x1x1xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1025x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1025x1024xf32>

    return %1 : tensor<1x32x1025x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

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
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply(%5, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x3584xf16>, tensor<1x2x1x3584xf16> -> tensor<1x2x1x3584xf16>

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
