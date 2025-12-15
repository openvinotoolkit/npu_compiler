//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-concat-matmul %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @DecomposeConcatMatMul
// CHECK-SAME: ([[ARG0:%arg[0-9]+]]: tensor<1x256x2048x1xf32>, [[ARG1:%arg[0-9]+]]: tensor<1x256x2048x1xf32>,
// CHECK-SAME: [[ARG2:%arg[0-9]+]]: tensor<1x256x2048x1xf32>, [[ARG3:%arg[0-9]+]]: tensor<1x256x2048x1xf32>,
// CHECK-SAME: [[ARG4:%arg[0-9]+]]: tensor<1x256x2x4xf32>) -> tensor<1x256x2048x2xf32>
func.func @DecomposeConcatMatMul(%arg0: tensor<1x256x2048x1xf32>, %arg1: tensor<1x256x2048x1xf32>,
%arg2: tensor<1x256x2048x1xf32>, %arg3: tensor<1x256x2048x1xf32>, %arg4: tensor<1x256x2x4xf32>) -> tensor<1x256x2048x2xf32> {
    %concat = IE.Concat(%arg0, %arg1, %arg2, %arg3) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]]} : tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32> -> tensor<1x256x2048x4xf32>
    %matmul = IE.MatMul(%concat, %arg4) {transpose_b} : tensor<1x256x2048x4xf32>, tensor<1x256x2x4xf32> -> tensor<1x256x2048x2xf32>

    return %matmul : tensor<1x256x2048x2xf32>

    // Check that the Concat+MatMul is decomposed into Multiply+Add operations
    // CHECK-NOT: IE.MatMul

    // Check weight slicing along K dimension (last dimension of input2)
    // CHECK: [[SLICE0:%.+]] = IE.Slice [[ARG4]] [0, 0, 0, 0] [1, 256, 2, 1] : tensor<1x256x2x4xf32> to tensor<1x256x2x1xf32>
    // CHECK: [[SLICE1:%.+]] = IE.Slice [[ARG4]] [0, 0, 0, 1] [1, 256, 2, 1] : tensor<1x256x2x4xf32> to tensor<1x256x2x1xf32>
    // CHECK: [[SLICE2:%.+]] = IE.Slice [[ARG4]] [0, 0, 0, 2] [1, 256, 2, 1] : tensor<1x256x2x4xf32> to tensor<1x256x2x1xf32>
    // CHECK: [[SLICE3:%.+]] = IE.Slice [[ARG4]] [0, 0, 0, 3] [1, 256, 2, 1] : tensor<1x256x2x4xf32> to tensor<1x256x2x1xf32>

    // Check further slicing along output dimension (second-to-last dimension)
    // CHECK: [[SUBSLICE0_0:%.+]] = IE.Slice [[SLICE0]] [0, 0, 0, 0] [1, 256, 1, 1] : tensor<1x256x2x1xf32> to tensor<1x256x1x1xf32>
    // CHECK: [[SUBSLICE0_1:%.+]] = IE.Slice [[SLICE0]] [0, 0, 1, 0] [1, 256, 1, 1] : tensor<1x256x2x1xf32> to tensor<1x256x1x1xf32>
    // CHECK: [[SUBSLICE1_0:%.+]] = IE.Slice [[SLICE1]] [0, 0, 0, 0] [1, 256, 1, 1] : tensor<1x256x2x1xf32> to tensor<1x256x1x1xf32>
    // CHECK: [[SUBSLICE1_1:%.+]] = IE.Slice [[SLICE1]] [0, 0, 1, 0] [1, 256, 1, 1] : tensor<1x256x2x1xf32> to tensor<1x256x1x1xf32>
    // CHECK: [[SUBSLICE2_0:%.+]] = IE.Slice [[SLICE2]] [0, 0, 0, 0] [1, 256, 1, 1] : tensor<1x256x2x1xf32> to tensor<1x256x1x1xf32>
    // CHECK: [[SUBSLICE2_1:%.+]] = IE.Slice [[SLICE2]] [0, 0, 1, 0] [1, 256, 1, 1] : tensor<1x256x2x1xf32> to tensor<1x256x1x1xf32>
    // CHECK: [[SUBSLICE3_0:%.+]] = IE.Slice [[SLICE3]] [0, 0, 0, 0] [1, 256, 1, 1] : tensor<1x256x2x1xf32> to tensor<1x256x1x1xf32>
    // CHECK: [[SUBSLICE3_1:%.+]] = IE.Slice [[SLICE3]] [0, 0, 1, 0] [1, 256, 1, 1] : tensor<1x256x2x1xf32> to tensor<1x256x1x1xf32>

    // Check element-wise multiplications for output channel 0
    // CHECK: [[MUL0_0:%.+]] = IE.Multiply([[ARG0]], [[SUBSLICE0_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x1x1xf32> -> tensor<1x256x2048x1xf32>
    // CHECK: [[MUL1_0:%.+]] = IE.Multiply([[ARG1]], [[SUBSLICE1_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x1x1xf32> -> tensor<1x256x2048x1xf32>
    // CHECK: [[MUL2_0:%.+]] = IE.Multiply([[ARG2]], [[SUBSLICE2_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x1x1xf32> -> tensor<1x256x2048x1xf32>
    // CHECK: [[MUL3_0:%.+]] = IE.Multiply([[ARG3]], [[SUBSLICE3_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x1x1xf32> -> tensor<1x256x2048x1xf32>

    // Check additions for output channel 0
    // CHECK: [[ADD0_0:%.+]] = IE.Add([[MUL0_0]], [[MUL1_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32> -> tensor<1x256x2048x1xf32>
    // CHECK: [[ADD1_0:%.+]] = IE.Add([[ADD0_0]], [[MUL2_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32> -> tensor<1x256x2048x1xf32>
    // CHECK: [[RESULT0:%.+]] = IE.Add([[ADD1_0]], [[MUL3_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32> -> tensor<1x256x2048x1xf32>

    // Check element-wise multiplications for output channel 1
    // CHECK: [[MUL0_1:%.+]] = IE.Multiply([[ARG0]], [[SUBSLICE0_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x1x1xf32> -> tensor<1x256x2048x1xf32>
    // CHECK: [[MUL1_1:%.+]] = IE.Multiply([[ARG1]], [[SUBSLICE1_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x1x1xf32> -> tensor<1x256x2048x1xf32>
    // CHECK: [[MUL2_1:%.+]] = IE.Multiply([[ARG2]], [[SUBSLICE2_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x1x1xf32> -> tensor<1x256x2048x1xf32>
    // CHECK: [[MUL3_1:%.+]] = IE.Multiply([[ARG3]], [[SUBSLICE3_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x1x1xf32> -> tensor<1x256x2048x1xf32>

    // Check additions for output channel 1
    // CHECK: [[ADD0_1:%.+]] = IE.Add([[MUL0_1]], [[MUL1_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32> -> tensor<1x256x2048x1xf32>
    // CHECK: [[ADD1_1:%.+]] = IE.Add([[ADD0_1]], [[MUL2_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32> -> tensor<1x256x2048x1xf32>
    // CHECK: [[RESULT1:%.+]] = IE.Add([[ADD1_1]], [[MUL3_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32> -> tensor<1x256x2048x1xf32>

    // Check final concatenation along the last dimension
    // CHECK: [[FINAL_RESULT:%.+]] = IE.Concat([[RESULT0]], [[RESULT1]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32> -> tensor<1x256x2048x2xf32>

    // CHECK: return [[FINAL_RESULT]] : tensor<1x256x2048x2xf32>
}

// -----

// CHECK-LABEL: @DecomposeConcatMatMul3Inputs
// CHECK-SAME: ([[ARG0:%arg[0-9]+]]: tensor<1x128x1024x1xf32>, [[ARG1:%arg[0-9]+]]: tensor<1x128x1024x1xf32>,
// CHECK-SAME: [[ARG2:%arg[0-9]+]]: tensor<1x128x1024x1xf32>, [[ARG3:%arg[0-9]+]]: tensor<1x128x3x3xf32>) -> tensor<1x128x1024x3xf32>
func.func @DecomposeConcatMatMul3Inputs(%arg0: tensor<1x128x1024x1xf32>, %arg1: tensor<1x128x1024x1xf32>,
%arg2: tensor<1x128x1024x1xf32>, %arg3: tensor<1x128x3x3xf32>) -> tensor<1x128x1024x3xf32> {
    %concat = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]} : tensor<1x128x1024x1xf32>, tensor<1x128x1024x1xf32>, tensor<1x128x1024x1xf32> -> tensor<1x128x1024x3xf32>
    %matmul = IE.MatMul(%concat, %arg3) {transpose_b} : tensor<1x128x1024x3xf32>, tensor<1x128x3x3xf32> -> tensor<1x128x1024x3xf32>

    return %matmul : tensor<1x128x1024x3xf32>

    // Check that the Concat+MatMul is decomposed
    // CHECK-NOT: IE.MatMul

    // Check weight slicing along K dimension (last dimension of input2)
    // CHECK: [[SLICE0:%.+]] = IE.Slice [[ARG3]] [0, 0, 0, 0] [1, 128, 3, 1] : tensor<1x128x3x3xf32> to tensor<1x128x3x1xf32>
    // CHECK: [[SLICE1:%.+]] = IE.Slice [[ARG3]] [0, 0, 0, 1] [1, 128, 3, 1] : tensor<1x128x3x3xf32> to tensor<1x128x3x1xf32>
    // CHECK: [[SLICE2:%.+]] = IE.Slice [[ARG3]] [0, 0, 0, 2] [1, 128, 3, 1] : tensor<1x128x3x3xf32> to tensor<1x128x3x1xf32>

    // Check further slicing along output dimension (second-to-last dimension)
    // For output channel 0
    // CHECK-DAG: [[SUBSLICE0_0:%.+]] = IE.Slice [[SLICE0]] [0, 0, 0, 0] [1, 128, 1, 1] : tensor<1x128x3x1xf32> to tensor<1x128x1x1xf32>
    // CHECK-DAG: [[SUBSLICE1_0:%.+]] = IE.Slice [[SLICE1]] [0, 0, 0, 0] [1, 128, 1, 1] : tensor<1x128x3x1xf32> to tensor<1x128x1x1xf32>
    // CHECK-DAG: [[SUBSLICE2_0:%.+]] = IE.Slice [[SLICE2]] [0, 0, 0, 0] [1, 128, 1, 1] : tensor<1x128x3x1xf32> to tensor<1x128x1x1xf32>

    // For output channel 1
    // CHECK-DAG: [[SUBSLICE0_1:%.+]] = IE.Slice [[SLICE0]] [0, 0, 1, 0] [1, 128, 1, 1] : tensor<1x128x3x1xf32> to tensor<1x128x1x1xf32>
    // CHECK-DAG: [[SUBSLICE1_1:%.+]] = IE.Slice [[SLICE1]] [0, 0, 1, 0] [1, 128, 1, 1] : tensor<1x128x3x1xf32> to tensor<1x128x1x1xf32>
    // CHECK-DAG: [[SUBSLICE2_1:%.+]] = IE.Slice [[SLICE2]] [0, 0, 1, 0] [1, 128, 1, 1] : tensor<1x128x3x1xf32> to tensor<1x128x1x1xf32>

    // For output channel 2
    // CHECK-DAG: [[SUBSLICE0_2:%.+]] = IE.Slice [[SLICE0]] [0, 0, 2, 0] [1, 128, 1, 1] : tensor<1x128x3x1xf32> to tensor<1x128x1x1xf32>
    // CHECK-DAG: [[SUBSLICE1_2:%.+]] = IE.Slice [[SLICE1]] [0, 0, 2, 0] [1, 128, 1, 1] : tensor<1x128x3x1xf32> to tensor<1x128x1x1xf32>
    // CHECK-DAG: [[SUBSLICE2_2:%.+]] = IE.Slice [[SLICE2]] [0, 0, 2, 0] [1, 128, 1, 1] : tensor<1x128x3x1xf32> to tensor<1x128x1x1xf32>

    // Check element-wise multiplications for output channel 0
    // CHECK: [[MUL0_0:%.+]] = IE.Multiply([[ARG0]], [[SUBSLICE0_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x1024x1xf32>
    // CHECK: [[MUL1_0:%.+]] = IE.Multiply([[ARG1]], [[SUBSLICE1_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x1024x1xf32>
    // CHECK: [[MUL2_0:%.+]] = IE.Multiply([[ARG2]], [[SUBSLICE2_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x1024x1xf32>

    // Check additions for output channel 0
    // CHECK: [[ADD0_0:%.+]] = IE.Add([[MUL0_0]], [[MUL1_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1024x1xf32> -> tensor<1x128x1024x1xf32>
    // CHECK: [[RESULT0:%.+]] = IE.Add([[ADD0_0]], [[MUL2_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1024x1xf32> -> tensor<1x128x1024x1xf32>

    // Check element-wise multiplications for output channel 1
    // CHECK: [[MUL0_1:%.+]] = IE.Multiply([[ARG0]], [[SUBSLICE0_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x1024x1xf32>
    // CHECK: [[MUL1_1:%.+]] = IE.Multiply([[ARG1]], [[SUBSLICE1_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x1024x1xf32>
    // CHECK: [[MUL2_1:%.+]] = IE.Multiply([[ARG2]], [[SUBSLICE2_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x1024x1xf32>

    // Check additions for output channel 1
    // CHECK: [[ADD0_1:%.+]] = IE.Add([[MUL0_1]], [[MUL1_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1024x1xf32> -> tensor<1x128x1024x1xf32>
    // CHECK: [[RESULT1:%.+]] = IE.Add([[ADD0_1]], [[MUL2_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1024x1xf32> -> tensor<1x128x1024x1xf32>

    // Check element-wise multiplications for output channel 2
    // CHECK: [[MUL0_2:%.+]] = IE.Multiply([[ARG0]], [[SUBSLICE0_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x1024x1xf32>
    // CHECK: [[MUL1_2:%.+]] = IE.Multiply([[ARG1]], [[SUBSLICE1_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x1024x1xf32>
    // CHECK: [[MUL2_2:%.+]] = IE.Multiply([[ARG2]], [[SUBSLICE2_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x1024x1xf32>

    // Check additions for output channel 2
    // CHECK: [[ADD0_2:%.+]] = IE.Add([[MUL0_2]], [[MUL1_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1024x1xf32> -> tensor<1x128x1024x1xf32>
    // CHECK: [[RESULT2:%.+]] = IE.Add([[ADD0_2]], [[MUL2_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1024x1xf32>, tensor<1x128x1024x1xf32> -> tensor<1x128x1024x1xf32>

    // Check final concatenation along the last dimension
    // CHECK: [[FINAL_RESULT:%.+]] = IE.Concat([[RESULT0]], [[RESULT1]], [[RESULT2]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x128x1024x1xf32>, tensor<1x128x1024x1xf32>, tensor<1x128x1024x1xf32> -> tensor<1x128x1024x3xf32>

    // CHECK: return [[FINAL_RESULT]] : tensor<1x128x1024x3xf32>
}

// -----

// CHECK-LABEL: @NoDecomposeNoTransposeB
func.func @NoDecomposeNoTransposeB(%arg0: tensor<1x256x2048x1xf32>, %arg1: tensor<1x256x2048x1xf32>,
%arg2: tensor<1x256x2048x1xf32>, %arg3: tensor<1x256x2048x1xf32>, %arg4: tensor<1x256x4x2xf32>) -> tensor<1x256x2048x2xf32> {
    %concat = IE.Concat(%arg0, %arg1, %arg2, %arg3) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]]} : tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32> -> tensor<1x256x2048x4xf32>
    %matmul = IE.MatMul(%concat, %arg4) : tensor<1x256x2048x4xf32>, tensor<1x256x4x2xf32> -> tensor<1x256x2048x2xf32>

    return %matmul : tensor<1x256x2048x2xf32>

    // Should not decompose since transpose_b is not set
    // CHECK: IE.Concat
    // CHECK: IE.MatMul
    // CHECK-NOT: IE.Slice
    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Add
    }

// -----

// CHECK-LABEL: @NoDecomposeTooManyInputs
func.func @NoDecomposeTooManyInputs(%arg0: tensor<1x256x2048x1xf32>, %arg1: tensor<1x256x2048x1xf32>,
    %arg2: tensor<1x256x2048x1xf32>, %arg3: tensor<1x256x2048x1xf32>,
    %arg4: tensor<1x256x2048x1xf32>,
    %arg5: tensor<1x256x2x5xf32>) -> tensor<1x256x2048x2xf32> {
    %concat = IE.Concat(%arg0, %arg1, %arg2, %arg3, %arg4) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4]]} : tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32>, tensor<1x256x2048x1xf32> -> tensor<1x256x2048x5xf32>
    %matmul = IE.MatMul(%concat, %arg5) {transpose_b} : tensor<1x256x2048x5xf32>, tensor<1x256x2x5xf32> -> tensor<1x256x2048x2xf32>

    return %matmul : tensor<1x256x2048x2xf32>

    // Should not decompose since there are more than 4 input slices
    // CHECK: IE.Concat
    // CHECK: IE.MatMul
    // CHECK-NOT: IE.Slice
    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Add
}

// -----

// CHECK-LABEL: @NoDecomposeWrongAxis
func.func @NoDecomposeWrongAxis(%arg0: tensor<1x256x512x4xf32>, %arg1: tensor<1x256x512x4xf32>,
%arg2: tensor<1x256x512x4xf32>, %arg3: tensor<1x256x512x4xf32>, %arg4: tensor<1x256x2x4xf32>) -> tensor<1x256x2048x2xf32> {
    %concat = IE.Concat(%arg0, %arg1, %arg2, %arg3) {static_offsets = [[0, 0, 0, 0], [0, 0, 512, 0], [0, 0, 1024, 0], [0, 0, 1536, 0]]} : tensor<1x256x512x4xf32>, tensor<1x256x512x4xf32>, tensor<1x256x512x4xf32>, tensor<1x256x512x4xf32> -> tensor<1x256x2048x4xf32>
    %matmul = IE.MatMul(%concat, %arg4) {transpose_b} : tensor<1x256x2048x4xf32>, tensor<1x256x2x4xf32> -> tensor<1x256x2048x2xf32>

    return %matmul : tensor<1x256x2048x2xf32>

    // Should not decompose since concat is along wrong axis (axis 2 instead of axis 3)
    // CHECK: IE.Concat
    // CHECK: IE.MatMul
    // CHECK-NOT: IE.Slice
    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Add
}

// -----

// CHECK-LABEL: @DecomposeMinimalCase
// CHECK-SAME: ([[ARG0:%arg[0-9]+]]: tensor<1x64x512x1xf32>, [[ARG1:%arg[0-9]+]]: tensor<1x64x512x1xf32>,
// CHECK-SAME: [[ARG2:%arg[0-9]+]]: tensor<1x64x1x2xf32>) -> tensor<1x64x512x1xf32>
func.func @DecomposeMinimalCase(%arg0: tensor<1x64x512x1xf32>, %arg1: tensor<1x64x512x1xf32>,
%arg2: tensor<1x64x1x2xf32>) -> tensor<1x64x512x1xf32> {
    %concat = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1]]} : tensor<1x64x512x1xf32>, tensor<1x64x512x1xf32> -> tensor<1x64x512x2xf32>
    %matmul = IE.MatMul(%concat, %arg2) {transpose_b} : tensor<1x64x512x2xf32>, tensor<1x64x1x2xf32> -> tensor<1x64x512x1xf32>

    return %matmul : tensor<1x64x512x1xf32>

    // Check that the Concat+MatMul is decomposed
    // CHECK-NOT: IE.MatMul

    // Check weight slicing along K dimension (last dimension of input2)
    // CHECK: [[SLICE0:%.+]] = IE.Slice [[ARG2]] [0, 0, 0, 0] [1, 64, 1, 1] : tensor<1x64x1x2xf32> to tensor<1x64x1x1xf32>
    // CHECK: [[SLICE1:%.+]] = IE.Slice [[ARG2]] [0, 0, 0, 1] [1, 64, 1, 1] : tensor<1x64x1x2xf32> to tensor<1x64x1x1xf32>

    // Check element-wise multiplications (only one output channel)
    // CHECK: [[MUL0:%.+]] = IE.Multiply([[ARG0]], [[SLICE0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x512x1xf32>, tensor<1x64x1x1xf32> -> tensor<1x64x512x1xf32>
    // CHECK: [[MUL1:%.+]] = IE.Multiply([[ARG1]], [[SLICE1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x512x1xf32>, tensor<1x64x1x1xf32> -> tensor<1x64x512x1xf32>

    // Check addition to combine results
    // CHECK: [[FINAL_RESULT:%.+]] = IE.Add([[MUL0]], [[MUL1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x512x1xf32>, tensor<1x64x512x1xf32> -> tensor<1x64x512x1xf32>

    // CHECK: return [[FINAL_RESULT]] : tensor<1x64x512x1xf32>
}

// -----

// CHECK-LABEL: @DecomposeSingleOutputChannel
// CHECK-SAME: ([[ARG0:%arg[0-9]+]]: tensor<1x32x256x1xf32>, [[ARG1:%arg[0-9]+]]: tensor<1x32x256x1xf32>,
// CHECK-SAME: [[ARG2:%arg[0-9]+]]: tensor<1x32x256x1xf32>, [[ARG3:%arg[0-9]+]]: tensor<1x32x1x3xf32>) -> tensor<1x32x256x1xf32>
func.func @DecomposeSingleOutputChannel(%arg0: tensor<1x32x256x1xf32>, %arg1: tensor<1x32x256x1xf32>,
%arg2: tensor<1x32x256x1xf32>, %arg3: tensor<1x32x1x3xf32>) -> tensor<1x32x256x1xf32> {
    %concat = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]} : tensor<1x32x256x1xf32>, tensor<1x32x256x1xf32>, tensor<1x32x256x1xf32> -> tensor<1x32x256x3xf32>
    %matmul = IE.MatMul(%concat, %arg3) {transpose_b} : tensor<1x32x256x3xf32>, tensor<1x32x1x3xf32> -> tensor<1x32x256x1xf32>

    return %matmul : tensor<1x32x256x1xf32>

    // Check that the Concat+MatMul is decomposed
    // CHECK-NOT: IE.MatMul

    // Check weight slicing along K dimension (last dimension of input2)
    // CHECK: [[SLICE0:%.+]] = IE.Slice [[ARG3]] [0, 0, 0, 0] [1, 32, 1, 1] : tensor<1x32x1x3xf32> to tensor<1x32x1x1xf32>
    // CHECK: [[SLICE1:%.+]] = IE.Slice [[ARG3]] [0, 0, 0, 1] [1, 32, 1, 1] : tensor<1x32x1x3xf32> to tensor<1x32x1x1xf32>
    // CHECK: [[SLICE2:%.+]] = IE.Slice [[ARG3]] [0, 0, 0, 2] [1, 32, 1, 1] : tensor<1x32x1x3xf32> to tensor<1x32x1x1xf32>

    // Check element-wise multiplications for single output channel
    // CHECK: [[MUL0:%.+]] = IE.Multiply([[ARG0]], [[SLICE0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x256x1xf32>, tensor<1x32x1x1xf32> -> tensor<1x32x256x1xf32>
    // CHECK: [[MUL1:%.+]] = IE.Multiply([[ARG1]], [[SLICE1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x256x1xf32>, tensor<1x32x1x1xf32> -> tensor<1x32x256x1xf32>
    // CHECK: [[MUL2:%.+]] = IE.Multiply([[ARG2]], [[SLICE2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x256x1xf32>, tensor<1x32x1x1xf32> -> tensor<1x32x256x1xf32>

    // Check additions to combine results (2 additions for 3 terms)
    // CHECK: [[ADD0:%.+]] = IE.Add([[MUL0]], [[MUL1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x256x1xf32>, tensor<1x32x256x1xf32> -> tensor<1x32x256x1xf32>
    // CHECK: [[FINAL_RESULT:%.+]] = IE.Add([[ADD0]], [[MUL2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x256x1xf32>, tensor<1x32x256x1xf32> -> tensor<1x32x256x1xf32>

    // CHECK: return [[FINAL_RESULT]] : tensor<1x32x256x1xf32>
}
