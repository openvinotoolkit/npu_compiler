//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --run-batch-op-processing-rewriters="enable-grouped-matmul=true rewriter=matmul-inputs-to-2d-set" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @MatMulInputsTo2dNotConverted
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<6x2x512xf32>
func.func @MatMulInputsTo2dNotConverted(%arg0: tensor<6x2x512xf32>) -> tensor<6x2x40xf32> {
    %cst = const.Declare tensor<6x512x40xf32> = dense<1.0> : tensor<6x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<6x2x512xf32>, tensor<6x512x40xf32> -> tensor<6x2x40xf32>

    return %0 : tensor<6x2x40xf32>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<6x512x40xf32> = dense<1.000000e+00> : tensor<6x512x40xf32>
    // CHECK:   [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[CST]]) : tensor<6x2x512xf32>, tensor<6x512x40xf32> -> tensor<6x2x40xf32>
    // CHECK:   return [[MATMUL]] : tensor<6x2x40xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInputsTo2dNotConverted
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x7x2x512xf32>
func.func @MatMul4dInputsTo2dNotConverted(%arg0: tensor<1x7x2x512xf32>) -> tensor<1x7x2x40xf32> {
    %cst = const.Declare tensor<1x7x512x40xf32> = dense<1.0> : tensor<1x7x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x7x2x512xf32>, tensor<1x7x512x40xf32> -> tensor<1x7x2x40xf32>

    return %0 : tensor<1x7x2x40xf32>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<1x7x512x40xf32> = dense<1.000000e+00> : tensor<1x7x512x40xf32>
    // CHECK:   [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[CST]]) : tensor<1x7x2x512xf32>, tensor<1x7x512x40xf32> -> tensor<1x7x2x40xf32>
    // CHECK:   return [[MATMUL]] : tensor<1x7x2x40xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInputs4dWeightsTo2dNotConverted
// CHECK-SAME:  [[ARG0:%.+]]: tensor<3x2x10x3xf32>, [[ARG1:%.+]]: tensor<3x2x10x3xf32>
func.func @MatMul4dInputs4dWeightsTo2dNotConverted(%arg0: tensor<3x2x10x3xf32>, %arg1: tensor<3x2x10x3xf32>) -> tensor<3x2x10x10xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<3x2x10x3xf32>, tensor<3x2x10x3xf32> -> tensor<3x2x10x10xf32>

    return %0 : tensor<3x2x10x10xf32>

    // CHECK:   [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b} : tensor<3x2x10x3xf32>, tensor<3x2x10x3xf32> -> tensor<3x2x10x10xf32>
    // CHECK:   return [[MATMUL]] : tensor<3x2x10x10xf32>
}

// -----
// CHECK-LABEL: @MatMul4dInputs4dWeightsTo2dTooBigPerGroup
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x2x4000x4000xf32>, [[ARG1:%.+]]: tensor<1x2x3000x4000xf32>
func.func @MatMul4dInputs4dWeightsTo2dTooBigPerGroup(%arg0: tensor<1x2x4000x4000xf32>, %arg1: tensor<1x2x3000x4000xf32>) -> tensor<1x2x4000x3000xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x2x4000x4000xf32>, tensor<1x2x3000x4000xf32> -> tensor<1x2x4000x3000xf32>

    return %0 : tensor<1x2x4000x3000xf32>

    // CHECK: [[SLICE0:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 1, 4000, 4000] : tensor<1x2x4000x4000xf32> to tensor<1x1x4000x4000xf32>
    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[SLICE0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [4000, 4000]} : tensor<1x1x4000x4000xf32> -> tensor<4000x4000xf32>
    // CHECK: [[SLICE1:%.+]] = IE.Slice [[ARG0]] [0, 1, 0, 0] [1, 1, 4000, 4000] : tensor<1x2x4000x4000xf32> to tensor<1x1x4000x4000xf32>
    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[SLICE1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [4000, 4000]} : tensor<1x1x4000x4000xf32> -> tensor<4000x4000xf32>
    // CHECK: [[SLICE2:%.+]] = IE.Slice [[ARG1]] [0, 0, 0, 0] [1, 1, 3000, 4000] : tensor<1x2x3000x4000xf32> to tensor<1x1x3000x4000xf32>
    // CHECK: [[RESHAPE2:%.+]] = IE.AffineReshape([[SLICE2]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [3000, 4000]} : tensor<1x1x3000x4000xf32> -> tensor<3000x4000xf32>
    // CHECK: [[SLICE3:%.+]] = IE.Slice [[ARG1]] [0, 1, 0, 0] [1, 1, 3000, 4000] : tensor<1x2x3000x4000xf32> to tensor<1x1x3000x4000xf32>
    // CHECK: [[RESHAPE3:%.+]] = IE.AffineReshape([[SLICE3]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [3000, 4000]} : tensor<1x1x3000x4000xf32> -> tensor<3000x4000xf32>
    // CHECK: [[MATMUL0:%.+]] = IE.MatMul([[RESHAPE0]], [[RESHAPE2]]) {transpose_b} : tensor<4000x4000xf32>, tensor<3000x4000xf32> -> tensor<4000x3000xf32>
    // CHECK: [[MATMUL1:%.+]] = IE.MatMul([[RESHAPE1]], [[RESHAPE3]]) {transpose_b} : tensor<4000x4000xf32>, tensor<3000x4000xf32> -> tensor<4000x3000xf32>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[MATMUL0]], [[MATMUL1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<4000x3000xf32>, tensor<4000x3000xf32> -> tensor<8000x3000xf32>
    // CHECK: [[RESHAPE4:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 4000, 3000]} : tensor<8000x3000xf32> -> tensor<1x2x4000x3000xf32>
    // CHECK: return [[RESHAPE4]] : tensor<1x2x4000x3000xf32>
}

// -----
// CHECK-LABEL: @MatMul4dInputs4dWeightsTo2dSmallOutputPerGroup
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x6x300x400xf32>, [[ARG1:%.+]]: tensor<1x6x300x400xf32>
func.func @MatMul4dInputs4dWeightsTo2dSmallOutputPerGroup(%arg0:  tensor<1x6x300x400xf32>, %arg1: tensor<1x6x300x400xf32>) ->  tensor<1x6x300x300xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x6x300x400xf32>, tensor<1x6x300x400xf32> -> tensor<1x6x300x300xf32>

    return %0 : tensor<1x6x300x300xf32>

    // CHECK:   [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) {transpose_b} : tensor<1x6x300x400xf32>, tensor<1x6x300x400xf32> -> tensor<1x6x300x300xf32>
    // CHECK:   return [[MATMUL]] : tensor<1x6x300x300xf32>
}

// -----
// CHECK-LABEL: @MatMul4dInputs4dWeightsTo2dBigOutputPerGroup
// CHECK-SAME:  [[ARG0:%.+]]:  tensor<1x6x300x400xf32>, [[ARG1:%.+]]: tensor<1x6x400x400xf32>
func.func @MatMul4dInputs4dWeightsTo2dBigOutputPerGroup(%arg0:  tensor<1x6x300x400xf32>, %arg1: tensor<1x6x400x400xf32>) ->  tensor<1x6x300x400xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x6x300x400xf32>, tensor<1x6x400x400xf32> -> tensor<1x6x300x400xf32>

    return %0 : tensor<1x6x300x400xf32>
    // CHECK:       IE.MatMul
    // CHECK-SAME:  {transpose_b} : tensor<300x400xf32>, tensor<400x400xf32>
    // CHECK:       IE.MatMul
    // CHECK-SAME:  {transpose_b} : tensor<300x400xf32>, tensor<400x400xf32>
    // CHECK:       IE.MatMul
    // CHECK-SAME:  {transpose_b} : tensor<300x400xf32>, tensor<400x400xf32>
    // CHECK:       IE.MatMul
    // CHECK-SAME:  {transpose_b} : tensor<300x400xf32>, tensor<400x400xf32>
    // CHECK:       IE.MatMul
    // CHECK-SAME:  {transpose_b} : tensor<300x400xf32>, tensor<400x400xf32>
    // CHECK:       IE.MatMul
    // CHECK-SAME:  {transpose_b} : tensor<300x400xf32>, tensor<400x400xf32>
    // CHECK:   IE.Concat
}

// Remaining tests are negative tests, enable-grouped-matmul=true does not prevent pass to work.
// -----

// CHECK-LABEL: @MatMul3dInputsBatch1To2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1024xf32>
func.func @MatMul3dInputsBatch1To2d(%arg0: tensor<1x1x1024xf32>) -> tensor<1x1x512xf32> {
    %cst = const.Declare tensor<1x1024x512xf32> = dense<1.0> : tensor<1x1024x512xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x1x1024xf32>, tensor<1x1024x512xf32> -> tensor<1x1x512xf32>

    return %0 : tensor<1x1x512xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1024x512xf32> = dense<1.000000e+00> : tensor<1x1024x512xf32>, [#const.Reshape<[1024, 512]>]
    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [1, 1024]} : tensor<1x1x1024xf32> -> tensor<1x1024xf32>
    // CHECK: [[MTML:%.+]] = IE.MatMul([[RESHAPE0]], [[CST]]) : tensor<1x1024xf32>, tensor<1024x512xf32> -> tensor<1x512xf32>
    // CHECK: [[OUT:%.+]] = IE.AffineReshape([[MTML]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 512]} : tensor<1x512xf32> -> tensor<1x1x512xf32>
    // CHECK: return [[OUT]] : tensor<1x1x512xf32>
}

// -----

// CHECK-LABEL: @NoChangesMatMul3dInput2DWeightsTo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x4x9728xf32>
func.func @NoChangesMatMul3dInput2DWeightsTo2d(%arg0: tensor<1x4x9728xf32>) -> tensor<1x4x512xf32> {
    %cst = const.Declare tensor<9728x512xf32> = dense<1.0> : tensor<9728x512xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x4x9728xf32>, tensor<9728x512xf32> -> tensor<1x4x512xf32>

    return %0 : tensor<1x4x512xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<9728x512xf32> = dense<1.000000e+00> : tensor<9728x512xf32>
    // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [4, 9728]} : tensor<1x4x9728xf32> -> tensor<4x9728xf32>
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[RESHAPE]], [[CST]]) : tensor<4x9728xf32>, tensor<9728x512xf32> -> tensor<4x512xf32>
    // CHECK: [[OUT:%.+]] = IE.AffineReshape([[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [1, 4, 512]} : tensor<4x512xf32> -> tensor<1x4x512xf32>
    // CHECK: return [[OUT]] : tensor<1x4x512xf32>
}

// -----

// CHECK-LABEL: @NoChangesMatMul3dInputWithMulChannel2DWeightsTo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<2x64x128xf32>
func.func @NoChangesMatMul3dInputWithMulChannel2DWeightsTo2d(%arg0: tensor<2x64x128xf32>) -> tensor<2x64x64xf32> {
    %cst = const.Declare tensor<128x64xf32> = dense<1.0> : tensor<128x64xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<2x64x128xf32>, tensor<128x64xf32> -> tensor<2x64x64xf32>

    return %0 : tensor<2x64x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<128x64xf32> = dense<1.000000e+00> : tensor<128x64xf32>
    // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [128, 128]} : tensor<2x64x128xf32> -> tensor<128x128xf32>
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[RESHAPE]], [[CST]]) : tensor<128x128xf32>, tensor<128x64xf32> -> tensor<128x64xf32>
    // CHECK: [[OUT:%.+]] = IE.AffineReshape([[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [2, 64, 64]} : tensor<128x64xf32> -> tensor<2x64x64xf32>
    // CHECK: return [[OUT]] : tensor<2x64x64xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInputWithMulChannel3dWeightsTo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x2x16x2xf32>
func.func @MatMul4dInputWithMulChannel3dWeightsTo2d(%arg0: tensor<1x2x16x2xf32>) -> tensor<1x2x16x2xf32> {
    %cst = const.Declare tensor<1x2x2xf32> = dense<1.0> : tensor<1x2x2xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x16x2xf32>, tensor<1x2x2xf32> -> tensor<1x2x16x2xf32>

    return %0 : tensor<1x2x16x2xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<1x2x2xf32>, [#const.Reshape<[2, 2]>]
    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 2]} : tensor<1x2x16x2xf32> -> tensor<32x2xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_IN]], [[CST]]) : tensor<32x2xf32>, tensor<2x2xf32> -> tensor<32x2xf32>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 16, 2]} : tensor<32x2xf32> -> tensor<1x2x16x2xf32>
    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x2x16x2xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInput2dWeightsNBatchTo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<16x2x16x2xf32>
func.func @MatMul4dInput2dWeightsNBatchTo2d(%arg0: tensor<16x2x16x2xf32>) -> tensor<16x2x16x4xf32> {
    %cst = const.Declare tensor<4x2xf32> = dense<1.0> : tensor<4x2xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<16x2x16x2xf32>, tensor<4x2xf32> -> tensor<16x2x16x4xf32>

    return %0 : tensor<16x2x16x4xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<4x2xf32> = dense<1.000000e+00> : tensor<4x2xf32>
    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 2]} : tensor<16x2x16x2xf32> -> tensor<512x2xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_IN]], [[CST]]) {transpose_b} : tensor<512x2xf32>, tensor<4x2xf32> -> tensor<512x4xf32>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [16, 2, 16, 4]} : tensor<512x4xf32> -> tensor<16x2x16x4xf32>
    // CHECK:       return [[RESHAPE_OUT]] : tensor<16x2x16x4xf32>
}

// -----

// CHECK-LABEL: @MatMul6dInput2dWeights1BatchTo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x8x16x2x16x2xf32>
func.func @MatMul6dInput2dWeights1BatchTo2d(%arg0: tensor<1x8x16x2x16x2xf32>) -> tensor<1x8x16x2x16x4xf32> {
    %cst = const.Declare tensor<4x2xf32> = dense<1.0> : tensor<4x2xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<1x8x16x2x16x2xf32>, tensor<4x2xf32> -> tensor<1x8x16x2x16x4xf32>

    return %0 : tensor<1x8x16x2x16x4xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<4x2xf32> = dense<1.000000e+00> : tensor<4x2xf32>
    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [0], [0], [1]], shape_value = [4096, 2]} : tensor<1x8x16x2x16x2xf32> -> tensor<4096x2xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_IN]], [[CST]]) {transpose_b} : tensor<4096x2xf32>, tensor<4x2xf32> -> tensor<4096x4xf32>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2, 3, 4], [5]], shape_value = [1, 8, 16, 2, 16, 4]} : tensor<4096x4xf32> -> tensor<1x8x16x2x16x4xf32>
    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x8x16x2x16x4xf32>
}

// -----

// CHECK-LABEL: @MatMul5dInput2dWeightsTo2dNoTranspose
// CHECK-SAME:  [[ARG0:%.+]]: tensor<5x6x7x8x16xf32>
func.func @MatMul5dInput2dWeightsTo2dNoTranspose(%arg0: tensor<5x6x7x8x16xf32>) -> tensor<5x6x7x8x32xf32> {
    %cst = const.Declare tensor<16x32xf32> = dense<1.0> : tensor<16x32xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<5x6x7x8x16xf32>, tensor<16x32xf32> -> tensor<5x6x7x8x32xf32>

    return %0 : tensor<5x6x7x8x32xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x32xf32> = dense<1.000000e+00> : tensor<16x32xf32>
    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [0], [1]], shape_value = [1680, 16]} : tensor<5x6x7x8x16xf32> -> tensor<1680x16xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_IN]], [[CST]]) : tensor<1680x16xf32>, tensor<16x32xf32> -> tensor<1680x32xf32>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2, 3], [4]], shape_value = [5, 6, 7, 8, 32]} : tensor<1680x32xf32> -> tensor<5x6x7x8x32xf32>
    // CHECK:       return [[RESHAPE_OUT]] : tensor<5x6x7x8x32xf32>
}

// -----

// CHECK-LABEL: @MatMul5dInput2dWeightsTransposeATo3d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<5x6x7x16x8xf32>
func.func @MatMul5dInput2dWeightsTransposeATo3d(%arg0: tensor<5x6x7x16x8xf32>) -> tensor<5x6x7x8x32xf32> {
    %cst = const.Declare tensor<16x32xf32> = dense<1.0> : tensor<16x32xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_a} : tensor<5x6x7x16x8xf32>, tensor<16x32xf32> -> tensor<5x6x7x8x32xf32>

    return %0 : tensor<5x6x7x8x32xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x32xf32> = dense<1.000000e+00> : tensor<16x32xf32>
    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1], [2]], shape_value = [210, 16, 8]} : tensor<5x6x7x16x8xf32> -> tensor<210x16x8xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_IN]], [[CST]]) {transpose_a} : tensor<210x16x8xf32>, tensor<16x32xf32> -> tensor<210x8x32xf32>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3], [4]], shape_value = [5, 6, 7, 8, 32]} : tensor<210x8x32xf32> -> tensor<5x6x7x8x32xf32>
    // CHECK:       return [[RESHAPE_OUT]] : tensor<5x6x7x8x32xf32>
}

// -----

// CHECK-LABEL: @MatMulVectorMatrixTo2D
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1024xf32>
func.func @MatMulVectorMatrixTo2D(%arg0: tensor<1024xf32>) -> tensor<1000xf32> {
    %cst = const.Declare tensor<1024x1000xf32> = dense<1.0> : tensor<1024x1000xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1024xf32>, tensor<1024x1000xf32> -> tensor<1000xf32>

    return %0 : tensor<1000xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1024x1000xf32> = dense<1.000000e+00> : tensor<1024x1000xf32>
    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1]], shape_value = [1, 1024]} : tensor<1024xf32> -> tensor<1x1024xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_IN]], [[CST]]) : tensor<1x1024xf32>, tensor<1024x1000xf32> -> tensor<1x1000xf32>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0]], shape_value = [1000]} : tensor<1x1000xf32> -> tensor<1000xf32>
    // CHECK:       return [[RESHAPE_OUT]] : tensor<1000xf32>
}

// -----

// CHECK-LABEL: @MatMulMatrixVectorTo2D
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1000x1024xf32>
func.func @MatMulMatrixVectorTo2D(%arg0: tensor<1000x1024xf32>) -> tensor<1000xf32> {
    %cst = const.Declare tensor<1024xf32> = dense<1.0> : tensor<1024xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1000x1024xf32>, tensor<1024xf32> -> tensor<1000xf32>

    return %0 : tensor<1000xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1024x1xf32> = dense<1.000000e+00> : tensor<1024xf32>, [#const.Reshape<[1024, 1]>]
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[CST]]) : tensor<1000x1024xf32>, tensor<1024x1xf32> -> tensor<1000x1xf32>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0]], shape_value = [1000]} : tensor<1000x1xf32> -> tensor<1000xf32>
    // CHECK:       return [[RESHAPE_OUT]] : tensor<1000xf32>
}

// -----

// CHECK-LABEL: @MatMul5dInputs3dWeightsTransposeBTo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x1x2400xf16>, [[ARG1:%.+]]: tensor<1x256x2400xf16>
func.func @MatMul5dInputs3dWeightsTransposeBTo2d(%arg0: tensor<1x1x1x1x2400xf16>, %arg1: tensor<1x256x2400xf16>) -> tensor<1x1x1x1x256xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x1x1x1x2400xf16>, tensor<1x256x2400xf16> -> tensor<1x1x1x1x256xf16>

    return %0 : tensor<1x1x1x1x256xf16>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [0], [1]], shape_value = [1, 2400]} : tensor<1x1x1x1x2400xf16> -> tensor<1x2400xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[ARG1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [256, 2400]} : tensor<1x256x2400xf16> -> tensor<256x2400xf16>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_0]], [[RESHAPE_1]]) {transpose_b} : tensor<1x2400xf16>, tensor<256x2400xf16> -> tensor<1x256xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2, 3], [4]], shape_value = [1, 1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x1x256xf16>
    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x1x1x1x256xf16>
}

// -----

// CHECK-LABEL: @MatMul5dInputs3dWeightsTransposeATo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x2400x1xf16>, [[ARG1:%.+]]: tensor<1x2400x256xf16>
func.func @MatMul5dInputs3dWeightsTransposeATo2d(%arg0: tensor<1x1x1x2400x1xf16>, %arg1: tensor<1x2400x256xf16>) -> tensor<1x1x1x1x256xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_a} : tensor<1x1x1x2400x1xf16>, tensor<1x2400x256xf16> -> tensor<1x1x1x1x256xf16>

    return %0 : tensor<1x1x1x1x256xf16>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [0], [1]], shape_value = [2400, 1]} : tensor<1x1x1x2400x1xf16> -> tensor<2400x1xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[ARG1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [2400, 256]} : tensor<1x2400x256xf16> -> tensor<2400x256xf16>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_0]], [[RESHAPE_1]]) {transpose_a} : tensor<2400x1xf16>, tensor<2400x256xf16> -> tensor<1x256xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MATMUL]]
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2, 3], [4]], shape_value = [1, 1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x1x256xf16>
    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x1x1x1x256xf16>
}

// CHECK-LABEL: @FuseReshapesAfterMatmulConcat
func.func @FuseReshapesAfterMatmulConcat(%arg0: tensor<1x8x4096x40xf32>, %arg1: tensor<1x8x4096x40xf32>) -> tensor<8x4096x4096xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x4096x40xf32>, tensor<1x8x4096x40xf32> -> tensor<1x8x4096x4096xf32>
    %1 = IE.Reshape(%0) {shape_value = [8, 4096, 4096]} : tensor<1x8x4096x4096xf32> -> tensor<8x4096x4096xf32>
    %2 = IE.SoftMax(%1) {axisInd = 2 : i64} : tensor<8x4096x4096xf32> -> tensor<8x4096x4096xf32>
    return %2 : tensor<8x4096x4096xf32>

    // CHECK: [[CONCAT:%.+]] = IE.Concat
    // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [8, 4096, 4096]} : tensor<32768x4096xf32> -> tensor<8x4096x4096xf32>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[RESHAPE]])
    // CHECK: return [[SOFTMAX]] : tensor<8x4096x4096xf32>
}

// -----

// CHECK-LABEL: @MatMulNot2GroupMatMulInputChannelOutOfNCELimit
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x8x32x9216xf16>, [[ARG1:%.+]]: tensor<1x8x32x9216xf16>
func.func @MatMulNot2GroupMatMulInputChannelOutOfNCELimit(%arg0: tensor<1x8x32x9216xf16>, %arg1: tensor<1x8x32x9216xf16>) -> tensor<1x8x32x32xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x32x9216xf16>, tensor<1x8x32x9216xf16> -> tensor<1x8x32x32xf16>
    return %0 : tensor<1x8x32x32xf16>

    // CHECK: [[SLICE0:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[SLICE0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE1:%.+]] = IE.Slice [[ARG0]] [0, 1, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[SLICE1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE2:%.+]] = IE.Slice [[ARG0]] [0, 2, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE2:%.+]] = IE.AffineReshape([[SLICE2]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE3:%.+]] = IE.Slice [[ARG0]] [0, 3, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE3:%.+]] = IE.AffineReshape([[SLICE3]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE4:%.+]] = IE.Slice [[ARG0]] [0, 4, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE4:%.+]] = IE.AffineReshape([[SLICE4]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE5:%.+]] = IE.Slice [[ARG0]] [0, 5, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE5:%.+]] = IE.AffineReshape([[SLICE5]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE6:%.+]] = IE.Slice [[ARG0]] [0, 6, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE6:%.+]] = IE.AffineReshape([[SLICE6]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE7:%.+]] = IE.Slice [[ARG0]] [0, 7, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE7:%.+]] = IE.AffineReshape([[SLICE7]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE8:%.+]] = IE.Slice [[ARG1]] [0, 0, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE8:%.+]] = IE.AffineReshape([[SLICE8]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE9:%.+]] = IE.Slice [[ARG1]] [0, 1, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE9:%.+]] = IE.AffineReshape([[SLICE9]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE10:%.+]] = IE.Slice [[ARG1]] [0, 2, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE10:%.+]] = IE.AffineReshape([[SLICE10]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE11:%.+]] = IE.Slice [[ARG1]] [0, 3, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE11:%.+]] = IE.AffineReshape([[SLICE11]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE12:%.+]] = IE.Slice [[ARG1]] [0, 4, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE12:%.+]] = IE.AffineReshape([[SLICE12]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE13:%.+]] = IE.Slice [[ARG1]] [0, 5, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE13:%.+]] = IE.AffineReshape([[SLICE13]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE14:%.+]] = IE.Slice [[ARG1]] [0, 6, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE14:%.+]] = IE.AffineReshape([[SLICE14]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[SLICE15:%.+]] = IE.Slice [[ARG1]] [0, 7, 0, 0] [1, 1, 32, 9216] : tensor<1x8x32x9216xf16> to tensor<1x1x32x9216xf16>
    // CHECK: [[RESHAPE15:%.+]] = IE.AffineReshape([[SLICE15]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 9216]} : tensor<1x1x32x9216xf16> -> tensor<32x9216xf16>
    // CHECK: [[MATMUL0:%.+]] = IE.MatMul([[RESHAPE0]], [[RESHAPE8]]) {transpose_b} : tensor<32x9216xf16>, tensor<32x9216xf16> -> tensor<32x32xf16>
    // CHECK: [[MATMUL1:%.+]] = IE.MatMul([[RESHAPE1]], [[RESHAPE9]]) {transpose_b} : tensor<32x9216xf16>, tensor<32x9216xf16> -> tensor<32x32xf16>
    // CHECK: [[MATMUL2:%.+]] = IE.MatMul([[RESHAPE2]], [[RESHAPE10]]) {transpose_b} : tensor<32x9216xf16>, tensor<32x9216xf16> -> tensor<32x32xf16>
    // CHECK: [[MATMUL3:%.+]] = IE.MatMul([[RESHAPE3]], [[RESHAPE11]]) {transpose_b} : tensor<32x9216xf16>, tensor<32x9216xf16> -> tensor<32x32xf16>
    // CHECK: [[MATMUL4:%.+]] = IE.MatMul([[RESHAPE4]], [[RESHAPE12]]) {transpose_b} : tensor<32x9216xf16>, tensor<32x9216xf16> -> tensor<32x32xf16>
    // CHECK: [[MATMUL5:%.+]] = IE.MatMul([[RESHAPE5]], [[RESHAPE13]]) {transpose_b} : tensor<32x9216xf16>, tensor<32x9216xf16> -> tensor<32x32xf16>
    // CHECK: [[MATMUL6:%.+]] = IE.MatMul([[RESHAPE6]], [[RESHAPE14]]) {transpose_b} : tensor<32x9216xf16>, tensor<32x9216xf16> -> tensor<32x32xf16>
    // CHECK: [[MATMUL7:%.+]] = IE.MatMul([[RESHAPE7]], [[RESHAPE15]]) {transpose_b} : tensor<32x9216xf16>, tensor<32x9216xf16> -> tensor<32x32xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[MATMUL0]], [[MATMUL1]], [[MATMUL2]], [[MATMUL3]], [[MATMUL4]], [[MATMUL5]], [[MATMUL6]], [[MATMUL7]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf16> -> tensor<256x32xf16>
    // CHECK: [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 8, 32, 32]} : tensor<256x32xf16> -> tensor<1x8x32x32xf16>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x8x32x32xf16>
}

// -----

// CHECK-LABEL: @ConvertMatMulWithBatchAndTransposeBToGroupConv
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x16x4x2xf32>, [[INPUT1:%.+]]: tensor<1x16x1x2xf32>
func.func @ConvertMatMulWithBatchAndTransposeBToGroupConv(%arg0: tensor<1x16x4x2xf32>, %arg1: tensor<1x16x1x2xf32>) -> tensor<1x16x4x1xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x16x4x2xf32>, tensor<1x16x1x2xf32> -> tensor<1x16x4x1xf32>

    return %0 : tensor<1x16x4x1xf32>

    // CHECK: [[AFFINERESHAPE_0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK: [[AFFINERESHAPE_1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK: [[AFFINERESHAPE_ACT:%.+]] = IE.AffineReshape([[AFFINERESHAPE_0]])
    // CHECK: [[AFFINERESHAPE_WT:%.+]] = IE.AffineReshape([[AFFINERESHAPE_1]])
    // CHECK: [[GROUPCONV:%.+]] = IE.GroupConvolution([[AFFINERESHAPE_ACT]], [[AFFINERESHAPE_WT]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x4x2xf32>, tensor<16x1x1x2xf32> -> tensor<1x16x4x1xf32>

    // CHECK: return [[GROUPCONV]] : tensor<1x16x4x1xf32>
}

// -----

// CHECK-LABEL: @ConvertMatMulWithBatchAndTransposeAToGroupConv
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x16x2x4xf32>, [[INPUT1:%.+]]: tensor<1x16x2x1xf32>
func.func @ConvertMatMulWithBatchAndTransposeAToGroupConv(%arg0: tensor<1x16x2x4xf32>, %arg1: tensor<1x16x2x1xf32>) -> tensor<1x16x4x1xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_a} : tensor<1x16x2x4xf32>, tensor<1x16x2x1xf32> -> tensor<1x16x4x1xf32>

    return %0 : tensor<1x16x4x1xf32>

    // CHECK: [[RESHAPE_0:%.+]] = IE.Reshape([[INPUT0]])
    // CHECK: [[RESHAPE_1:%.+]] = IE.Reshape([[INPUT1]])
    // CHECK: [[AFFINERESHAPE_ACT:%.+]] = IE.AffineReshape([[RESHAPE_0]])
    // CHECK: [[AFFINERESHAPE_WT:%.+]] = IE.AffineReshape([[RESHAPE_1]])
    // CHECK: [[GROUPCONV:%.+]] = IE.GroupConvolution([[AFFINERESHAPE_ACT]], [[AFFINERESHAPE_WT]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x4x2xf32>, tensor<16x1x1x2xf32> -> tensor<1x16x4x1xf32>

    // CHECK: return [[GROUPCONV]] : tensor<1x16x4x1xf32>
}

// -----

// CHECK-LABEL: @ConvertMatMulWithHugeBatchToGroupConv
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x17408x2x2xf32>, [[INPUT1:%.+]]: tensor<1x17408x1x2xf32>
func.func @ConvertMatMulWithHugeBatchToGroupConv(%arg0: tensor<1x17408x2x2xf32>, %arg1: tensor<1x17408x1x2xf32>) -> tensor<1x17408x2x1xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x17408x2x2xf32>, tensor<1x17408x1x2xf32> -> tensor<1x17408x2x1xf32>

    return %0 : tensor<1x17408x2x1xf32>

    // CHECK: [[AFFINERESHAPE_0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK: [[AFFINERESHAPE_1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK: [[SLICE_0_ACT:%.+]] = IE.Slice [[AFFINERESHAPE_0]] [0, 0, 0] [8192, 2, 2] : tensor<17408x2x2xf32> to tensor<8192x2x2xf32>
    // CHECK: [[AFFINERESHAPE_SLICE_0_ACT:%.+]] = IE.AffineReshape([[SLICE_0_ACT]])
    // CHECK: [[SLICE_0_WT:%.+]] = IE.Slice [[AFFINERESHAPE_1]] [0, 0, 0] [8192, 1, 2] : tensor<17408x1x2xf32> to tensor<8192x1x2xf32>
    // CHECK: [[AFFINERESHAPE_SLICE_0_WT:%.+]] = IE.AffineReshape([[SLICE_0_WT]])
    // CHECK: [[SLICE_0_GROUPCONV:%.+]] = IE.GroupConvolution([[AFFINERESHAPE_SLICE_0_ACT]], [[AFFINERESHAPE_SLICE_0_WT]]) {dilations = [1, 1], groups = 8192 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x8192x2x2xf32>, tensor<8192x1x1x2xf32> -> tensor<1x8192x2x1xf32>

    // CHECK: [[SLICE_1_ACT:%.+]] = IE.Slice [[AFFINERESHAPE_0]] [8192, 0, 0] [8192, 2, 2] : tensor<17408x2x2xf32> to tensor<8192x2x2xf32>
    // CHECK: [[AFFINERESHAPE_SLICE_1_ACT:%.+]] = IE.AffineReshape([[SLICE_1_ACT]])
    // CHECK: [[SLICE_1_WT:%.+]] = IE.Slice [[AFFINERESHAPE_1]] [8192, 0, 0] [8192, 1, 2] : tensor<17408x1x2xf32> to tensor<8192x1x2xf32>
    // CHECK: [[AFFINERESHAPE_SLICE_1_WT:%.+]] = IE.AffineReshape([[SLICE_1_WT]])
    // CHECK: [[SLICE_1_GROUPCONV:%.+]] = IE.GroupConvolution([[AFFINERESHAPE_SLICE_1_ACT]], [[AFFINERESHAPE_SLICE_1_WT]]) {dilations = [1, 1], groups = 8192 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x8192x2x2xf32>, tensor<8192x1x1x2xf32> -> tensor<1x8192x2x1xf32>

    // CHECK: [[SLICE_2_ACT:%.+]] = IE.Slice [[AFFINERESHAPE_0]] [16384, 0, 0] [1024, 2, 2] : tensor<17408x2x2xf32> to tensor<1024x2x2xf32>
    // CHECK: [[AFFINERESHAPE_SLICE_2_ACT:%.+]] = IE.AffineReshape([[SLICE_2_ACT]])
    // CHECK: [[SLICE_2_WT:%.+]] = IE.Slice [[AFFINERESHAPE_1]] [16384, 0, 0] [1024, 1, 2] : tensor<17408x1x2xf32> to tensor<1024x1x2xf32>
    // CHECK: [[AFFINERESHAPE_SLICE_2_WT:%.+]] = IE.AffineReshape([[SLICE_2_WT]])
    // CHECK: [[SLICE_2_GROUPCONV:%.+]] = IE.GroupConvolution([[AFFINERESHAPE_SLICE_2_ACT]], [[AFFINERESHAPE_SLICE_2_WT]]) {dilations = [1, 1], groups = 1024 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x2x2xf32>, tensor<1024x1x1x2xf32> -> tensor<1x1024x2x1xf32>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[SLICE_0_GROUPCONV]], [[SLICE_1_GROUPCONV]], [[SLICE_2_GROUPCONV]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x8192x2x1xf32>, tensor<1x8192x2x1xf32>, tensor<1x1024x2x1xf32> -> tensor<1x17408x2x1xf32>

    // CHECK: return [[CONCAT]] : tensor<1x17408x2x1xf32>
}

// -----

// CHECK-LABEL: @NotConvertMatMulWithBatchToGroupConvInCaseOutputChannelNonOne
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x16x4x2xf32>, [[INPUT1:%.+]]: tensor<1x16x3x2xf32>
func.func @NotConvertMatMulWithBatchToGroupConvInCaseOutputChannelNonOne(%arg0: tensor<1x16x4x2xf32>, %arg1: tensor<1x16x3x2xf32>) -> tensor<1x16x4x3xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x16x4x2xf32>, tensor<1x16x3x2xf32> -> tensor<1x16x4x3xf32>

    return %0 : tensor<1x16x4x3xf32>

    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[INPUT0]], [[INPUT1]]) {transpose_b} : tensor<1x16x4x2xf32>, tensor<1x16x3x2xf32> -> tensor<1x16x4x3xf32>

    // CHECK: return [[MATMUL]] : tensor<1x16x4x3xf32>
}

// -----

// CHECK-LABEL: @BroadcastMatmulIsUnrolled
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x3x3xf32>, [[ARG1:%.+]]: tensor<1x4x3x3xf32>
func.func @BroadcastMatmulIsUnrolled(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x4x3x3xf32>) -> tensor<1x4x3x3xf32> {
    %0 = IE.MatMul(%arg0, %arg1) : tensor<1x1x3x3xf32>, tensor<1x4x3x3xf32> -> tensor<1x4x3x3xf32>
    return %0 : tensor<1x4x3x3xf32>

    // input1 (batch=1) is reshaped once and reused for all 4 matmuls.
    // CHECK:       [[RESHAPE_IN0:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [3, 3]} : tensor<1x1x3x3xf32> -> tensor<3x3xf32>
    // input2 (batch=4) is sliced into 4 individual matrices.
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[ARG1]] [0, 0, 0, 0] [1, 1, 3, 3] : tensor<1x4x3x3xf32> to tensor<1x1x3x3xf32>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[SLICE0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [3, 3]} : tensor<1x1x3x3xf32> -> tensor<3x3xf32>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ARG1]] [0, 1, 0, 0] [1, 1, 3, 3] : tensor<1x4x3x3xf32> to tensor<1x1x3x3xf32>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[SLICE1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [3, 3]} : tensor<1x1x3x3xf32> -> tensor<3x3xf32>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[ARG1]] [0, 2, 0, 0] [1, 1, 3, 3] : tensor<1x4x3x3xf32> to tensor<1x1x3x3xf32>
    // CHECK:       [[RESHAPE2:%.+]] = IE.AffineReshape([[SLICE2]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [3, 3]} : tensor<1x1x3x3xf32> -> tensor<3x3xf32>
    // CHECK:       [[SLICE3:%.+]] = IE.Slice [[ARG1]] [0, 3, 0, 0] [1, 1, 3, 3] : tensor<1x4x3x3xf32> to tensor<1x1x3x3xf32>
    // CHECK:       [[RESHAPE3:%.+]] = IE.AffineReshape([[SLICE3]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [3, 3]} : tensor<1x1x3x3xf32> -> tensor<3x3xf32>
    // CHECK:       [[MATMUL0:%.+]] = IE.MatMul([[RESHAPE_IN0]], [[RESHAPE0]]) : tensor<3x3xf32>, tensor<3x3xf32> -> tensor<3x3xf32>
    // CHECK:       [[MATMUL1:%.+]] = IE.MatMul([[RESHAPE_IN0]], [[RESHAPE1]]) : tensor<3x3xf32>, tensor<3x3xf32> -> tensor<3x3xf32>
    // CHECK:       [[MATMUL2:%.+]] = IE.MatMul([[RESHAPE_IN0]], [[RESHAPE2]]) : tensor<3x3xf32>, tensor<3x3xf32> -> tensor<3x3xf32>
    // CHECK:       [[MATMUL3:%.+]] = IE.MatMul([[RESHAPE_IN0]], [[RESHAPE3]]) : tensor<3x3xf32>, tensor<3x3xf32> -> tensor<3x3xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[MATMUL0]], [[MATMUL1]], [[MATMUL2]], [[MATMUL3]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32> -> tensor<12x3xf32>
    // CHECK:       [[OUT:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 4, 3, 3]} : tensor<12x3xf32> -> tensor<1x4x3x3xf32>
    // CHECK:       return [[OUT]] : tensor<1x4x3x3xf32>
}

// -----

// CHECK-LABEL: @NotConvertMatMulWithBatchToGroupConvInCaseLargeInputChannel
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x16x4x16xf32>, [[INPUT1:%.+]]: tensor<1x16x1x16xf32>
func.func @NotConvertMatMulWithBatchToGroupConvInCaseLargeInputChannel(%arg0: tensor<1x16x4x16xf32>, %arg1: tensor<1x16x1x16xf32>) -> tensor<1x16x4x1xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x16x4x16xf32>, tensor<1x16x1x16xf32> -> tensor<1x16x4x1xf32>

    return %0 : tensor<1x16x4x1xf32>

    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[INPUT0]], [[INPUT1]]) {transpose_b} : tensor<1x16x4x16xf32>, tensor<1x16x1x16xf32> -> tensor<1x16x4x1xf32>

    // CHECK: return [[MATMUL]] : tensor<1x16x4x1xf32>
}

// -----

// Test: Broadcastable multi-batch MatMul that would collapse into incompatible flattened batches.
// [2,3,4,16] x [2,1,16,8]: after adjustTo3DShape → [6,4,16] x [2,16,8], batches 6 vs 2 are
// incompatible (neither equal nor 1). The rewriter must leave the MatMul unchanged.

// CHECK-LABEL: @MatMulBroadcastBatchIncompatibleAfterFlatten
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<2x3x4x16xf16>, [[INPUT1:%.+]]: tensor<2x1x16x8xf16>
func.func @MatMulBroadcastBatchIncompatibleAfterFlatten(%arg0: tensor<2x3x4x16xf16>, %arg1: tensor<2x1x16x8xf16>) -> tensor<2x3x4x8xf16> {
    %0 = IE.MatMul(%arg0, %arg1) : tensor<2x3x4x16xf16>, tensor<2x1x16x8xf16> -> tensor<2x3x4x8xf16>

    return %0 : tensor<2x3x4x8xf16>

    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[INPUT0]], [[INPUT1]]) : tensor<2x3x4x16xf16>, tensor<2x1x16x8xf16> -> tensor<2x3x4x8xf16>

    // CHECK: return [[MATMUL]] : tensor<2x3x4x8xf16>
}

// -----

// Test: Per-dim batch mismatch where flattened products coincidentally match.
// [2,1,4,16] x [1,2,16,8]: both flatten to batch=2, but per-dim batches [2,1] vs [1,2] differ.
// Collapsing would incorrectly pair elements. The rewriter must leave the MatMul unchanged.

// CHECK-LABEL: @MatMulPerDimBatchMismatchSameProduct
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<2x1x4x16xf16>, [[INPUT1:%.+]]: tensor<1x2x16x8xf16>
func.func @MatMulPerDimBatchMismatchSameProduct(%arg0: tensor<2x1x4x16xf16>, %arg1: tensor<1x2x16x8xf16>) -> tensor<2x2x4x8xf16> {
    %0 = IE.MatMul(%arg0, %arg1) : tensor<2x1x4x16xf16>, tensor<1x2x16x8xf16> -> tensor<2x2x4x8xf16>

    return %0 : tensor<2x2x4x8xf16>

    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[INPUT0]], [[INPUT1]]) : tensor<2x1x4x16xf16>, tensor<1x2x16x8xf16> -> tensor<2x2x4x8xf16>

    // CHECK: return [[MATMUL]] : tensor<2x2x4x8xf16>
}
