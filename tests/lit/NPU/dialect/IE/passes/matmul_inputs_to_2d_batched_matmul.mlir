//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --matmul-inputs-to-2d="enable-grouped-matmul=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

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

    // CHECK: [[SLICE0:%.*]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 1, 4000, 4000] : tensor<1x2x4000x4000xf32> to tensor<1x1x4000x4000xf32>
    // CHECK: [[RESHAPE0:%.*]] = IE.AffineReshape([[SLICE0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [4000, 4000]} : tensor<1x1x4000x4000xf32> -> tensor<4000x4000xf32>
    // CHECK: [[SLICE1:%.*]] = IE.Slice [[ARG0]] [0, 1, 0, 0] [1, 1, 4000, 4000] : tensor<1x2x4000x4000xf32> to tensor<1x1x4000x4000xf32>
    // CHECK: [[RESHAPE1:%.*]] = IE.AffineReshape([[SLICE1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [4000, 4000]} : tensor<1x1x4000x4000xf32> -> tensor<4000x4000xf32>
    // CHECK: [[SLICE2:%.*]] = IE.Slice [[ARG1]] [0, 0, 0, 0] [1, 1, 3000, 4000] : tensor<1x2x3000x4000xf32> to tensor<1x1x3000x4000xf32>
    // CHECK: [[RESHAPE2:%.*]] = IE.AffineReshape([[SLICE2]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [3000, 4000]} : tensor<1x1x3000x4000xf32> -> tensor<3000x4000xf32>
    // CHECK: [[SLICE3:%.*]] = IE.Slice [[ARG1]] [0, 1, 0, 0] [1, 1, 3000, 4000] : tensor<1x2x3000x4000xf32> to tensor<1x1x3000x4000xf32>
    // CHECK: [[RESHAPE3:%.*]] = IE.AffineReshape([[SLICE3]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [3000, 4000]} : tensor<1x1x3000x4000xf32> -> tensor<3000x4000xf32>
    // CHECK: [[MATMUL0:%.*]] = IE.MatMul([[RESHAPE0]], [[RESHAPE2]]) {transpose_b} : tensor<4000x4000xf32>, tensor<3000x4000xf32> -> tensor<4000x3000xf32>
    // CHECK: [[MATMUL1:%.*]] = IE.MatMul([[RESHAPE1]], [[RESHAPE3]]) {transpose_b} : tensor<4000x4000xf32>, tensor<3000x4000xf32> -> tensor<4000x3000xf32>
    // CHECK: [[CONCAT:%.*]] = IE.Concat([[MATMUL0]], [[MATMUL1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<4000x3000xf32>, tensor<4000x3000xf32> -> tensor<8000x3000xf32>
    // CHECK: [[RESHAPE4:%.*]] = IE.AffineReshape([[CONCAT]])
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

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<1024x512xf32> = dense<1.000000e+00> : tensor<1x1024x512xf32>, [#const.Reshape<[1024, 512]>]
    // CHECK: %[[RESHAPE0:.*]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [1, 1024]} : tensor<1x1x1024xf32> -> tensor<1x1024xf32>
    // CHECK: %[[MTML:.*]] = IE.MatMul(%[[RESHAPE0]], %[[CST]]) : tensor<1x1024xf32>, tensor<1024x512xf32> -> tensor<1x512xf32>
    // CHECK: %[[OUT:.*]] = IE.AffineReshape(%[[MTML]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 512]} : tensor<1x512xf32> -> tensor<1x1x512xf32>
    // CHECK: return %[[OUT]] : tensor<1x1x512xf32>
}

// -----

// CHECK-LABEL: @NoChangesMatMul3dInput2DWeightsTo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x4x9728xf32>
func.func @NoChangesMatMul3dInput2DWeightsTo2d(%arg0: tensor<1x4x9728xf32>) -> tensor<1x4x512xf32> {
    %cst = const.Declare tensor<9728x512xf32> = dense<1.0> : tensor<9728x512xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x4x9728xf32>, tensor<9728x512xf32> -> tensor<1x4x512xf32>

    return %0 : tensor<1x4x512xf32>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<9728x512xf32> = dense<1.000000e+00> : tensor<9728x512xf32>
    // CHECK: %[[RESHAPE:.*]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [4, 9728]} : tensor<1x4x9728xf32> -> tensor<4x9728xf32>
    // CHECK: %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE]], %[[CST]]) : tensor<4x9728xf32>, tensor<9728x512xf32> -> tensor<4x512xf32>
    // CHECK: %[[OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [1, 4, 512]} : tensor<4x512xf32> -> tensor<1x4x512xf32>
    // CHECK: return %[[OUT]] : tensor<1x4x512xf32>
}

// -----

// CHECK-LABEL: @NoChangesMatMul3dInputWithMulChannel2DWeightsTo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<2x64x128xf32>
func.func @NoChangesMatMul3dInputWithMulChannel2DWeightsTo2d(%arg0: tensor<2x64x128xf32>) -> tensor<2x64x64xf32> {
    %cst = const.Declare tensor<128x64xf32> = dense<1.0> : tensor<128x64xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<2x64x128xf32>, tensor<128x64xf32> -> tensor<2x64x64xf32>

    return %0 : tensor<2x64x64xf32>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<128x64xf32> = dense<1.000000e+00> : tensor<128x64xf32>
    // CHECK: %[[RESHAPE:.*]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [128, 128]} : tensor<2x64x128xf32> -> tensor<128x128xf32>
    // CHECK: %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE]], %[[CST]]) : tensor<128x128xf32>, tensor<128x64xf32> -> tensor<128x64xf32>
    // CHECK: %[[OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [2, 64, 64]} : tensor<128x64xf32> -> tensor<2x64x64xf32>
    // CHECK: return %[[OUT]] : tensor<2x64x64xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInputWithMulChannel3dWeightsTo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x2x16x2xf32>
func.func @MatMul4dInputWithMulChannel3dWeightsTo2d(%arg0: tensor<1x2x16x2xf32>) -> tensor<1x2x16x2xf32> {
    %cst = const.Declare tensor<1x2x2xf32> = dense<1.0> : tensor<1x2x2xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x16x2xf32>, tensor<1x2x2xf32> -> tensor<1x2x16x2xf32>

    return %0 : tensor<1x2x16x2xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<1x2x2xf32>, [#const.Reshape<[2, 2]>]
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 2]} : tensor<1x2x16x2xf32> -> tensor<32x2xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) : tensor<32x2xf32>, tensor<2x2xf32> -> tensor<32x2xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 16, 2]} : tensor<32x2xf32> -> tensor<1x2x16x2xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1x2x16x2xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInput2dWeightsNBatchTo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<16x2x16x2xf32>
func.func @MatMul4dInput2dWeightsNBatchTo2d(%arg0: tensor<16x2x16x2xf32>) -> tensor<16x2x16x4xf32> {
    %cst = const.Declare tensor<4x2xf32> = dense<1.0> : tensor<4x2xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<16x2x16x2xf32>, tensor<4x2xf32> -> tensor<16x2x16x4xf32>

    return %0 : tensor<16x2x16x4xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<4x2xf32> = dense<1.000000e+00> : tensor<4x2xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 2]} : tensor<16x2x16x2xf32> -> tensor<512x2xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) {transpose_b} : tensor<512x2xf32>, tensor<4x2xf32> -> tensor<512x4xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [16, 2, 16, 4]} : tensor<512x4xf32> -> tensor<16x2x16x4xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<16x2x16x4xf32>
}

// -----

// CHECK-LABEL: @MatMul6dInput2dWeights1BatchTo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x8x16x2x16x2xf32>
func.func @MatMul6dInput2dWeights1BatchTo2d(%arg0: tensor<1x8x16x2x16x2xf32>) -> tensor<1x8x16x2x16x4xf32> {
    %cst = const.Declare tensor<4x2xf32> = dense<1.0> : tensor<4x2xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<1x8x16x2x16x2xf32>, tensor<4x2xf32> -> tensor<1x8x16x2x16x4xf32>

    return %0 : tensor<1x8x16x2x16x4xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<4x2xf32> = dense<1.000000e+00> : tensor<4x2xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [0], [0], [1]], shape_value = [4096, 2]} : tensor<1x8x16x2x16x2xf32> -> tensor<4096x2xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) {transpose_b} : tensor<4096x2xf32>, tensor<4x2xf32> -> tensor<4096x4xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2, 3, 4], [5]], shape_value = [1, 8, 16, 2, 16, 4]} : tensor<4096x4xf32> -> tensor<1x8x16x2x16x4xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1x8x16x2x16x4xf32>
}

// -----

// CHECK-LABEL: @MatMul5dInput2dWeightsTo2dNoTranspose
// CHECK-SAME:  [[ARG0:%.+]]: tensor<5x6x7x8x16xf32>
func.func @MatMul5dInput2dWeightsTo2dNoTranspose(%arg0: tensor<5x6x7x8x16xf32>) -> tensor<5x6x7x8x32xf32> {
    %cst = const.Declare tensor<16x32xf32> = dense<1.0> : tensor<16x32xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<5x6x7x8x16xf32>, tensor<16x32xf32> -> tensor<5x6x7x8x32xf32>

    return %0 : tensor<5x6x7x8x32xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<16x32xf32> = dense<1.000000e+00> : tensor<16x32xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [0], [1]], shape_value = [1680, 16]} : tensor<5x6x7x8x16xf32> -> tensor<1680x16xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) : tensor<1680x16xf32>, tensor<16x32xf32> -> tensor<1680x32xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2, 3], [4]], shape_value = [5, 6, 7, 8, 32]} : tensor<1680x32xf32> -> tensor<5x6x7x8x32xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<5x6x7x8x32xf32>
}

// -----

// CHECK-LABEL: @MatMul5dInput2dWeightsTransposeATo3d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<5x6x7x16x8xf32>
func.func @MatMul5dInput2dWeightsTransposeATo3d(%arg0: tensor<5x6x7x16x8xf32>) -> tensor<5x6x7x8x32xf32> {
    %cst = const.Declare tensor<16x32xf32> = dense<1.0> : tensor<16x32xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_a} : tensor<5x6x7x16x8xf32>, tensor<16x32xf32> -> tensor<5x6x7x8x32xf32>

    return %0 : tensor<5x6x7x8x32xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<16x32xf32> = dense<1.000000e+00> : tensor<16x32xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1], [2]], shape_value = [210, 16, 8]} : tensor<5x6x7x16x8xf32> -> tensor<210x16x8xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) {transpose_a} : tensor<210x16x8xf32>, tensor<16x32xf32> -> tensor<210x8x32xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3], [4]], shape_value = [5, 6, 7, 8, 32]} : tensor<210x8x32xf32> -> tensor<5x6x7x8x32xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<5x6x7x8x32xf32>
}

// -----

// CHECK-LABEL: @MatMulVectorMatrixTo2D
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1024xf32>
func.func @MatMulVectorMatrixTo2D(%arg0: tensor<1024xf32>) -> tensor<1000xf32> {
    %cst = const.Declare tensor<1024x1000xf32> = dense<1.0> : tensor<1024x1000xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1024xf32>, tensor<1024x1000xf32> -> tensor<1000xf32>

    return %0 : tensor<1000xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<1024x1000xf32> = dense<1.000000e+00> : tensor<1024x1000xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1]], shape_value = [1, 1024]} : tensor<1024xf32> -> tensor<1x1024xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) : tensor<1x1024xf32>, tensor<1024x1000xf32> -> tensor<1x1000xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0]], shape_value = [1000]} : tensor<1x1000xf32> -> tensor<1000xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1000xf32>
}

// -----

// CHECK-LABEL: @MatMulMatrixVectorTo2D
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1000x1024xf32>
func.func @MatMulMatrixVectorTo2D(%arg0: tensor<1000x1024xf32>) -> tensor<1000xf32> {
    %cst = const.Declare tensor<1024xf32> = dense<1.0> : tensor<1024xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1000x1024xf32>, tensor<1024xf32> -> tensor<1000xf32>

    return %0 : tensor<1000xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<1024x1xf32> = dense<1.000000e+00> : tensor<1024xf32>, [#const.Reshape<[1024, 1]>]
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul([[ARG0]], %[[CST]]) : tensor<1000x1024xf32>, tensor<1024x1xf32> -> tensor<1000x1xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0]], shape_value = [1000]} : tensor<1000x1xf32> -> tensor<1000xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1000xf32>
}

// -----

// CHECK-LABEL: @MatMul5dInputs3dWeightsTransposeBTo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x1x2400xf16>, [[ARG1:%.+]]: tensor<1x256x2400xf16>
func.func @MatMul5dInputs3dWeightsTransposeBTo2d(%arg0: tensor<1x1x1x1x2400xf16>, %arg1: tensor<1x256x2400xf16>) -> tensor<1x1x1x1x256xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x1x1x1x2400xf16>, tensor<1x256x2400xf16> -> tensor<1x1x1x1x256xf16>

    return %0 : tensor<1x1x1x1x256xf16>

    // CHECK:       %[[RESHAPE_0:.*]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [0], [1]], shape_value = [1, 2400]} : tensor<1x1x1x1x2400xf16> -> tensor<1x2400xf16>
    // CHECK:       %[[RESHAPE_1:.*]] = IE.AffineReshape([[ARG1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [256, 2400]} : tensor<1x256x2400xf16> -> tensor<256x2400xf16>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_0]], %[[RESHAPE_1]]) {transpose_b} : tensor<1x2400xf16>, tensor<256x2400xf16> -> tensor<1x256xf16>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2, 3], [4]], shape_value = [1, 1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x1x256xf16>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1x1x1x1x256xf16>
}

// -----

// CHECK-LABEL: @MatMul5dInputs3dWeightsTransposeATo2d
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x2400x1xf16>, [[ARG1:%.+]]: tensor<1x2400x256xf16>
func.func @MatMul5dInputs3dWeightsTransposeATo2d(%arg0: tensor<1x1x1x2400x1xf16>, %arg1: tensor<1x2400x256xf16>) -> tensor<1x1x1x1x256xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_a} : tensor<1x1x1x2400x1xf16>, tensor<1x2400x256xf16> -> tensor<1x1x1x1x256xf16>

    return %0 : tensor<1x1x1x1x256xf16>

    // CHECK:       %[[RESHAPE_0:.*]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [0], [1]], shape_value = [2400, 1]} : tensor<1x1x1x2400x1xf16> -> tensor<2400x1xf16>
    // CHECK:       %[[RESHAPE_1:.*]] = IE.AffineReshape([[ARG1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [2400, 256]} : tensor<1x2400x256xf16> -> tensor<2400x256xf16>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_0]], %[[RESHAPE_1]]) {transpose_a} : tensor<2400x1xf16>, tensor<2400x256xf16> -> tensor<1x256xf16>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]]
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2, 3], [4]], shape_value = [1, 1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x1x256xf16>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1x1x1x1x256xf16>
}

// CHECK-LABEL: @FuseReshapesAfterMatmulConcat
func.func @FuseReshapesAfterMatmulConcat(%arg0: tensor<1x8x4096x40xf32>, %arg1: tensor<1x8x4096x40xf32>) -> tensor<8x4096x4096xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x4096x40xf32>, tensor<1x8x4096x40xf32> -> tensor<1x8x4096x4096xf32>
    %1 = IE.Reshape(%0) {shape_value = [8, 4096, 4096]} : tensor<1x8x4096x4096xf32> -> tensor<8x4096x4096xf32>
    %2 = IE.SoftMax(%1) {axisInd = 2 : i64} : tensor<8x4096x4096xf32> -> tensor<8x4096x4096xf32>
    return %2 : tensor<8x4096x4096xf32>

    // CHECK: %[[CONCAT:.+]] = IE.Concat
    // CHECK: %[[RESHAPE:.+]] = IE.AffineReshape(%[[CONCAT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [8, 4096, 4096]} : tensor<32768x4096xf32> -> tensor<8x4096x4096xf32>
    // CHECK: %[[SOFTMAX:.+]] = IE.SoftMax(%[[RESHAPE]])
    // CHECK: return %[[SOFTMAX]] : tensor<8x4096x4096xf32>
}
