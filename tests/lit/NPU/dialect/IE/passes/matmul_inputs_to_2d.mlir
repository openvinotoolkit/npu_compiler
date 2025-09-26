//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --matmul-inputs-to-2d %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @MatMulInputsTo2d
func.func @MatMulInputsTo2d(%arg0: tensor<2x1x512xf32>) -> tensor<2x1x40xf32> {
    %cst = const.Declare tensor<2x512x40xf32> = dense<1.0> : tensor<2x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<2x1x512xf32>, tensor<2x512x40xf32> -> tensor<2x1x40xf32>

    return %0 : tensor<2x1x40xf32>

    // CHECK-DAG:   %[[CST_1_2D:.*]] = const.Declare tensor<512x40xf32> = dense<1.000000e+00> : tensor<2x512x40xf32>, [#const.SubView<[0, 0, 0], [1, 512, 40]>, #const.AffineReshape<{{\[\[}}0], [0], [1]], [512, 40]>]
    // CHECK-DAG:   %[[CST_2_2D:.*]] = const.Declare tensor<512x40xf32> = dense<1.000000e+00> : tensor<2x512x40xf32>, [#const.SubView<[1, 0, 0], [1, 512, 40]>, #const.AffineReshape<{{\[\[}}0], [0], [1]], [512, 40]>]
    // CHECK:       %[[IN_1:.*]] = IE.Slice %arg0 [0, 0, 0] [1, 1, 512] : tensor<2x1x512xf32> to tensor<1x1x512xf32>
    // CHECK:       %[[IN_1_2D:.*]] = IE.AffineReshape(%[[IN_1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[IN_2:.*]] = IE.Slice %arg0 [1, 0, 0] [1, 1, 512] : tensor<2x1x512xf32> to tensor<1x1x512xf32>
    // CHECK:       %[[IN_2_2D:.*]] = IE.AffineReshape(%[[IN_2]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[MATMUL_1:.*]] = IE.MatMul(%[[IN_1_2D]], %[[CST_1_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[MATMUL_2:.*]] = IE.MatMul(%[[IN_2_2D]], %[[CST_2_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[MATMUL_1]], %[[MATMUL_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x40xf32>, tensor<1x40xf32> -> tensor<2x40xf32>
    // CHECK:       %[[OUT:.*]] = IE.AffineReshape(%[[CONCAT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [2, 1, 40]} : tensor<2x40xf32> -> tensor<2x1x40xf32>
    // CHECK:       return %[[OUT]] : tensor<2x1x40xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInputsTo2d
func.func @MatMul4dInputsTo2d(%arg0: tensor<1x2x1x512xf32>) -> tensor<1x2x1x40xf32> {
    %cst = const.Declare tensor<1x2x512x40xf32> = dense<1.0> : tensor<1x2x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x1x512xf32>, tensor<1x2x512x40xf32> -> tensor<1x2x1x40xf32>

    return %0 : tensor<1x2x1x40xf32>

    // CHECK-DAG:   %[[CST_1_2D:.*]] = const.Declare tensor<512x40xf32> = dense<1.000000e+00> : tensor<1x2x512x40xf32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 512, 40]>, #const.AffineReshape<{{\[\[}}0], [0], [0], [1]], [512, 40]>]
    // CHECK-DAG:   %[[CST_2_2D:.*]] = const.Declare tensor<512x40xf32> = dense<1.000000e+00> : tensor<1x2x512x40xf32>, [#const.SubView<[0, 1, 0, 0], [1, 1, 512, 40]>, #const.AffineReshape<{{\[\[}}0], [0], [0], [1]], [512, 40]>]
    // CHECK:       %[[IN_1:.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:       %[[IN_1_2D:.*]] = IE.AffineReshape(%[[IN_1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[IN_2:.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:       %[[IN_2_2D:.*]] = IE.AffineReshape(%[[IN_2]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[MATMUL_1:.*]] = IE.MatMul(%[[IN_1_2D]], %[[CST_1_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[MATMUL_2:.*]] = IE.MatMul(%[[IN_2_2D]], %[[CST_2_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[MATMUL_1]], %[[MATMUL_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x40xf32>, tensor<1x40xf32> -> tensor<2x40xf32>
    // CHECK:       %[[OUT:.*]] = IE.AffineReshape(%[[CONCAT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 1, 40]} : tensor<2x40xf32> -> tensor<1x2x1x40xf32>
    // CHECK:       return %[[OUT]] : tensor<1x2x1x40xf32>
}

// -----

// CHECK-LABEL: @MatMul3dInputsBatch1To2d
func.func @MatMul3dInputsBatch1To2d(%arg0: tensor<1x1x1024xf32>) -> tensor<1x1x512xf32> {
    %cst = const.Declare tensor<1x1024x512xf32> = dense<1.0> : tensor<1x1024x512xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x1x1024xf32>, tensor<1x1024x512xf32> -> tensor<1x1x512xf32>

    return %0 : tensor<1x1x512xf32>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<1024x512xf32> = dense<1.000000e+00> : tensor<1x1024x512xf32>, [#const.Reshape<[1024, 512]>]
    // CHECK: %[[RESHAPE0:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [1, 1024]} : tensor<1x1x1024xf32> -> tensor<1x1024xf32>
    // CHECK: %[[MTML:.*]] = IE.MatMul(%[[RESHAPE0]], %[[CST]]) : tensor<1x1024xf32>, tensor<1024x512xf32> -> tensor<1x512xf32>
    // CHECK: %[[OUT:.*]] = IE.AffineReshape(%[[MTML]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 512]} : tensor<1x512xf32> -> tensor<1x1x512xf32>
    // CHECK: return %[[OUT]] : tensor<1x1x512xf32>
}

// -----

// CHECK-LABEL: @NoChangesMatMul3dInput2DWeightsTo2d
func.func @NoChangesMatMul3dInput2DWeightsTo2d(%arg0: tensor<1x4x9728xf32>) -> tensor<1x4x512xf32> {
    %cst = const.Declare tensor<9728x512xf32> = dense<1.0> : tensor<9728x512xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x4x9728xf32>, tensor<9728x512xf32> -> tensor<1x4x512xf32>

    return %0 : tensor<1x4x512xf32>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<9728x512xf32> = dense<1.000000e+00> : tensor<9728x512xf32>
    // CHECK: %[[RESHAPE:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [4, 9728]} : tensor<1x4x9728xf32> -> tensor<4x9728xf32>
    // CHECK: %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE]], %[[CST]]) : tensor<4x9728xf32>, tensor<9728x512xf32> -> tensor<4x512xf32>
    // CHECK: %[[OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [1, 4, 512]} : tensor<4x512xf32> -> tensor<1x4x512xf32>
    // CHECK: return %[[OUT]] : tensor<1x4x512xf32>
}

// -----

// CHECK-LABEL: @NoChangesMatMul3dInputWithMulChannel2DWeightsTo2d
func.func @NoChangesMatMul3dInputWithMulChannel2DWeightsTo2d(%arg0: tensor<2x64x128xf32>) -> tensor<2x64x64xf32> {
    %cst = const.Declare tensor<128x64xf32> = dense<1.0> : tensor<128x64xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<2x64x128xf32>, tensor<128x64xf32> -> tensor<2x64x64xf32>

    return %0 : tensor<2x64x64xf32>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<128x64xf32> = dense<1.000000e+00> : tensor<128x64xf32>
    // CHECK: %[[RESHAPE:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [128, 128]} : tensor<2x64x128xf32> -> tensor<128x128xf32>
    // CHECK: %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE]], %[[CST]]) : tensor<128x128xf32>, tensor<128x64xf32> -> tensor<128x64xf32>
    // CHECK: %[[OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [2, 64, 64]} : tensor<128x64xf32> -> tensor<2x64x64xf32>
    // CHECK: return %[[OUT]] : tensor<2x64x64xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInputWithMulChannel3dWeightsTo2d
func.func @MatMul4dInputWithMulChannel3dWeightsTo2d(%arg0: tensor<1x2x16x2xf32>) -> tensor<1x2x16x2xf32> {
    %cst = const.Declare tensor<1x2x2xf32> = dense<1.0> : tensor<1x2x2xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x16x2xf32>, tensor<1x2x2xf32> -> tensor<1x2x16x2xf32>

    return %0 : tensor<1x2x16x2xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<1x2x2xf32>, [#const.Reshape<[2, 2]>]
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [32, 2]} : tensor<1x2x16x2xf32> -> tensor<32x2xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) : tensor<32x2xf32>, tensor<2x2xf32> -> tensor<32x2xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 16, 2]} : tensor<32x2xf32> -> tensor<1x2x16x2xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1x2x16x2xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInput2dWeightsNBatchTo2d
func.func @MatMul4dInput2dWeightsNBatchTo2d(%arg0: tensor<16x2x16x2xf32>) -> tensor<16x2x16x4xf32> {
    %cst = const.Declare tensor<4x2xf32> = dense<1.0> : tensor<4x2xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<16x2x16x2xf32>, tensor<4x2xf32> -> tensor<16x2x16x4xf32>

    return %0 : tensor<16x2x16x4xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<4x2xf32> = dense<1.000000e+00> : tensor<4x2xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 2]} : tensor<16x2x16x2xf32> -> tensor<512x2xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) {transpose_b} : tensor<512x2xf32>, tensor<4x2xf32> -> tensor<512x4xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [16, 2, 16, 4]} : tensor<512x4xf32> -> tensor<16x2x16x4xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<16x2x16x4xf32>
}

// -----

// CHECK-LABEL: @MatMul6dInput2dWeights1BatchTo2d
func.func @MatMul6dInput2dWeights1BatchTo2d(%arg0: tensor<1x8x16x2x16x2xf32>) -> tensor<1x8x16x2x16x4xf32> {
    %cst = const.Declare tensor<4x2xf32> = dense<1.0> : tensor<4x2xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<1x8x16x2x16x2xf32>, tensor<4x2xf32> -> tensor<1x8x16x2x16x4xf32>

    return %0 : tensor<1x8x16x2x16x4xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<4x2xf32> = dense<1.000000e+00> : tensor<4x2xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [0], [0], [1]], shape_value = [4096, 2]} : tensor<1x8x16x2x16x2xf32> -> tensor<4096x2xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) {transpose_b} : tensor<4096x2xf32>, tensor<4x2xf32> -> tensor<4096x4xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2, 3, 4], [5]], shape_value = [1, 8, 16, 2, 16, 4]} : tensor<4096x4xf32> -> tensor<1x8x16x2x16x4xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1x8x16x2x16x4xf32>
}

// -----

// CHECK-LABEL: @MatMul5dInput2dWeightsTo2dNoTranspose
func.func @MatMul5dInput2dWeightsTo2dNoTranspose(%arg0: tensor<5x6x7x8x16xf32>) -> tensor<5x6x7x8x32xf32> {
    %cst = const.Declare tensor<16x32xf32> = dense<1.0> : tensor<16x32xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<5x6x7x8x16xf32>, tensor<16x32xf32> -> tensor<5x6x7x8x32xf32>

    return %0 : tensor<5x6x7x8x32xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<16x32xf32> = dense<1.000000e+00> : tensor<16x32xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [0], [1]], shape_value = [1680, 16]} : tensor<5x6x7x8x16xf32> -> tensor<1680x16xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) : tensor<1680x16xf32>, tensor<16x32xf32> -> tensor<1680x32xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]]
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2, 3], [4]], shape_value = [5, 6, 7, 8, 32]} : tensor<1680x32xf32> -> tensor<5x6x7x8x32xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<5x6x7x8x32xf32>
}

// -----

// CHECK-LABEL: @MatMul5dInput2dWeightsTransposeATo3d
func.func @MatMul5dInput2dWeightsTransposeATo3d(%arg0: tensor<5x6x7x16x8xf32>) -> tensor<5x6x7x8x32xf32> {
    %cst = const.Declare tensor<16x32xf32> = dense<1.0> : tensor<16x32xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_a} : tensor<5x6x7x16x8xf32>, tensor<16x32xf32> -> tensor<5x6x7x8x32xf32>

    return %0 : tensor<5x6x7x8x32xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<16x32xf32> = dense<1.000000e+00> : tensor<16x32xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1], [2]], shape_value = [210, 16, 8]} : tensor<5x6x7x16x8xf32> -> tensor<210x16x8xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) {transpose_a} : tensor<210x16x8xf32>, tensor<16x32xf32> -> tensor<210x8x32xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3], [4]], shape_value = [5, 6, 7, 8, 32]} : tensor<210x8x32xf32> -> tensor<5x6x7x8x32xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<5x6x7x8x32xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInputs4dWeightsTo2d
func.func @MatMul4dInputs4dWeightsTo2d(%arg0: tensor<2x2x10x3xf32>, %arg1: tensor<2x2x10x3xf32>) -> tensor<2x2x10x10xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<2x2x10x3xf32>, tensor<2x2x10x3xf32> -> tensor<2x2x10x10xf32>

    return %0 : tensor<2x2x10x10xf32>

    // CHECK: %[[RESHAPE_IN1:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1], [2]], shape_value = [4, 10, 3]} : tensor<2x2x10x3xf32> -> tensor<4x10x3xf32>
    // CHECK: %[[RESHAPE_IN2:.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1], [2]], shape_value = [4, 10, 3]} : tensor<2x2x10x3xf32> -> tensor<4x10x3xf32>
    // CHECK: %[[SLICE1:.*]] = IE.Slice %[[RESHAPE_IN1:.*]] [0, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_1:.*]] = IE.AffineReshape(%[[SLICE1:.*]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE2:.*]] = IE.Slice %[[RESHAPE_IN1:.*]] [1, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_2:.*]] = IE.AffineReshape(%[[SLICE2:.*]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE3:.*]] = IE.Slice %[[RESHAPE_IN1:.*]] [2, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_3:.*]] = IE.AffineReshape(%[[SLICE3:.*]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE4:.*]] = IE.Slice %[[RESHAPE_IN1:.*]] [3, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_4:.*]] = IE.AffineReshape(%[[SLICE4:.*]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE5:.*]] = IE.Slice %[[RESHAPE_IN2:.*]] [0, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_5:.*]] = IE.AffineReshape(%[[SLICE5:.*]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE6:.*]] = IE.Slice %[[RESHAPE_IN2:.*]] [1, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_6:.*]] = IE.AffineReshape(%[[SLICE6:.*]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE7:.*]] = IE.Slice %[[RESHAPE_IN2:.*]] [2, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_7:.*]] = IE.AffineReshape(%[[SLICE7:.*]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE8:.*]] = IE.Slice %[[RESHAPE_IN2:.*]] [3, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_8:.*]] = IE.AffineReshape(%[[SLICE8:.*]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[MATMUL_1:.*]] = IE.MatMul(%[[RESHAPE_1:.*]], %[[RESHAPE_5:.*]]) {transpose_b} : tensor<10x3xf32>, tensor<10x3xf32> -> tensor<10x10xf32>
    // CHECK: %[[MATMUL_2:.*]] = IE.MatMul(%[[RESHAPE_2:.*]], %[[RESHAPE_6:.*]]) {transpose_b} : tensor<10x3xf32>, tensor<10x3xf32> -> tensor<10x10xf32>
    // CHECK: %[[MATMUL_3:.*]] = IE.MatMul(%[[RESHAPE_3:.*]], %[[RESHAPE_7:.*]]) {transpose_b} : tensor<10x3xf32>, tensor<10x3xf32> -> tensor<10x10xf32>
    // CHECK: %[[MATMUL_4:.*]] = IE.MatMul(%[[RESHAPE_4:.*]], %[[RESHAPE_8:.*]]) {transpose_b} : tensor<10x3xf32>, tensor<10x3xf32> -> tensor<10x10xf32>
    // CHECK: %[[CONCAT:.*]] = IE.Concat(%[[MATMUL_1:.*]], %[[MATMUL_2:.*]], %[[MATMUL_3:.*]], %[[MATMUL_4:.*]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32> -> tensor<40x10xf32>
    // CHECK: %[[RESHAPE_9:.*]] = IE.AffineReshape(%[[CONCAT:.*]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2]], shape_value = [4, 10, 10]} : tensor<40x10xf32> -> tensor<4x10x10xf32>
    // CHECK: %[[RESHAPE_10:.*]] = IE.AffineReshape(%[[RESHAPE_9:.*]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [3]], shape_value = [2, 2, 10, 10]} : tensor<4x10x10xf32> -> tensor<2x2x10x10xf32>
    // CHECK: return %[[RESHAPE_10:.*]] : tensor<2x2x10x10xf32>
}

// -----

// CHECK-LABEL: @MatMulVectorMatrixTo2D
func.func @MatMulVectorMatrixTo2D(%arg0: tensor<1024xf32>) -> tensor<1000xf32> {
    %cst = const.Declare tensor<1024x1000xf32> = dense<1.0> : tensor<1024x1000xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1024xf32>, tensor<1024x1000xf32> -> tensor<1000xf32>

    return %0 : tensor<1000xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<1024x1000xf32> = dense<1.000000e+00> : tensor<1024x1000xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1]], shape_value = [1, 1024]} : tensor<1024xf32> -> tensor<1x1024xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) : tensor<1x1024xf32>, tensor<1024x1000xf32> -> tensor<1x1000xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0]], shape_value = [1000]} : tensor<1x1000xf32> -> tensor<1000xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1000xf32>
}

// -----

// CHECK-LABEL: @MatMulMatrixVectorTo2D
func.func @MatMulMatrixVectorTo2D(%arg0: tensor<1000x1024xf32>) -> tensor<1000xf32> {
    %cst = const.Declare tensor<1024xf32> = dense<1.0> : tensor<1024xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1000x1024xf32>, tensor<1024xf32> -> tensor<1000xf32>

    return %0 : tensor<1000xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<1024x1xf32> = dense<1.000000e+00> : tensor<1024xf32>, [#const.Reshape<[1024, 1]>]
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%arg0, %[[CST]]) : tensor<1000x1024xf32>, tensor<1024x1xf32> -> tensor<1000x1xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0]], shape_value = [1000]} : tensor<1000x1xf32> -> tensor<1000xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1000xf32>
}

// -----

// CHECK-LABEL: @MatMul5dInputs3dWeightsTransposeBTo2d
func.func @MatMul5dInputs3dWeightsTransposeBTo2d(%arg0: tensor<1x1x1x1x2400xf16>, %arg1: tensor<1x256x2400xf16>) -> tensor<1x1x1x1x256xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x1x1x1x2400xf16>, tensor<1x256x2400xf16> -> tensor<1x1x1x1x256xf16>

    return %0 : tensor<1x1x1x1x256xf16>

    // CHECK:       %[[RESHAPE_0:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [0], [1]], shape_value = [1, 2400]} : tensor<1x1x1x1x2400xf16> -> tensor<1x2400xf16>
    // CHECK:       %[[RESHAPE_1:.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [256, 2400]} : tensor<1x256x2400xf16> -> tensor<256x2400xf16>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_0]], %[[RESHAPE_1]]) {transpose_b} : tensor<1x2400xf16>, tensor<256x2400xf16> -> tensor<1x256xf16>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2, 3], [4]], shape_value = [1, 1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x1x256xf16>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1x1x1x1x256xf16>
}

// -----

// CHECK-LABEL: @MatMul5dInputs3dWeightsTransposeATo2d
func.func @MatMul5dInputs3dWeightsTransposeATo2d(%arg0: tensor<1x1x1x2400x1xf16>, %arg1: tensor<1x2400x256xf16>) -> tensor<1x1x1x1x256xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_a} : tensor<1x1x1x2400x1xf16>, tensor<1x2400x256xf16> -> tensor<1x1x1x1x256xf16>

    return %0 : tensor<1x1x1x1x256xf16>

    // CHECK:       %[[RESHAPE_0:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [0], [1]], shape_value = [2400, 1]} : tensor<1x1x1x2400x1xf16> -> tensor<2400x1xf16>
    // CHECK:       %[[RESHAPE_1:.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1]], shape_value = [2400, 256]} : tensor<1x2400x256xf16> -> tensor<2400x256xf16>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_0]], %[[RESHAPE_1]]) {transpose_a} : tensor<2400x1xf16>, tensor<2400x256xf16> -> tensor<1x256xf16>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.AffineReshape(%[[MATMUL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2, 3], [4]], shape_value = [1, 1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x1x256xf16>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1x1x1x1x256xf16>
}

// -----

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

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @matmulBroadcastSupport(
// CHECK-SAME:  [[ARG_0:.+]]: tensor<1x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 35, 512]> : tensor<3xsi64>, order = #CHW}>) -> tensor<1x2x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 512]> : tensor<4xsi64>, order = #NCHW}> {
func.func @matmulBroadcastSupport(%arg0: tensor<1x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 35, 512]> : tensor<3xsi64>, order = #CHW}>) -> tensor<1x2x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 512]> : tensor<4xsi64>, order = #NCHW}> {
    %cst = const.Declare tensor<4xsi32> = dense<[1, 2, -1, 512]> : tensor<4xsi32>
    %cst_0 = const.Declare tensor<4xsi32> = dense<[1, 2, -9223372036854775808, 512]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    %cst_1 = const.Declare tensor<1x2x512x512xf32> = dense<0.000000e+00> : tensor<2x512x512xf32>, [#const.Reshape<[1, 2, 512, 512]>]
    %cst_3 = const.Declare tensor<4xsi32> = dense<[1, 1, -1, 512]> : tensor<4xsi32>

    %0 = IE.DynamicReshape(%arg0, %cst_3) {output_bounds = [1, 1, 35, 512], output_shape = [1, 1, -9223372036854775808, 512]} : tensor<1x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 35, 512]> : tensor<3xsi64>, order = #CHW}>, tensor<4xsi32> -> tensor<1x1x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>
    %1 = IE.DynamicBroadcast(%0, %cst_0) {mode = #IE.broadcast_type<NUMPY>, output_bounds = [1, 1, 35, 512], output_shape = [1, 1, -9223372036854775808, 512]} : tensor<1x1x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi32> -> tensor<1x1x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>
    %2 = IE.DynamicExpand(%1) : tensor<1x1x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x35x512xf32>
    %3 = IE.MatMul(%2, %cst_1) {transpose_b} : tensor<1x1x35x512xf32>, tensor<1x2x512x512xf32> -> tensor<1x2x35x512xf32>
    %4 = IE.DynamicReshape(%3, %cst) {output_bounds = [1, 2, 35, 512], output_shape = [1, 2, -9223372036854775808, 512]} : tensor<1x2x35x512xf32>, tensor<4xsi32> -> tensor<1x2x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 512]> : tensor<4xsi64>, order = #NCHW}>

    return %4 : tensor<1x2x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 512]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[CST:%.+]] = const.Declare tensor<512x512xf32> = dense<0.000000e+00> : tensor<2x512x512xf32>, [#const.SubView<[1, 0, 0], [1, 512, 512]>, #const.Reshape<[1, 1, 512, 512]>, #const.AffineReshape<{{\[\[}}0], [0], [0], [1]], [512, 512]>]
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<512x512xf32> = dense<0.000000e+00> : tensor<2x512x512xf32>, [#const.SubView<[0, 0, 0], [1, 512, 512]>, #const.Reshape<[1, 1, 512, 512]>, #const.AffineReshape<{{\[\[}}0], [0], [0], [1]], [512, 512]>]
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 2, -1, 512]> : tensor<4xsi32>
    // CHECK: [[CST_2:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 2, -9223372036854775808, 512]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK: [[CST_3:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 1, -1, 512]> : tensor<4xsi32>
    // CHECK: [[DYN_RESHAPE:%.+]] = IE.DynamicReshape([[ARG_0]], [[CST_3]]) {output_bounds = [1, 1, 35, 512], output_shape = [1, 1, -9223372036854775808, 512]} : tensor<1x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 35, 512]> : tensor<3xsi64>, order = #CHW}>, tensor<4xsi32> -> tensor<1x1x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[DYN_BROADCAST:%.+]] = IE.DynamicBroadcast([[DYN_RESHAPE]], [[CST_2]]) {mode = #IE.broadcast_type<NUMPY>, output_bounds = [1, 1, 35, 512], output_shape = [1, 1, -9223372036854775808, 512]} : tensor<1x1x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi32> -> tensor<1x1x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[DYN_EXPAND:%.+]] = IE.DynamicExpand([[DYN_BROADCAST]]) : tensor<1x1x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x35x512xf32>
    // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[DYN_EXPAND]])
    // CHECK: [[MAT_MUL:%.+]] = IE.MatMul([[RESHAPE]], [[CST_0]]) {transpose_b} : tensor<35x512xf32>, tensor<512x512xf32> -> tensor<35x512xf32>
    // CHECK: [[MAT_MUL_0:%.+]] = IE.MatMul([[RESHAPE]], [[CST]]) {transpose_b} : tensor<35x512xf32>, tensor<512x512xf32> -> tensor<35x512xf32>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[MAT_MUL]], [[MAT_MUL_0]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<35x512xf32>, tensor<35x512xf32> -> tensor<70x512xf32>
    // CHECK: [[RESHAPE_0:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK: [[DYN_RESHAPE_0:%.+]] = IE.DynamicReshape([[RESHAPE_0]], [[CST_1]]) {output_bounds = [1, 2, 35, 512], output_shape = [1, 2, -9223372036854775808, 512]} : tensor<1x2x35x512xf32>, tensor<4xsi32> -> tensor<1x2x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 512]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: return [[DYN_RESHAPE_0]] : tensor<1x2x?x512xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 512]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

// CHECK-LABEL: func.func @matmulBroadcastSupportReproducer(
// CHECK-SAME:  [[ARG_0:.+]]: tensor<1x1x4x3xf16>) -> tensor<1x6x4x10xf16> {
func.func @matmulBroadcastSupportReproducer(%arg0: tensor<1x1x4x3xf16>) -> tensor<1x6x4x10xf16> {
    %cst = const.Declare tensor<1x6x3x10xf32> = dense<0.000000e+00> : tensor<1x6x3x10xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x1x4x3xf16> -> tensor<1x1x4x3xf32>
    %1 = IE.MatMul(%0, %cst) : tensor<1x1x4x3xf32>, tensor<1x6x3x10xf32> -> tensor<1x6x4x10xf32>
    %2 = IE.Convert(%1) {dstElemType = f16} : tensor<1x6x4x10xf32> -> tensor<1x6x4x10xf16>
    return %2 : tensor<1x6x4x10xf16>

    // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG_0]]) {dstElemType = f32} : tensor<1x1x4x3xf16> -> tensor<1x1x4x3xf32>
    // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[CONVERT]])
    // CHECK: [[MAT_MUL_0:%.+]] = IE.MatMul([[RESHAPE]], {{%.+}}) : tensor<4x3xf32>, tensor<3x10xf32> -> tensor<4x10xf32>
    // CHECK: [[MAT_MUL_1:%.+]] = IE.MatMul([[RESHAPE]], {{%.+}}) : tensor<4x3xf32>, tensor<3x10xf32> -> tensor<4x10xf32>
    // CHECK: [[MAT_MUL_2:%.+]] = IE.MatMul([[RESHAPE]], {{%.+}}) : tensor<4x3xf32>, tensor<3x10xf32> -> tensor<4x10xf32>
    // CHECK: [[MAT_MUL_3:%.+]] = IE.MatMul([[RESHAPE]], {{%.+}}) : tensor<4x3xf32>, tensor<3x10xf32> -> tensor<4x10xf32>
    // CHECK: [[MAT_MUL_4:%.+]] = IE.MatMul([[RESHAPE]], {{%.+}}) : tensor<4x3xf32>, tensor<3x10xf32> -> tensor<4x10xf32>
    // CHECK: [[MAT_MUL_5:%.+]] = IE.MatMul([[RESHAPE]], {{%.+}}) : tensor<4x3xf32>, tensor<3x10xf32> -> tensor<4x10xf32>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[MAT_MUL_0]], [[MAT_MUL_1]], [[MAT_MUL_2]], [[MAT_MUL_3]], [[MAT_MUL_4]], [[MAT_MUL_5]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<4x10xf32>, tensor<4x10xf32>, tensor<4x10xf32>, tensor<4x10xf32>, tensor<4x10xf32>, tensor<4x10xf32> -> tensor<24x10xf32>
    // CHECK: [[RESHAPE_1:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK: [[CONVERT_1:%.+]] = IE.Convert([[RESHAPE_1]]) {dstElemType = f16} : tensor<1x6x4x10xf32> -> tensor<1x6x4x10xf16>
    // CHECK: return [[CONVERT_1]] : tensor<1x6x4x10xf16>
}

// -----

// CHECK-LABEL: func.func @SwapInputsWhenHeightOne(
// CHECK-SAME:  [[ARG_0:.+]]: tensor<1x6336x1x64xf32>) -> tensor<1x6336x32x1xf32> {
func.func @SwapInputsWhenHeightOne(%arg0: tensor<1x6336x1x64xf32>) -> tensor<1x6336x32x1xf32> {
    %cst = const.Declare tensor<1x1x32x64xf32> = dense<0.000000e+00> : tensor<1x1x32x64xf32>
    %0 = IE.MatMul(%cst, %arg0) {transpose_b} : tensor<1x1x32x64xf32>, tensor<1x6336x1x64xf32> -> tensor<1x6336x32x1xf32>
    return %0 : tensor<1x6336x32x1xf32>

    // CHECK:               [[CST:%.+]] = const.Declare tensor<32x64xf32> = dense<0.000000e+00> : tensor<1x1x32x64xf32>, [#const.Reshape<[32, 64]>]
    // CHECK:               [[RESHAPE_0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [6336, 64]} : tensor<1x6336x1x64xf32> -> tensor<6336x64xf32>
    // CHECK:               [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_0]], [[CST]]) {transpose_b} : tensor<6336x64xf32>, tensor<32x64xf32> -> tensor<6336x32xf32>
    // CHECK:               [[RESHAPE_1:%.+]] = IE.AffineReshape([[MATMUL]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 6336, 1, 32]} : tensor<6336x32xf32> -> tensor<1x6336x1x32xf32>
    // CHECK:               [[RESHAPE_2:%.+]] = IE.AffineReshape([[RESHAPE_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 6336, 32, 1]} : tensor<1x6336x1x32xf32> -> tensor<1x6336x32x1xf32>
    // CHECK:               return [[RESHAPE_2]] : tensor<1x6336x32x1xf32>
}

// -----

// CHECK-LABEL: func.func @SwapInputsWhenWidthOne(
// CHECK-SAME:  [[ARG_0:.+]]: tensor<1x6336x64x1xf32>) -> tensor<1x6336x32x1xf32> {
func.func @SwapInputsWhenWidthOne(%arg0: tensor<1x6336x64x1xf32>) -> tensor<1x6336x32x1xf32> {
    %cst = const.Declare tensor<1x1x32x64xf32> = dense<0.000000e+00> : tensor<1x1x32x64xf32>
    %0 = IE.MatMul(%cst, %arg0) : tensor<1x1x32x64xf32>, tensor<1x6336x64x1xf32> -> tensor<1x6336x32x1xf32>
    return %0 : tensor<1x6336x32x1xf32>

    // CHECK:               [[CST:%.+]] = const.Declare tensor<32x64xf32> = dense<0.000000e+00> : tensor<1x1x32x64xf32>, [#const.Reshape<[32, 64]>]
    // CHECK:               [[RESHAPE_0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2]], shape_value = [6336, 64, 1]} : tensor<1x6336x64x1xf32> -> tensor<6336x64x1xf32>
    // CHECK:               [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_0]], [[CST]]) {transpose_a, transpose_b} : tensor<6336x64x1xf32>, tensor<32x64xf32> -> tensor<6336x1x32xf32>
    // CHECK:               [[RESHAPE_1:%.+]] = IE.AffineReshape([[MATMUL]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 6336, 1, 32]} : tensor<6336x1x32xf32> -> tensor<1x6336x1x32xf32>
    // CHECK:               [[RESHAPE_2:%.+]] = IE.AffineReshape([[RESHAPE_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 6336, 32, 1]} : tensor<1x6336x1x32xf32> -> tensor<1x6336x32x1xf32>
    // CHECK:               return [[RESHAPE_2]] : tensor<1x6336x32x1xf32>
}

// -----

// CHECK-LABEL: func.func @NotSwapInputsWhenHeightOne(
// CHECK-SAME:  [[ARG_0:.+]]: tensor<1x2x1x64xf32>) -> tensor<1x2x32x64xf32> {
func.func @NotSwapInputsWhenHeightOne(%arg0: tensor<1x2x1x64xf32>) -> tensor<1x2x32x64xf32> {
    %cst = const.Declare tensor<1x1x32x1xf32> = dense<0.000000e+00> : tensor<1x1x32x1xf32>
    %0 = IE.MatMul(%cst, %arg0) : tensor<1x1x32x1xf32>, tensor<1x2x1x64xf32> -> tensor<1x2x32x64xf32>
    return %0 : tensor<1x2x32x64xf32>

    // CHECK:               [[CST:%.+]] = const.Declare tensor<32x1xf32> = dense<0.000000e+00> : tensor<1x1x32x1xf32>, [#const.Reshape<[32, 1]>]
    // CHECK:               [[SLICE_0:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 1, 1, 64] : tensor<1x2x1x64xf32> to tensor<1x1x1x64xf32>
    // CHECK:               [[RESHAPE_0:%.+]] = IE.AffineReshape([[SLICE_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 64]} : tensor<1x1x1x64xf32> -> tensor<1x64xf32>
    // CHECK:               [[SLICE_1:%.+]] = IE.Slice [[ARG_0]] [0, 1, 0, 0] [1, 1, 1, 64] : tensor<1x2x1x64xf32> to tensor<1x1x1x64xf32>
    // CHECK:               [[RESHAPE_1:%.+]] = IE.AffineReshape([[SLICE_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 64]} : tensor<1x1x1x64xf32> -> tensor<1x64xf32>
    // CHECK:               [[MATMUL_0:%.+]] = IE.MatMul([[CST]], [[RESHAPE_0]]) : tensor<32x1xf32>, tensor<1x64xf32> -> tensor<32x64xf32>
    // CHECK:               [[MATMUL_1:%.+]] = IE.MatMul([[CST]], [[RESHAPE_1]]) : tensor<32x1xf32>, tensor<1x64xf32> -> tensor<32x64xf32>
    // CHECK:               [[CONCAT:%.+]] = IE.Concat([[MATMUL_0]], [[MATMUL_1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<32x64xf32>, tensor<32x64xf32> -> tensor<64x64xf32>
    // CHECK:               [[RESHAPE_2:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 32, 64]} : tensor<64x64xf32> -> tensor<1x2x32x64xf32>
    // CHECK:               return [[RESHAPE_2]] : tensor<1x2x32x64xf32>
}

// -----

// CHECK-LABEL: func.func @UnrollMatmulSoftmaxMatmul(
func.func @UnrollMatmulSoftmaxMatmul(%arg0: tensor<1x2x730x64xf32>, %arg1: tensor<1x2x730x64xf32>, %arg2: tensor<1x2x64x730xf32>) -> tensor<1x2x730x64xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x2x730x64xf32>, tensor<1x2x730x64xf32> -> tensor<1x2x730x730xf32>
    %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x2x730x730xf32> -> tensor<1x2x730x730xf32>
    %2 = IE.MatMul(%1, %arg2) {transpose_b} : tensor<1x2x730x730xf32>, tensor<1x2x64x730xf32> -> tensor<1x2x730x64xf32>

    return %2 : tensor<1x2x730x64xf32>

    // CHECK:       [[IN_0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 730, 64] : tensor<1x2x730x64xf32> to tensor<1x1x730x64xf32>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[IN_0]])
    // CHECK-SAME{LITERAL}:             {dim_mapping = [[0], [0], [0], [1]], shape_value = [730, 64]} : tensor<1x1x730x64xf32> -> tensor<730x64xf32>
    // CHECK:       [[IN_2:%.+]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 730, 64] : tensor<1x2x730x64xf32> to tensor<1x1x730x64xf32>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.AffineReshape([[IN_2]])
    // CHECK-SAME{LITERAL}:             {dim_mapping = [[0], [0], [0], [1]], shape_value = [730, 64]} : tensor<1x1x730x64xf32> -> tensor<730x64xf32>
    // CHECK:       [[IN_4:%.+]] = IE.Slice %arg1 [0, 0, 0, 0] [1, 1, 730, 64] : tensor<1x2x730x64xf32> to tensor<1x1x730x64xf32>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.AffineReshape([[IN_4]])
    // CHECK-SAME{LITERAL}:             {dim_mapping = [[0], [0], [0], [1]], shape_value = [730, 64]} : tensor<1x1x730x64xf32> -> tensor<730x64xf32>
    // CHECK:       [[IN_6:%.+]] = IE.Slice %arg1 [0, 1, 0, 0] [1, 1, 730, 64] : tensor<1x2x730x64xf32> to tensor<1x1x730x64xf32>
    // CHECK:       [[RESHAPE_7:%.+]] = IE.AffineReshape([[IN_6]])
    // CHECK-SAME{LITERAL}:             {dim_mapping = [[0], [0], [0], [1]], shape_value = [730, 64]} : tensor<1x1x730x64xf32> -> tensor<730x64xf32>

    // CHECK:       [[MATMUL_8:%.+]] = IE.MatMul([[RESHAPE_1]], [[RESHAPE_5]]) {transpose_b} : tensor<730x64xf32>, tensor<730x64xf32> -> tensor<730x730xf32>
    // CHECK:       [[MATMUL_9:%.+]] = IE.MatMul([[RESHAPE_3]], [[RESHAPE_7]]) {transpose_b} : tensor<730x64xf32>, tensor<730x64xf32> -> tensor<730x730xf32>

    // CHECK:       [[CONCAT_10:%.+]] = IE.Concat([[MATMUL_8]], [[MATMUL_9]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<730x730xf32>, tensor<730x730xf32> -> tensor<1460x730xf32>

    // CHECK:       [[RESHAPE_11:%.+]] = IE.AffineReshape([[CONCAT_10]])
    // CHECK-SAME{LITERAL}:             {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 730, 730]} : tensor<1460x730xf32> -> tensor<1x2x730x730xf32>
    // CHECK:       [[SOFTMAX_12:%.+]] = IE.SoftMax([[RESHAPE_11]]) {axisInd = 3 : i64} : tensor<1x2x730x730xf32> -> tensor<1x2x730x730xf32>
    // CHECK:       [[SLICE_13:%.+]] = IE.Slice [[SOFTMAX_12]] [0, 0, 0, 0] [1, 1, 730, 730] : tensor<1x2x730x730xf32> to tensor<1x1x730x730xf32>
    // CHECK:       [[RESHAPE_14:%.+]] = IE.AffineReshape([[SLICE_13]])
    // CHECK-SAME{LITERAL}:             {dim_mapping = [[0], [0], [0], [1]], shape_value = [730, 730]} : tensor<1x1x730x730xf32> -> tensor<730x730xf32>
    // CHECK:       [[SLICE_15:%.+]] = IE.Slice [[SOFTMAX_12]] [0, 1, 0, 0] [1, 1, 730, 730] : tensor<1x2x730x730xf32> to tensor<1x1x730x730xf32>
    // CHECK:       [[RESHAPE_16:%.+]] = IE.AffineReshape([[SLICE_15]])
    // CHECK-SAME{LITERAL}:             {dim_mapping = [[0], [0], [0], [1]], shape_value = [730, 730]} : tensor<1x1x730x730xf32> -> tensor<730x730xf32>
    // CHECK:       [[SLICE_17:%.+]] = IE.Slice %arg2 [0, 0, 0, 0] [1, 1, 64, 730] : tensor<1x2x64x730xf32> to tensor<1x1x64x730xf32>
    // CHECK:       [[RESHAPE_18:%.+]] = IE.AffineReshape([[SLICE_17]])
    // CHECK-SAME{LITERAL}:             {dim_mapping = [[0], [0], [0], [1]], shape_value = [64, 730]} : tensor<1x1x64x730xf32> -> tensor<64x730xf32>
    // CHECK:       [[SLICE_19:%.+]] = IE.Slice %arg2 [0, 1, 0, 0] [1, 1, 64, 730] : tensor<1x2x64x730xf32> to tensor<1x1x64x730xf32>
    // CHECK:       [[RESHAPE_20:%.+]] = IE.AffineReshape([[SLICE_19]])
    // CHECK-SAME{LITERAL}:             {dim_mapping = [[0], [0], [0], [1]], shape_value = [64, 730]} : tensor<1x1x64x730xf32> -> tensor<64x730xf32>

    // CHECK:       [[MATMUL_21:%.+]] = IE.MatMul([[RESHAPE_14]], [[RESHAPE_18]]) {transpose_b} : tensor<730x730xf32>, tensor<64x730xf32> -> tensor<730x64xf32>
    // CHECK:       [[MATMUL_22:%.+]] = IE.MatMul([[RESHAPE_16]], [[RESHAPE_20]]) {transpose_b} : tensor<730x730xf32>, tensor<64x730xf32> -> tensor<730x64xf32>
    // CHECK:       [[CONCAT_23:%.+]] = IE.Concat([[MATMUL_21]], [[MATMUL_22]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<730x64xf32>, tensor<730x64xf32> -> tensor<1460x64xf32>

    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[CONCAT_23]])
    // CHECK-SAME{LITERAL}:             {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 730, 64]} : tensor<1460x64xf32> -> tensor<1x2x730x64xf32>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x2x730x64xf32>
}
