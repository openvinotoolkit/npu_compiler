//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --initial-transformations %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @FullyConnected
func.func @FullyConnected(%arg0: tensor<1x16xf32>) -> tensor<1x64xf32> {
    %weights = const.Declare tensor<64x16xf32> = dense<1.0> : tensor<64x16xf32>
    %bias = const.Declare tensor<1x64xf32> = dense<1.0> : tensor<1x64xf32>
    %0 = IE.FullyConnected(%arg0, %weights, %bias) : tensor<1x16xf32>, tensor<64x16xf32>, tensor<1x64xf32> -> tensor<1x64xf32>

    return %0 : tensor<1x64xf32>

    // CHECK-NOT:   IE.Convolution
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<64x16xf32> = dense<1.000000e+00> : tensor<64x16xf32>
    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x64xf32> = dense<1.000000e+00> : tensor<1x64xf32>
    // CHECK:       [[FC:%.+]] = IE.FullyConnected(%arg0, [[WEIGHTS]], [[BIAS]])
    // CHECK:       return [[FC]]
}

// -----

// CHECK-LABEL: @MatMul4dInputsTo2d
func.func @MatMul4dInputsTo2d(%arg0: tensor<1x2x1x512xf32>) -> tensor<1x2x1x40xf32> {
    %cst = const.Declare tensor<1x2x512x40xf32> = dense<1.0> : tensor<1x2x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x1x512xf32>, tensor<1x2x512x40xf32> -> tensor<1x2x1x40xf32>

    return %0 : tensor<1x2x1x40xf32>

    // CHECK-DAG:      [[CST_0:%.+]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00>
    // CHECK-DAG:      [[CST_1:%.+]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00>
    // CHECK:          [[IN_1:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:          [[IN_1_2D:%.+]] = IE.AffineReshape([[IN_1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:          [[IN_2:%.+]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:          [[IN_2_2D:%.+]] = IE.AffineReshape([[IN_2]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>

    // CHECK:          [[FC_1:%.+]] = IE.FullyConnected([[IN_1_2D]], [[CST_1]])
    // CHECK-SAME:           : tensor<1x512xf32>, tensor<40x512xf32> -> tensor<1x40xf32>

    // CHECK:          [[FC_2:%.+]] = IE.FullyConnected([[IN_2_2D]], [[CST_0]])
    // CHECK-SAME:           : tensor<1x512xf32>, tensor<40x512xf32> -> tensor<1x40xf32>

    // CHECK:          [[OUT_1_4D:%.+]] = IE.AffineReshape([[FC_1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 40]} : tensor<1x40xf32> -> tensor<1x1x1x40xf32>
    // CHECK:          [[OUT_2_4D:%.+]] = IE.AffineReshape([[FC_2]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 40]} : tensor<1x40xf32> -> tensor<1x1x1x40xf32>

    // CHECK:          [[CONCAT:%.+]] = IE.Concat([[OUT_1_4D]], [[OUT_2_4D]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x40xf32>, tensor<1x1x1x40xf32> -> tensor<1x2x1x40xf32>

    // CHECK return [[OUT]] : tensor<1x2x1x40xf32>
}

// -----

// CHECK-LABEL: @UnrollMatMulAndPropagate
func.func @UnrollMatMulAndPropagate(%arg0: tensor<1x8x4096x40xf32>, %arg1: tensor<1x8x4096x40xf32>) -> tensor<8x4096x4096xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x4096x40xf32>, tensor<1x8x4096x40xf32> -> tensor<1x8x4096x4096xf32>
    %1 = IE.Reshape(%0) {shape_value = [8, 4096, 4096]} : tensor<1x8x4096x4096xf32> -> tensor<8x4096x4096xf32>
    %2 = IE.SoftMax(%1) {axisInd = 2 : i64} : tensor<8x4096x4096xf32> -> tensor<8x4096x4096xf32>
    return %2 : tensor<8x4096x4096xf32>

    // CHECK:       %[[FC:.+]] = IE.FullyConnected
    // CHECK:       %[[AFFINE:.+]] = IE.AffineReshape(%[[FC]])
    // CHECK-SAME{LITERAL}  {dim_mapping = [[0, 1], [2]], shape_value = [1, 4096, 4096]} : tensor<4096x4096xf32> -> tensor<1x4096x4096xf32>
    // CHECK:       %[[SOFTMAX:.+]] = IE.SoftMax(%[[AFFINE]])
    // CHECK:       %[[CONCAT:.+]] = IE.Concat(%[[SOFTMAX]],
    // CHECK:       return %[[CONCAT]] : tensor<8x4096x4096xf32>
}
