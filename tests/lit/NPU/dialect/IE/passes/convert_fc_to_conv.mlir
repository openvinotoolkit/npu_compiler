//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-fc-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertFullyConnectedToConvolution
func.func @ConvertFullyConnectedToConvolution(%arg0: tensor<1x16xf32>) -> tensor<1x64xf32> {
    %weights = const.Declare tensor<64x16xf32> = dense<1.0> : tensor<64x16xf32>
    %bias = const.Declare tensor<1x64xf32> = dense<1.0> : tensor<1x64xf32>
    %0 = IE.FullyConnected(%arg0, %weights, %bias) : tensor<1x16xf32>, tensor<64x16xf32>, tensor<1x64xf32> -> tensor<1x64xf32>

    return %0 : tensor<1x64xf32>

    // CHECK-NOT:   IE.FullyConnected
    // CHECK-DAG:       %[[WEIGHTS:.*]] = const.Declare tensor<64x16xf32> = dense<1.000000e+00> : tensor<64x16xf32>
    // CHECK-DAG:       %[[BIAS:.*]] = const.Declare tensor<1x64xf32> = dense<1.000000e+00> : tensor<1x64xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 1, 1]} : tensor<1x16xf32> -> tensor<1x16x1x1xf32>
    // CHECK:       %[[VAL1:.*]] = IE.Reshape(%[[WEIGHTS]]) {shape_value = [64, 16, 1, 1]} : tensor<64x16xf32> -> tensor<64x16x1x1xf32>
    // CHECK:       %[[VAL2:.*]] = IE.Reshape(%[[BIAS]]) {shape_value = [1, 64, 1, 1]} : tensor<1x64xf32> -> tensor<1x64x1x1xf32>
    // CHECK:       %[[CONV:.*]] = IE.Convolution(%[[VAL0]], %[[VAL1]], %[[VAL2]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       %[[VAL3:.*]] = IE.Reshape(%[[CONV]]) {shape_value = [1, 64]} : tensor<1x64x1x1xf32> -> tensor<1x64xf32>
    // CHECK:       return %[[VAL3]]
}
