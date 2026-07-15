//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: not vpux-opt --split-input-file --init-compiler="platform=%platform%" --debatcher="debatcher-input-coefficients-partitions=[1-2],[2-2]" %s 2>&1 | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK: DebatchCoeffDescription expects the batch position to be 0, got: d1 in [1-2]
func.func @SingleInputSingleOutputBatched(%arg: tensor<6x3x62x62xf32>) -> tensor<6x48x60x57xf32> {
    %cst = const.Declare tensor<48x3x3x6xf32> = dense<1.0> : tensor<48x3x3x6xf32>
    %0 = IE.Convolution(%arg, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<6x3x62x62xf32>, tensor<48x3x3x6xf32> -> tensor<6x48x60x57xf32>
    %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<6x48x60x57xf32> -> tensor<6x48x60x57xf32>
    return %1 : tensor<6x48x60x57xf32>
}

// -----

// CHECK: DebatchCoeffDescription expects the batch position to be 0, got: d1 in [1-2]
func.func @MultipleInputSingleOutputBatched(%arg0: tensor<6x3x62x62xf32>, %arg1: tensor<6x48x60x57xf32>) -> tensor<6x48x60x57xf32> {
    %cst = const.Declare tensor<48x3x3x6xf32> = dense<1.0> : tensor<48x3x3x6xf32>
    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<6x3x62x62xf32>, tensor<48x3x3x6xf32> -> tensor<6x48x60x57xf32>
    %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<6x48x60x57xf32> -> tensor<6x48x60x57xf32>
    %2 = IE.Add(%1, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<6x48x60x57xf32>, tensor<6x48x60x57xf32> -> tensor<6x48x60x57xf32>
    %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<6x48x60x57xf32> -> tensor<6x48x60x57xf32>
    return %3: tensor<6x48x60x57xf32>
}
