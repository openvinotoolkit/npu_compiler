//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @FuseConvAndBias
func @FuseConvAndBias(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %filters = const.Declare tensor<16x3x3x3xf32> = dense<1.0> : tensor<16x3x3x3xf32>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x3x300x300xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x300x300xf32>

    %bias = const.Declare tensor<1x16x1x1xf32> = dense<1.0> : tensor<1x16x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias)
        {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} :
        tensor<1x16x300x300xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x300x300xf32>

    return %1 : tensor<1x16x300x300xf32>

    // CHECK-DAG:   %[[FILTERS:.*]] = const.Declare tensor<16x3x3x3xf32> = dense<1.000000e+00> : tensor<16x3x3x3xf32>
    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Convolution(%arg0, %[[FILTERS]], %[[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return %[[VAL0]]
}

// CHECK-LABEL: @GroupsToAttr
func @GroupsToAttr(%arg0: tensor<1x16x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %filters = const.Declare tensor<16x1x1x3x3xf32> = dense<1.0> : tensor<16x1x1x3x3xf32>
    %0 = IE.GroupConvolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x16x300x300xf32>, tensor<16x1x1x3x3xf32> -> tensor<1x16x300x300xf32>

    return %0 : tensor<1x16x300x300xf32>

    // CHECK:       %[[FILTERS:.*]] = const.Declare tensor<16x1x3x3xf32> =
    // CHECK-SAM:       dense<1.000000e+00> : tensor<16x1x1x3x3xf32>, [#const.Reshape<[16, 1, 3, 3]>]
    // CHECK:       %[[VAL0:.*]] = IE.GroupConvolution(%arg0, %[[FILTERS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 16
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return %[[VAL0]]
}
