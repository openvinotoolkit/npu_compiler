//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --legalize-dilated-conv %s | FileCheck %s
// REQUIRES: platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @LegalizeDilatedConvolutionTwoDimension
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x30x30xf16>)
func.func @LegalizeDilatedConvolutionTwoDimension(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x32x18x24xf16> {
    %filter = const.Declare tensor<32x3x3x3xf16> = dense<1.0> : tensor<32x3x3x3xf16>
    %bias = const.Declare tensor<1x32x1x1xf16> = dense<1.0> : tensor<1x32x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {
        clamp = {min = 0.000000e+00 : f64, max = 1.000000e+00 : f64},
        dilations = [8, 6], pads_begin = [1, 2], pads_end = [3, 4], strides = [1, 1]
    } : tensor<1x3x30x30xf16>, tensor<32x3x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x18x24xf16>
    return %0 : tensor<1x32x18x24xf16>

    // CHECK-DAG: [[FILTERS:%.+]] = const.Declare tensor<32x3x3x3xf16> = dense<1.000000e+00> : tensor<32x3x3x3xf16>
    // CHECK-DAG: [[BIAS:%.+]] = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>

    // CHECK: [[FILTERS_SLICE0:%.+]] = IE.Slice [[FILTERS]] [0, 0, 0, 0] [32, 3, 2, 3] : tensor<32x3x3x3xf16> to tensor<32x3x2x3xf16>
    // CHECK: [[EXPAND_DILATED0:%.+]] = IE.ExpandDilated([[FILTERS_SLICE0]]) {dilations = [8, 6]} : tensor<32x3x2x3xf16> -> tensor<32x3x9x13xf16>

    // CHECK: [[FILTERS_SLICE1:%.+]] = IE.Slice [[FILTERS]] [0, 0, 2, 0] [32, 3, 1, 3] : tensor<32x3x3x3xf16> to tensor<32x3x1x3xf16>
    // CHECK: [[EXPAND_DILATED1:%.+]] = IE.ExpandDilated([[FILTERS_SLICE1]]) {dilations = [8, 6]} : tensor<32x3x1x3xf16> -> tensor<32x3x1x13xf16>

    // CHECK-DAG: [[BIAS_1:%.+]] = const.Declare tensor<1x32x1x1xf16> = dense<0.000000e+00> : tensor<1x32x1x1xf16>

    // CHECK: [[ACT_SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 25, 30] : tensor<1x3x30x30xf16> to tensor<1x3x25x30xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[ACT_SLICE0]], [[EXPAND_DILATED0]], [[BIAS]]) {dilations = [1, 1], pads_begin = [1, 2], pads_end = [0, 4], strides = [1, 1]} : tensor<1x3x25x30xf16>, tensor<32x3x9x13xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x18x24xf16>

    // CHECK: [[ACT_SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 15, 0] [1, 3, 15, 30] : tensor<1x3x30x30xf16> to tensor<1x3x15x30xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[ACT_SLICE1]], [[EXPAND_DILATED1]], [[BIAS_1]]) {dilations = [1, 1], pads_begin = [0, 2], pads_end = [3, 4], strides = [1, 1]} : tensor<1x3x15x30xf16>, tensor<32x3x1x13xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x18x24xf16>

    // CHECK: [[ADD0:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}} : tensor<1x32x18x24xf16>, tensor<1x32x18x24xf16> -> tensor<1x32x18x24xf16>

    // CHECK: return [[ADD0]]
}
