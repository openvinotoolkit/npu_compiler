//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW enable-se-ptrs-operations=true" --expand-activation-channels --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TransposedConvolution
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x20x23x30xf16, {order = #NHWC}>)
func.func @TransposedConvolution(%input: tensor<1x20x23x30xf16, {order = #NHWC}>) -> tensor<1x20x46x60xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<20x20x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<20x20x2x2xf16, {order = #NHWC}>
    %output = IE.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x20x23x30xf16, {order = #NHWC}>, tensor<20x20x2x2xf16, {order = #NHWC}> -> tensor<1x20x46x60xf16, {order = #NHWC}>
    return %output : tensor<1x20x46x60xf16, {order = #NHWC}>


    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<32x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<20x20x2x2xf16, {order = #NHWC}>, [#const.PadWithZero<[0, 0, 0, 0], [12, 12, 0, 0]>]
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
    // CHECK-SAME:      : tensor<1x20x23x30xf16, {order = #NHWC}> -> tensor<1x32x23x30xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT:%.+]] = IE.TransposedConvolution([[EXPAND]], [[WEIGHTS]]) {
    // CHECK-SAME:          dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
    // CHECK-SAME:      } : tensor<1x32x23x30xf16, {order = #NHWC}>, tensor<32x32x2x2xf16, {order = #NHWC}> -> tensor<1x32x46x60xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_SLICE:%.+]] = IE.Slice [[OUTPUT]] [0, 0, 0, 0] [1, 20, 46, 60]
    // CHECK-SAME:      : tensor<1x32x46x60xf16, {order = #NHWC}> to tensor<1x20x46x60xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT_SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TransposedConvolutionWithBias
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x20x23x30xf16, {order = #NHWC}>)
func.func @TransposedConvolutionWithBias(%input: tensor<1x20x23x30xf16, {order = #NHWC}>) -> tensor<1x20x46x60xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<20x20x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<20x20x2x2xf16, {order = #NHWC}>
    %bias = const.Declare tensor<1x20x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x20x1x1xf16, {order = #NHWC}>
    %output = IE.TransposedConvolution(%input, %weights, %bias) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x20x23x30xf16, {order = #NHWC}>, tensor<20x20x2x2xf16, {order = #NHWC}>, tensor<1x20x1x1xf16, {order = #NHWC}> -> tensor<1x20x46x60xf16, {order = #NHWC}>
    return %output : tensor<1x20x46x60xf16, {order = #NHWC}>


    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<32x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<20x20x2x2xf16, {order = #NHWC}>, [#const.PadWithZero<[0, 0, 0, 0], [12, 12, 0, 0]>]
    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x20x1x1xf16, {order = #NHWC}>, [#const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
    // CHECK:           [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
    // CHECK-SAME:          : tensor<1x20x23x30xf16, {order = #NHWC}> -> tensor<1x32x23x30xf16, {order = #NHWC}>
    // CHECK:           [[OUTPUT:%.+]] = IE.TransposedConvolution([[EXPAND]], [[WEIGHTS]], [[BIAS]]) {
    // CHECK-SAME:              dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
    // CHECK-SAME:          } : tensor<1x32x23x30xf16, {order = #NHWC}>, tensor<32x32x2x2xf16, {order = #NHWC}>, tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x46x60xf16, {order = #NHWC}>
    // CHECK:           [[OUTPUT_SLICE:%.+]] = IE.Slice [[OUTPUT]] [0, 0, 0, 0] [1, 20, 46, 60]
    // CHECK-SAME:          : tensor<1x32x46x60xf16, {order = #NHWC}> to tensor<1x20x46x60xf16, {order = #NHWC}>
    // CHECK:           return [[OUTPUT_SLICE]]
}
