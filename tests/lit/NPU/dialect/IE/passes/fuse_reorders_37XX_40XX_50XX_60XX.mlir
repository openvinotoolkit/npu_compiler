//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-reorders %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToTransposedConv
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x2x256x144xf16, {order = #NHWC}>
func.func @FusePermuteToTransposedConv(%arg0: tensor<1x2x256x144xf16, {order = #NHWC}>) -> tensor<1x2x512x288xf16> {
    %cst = const.Declare tensor<2x2x4x4xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<2x2x4x4xf16>, [#const.Reorder<#NHWC>]

    %TransposedConv = IE.TransposedConvolution(%arg0, %cst) {
        dilations = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        spatial_output_padding = [0, 0],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [2, 2]
    } : tensor<1x2x256x144xf16, {order = #NHWC}>,
        tensor<2x2x4x4xf16, {order = #NHWC}> -> tensor<1x2x512x288xf16, {order = #NHWC}>

    %Reorder = IE.Reorder(%TransposedConv) {
        dstOrder = #NCHW
    } : tensor<1x2x512x288xf16, {order = #NHWC}> -> tensor<1x2x512x288xf16>

    return %Reorder : tensor<1x2x512x288xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<2x2x4x4xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<2x2x4x4xf16>, [#const.Reorder<#NHWC>]

    // CHECK:   [[TransposedConv:%.+]] = IE.TransposedConvolution([[ARG_0]], [[CST]]) {
    // CHECK-SAME:  dilations = [1, 1],
    // CHECK-SAME:  operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:  pads_begin = [1, 1],
    // CHECK-SAME:  pads_end = [1, 1],
    // CHECK-SAME:  spatial_output_padding = [0, 0],
    // CHECK-SAME:  strides = [2, 2]
    // CHECK-SAME: } : tensor<1x2x256x144xf16, {order = #NHWC}>,
    // CHECK-SAME: tensor<2x2x4x4xf16, {order = #NHWC}> -> tensor<1x2x512x288xf16>

    // CHECK:   return [[TransposedConv]] : tensor<1x2x512x288xf16>
}
