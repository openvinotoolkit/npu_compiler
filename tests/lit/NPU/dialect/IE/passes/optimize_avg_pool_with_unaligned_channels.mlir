//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-avg-pool-with-unaligned-channels %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertAvgPoolToConv
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x3x640x640xf16, {order = #NHWC}>
func.func @ConvertAvgPoolToConv(%arg0 : tensor<1x3x640x640xf16, {order = #NHWC}>) -> (tensor<1x3x320x320xf16, {order = #NHWC}>) {
    %pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x3x640x640xf16, {order = #NHWC}> -> tensor<1x3x320x320xf16, {order = #NHWC}>

    return %pool : tensor<1x3x320x320xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x2x2xf16, {order = #NHWC}> = dense<[
    // CHECK-SAME{LITERAL}:   [[[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]],
    // CHECK-SAME{LITERAL}:   [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]],
    // CHECK-SAME{LITERAL}:   [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]]]
    // CHECK-SAME:          ]> : tensor<3x3x2x2xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
    // CHECK-SAME:          dilations = [1, 1],
    // CHECK-SAME:          pads_begin = [0, 0],
    // CHECK-SAME:          pads_end = [0, 0],
    // CHECK-SAME:          strides = [2, 2]
    // CHECK-SAME:          } : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<3x3x2x2xf16, {order = #NHWC}> -> tensor<1x3x320x320xf16, {order = #NHWC}>

    // CHECK: return [[CONV]] : tensor<1x3x320x320xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertAvgPoolWithStaticScaleToConv
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x3x640x640xf16, {order = #NHWC}>
func.func @ConvertAvgPoolWithStaticScaleToConv(%arg0 : tensor<1x3x640x640xf16, {order = #NHWC}>) -> (tensor<1x3x320x320xf16, {order = #NHWC}>) {
    %pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        static_scale = 0.135327876 : f32,
        strides = [2, 2]
    } : tensor<1x3x640x640xf16, {order = #NHWC}> -> tensor<1x3x320x320xf16, {order = #NHWC}>

    return %pool : tensor<1x3x320x320xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x2x2xf16, {order = #NHWC}> = dense<[
    // CHECK-SAME{LITERAL}:   [[[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]],
    // CHECK-SAME{LITERAL}:   [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]],
    // CHECK-SAME{LITERAL}:   [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[2.500000e-01, 2.500000e-01], [2.500000e-01, 2.500000e-01]]]
    // CHECK-SAME:          ]> : tensor<3x3x2x2xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
    // CHECK-SAME:          dilations = [1, 1],
    // CHECK-SAME:          pads_begin = [0, 0],
    // CHECK-SAME:          pads_end = [0, 0],
    // CHECK-SAME:          static_scale = 0.135327876 : f32,
    // CHECK-SAME:          strides = [2, 2]
    // CHECK-SAME:          } : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<3x3x2x2xf16, {order = #NHWC}> -> tensor<1x3x320x320xf16, {order = #NHWC}>

    // CHECK: return [[CONV]] : tensor<1x3x320x320xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotConvertAvgPoolWithAlignedChannelNums
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x640x640xf16, {order = #NHWC}>
func.func @NotConvertAvgPoolWithAlignedChannelNums(%arg0 : tensor<1x16x640x640xf16, {order = #NHWC}>) -> (tensor<1x16x320x320xf16, {order = #NHWC}>) {
    %pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x16x640x640xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>

    return %pool : tensor<1x16x320x320xf16, {order = #NHWC}>
    // CHECK:       [[POOL:%.+]] = IE.AvgPool([[INPUT]])
    // CHECK-SAME{LITERAL}:   {exclude_pads, kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x16x640x640xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>

    // CHECK: return [[POOL]] : tensor<1x16x320x320xf16, {order = #NHWC}>
}
