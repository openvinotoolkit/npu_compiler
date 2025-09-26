//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-convolution-input-shape %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @ReshapeInputFor1x1Conv
func.func @ReshapeInputFor1x1Conv(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x4096x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4096x1xf16>
    return %0 : tensor<1x320x4096x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 1024, 4]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x1024x4xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1024x4xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4096, 1]} : tensor<1x320x1024x4xf16> -> tensor<1x320x4096x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x4096x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeInputFor1x1ConvWithInputHeightNotDivisibleByFour
func.func @ReshapeInputFor1x1ConvWithInputHeightNotDivisibleByFour(%arg0: tensor<1x1280x77x1xf16>) -> tensor<1x320x77x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x77x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x77x1xf16>
    return %0 : tensor<1x320x77x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 11, 7]} : tensor<1x1280x77x1xf16> -> tensor<1x1280x11x7xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x11x7xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x11x7xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 77, 1]} : tensor<1x320x11x7xf16> -> tensor<1x320x77x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x77x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeInputFor1x1ConvWithInputHeightNeedExpand
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x256x151x1xf16, {order = #NHWC}>)
func.func @ReshapeInputFor1x1ConvWithInputHeightNeedExpand(%arg0: tensor<1x256x151x1xf16, {order = #NHWC}>)
        -> tensor<1x256x151x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<1.0>
        : tensor<256x256x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x256x151x1xf16, {order = #NHWC}>,
        tensor<256x256x1x1xf16, {order = #NHWC}>
            -> tensor<1x256x151x1xf16, {order = #NHWC}>
    return %conv : tensor<1x256x151x1xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1, 0]}
    // CHECK-SAME:      : tensor<1x256x151x1xf16, {order = #NHWC}> -> tensor<1x256x152x1xf16, {order = #NHWC}>
    // CHECK:       [[IN_RESHAPE:%.+]] = IE.AffineReshape([[EXPAND]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 256, 38, 4]}
    // CHECK-SAME:      -> tensor<1x256x38x4xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[IN_RESHAPE]], [[CST]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x256x38x4xf16, {order = #NHWC}>, tensor<256x256x1x1xf16, {order = #NHWC}> -> tensor<1x256x38x4xf16, {order = #NHWC}>
    // CHECK:       [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 256, 152, 1]}
    // CHECK-SAME:      : tensor<1x256x38x4xf16, {order = #NHWC}> -> tensor<1x256x152x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[OUT_RESHAPE]] [0, 0, 0, 0] [1, 256, 151, 1]
    // CHECK-SAME:      : tensor<1x256x152x1xf16, {order = #NHWC}> to tensor<1x256x151x1xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE]] : tensor<1x256x151x1xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @ReshapeInputFor1x1ConvWithInputHeightBePrimeNumbers
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x1280x4091x1xf16>)
func.func @ReshapeInputFor1x1ConvWithInputHeightBePrimeNumbers(%arg0: tensor<1x1280x4091x1xf16>) -> tensor<1x320x4091x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4091x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4091x1xf16>
    return %0 : tensor<1x320x4091x1xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<320x1280x1x1xf16>
    // CHECK:       [[CST_1:%.+]] = const.Declare tensor<1x320x1x1xf16>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1, 0]}
    // CHECK-SAME:      : tensor<1x1280x4091x1xf16> -> tensor<1x1280x4092x1xf16>
    // CHECK:       [[IN_RESHAPE:%.+]] = IE.AffineReshape([[EXPAND]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 1023, 4]}
    // CHECK-SAME:      : tensor<1x1280x4092x1xf16> -> tensor<1x1280x1023x4xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[IN_RESHAPE]], [[CST]], [[CST_1]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x1280x1023x4xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1023x4xf16>
    // CHECK:       [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4092, 1]}
    // CHECK-SAME:      : tensor<1x320x1023x4xf16> -> tensor<1x320x4092x1xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[OUT_RESHAPE]] [0, 0, 0, 0] [1, 320, 4091, 1]
    // CHECK-SAME:      : tensor<1x320x4092x1xf16> to tensor<1x320x4091x1xf16>
    // CHECK:       return [[SLICE]] : tensor<1x320x4091x1xf16>
}

// -----

// CHECK-LABEL: @NotReshapeInputFor1x1ConvMismatchedFilterShapeAlignment
func.func @NotReshapeInputFor1x1ConvMismatchedFilterShapeAlignment(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x4095x1xf16> {
    %filter = const.Declare tensor<320x1280x2x1xf16> = dense<1.000000e+00> : tensor<320x1280x2x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    return %0 : tensor<1x320x4095x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x2x1xf16> = dense<1.000000e+00> : tensor<320x1280x2x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x4095x1xf16>
}

// -----

// CHECK-LABEL: @NotReshapeInputForNon1x1Conv
func.func @NotReshapeInputForNon1x1Conv(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x2048x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    return %0 : tensor<1x320x2048x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x2048x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeInputFor1x1GroupConv
func.func @ReshapeInputFor1x1GroupConv(%arg0: tensor<1x320x4096x1xf16>) -> tensor<1x320x4096x1xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4096x1xf16>
    return %0 : tensor<1x320x4096x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 320, 1024, 4]} : tensor<1x320x4096x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x1024x4xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4096, 1]} : tensor<1x320x1024x4xf16> -> tensor<1x320x4096x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x4096x1xf16>
}

// -----

// CHECK-LABEL: @NotReshapeInputForNon1x1GroupConv
func.func @NotReshapeInputForNon1x1GroupConv(%arg0: tensor<1x320x4096x1xf16>) -> tensor<1x320x2048x1xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x320x4096x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    return %0 : tensor<1x320x2048x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x320x4096x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x2048x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeInputFor1x1GroupConvWithInputHeightBePrimeNumbers
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x320x4091x1xf16>)
func.func @ReshapeInputFor1x1GroupConvWithInputHeightBePrimeNumbers(%arg0: tensor<1x320x4091x1xf16>) -> tensor<1x320x4091x1xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4091x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4091x1xf16>
    return %0 : tensor<1x320x4091x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1, 0]}
    // CHECK-SAME:      : tensor<1x320x4091x1xf16> -> tensor<1x320x4092x1xf16>
    // CHECK:       [[IN_RESHAPE:%.+]] = IE.AffineReshape([[EXPAND]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 320, 1023, 4]}
    // CHECK-SAME:      : tensor<1x320x4092x1xf16> -> tensor<1x320x1023x4xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[IN_RESHAPE]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:     {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x320x1023x4xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1023x4xf16>
    // CHECK:       [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4092, 1]}
    // CHECK-SAME:      : tensor<1x320x1023x4xf16> -> tensor<1x320x4092x1xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[OUT_RESHAPE]] [0, 0, 0, 0] [1, 320, 4091, 1]
    // CHECK-SAME:      : tensor<1x320x4092x1xf16> to tensor<1x320x4091x1xf16>
    // CHECK:       return [[SLICE]] : tensor<1x320x4091x1xf16>
}

// -----

// CHECK-LABEL: @NotReshapeInputFor1x1GroupConvMismatchedFilterShapeAlignment
func.func @NotReshapeInputFor1x1GroupConvMismatchedFilterShapeAlignment(%arg0: tensor<1x320x4096x1xf16>) -> tensor<1x320x4095x1xf16> {
    %filter = const.Declare tensor<320x1x2x1xf16> = dense<1.000000e+00> : tensor<320x1x2x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<320x1x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    return %0 : tensor<1x320x4095x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x2x1xf16> = dense<1.000000e+00> : tensor<320x1x2x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<320x1x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x4095x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeSingleConstGroupConv
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1280x1x1xf16, {order = #NHWC}>
func.func @ReshapeSingleConstGroupConv(%arg0: tensor<1x1280x1x1xf16, {order = #NHWC}>) -> tensor<1x1280x1x1xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<1280x1x1x1xf16> = dense<1.000000e+00> : tensor<1280x1x1x1xf16>
    %bias = const.Declare tensor<1x1280x1x1xf16> = dense<1.000000e+00> : tensor<1x1280x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 1280 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1x1xf16, {order = #NHWC}>, tensor<1280x1x1x1xf16>, tensor<1x1280x1x1xf16> -> tensor<1x1280x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1280x1x1xf16, {order = #NHWC}>


    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<80x1x1x1xf16> = dense<1.000000e+00> : tensor<1280x1x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 1, 1]>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x80x1x1xf16> = dense<1.000000e+00> : tensor<1x1280x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 80, 1, 1]>]
    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 80, 4, 4]} inputs([[INPUT]] : tensor<1x1280x1x1xf16, {order = #NHWC}>) -> tensor<1x80x4x4xf16, {order = #NHWC}>
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 80 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<80x1x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 1280, 1, 1]} inputs([[GROUPCONV]] : tensor<1x80x4x4xf16, {order = #NHWC}>) -> tensor<1x1280x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[SHAPECAST_OUT]] : tensor<1x1280x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeSingleConstGroupConvPostOp
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x256x1x1xf16, {order = #NHWC}>
func.func @ReshapeSingleConstGroupConvPostOp(%arg0: tensor<1x256x1x1xf16, {order = #NHWC}>) -> tensor<1x256x1x1xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<256x1x1x1xf16> = dense<1> : tensor<256x1x1x1xui8>, [#const.CastElemType<f16>]
    %0 = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>, strides = [1, 1]} : tensor<1x256x1x1xf16, {order = #NHWC}>, tensor<256x1x1x1xf16> -> tensor<1x256x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x256x1x1xf16, {order = #NHWC}>


    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<1> : tensor<256x1x1x1xui8>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 4, 4]} inputs([[INPUT]] : tensor<1x256x1x1xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}>
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[FILTER]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>, strides = [1, 1]} : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<16x1x1x1xf16> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 256, 1, 1]} inputs([[GROUPCONV]] : tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x256x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[SHAPECAST_OUT]] : tensor<1x256x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeSingleConstGroupConvInvalidPostOp
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x256x1x1xf16, {order = #NHWC}>
func.func @NotReshapeSingleConstGroupConvInvalidPostOp(%arg0: tensor<1x256x1x1xf16, {order = #NHWC}>) -> tensor<1x256x1x1xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<256x1x1x1xf16> = dense<1> : tensor<256x1x1x1xui8>, [#const.CastElemType<f16>]
    %0 = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PRelu<negative_slope=[1.000000e-01, 2.000000e-01]>, strides = [1, 1]} : tensor<1x256x1x1xf16, {order = #NHWC}>, tensor<256x1x1x1xf16> -> tensor<1x256x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x256x1x1xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.ShapeCast

    // CHECK:       [[FILTER:%.+]] = const.Declare
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]])
    // CHECK:       return [[GROUPCONV]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.026685049019607842>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeSingleConstGroupConvQuantPerTensor
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x256x1x1x!qElemType, {order = #NHWC}>
func.func @ReshapeSingleConstGroupConvQuantPerTensor(%arg0: tensor<1x256x1x1x!qElemType, {order = #NHWC}>) -> tensor<1x256x1x1x!qElemType, {order = #NHWC}> {
    %filter = const.Declare tensor<256x1x1x1x!qElemType> = dense<1> : tensor<256x1x1x1xui8>, [#const.CastElemType<!qElemType>]
    %0 = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x1x1x!qElemType, {order = #NHWC}>, tensor<256x1x1x1x!qElemType> -> tensor<1x256x1x1x!qElemType, {order = #NHWC}>
    return %0 : tensor<1x256x1x1x!qElemType, {order = #NHWC}>


    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<16x1x1x1x!qElemType> = dense<1> : tensor<256x1x1x1xui8>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 1]>, #const.CastElemType<!qElemType>]
    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 4, 4]} inputs([[INPUT]] : tensor<1x256x1x1x!qElemType, {order = #NHWC}>) -> tensor<1x16x4x4x!qElemType, {order = #NHWC}>
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[FILTER]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x4x4x!qElemType, {order = #NHWC}>, tensor<16x1x1x1x!qElemType> -> tensor<1x16x4x4x!qElemType, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 256, 1, 1]} inputs([[GROUPCONV]] : tensor<1x16x4x4x!qElemType, {order = #NHWC}>) -> tensor<1x256x1x1x!qElemType, {order = #NHWC}>
    // CHECK:       return [[SHAPECAST_OUT]] : tensor<1x256x1x1x!qElemType, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeSingleConstGroupConvQuantPerChannel
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x256x1x1x!qElemType, {order = #NHWC}>
func.func @NotReshapeSingleConstGroupConvQuantPerChannel(%arg0: tensor<1x256x1x1x!qElemType, {order = #NHWC}>) -> tensor<1x256x1x1x!qElemType, {order = #NHWC}> {
    %filter = const.Declare tensor<256x1x1x1xf16> = dense<1.0> : tensor<256x1x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x1x1x!qElemType, {order = #NHWC}>, tensor<256x1x1x1xf16> -> tensor<1x256x1x1x!qElemType, {order = #NHWC}>
    return %0 : tensor<1x256x1x1x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   IE.ShapeCast

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<256x1x1x1xf16>
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]])
    // CHECK:       return [[GROUPCONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeSingleConstGroupConvForNCHWOut
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1280x1x1xf16, {order = #NHWC}>
func.func @NotReshapeSingleConstGroupConvForNCHWOut(%arg0: tensor<1x1280x1x1xf16, {order = #NHWC}>) -> tensor<1x1280x1x1xf16> {
    %filter = const.Declare tensor<1280x1x1x1xf16> = dense<1.000000e+00> : tensor<1280x1x1x1xf16>
    %bias = const.Declare tensor<1x1280x1x1xf16> = dense<1.000000e+00> : tensor<1x1280x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 1280 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1x1xf16, {order = #NHWC}>, tensor<1280x1x1x1xf16>, tensor<1x1280x1x1xf16> -> tensor<1x1280x1x1xf16>
    return %0 : tensor<1x1280x1x1xf16>

    // CHECK-DAG:   [[BIAS:%.+]]  = const.Declare
    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeForBiasIsNotConst
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1280x1x1xf16, {order = #NHWC}>
func.func @NotReshapeForBiasIsNotConst(%arg0: tensor<1x1280x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x1280x1x1xf16>) -> tensor<1x1280x1x1xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<1280x1x1x1xf16> = dense<1.000000e+00> : tensor<1280x1x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %arg1) {dilations = [1, 1], groups = 1280 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1x1xf16, {order = #NHWC}>, tensor<1280x1x1x1xf16>, tensor<1x1280x1x1xf16> -> tensor<1x1280x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1280x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeForInputHAndWIsNotOne
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1280x2x3xf16, {order = #NHWC}>
func.func @NotReshapeForInputHAndWIsNotOne(%arg0: tensor<1x1280x2x3xf16, {order = #NHWC}>) -> tensor<1x1280x2x3xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<1280x1x1x1xf16> = dense<1.000000e+00> : tensor<1280x1x1x1xf16>
    %bias = const.Declare tensor<1x1280x1x1xf16> = dense<1.000000e+00> : tensor<1x1280x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 1280 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x2x3xf16, {order = #NHWC}>, tensor<1280x1x1x1xf16>, tensor<1x1280x1x1xf16> -> tensor<1x1280x2x3xf16, {order = #NHWC}>
    return %0 : tensor<1x1280x2x3xf16, {order = #NHWC}>

    // CHECK-DAG:   [[BIAS:%.+]]  = const.Declare
    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeForRemainingChannelNotAlign
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x64x1x1xf16, {order = #NHWC}>
func.func @NotReshapeForRemainingChannelNotAlign(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x64x1x1xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<64x1x1x1xf16> = dense<1.000000e+00> : tensor<64x1x1x1xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16, {order = #NHWC}>, tensor<64x1x1x1xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x64x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[BIAS:%.+]]  = const.Declare
    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeForKernelIsNot1X1
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1280x4x1xf16, {order = #NHWC}>
func.func @NotReshapeForKernelIsNot1X1(%arg0: tensor<1x1280x4x1xf16, {order = #NHWC}>) -> tensor<1x1280x1x1xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<1280x1x4x1xf16> = dense<1.000000e+00> : tensor<1280x1x4x1xf16>
    %bias = const.Declare tensor<1x1280x1x1xf16> = dense<1.000000e+00> : tensor<1x1280x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 1280 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4x1xf16, {order = #NHWC}>, tensor<1280x1x4x1xf16>, tensor<1x1280x1x1xf16> -> tensor<1x1280x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1280x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[BIAS:%.+]]  = const.Declare
    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV]]
}

// -----

// CHECK: @ReshapeInputFor1x1ConvHeight1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1280x1x4096xf16>)
func.func @ReshapeInputFor1x1ConvHeight1(%arg0: tensor<1x1280x1x4096xf16>) -> tensor<1x320x1x4096xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1x4096xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1x4096xf16>
    return %0 : tensor<1x320x1x4096xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>

    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 1280, 1024, 4]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x1024x4xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x1280x1024x4xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1024x4xf16>

    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 320, 1, 4096]} : tensor<1x320x1024x4xf16> -> tensor<1x320x1x4096xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x1x4096xf16>
}

// -----

// CHECK: @ReshapeInputFor1x1GroupConvHeight1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x320x1x4096xf16>)
func.func @ReshapeInputFor1x1GroupConvHeight1(%arg0: tensor<1x320x1x4096xf16>) -> tensor<1x320x1x4096xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x1x4096xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1x4096xf16>
    return %0 : tensor<1x320x1x4096xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4, 1024]} : tensor<1x320x1x4096xf16> -> tensor<1x320x4x1024xf16>

    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[RESHAPE0]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:        {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:     : tensor<1x320x4x1024xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4x1024xf16>

    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 320, 1, 4096]} : tensor<1x320x4x1024xf16> -> tensor<1x320x1x4096xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x1x4096xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeInputForAddOpWithConstInputWidthEQOne
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x108864x2x1xf16, {order = #NHWC}>
func.func @ReshapeInputForAddOpWithConstInputWidthEQOne(%arg0: tensor<1x108864x2x1xf16, {order = #NHWC}>) -> tensor<1x108864x2x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x108864x2x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x2x1xf32>, [#const.Reshape<[1, 1, 2, 1]>, #const.CastElemType<f16>, #const.Broadcast<1 : i64, 108864 : i64>, #const.Reorder<#NHWC>]
    %0 = IE.Add(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x108864x2x1xf16, {order = #NHWC}>, tensor<1x108864x2x1xf16, {order = #NHWC}> -> tensor<1x108864x2x1xf16, {order = #NHWC}>
    return %0 : tensor<1x108864x2x1xf16, {order = #NHWC}>

    // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<1x3888x14x4xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x2x1xf32>, [#const.Reshape<[1, 1, 2, 1]>, #const.CastElemType<f16>, #const.Broadcast<1 : i64, 108864 : i64>, #const.Reorder<#NHWC>, #const.Reshape<[1, 3888, 14, 4]>]
    // CHECK:      [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 3888, 14, 4]}
    // CHECK:          inputs([[INPUT]] : tensor<1x108864x2x1xf16, {order = #NHWC}>) -> tensor<1x3888x14x4xf16, {order = #NHWC}>
    // CHECK:      [[ADD:%.+]] = IE.Add([[SHAPECAST_IN]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:          tensor<1x3888x14x4xf16, {order = #NHWC}>, tensor<1x3888x14x4xf16, {order = #NHWC}> -> tensor<1x3888x14x4xf16, {order = #NHWC}>
    // CHECK:      [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 108864, 2, 1]}
    // CHECK:          inputs([[ADD]] : tensor<1x3888x14x4xf16, {order = #NHWC}>) -> tensor<1x108864x2x1xf16, {order = #NHWC}>
    // CHECK:      return [[SHAPECAST_OUT]] : tensor<1x108864x2x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeInputForAddOpWithConstInputWidthEQOneAndSmallChannel
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x256x2x1xf16, {order = #NHWC}>
func.func @ReshapeInputForAddOpWithConstInputWidthEQOneAndSmallChannel(%arg0: tensor<1x256x2x1xf16, {order = #NHWC}>) -> tensor<1x256x2x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x256x2x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x2x1xf32>, [#const.Reshape<[1, 1, 2, 1]>, #const.CastElemType<f16>, #const.Broadcast<1 : i64, 256 : i64>, #const.Reorder<#NHWC>]
    %0 = IE.Add(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x1xf16, {order = #NHWC}>, tensor<1x256x2x1xf16, {order = #NHWC}> -> tensor<1x256x2x1xf16, {order = #NHWC}>
    return %0 : tensor<1x256x2x1xf16, {order = #NHWC}>

    // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<1x64x2x4xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x2x1xf32>, [#const.Reshape<[1, 1, 2, 1]>, #const.CastElemType<f16>, #const.Broadcast<1 : i64, 256 : i64>, #const.Reorder<#NHWC>, #const.Reshape<[1, 64, 2, 4]>]
    // CHECK:      [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 64, 2, 4]}
    // CHECK:          inputs([[INPUT]] : tensor<1x256x2x1xf16, {order = #NHWC}>) -> tensor<1x64x2x4xf16, {order = #NHWC}>
    // CHECK:      [[ADD:%.+]] = IE.Add([[SHAPECAST_IN]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:          tensor<1x64x2x4xf16, {order = #NHWC}>, tensor<1x64x2x4xf16, {order = #NHWC}> -> tensor<1x64x2x4xf16, {order = #NHWC}>
    // CHECK:      [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 256, 2, 1]}
    // CHECK:          inputs([[ADD]] : tensor<1x64x2x4xf16, {order = #NHWC}>) -> tensor<1x256x2x1xf16, {order = #NHWC}>
    // CHECK:      return [[SHAPECAST_OUT]] : tensor<1x256x2x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeInputForAddOpWithConstInputHWNotEQOne
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x12096x9x2xf16, {order = #NHWC}>
func.func @ReshapeInputForAddOpWithConstInputHWNotEQOne(%arg0: tensor<1x12096x9x2xf16, {order = #NHWC}>) -> tensor<1x12096x9x2xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x12096x9x2xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1344x9x9x2xf32>, [#const.Reshape<[1, 12096, 9, 2]>, #const.CastElemType<f16>, #const.LayoutCast<#NHWC>]
    %0 = IE.Add(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12096x9x2xf16, {order = #NHWC}>, tensor<1x12096x9x2xf16, {order = #NHWC}> -> tensor<1x12096x9x2xf16, {order = #NHWC}>
    return %0 : tensor<1x12096x9x2xf16, {order = #NHWC}>

    // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<1x6048x9x4xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1344x9x9x2xf32>, [#const.Reshape<[1, 12096, 9, 2]>, #const.CastElemType<f16>, #const.LayoutCast<#NHWC>, #const.Reshape<[1, 6048, 9, 4]>]
    // CHECK:      [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 6048, 9, 4]}
    // CHECK:          inputs([[INPUT]] : tensor<1x12096x9x2xf16, {order = #NHWC}>) -> tensor<1x6048x9x4xf16, {order = #NHWC}>
    // CHECK:      [[ADD:%.+]] = IE.Add([[SHAPECAST_IN]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:          tensor<1x6048x9x4xf16, {order = #NHWC}>, tensor<1x6048x9x4xf16, {order = #NHWC}> -> tensor<1x6048x9x4xf16, {order = #NHWC}>
    // CHECK:      [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 12096, 9, 2]}
    // CHECK:          inputs([[ADD]] : tensor<1x6048x9x4xf16, {order = #NHWC}>) -> tensor<1x12096x9x2xf16, {order = #NHWC}>
    // CHECK:      return [[SHAPECAST_OUT]] : tensor<1x12096x9x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeInputForAddOpWithConstInputHeightGreaterFour
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x256x7x1xf16, {order = #NHWC}>
func.func @NotReshapeInputForAddOpWithConstInputHeightGreaterFour(%arg0: tensor<1x256x7x1xf16, {order = #NHWC}>) -> tensor<1x256x7x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x256x7x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x7x1xf32>, [#const.Reshape<[1, 1, 7, 1]>, #const.CastElemType<f16>, #const.Broadcast<1 : i64, 256 : i64>, #const.Reorder<#NHWC>]
    %0 = IE.Add(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x7x1xf16, {order = #NHWC}>, tensor<1x256x7x1xf16, {order = #NHWC}> -> tensor<1x256x7x1xf16, {order = #NHWC}>
    return %0 : tensor<1x256x7x1xf16, {order = #NHWC}>

    // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<1x256x7x1xf16, {order = #NHWC}>
    // CHECK:      [[ADD:%.+]] = IE.Add([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:          tensor<1x256x7x1xf16, {order = #NHWC}>, tensor<1x256x7x1xf16, {order = #NHWC}> -> tensor<1x256x7x1xf16, {order = #NHWC}>
    // CHECK:      return [[ADD]] : tensor<1x256x7x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeInputForAddOpWithConstInputHWNotEQOneAndSmallChannel
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x256x2x3xf16, {order = #NHWC}>
func.func @NotReshapeInputForAddOpWithConstInputHWNotEQOneAndSmallChannel(%arg0: tensor<1x256x2x3xf16, {order = #NHWC}>) -> tensor<1x256x2x3xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x256x2x3xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x2x3x1xf32>, [#const.Reshape<[1, 1, 2, 3]>, #const.CastElemType<f16>, #const.Broadcast<1 : i64, 256 : i64>, #const.Reorder<#NHWC>]
    %0 = IE.Add(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x3xf16, {order = #NHWC}>, tensor<1x256x2x3xf16, {order = #NHWC}> -> tensor<1x256x2x3xf16, {order = #NHWC}>
    return %0 : tensor<1x256x2x3xf16, {order = #NHWC}>

    // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<1x256x2x3xf16, {order = #NHWC}>
    // CHECK:      [[ADD:%.+]] = IE.Add([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:          tensor<1x256x2x3xf16, {order = #NHWC}>, tensor<1x256x2x3xf16, {order = #NHWC}> -> tensor<1x256x2x3xf16, {order = #NHWC}>
    // CHECK:      return [[ADD]]  : tensor<1x256x2x3xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeInputForAddOpWithConstInputDueToSmallChannel
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x32x9x2xf16, {order = #NHWC}>
func.func @NotReshapeInputForAddOpWithConstInputDueToSmallChannel(%arg0: tensor<1x32x9x2xf16, {order = #NHWC}>) -> tensor<1x32x9x2xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x32x9x2xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<32x1x9x2xf32>, [#const.Reshape<[1, 32, 9, 2]>, #const.CastElemType<f16>, #const.LayoutCast<#NHWC>]
    %0 = IE.Add(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x9x2xf16, {order = #NHWC}>, tensor<1x32x9x2xf16, {order = #NHWC}> -> tensor<1x32x9x2xf16, {order = #NHWC}>
    return %0 : tensor<1x32x9x2xf16, {order = #NHWC}>

    // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<1x32x9x2xf16, {order = #NHWC}>
    // CHECK:      [[ADD:%.+]] = IE.Add([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:          tensor<1x32x9x2xf16, {order = #NHWC}>, tensor<1x32x9x2xf16, {order = #NHWC}> -> tensor<1x32x9x2xf16, {order = #NHWC}>
    // CHECK:      return [[ADD]] : tensor<1x32x9x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeInputForAddOpWithConstInputDueToBigChannelAndHW
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x108864x1024x1048xf16, {order = #NHWC}>
func.func @NotReshapeInputForAddOpWithConstInputDueToBigChannelAndHW(%arg0: tensor<1x108864x1024x1048xf16, {order = #NHWC}>)
          -> tensor<1x108864x1024x1048xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x108864x1024x1048xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x108864x1024x1048xf16>, [#const.LayoutCast<#NHWC>]
    %0 = IE.Add(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x108864x1024x1048xf16, {order = #NHWC}>, tensor<1x108864x1024x1048xf16, {order = #NHWC}> -> tensor<1x108864x1024x1048xf16, {order = #NHWC}>
    return %0 : tensor<1x108864x1024x1048xf16, {order = #NHWC}>

    // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<1x108864x1024x1048xf16, {order = #NHWC}>
    // CHECK:      [[ADD:%.+]] = IE.Add([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:          tensor<1x108864x1024x1048xf16, {order = #NHWC}>, tensor<1x108864x1024x1048xf16, {order = #NHWC}>
    // CHECK:          -> tensor<1x108864x1024x1048xf16, {order = #NHWC}>
    // CHECK:      return [[ADD]] : tensor<1x108864x1024x1048xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.026685049019607842>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeInputForAddOp
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x245760x1x1xf16, {order = #NHWC}>
func.func @ReshapeInputForAddOp(%arg0: tensor<1x245760x1x1xf16, {order = #NHWC}>) -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x245760x1x1xf16, {order = #NHWC}>, tensor<1x245760x1x1xf16, {order = #NHWC}> -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x245760x1x1x!qElemType, {order = #NHWC}>

    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 7680, 8, 4]} inputs([[INPUT]] : tensor<1x245760x1x1xf16, {order = #NHWC}>) -> tensor<1x7680x8x4xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPECAST_IN]], [[SHAPECAST_IN]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x7680x8x4xf16, {order = #NHWC}>, tensor<1x7680x8x4xf16, {order = #NHWC}> -> tensor<1x7680x8x4x!qElemType, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 245760, 1, 1]} inputs([[ADD]] : tensor<1x7680x8x4x!qElemType, {order = #NHWC}>) -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}>
    // CHECK:       return [[SHAPECAST_OUT]] : tensor<1x245760x1x1x!qElemType, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.026685049019607842>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeInputForAddPostOp
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x245760x1x1xf16, {order = #NHWC}>
func.func @ReshapeInputForAddPostOp(%arg0: tensor<1x245760x1x1xf16, {order = #NHWC}>) -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>} : tensor<1x245760x1x1xf16, {order = #NHWC}>, tensor<1x245760x1x1xf16, {order = #NHWC}> -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x245760x1x1x!qElemType, {order = #NHWC}>

    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 7680, 8, 4]} inputs([[INPUT]] : tensor<1x245760x1x1xf16, {order = #NHWC}>) -> tensor<1x7680x8x4xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPECAST_IN]], [[SHAPECAST_IN]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>} : tensor<1x7680x8x4xf16, {order = #NHWC}>, tensor<1x7680x8x4xf16, {order = #NHWC}> -> tensor<1x7680x8x4x!qElemType, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 245760, 1, 1]} inputs([[ADD]] : tensor<1x7680x8x4x!qElemType, {order = #NHWC}>) -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}>
    // CHECK:       return [[SHAPECAST_OUT]] : tensor<1x245760x1x1x!qElemType, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.026685049019607842>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeInputForAddInvalidPostOp
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x245760x1x1xf16, {order = #NHWC}>
func.func @NotReshapeInputForAddInvalidPostOp(%arg0: tensor<1x245760x1x1xf16, {order = #NHWC}>) -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, post_op = #IE.PRelu<negative_slope=[1.000000e-01, 2.000000e-01]>} : tensor<1x245760x1x1xf16, {order = #NHWC}>, tensor<1x245760x1x1xf16, {order = #NHWC}> -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x245760x1x1x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   IE.ShapeCast

    // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]])
    // CHECK:       return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeInputForAddOpQuantPerChannel
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x256x1x1xf16, {order = #NHWC}>
func.func @NotReshapeInputForAddOpQuantPerChannel(%arg0: tensor<1x256x1x1xf16, {order = #NHWC}>) -> tensor<1x256x1x1x!qElemType, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x256x1x1xf16, {order = #NHWC}>, tensor<1x256x1x1xf16, {order = #NHWC}> -> tensor<1x256x1x1x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x1x1x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   IE.ShapeCast

    // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]])
    // CHECK:       return [[ADD]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ShapeCastToAlignExpandedDWConv
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x3x640x640xf16, {order = #NHWC}>
func.func @ShapeCastToAlignExpandedDWConv(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x16x320x640xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x640x640xf16, {order  = #NHWC}> -> tensor<1x16x640x640xf16, {order = #NHWC}>
    %conv = IE.GroupConvolution(%expand, %filter, %bias) {
        dilations = [1, 1], groups = 16, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]
    } : tensor<1x16x640x640xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>

    return %conv : tensor<1x16x320x640xf16, {order = #NHWC}>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<48x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 48 : i64>, #const.Reshape<[48, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    // CHECK:        [[IN_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 48, 640, 40]} inputs([[INPUT]] : tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x48x640x40xf16, {order = #NHWC}>
    // CHECK:        [[GRP_CONV:%.+]] = IE.GroupConvolution([[IN_SHAPECAST]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 48 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x48x640x40xf16, {order = #NHWC}>, tensor<48x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x40xf16, {order = #NHWC}>
    // CHECK:        [[OUT_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 3, 320, 640]} inputs([[GRP_CONV]] : tensor<1x48x320x40xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:        [[EXPAND:%.+]] = IE.Expand([[OUT_SHAPECAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>
    // CHECK:        return [[EXPAND]] : tensor<1x16x320x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ShapeCastToAlignExpandedDWConvPostOp
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x3x640x640xf16, {order = #NHWC}>
func.func @ShapeCastToAlignExpandedDWConvPostOp(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x16x320x640xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x640x640xf16, {order  = #NHWC}> -> tensor<1x16x640x640xf16, {order = #NHWC}>
    %conv = IE.GroupConvolution(%expand, %filter, %bias) {
        dilations = [1, 1], groups = 16, pads_begin = [0, 0], pads_end = [0, 0],
        post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>, strides = [2, 1]
    } : tensor<1x16x640x640xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>

    return %conv : tensor<1x16x320x640xf16, {order = #NHWC}>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<48x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 48 : i64>, #const.Reshape<[48, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    // CHECK:        [[IN_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 48, 640, 40]} inputs([[INPUT]] : tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x48x640x40xf16, {order = #NHWC}>
    // CHECK:        [[GRP_CONV:%.+]] = IE.GroupConvolution([[IN_SHAPECAST]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 48 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>, strides = [2, 1]} : tensor<1x48x640x40xf16, {order = #NHWC}>, tensor<48x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x40xf16, {order = #NHWC}>
    // CHECK:        [[OUT_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 3, 320, 640]} inputs([[GRP_CONV]] : tensor<1x48x320x40xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:        [[EXPAND:%.+]] = IE.Expand([[OUT_SHAPECAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>
    // CHECK:        return [[EXPAND]] : tensor<1x16x320x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotShapeCastToAlignExpandedDWConvInvalidPostOp
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x3x640x640xf16, {order = #NHWC}>
func.func @NotShapeCastToAlignExpandedDWConvInvalidPostOp(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x16x320x640xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x640x640xf16, {order  = #NHWC}> -> tensor<1x16x640x640xf16, {order = #NHWC}>
    %conv = IE.GroupConvolution(%expand, %filter, %bias) {
        dilations = [1, 1], groups = 16, pads_begin = [0, 0], pads_end = [0, 0],
        post_op = #IE.PRelu<negative_slope=[1.000000e-01, 2.000000e-01]>, strides = [2, 1]
    } : tensor<1x16x640x640xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>

    return %conv : tensor<1x16x320x640xf16, {order = #NHWC}>

    // CHECK-NOT:    IE.ShapeCast

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.000000e+00>
    // CHECK:        [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK:        [[GRP_CONV:%.+]] = IE.GroupConvolution([[EXPAND]], [[FILTER]], [[BIAS]])
    // CHECK:        return [[GRP_CONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0039085829959196201>

// CHECK-LABEL: @ShapeCastToAlignExpandedDWConvQuantPerTensor
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x3x640x640x!qElemType, {order = #NHWC}>
func.func @ShapeCastToAlignExpandedDWConvQuantPerTensor(%arg0: tensor<1x3x640x640x!qElemType, {order = #NHWC}>) -> tensor<1x16x320x640xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x640x640x!qElemType, {order  = #NHWC}> -> tensor<1x16x640x640x!qElemType, {order = #NHWC}>
    %conv = IE.GroupConvolution(%expand, %filter, %bias) {
        dilations = [1, 1], groups = 16, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]
    } : tensor<1x16x640x640x!qElemType, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>

    return %conv : tensor<1x16x320x640xf16, {order = #NHWC}>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<48x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 48 : i64>, #const.Reshape<[48, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    // CHECK:        [[IN_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 48, 640, 40]} inputs([[INPUT]] : tensor<1x3x640x640x!qElemType, {order = #NHWC}>) -> tensor<1x48x640x40x!qElemType, {order = #NHWC}>
    // CHECK:        [[GRP_CONV:%.+]] = IE.GroupConvolution([[IN_SHAPECAST]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 48 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x48x640x40x!qElemType, {order = #NHWC}>, tensor<48x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x40xf16, {order = #NHWC}>
    // CHECK:        [[OUT_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 3, 320, 640]} inputs([[GRP_CONV]] : tensor<1x48x320x40xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:        [[EXPAND:%.+]] = IE.Expand([[OUT_SHAPECAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>
    // CHECK:        return [[EXPAND]] : tensor<1x16x320x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16:1, {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8}>

// CHECK-LABEL: @NotShapeCastToAlignExpandedDWConvQuantPerChannel
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x3x640x640xf16, {order = #NHWC}>
func.func @NotShapeCastToAlignExpandedDWConvQuantPerChannel(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x16x320x640x!qElemType, {order = #NHWC}> {
    %filter = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x640x640xf16, {order  = #NHWC}> -> tensor<1x16x640x640xf16, {order = #NHWC}>
    %conv = IE.GroupConvolution(%expand, %filter, %bias) {
        dilations = [1, 1], groups = 16, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]
    } : tensor<1x16x640x640xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x320x640x!qElemType, {order = #NHWC}>

    return %conv : tensor<1x16x320x640x!qElemType, {order = #NHWC}>

    // CHECK-NOT:    IE.ShapeCast

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:        [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK:        [[GRP_CONV:%.+]] = IE.GroupConvolution([[EXPAND]], [[FILTER]], [[BIAS]])
    // CHECK:        return [[GRP_CONV]]
}
