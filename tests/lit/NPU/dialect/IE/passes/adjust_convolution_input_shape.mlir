//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --adjust-convolution-input-shape %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

// CHECK-LABEL: @ReshapeInputFor1x1Conv
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1280x4096x1xf16>
func.func @ReshapeInputFor1x1Conv(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x4096x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4096x1xf16>
    return %0 : tensor<1x320x4096x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 1024, 4]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x1024x4xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1024x4xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4096, 1]} : tensor<1x320x1024x4xf16> -> tensor<1x320x4096x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x4096x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeInputFor1x1ConvWithInputHeightNotDivisibleByFour
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1280x77x1xf16>
func.func @ReshapeInputFor1x1ConvWithInputHeightNotDivisibleByFour(%arg0: tensor<1x1280x77x1xf16>) -> tensor<1x320x77x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x77x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x77x1xf16>
    return %0 : tensor<1x320x77x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG_0]])
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
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1280x4096x1xf16>
func.func @NotReshapeInputFor1x1ConvMismatchedFilterShapeAlignment(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x4095x1xf16> {
    %filter = const.Declare tensor<320x1280x2x1xf16> = dense<1.000000e+00> : tensor<320x1280x2x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    return %0 : tensor<1x320x4095x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x2x1xf16> = dense<1.000000e+00> : tensor<320x1280x2x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x4095x1xf16>
}

// -----

// CHECK-LABEL: @NotReshapeInputForNon1x1Conv
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1280x4096x1xf16>
func.func @NotReshapeInputForNon1x1Conv(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x2048x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    return %0 : tensor<1x320x2048x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x2048x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeInputFor1x1GroupConv
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x320x4096x1xf16>
func.func @ReshapeInputFor1x1GroupConv(%arg0: tensor<1x320x4096x1xf16>) -> tensor<1x320x4096x1xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4096x1xf16>
    return %0 : tensor<1x320x4096x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 320, 1024, 4]} : tensor<1x320x4096x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x1024x4xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4096, 1]} : tensor<1x320x1024x4xf16> -> tensor<1x320x4096x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x4096x1xf16>
}

// -----

// CHECK-LABEL: @NotReshapeInputForNon1x1GroupConv
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x320x4096x1xf16>
func.func @NotReshapeInputForNon1x1GroupConv(%arg0: tensor<1x320x4096x1xf16>) -> tensor<1x320x2048x1xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x320x4096x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    return %0 : tensor<1x320x2048x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[ARG_0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x320x4096x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
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
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x320x4096x1xf16>
func.func @NotReshapeInputFor1x1GroupConvMismatchedFilterShapeAlignment(%arg0: tensor<1x320x4096x1xf16>) -> tensor<1x320x4095x1xf16> {
    %filter = const.Declare tensor<320x1x2x1xf16> = dense<1.000000e+00> : tensor<320x1x2x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<320x1x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    return %0 : tensor<1x320x4095x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x2x1xf16> = dense<1.000000e+00> : tensor<320x1x2x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[ARG_0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<320x1x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
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

    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 80, 4, 4]} inputs([[INPUT]] : tensor<1x1280x1x1xf16, {order = #NHWC}>) -> tensor<1x80x4x4xf16, {order = #NHWC}>
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<80x1x1x1xf16> = dense<1.000000e+00> : tensor<1280x1x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 1, 1]>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x80x1x1xf16> = dense<1.000000e+00> : tensor<1x1280x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 80, 1, 1]>]
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

    // CHECK:   [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 4, 4]} inputs([[INPUT]] : tensor<1x256x1x1xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}>
    // CHECK:   [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<1> : tensor<256x1x1x1xui8>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:   [[GROUPCONV:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[FILTER]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>, strides = [1, 1]} : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<16x1x1x1xf16> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 256, 1, 1]} inputs([[GROUPCONV]] : tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x256x1x1xf16, {order = #NHWC}>
    // CHECK:   return [[SHAPECAST_OUT]] : tensor<1x256x1x1xf16, {order = #NHWC}>
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

    // CHECK:   [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 4, 4]} inputs([[INPUT]] : tensor<1x256x1x1x!qElemType, {order = #NHWC}>) -> tensor<1x16x4x4x!qElemType, {order = #NHWC}>
    // CHECK:   [[FILTER:%.+]] = const.Declare tensor<16x1x1x1x!qElemType> = dense<1> : tensor<256x1x1x1xui8>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 1]>, #const.CastElemType<!qElemType>]
    // CHECK:   [[GROUPCONV:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[FILTER]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x4x4x!qElemType, {order = #NHWC}>, tensor<16x1x1x1x!qElemType> -> tensor<1x16x4x4x!qElemType, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 256, 1, 1]} inputs([[GROUPCONV]] : tensor<1x16x4x4x!qElemType, {order = #NHWC}>) -> tensor<1x256x1x1x!qElemType, {order = #NHWC}>
    // CHECK:   return [[SHAPECAST_OUT]] : tensor<1x256x1x1x!qElemType, {order = #NHWC}>
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

    // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<1x108864x2x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x2x1xf32>, [#const.Reshape<[1, 1, 2, 1]>, #const.CastElemType<f16>, #const.Broadcast<1 : i64, 108864 : i64>, #const.Reorder<#NHWC>]
    // CHECK:      [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 3888, 14, 4]}
    // CHECK:          inputs([[INPUT]] : tensor<1x108864x2x1xf16, {order = #NHWC}>) -> tensor<1x3888x14x4xf16, {order = #NHWC}>
    // CHECK:      [[SHAPECAST_IN2:%.+]] = IE.ShapeCast {shape = [1, 3888, 14, 4]}
    // CHECK:          inputs([[CST]] : tensor<1x108864x2x1xf16, {order = #NHWC}>) -> tensor<1x3888x14x4xf16, {order = #NHWC}>
    // CHECK:      [[ADD:%.+]] = IE.Add([[SHAPECAST_IN]], [[SHAPECAST_IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
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

    // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<1x256x2x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x2x1xf32>, [#const.Reshape<[1, 1, 2, 1]>, #const.CastElemType<f16>, #const.Broadcast<1 : i64, 256 : i64>, #const.Reorder<#NHWC>]
    // CHECK:      [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 64, 2, 4]}
    // CHECK:          inputs([[INPUT]] : tensor<1x256x2x1xf16, {order = #NHWC}>) -> tensor<1x64x2x4xf16, {order = #NHWC}>
    // CHECK:      [[SHAPECAST_IN2:%.+]] = IE.ShapeCast {shape = [1, 64, 2, 4]}
    // CHECK:          inputs([[CST]] : tensor<1x256x2x1xf16, {order = #NHWC}>) -> tensor<1x64x2x4xf16, {order = #NHWC}>
    // CHECK:      [[ADD:%.+]] = IE.Add([[SHAPECAST_IN]], [[SHAPECAST_IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
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

    // CHECK:  [[CST:%.+]] = const.Declare tensor<1x12096x9x2xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1344x9x9x2xf32>, [#const.Reshape<[1, 12096, 9, 2]>, #const.CastElemType<f16>, #const.LayoutCast<#NHWC>]
    // CHECK:      [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 6048, 9, 4]}
    // CHECK:          inputs([[INPUT]] : tensor<1x12096x9x2xf16, {order = #NHWC}>) -> tensor<1x6048x9x4xf16, {order = #NHWC}>
    // CHECK:      [[SHAPECAST_IN2:%.+]] = IE.ShapeCast {shape = [1, 6048, 9, 4]}
    // CHECK:          inputs([[CST]] : tensor<1x12096x9x2xf16, {order = #NHWC}>) -> tensor<1x6048x9x4xf16, {order = #NHWC}>
    // CHECK:      [[ADD:%.+]] = IE.Add([[SHAPECAST_IN]], [[SHAPECAST_IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
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

    // CHECK:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    // CHECK:   [[IN_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 48, 640, 40]} inputs([[INPUT]] : tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x48x640x40xf16, {order = #NHWC}>
    // CHECK:   [[FILTER:%.+]] = const.Declare tensor<48x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 48 : i64>, #const.Reshape<[48, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK:   [[GRP_CONV:%.+]] = IE.GroupConvolution([[IN_SHAPECAST]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 48 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x48x640x40xf16, {order = #NHWC}>, tensor<48x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x40xf16, {order = #NHWC}>
    // CHECK:   [[OUT_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 3, 320, 640]} inputs([[GRP_CONV]] : tensor<1x48x320x40xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[OUT_SHAPECAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>
    // CHECK:   return [[EXPAND]] : tensor<1x16x320x640xf16, {order = #NHWC}>
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

    // CHECK:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    // CHECK:   [[IN_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 48, 640, 40]} inputs([[INPUT]] : tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x48x640x40xf16, {order = #NHWC}>
    // CHECK:   [[FILTER:%.+]] = const.Declare tensor<48x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 48 : i64>, #const.Reshape<[48, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK:   [[GRP_CONV:%.+]] = IE.GroupConvolution([[IN_SHAPECAST]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 48 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>, strides = [2, 1]} : tensor<1x48x640x40xf16, {order = #NHWC}>, tensor<48x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x40xf16, {order = #NHWC}>
    // CHECK:   [[OUT_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 3, 320, 640]} inputs([[GRP_CONV]] : tensor<1x48x320x40xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[OUT_SHAPECAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>
    // CHECK:   return [[EXPAND]] : tensor<1x16x320x640xf16, {order = #NHWC}>
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

    // CHECK:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    // CHECK:   [[IN_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 48, 640, 40]} inputs([[INPUT]] : tensor<1x3x640x640x!qElemType, {order = #NHWC}>) -> tensor<1x48x640x40x!qElemType, {order = #NHWC}>
    // CHECK:   [[FILTER:%.+]] = const.Declare tensor<48x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 48 : i64>, #const.Reshape<[48, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK:   [[GRP_CONV:%.+]] = IE.GroupConvolution([[IN_SHAPECAST]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 48 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x48x640x40x!qElemType, {order = #NHWC}>, tensor<48x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x40xf16, {order = #NHWC}>
    // CHECK:   [[OUT_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 3, 320, 640]} inputs([[GRP_CONV]] : tensor<1x48x320x40xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[OUT_SHAPECAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>
    // CHECK:   return [[EXPAND]] : tensor<1x16x320x640xf16, {order = #NHWC}>
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

// -----

// CHECK-LABEL: @ReshapeInputFor1x1ConvWithLargeHeight
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x96x65536x1xf16>
func.func @ReshapeInputFor1x1ConvWithLargeHeight(%arg0: tensor<1x96x65536x1xf16>) -> tensor<1x96x65536x1xf16> {
    %filter = const.Declare tensor<96x96x1x1xf16> = dense<1.000000e+00> : tensor<96x96x1x1xf16>
    %bias = const.Declare tensor<1x96x1x1xf16> = dense<1.000000e+00> : tensor<1x96x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x65536x1xf16>, tensor<96x96x1x1xf16>, tensor<1x96x1x1xf16> -> tensor<1x96x65536x1xf16>
    return %0 : tensor<1x96x65536x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<96x96x1x1xf16> = dense<1.000000e+00> : tensor<96x96x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x96x1x1xf16> = dense<1.000000e+00> : tensor<1x96x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 96, 8192, 8]} : tensor<1x96x65536x1xf16> -> tensor<1x96x8192x8xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x8192x8xf16>, tensor<96x96x1x1xf16>, tensor<1x96x1x1xf16> -> tensor<1x96x8192x8xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 96, 65536, 1]} : tensor<1x96x8192x8xf16> -> tensor<1x96x65536x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x96x65536x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeInputFor1x1ConvWithLargeHeightAndExpand
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x96x65521x1xf16>
func.func @ReshapeInputFor1x1ConvWithLargeHeightAndExpand(%arg0: tensor<1x96x65521x1xf16>) -> tensor<1x96x65521x1xf16> {
    %filter = const.Declare tensor<96x96x1x1xf16> = dense<1.000000e+00> : tensor<96x96x1x1xf16>
    %bias = const.Declare tensor<1x96x1x1xf16> = dense<1.000000e+00> : tensor<1x96x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x65521x1xf16>, tensor<96x96x1x1xf16>, tensor<1x96x1x1xf16> -> tensor<1x96x65521x1xf16>
    return %0 : tensor<1x96x65521x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<96x96x1x1xf16> = dense<1.000000e+00> : tensor<96x96x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x96x1x1xf16> = dense<1.000000e+00> : tensor<1x96x1x1xf16>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 7, 0]} : tensor<1x96x65521x1xf16> -> tensor<1x96x65528x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[EXPAND]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 96, 8191, 8]} : tensor<1x96x65528x1xf16> -> tensor<1x96x8191x8xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x8191x8xf16>, tensor<96x96x1x1xf16>, tensor<1x96x1x1xf16> -> tensor<1x96x8191x8xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 96, 65528, 1]} : tensor<1x96x8191x8xf16> -> tensor<1x96x65528x1xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[RESHAPE1]] [0, 0, 0, 0] [1, 96, 65521, 1] : tensor<1x96x65528x1xf16> to tensor<1x96x65521x1xf16>
    // CHECK:       return [[SLICE]] : tensor<1x96x65521x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeInputFor1x1ConvWithLargeHeightWithSqrt
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x96x65535x1xf16>
func.func @ReshapeInputFor1x1ConvWithLargeHeightWithSqrt(%arg0: tensor<1x96x65535x1xf16>) -> tensor<1x96x65535x1xf16> {
    %filter = const.Declare tensor<96x96x1x1xf16> = dense<1.000000e+00> : tensor<96x96x1x1xf16>
    %bias = const.Declare tensor<1x96x1x1xf16> = dense<1.000000e+00> : tensor<1x96x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x65535x1xf16>, tensor<96x96x1x1xf16>, tensor<1x96x1x1xf16> -> tensor<1x96x65535x1xf16>
    return %0 : tensor<1x96x65535x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<96x96x1x1xf16> = dense<1.000000e+00> : tensor<96x96x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x96x1x1xf16> = dense<1.000000e+00> : tensor<1x96x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 96, 257, 255]} : tensor<1x96x65535x1xf16> -> tensor<1x96x257x255xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x257x255xf16>, tensor<96x96x1x1xf16>, tensor<1x96x1x1xf16> -> tensor<1x96x257x255xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 96, 65535, 1]} : tensor<1x96x257x255xf16> -> tensor<1x96x65535x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x96x65535x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeInputFor1x1ConvWithLargeWidth
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x96x1x65536xf16>
func.func @ReshapeInputFor1x1ConvWithLargeWidth(%arg0: tensor<1x96x1x65536xf16>) -> tensor<1x96x1x65536xf16> {
    %filter = const.Declare tensor<96x96x1x1xf16> = dense<1.000000e+00> : tensor<96x96x1x1xf16>
    %bias = const.Declare tensor<1x96x1x1xf16> = dense<1.000000e+00> : tensor<1x96x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x1x65536xf16>, tensor<96x96x1x1xf16>, tensor<1x96x1x1xf16> -> tensor<1x96x1x65536xf16>
    return %0 : tensor<1x96x1x65536xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<96x96x1x1xf16> = dense<1.000000e+00> : tensor<96x96x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x96x1x1xf16> = dense<1.000000e+00> : tensor<1x96x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 96, 8192, 8]} : tensor<1x96x1x65536xf16> -> tensor<1x96x8192x8xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x8192x8xf16>, tensor<96x96x1x1xf16>, tensor<1x96x1x1xf16> -> tensor<1x96x8192x8xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 96, 1, 65536]} : tensor<1x96x8192x8xf16> -> tensor<1x96x1x65536xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x96x1x65536xf16>
}

// -----

// CHECK-LABEL: @ReshapeInputFor1x1ConvWithHugeHeight
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x96x268435456x1xf16>
func.func @ReshapeInputFor1x1ConvWithHugeHeight(%arg0: tensor<1x96x268435456x1xf16>) -> tensor<1x96x268435456x1xf16> {
    %filter = const.Declare tensor<96x96x1x1xf16> = dense<1.000000e+00> : tensor<96x96x1x1xf16>
    %bias = const.Declare tensor<1x96x1x1xf16> = dense<1.000000e+00> : tensor<1x96x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x268435456x1xf16>, tensor<96x96x1x1xf16>, tensor<1x96x1x1xf16> -> tensor<1x96x268435456x1xf16>
    return %0 : tensor<1x96x268435456x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<96x96x1x1xf16> = dense<1.000000e+00> : tensor<96x96x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x96x1x1xf16> = dense<1.000000e+00> : tensor<1x96x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 96, 32768, 8192]} : tensor<1x96x268435456x1xf16> -> tensor<1x96x32768x8192xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x32768x8192xf16>, tensor<96x96x1x1xf16>, tensor<1x96x1x1xf16> -> tensor<1x96x32768x8192xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 96, 268435456, 1]} : tensor<1x96x32768x8192xf16> -> tensor<1x96x268435456x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x96x268435456x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeInputFor1x1ConvWithShapeCastInput
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1280x16x197xf16>
func.func @ReshapeInputFor1x1ConvWithShapeCastInput(%arg0: tensor<1x1280x16x197xf16>) -> tensor<1x320x3152x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %softmax = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x1280x16x197xf16> -> tensor<1x1280x16x197xf16>
    %shapecast = IE.ShapeCast {shape = [1, 1280, 3152, 1]} inputs(%softmax: tensor<1x1280x16x197xf16>) -> tensor<1x1280x3152x1xf16>
    %0 = IE.Convolution(%shapecast, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x3152x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x3152x1xf16>
    return %0 : tensor<1x320x3152x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 3 : i64} : tensor<1x1280x16x197xf16> -> tensor<1x1280x16x197xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[SOFTMAX]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x16x197xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x16x197xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 3152, 1]} : tensor<1x320x16x197xf16> -> tensor<1x320x3152x1xf16>
    // CHECK:       return [[RESHAPE]] : tensor<1x320x3152x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeConv1DWithHaloK11
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x512x1x512xf16>
func.func @ReshapeConv1DWithHaloK11(%arg0: tensor<1x512x1x512xf16>) -> tensor<1x512x1x512xf16> {
    %filter = const.Declare tensor<512x512x1x11xf16> = dense<1.000000e+00> : tensor<512x512x1x11xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 5], pads_end = [0, 5], strides = [1, 1]} : tensor<1x512x1x512xf16>, tensor<512x512x1x11xf16> -> tensor<1x512x1x512xf16>
    return %0 : tensor<1x512x1x512xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<512x512x1x11xf16> = dense<1.000000e+00> : tensor<512x512x1x11xf16>
    // CHECK-DAG:   [[PAD_BEGIN:%.+]] = const.Declare tensor<1x512x1x5xf16> = dense<0.000000e+00> : tensor<1x512x1x5xf16>
    // CHECK-DAG:   [[PAD_END:%.+]] = const.Declare tensor<1x512x1x5xf16> = dense<0.000000e+00> : tensor<1x512x1x5xf16>
    // CHECK:       [[PAD:%.+]] = IE.Concat([[PAD_BEGIN]], [[ARG_0]], [[PAD_END]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:    : tensor<1x512x1x5xf16>, tensor<1x512x1x512xf16>, tensor<1x512x1x5xf16> -> tensor<1x512x1x522xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 0] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 16] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 32] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 48] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_4:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 64] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_5:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 80] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_6:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 96] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_7:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 112] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_8:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 128] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_9:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 144] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_10:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 160] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_11:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 176] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_12:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 192] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_13:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 208] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_14:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 224] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_15:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 240] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_16:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 256] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_17:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 272] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_18:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 288] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_19:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 304] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_20:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 320] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_21:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 336] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_22:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 352] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_23:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 368] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_24:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 384] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_25:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 400] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_26:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 416] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_27:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 432] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_28:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 448] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_29:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 464] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_30:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 480] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_31:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 496] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SLICE_0]], [[SLICE_1]], [[SLICE_2]], [[SLICE_3]], [[SLICE_4]], [[SLICE_5]], [[SLICE_6]], [[SLICE_7]], [[SLICE_8]], [[SLICE_9]], [[SLICE_10]], [[SLICE_11]], [[SLICE_12]], [[SLICE_13]], [[SLICE_14]], [[SLICE_15]], [[SLICE_16]], [[SLICE_17]], [[SLICE_18]], [[SLICE_19]], [[SLICE_20]], [[SLICE_21]], [[SLICE_22]], [[SLICE_23]], [[SLICE_24]], [[SLICE_25]], [[SLICE_26]], [[SLICE_27]], [[SLICE_28]], [[SLICE_29]], [[SLICE_30]], [[SLICE_31]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK-SAME:    -> tensor<1x512x32x26xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[CONCAT]], [[FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x32x26xf16>, tensor<512x512x1x11xf16> -> tensor<1x512x32x16xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 512]} : tensor<1x512x32x16xf16> -> tensor<1x512x1x512xf16>
    // CHECK:       return [[RESHAPE]] : tensor<1x512x1x512xf16>
}

// -----

// CHECK-LABEL: @ReshapeConv1DWithHaloK11Bias
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x512x1x512xf16>
func.func @ReshapeConv1DWithHaloK11Bias(%arg0: tensor<1x512x1x512xf16>) -> tensor<1x512x1x512xf16> {
    %filter = const.Declare tensor<512x512x1x11xf16> = dense<1.000000e+00> : tensor<512x512x1x11xf16>
    %bias = const.Declare tensor<1x512x1x1xf16> = dense<5.000000e-01> : tensor<1x512x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 5], pads_end = [0, 5], strides = [1, 1]} : tensor<1x512x1x512xf16>, tensor<512x512x1x11xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x1x512xf16>
    return %0 : tensor<1x512x1x512xf16>

    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x512x1x1xf16> = dense<5.000000e-01> : tensor<1x512x1x1xf16>
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<512x512x1x11xf16> = dense<1.000000e+00> : tensor<512x512x1x11xf16>
    // CHECK-DAG:   [[PAD_BEGIN:%.+]] = const.Declare tensor<1x512x1x5xf16> = dense<0.000000e+00> : tensor<1x512x1x5xf16>
    // CHECK-DAG:   [[PAD_END:%.+]] = const.Declare tensor<1x512x1x5xf16> = dense<0.000000e+00> : tensor<1x512x1x5xf16>
    // CHECK:       [[PAD:%.+]] = IE.Concat([[PAD_BEGIN]], [[ARG_0]], [[PAD_END]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:    : tensor<1x512x1x5xf16>, tensor<1x512x1x512xf16>, tensor<1x512x1x5xf16> -> tensor<1x512x1x522xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 0] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 16] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 32] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 48] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_4:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 64] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_5:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 80] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_6:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 96] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_7:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 112] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_8:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 128] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_9:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 144] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_10:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 160] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_11:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 176] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_12:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 192] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_13:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 208] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_14:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 224] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_15:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 240] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_16:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 256] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_17:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 272] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_18:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 288] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_19:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 304] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_20:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 320] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_21:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 336] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_22:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 352] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_23:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 368] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_24:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 384] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_25:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 400] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_26:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 416] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_27:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 432] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_28:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 448] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_29:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 464] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_30:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 480] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_31:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 496] [1, 512, 1, 26] : tensor<1x512x1x522xf16> to tensor<1x512x1x26xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SLICE_0]], [[SLICE_1]], [[SLICE_2]], [[SLICE_3]], [[SLICE_4]], [[SLICE_5]], [[SLICE_6]], [[SLICE_7]], [[SLICE_8]], [[SLICE_9]], [[SLICE_10]], [[SLICE_11]], [[SLICE_12]], [[SLICE_13]], [[SLICE_14]], [[SLICE_15]], [[SLICE_16]], [[SLICE_17]], [[SLICE_18]], [[SLICE_19]], [[SLICE_20]], [[SLICE_21]], [[SLICE_22]], [[SLICE_23]], [[SLICE_24]], [[SLICE_25]], [[SLICE_26]], [[SLICE_27]], [[SLICE_28]], [[SLICE_29]], [[SLICE_30]], [[SLICE_31]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK-SAME:    -> tensor<1x512x32x26xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[CONCAT]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x32x26xf16>, tensor<512x512x1x11xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x32x16xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 512]} : tensor<1x512x32x16xf16> -> tensor<1x512x1x512xf16>
    // CHECK:       return [[RESHAPE]] : tensor<1x512x1x512xf16>
}

// -----

// CHECK-LABEL: @ReshapeConv1DLayoutHK11
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x512x512x1xf16>
func.func @ReshapeConv1DLayoutHK11(%arg0: tensor<1x512x512x1xf16>) -> tensor<1x512x512x1xf16> {
    %filter = const.Declare tensor<512x512x11x1xf16> = dense<1.000000e+00> : tensor<512x512x11x1xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [5, 0], pads_end = [5, 0], strides = [1, 1]} : tensor<1x512x512x1xf16>, tensor<512x512x11x1xf16> -> tensor<1x512x512x1xf16>
    return %0 : tensor<1x512x512x1xf16>

    // CHECK-DAG:   [[PAD_BEGIN:%.+]] = const.Declare tensor<1x512x5x1xf16> = dense<0.000000e+00> : tensor<1x512x5x1xf16>
    // CHECK-DAG:   [[PAD_END:%.+]] = const.Declare tensor<1x512x5x1xf16> = dense<0.000000e+00> : tensor<1x512x5x1xf16>
    // CHECK:       [[PAD:%.+]] = IE.Concat([[PAD_BEGIN]], [[ARG_0]], [[PAD_END]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK-SAME:    : tensor<1x512x5x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x5x1xf16> -> tensor<1x512x522x1xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[SLICE_0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[PAD]] [0, 0, 16, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[SLICE_1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[PAD]] [0, 0, 32, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[SLICE_2]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice [[PAD]] [0, 0, 48, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.AffineReshape([[SLICE_3]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_4:%.+]] = IE.Slice [[PAD]] [0, 0, 64, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.AffineReshape([[SLICE_4]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_5:%.+]] = IE.Slice [[PAD]] [0, 0, 80, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.AffineReshape([[SLICE_5]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_6:%.+]] = IE.Slice [[PAD]] [0, 0, 96, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_6:%.+]] = IE.AffineReshape([[SLICE_6]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_7:%.+]] = IE.Slice [[PAD]] [0, 0, 112, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_7:%.+]] = IE.AffineReshape([[SLICE_7]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_8:%.+]] = IE.Slice [[PAD]] [0, 0, 128, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_8:%.+]] = IE.AffineReshape([[SLICE_8]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_9:%.+]] = IE.Slice [[PAD]] [0, 0, 144, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_9:%.+]] = IE.AffineReshape([[SLICE_9]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_10:%.+]] = IE.Slice [[PAD]] [0, 0, 160, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_10:%.+]] = IE.AffineReshape([[SLICE_10]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_11:%.+]] = IE.Slice [[PAD]] [0, 0, 176, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_11:%.+]] = IE.AffineReshape([[SLICE_11]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_12:%.+]] = IE.Slice [[PAD]] [0, 0, 192, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_12:%.+]] = IE.AffineReshape([[SLICE_12]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_13:%.+]] = IE.Slice [[PAD]] [0, 0, 208, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_13:%.+]] = IE.AffineReshape([[SLICE_13]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_14:%.+]] = IE.Slice [[PAD]] [0, 0, 224, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_14:%.+]] = IE.AffineReshape([[SLICE_14]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_15:%.+]] = IE.Slice [[PAD]] [0, 0, 240, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_15:%.+]] = IE.AffineReshape([[SLICE_15]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_16:%.+]] = IE.Slice [[PAD]] [0, 0, 256, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_16:%.+]] = IE.AffineReshape([[SLICE_16]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_17:%.+]] = IE.Slice [[PAD]] [0, 0, 272, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_17:%.+]] = IE.AffineReshape([[SLICE_17]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_18:%.+]] = IE.Slice [[PAD]] [0, 0, 288, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_18:%.+]] = IE.AffineReshape([[SLICE_18]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_19:%.+]] = IE.Slice [[PAD]] [0, 0, 304, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_19:%.+]] = IE.AffineReshape([[SLICE_19]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_20:%.+]] = IE.Slice [[PAD]] [0, 0, 320, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_20:%.+]] = IE.AffineReshape([[SLICE_20]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_21:%.+]] = IE.Slice [[PAD]] [0, 0, 336, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_21:%.+]] = IE.AffineReshape([[SLICE_21]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_22:%.+]] = IE.Slice [[PAD]] [0, 0, 352, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_22:%.+]] = IE.AffineReshape([[SLICE_22]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_23:%.+]] = IE.Slice [[PAD]] [0, 0, 368, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_23:%.+]] = IE.AffineReshape([[SLICE_23]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_24:%.+]] = IE.Slice [[PAD]] [0, 0, 384, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_24:%.+]] = IE.AffineReshape([[SLICE_24]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_25:%.+]] = IE.Slice [[PAD]] [0, 0, 400, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_25:%.+]] = IE.AffineReshape([[SLICE_25]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_26:%.+]] = IE.Slice [[PAD]] [0, 0, 416, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_26:%.+]] = IE.AffineReshape([[SLICE_26]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_27:%.+]] = IE.Slice [[PAD]] [0, 0, 432, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_27:%.+]] = IE.AffineReshape([[SLICE_27]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_28:%.+]] = IE.Slice [[PAD]] [0, 0, 448, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_28:%.+]] = IE.AffineReshape([[SLICE_28]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_29:%.+]] = IE.Slice [[PAD]] [0, 0, 464, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_29:%.+]] = IE.AffineReshape([[SLICE_29]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_30:%.+]] = IE.Slice [[PAD]] [0, 0, 480, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_30:%.+]] = IE.AffineReshape([[SLICE_30]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[SLICE_31:%.+]] = IE.Slice [[PAD]] [0, 0, 496, 0] [1, 512, 26, 1] : tensor<1x512x522x1xf16> to tensor<1x512x26x1xf16>
    // CHECK:       [[RESHAPE_31:%.+]] = IE.AffineReshape([[SLICE_31]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 1, 26]} : tensor<1x512x26x1xf16> -> tensor<1x512x1x26xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_0]], [[RESHAPE_1]], [[RESHAPE_2]], [[RESHAPE_3]], [[RESHAPE_4]], [[RESHAPE_5]], [[RESHAPE_6]], [[RESHAPE_7]], [[RESHAPE_8]], [[RESHAPE_9]], [[RESHAPE_10]], [[RESHAPE_11]], [[RESHAPE_12]], [[RESHAPE_13]], [[RESHAPE_14]], [[RESHAPE_15]], [[RESHAPE_16]], [[RESHAPE_17]], [[RESHAPE_18]], [[RESHAPE_19]], [[RESHAPE_20]], [[RESHAPE_21]], [[RESHAPE_22]], [[RESHAPE_23]], [[RESHAPE_24]], [[RESHAPE_25]], [[RESHAPE_26]], [[RESHAPE_27]], [[RESHAPE_28]], [[RESHAPE_29]], [[RESHAPE_30]], [[RESHAPE_31]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK-SAME:    -> tensor<1x512x32x26xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<512x512x1x11xf16> = dense<1.000000e+00> : tensor<512x512x11x1xf16>, [#const.AffineReshape<{{\[\[}}0], [1], [2, 3], [3]], [512, 512, 1, 11]>]
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[CONCAT]], [[FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x32x26xf16>, tensor<512x512x1x11xf16> -> tensor<1x512x32x16xf16>
    // CHECK:       [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 512, 512, 1]} : tensor<1x512x32x16xf16> -> tensor<1x512x512x1xf16>
    // CHECK:       return [[OUT_RESHAPE]] : tensor<1x512x512x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeConv1DWithHaloK11Group
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1024x1x512xf16>
func.func @ReshapeConv1DWithHaloK11Group(%arg0: tensor<1x1024x1x512xf16>) -> tensor<1x1024x1x512xf16> {
    %filter = const.Declare tensor<1024x256x1x11xf16> = dense<1.000000e+00> : tensor<1024x256x1x11xf16>
    %0 = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 5], pads_end = [0, 5], strides = [1, 1]} : tensor<1x1024x1x512xf16>, tensor<1024x256x1x11xf16> -> tensor<1x1024x1x512xf16>
    return %0 : tensor<1x1024x1x512xf16>

    // CHECK-DAG:   [[PAD_BEGIN:%.+]] = const.Declare tensor<1x1024x1x5xf16> = dense<0.000000e+00> : tensor<1x1024x1x5xf16>
    // CHECK-DAG:   [[PAD_END:%.+]] = const.Declare tensor<1x1024x1x5xf16> = dense<0.000000e+00> : tensor<1x1024x1x5xf16>
    // CHECK:       [[PAD:%.+]] = IE.Concat([[PAD_BEGIN]], [[ARG_0]], [[PAD_END]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:    : tensor<1x1024x1x5xf16>, tensor<1x1024x1x512xf16>, tensor<1x1024x1x5xf16> -> tensor<1x1024x1x522xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 0] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 16] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 32] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 48] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_4:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 64] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_5:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 80] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_6:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 96] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_7:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 112] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_8:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 128] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_9:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 144] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_10:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 160] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_11:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 176] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_12:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 192] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_13:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 208] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_14:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 224] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_15:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 240] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_16:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 256] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_17:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 272] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_18:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 288] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_19:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 304] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_20:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 320] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_21:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 336] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_22:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 352] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_23:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 368] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_24:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 384] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_25:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 400] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_26:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 416] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_27:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 432] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_28:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 448] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_29:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 464] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_30:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 480] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[SLICE_31:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 496] [1, 1024, 1, 26] : tensor<1x1024x1x522xf16> to tensor<1x1024x1x26xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SLICE_0]], [[SLICE_1]], [[SLICE_2]], [[SLICE_3]], [[SLICE_4]], [[SLICE_5]], [[SLICE_6]], [[SLICE_7]], [[SLICE_8]], [[SLICE_9]], [[SLICE_10]], [[SLICE_11]], [[SLICE_12]], [[SLICE_13]], [[SLICE_14]], [[SLICE_15]], [[SLICE_16]], [[SLICE_17]], [[SLICE_18]], [[SLICE_19]], [[SLICE_20]], [[SLICE_21]], [[SLICE_22]], [[SLICE_23]], [[SLICE_24]], [[SLICE_25]], [[SLICE_26]], [[SLICE_27]], [[SLICE_28]], [[SLICE_29]], [[SLICE_30]], [[SLICE_31]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK-SAME:    -> tensor<1x1024x32x26xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[CONCAT]], {{.*}}) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x32x26xf16>, tensor<1024x256x1x11xf16> -> tensor<1x1024x32x16xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1024, 1, 512]} : tensor<1x1024x32x16xf16> -> tensor<1x1024x1x512xf16>
    // CHECK:       return [[RESHAPE]] : tensor<1x1024x1x512xf16>
}

// -----

// CHECK-LABEL: @SkipConv1DDepthwiseGroupSmallMAC
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x256x1x9600xf16>
func.func @SkipConv1DDepthwiseGroupSmallMAC(%arg0: tensor<1x256x1x9600xf16>) -> tensor<1x256x1x9600xf16> {
    %filter = const.Declare tensor<256x1x1x11xf16> = dense<1.000000e+00> : tensor<256x1x1x11xf16>
    %0 = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 5], pads_end = [0, 5], strides = [1, 1]} : tensor<1x256x1x9600xf16>, tensor<256x1x1x11xf16> -> tensor<1x256x1x9600xf16>
    return %0 : tensor<1x256x1x9600xf16>

    // CHECK-NOT:   IE.Slice
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[ARG_0]], {{.*}}) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 5], pads_end = [0, 5], strides = [1, 1]}
    // CHECK-SAME:    -> tensor<1x256x1x9600xf16>
    // CHECK:       return [[CONV]] : tensor<1x256x1x9600xf16>
}

// -----

// CHECK-LABEL: @SkipConv1DWithStrideTwo
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x256x1x1600xf16>
func.func @SkipConv1DWithStrideTwo(%arg0: tensor<1x256x1x1600xf16>) -> tensor<1x256x1x800xf16> {
    %filter = const.Declare tensor<256x256x1x11xf16> = dense<1.000000e+00> : tensor<256x256x1x11xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 5], pads_end = [0, 4], strides = [1, 2]} : tensor<1x256x1x1600xf16>, tensor<256x256x1x11xf16> -> tensor<1x256x1x800xf16>
    return %0 : tensor<1x256x1x800xf16>

    // CHECK-NOT:   IE.Slice
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG_0]], {{.*}}) {dilations = [1, 1], pads_begin = [0, 5], pads_end = [0, 4], strides = [1, 2]}
    // CHECK:       return [[CONV]] : tensor<1x256x1x800xf16>
}

// -----

// CHECK-LABEL: @SkipConv1DWithKernelOne
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1280x1x4096xf16>
func.func @SkipConv1DWithKernelOne(%arg0: tensor<1x1280x1x4096xf16>) -> tensor<1x320x1x4096xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1x4096xf16>, tensor<320x1280x1x1xf16> -> tensor<1x320x1x4096xf16>
    return %0 : tensor<1x320x1x4096xf16>

    // CHECK-NOT:   IE.Slice
    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME:    : tensor<1x1280x1x4096xf16> -> tensor<1x1280x{{[0-9]+}}x{{[0-9]+}}xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE_IN]],
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x320x1x4096xf16>
}

// -----

// Mirrors the K=8, stride=4 TransposedConvolution (vocoder-style upsampling decoder,
// H-major 1D layout, i.e. W == 1) found in the target model. kernel % stride == 0 (q=2),
// so haloPad = q - 1 = 1.

// CHECK-LABEL: @ReshapeTransposedConv1DWithHalo
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x384x220x1xf16>
func.func @ReshapeTransposedConv1DWithHalo(%arg0: tensor<1x384x220x1xf16>) -> tensor<1x192x884x1xf16> {
    %filter = const.Declare tensor<192x384x8x1xf16> = dense<1.000000e+00> : tensor<192x384x8x1xf16>
    %0 = IE.TransposedConvolution(%arg0, %filter) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [4, 1]} : tensor<1x384x220x1xf16>, tensor<192x384x8x1xf16> -> tensor<1x192x884x1xf16>
    return %0 : tensor<1x192x884x1xf16>

    // CHECK-DAG:   [[PAD_BEGIN:%.+]] = const.Declare tensor<1x384x1x1xf16> = dense<0.000000e+00> : tensor<1x384x1x1xf16>
    // CHECK-DAG:   [[PAD_END:%.+]] = const.Declare tensor<1x384x1x1xf16> = dense<0.000000e+00> : tensor<1x384x1x1xf16>
    // CHECK:       [[PAD:%.+]] = IE.Concat([[PAD_BEGIN]], [[ARG_0]], [[PAD_END]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK-SAME:    : tensor<1x384x1x1xf16>, tensor<1x384x220x1xf16>, tensor<1x384x1x1xf16> -> tensor<1x384x222x1xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[SLICE_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 384, 1, 14]} : tensor<1x384x14x1xf16> -> tensor<1x384x1x14xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[PAD]] [0, 0, 13, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[SLICE_1]])
    // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[PAD]] [0, 0, 26, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[SLICE_2]])
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice [[PAD]] [0, 0, 39, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.AffineReshape([[SLICE_3]])
    // CHECK:       [[SLICE_4:%.+]] = IE.Slice [[PAD]] [0, 0, 52, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.AffineReshape([[SLICE_4]])
    // CHECK:       [[SLICE_5:%.+]] = IE.Slice [[PAD]] [0, 0, 65, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.AffineReshape([[SLICE_5]])
    // CHECK:       [[SLICE_6:%.+]] = IE.Slice [[PAD]] [0, 0, 78, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_6:%.+]] = IE.AffineReshape([[SLICE_6]])
    // CHECK:       [[SLICE_7:%.+]] = IE.Slice [[PAD]] [0, 0, 91, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_7:%.+]] = IE.AffineReshape([[SLICE_7]])
    // CHECK:       [[SLICE_8:%.+]] = IE.Slice [[PAD]] [0, 0, 104, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_8:%.+]] = IE.AffineReshape([[SLICE_8]])
    // CHECK:       [[SLICE_9:%.+]] = IE.Slice [[PAD]] [0, 0, 117, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_9:%.+]] = IE.AffineReshape([[SLICE_9]])
    // CHECK:       [[SLICE_10:%.+]] = IE.Slice [[PAD]] [0, 0, 130, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_10:%.+]] = IE.AffineReshape([[SLICE_10]])
    // CHECK:       [[SLICE_11:%.+]] = IE.Slice [[PAD]] [0, 0, 143, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_11:%.+]] = IE.AffineReshape([[SLICE_11]])
    // CHECK:       [[SLICE_12:%.+]] = IE.Slice [[PAD]] [0, 0, 156, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_12:%.+]] = IE.AffineReshape([[SLICE_12]])
    // CHECK:       [[SLICE_13:%.+]] = IE.Slice [[PAD]] [0, 0, 169, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_13:%.+]] = IE.AffineReshape([[SLICE_13]])
    // CHECK:       [[SLICE_14:%.+]] = IE.Slice [[PAD]] [0, 0, 182, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_14:%.+]] = IE.AffineReshape([[SLICE_14]])
    // CHECK:       [[SLICE_15:%.+]] = IE.Slice [[PAD]] [0, 0, 195, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_15:%.+]] = IE.AffineReshape([[SLICE_15]])
    // CHECK:       [[SLICE_16:%.+]] = IE.Slice [[PAD]] [0, 0, 208, 0] [1, 384, 14, 1] : tensor<1x384x222x1xf16> to tensor<1x384x14x1xf16>
    // CHECK:       [[RESHAPE_16:%.+]] = IE.AffineReshape([[SLICE_16]])
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_0]], [[RESHAPE_1]], [[RESHAPE_2]], [[RESHAPE_3]], [[RESHAPE_4]], [[RESHAPE_5]], [[RESHAPE_6]], [[RESHAPE_7]], [[RESHAPE_8]], [[RESHAPE_9]], [[RESHAPE_10]], [[RESHAPE_11]], [[RESHAPE_12]], [[RESHAPE_13]], [[RESHAPE_14]], [[RESHAPE_15]], [[RESHAPE_16]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK-SAME:    -> tensor<1x384x17x14xf16>
    // CHECK:       [[FILTER_T:%.+]] = const.Declare tensor<192x384x1x8xf16> = dense<1.000000e+00> : tensor<192x384x8x1xf16>, [#const.AffineReshape<{{\[\[}}0{{\]}}, {{\[}}1{{\]}}, {{\[}}2, 3{{\]}}, {{\[}}3{{\]\]}}, [192, 384, 1, 8]>]
    // CHECK:       [[CONV:%.+]] = IE.TransposedConvolution([[CONCAT]], [[FILTER_T]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 4]}
    // CHECK-SAME:    : tensor<1x384x17x14xf16>, tensor<192x384x1x8xf16> -> tensor<1x192x17x60xf16>
    // CHECK:       [[SLICE_VALID:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 4] [1, 192, 17, 52] : tensor<1x192x17x60xf16> to tensor<1x192x17x52xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[SLICE_VALID]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 192, 884, 1]} : tensor<1x192x17x52xf16> -> tensor<1x192x884x1xf16>
    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x192x884x1xf16>
}
