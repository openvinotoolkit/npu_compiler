//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-color-conversion --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @FuseYuvToRgbColorConversion
// CHECK-SAME:  ([[Y_INPUT:%.+]]: tensor<1x368x432x1xui8>, [[UV_INPUT:%.+]]: tensor<1x184x216x2xui8>)
func.func @FuseYuvToRgbColorConversion(%arg0: tensor<1x368x432x1xui8>, %arg1: tensor<1x184x216x2xui8>) -> tensor<1x3x368x432xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-128.0> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<127.0> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-128.0> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<127.0> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<-128.0> : tensor<1x1x1x1xf32>
    %cst_4 = const.Declare tensor<1x1x1x1xf32> = dense<127.0> : tensor<1x1x1x1xf32>
    %cst_5 = const.Declare tensor<1x1x1x1xf32> = dense<-128.0> : tensor<1x1x1x1xf32>
    %cst_6 = const.Declare tensor<1x1x1x1xf32> = dense<127.0> : tensor<1x1x1x1xf32>
    %cst_7 = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-276.928]], [[135.488]], [[-222.912]]]]> : tensor<1x3x1x1xf32>
    %conv_weights = const.Declare tensor<3x3x1x1xf32> = dense<[[[[1.164]], [[0.0]], [[1.596]]], [[[1.164]], [[-0.391]], [[-0.813]]], [[[1.164]], [[2.018]], [[0.0]]]]> : tensor<3x3x1x1xf32>

    // Y channel processing - Convert then AffineReshape
    %y_convert = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x368x432x1xui8> -> tensor<1x368x432x1xf32>
    %y_reshape = IE.AffineReshape(%y_convert) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 1, 368, 432]} : tensor<1x368x432x1xf32> -> tensor<1x1x368x432xf32>

    // UV channel processing - Convert, Transpose, then Interpolate
    %uv_convert = IE.Convert(%arg1) {dstElemType = f32} : tensor<1x184x216x2xui8> -> tensor<1x184x216x2xf32>
    %uv_transpose = IE.Transpose(%uv_convert) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x184x216x2xf32> -> tensor<1x2x184x216xf32>
    %uv_interpolate = IE.Interpolate(%uv_transpose) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [368, 432]} : tensor<1x2x184x216xf32> -> tensor<1x2x368x432xf32>
    %uv_fq = IE.FakeQuantize(%uv_interpolate, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x368x432xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x2x368x432xf32>

    // Concatenation and color conversion
    %concat = IE.Concat(%y_reshape, %uv_fq) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x368x432xf32>, tensor<1x2x368x432xf32> -> tensor<1x3x368x432xf32>
    %concat_fq = IE.FakeQuantize(%concat, %cst_3, %cst_4, %cst_5, %cst_6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x368x432xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x368x432xf32>
    %conv = IE.Convolution(%concat_fq, %conv_weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x368x432xf32>, tensor<3x3x1x1xf32> -> tensor<1x3x368x432xf32>
    %add = IE.Add(%conv, %cst_7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x368x432xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x368x432xf32>

    return %add : tensor<1x3x368x432xf32>

    // CHECK: [[Y_CONVERT:%.+]] = IE.Convert([[Y_INPUT]]) {dstElemType = f32} : tensor<1x368x432x1xui8> -> tensor<1x368x432x1xf32>
    // CHECK: [[UV_CONVERT:%.+]] = IE.Convert([[UV_INPUT]]) {dstElemType = f32} : tensor<1x184x216x2xui8> -> tensor<1x184x216x2xf32>
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_CONVERT]], [[UV_CONVERT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x368x432x1xf32>, tensor<1x184x216x2xf32> -> tensor<1x368x432x3xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[YUV_TO_RGB]]) {order_value = #NWCH} : tensor<1x368x432x3xf32> -> tensor<1x3x368x432xf32>
    // CHECK: return [[TRANSPOSE]] : tensor<1x3x368x432xf32>
}

// -----

// CHECK-LABEL: @FuseYuvToRgbWithoutFakeQuantize
// CHECK-SAME:  ([[Y_INPUT:%.+]]: tensor<1x480x640x1xui8>, [[UV_INPUT:%.+]]: tensor<1x240x320x2xui8>)
func.func @FuseYuvToRgbWithoutFakeQuantize(%arg0: tensor<1x480x640x1xui8>, %arg1: tensor<1x240x320x2xui8>) -> tensor<1x3x480x640xf32> {
    %cst_7 = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-276.928]], [[135.488]], [[-222.912]]]]> : tensor<1x3x1x1xf32>
    %conv_weights = const.Declare tensor<3x3x1x1xf32> = dense<[[[[1.164]], [[0.0]], [[1.596]]], [[[1.164]], [[-0.391]], [[-0.813]]], [[[1.164]], [[2.018]], [[0.0]]]]> : tensor<3x3x1x1xf32>

    // Y channel processing - Convert then AffineReshape
    %y_convert = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x480x640x1xui8> -> tensor<1x480x640x1xf32>
    %y_reshape = IE.AffineReshape(%y_convert) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 1, 480, 640]} : tensor<1x480x640x1xf32> -> tensor<1x1x480x640xf32>

    // UV channel processing - Convert, Transpose, then Interpolate (no FakeQuantize)
    %uv_convert = IE.Convert(%arg1) {dstElemType = f32} : tensor<1x240x320x2xui8> -> tensor<1x240x320x2xf32>
    %uv_transpose = IE.Transpose(%uv_convert) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x240x320x2xf32> -> tensor<1x2x240x320xf32>
    %uv_interpolate = IE.Interpolate(%uv_transpose) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [480, 640]} : tensor<1x2x240x320xf32> -> tensor<1x2x480x640xf32>

    // Concatenation and color conversion (no FakeQuantize on concat)
    %concat = IE.Concat(%y_reshape, %uv_interpolate) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x480x640xf32>, tensor<1x2x480x640xf32> -> tensor<1x3x480x640xf32>
    %conv = IE.Convolution(%concat, %conv_weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x480x640xf32>, tensor<3x3x1x1xf32> -> tensor<1x3x480x640xf32>
    %add = IE.Add(%conv, %cst_7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x480x640xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x480x640xf32>

    return %add : tensor<1x3x480x640xf32>

    // CHECK: [[Y_CONVERT:%.+]] = IE.Convert([[Y_INPUT]]) {dstElemType = f32} : tensor<1x480x640x1xui8> -> tensor<1x480x640x1xf32>
    // CHECK: [[UV_CONVERT:%.+]] = IE.Convert([[UV_INPUT]]) {dstElemType = f32} : tensor<1x240x320x2xui8> -> tensor<1x240x320x2xf32>
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_CONVERT]], [[UV_CONVERT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x480x640x1xf32>, tensor<1x240x320x2xf32> -> tensor<1x480x640x3xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[YUV_TO_RGB]]) {order_value = #NWCH} : tensor<1x480x640x3xf32> -> tensor<1x3x480x640xf32>
    // CHECK: return [[TRANSPOSE]] : tensor<1x3x480x640xf32>
}

// -----

// CHECK-LABEL: @FuseYuvToRgbWithScaling
// CHECK-SAME:  ([[Y_INPUT:%.+]]: tensor<1x320x240x1xui8>, [[UV_INPUT:%.+]]: tensor<1x160x120x2xui8>)
func.func @FuseYuvToRgbWithScaling(%arg0: tensor<1x320x240x1xui8>, %arg1: tensor<1x160x120x2xui8>) -> tensor<1x3x320x240xf32> {
    // Using scaled bias values (2x the standard values) to trigger multiply operation
    %cst_7 = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-553.856]], [[270.976]], [[-445.824]]]]> : tensor<1x3x1x1xf32>
    %conv_weights = const.Declare tensor<3x3x1x1xf32> = dense<[[[[1.164]], [[0.0]], [[1.596]]], [[[1.164]], [[-0.391]], [[-0.813]]], [[[1.164]], [[2.018]], [[0.0]]]]> : tensor<3x3x1x1xf32>

    // Y channel processing - Convert then AffineReshape
    %y_convert = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x320x240x1xui8> -> tensor<1x320x240x1xf32>
    %y_reshape = IE.AffineReshape(%y_convert) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 1, 320, 240]} : tensor<1x320x240x1xf32> -> tensor<1x1x320x240xf32>

    // UV channel processing - Convert, Transpose, then Interpolate
    %uv_convert = IE.Convert(%arg1) {dstElemType = f32} : tensor<1x160x120x2xui8> -> tensor<1x160x120x2xf32>
    %uv_transpose = IE.Transpose(%uv_convert) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x160x120x2xf32> -> tensor<1x2x160x120xf32>
    %uv_interpolate = IE.Interpolate(%uv_transpose) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [320, 240]} : tensor<1x2x160x120xf32> -> tensor<1x2x320x240xf32>

    // Concatenation and color conversion
    %concat = IE.Concat(%y_reshape, %uv_interpolate) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x320x240xf32>, tensor<1x2x320x240xf32> -> tensor<1x3x320x240xf32>
    %conv = IE.Convolution(%concat, %conv_weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x320x240xf32>, tensor<3x3x1x1xf32> -> tensor<1x3x320x240xf32>
    %add = IE.Add(%conv, %cst_7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x320x240xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x320x240xf32>

    return %add : tensor<1x3x320x240xf32>

    // CHECK: [[SCALE_FACTOR:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>
    // CHECK: [[Y_CONVERT:%.+]] = IE.Convert([[Y_INPUT]]) {dstElemType = f32} : tensor<1x320x240x1xui8> -> tensor<1x320x240x1xf32>
    // CHECK: [[UV_CONVERT:%.+]] = IE.Convert([[UV_INPUT]]) {dstElemType = f32} : tensor<1x160x120x2xui8> -> tensor<1x160x120x2xf32>
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_CONVERT]], [[UV_CONVERT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x320x240x1xf32>, tensor<1x160x120x2xf32> -> tensor<1x320x240x3xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[YUV_TO_RGB]]) {order_value = #NWCH} : tensor<1x320x240x3xf32> -> tensor<1x3x320x240xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[TRANSPOSE]], [[SCALE_FACTOR]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x320x240xf32>, tensor<1xf32> -> tensor<1x3x320x240xf32>
    // CHECK: return [[MULTIPLY]] : tensor<1x3x320x240xf32>
}
