//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-color-conversion --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FuseYuvToRgbWithClamp
// CHECK-SAME:  ([[Y_INPUT:%.+]]: tensor<1x32x32x1xf32>, [[UV_INPUT:%.+]]: tensor<1x16x16x2xf32>)
func.func @FuseYuvToRgbWithClamp(%arg0: tensor<1x32x32x1xf32>, %arg1: tensor<1x16x16x2xf32>) -> tensor<1x3x32x32xf32> {
    %cst = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-1.08561909]], [[0.53161991]], [[-0.87418211]]]]> : tensor<1x3x1x1xf32>
    %cst_0 = const.Declare tensor<3x3x1x1xf32> = dense<[[[[0.00456994]], [[0.00792123]], [[0.00000000]]], [[[0.00456994]], [[-0.00152331]], [[-0.00317720]]], [[[0.00456994]], [[0.00000000]], [[0.00626735]]]]> : tensor<3x3x1x1xf32>
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 32, 32]} : tensor<1x32x32x1xf32> -> tensor<1x1x32x32xf32>
    %1 = IE.Transpose(%arg1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x16x16x2xf32> -> tensor<1x2x16x16xf32>
    %2 = IE.Interpolate(%1) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [32, 32]} : tensor<1x2x16x16xf32> -> tensor<1x2x32x32xf32>
    %3 = IE.Concat(%0, %2) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x32x32xf32>, tensor<1x2x32x32xf32> -> tensor<1x3x32x32xf32>
    %4 = IE.Convolution(%3, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x32x32xf32>, tensor<3x3x1x1xf32> -> tensor<1x3x32x32xf32>
    %5 = IE.Add(%4, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x32x32xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x32x32xf32>
    %6 = IE.Clamp(%5) {max = 2.550000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1x3x32x32xf32> -> tensor<1x3x32x32xf32>
    return %6 : tensor<1x3x32x32xf32>

    // CHECK: [[SCALE_FACTOR:%.+]]  = const.Declare tensor<1xf32> = dense<0.00392186968> : tensor<1xf32>
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_INPUT]], [[UV_INPUT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<BGR>, scale = 1.000000e+00 : f32} : tensor<1x32x32x1xf32>, tensor<1x16x16x2xf32> -> tensor<1x32x32x3xf32>
    // CHECK: [[TRANSPOSE:%.+]]  = IE.Transpose([[YUV_TO_RGB]]) {order_value = #NWCH} : tensor<1x32x32x3xf32> -> tensor<1x3x32x32xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[TRANSPOSE]], [[SCALE_FACTOR]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x32x32xf32>, tensor<1xf32> -> tensor<1x3x32x32xf32>
    // CHECK: return [[MULTIPLY]] : tensor<1x3x32x32xf32>
}

// -----

// CHECK-LABEL: @FuseYuvToRgbWithFQ
// CHECK-SAME:  ([[Y_INPUT:%.+]]: tensor<1x368x432x1xui8>, [[UV_INPUT:%.+]]: tensor<1x184x216x2xui8>)
func.func @FuseYuvToRgbWithFQ(%arg0: tensor<1x368x432x1xui8>, %arg1: tensor<1x184x216x2xui8>) -> tensor<1x3x368x432xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-128.0> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<127.0> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-128.0> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<127.0> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<-128.0> : tensor<1x1x1x1xf32>
    %cst_4 = const.Declare tensor<1x1x1x1xf32> = dense<127.0> : tensor<1x1x1x1xf32>
    %cst_5 = const.Declare tensor<1x1x1x1xf32> = dense<-128.0> : tensor<1x1x1x1xf32>
    %cst_6 = const.Declare tensor<1x1x1x1xf32> = dense<127.0> : tensor<1x1x1x1xf32>
    %cst_7 = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-276.928]], [[135.488]], [[-222.912]]]]> : tensor<1x3x1x1xf32>
    %cst_8 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_9 = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %cst_10 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_11 = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %cst_12 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_13 = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %conv_weights_raw = const.Declare tensor<3x3x1x1xsi8> = dense<[[[[3]], [[5]], [[0]]], [[[3]], [[-1]], [[-2]]], [[[3]], [[0]], [[4]]]]> : tensor<3x3x1x1xsi8>
    %conv_weights_scale = const.Declare tensor<1x1x1x1xf32> = dense<0.00152331> : tensor<1x1x1x1xf32>
    %conv_weights_dq = IE.DynamicDequantize(%conv_weights_raw, %conv_weights_scale) {dstElemType = f32} : tensor<3x3x1x1xsi8>, tensor<1x1x1x1xf32> -> tensor<3x3x1x1xf32>

    %y_convert = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x368x432x1xui8> -> tensor<1x368x432x1xf32>
    %y_reshape = IE.AffineReshape(%y_convert) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 368, 432]} : tensor<1x368x432x1xf32> -> tensor<1x1x368x432xf32>
    %uv_convert = IE.Convert(%arg1) {dstElemType = f32} : tensor<1x184x216x2xui8> -> tensor<1x184x216x2xf32>
    %uv_transpose = IE.Transpose(%uv_convert) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x184x216x2xf32> -> tensor<1x2x184x216xf32>
    %uv_interpolate = IE.Interpolate(%uv_transpose) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [368, 432]} : tensor<1x2x184x216xf32> -> tensor<1x2x368x432xf32>
    %uv_fq = IE.FakeQuantize(%uv_interpolate, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x368x432xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x2x368x432xf32>
    %concat = IE.Concat(%y_reshape, %uv_fq) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x368x432xf32>, tensor<1x2x368x432xf32> -> tensor<1x3x368x432xf32>
    %concat_fq = IE.FakeQuantize(%concat, %cst_3, %cst_4, %cst_5, %cst_6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x368x432xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x368x432xf32>
    %conv = IE.Convolution(%concat_fq, %conv_weights_dq) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x368x432xf32>, tensor<3x3x1x1xf32> -> tensor<1x3x368x432xf32>
    %add = IE.Add(%conv, %cst_7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x368x432xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x368x432xf32>
    %final_fq = IE.FakeQuantize(%add, %cst_8, %cst_9, %cst_10, %cst_11) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x368x432xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x368x432xf32>

    return %final_fq : tensor<1x3x368x432xf32>

    // CHECK: [[Y_CONVERT:%.+]] = IE.Convert([[Y_INPUT]]) {dstElemType = f32} : tensor<1x368x432x1xui8> -> tensor<1x368x432x1xf32>
    // CHECK: [[UV_CONVERT:%.+]] = IE.Convert([[UV_INPUT]]) {dstElemType = f32} : tensor<1x184x216x2xui8> -> tensor<1x184x216x2xf32>
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_CONVERT]], [[UV_CONVERT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<BGR>, scale = 1.000000e+00 : f32} : tensor<1x368x432x1xf32>, tensor<1x184x216x2xf32> -> tensor<1x368x432x3xf32>
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
    %clamp = IE.Clamp(%add) {max = 2.550000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1x3x480x640xf32> -> tensor<1x3x480x640xf32>

    return %clamp : tensor<1x3x480x640xf32>

    // CHECK: [[SCALE_FACTOR:%.+]] = const.Declare tensor<1xf32> = dense<1.01575518> : tensor<1xf32>
    // CHECK: [[Y_CONVERT:%.+]] = IE.Convert([[Y_INPUT]]) {dstElemType = f32} : tensor<1x480x640x1xui8> -> tensor<1x480x640x1xf32>
    // CHECK: [[UV_CONVERT:%.+]] = IE.Convert([[UV_INPUT]]) {dstElemType = f32} : tensor<1x240x320x2xui8> -> tensor<1x240x320x2xf32>
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_CONVERT]], [[UV_CONVERT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>, scale = 1.000000e+00 : f32} : tensor<1x480x640x1xf32>, tensor<1x240x320x2xf32> -> tensor<1x480x640x3xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[YUV_TO_RGB]]) {order_value = #NWCH} : tensor<1x480x640x3xf32> -> tensor<1x3x480x640xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[TRANSPOSE]], [[SCALE_FACTOR]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x480x640xf32>, tensor<1xf32> -> tensor<1x3x480x640xf32>
    // CHECK: [[CLAMP:%.+]] = IE.Clamp([[MULTIPLY]]) {max = 2.550000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1x3x480x640xf32> -> tensor<1x3x480x640xf32>
    // CHECK: return [[CLAMP]] : tensor<1x3x480x640xf32>
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
    %clamp = IE.Clamp(%add) {max = 2.550000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1x3x320x240xf32> -> tensor<1x3x320x240xf32>
    return %clamp : tensor<1x3x320x240xf32>


    // CHECK: [[SCALE_FACTOR:%.+]] = const.Declare tensor<1xf32> = dense<2.03151035> : tensor<1xf32>
    // CHECK: [[Y_CONVERT:%.+]] = IE.Convert([[Y_INPUT]]) {dstElemType = f32} : tensor<1x320x240x1xui8> -> tensor<1x320x240x1xf32>
    // CHECK: [[UV_CONVERT:%.+]] = IE.Convert([[UV_INPUT]]) {dstElemType = f32} : tensor<1x160x120x2xui8> -> tensor<1x160x120x2xf32>
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_CONVERT]], [[UV_CONVERT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>, scale = 1.000000e+00 : f32} : tensor<1x320x240x1xf32>, tensor<1x160x120x2xf32> -> tensor<1x320x240x3xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[YUV_TO_RGB]]) {order_value = #NWCH} : tensor<1x320x240x3xf32> -> tensor<1x3x320x240xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[TRANSPOSE]], [[SCALE_FACTOR]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x320x240xf32>, tensor<1xf32> -> tensor<1x3x320x240xf32>
    // CHECK: [[CLAMP:%.+]] = IE.Clamp([[MULTIPLY]]) {max = 2.550000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1x3x320x240xf32> -> tensor<1x3x320x240xf32>
    // CHECK: return [[CLAMP]] : tensor<1x3x320x240xf32>

}

// -----

// CHECK-LABEL: @FuseYuvToRgbWithScalingAndFQ
// CHECK-SAME:  ([[Y_INPUT:%.+]]: tensor<1x320x240x1xui8>, [[UV_INPUT:%.+]]: tensor<1x160x120x2xui8>)
func.func @FuseYuvToRgbWithScalingAndFQ(%arg0: tensor<1x320x240x1xui8>, %arg1: tensor<1x160x120x2xui8>) -> tensor<1x3x320x240xf32> {
    // Scaled bias values (2x standard) to force post-fusion Multiply in non-shave mode.
    %cst_bias = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-553.856]], [[270.976]], [[-445.824]]]]> : tensor<1x3x1x1xf32>
    %cst_weights = const.Declare tensor<3x3x1x1xf32> = dense<[[[[1.164]], [[0.0]], [[1.596]]], [[[1.164]], [[-0.391]], [[-0.813]]], [[[1.164]], [[2.018]], [[0.0]]]]> : tensor<3x3x1x1xf32>
    %cst_in_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %cst_in_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %cst_out_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %cst_out_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %y_convert = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x320x240x1xui8> -> tensor<1x320x240x1xf32>
    %y_reshape = IE.AffineReshape(%y_convert) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 1, 320, 240]} : tensor<1x320x240x1xf32> -> tensor<1x1x320x240xf32>

    %uv_convert = IE.Convert(%arg1) {dstElemType = f32} : tensor<1x160x120x2xui8> -> tensor<1x160x120x2xf32>
    %uv_transpose = IE.Transpose(%uv_convert) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x160x120x2xf32> -> tensor<1x2x160x120xf32>
    %uv_interpolate = IE.Interpolate(%uv_transpose) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [320, 240]} : tensor<1x2x160x120xf32> -> tensor<1x2x320x240xf32>

    %concat = IE.Concat(%y_reshape, %uv_interpolate) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x320x240xf32>, tensor<1x2x320x240xf32> -> tensor<1x3x320x240xf32>
    %conv = IE.Convolution(%concat, %cst_weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x320x240xf32>, tensor<3x3x1x1xf32> -> tensor<1x3x320x240xf32>
    %add = IE.Add(%conv, %cst_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x320x240xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x320x240xf32>
    %fq = IE.FakeQuantize(%add, %cst_in_low, %cst_in_high, %cst_out_low, %cst_out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x320x240xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x320x240xf32>
    return %fq : tensor<1x3x320x240xf32>

    // CHECK: [[SCALE_FACTOR:%.+]] = const.Declare tensor<1xf32> = dense<2.03151035> : tensor<1xf32>
    // CHECK: [[Y_CONVERT:%.+]] = IE.Convert([[Y_INPUT]]) {dstElemType = f32} : tensor<1x320x240x1xui8> -> tensor<1x320x240x1xf32>
    // CHECK: [[UV_CONVERT:%.+]] = IE.Convert([[UV_INPUT]]) {dstElemType = f32} : tensor<1x160x120x2xui8> -> tensor<1x160x120x2xf32>
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_CONVERT]], [[UV_CONVERT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>, scale = 1.000000e+00 : f32} : tensor<1x320x240x1xf32>, tensor<1x160x120x2xf32> -> tensor<1x320x240x3xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[YUV_TO_RGB]]) {order_value = #NWCH} : tensor<1x320x240x3xf32> -> tensor<1x3x320x240xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[TRANSPOSE]], [[SCALE_FACTOR]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x320x240xf32>, tensor<1xf32> -> tensor<1x3x320x240xf32>
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[MULTIPLY]], [[CST_IN_LOW:%.+]], [[CST_IN_HIGH:%.+]], [[CST_OUT_LOW:%.+]], [[CST_OUT_HIGH:%.+]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x320x240xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x320x240xf32>
    // CHECK: return [[FQ]] : tensor<1x3x320x240xf32>
}

// -----

// CHECK-LABEL: @FuseYuvToBgr
// CHECK-SAME:  ([[Y_INPUT:%.+]]: tensor<1x32x32x1xf32>, [[UV_INPUT:%.+]]: tensor<1x16x16x2xf32>)
func.func @FuseYuvToBgr(%arg0: tensor<1x32x32x1xf32>, %arg1: tensor<1x16x16x2xf32>) -> tensor<1x3x32x32xf32> {
    %cst = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-276.928]], [[135.488]], [[-222.912]]]]> : tensor<1x3x1x1xf32>
    %cst_0 = const.Declare tensor<3x3x1x1xf32> = dense<[[[[1.164]], [[2.018]], [[0.000]]], [[[1.164]], [[-0.391]], [[-0.813]]], [[[1.164]], [[0.000]], [[1.596]]]]> : tensor<3x3x1x1xf32>
    %y_reshape = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 32, 32]} : tensor<1x32x32x1xf32> -> tensor<1x1x32x32xf32>
    %uv_transpose = IE.Transpose(%arg1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x16x16x2xf32> -> tensor<1x2x16x16xf32>
    %uv_interpolate = IE.Interpolate(%uv_transpose) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [32, 32]} : tensor<1x2x16x16xf32> -> tensor<1x2x32x32xf32>
    %concat = IE.Concat(%y_reshape, %uv_interpolate) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x32x32xf32>, tensor<1x2x32x32xf32> -> tensor<1x3x32x32xf32>
    %conv = IE.Convolution(%concat, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x32x32xf32>, tensor<3x3x1x1xf32> -> tensor<1x3x32x32xf32>
    %add = IE.Add(%conv, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x32x32xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x32x32xf32>
    %clamp = IE.Clamp(%add) {max = 2.550000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1x3x32x32xf32> -> tensor<1x3x32x32xf32>
    return %clamp : tensor<1x3x32x32xf32>

    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_INPUT]], [[UV_INPUT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<BGR>, scale = 1.000000e+00 : f32} : tensor<1x32x32x1xf32>, tensor<1x16x16x2xf32> -> tensor<1x32x32x3xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[YUV_TO_RGB]]) {order_value = #NWCH} : tensor<1x32x32x3xf32> -> tensor<1x3x32x32xf32>
    // CHECK: return [[TRANSPOSE]] : tensor<1x3x32x32xf32>
}

// -----

// CHECK-LABEL: @NoFuseYuvToRgbWithNonSplatFQ
// CHECK-SAME:  ([[Y_INPUT:%.+]]: tensor<1x32x32x1xf32>, [[UV_INPUT:%.+]]: tensor<1x16x16x2xf32>)
func.func @NoFuseYuvToRgbWithNonSplatFQ(%arg0: tensor<1x32x32x1xf32>, %arg1: tensor<1x16x16x2xf32>) -> tensor<1x3x32x32xf32> {
    // Non-splat output low/high constants - different values per channel
    %cst_output_low = const.Declare tensor<1x3x1x1xf32> = dense<[[[[0.0]], [[10.0]], [[20.0]]]]> : tensor<1x3x1x1xf32>
    %cst_output_high = const.Declare tensor<1x3x1x1xf32> = dense<[[[[255.0]], [[245.0]], [[235.0]]]]> : tensor<1x3x1x1xf32>
    %cst_input_low = const.Declare tensor<1x3x1x1xf32> = dense<0.0> : tensor<1x3x1x1xf32>
    %cst_input_high = const.Declare tensor<1x3x1x1xf32> = dense<255.0> : tensor<1x3x1x1xf32>
    %cst_bias = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-276.928]], [[135.488]], [[-222.912]]]]> : tensor<1x3x1x1xf32>
    %cst_weights = const.Declare tensor<3x3x1x1xf32> = dense<[[[[1.164]], [[0.0]], [[1.596]]], [[[1.164]], [[-0.391]], [[-0.813]]], [[[1.164]], [[2.018]], [[0.0]]]]> : tensor<3x3x1x1xf32>

    %y_reshape = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 32, 32]} : tensor<1x32x32x1xf32> -> tensor<1x1x32x32xf32>
    %uv_transpose = IE.Transpose(%arg1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x16x16x2xf32> -> tensor<1x2x16x16xf32>
    %uv_interpolate = IE.Interpolate(%uv_transpose) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [32, 32]} : tensor<1x2x16x16xf32> -> tensor<1x2x32x32xf32>
    %concat = IE.Concat(%y_reshape, %uv_interpolate) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x32x32xf32>, tensor<1x2x32x32xf32> -> tensor<1x3x32x32xf32>
    %conv = IE.Convolution(%concat, %cst_weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x32x32xf32>, tensor<3x3x1x1xf32> -> tensor<1x3x32x32xf32>
    %add = IE.Add(%conv, %cst_bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x32x32xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x32x32xf32>
    %fq = IE.FakeQuantize(%add, %cst_input_low, %cst_input_high, %cst_output_low, %cst_output_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x32x32xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x32x32xf32>
    return %fq : tensor<1x3x32x32xf32>

    // CHECK-NOT: IE.YuvToRgb
    // CHECK-DAG: [[CST_OUT_LOW:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<{{\[\[\[\[}}0.000000e+00]], {{\[\[}}1.000000e+01]], {{\[\[}}2.000000e+01]]]]>  : tensor<1x3x1x1xf32>
    // CHECK-DAG: [[CST_OUT_HIGH:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<{{\[\[\[\[}}2.550000e+02]], {{\[\[}}2.450000e+02]], {{\[\[}}2.350000e+02]]]]> : tensor<1x3x1x1xf32>
    // CHECK-DAG: [[CST_IN_LOW:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<0.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG: [[CST_IN_HIGH:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<2.550000e+02> : tensor<1x3x1x1xf32>
    // CHECK-DAG: [[CST_BIAS:%.+]] = const.Declare tensor<1x3x1x1xf32> = dense<{{\[\[\[\[}}-2.769280e+02]], {{\[\[}}1.354880e+02]], {{\[\[}}-2.229120e+02]]]]> : tensor<1x3x1x1xf32>
    // CHECK-DAG: [[CST_WEIGHTS:%.+]] = const.Declare tensor<3x3x1x1xf32> = dense<{{\[\[\[\[}}1.164000e+00]], {{\[\[}}0.000000e+00]], {{\[\[}}1.596000e+00]]], {{\[\[\[}}1.164000e+00]], {{\[\[}}-3.910000e-01]], {{\[\[}}-8.130000e-01]]], {{\[\[\[}}1.164000e+00]], {{\[\[}}2.018000e+00]], {{\[\[}}0.000000e+00]]]]> : tensor<3x3x1x1xf32>
    // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[Y_INPUT]]) {dim_mapping = {{\[\[}}0, 1], [2], [3], [3]], shape_value = [1, 1, 32, 32]} : tensor<1x32x32x1xf32> -> tensor<1x1x32x32xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[UV_INPUT]]) {order_value = #NWCH} : tensor<1x16x16x2xf32> -> tensor<1x2x16x16xf32>
    // CHECK: [[INTERPOLATE:%.+]] = IE.Interpolate([[TRANSPOSE]]) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [32, 32]} : tensor<1x2x16x16xf32> -> tensor<1x2x32x32xf32>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[RESHAPE]], [[INTERPOLATE]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x32x32xf32>, tensor<1x2x32x32xf32> -> tensor<1x3x32x32xf32>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[CONCAT]], [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x32x32xf32>, tensor<3x3x1x1xf32> -> tensor<1x3x32x32xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[CONV]], [[CST_BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x32x32xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x32x32xf32>
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[ADD]], [[CST_IN_LOW]], [[CST_IN_HIGH]], [[CST_OUT_LOW]], [[CST_OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x32x32xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x32x32xf32>
    // CHECK: return [[FQ]] : tensor<1x3x32x32xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @FuseDynamicColorSpaceConversion
// CHECK-SAME:  [[Y_INPUT:%[^:]+]]: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 1]> : tensor<4xsi64>}>,
// CHECK-SAME:  [[UV_INPUT:%[^:]+]]: tensor<1x?x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 2]> : tensor<4xsi64>}>
func.func @FuseDynamicColorSpaceConversion(%arg0: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 1]> : tensor<4xsi64>}>, %arg1: tensor<1x?x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 2]> : tensor<4xsi64>}>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}> {
    %cst = const.Declare tensor<4xsi64> = dense<[0, 3, 1, 2]> : tensor<4xsi64>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<242.701309> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<244.095093> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-1.0332917]], [[0.529001474]], [[-0.884533047]]]]> : tensor<1x3x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %cst_4 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %conv_weights_raw = const.Declare tensor<3x3x1x1xsi8> = dense<[[[[3]], [[5]], [[0]]], [[[3]], [[-1]], [[-2]]], [[[3]], [[0]], [[4]]]]> : tensor<3x3x1x1xsi8>
    %conv_weights_scale = const.Declare tensor<1x1x1x1xf32> = dense<0.00152331> : tensor<1x1x1x1xf32>
    %0 = IE.DynamicDequantize(%conv_weights_raw, %conv_weights_scale) {dstElemType = f32} : tensor<3x3x1x1xsi8>, tensor<1x1x1x1xf32> -> tensor<3x3x1x1xf32>
    %1 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 1]> : tensor<4xsi64>}> -> tensor<1x?x?x1xf32, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 1]> : tensor<4xsi64>}>
    %2 = IE.ShapeOf(%1) {dstElemType = si64} : tensor<1x?x?x1xf32, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 1]> : tensor<4xsi64>}> -> tensor<4xsi64>
    %3 = IE.Gather(%2, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<4xsi64>, tensor<4xsi64> -> tensor<4xsi64>
    %4 = IE.DynamicReshape(%1, %3) {output_bounds = [1, 1, 1440, 2560], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x?x?x1xf32, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 1]> : tensor<4xsi64>}>, tensor<4xsi64> -> tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1440, 2560]> : tensor<4xsi64>}>
    %5 = IE.Convert(%arg1) {dstElemType = f32} : tensor<1x?x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 2]> : tensor<4xsi64>}> -> tensor<1x?x?x2xf32, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 2]> : tensor<4xsi64>}>
    %6 = IE.Transpose(%5) {order_value = #NWCH} : tensor<1x?x?x2xf32, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 2]> : tensor<4xsi64>}> -> tensor<1x2x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 720, 1280]> : tensor<4xsi64>}>
    %7 = IE.Interpolate(%6) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [1440, 2560]} : tensor<1x2x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 720, 1280]> : tensor<4xsi64>}> -> tensor<1x2x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 1440, 2560]> : tensor<4xsi64>}>
    %8 = IE.FakeQuantize(%7, %cst_4, %cst_0, %cst_4, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 1440, 2560]> : tensor<4xsi64>}>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x2x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 1440, 2560]> : tensor<4xsi64>}>
    %9 = IE.Concat(%4, %8) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1440, 2560]> : tensor<4xsi64>}>, tensor<1x2x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 1440, 2560]> : tensor<4xsi64>}> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}>
    %10 = IE.FakeQuantize(%9, %cst_4, %cst_1, %cst_4, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}>
    %11 = IE.Convolution(%10, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}>, tensor<3x3x1x1xf32> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}>
    %12 = IE.Add(%11, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}>, tensor<1x3x1x1xf32> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}>
    %13 = IE.FakeQuantize(%12, %cst_4, %cst_3, %cst_4, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}>
    %14 = IE.Convert(%13) {dstElemType = f16} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}>
    return %14 : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}>


    // CHECK:       [[SCALE_CONST:%.+]] = const.Declare tensor<1xf32> = dense<0.00386792072> : tensor<1xf32>

    // CHECK:       [[Y:%.+]] = IE.Convert([[Y_INPUT]]) {dstElemType = f32}
    // CHECK-SAME:      -> tensor<1x?x?x1xf32, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 1]> : tensor<4xsi64>}>
    // CHECK:       [[UV:%.+]] = IE.Convert([[UV_INPUT]]) {dstElemType = f32}
    // CHECK-SAME:      -> tensor<1x?x?x2xf32, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 2]> : tensor<4xsi64>}>

    // CHECK:       [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y]], [[UV]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<BGR>, scale = 1.000000e+00 : f32}
    // CHECK-SAME:      -> tensor<1x?x?x3xf32, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[YUV_TO_RGB]]) {order_value = #NWCH}
    // CHECK-SAME:      -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[TRANSPOSE]], [[SCALE_CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:       [[RESULT:%.+]] = IE.Convert([[MULTIPLY]]) {dstElemType = f16} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1440, 2560]> : tensor<4xsi64>}>
    // CHECK:       return [[RESULT]]
}
