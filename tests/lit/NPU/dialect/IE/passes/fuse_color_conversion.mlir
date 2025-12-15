//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-color-conversion --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

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
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_INPUT]], [[UV_INPUT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<BGR>} : tensor<1x32x32x1xf32>, tensor<1x16x16x2xf32> -> tensor<1x32x32x3xf32>
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
    %cst_14 = const.Declare tensor<1x1x1x1xf32> = dense<-0.00317719812> : tensor<1x1x1x1xf32>
    %cst_15 = const.Declare tensor<1x1x1x1xf32> = dense<0.00792123377> : tensor<1x1x1x1xf32>
    %cst_16 = const.Declare tensor<3x3x1x1xf32> = dense<[[[[0.00456994]], [[0.00792123]], [[0.00000000]]], [[[0.00456994]], [[-0.00152331]], [[-0.00317720]]], [[[0.00456994]], [[0.00000000]], [[0.00626735]]]]> : tensor<3x3x1x1xf32>

    %conv_weights_fq = IE.FakeQuantize(%cst_16, %cst_12, %cst_13, %cst_14, %cst_15) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<3x3x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<3x3x1x1xf32>
    %y_convert = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x368x432x1xui8> -> tensor<1x368x432x1xf32>
    %y_reshape = IE.AffineReshape(%y_convert) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 368, 432]} : tensor<1x368x432x1xf32> -> tensor<1x1x368x432xf32>
    %uv_convert = IE.Convert(%arg1) {dstElemType = f32} : tensor<1x184x216x2xui8> -> tensor<1x184x216x2xf32>
    %uv_transpose = IE.Transpose(%uv_convert) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x184x216x2xf32> -> tensor<1x2x184x216xf32>
    %uv_interpolate = IE.Interpolate(%uv_transpose) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [368, 432]} : tensor<1x2x184x216xf32> -> tensor<1x2x368x432xf32>
    %uv_fq = IE.FakeQuantize(%uv_interpolate, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x368x432xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x2x368x432xf32>
    %concat = IE.Concat(%y_reshape, %uv_fq) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x368x432xf32>, tensor<1x2x368x432xf32> -> tensor<1x3x368x432xf32>
    %concat_fq = IE.FakeQuantize(%concat, %cst_3, %cst_4, %cst_5, %cst_6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x368x432xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x368x432xf32>
    %conv = IE.Convolution(%concat_fq, %conv_weights_fq) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x368x432xf32>, tensor<3x3x1x1xf32> -> tensor<1x3x368x432xf32>
    %add = IE.Add(%conv, %cst_7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x368x432xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x368x432xf32>
    %final_fq = IE.FakeQuantize(%add, %cst_8, %cst_9, %cst_10, %cst_11) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x368x432xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x368x432xf32>

    return %final_fq : tensor<1x3x368x432xf32>

    // CHECK: [[SCALE_FACTOR:%.+]] = const.Declare tensor<1xf32> = dense<1.01575518> : tensor<1xf32>
    // CHECK: [[Y_CONVERT:%.+]] = IE.Convert([[Y_INPUT]]) {dstElemType = f32} : tensor<1x368x432x1xui8> -> tensor<1x368x432x1xf32>
    // CHECK: [[UV_CONVERT:%.+]] = IE.Convert([[UV_INPUT]]) {dstElemType = f32} : tensor<1x184x216x2xui8> -> tensor<1x184x216x2xf32>
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_CONVERT]], [[UV_CONVERT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x368x432x1xf32>, tensor<1x184x216x2xf32> -> tensor<1x368x432x3xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[YUV_TO_RGB]]) {order_value = #NWCH} : tensor<1x368x432x3xf32> -> tensor<1x3x368x432xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[TRANSPOSE]], [[SCALE_FACTOR]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x368x432xf32>, tensor<1xf32> -> tensor<1x3x368x432xf32>
    // CHECK: return [[MULTIPLY]] : tensor<1x3x368x432xf32>

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
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_CONVERT]], [[UV_CONVERT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x480x640x1xf32>, tensor<1x240x320x2xf32> -> tensor<1x480x640x3xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[YUV_TO_RGB]]) {order_value = #NWCH} : tensor<1x480x640x3xf32> -> tensor<1x3x480x640xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[TRANSPOSE]], [[SCALE_FACTOR]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x480x640xf32>, tensor<1xf32> -> tensor<1x3x480x640xf32>
    // CHECK: return [[MULTIPLY]] : tensor<1x3x480x640xf32>
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
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_CONVERT]], [[UV_CONVERT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x320x240x1xf32>, tensor<1x160x120x2xf32> -> tensor<1x320x240x3xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[YUV_TO_RGB]]) {order_value = #NWCH} : tensor<1x320x240x3xf32> -> tensor<1x3x320x240xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[TRANSPOSE]], [[SCALE_FACTOR]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x320x240xf32>, tensor<1xf32> -> tensor<1x3x320x240xf32>
    // CHECK: return [[MULTIPLY]] : tensor<1x3x320x240xf32>

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

    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y_INPUT]], [[UV_INPUT]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<BGR>} : tensor<1x32x32x1xf32>, tensor<1x16x16x2xf32> -> tensor<1x32x32x3xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[YUV_TO_RGB]]) {order_value = #NWCH} : tensor<1x32x32x3xf32> -> tensor<1x3x32x32xf32>
    // CHECK: return [[TRANSPOSE]] : tensor<1x3x32x32xf32>
}
