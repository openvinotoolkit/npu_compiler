//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --m2i-batchnorm-fusion --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @FuseInterpMultAddTask
func.func @FuseInterpMultAddTask(%arg0: tensor<1x3x64x64xf16>) -> tensor<1x3x256x256xf16> {
    %interp = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, axes_attr = [2, 3], scales_attr = [0.0, 0.0], sizes_attr = [256, 256]} : tensor<1x3x64x64xf16> -> tensor<1x3x256x256xf16>
    %cst_0 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[3.635250e-01]], [[3.640140e-01]]]]> : tensor<1x3x1x1xf16>
    %mul = IE.Multiply(%interp, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x256x256xf16>
    %cst_1 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[2.653810e-01]], [[6.357420e-01]]]]> : tensor<1x3x1x1xf16>
    %add = IE.Add(%mul, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x256x256xf16>
    return %add : tensor<1x3x256x256xf16>

    // CHECK: [[INTERP:%.+]] = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [0.000000e+00, 0.000000e+00], sizes_attr = [256, 256]} : tensor<1x3x64x64xf16> -> tensor<1x3x256x256xf16>
    // CHECK: [[BATCHNORM:%.+]] = IE.BatchNormInference([[INTERP]]) {beta_value = [0.000000e+00, 0.265380859375, 0.6357421875], eps = 0.000000e+00 : f64, gamma_value = [0.000000e+00, 0.363525390625, 0.364013671875], mean_value = [0.000000e+00, 0.000000e+00, 0.000000e+00], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, variance_value = [1.000000e+00, 1.000000e+00, 1.000000e+00]} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16>
    // CHECK: return [[BATCHNORM]] : tensor<1x3x256x256xf16>
}

// -----

// CHECK-LABEL: @FuseInterpConvertMultAddTask
func.func @FuseInterpConvertMultAddTask(%arg0: tensor<1x3x64x64xui8>) -> tensor<1x3x256x256xf16> {
    %interp = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, axes_attr = [2, 3], scales_attr = [0.0, 0.0], sizes_attr = [256, 256]} : tensor<1x3x64x64xui8> -> tensor<1x3x256x256xui8>
    %convert = IE.Convert(%interp) {dstElemType = f16} : tensor<1x3x256x256xui8> -> tensor<1x3x256x256xf16>
    %cst_0 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[3.635250e-01]], [[3.640140e-01]]]]> : tensor<1x3x1x1xf16>
    %mul = IE.Multiply(%convert, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x256x256xf16>
    %cst_1 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[2.653810e-01]], [[6.357420e-01]]]]> : tensor<1x3x1x1xf16>
    %add = IE.Add(%mul, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x256x256xf16>
    return %add : tensor<1x3x256x256xf16>

    // CHECK: [[INTERP:%.+]] = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [0.000000e+00, 0.000000e+00], sizes_attr = [256, 256]} : tensor<1x3x64x64xui8> -> tensor<1x3x256x256xui8>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INTERP]]) {dstElemType = f16} : tensor<1x3x256x256xui8> -> tensor<1x3x256x256xf16>
    // CHECK: [[BATCHNORM:%.+]] = IE.BatchNormInference([[CONVERT]]) {beta_value = [0.000000e+00, 0.265380859375, 0.6357421875], eps = 0.000000e+00 : f64, gamma_value = [0.000000e+00, 0.363525390625, 0.364013671875], mean_value = [0.000000e+00, 0.000000e+00, 0.000000e+00], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, variance_value = [1.000000e+00, 1.000000e+00, 1.000000e+00]} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16>
    // CHECK: return [[BATCHNORM]] : tensor<1x3x256x256xf16>
}

// -----

// CHECK-LABEL: @FuseConvertInterpMultAddTask
func.func @FuseConvertInterpMultAddTask(%arg0: tensor<1x3x64x64xui8>) -> tensor<1x3x256x256xf16> {
    %convert = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x64x64xui8> -> tensor<1x3x64x64xf16>
    %interp = IE.Interpolate(%convert) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, axes_attr = [2, 3], scales_attr = [0.0, 0.0], sizes_attr = [256, 256]} : tensor<1x3x64x64xf16> -> tensor<1x3x256x256xf16>
    %cst_0 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[3.635250e-01]], [[3.640140e-01]]]]> : tensor<1x3x1x1xf16>
    %mul = IE.Multiply(%interp, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x256x256xf16>
    %cst_1 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[2.653810e-01]], [[6.357420e-01]]]]> : tensor<1x3x1x1xf16>
    %add = IE.Add(%mul, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x256x256xf16>
    return %add : tensor<1x3x256x256xf16>

    // CHECK: [[CONVERT:%.+]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x64x64xui8> -> tensor<1x3x64x64xf16>
    // CHECK: [[INTERP:%.+]] = IE.Interpolate([[CONVERT]]) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [0.000000e+00, 0.000000e+00], sizes_attr = [256, 256]} : tensor<1x3x64x64xf16> -> tensor<1x3x256x256xf16>
    // CHECK: [[BATCHNORM:%.+]] = IE.BatchNormInference([[INTERP]]) {beta_value = [0.000000e+00, 0.265380859375, 0.6357421875], eps = 0.000000e+00 : f64, gamma_value = [0.000000e+00, 0.363525390625, 0.364013671875], mean_value = [0.000000e+00, 0.000000e+00, 0.000000e+00], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, variance_value = [1.000000e+00, 1.000000e+00, 1.000000e+00]} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16>
    // CHECK: return [[BATCHNORM]] : tensor<1x3x256x256xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseCSCConvertPermuteMultAddTask
func.func @FuseCSCConvertPermuteMultAddTask(%arg0: tensor<1x360x320x1xui8, {order = #NHWC}>) -> tensor<1x3x240x320xf16, {order = #NCHW}> {
    %csc = IE.YuvToRgb(%arg0) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 0, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x360x320x1xui8, {order = #NHWC}> -> tensor<1x240x320x3xui8, {order = #NHWC}>
    %convert = IE.Convert(%csc) {dstElemType = f16} : tensor<1x240x320x3xui8, {order = #NHWC}> -> tensor<1x240x320x3xf16, {order = #NHWC}>
    %permute = IE.MemPermute(%convert) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x240x320x3xf16, {order = #NHWC}> -> tensor<1x3x240x320xf16, {order = #NCHW}>
    %cst_0 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[3.635250e-01]], [[3.640140e-01]]]]> : tensor<1x3x1x1xf16>
    %mul = IE.Multiply(%permute, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x240x320xf16, {order = #NCHW}>, tensor<1x3x1x1xf16> -> tensor<1x3x240x320xf16, {order = #NCHW}>
    %cst_1 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[2.653810e-01]], [[6.357420e-01]]]]> : tensor<1x3x1x1xf16>
    %add = IE.Add(%mul, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x240x320xf16, {order = #NCHW}>, tensor<1x3x1x1xf16> -> tensor<1x3x240x320xf16, {order = #NCHW}>
    return %add : tensor<1x3x240x320xf16, {order = #NCHW}>

    // CHECK: [[LUMA:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 240, 320, 1] : tensor<1x360x320x1xui8, {order = #NHWC}> to tensor<1x240x320x1xui8, {order = #NHWC}>
    // CHECK: [[CHROMA:%.*]] = IE.Slice %arg0 [0, 240, 0, 0] [1, 120, 320, 1] : tensor<1x360x320x1xui8, {order = #NHWC}> to tensor<1x120x320x1xui8, {order = #NHWC}>
    // CHECK: [[RESHAPE:%.*]] = IE.Reshape([[CHROMA]]) {shape_value = [1, 120, 160, 2]} : tensor<1x120x320x1xui8, {order = #NHWC}> -> tensor<1x120x160x2xui8>
    // CHECK: [[CSC:%.+]] = IE.YuvToRgb([[LUMA]], [[RESHAPE]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x240x320x1xui8, {order = #NHWC}>, tensor<1x120x160x2xui8> -> tensor<1x240x320x3xui8, {order = #NHWC}>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[CSC]]) {dstElemType = f16} : tensor<1x240x320x3xui8, {order = #NHWC}> -> tensor<1x240x320x3xf16, {order = #NHWC}>
    // CHECK: [[MEMPERM:%.+]] = IE.MemPermute([[CONVERT]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x240x320x3xf16, {order = #NHWC}> -> tensor<1x3x240x320xf16, {order = #NCHW}>
    // CHECK: [[BATCHNORM:%.+]] = IE.BatchNormInference([[MEMPERM]]) {beta_value = [0.000000e+00, 0.265380859375, 0.6357421875], eps = 0.000000e+00 : f64, gamma_value = [0.000000e+00, 0.363525390625, 0.364013671875], mean_value = [0.000000e+00, 0.000000e+00, 0.000000e+00], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, variance_value = [1.000000e+00, 1.000000e+00, 1.000000e+00]} : tensor<1x3x240x320xf16, {order = #NCHW}> -> tensor<1x3x240x320xf16, {order = #NCHW}>
    // CHECK: return [[BATCHNORM]] : tensor<1x3x240x320xf16, {order = #NCHW}>
}

// -----

// CHECK-LABEL: @FuseCSCConvertTransposeMultAddTask
func.func @FuseCSCConvertTransposeMultAddTask(%arg0: tensor<1x360x320x1xui8>) -> tensor<1x3x240x320xf16> {
    %csc = IE.YuvToRgb(%arg0) {inFmt = #IE.color_fmt<I420>, operandSegmentSizes = array<i32: 1, 0, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x360x320x1xui8> -> tensor<1x240x320x3xui8>
    %convert = IE.Convert(%csc) {dstElemType = f16} : tensor<1x240x320x3xui8> -> tensor<1x240x320x3xf16>
    %cst_0 = const.Declare tensor<4xsi64> = dense<[0, 3, 1, 2]> : tensor<4xsi64>
    %transpose = IE.Transpose(%convert, %cst_0) : tensor<1x240x320x3xf16>, tensor<4xsi64> -> tensor<1x3x240x320xf16>
    %cst_1 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[3.635250e-01]], [[3.640140e-01]]]]> : tensor<1x3x1x1xf16>
    %mul = IE.Multiply(%transpose, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x240x320xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x240x320xf16>
    %cst_2 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[2.653810e-01]], [[6.357420e-01]]]]> : tensor<1x3x1x1xf16>
    %add = IE.Add(%mul, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x240x320xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x240x320xf16>
    return %add : tensor<1x3x240x320xf16>

    // CHECK: [[SLICE_Y:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 240, 320, 1] : tensor<1x360x320x1xui8> to tensor<1x240x320x1xui8>
    // CHECK: [[SLICE_UV:%.+]] = IE.Slice %arg0 [0, 240, 0, 0] [1, 120, 320, 1] : tensor<1x360x320x1xui8> to tensor<1x120x320x1xui8>
    // CHECK: [[RESHAPE_UV:%.+]] = IE.Reshape([[SLICE_UV]]) {shape_value = [1, 240, 160, 1]} : tensor<1x120x320x1xui8> -> tensor<1x240x160x1xui8>
    // CHECK: [[SLICE_U:%.+]] = IE.Slice [[RESHAPE_UV]] [0, 0, 0, 0] [1, 120, 160, 1] : tensor<1x240x160x1xui8> to tensor<1x120x160x1xui8>
    // CHECK: [[SLICE_V:%.+]] = IE.Slice [[RESHAPE_UV]] [0, 120, 0, 0] [1, 120, 160, 1] : tensor<1x240x160x1xui8> to tensor<1x120x160x1xui8>
    // CHECK: [[CSC:%.+]] = IE.YuvToRgb([[SLICE_Y]], [[SLICE_U]], [[SLICE_V]]) {inFmt = #IE.color_fmt<I420>, operandSegmentSizes = array<i32: 1, 1, 1>, outFmt = #IE.color_fmt<RGB>} : tensor<1x240x320x1xui8>, tensor<1x120x160x1xui8>, tensor<1x120x160x1xui8> -> tensor<1x240x320x3xui8>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[CSC]]) {dstElemType = f16} : tensor<1x240x320x3xui8> -> tensor<1x240x320x3xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[CONVERT]]) {order_value = #NWCH} : tensor<1x240x320x3xf16> -> tensor<1x3x240x320xf16>
    // CHECK: [[BATCHNORM:%.+]] = IE.BatchNormInference([[TRANSPOSE]]) {beta_value = [0.000000e+00, 0.265380859375, 0.6357421875], eps = 0.000000e+00 : f64, gamma_value = [0.000000e+00, 0.363525390625, 0.364013671875], mean_value = [0.000000e+00, 0.000000e+00, 0.000000e+00], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, variance_value = [1.000000e+00, 1.000000e+00, 1.000000e+00]} : tensor<1x3x240x320xf16> -> tensor<1x3x240x320xf16>
    // CHECK: return [[BATCHNORM]] : tensor<1x3x240x320xf16>
}

// -----

// CHECK-LABEL: @FuseMultAddInterpTask
func.func @FuseMultAddInterpTask(%arg0: tensor<1x3x64x64xf16>) -> tensor<1x3x256x256xf16> {
    %cst_0 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[3.635250e-01]], [[3.640140e-01]]]]> : tensor<1x3x1x1xf16>
    %mul = IE.Multiply(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x64x64xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x64x64xf16>
    %cst_1 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[2.653810e-01]], [[6.357420e-01]]]]> : tensor<1x3x1x1xf16>
    %add = IE.Add(%mul, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x64x64xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x64x64xf16>
    %interp = IE.Interpolate(%add) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, axes_attr = [2, 3], scales_attr = [0.0, 0.0], sizes_attr = [256, 256]} : tensor<1x3x64x64xf16> -> tensor<1x3x256x256xf16>
    return %interp : tensor<1x3x256x256xf16>

    // CHECK: [[BATCHNORM:%.+]] = IE.BatchNormInference(%arg0) {beta_value = [0.000000e+00, 0.265380859375, 0.6357421875], eps = 0.000000e+00 : f64, gamma_value = [0.000000e+00, 0.363525390625, 0.364013671875], mean_value = [0.000000e+00, 0.000000e+00, 0.000000e+00], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, variance_value = [1.000000e+00, 1.000000e+00, 1.000000e+00]} : tensor<1x3x64x64xf16> -> tensor<1x3x64x64xf16>
    // CHECK: [[INTERP:%.+]] = IE.Interpolate([[BATCHNORM]]) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [0.000000e+00, 0.000000e+00], sizes_attr = [256, 256]} : tensor<1x3x64x64xf16> -> tensor<1x3x256x256xf16>
    // CHECK: return [[INTERP]] : tensor<1x3x256x256xf16>
}

// -----

// CHECK-LABEL: @FuseMultAddConvertTransposeInterpTask
func.func @FuseMultAddConvertTransposeInterpTask(%arg0: tensor<1x3x64x64xf16>) -> tensor<1x240x256x3xui8> {
    %cst_0 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[3.635250e-01]], [[3.640140e-01]]]]> : tensor<1x3x1x1xf16>
    %mul = IE.Multiply(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x64x64xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x64x64xf16>
    %cst_1 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[2.653810e-01]], [[6.357420e-01]]]]> : tensor<1x3x1x1xf16>
    %add = IE.Add(%mul, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x64x64xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x64x64xf16>
    %convert = IE.Convert(%add) {dstElemType = ui8} : tensor<1x3x64x64xf16> -> tensor<1x3x64x64xui8>
    %cst_2 = const.Declare tensor<4xsi64> = dense<[0, 2, 3, 1]> : tensor<4xsi64>
    %transpose = IE.Transpose(%convert, %cst_2) : tensor<1x3x64x64xui8>, tensor<4xsi64> -> tensor<1x64x64x3xui8>
    %interp = IE.Interpolate(%transpose) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, axes_attr = [1, 2], scales_attr = [0.0, 0.0], sizes_attr = [240, 256]} : tensor<1x64x64x3xui8> -> tensor<1x240x256x3xui8>
    return %interp : tensor<1x240x256x3xui8>

    // CHECK: [[BATCHNORM:%.+]] = IE.BatchNormInference(%arg0) {beta_value = [0.000000e+00, 0.265380859375, 0.6357421875], eps = 0.000000e+00 : f64, gamma_value = [0.000000e+00, 0.363525390625, 0.364013671875], mean_value = [0.000000e+00, 0.000000e+00, 0.000000e+00], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, variance_value = [1.000000e+00, 1.000000e+00, 1.000000e+00]} : tensor<1x3x64x64xf16> -> tensor<1x3x64x64xf16>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[BATCHNORM]]) {dstElemType = ui8} : tensor<1x3x64x64xf16> -> tensor<1x3x64x64xui8>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[CONVERT]]) {order_value = #NHWC} : tensor<1x3x64x64xui8> -> tensor<1x64x64x3xui8>
    // CHECK: [[INTERP:%.+]] = IE.Interpolate([[TRANSPOSE]]) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, axes_attr = [1, 2], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [0.000000e+00, 0.000000e+00], sizes_attr = [240, 256]} : tensor<1x64x64x3xui8> -> tensor<1x240x256x3xui8>
    // CHECK: return [[INTERP]] : tensor<1x240x256x3xui8>
}

// -----

// CHECK-LABEL: @FuseMultAddDoublePatternTask
func.func @FuseMultAddDoublePatternTask(%arg0: tensor<1x48x48x3xf16>) -> tensor<1x240x256x3xui8> {
    %interp0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, axes_attr = [1, 2], scales_attr = [0.0, 0.0], sizes_attr = [64, 64]} : tensor<1x48x48x3xf16> -> tensor<1x64x64x3xf16>
    %cst_0 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[3.635250e-01]], [[3.640140e-01]]]]> : tensor<1x3x1x1xf16>
    %permute = IE.MemPermute(%interp0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x64x64x3xf16> -> tensor<1x3x64x64xf16>
    %mul = IE.Multiply(%permute, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x64x64xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x64x64xf16>
    %cst_1 = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.000000e+00]], [[2.653810e-01]], [[6.357420e-01]]]]> : tensor<1x3x1x1xf16>
    %add = IE.Add(%mul, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x64x64xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x64x64xf16>
    %convert = IE.Convert(%add) {dstElemType = ui8} : tensor<1x3x64x64xf16> -> tensor<1x3x64x64xui8>
    %cst_2 = const.Declare tensor<4xsi64> = dense<[0, 2, 3, 1]> : tensor<4xsi64>
    %transpose = IE.Transpose(%convert, %cst_2) : tensor<1x3x64x64xui8>, tensor<4xsi64> -> tensor<1x64x64x3xui8>
    %interp = IE.Interpolate(%transpose) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, axes_attr = [1, 2], scales_attr = [0.0, 0.0], sizes_attr = [240, 256]} : tensor<1x64x64x3xui8> -> tensor<1x240x256x3xui8>
    return %interp : tensor<1x240x256x3xui8>

    // CHECK: [[INTERP0:%.+]] = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, axes_attr = [1, 2], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [0.000000e+00, 0.000000e+00], sizes_attr = [64, 64]} : tensor<1x48x48x3xf16> -> tensor<1x64x64x3xf16>
    // CHECK: [[MEMPERM:%.+]] = IE.MemPermute([[INTERP0]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x64x64x3xf16> -> tensor<1x3x64x64xf16>
    // CHECK: [[BATCHNORM:%.+]] = IE.BatchNormInference([[MEMPERM]]) {beta_value = [0.000000e+00, 0.265380859375, 0.6357421875], eps = 0.000000e+00 : f64, gamma_value = [0.000000e+00, 0.363525390625, 0.364013671875], mean_value = [0.000000e+00, 0.000000e+00, 0.000000e+00], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, variance_value = [1.000000e+00, 1.000000e+00, 1.000000e+00]} : tensor<1x3x64x64xf16> -> tensor<1x3x64x64xf16>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[BATCHNORM]]) {dstElemType = ui8} : tensor<1x3x64x64xf16> -> tensor<1x3x64x64xui8>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[CONVERT]]) {order_value = #NHWC} : tensor<1x3x64x64xui8> -> tensor<1x64x64x3xui8>
    // CHECK: [[INTERP:%.+]] = IE.Interpolate([[TRANSPOSE]]) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = 0.000000e+00 : f64>, axes_attr = [1, 2], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [0.000000e+00, 0.000000e+00], sizes_attr = [240, 256]} : tensor<1x64x64x3xui8> -> tensor<1x240x256x3xui8>
    // CHECK: return [[INTERP]] : tensor<1x240x256x3xui8>
}
