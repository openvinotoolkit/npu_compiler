//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-reorder-to-nce %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PropagateReorderToConv
func.func @PropagateReorderToConv(%arg0: tensor<1x3x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16> {
    %cst = const.Declare tensor<3x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %1 = IE.Tanh(%0) : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>
    return %2 : tensor<1x3x512x512xf16>

    //CHECK:      [[CONV:%.*]] = IE.Convolution
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:      [[REORDER:%.*]] = IE.Reorder([[CONV]]) {dstOrder = #NCHW}
    //CHECK:      [[TANH:%.*]] = IE.Tanh([[REORDER]])
    //CHECK-SAME: tensor<1x3x512x512xf16> -> tensor<1x3x512x512xf16>
    //CHECK:      return [[TANH]] : tensor<1x3x512x512xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PropagateReorderToConvWithConvert
func.func @PropagateReorderToConvWithConvert(%arg0: tensor<1x3x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf32> {
    %cst = const.Declare tensor<3x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %1 = IE.Tanh(%0) : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>
    %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x3x512x512xf16> -> tensor<1x3x512x512xf32>
    return %3 : tensor<1x3x512x512xf32>

    //CHECK:      [[CONV:%.*]] = IE.Convolution
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:      [[REORDER:%.*]] = IE.Reorder([[CONV]]) {dstOrder = #NCHW}
    //CHECK:      [[TANH:%.*]] = IE.Tanh([[REORDER]])
    //CHECK-SAME: tensor<1x3x512x512xf16> -> tensor<1x3x512x512xf16>
    //CHECK:      [[CONVERT:%.*]] = IE.Convert([[TANH]]) {dstElemType = f32}
    //CHECK-SAME: tensor<1x3x512x512xf16> -> tensor<1x3x512x512xf32>
    //CHECK:      return [[CONVERT]] : tensor<1x3x512x512xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NoPropagateReorderWithoutNCE
func.func @NoPropagateReorderWithoutNCE(%arg0: tensor<1x3x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16> {
    %0 = IE.Tanh(%arg0) : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>
    return %1 : tensor<1x3x512x512xf16>

    //CHECK:      [[TANH:%.*]] = IE.Tanh
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:      [[REORDER:%.*]] = IE.Reorder([[TANH]]) {dstOrder = #NCHW}
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>
    //CHECK:      return [[REORDER]] : tensor<1x3x512x512xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NoPropagateReorderTwoBranches
func.func @NoPropagateReorderTwoBranches(%arg0: tensor<1x3x512x512xf16, {order = #NHWC}>) -> (tensor<1x3x512x512xf16>, tensor<1x3x512x512xf16, {order = #NHWC}>) {
    %cst = const.Declare tensor<3x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %1 = IE.Tanh(%0) : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>
    %3 = IE.MaxPool(%0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    return %2, %3 : tensor<1x3x512x512xf16>, tensor<1x3x512x512xf16, {order = #NHWC}>

    //CHECK:      [[CONV:%.*]] = IE.Convolution
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:      [[TANH:%.*]] = IE.Tanh([[CONV]])
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:      [[REORDER:%.*]] = IE.Reorder([[TANH]]) {dstOrder = #NCHW}
    //CHECK:      [[MAXPOOL:%.*]] = IE.MaxPool([[CONV]])
    //CHECK:      return [[REORDER]], [[MAXPOOL]] : tensor<1x3x512x512xf16>, tensor<1x3x512x512xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>

// CHECK-LABEL: @NoPropagateNoSupportedReorder
func.func @NoPropagateNoSupportedReorder(%arg0: tensor<1x3x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16, {order = #map}> {
    %cst = const.Declare tensor<3x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %1 = IE.MVN(%0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #map} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #map}>
    return %2 : tensor<1x3x512x512xf16, {order = #map}>

    //CHECK:      [[CONV:%.*]] = IE.Convolution
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:      [[MVN:%.*]] = IE.MVN([[CONV]])
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:      [[REORDER:%.*]] = IE.Reorder([[MVN]]) {dstOrder = #map}
    //CHECK:      return [[REORDER]] : tensor<1x3x512x512xf16, {order = #map}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SkipReorderWithBlockArg
func.func @SkipReorderWithBlockArg(%arg0: tensor<1x3x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>
    return %0 : tensor<1x3x512x512xf16>

    // CHECK:   [[REORDER:%.*]] = IE.Reorder(%arg0) {
    // CHECK-SAME:      dstOrder = #NCHW
    // CHECK-SAME:  } : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>

    // CHECK:   return %0 : tensor<1x3x512x512xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PropagateReorderToConvBeforeInterpolate
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @PropagateReorderToConvBeforeInterpolate(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x2x256x256xf16> {
    %cst = const.Declare tensor<2x32x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<2x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<2x32x1x1xf16, {order = #NHWC}> -> tensor<1x2x64x64xf16, {order = #NHWC}>
    %1 = IE.Interpolate(%0)
             {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <FLOOR>, antialias = false,
             pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
             operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 2, 256, 256]
             } : tensor<1x2x64x64xf16, {order = #NHWC}> -> tensor<1x2x256x256xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x2x256x256xf16, {order = #NHWC}> -> tensor<1x2x256x256xf16>
    return %2 : tensor<1x2x256x256xf16>

    //CHECK:      [[CONV:%.+]] = IE.Convolution
    //CHECK-SAME: tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<2x32x1x1xf16, {order = #NHWC}> -> tensor<1x2x64x64xf16, {order = #NHWC}>
    //CHECK:      [[REORDER:%.+]] = IE.Reorder([[CONV]]) {dstOrder = #NCHW}
    //CHECK:      [[INTERPOLATE:%.+]] = IE.Interpolate([[REORDER]])
    //CHECK-SAME: tensor<1x2x64x64xf16> -> tensor<1x2x256x256xf16>
    //CHECK:      return [[INTERPOLATE]] : tensor<1x2x256x256xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NotPropagateReorderToConvBeforeSoftMax
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x64x368x480xf16, {order = #NHWC}>
func.func @NotPropagateReorderToConvBeforeSoftMax(%arg0: tensor<1x64x368x480xf16, {order = #NHWC}>) -> tensor<1x12x368x480xf16> {
    %cst = const.Declare tensor<12x64x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<12x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x368x480xf16, {order = #NHWC}>, tensor<12x64x1x1xf16, {order = #NHWC}> -> tensor<1x12x368x480xf16, {order = #NHWC}>
    %1 = IE.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x12x368x480xf16, {order = #NHWC}> -> tensor<1x12x368x480xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x12x368x480xf16, {order = #NHWC}> -> tensor<1x12x368x480xf16>
    return %2 : tensor<1x12x368x480xf16>

    //CHECK:      [[CONV:%.+]] = IE.Convolution
    //CHECK-SAME: tensor<1x64x368x480xf16, {order = #NHWC}>, tensor<12x64x1x1xf16, {order = #NHWC}> -> tensor<1x12x368x480xf16, {order = #NHWC}>
    //CHECK:      [[SOFTMAX:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64}
    //CHECK:      [[REORDER:%.+]] = IE.Reorder([[SOFTMAX]]) {dstOrder = #NCHW}
    //CHECK:      return [[REORDER]] : tensor<1x12x368x480xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NotPropagateForMultiUsers
func.func @NotPropagateForMultiUsers(%arg0: tensor<1x3x512x512xf16, {order = #NHWC}>) -> (tensor<1x3x512x512xf16>, tensor<1x3x512x512xf16, {order = #NHWC}>) {
    %cst = const.Declare tensor<3x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %1 = IE.Tanh(%0) : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>
    %3 = IE.MaxPool(%1) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    return %2, %3 : tensor<1x3x512x512xf16>, tensor<1x3x512x512xf16, {order = #NHWC}>

    //CHECK-DAG:        [[WEIGHT:%.*]] = const.Declare tensor<3x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:            [[CONV:%.*]] = IE.Convolution(%arg0, [[WEIGHT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:            [[TANH:%.*]] = IE.Tanh([[CONV]]) : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:            [[REORDER:%.*]] = IE.Reorder([[TANH]]) {dstOrder = #NCHW} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>
    //CHECK:            [[POOL:%.*]] = IE.MaxPool([[TANH]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:            return [[REORDER]], [[POOL]]  : tensor<1x3x512x512xf16>, tensor<1x3x512x512xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#CN = affine_map<(d0, d1) -> (d1, d0)>

func.func @NotPropagateViewLike(%arg: tensor<1280x5120xf16>) -> (tensor<1280x5120x1x1xf16, {order = #NHWC}>, tensor<5120x1280xf16, {order = #CN}>) {
    %affine_reshape = IE.AffineReshape(%arg) {dim_mapping = [[0], [1, 2, 3]], shape_value = [1280, 5120, 1, 1]} : tensor<1280x5120xf16> -> tensor<1280x5120x1x1xf16>
    %reorder_0 = IE.Reorder(%affine_reshape) {dstOrder = #NHWC} : tensor<1280x5120x1x1xf16> -> tensor<1280x5120x1x1xf16, {order = #NHWC}>
    %shape_cast = IE.ShapeCast {shape = [5120, 1280]} inputs(%arg : tensor<1280x5120xf16>) -> tensor<5120x1280xf16>
    %reorder_1 = IE.Reorder(%shape_cast) {dstOrder = #CN} : tensor<5120x1280xf16> -> tensor<5120x1280xf16, {order = #CN}>
    return %reorder_0, %reorder_1 : tensor<1280x5120x1x1xf16, {order = #NHWC}>, tensor<5120x1280xf16, {order = #CN}>
}

    // CHECK: func.func @NotPropagateViewLike([[ARG:%.+]]: tensor<1280x5120xf16>) -> (tensor<1280x5120x1x1xf16, {order = #NHWC}>, tensor<5120x1280xf16, {order = #CN}>) {
    // CHECK:     [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[ARG]]) {dim_mapping = {{\[\[}}0], [1, 2, 3]], shape_value = [1280, 5120, 1, 1]} : tensor<1280x5120xf16> -> tensor<1280x5120x1x1xf16>
    // CHECK:     [[REORDER_0:%.+]] = IE.Reorder([[AFFINE_RESHAPE]]) {dstOrder = #NHWC} : tensor<1280x5120x1x1xf16> -> tensor<1280x5120x1x1xf16, {order = #NHWC}>
    // CHECK:     [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [5120, 1280]} inputs([[ARG]] : tensor<1280x5120xf16>) -> tensor<5120x1280xf16>
    // CHECK:     [[REORDER_1:%.+]] = IE.Reorder([[SHAPE_CAST]]) {dstOrder = #CN} : tensor<5120x1280xf16> -> tensor<5120x1280xf16, {order = #CN}>
    // CHECK:     return [[REORDER_0]], [[REORDER_1]]
