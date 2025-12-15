//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% enable-auto-padding-odu" --adjust-convolution-shape %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FoldStrideIntoKernel
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @FoldStrideIntoKernel(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]} : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x2xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  return %0 : tensor<1x2x128x64xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x16x1x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT:%.+]] = IE.ShapeCast {shape = [1, 16, 128, 64]}
  // CHECK-SAME:      inputs([[ARG0]] : tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x16x128x64xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT]], [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldStrideIntoKernelForDifferentKXGreaterStride
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @NotFoldStrideIntoKernelForDifferentKXGreaterStride(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<2x8x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x4xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 2], strides = [1, 2]} : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x4xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
    return %0 : tensor<1x2x128x64xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x8x1x4xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[ARG0]], [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 2], strides = [1, 2]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldStrideIntoKernelForWmodStideNone0
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @NotFoldStrideIntoKernelForWmodStideNone0(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x43xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<2x8x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]} : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x3xf16, {order = #NHWC}> -> tensor<1x2x128x43xf16, {order = #NHWC}>
    return %0 : tensor<1x2x128x43xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x8x1x3xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[ARG0]], [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldStrideIntoKernelWhenChannelAligned
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x16x128x128xf16, {order = #NHWC}>)
func.func @NotFoldStrideIntoKernelWhenChannelAligned(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x16x128x43xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<16x16x1x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<16x16x1x3xf16, {order = #NHWC}> -> tensor<1x16x128x43xf16, {order = #NHWC}>
    return %0 : tensor<1x16x128x43xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x3xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[ARG0]], [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShape
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x1080x1920xf16, {order = #NHWC}>)
func.func @AdjustConvolutionShape(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<48x48x1x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs([[ARG0]] : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeNoneSplatBias
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x1080x1920xf16, {order = #NHWC}>)
func.func @AdjustConvolutionShapeNoneSplatBias(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x3x1x1xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}>, tensor<1x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x48x1x1xf16, {order = #NHWC}> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Reorder<#NHWC>, #const.Broadcast<1 : i64, 48 : i64>, #const.Reshape<[1, 48, 1, 1]>]
  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<48x48x1x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs([[ARG0]] : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPaddingRight
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x1080x1920xf16, {order = #NHWC}>)
func.func @AdjustConvolutionShapeWithKXGreat1andPaddingRight(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x2xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<48x48x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs([[ARG0]] : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPaddingLeft
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x1080x1920xf16, {order = #NHWC}>)
func.func @AdjustConvolutionShapeWithKXGreat1andPaddingLeft(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x2xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<48x48x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs([[ARG0]] : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPaddingLeftRight
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x1080x1920xf16, {order = #NHWC}>)
func.func @AdjustConvolutionShapeWithKXGreat1andPaddingLeftRight(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x3xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<48x48x1x3xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs([[ARG0]] : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPaddingStride
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x1080x1920xf16, {order = #NHWC}>)
func.func @AdjustConvolutionShapeWithKXGreat1andPaddingStride(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 0], strides = [1, 2]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x3xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x960xf16, {order = #NHWC}>

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<48x96x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 96, 1080, 60]} inputs([[ARG0]] : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x96x1080x60xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 960]} inputs([[CONV_RET]] : tensor<1x48x1080x60xf16, {order = #NHWC}>) -> tensor<1x3x1080x960xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotAdjustConvolutionShapeWhenTensorFitCMX
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x12x2x8xf16, {order = #NHWC}>)
func.func @NotAdjustConvolutionShapeWhenTensorFitCMX(%arg0: tensor<1x12x2x8xf16, {order = #NHWC}>) -> tensor<1x2x2x8xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x12x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x12x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x12x2x8xf16, {order = #NHWC}>, tensor<2x12x1x2xf16, {order = #NHWC}> -> tensor<1x2x2x8xf16, {order = #NHWC}>
  return %0 : tensor<1x2x2x8xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<2x12x1x2xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[ARG0]], [[CST_WEIGHTS_0]])
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat3andPaddingBeginStride
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x4x1080x1920xf16, {order = #NHWC}>
func.func @AdjustConvolutionShapeWithKXGreat3andPaddingBeginStride(%arg0: tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<4x4x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 2], pads_end = [0, 0], strides = [1, 2]} : tensor<1x4x1080x1920xf16, {order = #NHWC}>, tensor<4x4x1x4xf16, {order = #NHWC}> -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x4x1080x960xf16, {order = #NHWC}>

  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<16x32x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 32, 1080, 240]} inputs([[ARG0]] : tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x32x1080x240xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 4, 1080, 960]} inputs([[CONV_RET]] : tensor<1x16x1080x240xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>

// CHECK: func.func @NotAdjustConvWithMixedPrecisionFloatOutputQuantInput([[INPUT_DATA:%.+]]: tensor<1x3x320x320x!qElemType, {order = #NHWC}>)
func.func @NotAdjustConvWithMixedPrecisionFloatOutputQuantInput(%arg0: tensor<1x3x320x320x!qElemType, {order = #NHWC}>) -> tensor<1x32x160x160xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<32x3x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<32x3x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]}
            : tensor<1x3x320x320x!qElemType, {order = #NHWC}>, tensor<32x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>

  return %result : tensor<1x32x160x160xf16, {order = #NHWC}>

  //CHECK:       [[VAL0:%.*]] = const.Declare tensor<32x3x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  //CHECK:       [[VAL1:%.*]] = IE.Convolution([[INPUT_DATA]], [[VAL0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x320x320x!qElemType, {order = #NHWC}>, tensor<32x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>
  //CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>

// CHECK: func.func @NotAdjustConvWithMixedPrecisionFloatInputQuantWeights([[INPUT_DATA:%.+]]: tensor<1x3x320x320xf16, {order = #NHWC}>)
func.func @NotAdjustConvWithMixedPrecisionFloatInputQuantWeights(%arg0: tensor<1x3x320x320xf16, {order = #NHWC}>) -> tensor<1x32x160x160xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<32x3x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<32x3x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]}
            : tensor<1x3x320x320xf16, {order = #NHWC}>, tensor<32x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>

  return %result : tensor<1x32x160x160xf16, {order = #NHWC}>

  //CHECK:       [[VAL0:%.*]] = const.Declare tensor<32x3x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  //CHECK:       [[VAL1:%.*]] = IE.Convolution([[INPUT_DATA]], [[VAL0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x320x320xf16, {order = #NHWC}>, tensor<32x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>
  //CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>

// CHECK: func.func @NotAdjustConvWithMixedPrecisionFloatInputQuantOutput([[INPUT_DATA:%.+]]: tensor<1x3x320x320xf16, {order = #NHWC}>)
func.func @NotAdjustConvWithMixedPrecisionFloatInputQuantOutput(%arg0: tensor<1x3x320x320xf16, {order = #NHWC}>) -> tensor<1x32x160x160x!qElemType, {order = #NHWC}> {
  %weights = const.Declare tensor<32x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x3x3x3xf16>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]}
            : tensor<1x3x320x320xf16, {order = #NHWC}>, tensor<32x3x3x3xf16, {order = #NHWC}> -> tensor<1x32x160x160x!qElemType, {order = #NHWC}>

  return %result : tensor<1x32x160x160x!qElemType, {order = #NHWC}>

  //CHECK:       [[VAL0:%.*]] = const.Declare tensor<32x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.Reorder<#NHWC>]
  //CHECK:       [[VAL1:%.*]] = IE.Convolution([[INPUT_DATA]], [[VAL0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x320x320xf16, {order = #NHWC}>, tensor<32x3x3x3xf16, {order = #NHWC}> -> tensor<1x32x160x160x!qElemType, {order = #NHWC}>
  //CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @Adjust1x1ConvUnalignedChannelByPadding([[INPUT_DATA:%.+]]: tensor<1x3x320x639xf16, {order = #NHWC}>)
func.func @Adjust1x1ConvUnalignedChannelByPadding(%arg0: tensor<1x3x320x639xf16, {order = #NHWC}>) -> tensor<1x3x320x320xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
            : tensor<1x3x320x639xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x320x320xf16, {order = #NHWC}>

  return %result : tensor<1x3x320x320xf16, {order = #NHWC}>

    // CHECK-DAG:    [[NEW_FILTER:%.+]] = const.Declare tensor<48x96x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:    [[CST_PAD:%.+]] = const.Declare tensor<1x3x320x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x3x320x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:    [[CONCAT_PAD:%.+]] = IE.Concat([[INPUT_DATA]], [[CST_PAD]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 639]]}
    // CHECK:      tensor<1x3x320x639xf16, {order = #NHWC}>, tensor<1x3x320x1xf16, {order = #NHWC}> -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:    [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 96, 320, 20]} inputs([[CONCAT_PAD]] : tensor<1x3x320x640xf16, {order = #NHWC}>) -> tensor<1x96x320x20xf16, {order = #NHWC}>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[SHAPE_CAST_IN]], [[NEW_FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x320x20xf16, {order = #NHWC}>, tensor<48x96x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x20xf16, {order = #NHWC}>
    // CHECK:    [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 320, 320]} inputs([[CONV]] : tensor<1x48x320x20xf16, {order = #NHWC}>) -> tensor<1x3x320x320xf16, {order = #NHWC}>
    // CHECK:    return [[SHAPE_CAST_OUT]] : tensor<1x3x320x320xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @Adjust1x1ConvUnalignedChannelByPadding2([[INPUT_DATA:%.+]]: tensor<1x3x320x638xf16, {order = #NHWC}>)
func.func @Adjust1x1ConvUnalignedChannelByPadding2(%arg0: tensor<1x3x320x638xf16, {order = #NHWC}>) -> tensor<1x3x320x160xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 4]}
            : tensor<1x3x320x638xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x320x160xf16, {order = #NHWC}>

  return %result : tensor<1x3x320x160xf16, {order = #NHWC}>

    // CHECK-DAG:    [[NEW_FILTER:%.+]] = const.Declare tensor<48x192x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:    [[CST_PAD:%.+]] = const.Declare tensor<1x3x320x2xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x3x320x2xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:    [[CONCAT_PAD:%.+]] = IE.Concat([[INPUT_DATA]], [[CST_PAD]])
    // CHECK{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 638]]}
    // CHECK:       tensor<1x3x320x638xf16, {order = #NHWC}>, tensor<1x3x320x2xf16, {order = #NHWC}> -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:    [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 192, 320, 10]} inputs([[CONCAT_PAD]] : tensor<1x3x320x640xf16, {order = #NHWC}>) -> tensor<1x192x320x10xf16, {order = #NHWC}>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[SHAPE_CAST_IN]], [[NEW_FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x192x320x10xf16, {order = #NHWC}>, tensor<48x192x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x10xf16, {order = #NHWC}>
    // CHECK:    [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 320, 160]} inputs([[CONV]] : tensor<1x48x320x10xf16, {order = #NHWC}>) -> tensor<1x3x320x160xf16, {order = #NHWC}>
    // CHECK:    return [[SHAPE_CAST_OUT]] : tensor<1x3x320x160xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @NotAdjust1x1ConvPadNumGreaterThanStride([[INPUT_DATA:%.+]]: tensor<1x3x320x635xf16, {order = #NHWC}>)
func.func @NotAdjust1x1ConvPadNumGreaterThanStride(%arg0: tensor<1x3x320x635xf16, {order = #NHWC}>) -> tensor<1x3x320x318xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
            : tensor<1x3x320x635xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x320x318xf16, {order = #NHWC}>

  return %result : tensor<1x3x320x318xf16, {order = #NHWC}>

    //CHECK-DAG: [[CST:%.+]] = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK:     [[CONV:%.+]] = IE.Convolution([[INPUT_DATA]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]} : tensor<1x3x320x635xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x320x318xf16, {order = #NHWC}>
    //CHECK:    return [[CONV]] : tensor<1x3x320x318xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @NotAdjustConvCaseWithExtraPadding([[INPUT_DATA:%.+]]: tensor<1x10x10x13xf16, {order = #NHWC}>)
func.func @NotAdjustConvCaseWithExtraPadding(%arg0: tensor<1x10x10x13xf16, {order = #NHWC}>) -> tensor<1x10x10x5xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<10x10x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<10x10x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 3]}
               : tensor<1x10x10x13xf16, {order = #NHWC}>, tensor<10x10x1x1xf16, {order = #NHWC}> -> tensor<1x10x10x5xf16, {order = #NHWC}>

  return %result : tensor<1x10x10x5xf16, {order = #NHWC}>

  //CHECK:    [[WEIGHTS:%.+]] = const.Declare tensor<10x10x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<10x10x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>]
  //CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT_DATA]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 3]} : tensor<1x10x10x13xf16, {order = #NHWC}>, tensor<10x10x1x1xf16, {order = #NHWC}> -> tensor<1x10x10x5xf16, {order = #NHWC}>
  //CHECK:    return [[CONV]] : tensor<1x10x10x5xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @PreserveOutElemTypeFoldStrideIntoKernel
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @PreserveOutElemTypeFoldStrideIntoKernel(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x64xf32, {order = #NHWC}> {
  %cst = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]} : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x2xf16, {order = #NHWC}> -> tensor<1x2x128x64xf32, {order = #NHWC}>
  return %0 : tensor<1x2x128x64xf32, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x16x1x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT:%.+]] = IE.ShapeCast {shape = [1, 16, 128, 64]}
  // CHECK-SAME:      inputs([[ARG0]] : tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x16x128x64xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT]], [[CST_WEIGHTS]])
  // CHECK-SAME:       {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x2x128x64xf32, {order = #NHWC}>
  // CHECK:       return [[CONV_RET]] : tensor<1x2x128x64xf32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @PreserveOutElemTypeAdjustConvolutionShape
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x1080x1920xf16, {order = #NHWC}>)
func.func @PreserveOutElemTypeAdjustConvolutionShape(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf32, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf32, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf32, {order = #NHWC}>

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<48x48x1x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]}
  // CHECK-SAME:      inputs([[ARG0]] : tensor<1x3x1080x1920xf16, {order = #NHWC}>)
  // CHECK-SAME:    -> tensor<1x48x1080x120xf16, {order = #NHWC}>

  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK-SAME:      -> tensor<1x48x1080x120xf32, {order = #NHWC}>

  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]}
  // CHECK-SAME:      inputs([[CONV_RET]] : tensor<1x48x1080x120xf32, {order = #NHWC}>)
  // CHECK-SAME:    -> tensor<1x3x1080x1920xf32, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}
