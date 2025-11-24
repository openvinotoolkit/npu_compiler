//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-convolution-shape %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FoldStrideIntoKernel
func.func @FoldStrideIntoKernel(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]} : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x2xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  return %0 : tensor<1x2x128x64xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x16x1x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT:%.+]] = IE.ShapeCast {shape = [1, 16, 128, 64]}
  // CHECK-SAME:      inputs(%arg0 : tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x16x128x64xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT]], [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldStrideIntoKernelForDifferentKXGreaterStride
func.func @NotFoldStrideIntoKernelForDifferentKXGreaterStride(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<2x8x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x4xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 2], strides = [1, 2]} : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x4xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
    return %0 : tensor<1x2x128x64xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x8x1x4xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution(%arg0, [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 2], strides = [1, 2]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldStrideIntoKernelForWmodStideNone0
func.func @NotFoldStrideIntoKernelForWmodStideNone0(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x43xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<2x8x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]} : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x3xf16, {order = #NHWC}> -> tensor<1x2x128x43xf16, {order = #NHWC}>
    return %0 : tensor<1x2x128x43xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x8x1x3xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution(%arg0, [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldStrideIntoKernelWhenChannelAligned
func.func @NotFoldStrideIntoKernelWhenChannelAligned(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x16x128x43xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<16x16x1x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<16x16x1x3xf16, {order = #NHWC}> -> tensor<1x16x128x43xf16, {order = #NHWC}>
    return %0 : tensor<1x16x128x43xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x3xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution(%arg0, [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShape
func.func @AdjustConvolutionShape(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<48x48x1x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeNoneSplatBias
func.func @AdjustConvolutionShapeNoneSplatBias(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x3x1x1xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}>, tensor<1x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x48x1x1xf16, {order = #NHWC}> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Reorder<#NHWC>, #const.Broadcast<1 : i64, 48 : i64>, #const.Reshape<[1, 48, 1, 1]>]
  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<48x48x1x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPadingRight
func.func @AdjustConvolutionShapeWithKXGreat1andPadingRight(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x2xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<48x48x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPaddingLeft
func.func @AdjustConvolutionShapeWithKXGreat1andPaddingLeft(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x2xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<48x48x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPaddingLeftRight
func.func @AdjustConvolutionShapeWithKXGreat1andPaddingLeftRight(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x3xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[FILTER_CST:%.+]] =  const.Declare tensor<48x48x1x3xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPaddingStride
func.func @AdjustConvolutionShapeWithKXGreat1andPaddingStride(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 0], strides = [1, 2]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x3xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x960xf16, {order = #NHWC}>

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[FILTER_CST:%.+]] =  const.Declare tensor<48x96x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 96, 1080, 60]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x96x1080x60xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 960]} inputs([[CONV_RET]] : tensor<1x48x1080x60xf16, {order = #NHWC}>) -> tensor<1x3x1080x960xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotAdjustConvolutionShapeWhenTensorFitCMX
func.func @NotAdjustConvolutionShapeWhenTensorFitCMX(%arg0: tensor<1x12x2x8xf16, {order = #NHWC}>) -> tensor<1x2x2x8xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x12x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x12x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x12x2x8xf16, {order = #NHWC}>, tensor<2x12x1x2xf16, {order = #NHWC}> -> tensor<1x2x2x8xf16, {order = #NHWC}>
  return %0 : tensor<1x2x2x8xf16, {order = #NHWC}>

  // CHECK:       [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<2x12x1x2xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution(%arg0, [[CST_WEIGHTS_0]])
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat3andPaddingBeginStride
func.func @AdjustConvolutionShapeWithKXGreat3andPaddingBeginStride(%arg0: tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<4x4x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 2], pads_end = [0, 0], strides = [1, 2]} : tensor<1x4x1080x1920xf16, {order = #NHWC}>, tensor<4x4x1x4xf16, {order = #NHWC}> -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x4x1080x960xf16, {order = #NHWC}>

  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<16x32x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 32, 1080, 240]} inputs(%arg0 : tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x32x1080x240xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 4, 1080, 960]} inputs([[CONV_RET]] : tensor<1x16x1080x240xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat3andPaddingEndStride
func.func @AdjustConvolutionShapeWithKXGreat3andPaddingEndStride(%arg0: tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<4x4x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 2], strides = [1, 2]} : tensor<1x4x1080x1920xf16, {order = #NHWC}>, tensor<4x4x1x4xf16, {order = #NHWC}> -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x4x1080x960xf16, {order = #NHWC}>

  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<16x32x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 32, 1080, 240]} inputs(%arg0 : tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x32x1080x240xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 4, 1080, 960]} inputs([[CONV_RET]] : tensor<1x16x1080x240xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat3andPaddingBeginEndStride
func.func @AdjustConvolutionShapeWithKXGreat3andPaddingBeginEndStride(%arg0: tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<4x4x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 2]} : tensor<1x4x1080x1920xf16, {order = #NHWC}>, tensor<4x4x1x4xf16, {order = #NHWC}> -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x4x1080x960xf16, {order = #NHWC}>

  // CHECK-DAG:   [[FILTER_CST:%.+]] = const.Declare tensor<16x32x1x3xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 32, 1080, 240]} inputs(%arg0 : tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x32x1080x240xf16, {order = #NHWC}>
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

  //CHECK-DAG:   [[VAL0:%.*]] = const.Declare tensor<32x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.Reorder<#NHWC>]
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

    // CHECK-DAG:  [[NEW_FILTER:%.+]] = const.Declare tensor<48x96x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:  [[CST_PAD:%.+]] = const.Declare tensor<1x3x320x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x3x320x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:  [[CONCAT_PAD:%.+]] = IE.Concat([[INPUT_DATA]], [[CST_PAD]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 639]]}
    // CHECK:           tensor<1x3x320x639xf16, {order = #NHWC}>, tensor<1x3x320x1xf16, {order = #NHWC}> -> tensor<1x3x320x640xf16, {order = #NHWC}>
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
    // CHECK:          tensor<1x3x320x638xf16, {order = #NHWC}>, tensor<1x3x320x2xf16, {order = #NHWC}> -> tensor<1x3x320x640xf16, {order = #NHWC}>
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

// CHECK: func.func @AdjustConvolutionShapeWithAlginedICButUnalignedOC([[INPUT_DATA:%.+]]: tensor<1x4x512x512xf16, {order = #NHWC}>)
func.func @AdjustConvolutionShapeWithAlginedICButUnalignedOC(%arg0: tensor<1x4x512x512xf16, {order = #NHWC}>) -> tensor<1x4x256x256xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<4x4x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<4x4x2x2xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]}
                : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<4x4x2x2xf16, {order = #NHWC}> -> tensor<1x4x256x256xf16, {order = #NHWC}>


  return %result : tensor<1x4x256x256xf16, {order = #NHWC}>

  // CHECK-DAG:    [[CONCAT:%.+]] = const.Declare tensor<16x32x2x1xf16, {order = #NHWC}>
  // CHECK:        [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 8, 512, 256]} inputs([[INPUT_DATA]] : tensor<1x4x512x512xf16, {order = #NHWC}>) -> tensor<1x8x512x256xf16, {order = #NHWC}>
  // CHECK:        [[SHAPE_CAST_2:%.+]] = IE.ShapeCast {shape = [1, 32, 512, 64]} inputs([[SHAPE_CAST_0]] : tensor<1x8x512x256xf16, {order = #NHWC}>) -> tensor<1x32x512x64xf16, {order = #NHWC}>

  // CHECK:        [[CONV:%.+]] = IE.Convolution([[SHAPE_CAST_2]], [[CONCAT]]) {
  // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]}
  // CHECK-SAME:      : tensor<1x32x512x64xf16, {order = #NHWC}>, tensor<16x32x2x1xf16, {order = #NHWC}> -> tensor<1x16x256x64xf16, {order = #NHWC}>

  // CHECK:        [[SHAPE_CAST_3:%.+]] = IE.ShapeCast {shape = [1, 4, 256, 256]} inputs([[CONV]] : tensor<1x16x256x64xf16, {order = #NHWC}>) -> tensor<1x4x256x256xf16, {order = #NHWC}>

  // CHECK:        return [[SHAPE_CAST_3]] : tensor<1x4x256x256xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @SkipShapeAdjustmentWithConsideringAdjacentConvLayers([[INPUT_DATA:%.+]]: tensor<1x128x80x80xf16, {order = #NHWC}>)
func.func @SkipShapeAdjustmentWithConsideringAdjacentConvLayers(%arg0: tensor<1x128x80x80xf16, {order = #NHWC}>) -> tensor<1x128x80x80xf16, {order = #NHWC}> {
  %weights_1 = const.Declare tensor<88x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<88x128x1x1xf16>, [#const.Reorder<#NHWC>]
  %weights_2 = const.Declare tensor<88x88x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<88x88x3x3xf16>, [#const.Reorder<#NHWC>]
  %weights_3 = const.Declare tensor<128x88x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<128x88x1x1xf16>, [#const.Reorder<#NHWC>]

  %conv_1 = IE.Convolution(%arg0, %weights_1) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]}
                : tensor<1x128x80x80xf16, {order = #NHWC}>, tensor<88x128x1x1xf16, {order = #NHWC}> -> tensor<1x88x80x80xf16, {order = #NHWC}>
  %conv_2 = IE.Convolution(%conv_1, %weights_2) {
              dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.Relu<>, strides = [1, 1]}
                : tensor<1x88x80x80xf16, {order = #NHWC}>, tensor<88x88x3x3xf16, {order = #NHWC}> -> tensor<1x88x80x80xf16, {order = #NHWC}>
  %conv_3 = IE.Convolution(%conv_2, %weights_3) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
                : tensor<1x88x80x80xf16, {order = #NHWC}>, tensor<128x88x1x1xf16, {order = #NHWC}> -> tensor<1x128x80x80xf16, {order = #NHWC}>

  return %conv_3 : tensor<1x128x80x80xf16, {order = #NHWC}>

  // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<88x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<88x128x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[WEIGHTS_2:%.+]] = const.Declare tensor<88x88x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<88x88x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[WEIGHTS_3:%.+]] = const.Declare tensor<128x88x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x88x1x1xf16>, [#const.Reorder<#NHWC>]

  // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[INPUT_DATA]], [[WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x128x80x80xf16, {order = #NHWC}>, tensor<88x128x1x1xf16, {order = #NHWC}> -> tensor<1x88x80x80xf16, {order = #NHWC}>
  // CHECK:       [[CONV_2:%.+]] = IE.Convolution([[CONV_1]], [[WEIGHTS_2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x88x80x80xf16, {order = #NHWC}>, tensor<88x88x3x3xf16, {order = #NHWC}> -> tensor<1x88x80x80xf16, {order = #NHWC}>
  // CHECK:       [[CONV_3:%.+]] = IE.Convolution([[CONV_2]], [[WEIGHTS_3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x88x80x80xf16, {order = #NHWC}>, tensor<128x88x1x1xf16, {order = #NHWC}> -> tensor<1x128x80x80xf16, {order = #NHWC}>

  // CHECK:       return [[CONV_3]] : tensor<1x128x80x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @NotAdjustConvWithAsymmetricStrides
// CHECK-SAME:  [[INPUT_DATA:%.+]]: tensor<1x1x1x256xf16, {order = #NHWC}>
func.func @NotAdjustConvWithAsymmetricStrides(%arg0: tensor<1x1x1x256xf16, {order = #NHWC}>) -> tensor<1x64x1x64xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<64x1x1x4xf16, {order = #NHWC}> = dense<1.0> : tensor<64x1x1x4xf16>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 4]}
            : tensor<1x1x1x256xf16, {order = #NHWC}>, tensor<64x1x1x4xf16, {order = #NHWC}> -> tensor<1x64x1x64xf16, {order = #NHWC}>

  return %result : tensor<1x64x1x64xf16, {order = #NHWC}>

  // CHECK:       [[CST:%.+]] = const.Declare tensor<64x1x1x4xf16, {order = #NHWC}>
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT_DATA]], [[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 4]} :
  // CHECK-SAME:  tensor<1x1x1x256xf16, {order = #NHWC}>,
  // CHECK-SAME:  tensor<64x1x1x4xf16, {order = #NHWC}> -> tensor<1x64x1x64xf16, {order = #NHWC}>
  // CHECK:       return [[CONV]]
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


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK-LABEL: @ConvertGroupConv
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x64x128xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>, [[ARG2:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>)
func.func @ConvertGroupConv(%arg0: tensor<1x1x64x128xf16, {order = #NHWC}>, %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x1x1x1xf16, {order = #NHWC}>) -> tensor<1x1x64x128x!qElemType, {order = #NHWC}> {
  %0 = IE.GroupConvolution(%arg0, %arg1, %arg2) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x64x128xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x64x128x!qElemType, {order = #NHWC}>
  return %0 : tensor<1x1x64x128x!qElemType, {order = #NHWC}>

  // CHECK-DAG:   [[REPEAT_CONST:%.+]] = const.Declare tensor<4xsi32> = dense<[16, 1, 1, 1]> : tensor<4xsi32>
  // CHECK:       [[IN_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 16, 64, 8]} inputs([[ARG0]] : tensor<1x1x64x128xf16, {order = #NHWC}>) -> tensor<1x16x64x8xf16, {order = #NHWC}>
  // CHECK:       [[FILTER_TILE:%.+]] = IE.Tile([[ARG1]], [[REPEAT_CONST]]) : tensor<1x1x1x1xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<16x1x1x1xf16, {order = #NHWC}>
  // CHECK:       [[BIAS_TILE:%.+]] = IE.Tile([[ARG2]], [[REPEAT_CONST]]) : tensor<1x1x1x1xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<16x1x1x1xf16, {order = #NHWC}>
  // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[IN_SHAPECAST]], [[FILTER_TILE]], [[BIAS_TILE]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x64x8xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x64x8x!qElemType, {order = #NHWC}>
  // CHECK:       [[OUT_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 1, 64, 128]} inputs([[GROUPCONV]] : tensor<1x16x64x8x!qElemType, {order = #NHWC}>) -> tensor<1x1x64x128x!qElemType, {order = #NHWC}>
  // CHECK:       return [[OUT_SHAPECAST]]  : tensor<1x1x64x128x!qElemType, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK-LABEL: @ConvertGroupConvAdjustHW
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x220x200xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>, [[ARG2:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>)
func.func @ConvertGroupConvAdjustHW(%arg0: tensor<1x1x220x200xf16, {order = #NHWC}>, %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x1x1x1xf16, {order = #NHWC}>) -> tensor<1x1x220x200x!qElemType, {order = #NHWC}> {
  %0 = IE.GroupConvolution(%arg0, %arg1, %arg2) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x220x200xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x220x200x!qElemType, {order = #NHWC}>
  return %0 : tensor<1x1x220x200x!qElemType, {order = #NHWC}>

  // CHECK-DAG:   [[REPEAT_CONST:%.+]] = const.Declare tensor<4xsi32> = dense<[16, 1, 1, 1]> : tensor<4xsi32>
  // CHECK:       [[IN_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 16, 55, 50]} inputs([[ARG0]] : tensor<1x1x220x200xf16, {order = #NHWC}>) -> tensor<1x16x55x50xf16, {order = #NHWC}>
  // CHECK:       [[FILTER_TILE:%.+]] = IE.Tile([[ARG1]], [[REPEAT_CONST]]) : tensor<1x1x1x1xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<16x1x1x1xf16, {order = #NHWC}>
  // CHECK:       [[BIAS_TILE:%.+]] = IE.Tile([[ARG2]], [[REPEAT_CONST]]) : tensor<1x1x1x1xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<16x1x1x1xf16, {order = #NHWC}>
  // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[IN_SHAPECAST]], [[FILTER_TILE]], [[BIAS_TILE]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x55x50xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x55x50x!qElemType, {order = #NHWC}>
  // CHECK:       [[OUT_SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 1, 220, 200]} inputs([[GROUPCONV]] : tensor<1x16x55x50x!qElemType, {order = #NHWC}>) -> tensor<1x1x220x200x!qElemType, {order = #NHWC}>
  // CHECK:       return [[OUT_SHAPECAST]]  : tensor<1x1x220x200x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
!qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK-LABEL: @NotConvertGroupConvForODUPermute
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x64x128xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>, [[ARG2:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>)
func.func @NotConvertGroupConvForODUPermute(%arg0: tensor<1x1x64x128xf16, {order = #NHWC}>, %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x1x1x1xf16, {order = #NHWC}>) -> tensor<1x1x64x128x!qElemType, {order = #NWHC}> {
  %0 = IE.GroupConvolution(%arg0, %arg1, %arg2) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x64x128xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x64x128x!qElemType, {order = #NWHC}>
  return %0 : tensor<1x1x64x128x!qElemType, {order = #NWHC}>

  // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution
  // CHECK:       return [[GROUPCONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK-LABEL: @NotConvertGroupConvForNonSingleDataKernel
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x64x128xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x1x1x2xf16, {order = #NHWC}>, [[ARG2:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>)
func.func @NotConvertGroupConvForNonSingleDataKernel(%arg0: tensor<1x1x64x128xf16, {order = #NHWC}>, %arg1: tensor<1x1x1x2xf16, {order = #NHWC}>, %arg2: tensor<1x1x1x1xf16, {order = #NHWC}>) -> tensor<1x1x64x127x!qElemType, {order = #NHWC}> {
  %0 = IE.GroupConvolution(%arg0, %arg1, %arg2) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x64x128xf16, {order = #NHWC}>, tensor<1x1x1x2xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x64x127x!qElemType, {order = #NHWC}>
  return %0 : tensor<1x1x64x127x!qElemType, {order = #NHWC}>

  // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution
  // CHECK:       return [[GROUPCONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK-LABEL: @NotConvertGroupConvForPadding
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x64x128xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>, [[ARG2:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>)
func.func @NotConvertGroupConvForPadding(%arg0: tensor<1x1x64x128xf16, {order = #NHWC}>, %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x1x1x1xf16, {order = #NHWC}>) -> tensor<1x1x64x129x!qElemType, {order = #NHWC}> {
  %0 = IE.GroupConvolution(%arg0, %arg1, %arg2) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x64x128xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x64x129x!qElemType, {order = #NHWC}>
  return %0 : tensor<1x1x64x129x!qElemType, {order = #NHWC}>

  // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution
  // CHECK:       return [[GROUPCONV]]
}
