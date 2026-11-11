//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% enable-auto-padding-odu" --adjust-convolution-shape %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% enable-auto-padding-odu" --run-mem-permute-processing-rewriters="rewriter=adjust-convolution-shape-set" %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:0, {1.000000e+00, 1.100000e+00, 1.200000e+00}>

// CHECK-LABEL: @AdjustQuantizedConvPerAxisFilter9x9
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x1080x1920x!qElemType, {order = #NHWC}>)
func.func @AdjustQuantizedConvPerAxisFilter9x9(%arg0: tensor<1x32x1080x1920x!qElemType, {order = #NHWC}>) -> tensor<1x3x1080x1920x!qElemType, {order = #NHWC}> {
  %weights = const.Declare tensor<3x32x9x9x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<3x32x9x9xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [4, 4], pads_end = [4, 4], strides = [1, 1]}
            : tensor<1x32x1080x1920x!qElemType, {order = #NHWC}>, tensor<3x32x9x9x!qElemType1, {order = #NHWC}> -> tensor<1x3x1080x1920x!qElemType, {order = #NHWC}>

  return %result : tensor<1x3x1080x1920x!qElemType, {order = #NHWC}>

  // CHECK-DAG:  [[FILTER_CST:%.+]] = const.Declare tensor<48x512x9x3x!qElemType1, {order = #NHWC}>
  // CHECK:      [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 512, 1080, 120]}
  // CHECK-SAME: inputs([[ARG0]] : tensor<1x32x1080x1920x!qElemType, {order = #NHWC}>) -> tensor<1x512x1080x120x!qElemType, {order = #NHWC}>
  // CHECK:      [[CONV:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]])
  // CHECK-SAME: -> tensor<1x48x1080x120x!qElemType, {order = #NHWC}>
  // CHECK:      [[CONV_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]}
  // CHECK-SAME: inputs([[CONV]] : tensor<1x48x1080x120x!qElemType, {order = #NHWC}>) -> tensor<1x3x1080x1920x!qElemType, {order = #NHWC}>
  // CHECK:      return [[CONV_CAST]] : tensor<1x3x1080x1920x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:0, {1.000000e+00, 1.100000e+00, 1.200000e+00}>

// CHECK-LABEL: @NotAdjustQuantizedConvPerAxisFilter3x3
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x320x320x!qElemType, {order = #NHWC}>)
func.func @NotAdjustQuantizedConvPerAxisFilter3x3(%arg0: tensor<1x3x320x320x!qElemType, {order = #NHWC}>) -> tensor<1x3x320x320x!qElemType, {order = #NHWC}> {
  %weights = const.Declare tensor<3x3x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
            : tensor<1x3x320x320x!qElemType, {order = #NHWC}>, tensor<3x3x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x3x320x320x!qElemType, {order = #NHWC}>

  return %result : tensor<1x3x320x320x!qElemType, {order = #NHWC}>

  // CHECK:      [[WEIGHTS:%.+]] = const.Declare tensor<3x3x3x3x!qElemType1, {order = #NHWC}>
  // CHECK:      [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WEIGHTS]])
  // CHECK-SAME: -> tensor<1x3x320x320x!qElemType, {order = #NHWC}>
  // CHECK:      return [[CONV]] : tensor<1x3x320x320x!qElemType, {order = #NHWC}>
}
