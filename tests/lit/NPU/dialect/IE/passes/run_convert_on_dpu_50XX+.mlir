//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --run-f16-to-f32-convert-on-dpu %s | FileCheck %s
// REQUIRES: arch-NPU50XX


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FoldConvertIntoMaxPool
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @FoldConvertIntoMaxPool(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x64x14x14xf32, {order = #NHWC}> {
  %0 = IE.MaxPool(%arg0) {
      kernel_size = [2, 2],
      pads_begin = [0, 0],
      pads_end = [0, 0],
      rounding_type = #IE.rounding_type<FLOOR>,
      strides = [2, 2]
  } : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x14x14xf16, {order = #NHWC}>
  %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x64x14x14xf16, {order = #NHWC}> -> tensor<1x64x14x14xf32, {order = #NHWC}>
  return %1 : tensor<1x64x14x14xf32, {order = #NHWC}>

  // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[INPUT]])
  // CHECK-SAME:    : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x14x14xf32, {order = #NHWC}>
  // CHECK-NOT:  IE.Convert

  // CHECK:       return [[MAXPOOL]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FoldConvertIntoEltwiseWithClamp
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @FoldConvertIntoEltwiseWithClamp(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x64x28x28xf32, {order = #NHWC}> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.Clamp<min = 0.000000e+00 : f64, max = 6.000000e+00 : f64>
  }
      : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
      -> tensor<1x64x28x28xf16, {order = #NHWC}>
  %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf32, {order = #NHWC}>
  return %1 : tensor<1x64x28x28xf32, {order = #NHWC}>

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]])
  // CHECK-SAME:    : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
  // CHECK-SAME:    -> tensor<1x64x28x28xf32, {order = #NHWC}>
  // CHECK-NOT:   IE.Convert
  // CHECK:       return [[ADD]]
}

// -----

!qElemType = !quant.uniform<i8:f16, -0.078737745098039214>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FoldConvertIntoConvQuantInput
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x8x128x128x!qElemType, {order = #NHWC}>)
func.func @FoldConvertIntoConvQuantInput(%arg0: tensor<1x8x128x128x!qElemType, {order = #NHWC}>) -> tensor<1x2x128x64xf32, {order = #NHWC}> {
  %cst = const.Declare tensor<2x8x1x2x!qElemType, {order = #NHWC}> = dense<125>
    : tensor<2x8x1x2xui8>, [#const.Reorder<#NHWC>, #const.Quantize<!qElemType>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
    : tensor<1x8x128x128x!qElemType, {order = #NHWC}>, tensor<2x8x1x2x!qElemType, {order = #NHWC}>
    -> tensor<1x2x128x64xf16, {order = #NHWC}>
  %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x2x128x64xf32, {order = #NHWC}>
  return %1 : tensor<1x2x128x64xf32, {order = #NHWC}>

  // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], %{{.+}})
  // CHECK-SAME:    : tensor<1x8x128x128x!qElemType, {order = #NHWC}>, tensor<2x8x1x2x!qElemType, {order = #NHWC}>
  // CHECK-SAME:    -> tensor<1x2x128x64xf32, {order = #NHWC}>
  // CHECK-NOT:   IE.Convert
  // CHECK:       return [[CONV]]
}
