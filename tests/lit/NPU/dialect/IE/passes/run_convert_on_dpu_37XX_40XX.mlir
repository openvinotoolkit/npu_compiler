//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --run-f16-to-f32-convert-on-dpu %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldConvertIntoMaxPool
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @NotFoldConvertIntoMaxPool(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x64x14x14xf32, {order = #NHWC}> {
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
  // CHECK-SAME:    : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x14x14xf16, {order = #NHWC}>
  // CHECK-NEXT:  [[RET:%.+]] = IE.Convert([[MAXPOOL]])
  // CHECK-SAME:    -> tensor<1x64x14x14xf32, {order = #NHWC}>
  // CHECK:       return [[RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldConvertIntoEltwiseWithClamp
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @NotFoldConvertIntoEltwiseWithClamp(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x64x28x28xf32, {order = #NHWC}> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 6.0, min = 0.0}>
  }
      : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
      -> tensor<1x64x28x28xf16, {order = #NHWC}>
  %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf32, {order = #NHWC}>
  return %1 : tensor<1x64x28x28xf32, {order = #NHWC}>

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]])
  // CHECK-SAME:    : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf16, {order = #NHWC}>
  // CHECK-NEXT:  [[RET:%.+]] = IE.Convert([[ADD]])
  // CHECK-SAME:    -> tensor<1x64x28x28xf32, {order = #NHWC}>
  // CHECK:       return [[RET]]
}

// -----

!qElemType = !quant.uniform<i8:f16, -0.078737745098039214>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFoldConvertIntoConvQuantInput
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x8x128x128x!qElemType, {order = #NHWC}>)
func.func @NotFoldConvertIntoConvQuantInput(%arg0: tensor<1x8x128x128x!qElemType, {order = #NHWC}>) -> tensor<1x2x128x64xf32, {order = #NHWC}> {
  %cst = const.Declare tensor<2x8x1x2x!qElemType, {order = #NHWC}> = dense<125>
    : tensor<2x8x1x2xui8>, [#const.Reorder<#NHWC>, #const.Quantize<!qElemType>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
    : tensor<1x8x128x128x!qElemType, {order = #NHWC}>, tensor<2x8x1x2x!qElemType, {order = #NHWC}>
    -> tensor<1x2x128x64xf16, {order = #NHWC}>
  %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x2x128x64xf32, {order = #NHWC}>
  return %1 : tensor<1x2x128x64xf32, {order = #NHWC}>

  // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], %{{.+}})
  // CHECK-SAME:    : tensor<1x8x128x128x!qElemType, {order = #NHWC}>, tensor<2x8x1x2x!qElemType, {order = #NHWC}>
  // CHECK-SAME:    -> tensor<1x2x128x64xf16, {order = #NHWC}>
  // CHECK-NEXT:  [[RET:%.+]] = IE.Convert([[CONV]]) {dstElemType = f32}
  // CHECK-SAME:    -> tensor<1x2x128x64xf32, {order = #NHWC}>
  // CHECK:       return [[RET]]
}
