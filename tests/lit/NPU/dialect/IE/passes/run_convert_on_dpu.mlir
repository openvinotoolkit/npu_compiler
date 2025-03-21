//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --run-f16-to-f32-convert-on-dpu %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FoldConvertIntoConv
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @FoldConvertIntoConv(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x64xf32, {order = #NHWC}> {
  %cst = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
    : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x2xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x2x128x64xf32, {order = #NHWC}>
  return %1 : tensor<1x2x128x64xf32, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT]], [[CST_WEIGHTS]])
  // CHECK-SAME:        -> tensor<1x2x128x64xf32, {order = #NHWC}>

  // CHECK-NOT: IE.Convert
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FoldConvertIntoDepthConv
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x320x4096x1xf16, {order = #NHWC}>)
func.func @FoldConvertIntoDepthConv(%arg0: tensor<1x320x4096x1xf16, {order = #NHWC}>) -> tensor<1x320x4096x1xf32, {order = #NHWC}> {
  %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
  %0 = IE.GroupConvolution(%arg0, %filter)
    {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    : tensor<1x320x4096x1xf16, {order = #NHWC}>, tensor<320x1x1x1xf16> -> tensor<1x320x4096x1xf16, {order = #NHWC}>
  %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x320x4096x1xf16, {order = #NHWC}> -> tensor<1x320x4096x1xf32, {order = #NHWC}>
  return %1 : tensor<1x320x4096x1xf32, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<320x1x1x1xf16>
  // CHECK:       [[RET:%.+]] = IE.GroupConvolution([[INPUT]], [[CST_WEIGHTS]])
  // CHECK-SAME:    : tensor<1x320x4096x1xf16, {order = #NHWC}>, tensor<320x1x1x1xf16> -> tensor<1x320x4096x1xf32, {order = #NHWC}>

  // CHECK-NOT: IE.Convert
  // CHECK:       return [[RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FoldConvertIntoAvgPool
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @FoldConvertIntoAvgPool(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x64x14x14xf32, {order = #NHWC}> {
  %0 = IE.AvgPool(%arg0) {
      kernel_size = [2, 2],
      pads_begin = [0, 0],
      pads_end = [0, 0],
      rounding_type = #IE.rounding_type<FLOOR>,
      strides = [2, 2]
  } : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x14x14xf16, {order = #NHWC}>
  %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x64x14x14xf16, {order = #NHWC}> -> tensor<1x64x14x14xf32, {order = #NHWC}>
  return %1 : tensor<1x64x14x14xf32, {order = #NHWC}>

  // CHECK:       [[RET:%.+]] = IE.AvgPool([[INPUT]])
  // CHECK-SAME:    : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x14x14xf32, {order = #NHWC}>

  // CHECK-NOT: IE.Convert
  // CHECK:       return [[RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FoldConvertIntoEltwise
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @FoldConvertIntoEltwise(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x64x28x28xf32, {order = #NHWC}> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>
  }
      : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
      -> tensor<1x64x28x28xf16, {order = #NHWC}>
  %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf32, {order = #NHWC}>
  return %1 : tensor<1x64x28x28xf32, {order = #NHWC}>

  // CHECK:       [[RET:%.+]] = IE.Add([[INPUT]], [[INPUT]])
  // CHECK-SAME:    : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf32, {order = #NHWC}>

  // CHECK-NOT: IE.Convert
  // CHECK:       return [[RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFoldConvertIntoConvNotSingleUser
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @NotFoldConvertIntoConvNotSingleUser(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>)
    -> (tensor<1x2x128x64xf32, {order = #NHWC}>, tensor<1x2x128x64xf16, {order = #NHWC}>) {
  %cst = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}> = dense<1.25000e+00> : tensor<2x8x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
    : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x2xf16, {order = #NHWC}>
    -> tensor<1x2x128x64xf16, {order = #NHWC}>
  %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x2x128x64xf32, {order = #NHWC}>
  return %1, %0 : tensor<1x2x128x64xf32, {order = #NHWC}>, tensor<1x2x128x64xf16, {order = #NHWC}>

  // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], %{{.+}})
  // CHECK-SAME:    -> tensor<1x2x128x64xf16, {order = #NHWC}>
  // CHECK-NEXT:  [[RET:%.+]] = IE.Convert([[CONV]]) {dstElemType = f32}
  // CHECK-SAME:    -> tensor<1x2x128x64xf32, {order = #NHWC}>
  // CHECK:       return [[RET]], [[CONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFoldConvertNoParentOp
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x128x64xf16, {order = #NHWC}>)
func.func @NotFoldConvertNoParentOp(%arg0: tensor<1x2x128x64xf16, {order = #NHWC}>)
    -> tensor<1x2x128x64xf32, {order = #NHWC}> {
  %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x2x128x64xf32, {order = #NHWC}>
  return %0 : tensor<1x2x128x64xf32, {order = #NHWC}>

  // CHECK:  [[RET:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32}
  // CHECK-SAME:    -> tensor<1x2x128x64xf32, {order = #NHWC}>
  // CHECK:       return [[RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFoldConvertNotDPUParentOp
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x128x64xf16, {order = #NHWC}>)
func.func @NotFoldConvertNotDPUParentOp(%arg0: tensor<1x2x128x64xf16, {order = #NHWC}>)
    -> tensor<1x2x128x64xf32, {order = #NHWC}> {
  %0 = IE.SoftMax(%arg0) {axisInd = 2} : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x2x128x64xf32, {order = #NHWC}>
  return %1 : tensor<1x2x128x64xf32, {order = #NHWC}>

  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]])
  // CHECK-SAME:    -> tensor<1x2x128x64xf16, {order = #NHWC}>
  // CHECK-NEXT:  [[RET:%.+]] = IE.Convert([[SOFTMAX]]) {dstElemType = f32}
  // CHECK-SAME:    -> tensor<1x2x128x64xf32, {order = #NHWC}>
  // CHECK:       return [[RET]]
}
