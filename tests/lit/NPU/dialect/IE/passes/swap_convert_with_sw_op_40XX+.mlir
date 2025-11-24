//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-convert-with-sw-op %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @SwapConvert
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @SwapConvert(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x1x64x256000xf32> {
  %cst = const.Declare tensor<2000x8x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2000x8x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
    : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2000x8x1x2xf16, {order = #NHWC}> -> tensor<1x2000x128x64xf16, {order = #NHWC}>
  %1 = IE.Reshape(%0) { shape_value = [1, 256000, 1, 64] } : tensor<1x2000x128x64xf16, {order = #NHWC}> -> tensor<1x256000x1x64xf16>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x256000x1x64xf16> -> tensor<1x1x64x256000xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x1x64x256000xf16> -> tensor<1x1x64x256000xf32>
  return %3 : tensor<1x1x64x256000xf32>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2000x8x1x2xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT]], [[CST_WEIGHTS]])
  // CHECK-SAME:        -> tensor<1x2000x128x64xf16, {order = #NHWC}>

  // CHECK:       [[RET:%.+]] = IE.Convert([[CONV_RET]]) {dstElemType = f32}
  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[RET]])
  // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
  // CHECK:       return [[TRANSPOSE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @SmallSizeConv
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @SmallSizeConv(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x1x64x256xf32> {
  %cst = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
    : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x2xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  %1 = IE.Reshape(%0) { shape_value = [1, 256, 1, 64] } : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x256x1x64xf16>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x256x1x64xf16> -> tensor<1x1x64x256xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x1x64x256xf16> -> tensor<1x1x64x256xf32>
  return %3 : tensor<1x1x64x256xf32>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT]], [[CST_WEIGHTS]])
  // CHECK-SAME:        -> tensor<1x2x128x64xf16, {order = #NHWC}>

  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[CONV_RET]])
  // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
  // CHECK:       [[RET:%.+]] = IE.Convert([[TRANSPOSE]]) {dstElemType = f32}
  // CHECK-SAME:    -> tensor<1x1x64x256xf32
  // CHECK:       return [[RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvNotSingleUser
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @ConvNotSingleUser(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>)
    -> (tensor<1x1x64x256xf32>, tensor<1x2x128x64xf16, {order = #NHWC}>) {
  %cst = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}> = dense<1.25000e+00> : tensor<2x8x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
    : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x2xf16, {order = #NHWC}>
    -> tensor<1x2x128x64xf16, {order = #NHWC}>
  %1 = IE.Reshape(%0) { shape_value = [1, 256, 1, 64] } : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x256x1x64xf16>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x256x1x64xf16> -> tensor<1x1x64x256xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x1x64x256xf16> -> tensor<1x1x64x256xf32>
  return %3, %0 : tensor<1x1x64x256xf32>, tensor<1x2x128x64xf16, {order = #NHWC}>

  // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], %{{.+}})
  // CHECK-SAME:    -> tensor<1x2x128x64xf16, {order = #NHWC}>
  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[CONV]])
  // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
  // CHECK:       [[RET:%.+]] = IE.Convert([[TRANSPOSE]]) {dstElemType = f32}
  // CHECK-SAME:    -> tensor<1x1x64x256xf32
  // CHECK:       return [[RET]], [[CONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NonDPUParentOp
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x128x64xf16, {order = #NHWC}>)
func.func @NonDPUParentOp(%arg0: tensor<1x2x128x64xf16, {order = #NHWC}>)
    -> tensor<1x1x64x256xf32> {
  %0 = IE.SoftMax(%arg0) {axisInd = 2} : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  %1 = IE.Reshape(%0) { shape_value = [1, 256, 1, 64] } : tensor<1x2x128x64xf16, {order = #NHWC}> -> tensor<1x256x1x64xf16>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x256x1x64xf16> -> tensor<1x1x64x256xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x1x64x256xf16> -> tensor<1x1x64x256xf32>
  return %3 : tensor<1x1x64x256xf32>

  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]])
  // CHECK-SAME:    -> tensor<1x2x128x64xf16, {order = #NHWC}>
  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SOFTMAX]])
  // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
  // CHECK:       [[RET:%.+]] = IE.Convert([[TRANSPOSE]]) {dstElemType = f32}
  // CHECK-SAME:    -> tensor<1x1x64x256xf32>
  // CHECK:       return [[RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @SwapAddWithConvert
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x16x270x54xsi32, {order = #NHWC}>)
func.func @SwapAddWithConvert(%arg0: tensor<1x16x270x54xsi32, {order = #NHWC}>) -> tensor<1x16x270x54xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<1x1x1x1xsi32, {order = #NHWC}> = dense<1> : tensor<1x1x1x1xsi32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<si32>, #const.Reorder<#NHWC>]
  %0 = IE.Add(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x270x54xsi32, {order = #NHWC}>, tensor<1x1x1x1xsi32, {order = #NHWC}> -> tensor<1x16x270x54xsi32, {order = #NHWC}>
  %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x16x270x54xsi32, {order = #NHWC}> -> tensor<1x16x270x54xf16, {order = #NHWC}>
  return %1 : tensor<1x16x270x54xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1> : tensor<1x1x1x1xsi32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<si32>, #const.Reorder<#NHWC>, #const.CastElemType<f16>]
  // CHECK:       [[CONVERT:%.+]]  = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x16x270x54xsi32, {order = #NHWC}> -> tensor<1x16x270x54xf16, {order = #NHWC}>
  // CHECK:       [[ADD:%.+]] = IE.Add([[CONVERT]], [[CST_WEIGHTS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x270x54xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x270x54xf16, {order = #NHWC}>

  // CHECK:       return [[ADD]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!convDynamicType = tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
!transposeDynamicType = tensor<1x?x1920x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NHWC}>
!convertDynamicType = tensor<1x?x1920x16xf32, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @SwapConvertDynamic
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]>
func.func @SwapConvertDynamic(%arg0: !convDynamicType) -> !convertDynamicType {
  %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    : !convDynamicType, tensor<16x16x1x1xf16, {order = #NHWC}> -> !convDynamicType
  %1 = IE.Transpose(%0) {order_value = #NHWC} : !convDynamicType -> !transposeDynamicType
  %2 = IE.Convert(%1) {dstElemType = f32} : !transposeDynamicType -> !convertDynamicType
  return %2 : !convertDynamicType

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT]], [[CST_WEIGHTS]])
  // CHECK-SAME:        -> tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]>

  // CHECK:       [[RET:%.+]] = IE.Convert([[CONV_RET]]) {dstElemType = f32}
  // CHECK-SAME:        -> tensor<1x16x?x1920xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]>
  // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[RET]])
  // CHECK:       return [[TRANSPOSE]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotSwapParentRankSizeOne
// CHECK-SAME: ([[INPUT:%.+]]: tensor<65536xf16>)
func.func @NotSwapParentRankSizeOne(%arg0: tensor<65536xf16>)
    -> tensor<1x3x64x256xf32> {
  %0 = IE.SoftMax(%arg0) {axisInd = 0} : tensor<65536xf16> -> tensor<65536xf16>
  %1 = IE.Reshape(%0) { shape_value = [1, 256, 3, 64] } : tensor<65536xf16> -> tensor<1x256x3x64xf16>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x256x3x64xf16> -> tensor<1x3x64x256xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x3x64x256xf16> -> tensor<1x3x64x256xf32>
  return %3 : tensor<1x3x64x256xf32>

  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]])
  // CHECK-SAME:    -> tensor<65536xf16>
  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SOFTMAX]])
  // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
  // CHECK:       [[RET:%.+]] = IE.Convert([[TRANSPOSE]]) {dstElemType = f32}
  // CHECK-SAME:    -> tensor<1x3x64x256xf32>
  // CHECK:       return [[RET]]
}
