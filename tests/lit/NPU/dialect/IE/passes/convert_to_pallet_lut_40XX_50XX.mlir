//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-to-pallet-lut --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010


!actType = !quant.uniform<i8:f16, 5.000000e-01>
!wgtType = !quant.uniform<u2:f16, 2.0:2>

// CHECK: !qElemType = !quant.uniform<i8:f16, 5.000000e-01>
// CHECK: !qElemType1 = !quant.uniform<!quant.quantile<ui2:si8, {-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00}>:f16, 2.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<u2:f16, 2.000000e+00:2>

// CHECK-LABEL: @ConvertInt8ActU2WgtSymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertInt8ActU2WgtSymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<3.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<3.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  // CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<i8:f16, 5.000000e-01>
!wgtType = !quant.uniform<i2:f16, 2.0:0>

// CHECK: !qElemType = !quant.uniform<i8:f16, 5.000000e-01>
// CHECK: !qElemType1 = !quant.uniform<!quant.quantile<ui2:si8, {0.000000e+00,1.000000e+00,-2.000000e+00,-1.000000e+00}>:f16, 2.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<i2:f16, 2.000000e+00>

// CHECK-LABEL: @ConvertInt8ActI2WgtSymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertInt8ActI2WgtSymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  // CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<i8:f16, 5.000000e-01>
!wgtType = !quant.uniform<i2:f16, 2.0:-1>

// CHECK: !qElemType = !quant.uniform<i8:f16, 5.000000e-01>
// CHECK: !qElemType1 = !quant.uniform<!quant.quantile<ui2:si8, {1.000000e+00,2.000000e+00,-1.000000e+00,0.000000e+00}>:f16, 2.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<i2:f16, 2.000000e+00:-1>

// CHECK-LABEL: @ConvertInt8ActI2WgtAsymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertInt8ActI2WgtAsymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  // CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<i8:f16, 5.000000e-01>
!wgtType = !quant.uniform<i2:f16, 2.000000e+00>

// CHECK: !qElemType = !quant.uniform<i8:f16, 5.000000e-01>
// CHECK: !qElemType1 = !quant.uniform<i2:f16, 2.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<!quant.quantile<ui2:si8, {0.000000e+00,1.000000e+00,-2.000000e+00,-1.000000e+00}>:f16, 2.000000e+00>

// CHECK-LABEL: @ConvertInt8ActI2WeightsAsInput
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>, [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType1>)
func.func @ConvertInt8ActI2WeightsAsInput(%arg0: tensor<1x16x16x16x!actType>, %arg1: tensor<16x16x1x1x!wgtType>) -> tensor<1x16x16x16xf16> {
  %wgtDequant = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %wgtDequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  // CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType2} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1x!qElemType2>
  // CHECK: [[WGTDEQUANT:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType2> -> tensor<16x16x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WGTDEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  // CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<i8:f16, 5.000000e-01>
!wgtType = !quant.uniform<u4:f16, 2.0>

// CHECK: !qElemType = !quant.uniform<i8:f16, 5.000000e-01>
// CHECK: !qElemType1 = !quant.uniform<!quant.quantile<ui4:si8, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01}>:f16, 2.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<u4:f16, 2.000000e+00>

// CHECK-LABEL: @ConvertInt8ActU4WgtSymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertInt8ActU4WgtSymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<3.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<3.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  // CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<i8:f16, 5.000000e-01>
!wgtType = !quant.uniform<u4:f16, 2.0:8>

// CHECK: !qElemType = !quant.uniform<i8:f16, 5.000000e-01>
// CHECK: !qElemType1 = !quant.uniform<!quant.quantile<ui4:si8, {-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00}>:f16, 2.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<u4:f16, 2.000000e+00:8>

// CHECK-LABEL: @ConvertInt8ActU4WgtAsymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertInt8ActU4WgtAsymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<3.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<3.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  // CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<i8:f16, 5.000000e-01>
!wgtType = !quant.uniform<i4:f16, 2.0>

// CHECK: !qElemType = !quant.uniform<i8:f16, 5.000000e-01>
// CHECK: !qElemType1 = !quant.uniform<!quant.quantile<ui4:si8, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>:f16, 2.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<i4:f16, 2.000000e+00>

// CHECK-LABEL: @ConvertInt8ActI4WgtSymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertInt8ActI4WgtSymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<3.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<3.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  // CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<i8:f16, 5.000000e-01>
!wgtType = !quant.uniform<i4:f16, 2.0:-3>

// CHECK: !qElemType = !quant.uniform<i8:f16, 5.000000e-01>
// CHECK: !qElemType1 = !quant.uniform<!quant.quantile<ui4:si8, {3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00}>:f16, 2.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<i4:f16, 2.000000e+00:-3>

// CHECK-LABEL: @ConvertInt8ActI4WgtAsymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertInt8ActI4WgtAsymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<3.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<3.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  // CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<u8:f16, 5.000000e-01>
!wgtType = !quant.uniform<u4:f16, 2.0:8>

// CHECK: !qElemType = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK: !qElemType1 = !quant.uniform<u4:f16, 2.000000e+00:8>

// CHECK-LABEL: @NoConvertU8ActU4Wgt
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @NoConvertU8ActU4Wgt(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<3.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  // CHECK-NOT: QuantileType
  // CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1>
  // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  // CHECK: return [[CONV]]
}
