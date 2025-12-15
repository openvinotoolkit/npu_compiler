//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-pallet-lut --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

!actType = !quant.uniform<f8E5M2:f16, 1.0>
!wgtType = !quant.uniform<u4:f16, 2.0:0>

// CHECK: !qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.quantile<u4:f8E4M3FN:f16, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01}:2.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<u4:f16, 2.000000e+00>

// CHECK-LABEL: @ConvertBF8ActU4WgtAsymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertBF8ActU4WgtAsymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<15.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<1.500000e+01> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<f8E5M2:f16, 1.0>
!wgtType = !quant.uniform<u4:f16, 2.0:8>

// CHECK: !qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.quantile<u4:f8E4M3FN:f16, {-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00}:2.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<u4:f16, 2.000000e+00:8>

// CHECK-LABEL: @ConvertBF8ActU4WgtSymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertBF8ActU4WgtSymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<15.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<1.500000e+01> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<f8E4M3FN:f16, 1.0>
!wgtType = !quant.uniform<u2:f16, 2.0:2>

// CHECK: !qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.quantile<u2:f8E4M3FN:f16, {-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00}:2.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<u2:f16, 2.000000e+00:2>

// CHECK-LABEL: @ConvertHF8ActU2WgtSymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertHF8ActU2WgtSymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<3.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<3.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<f8E4M3FN:f16, 1.0>
!wgtType = !quant.uniform<i2:f16, 2.0:0>

// CHECK: !qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.quantile<u2:f8E4M3FN:f16, {0.000000e+00,1.000000e+00,-2.000000e+00,-1.000000e+00}:2.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<i2:f16, 2.000000e+00>

// CHECK-LABEL: @ConvertHF8ActI2WgtSymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertHF8ActI2WgtSymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<f8E4M3FN:f16, 1.0>
!wgtType = !quant.uniform<i2:f16, 2.0:-1>

// CHECK: !qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.quantile<u2:f8E4M3FN:f16, {1.000000e+00,2.000000e+00,-1.000000e+00,0.000000e+00}:2.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<i2:f16, 2.000000e+00:-1>

// CHECK-LABEL: @ConvertHF8ActI2WgtAsymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertHF8ActI2WgtAsymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<f8E5M2:f16, 1.0>
!wgtType = !quant.uniform<i4:f16, 3.0:1>

// CHECK: !qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.quantile<u4:f8E4M3FN:f16, {-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00}:3.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<i4:f16, 3.000000e+00:1>

// CHECK-LABEL: @ConvertBF8ActI4WgtAsymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertBF8ActI4WgtAsymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<7.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<7.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!actType = !quant.uniform<f8E4M3FN:f16, 1.0>
!wgtType = !quant.uniform<i4:f16, 1.0>

// CHECK: !qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK: !quant.quantile<u4:f8E4M3FN:f16, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}:1.000000e+00>
// CHECK: !qElemType2 = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertHF8ActI4WgtSymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @ConvertHF8ActI4WgtSymmetricZp(%arg0: tensor<1x16x16x16x!actType>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!wgtType> = dense<7.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<7.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType1>]
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}


// -----

!actType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
!wgtType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<i4:f16, 1.000000e+00>
// CHECK: !qElemType2 = !quant.quantile<u4:f8E4M3FN:f16, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}:1.000000e+00>

// CHECK-LABEL: @ConvertHF8ActI4WeightsAsInput
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>, [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType1>)
func.func @ConvertHF8ActI4WeightsAsInput(%arg0: tensor<1x16x16x16x!actType>, %arg1: tensor<16x16x1x1x!wgtType>) -> tensor<1x16x16x16xf16> {
  %actDequant = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x16x16x16x!actType> -> tensor<1x16x16x16xf16>
  %wgtDequant = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%actDequant, %wgtDequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[ACTDEQUANT:%.+]] = IE.Dequantize([[ARG0]]) {dstElemType = f16} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType2} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1x!qElemType2>
  //CHECK: [[WGTDEQUANT:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType2> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ACTDEQUANT]], [[WGTDEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  //CHECK: return [[CONV]]
}


// -----

!wgtType = !quant.uniform<i4:f16, 1.000000e+00 : -1>

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00:-1>
// CHECK: !qElemType1 = !quant.quantile<u4:f16:f16, {1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00}:1.000000e+00>

// CHECK-LABEL: @ConvertFP16ActI4WeightsAsInput
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>, [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType>)
func.func @ConvertFP16ActI4WeightsAsInput(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1x!wgtType>) -> tensor<1x16x16x16xf16> {
  %wgtDequant = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %wgtDequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType1} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1x!qElemType1>
  //CHECK: [[WGTDEQUANT:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WGTDEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  //CHECK: return [[CONV]]
}


// -----

!wgtType = !quant.uniform<i4:f16, 1.000000e+00 : 0>

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.quantile<u4:f16:f16, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}:1.000000e+00>
// CHECK-LABEL: @ConvertFP16ActI4WeightsAsInputSymmetric
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>, [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType>)
func.func @ConvertFP16ActI4WeightsAsInputSymmetric(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1x!wgtType>) -> tensor<1x16x16x16xf16> {
  %wgtDequant = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %wgtDequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType1} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1x!qElemType1>
  //CHECK: [[WGTDEQUANT:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WGTDEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  //CHECK: return [[CONV]]
}


// -----

!wgtType = !quant.uniform<i2:f16, 1.000000e+00 : 0>

// CHECK: !qElemType = !quant.uniform<i2:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.quantile<u2:f16:f16, {0.000000e+00,1.000000e+00,-2.000000e+00,-1.000000e+00}:1.000000e+00>
// CHECK-LABEL: @ConvertFP16ActI2WeightsAsInputSymmetric
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>, [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType>)
func.func @ConvertFP16ActI2WeightsAsInputSymmetric(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1x!wgtType>) -> tensor<1x16x16x16xf16> {
  %wgtDequant = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %wgtDequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType1} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1x!qElemType1>
  //CHECK: [[WGTDEQUANT:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WGTDEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  //CHECK: return [[CONV]]
}

// -----

!wgtType = !quant.uniform<u2:f16, 1.000000e+00 : 2>

// CHECK: !qElemType = !quant.uniform<u2:f16, 1.000000e+00:2>
// CHECK: !qElemType1 = !quant.quantile<u2:f16:f16, {-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00}:1.000000e+00>
// CHECK-LABEL: @ConvertFP16ActU2WeightsAsInputSymmetric
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>, [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType>)
func.func @ConvertFP16ActU2WeightsAsInputSymmetric(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1x!wgtType>) -> tensor<1x16x16x16xf16> {
  %wgtDequant = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %wgtDequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType1} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1x!qElemType1>
  //CHECK: [[WGTDEQUANT:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WGTDEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  //CHECK: return [[CONV]]
}
