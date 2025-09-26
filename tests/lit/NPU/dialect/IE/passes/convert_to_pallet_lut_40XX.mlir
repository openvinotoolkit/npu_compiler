//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-pallet-lut --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX


// Only the asymmetric quantized i4/u4/i2/u2 are converted to palletized types when activations are fp16

!qElemType = !quant.uniform<i4:f16, 1.0:0>
// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @DontConvertI4UniformSymmetricZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>)
func.func @DontConvertI4UniformSymmetricZp(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<7.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<u4:f16:1, {2.0:8,1.0:8,1.5:8,2.5:8,2.0:8,1.0:8,1.5:8,2.5:8,2.0:8,1.0:8,1.5:8,2.5:8,2.0:8,1.0:8,1.5:8,2.5:8}>
// CHECK: !qElemType = !quant.uniform<u4:f16:1, {2.000000e+00:8,1.000000e+00:8,1.500000e+00:8,2.500000e+00:8,2.000000e+00:8,1.000000e+00:8,1.500000e+00:8,2.500000e+00:8,2.000000e+00:8,1.000000e+00:8,1.500000e+00:8,2.500000e+00:8,2.000000e+00:8,1.000000e+00:8,1.500000e+00:8,2.500000e+00:8}>

// CHECK-LABEL: @DontConvertU4UniformSymmZpPerAxisType
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>)
func.func @DontConvertU4UniformSymmZpPerAxisType(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<15.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<i2:f16, 2.0:0>
// CHECK: !qElemType = !quant.uniform<i2:f16, 2.000000e+00>

// CHECK-LABEL: @DontConvertI2UniformSymmZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>)
func.func @DontConvertI2UniformSymmZp(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<-2.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<u2:f16, 2.0:2>
// CHECK: !qElemType = !quant.uniform<u2:f16, 2.000000e+00:2>

// CHECK-LABEL: @DontConvertU2UniformSymmZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>)
func.func @DontConvertU2UniformSymmZp(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<3.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

// For the moment only the asymmetric quantized i4/u4/i2/u2 are converted to palletized types when activations are fp16.
// Notice that the lut is built using the weights binary encoding to determine the position in the lut.
// As a consequence for integer weights like int4, the lut content is not monotonic.
// For instance, a lut conversion for i4 quantized weights (considering zp = 0 here for simplicity) would be:
// !quant.uniform<i4:f16,1.0>  ==>  quant.quantile<u4:f16:f16, {0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0}:1.0>

!qElemType = !quant.uniform<i4:f16, 2.0:-2>
// CHECK: !qElemType = !quant.quantile<u4:f16:f16, {2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00}:2.000000e+00>

// CHECK-LABEL: @ConvertFp16ActI4WgtAsymmZpToFp16Quantile
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>)
func.func @ConvertFp16ActI4WgtAsymmZpToFp16Quantile(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<7.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<i4:f16, 2.0:-2>
!qElemType1 = !quant.uniform<i8:f16, 1.0:0>
// CHECK: !qElemType = !quant.uniform<i4:f16, 2.000000e+00:-2>
// CHECK: !qElemType1 = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @DontConvertI8ActI4WgtAsymmZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>)
func.func @DontConvertI8ActI4WgtAsymmZp(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %quant = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16x!qElemType1>
  %dqAct = IE.Dequantize(%quant) {dstElemType = f16} : tensor<1x16x16x16x!qElemType1> -> tensor<1x16x16x16xf16>
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<7.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %dqWeights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%dqAct, %dqWeights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> = dense<7.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  //CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16x!qElemType1>
  //CHECK: [[DQACT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} : tensor<1x16x16x16x!qElemType1> -> tensor<1x16x16x16xf16
  //CHECK: [[DQWGT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[DQACT]], [[DQWGT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<u4:f16:1, {2.0:0,1.0:0,1.5:0,2.5:0,2.0:0,1.0:0,1.5:0,2.5:0,2.0:0,1.0:0,1.5:0,2.5:0,2.0:0,1.0:0,1.5:0,2.5:0}>
// CHECK: !qElemType = !quant.quantile<u4:f16:f16:1, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01}:{2.000000e+00,1.000000e+00,1.500000e+00,2.500000e+00,2.000000e+00,1.000000e+00,1.500000e+00,2.500000e+00,2.000000e+00,1.000000e+00,1.500000e+00,2.500000e+00,2.000000e+00,1.000000e+00,1.500000e+00,2.500000e+00}>

// CHECK-LABEL: @ConvertU4UniformAsymmZpPerAxisTypeAllEqualZp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>)
func.func @ConvertU4UniformAsymmZpPerAxisTypeAllEqualZp(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<15.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<i2:f16, 2.0:-1>
// CHECK: !qElemType = !quant.quantile<u2:f16:f16, {1.000000e+00,2.000000e+00,-1.000000e+00,0.000000e+00}:2.000000e+00>

// CHECK-LABEL: @ConvertI2UniformAsymmZpToFp16Quantile
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>)
func.func @ConvertI2UniformAsymmZpToFp16Quantile(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<-2.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<u2:f16, 2.0:3>
// CHECK: !qElemType = !quant.quantile<u2:f16:f16, {-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00}:2.000000e+00>

// CHECK-LABEL: @ConvertU2UniformAsymmZpToFp16Quantile
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>)
func.func @ConvertU2UniformAsymmZpToFp16Quantile(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<3.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
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
