//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-mixed-precision="enable-float-in-quant-weights-mixed-mode=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
// CHECK-LABEL: @MixedPrecisionFloatInputQuantWeightsConv
func.func @MixedPrecisionFloatInputQuantWeightsConv(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType>

  //CHECK: [[VAL1:%.*]] = IE.Convolution(%arg0, [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
// CHECK-LABEL: @MixedPrecisionFloatInputQuantWeightsGroupConv
func.func @MixedPrecisionFloatInputQuantWeightsGroupConv(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x1x1x1x!qElemType> = dense<1.0> : tensor<16x1x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x1x1x1x!qElemType> -> tensor<16x1x1x1xf16>
  %result = IE.GroupConvolution(%arg0, %weights) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x1x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x1x1x1x!qElemType>

  //CHECK: [[VAL1:%.*]] = IE.GroupConvolution(%arg0, [[VAL0]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x1x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i8:f16:0, {1.153, 1.153, 1.153, 1.153, 1.153, 1.153, 1.153, 1.153, 1.153, 1.153, 1.153, 1.153, 1.153, 1.153, 1.153, 1.153}>
// CHECK-LABEL: @MixedPrecisionFloatInputQuantWeightsConvPerAxis
func.func @MixedPrecisionFloatInputQuantWeightsConvPerAxis(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType>

  //CHECK: [[VAL1:%.*]] = IE.Convolution(%arg0, [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195:10>
// CHECK-LABEL: @InvalidZPMixedPrecisionFloatInputQuantWeightsConv
func.func @InvalidZPMixedPrecisionFloatInputQuantWeightsConv(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
  //CHECK: [[VAL2:%.*]] = IE.Convolution(%arg0, [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195>
// CHECK-LABEL: @InvalidDTypeMixedPrecisionFloatInputQuantWeightsConv
func.func @InvalidDTypeMixedPrecisionFloatInputQuantWeightsConv(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
  //CHECK: [[VAL2:%.*]] = IE.Convolution(%arg0, [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
// CHECK-LABEL: @MixedPrecisionFloatInputQuantWeightsAndOutputConv
func.func @MixedPrecisionFloatInputQuantWeightsAndOutputConv(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16x!qElemType> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %conv = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  %result = IE.Quantize(%conv) {dstElemType = !qElemType} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16x!qElemType>
  return %result : tensor<1x16x16x16x!qElemType>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType>

  //CHECK: [[VAL1:%.*]] = IE.Convolution(%arg0, [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16x!qElemType>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.1534313725490195>
// CHECK-LABEL: @MixedPrecisionFloatInputQuantWeightsConv
func.func @MixedPrecisionFloatInputQuantWeightsConv(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType>

  //CHECK: [[VAL1:%.*]] = IE.Convolution(%arg0, [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]]
}

// -----

!quantFloatType = !QuantileFloat.quantileFloat<4, {-1.0, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0}>
!qElemType = !quant.quantile<i4:f16:f16, {-1.000000e+00,-8.000000e-01,-0.69999999999999996,-6.000000e-01,-5.000000e-01,-4.000000e-01,-3.000000e-01,0.000000e+00,1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,6.000000e-01,0.69999999999999996,1.000000e+00}:0.0029490152994791669>
// CHECK-LABEL: @MixedPrecisionFloatInputQuantWeightsConvNF4
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @MixedPrecisionFloatInputQuantWeightsConvNF4(%arg0: tensor<1x1x28x28xf16>) -> tensor<1x1x29x29xf16> {
  %weights = const.Declare tensor<1x1x2x2x!qElemType> = dense_resource<blob> : tensor<1x1x2x2x!quantFloatType>, [#const.ConvertElemType<f32>, #const.CastElemType<!quantFloatType>, #const.CastElemType<f16>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<1x1x2x2x!qElemType> -> tensor<1x1x2x2xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>

  return %result : tensor<1x1x29x29xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<1x1x2x2x!qElemType> = dense_resource<blob> : tensor<1x1x2x2x!QuantileFloat.quantileFloat<4, {-1.000000e+00,-8.000000e-01,-0.69999999999999996,-6.000000e-01,-5.000000e-01,-4.000000e-01,-3.000000e-01,0.000000e+00,1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,6.000000e-01,0.69999999999999996,1.000000e+00}>>, [#const.ConvertElemType<f32>, #const.CastElemType<!QuantileFloat.quantileFloat<4, {-1.000000e+00,-8.000000e-01,-0.69999999999999996,-6.000000e-01,-5.000000e-01,-4.000000e-01,-3.000000e-01,0.000000e+00,1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,6.000000e-01,0.69999999999999996,1.000000e+00}>>, #const.CastElemType<f16>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>]

  //CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2x!qElemType> -> tensor<1x1x29x29xf16>
  //CHECK: return [[CONV]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000002222"
    }
  }
#-}
