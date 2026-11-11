//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-to-pallet-lut --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU6010


!wgtType = !quant.uniform<i4:f16, 1.000000e+00 : -1>

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00:-1>
// CHECK: !qElemType1 = !quant.uniform<!quant.quantile<ui4:f16, {1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertFP16ActI4WeightsAsInputDynDequant
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>, [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType>, [[ARG2:%.+]]: tensor<1x16x1x1xf16>)
func.func @ConvertFP16ActI4WeightsAsInputDynDequant(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1x!wgtType>, %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
  %wgtDequant = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<16x16x1x1x!wgtType>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %wgtDequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType1} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1x!qElemType1>
  //CHECK: [[WGTDEQUANT:%.+]] = IE.DynamicDequantize([[QUANTIZECAST]], [[ARG2]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WGTDEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
!qElemType1 = !quant.uniform<u2:f16, 1.000000e+00:2>

// CHECK: !qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
// CHECK: !qElemType1 = !quant.uniform<!quant.quantile<ui2:si8, {-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertI8ActU2WeightsAsInput
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1536x1x1xf32>, [[ARG1:%.+]]: tensor<16x1536x1x1xui2>)
func.func @ConvertI8ActU2WeightsAsInput(%arg0: tensor<1x1536x1x1xf32>, %arg1: tensor<16x1536x1x1xui2>) -> tensor<1x16x1x1xf16> {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1536x1x1xf32> -> tensor<1x1536x1x1xf16>
  %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x1536x1x1xf16> -> tensor<1x1536x1x1x!qElemType>
  %2 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType1} : tensor<16x1536x1x1xui2> -> tensor<16x1536x1x1x!qElemType1>
  %3 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x1536x1x1x!qElemType> -> tensor<1x1536x1x1xf16>
  %4 = IE.Dequantize(%2) {dstElemType = f16} : tensor<16x1536x1x1x!qElemType1> -> tensor<16x1536x1x1xf16>
  %5 = IE.Convolution(%3, %4) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1536x1x1xf16>, tensor<16x1536x1x1xf16> -> tensor<1x16x1x1xf16>
  return %5 : tensor<1x16x1x1xf16>

  //CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x1536x1x1xf32> -> tensor<1x1536x1x1xf16>
  //CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x1536x1x1xf16> -> tensor<1x1536x1x1x!qElemType>
  //CHECK: [[DEQUANT_ACT:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x1536x1x1x!qElemType> -> tensor<1x1536x1x1xf16>
  //CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType1} : tensor<16x1536x1x1xui2> -> tensor<16x1536x1x1x!qElemType1>
  //CHECK: [[DEQUANT_WGT:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<16x1536x1x1x!qElemType1> -> tensor<16x1536x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[DEQUANT_ACT]], [[DEQUANT_WGT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1536x1x1xf16>, tensor<16x1536x1x1xf16> -> tensor<1x16x1x1xf16>

  //CHECK: return [[CONV]]
}

// -----

!wgtTypeU4PerAxis = !quant.uniform<u4:f16:0, {2.000000e+00:5, 3.000000e+00:10}>

// CHECK: !qElemType = !quant.uniform<!quant.quantile<ui4:ui8, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01}>:f16:0, {2.000000e+00:5,3.000000e+00:10}>

// CHECK-LABEL: @ConvertFP16ActU4PerAxisWeights
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>)
func.func @ConvertFP16ActU4PerAxisWeights(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x2x16x16xf16> {
  %weights = const.Declare tensor<2x16x1x1x!wgtTypeU4PerAxis> = dense<7.0> : tensor<2x16x1x1xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!wgtTypeU4PerAxis>]
  %wgtDequant = IE.Dequantize(%weights) {dstElemType = f16} : tensor<2x16x1x1x!wgtTypeU4PerAxis> -> tensor<2x16x1x1xf16>
  %result = IE.Convolution(%arg0, %wgtDequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<2x16x1x1xf16> -> tensor<1x2x16x16xf16>

  return %result : tensor<1x2x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<2x16x1x1x!qElemType>
  //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<2x16x1x1x!qElemType> -> tensor<2x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<2x16x1x1xf16> -> tensor<1x2x16x16xf16>

  //CHECK: return [[CONV]]
}

// -----

// FP4 weights with a per-tensor scale and fp8 activations. The LUT entries are the raw decoded FP4 values
// clamped to F8E4M3FN. Bit-pattern 1000 decodes to -0.0 in f4E2M1FN, so LUT[8] is -0.0.

!actType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
!wgtType = !quant.uniform<f4E2M1FN:f16, 5.000000e-01>

// CHECK: !qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<f4E2M1FN:f16, 5.000000e-01>
// CHECK: !qElemType2 = !quant.uniform<!quant.quantile<ui4:f8E4M3FN, {0.000000e+00,5.000000e-01,1.000000e+00,1.500000e+00,2.000000e+00,3.000000e+00,4.000000e+00,6.000000e+00,-0.000000e+00,-5.000000e-01,-1.000000e+00,-1.500000e+00,-2.000000e+00,-3.000000e+00,-4.000000e+00,-6.000000e+00}>:f16, 5.000000e-01>

// CHECK-LABEL: @ConvertFP8ActFP4Wgt
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>, [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType1>)
func.func @ConvertFP8ActFP4Wgt(%arg0: tensor<1x16x16x16x!actType>, %arg1: tensor<16x16x1x1x!wgtType>) -> tensor<1x16x16x16xf16> {
  %wgtDequant = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %wgtDequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!actType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType2} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1x!qElemType2>
  //CHECK: [[WGTDEQUANT:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType2> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WGTDEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  //CHECK: return [[CONV]]
}

// -----

// FP4 weights with per-channel scales (UniformQuantizedPerAxisType) and fp8 activations. Each channel has its
// own scale, and the zero-point is the single shared per-tensor value, which remains zero. The LUT is the same
// for all channels (raw decoded FP4 values clamped to F8E4M3FN), and the per-channel scales are preserved in the
// resulting quantized type, while the zero-point remains zero.

!actType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
!wgtType = !quant.uniform<f4E2M1FN:f16:0, {5.000000e-01,2.500000e-01,1.000000e+00,2.000000e+00}>

// CHECK: !qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<f4E2M1FN:f16:0, {5.000000e-01,2.500000e-01,1.000000e+00,2.000000e+00}>
// CHECK: !qElemType2 = !quant.uniform<!quant.quantile<ui4:f8E4M3FN, {0.000000e+00,5.000000e-01,1.000000e+00,1.500000e+00,2.000000e+00,3.000000e+00,4.000000e+00,6.000000e+00,-0.000000e+00,-5.000000e-01,-1.000000e+00,-1.500000e+00,-2.000000e+00,-3.000000e+00,-4.000000e+00,-6.000000e+00}>:f16:0, {5.000000e-01,2.500000e-01,1.000000e+00,2.000000e+00}>

// CHECK-LABEL: @ConvertFP8ActFP4WgtPerChannel
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x4x4x4x!qElemType>, [[ARG1:%.+]]: tensor<4x4x1x1x!qElemType1>)
func.func @ConvertFP8ActFP4WgtPerChannel(%arg0: tensor<1x4x4x4x!actType>, %arg1: tensor<4x4x1x1x!wgtType>) -> tensor<1x4x4x4xf16> {
  %wgtDequant = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<4x4x1x1x!wgtType> -> tensor<4x4x1x1xf16>
  %result = IE.Convolution(%arg0, %wgtDequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x4x4x!actType>, tensor<4x4x1x1xf16> -> tensor<1x4x4x4xf16>

  return %result : tensor<1x4x4x4xf16>

  //CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType2} : tensor<4x4x1x1x!qElemType1> -> tensor<4x4x1x1x!qElemType2>
  //CHECK: [[WGTDEQUANT:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<4x4x1x1x!qElemType2> -> tensor<4x4x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WGTDEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x4x4x!qElemType>, tensor<4x4x1x1xf16> -> tensor<1x4x4x4xf16>

  //CHECK: return [[CONV]]
}

// -----

// FP4 weights with a per-tensor scale and fp16 (plain, non-quantized) activations. Even for fp16 activations,
// FP4 weights are palletized into an F8E4M3FN LUT (not f16), matching the fp8-activation case. The LUT entries
// are the raw decoded FP4 values clamped to F8E4M3FN. Bit-pattern 1000 decodes to -0.0 in f4E2M1FN, so LUT[8]
// is -0.0.

!wgtType = !quant.uniform<f4E2M1FN:f16, 5.000000e-01>

// CHECK: !qElemType = !quant.uniform<f4E2M1FN:f16, 5.000000e-01>
// CHECK: !qElemType1 = !quant.uniform<!quant.quantile<ui4:f8E4M3FN, {0.000000e+00,5.000000e-01,1.000000e+00,1.500000e+00,2.000000e+00,3.000000e+00,4.000000e+00,6.000000e+00,-0.000000e+00,-5.000000e-01,-1.000000e+00,-1.500000e+00,-2.000000e+00,-3.000000e+00,-4.000000e+00,-6.000000e+00}>:f16, 5.000000e-01>

// CHECK-LABEL: @ConvertFP16ActFP4Wgt
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>, [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType>)
func.func @ConvertFP16ActFP4Wgt(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1x!wgtType>) -> tensor<1x16x16x16xf16> {
  %wgtDequant = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<16x16x1x1x!wgtType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %wgtDequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType1} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1x!qElemType1>
  //CHECK: [[WGTDEQUANT:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WGTDEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  //CHECK: return [[CONV]]
}

// -----

// FP4 weights with per-channel scales (UniformQuantizedPerAxisType) and fp16 (plain, non-quantized) activations.
// Each channel has its own scale, and the zero-point is the single shared per-tensor value, which remains zero.
// The LUT is the same for all channels (raw decoded FP4 values clamped to F8E4M3FN), and the per-channel scales
// are preserved in the resulting quantized type, while the zero-point remains zero.

!wgtType = !quant.uniform<f4E2M1FN:f16:0, {5.000000e-01,2.500000e-01,1.000000e+00,2.000000e+00}>

// CHECK: !qElemType = !quant.uniform<f4E2M1FN:f16:0, {5.000000e-01,2.500000e-01,1.000000e+00,2.000000e+00}>
// CHECK: !qElemType1 = !quant.uniform<!quant.quantile<ui4:f8E4M3FN, {0.000000e+00,5.000000e-01,1.000000e+00,1.500000e+00,2.000000e+00,3.000000e+00,4.000000e+00,6.000000e+00,-0.000000e+00,-5.000000e-01,-1.000000e+00,-1.500000e+00,-2.000000e+00,-3.000000e+00,-4.000000e+00,-6.000000e+00}>:f16:0, {5.000000e-01,2.500000e-01,1.000000e+00,2.000000e+00}>

// CHECK-LABEL: @ConvertFP16ActFP4WgtPerChannel
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x4x4x4xf16>, [[ARG1:%.+]]: tensor<4x4x1x1x!qElemType>)
func.func @ConvertFP16ActFP4WgtPerChannel(%arg0: tensor<1x4x4x4xf16>, %arg1: tensor<4x4x1x1x!wgtType>) -> tensor<1x4x4x4xf16> {
  %wgtDequant = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<4x4x1x1x!wgtType> -> tensor<4x4x1x1xf16>
  %result = IE.Convolution(%arg0, %wgtDequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x4x4xf16>, tensor<4x4x1x1xf16> -> tensor<1x4x4x4xf16>

  return %result : tensor<1x4x4x4xf16>

  //CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType1} : tensor<4x4x1x1x!qElemType> -> tensor<4x4x1x1x!qElemType1>
  //CHECK: [[WGTDEQUANT:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<4x4x1x1x!qElemType1> -> tensor<4x4x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WGTDEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x4x4xf16>, tensor<4x4x1x1xf16> -> tensor<1x4x4x4xf16>

  //CHECK: return [[CONV]]
}
