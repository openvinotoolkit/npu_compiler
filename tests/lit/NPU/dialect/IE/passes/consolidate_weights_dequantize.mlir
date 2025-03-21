//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --consolidate-weights-dequantize="enable-weights-dynamic-dequantization=true" --mlir-print-elementsattrs-with-hex-if-larger -1 %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK: !qElemType = !quant.uniform<u8:f32, 5.000000e-01>

// CHECK-LABEL: @StaticScaleDequantization
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @StaticScaleDequantization(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xui8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%weights) {dstElemType = f32} : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>
  %multiply = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %conv : tensor<1x4x28x28xf32>

  // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xui8> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f32} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf32>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  // CHECK: return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00:100>

// CHECK-LABEL: @StaticShiftDequantization
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @StaticShiftDequantization(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xsi8>) -> tensor<1x4x28x28xf16> {
  %shift = const.Declare tensor<1x1x1x1xf16> = dense<100.0> : tensor<1x1x1x1xf16>

  %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf16>
  %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %subtract) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>

  // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  // CHECK: return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f32, 5.000000e-01:100>

// CHECK-LABEL: @StaticScaleShiftDequantization
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @StaticScaleShiftDequantization(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xsi8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>
  %shift = const.Declare tensor<1x1x1x1xf32> = dense<100.0> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%weights) {dstElemType = f32} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf32>
  %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %conv : tensor<1x4x28x28xf32>

  // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f32} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf32>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  // CHECK: return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f32, 5.000000e-01:100>

// CHECK-LABEL: @StaticMultipleConvertsDequantization
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @StaticMultipleConvertsDequantization(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xsi8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>
  %shift = const.Declare tensor<1x1x1x1xf32> = dense<100.0> : tensor<1x1x1x1xf32>

  %convert_0 = IE.Convert(%weights) {dstElemType = ui8} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xui8>
  %convert_1 = IE.Convert(%convert_0) {dstElemType = f16} : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf16>
  %convert_2 = IE.Convert(%convert_1) {dstElemType = f32} : tensor<4x4x3x3xf16> -> tensor<4x4x3x3xf32>

  %subtract = IE.Subtract(%convert_2, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %conv : tensor<1x4x28x28xf32>

  // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f32} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf32>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  // CHECK: return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u4:f32, 5.000000e-01>

// CHECK-LABEL: @StaticScaleUI4Dequantization
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui4>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @StaticScaleUI4Dequantization(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xui4>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%weights) {dstElemType = f32} : tensor<4x4x3x3xui4> -> tensor<4x4x3x3xf32>
  %multiply = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %conv : tensor<1x4x28x28xf32>

  // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xui4> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f32} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf32>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  // CHECK: return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00:100>

// CHECK-LABEL: @StaticShiftUI4Dequantization
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xsi4>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @StaticShiftUI4Dequantization(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xsi4>) -> tensor<1x4x28x28xf16> {
  %shift = const.Declare tensor<1x1x1x1xf16> = dense<100.0> : tensor<1x1x1x1xf16>

  %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xsi4> -> tensor<4x4x3x3xf16>
  %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %subtract) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>

  // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xsi4> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  // CHECK: return [[CONV]]
}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<4, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

// CHECK: !quant.quantile<i4:f16:f32, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:5.000000e-01:100>

// CHECK-LABEL: @StaticScaleShiftNF4Dequantization
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3x!QuantileFloat.quantileFloat<4, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @StaticScaleShiftNF4Dequantization(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3x!quantileFloatType>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>
  %shift = const.Declare tensor<1x1x1x1xf32> = dense<100.0> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%weights) {dstElemType = f32} : tensor<4x4x3x3x!quantileFloatType> -> tensor<4x4x3x3xf32>
  %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %conv : tensor<1x4x28x28xf32>

  // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3x!QuantileFloat.quantileFloat<4, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f32} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf32>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  // CHECK: return [[CONV]]
}

// -----

#CNHW = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00:100>
// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @TransposeStaticShiftDequantization
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @TransposeStaticShiftDequantization(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xsi8>) -> tensor<1x4x28x28xf16> {
  %shift = const.Declare tensor<1x1x1x1xf16> = dense<100.0> : tensor<1x1x1x1xf16>

  %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf16>
  %transpose = IE.Transpose(%convert) {order_value = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>} : tensor<4x4x3x3xf16> -> tensor<4x4x3x3xf16>
  %subtract = IE.Subtract(%transpose, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %subtract) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>

  // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[TRANSPOSE:%.+]] = IE.Transpose([[QUANT_CAST]]) {order_value = #map} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[TRANSPOSE]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  // CHECK: return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, -5.000000e-01>

// CHECK-LABEL: @NegativeScaleDequantization
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @NegativeScaleDequantization(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xui8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<-0.5> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%weights) {dstElemType = f32} : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>
  %multiply = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %conv : tensor<1x4x28x28xf32>

  // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xui8> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f32} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf32>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  // CHECK:  return [[CONV]] : tensor<1x4x28x28xf32>
}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f32:1, {0.10000000149011612:100,0.20000000298023224:100,0.30000001192092896:100,0.40000000596046448:100}>

// CHECK-LABEL: @StaticPerAxisScalePerTensorShiftDequantization
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @StaticPerAxisScalePerTensorShiftDequantization(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xsi8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x4x1x1xf32> = dense<[[[[0.1]], [[0.2]], [[0.3]], [[0.4]]]]> : tensor<1x4x1x1xf32>
  %shift = const.Declare tensor<1x1x1x1xf32> = dense<100.0> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%weights) {dstElemType = f32} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf32>
  %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x4x1x1xf32> -> tensor<4x4x3x3xf32>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %conv : tensor<1x4x28x28xf32>

  // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f32} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf32>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  // CHECK: return [[CONV]]
}

// -----

// CHECK-NOT: !quant.uniform

// CHECK-LABEL: @NotStaticPerAxisScalePerAxisShiftDequantization
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @NotStaticPerAxisScalePerAxisShiftDequantization(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xsi8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x4x1x1xf32> = dense<[[[[0.1]], [[0.2]], [[0.3]], [[0.4]]]]> : tensor<1x4x1x1xf32>
  %shift = const.Declare tensor<1x4x1x1xf32> = dense<[[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]]> : tensor<1x4x1x1xf32>

  %convert = IE.Convert(%weights) {dstElemType = f32} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf32>
  %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x4x1x1xf32> -> tensor<4x4x3x3xf32>
  %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x4x1x1xf32> -> tensor<4x4x3x3xf32>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %conv : tensor<1x4x28x28xf32>

  // CHECK-NOT:  [[QUANT_CAST:%.+]] = IE.QuantizeCast
  // CHECK-NOT:  [[DEQUANT:%.+]] = IE.Dequantize

  // CHECK:  [[SCALE:%.+]] = const.Declare tensor<1x4x1x1xf32>
  // CHECK-SAME{LITERAL} = dense<[[[[1.000000e-01]], [[2.000000e-01]], [[3.000000e-01]], [[4.000000e-01]]]]>
  // CHECK:  [[SHIFT:%.+]] = const.Declare tensor<1x4x1x1xf32>
  // CHECK-SAME{LITERAL} = dense<[[[[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]]]]>

  // CHECK:  [[CONVERT:%.+]] = IE.Convert([[WEIGHTS]])
  // CHECK:  [[SUBTRACT:%.+]] = IE.Subtract([[CONVERT]], [[SHIFT]])
  // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[SUBTRACT]], [[SCALE]])
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[MULTIPLY]])

  // CHECK: return [[CONV]]
}

// -----

// CHECK-NOT: quant.uniform

// CHECK-LABEL: @NotConvertOnlyDequantization
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @NotConvertOnlyDequantization(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xsi8>) -> tensor<1x4x28x28xf16> {
  %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %convert) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>

  // CHECK-NOT:  [[QUANT_CAST:%.+]] = IE.QuantizeCast
  // CHECK-NOT:  [[DEQUANT:%.+]] = IE.Dequantize

  // CHECK:  [[CONVERT:%.+]] = IE.Convert([[WEIGHTS]])
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CONVERT]])

  // CHECK: return [[CONV]]
}

// -----

// CHECK-NOT: quant.uniform

// CHECK-LABEL: @NotStaticScaleShiftDequantizationOnInvalidOp
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<1x4x28x28xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @NotStaticScaleShiftDequantizationOnInvalidOp(%input: tensor<1x4x28x28xf32>, %weights: tensor<1x4x28x28xsi8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>
  %shift = const.Declare tensor<1x1x1x1xf32> = dense<100.0> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%weights) {dstElemType = f32} : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf32>
  %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %add = IE.Add(%input, %multiply) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x4x28x28xf32> -> tensor<1x4x28x28xf32>

  return %add : tensor<1x4x28x28xf32>

  // CHECK-NOT:  IE.QuantizeCast
  // CHECK-NOT:  IE.Dequantize

  // CHECK-DAG:  [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01>
  // CHECK-DAG:  [[SHIFT:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+02>

  // CHECK:  [[CONVERT:%.+]] = IE.Convert([[WEIGHTS]])
  // CHECK:  [[SUBTRACT:%.+]] = IE.Subtract([[CONVERT]], [[SHIFT]])
  // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[SUBTRACT]], [[SCALE]])
  // CHECK:  [[ADD:%.+]] = IE.Add([[INPUT]], [[MULTIPLY]])

  // CHECK:  return [[ADD]]
}

// -----

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00:8>

// CHECK-LABEL: @DynamicScaleDequantization
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<16x16x1x1xsi4>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<1x16x1x1xf16>
func.func @DynamicScaleDequantization(%input: tensor<1x16x16x16xf16>, %weights: tensor<16x16x1x1xsi4>, %scale: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %zp = const.Declare tensor<1x16x1x1xsi4> = dense<8.0> : tensor<1x16x1x1xf16>,
              [#const.CastElemType<si4>]
    %zp_f16 = IE.Convert(%zp) { dstElemType = f16 } : tensor<1x16x1x1xsi4> -> tensor<1x16x1x1xf16>

    %weights_f16 = IE.Convert(%weights) { dstElemType = f16 } : tensor<16x16x1x1xsi4> -> tensor<16x16x1x1xf16>

    %subtract = IE.Subtract(%weights_f16, %zp_f16) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>
    %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

    %conv = IE.Convolution(%input, %multiply) {
              dilations = [1, 1],
              pads_begin = [0, 0],
              pads_end = [0, 0],
              strides = [1, 1]
          } : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>
              -> tensor<1x16x16x16xf16>

    return %conv : tensor<1x16x16x16xf16>

    // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<16x16x1x1xsi4> -> tensor<16x16x1x1x!qElemType>

    // CHECK:  [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANT_CAST]], [[SCALE]]) {dstElemType = f16}
    // CHECK-SAME:     : tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DYN_DEQUANT]])
    // CHECK-SAME:     {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:     : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    // CHECK:  return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

// CHECK: !qElemType = !quant.uniform<i4:f32, 1.000000e+00>

// CHECK-LABEL: @DynamicScaleDequantizationWithTranspose
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<28x512x128xsi4>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<28x1x512xf32>
func.func @DynamicScaleDequantizationWithTranspose(%weights: tensor<28x512x128xsi4>, %scale: tensor<28x1x512xf32>) -> tensor<28x128x512xf32> {
    %weights_f32 = IE.Convert(%weights) {dstElemType = f32} : tensor<28x512x128xsi4> -> tensor<28x512x128xf32>
    %transpose = IE.Transpose(%weights_f32) {order_value = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : tensor<28x512x128xf32> -> tensor<28x128x512xf32>
    %multiply = IE.Multiply(%transpose, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<28x128x512xf32>, tensor<28x1x512xf32> -> tensor<28x128x512xf32>

    return %multiply : tensor<28x128x512xf32>

    // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<28x512x128xsi4> -> tensor<28x512x128x!qElemType>
    // CHECK:  [[TRANSPOSE:%.+]] = IE.Transpose([[QUANT_CAST]]) {order_value = #map} : tensor<28x512x128x!qElemType> -> tensor<28x128x512x!qElemType>
    // CHECK:  [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[TRANSPOSE]], [[SCALE]]) {dstElemType = f32} : tensor<28x128x512x!qElemType>, tensor<28x1x512xf32> -> tensor<28x128x512xf32>
    // CHECK:  return [[DYN_DEQUANT]] : tensor<28x128x512xf32>
}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @DynamicScaleDequantizationForINT8Weights
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<73440x1536xsi8>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<73440x1xf16>,
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x1536xf32>
func.func @DynamicScaleDequantizationForINT8Weights(%weights: tensor<73440x1536xsi8>, %scale: tensor<73440x1xf16>, %input: tensor<1x1536xf32>) -> tensor<1x73440xf32> {
    %weights_f16 = IE.Convert(%weights) {dstElemType = f16} : tensor<73440x1536xsi8> -> tensor<73440x1536xf16>
    %multiply = IE.Multiply(%weights_f16, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<73440x1536xf16>, tensor<73440x1xf16> -> tensor<73440x1536xf16>
    %weights_f32 = IE.Convert(%multiply) {dstElemType = f32} : tensor<73440x1536xf16> -> tensor<73440x1536xf32>
    %fc = IE.FullyConnected(%input, %weights_f32) : tensor<1x1536xf32>, tensor<73440x1536xf32> -> tensor<1x73440xf32>

    return %fc: tensor<1x73440xf32>

    // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<73440x1536xsi8> -> tensor<73440x1536x!qElemType>
    // CHECK:  [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANT_CAST]], [[SCALE]]) {dstElemType = f16} : tensor<73440x1536x!qElemType>, tensor<73440x1xf16> -> tensor<73440x1536xf16>
    // CHECK:  [[CONVERT:%.+]] = IE.Convert([[DYN_DEQUANT]]) {dstElemType = f32} : tensor<73440x1536xf16> -> tensor<73440x1536xf32>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT]], [[CONVERT]]) : tensor<1x1536xf32>, tensor<73440x1536xf32> -> tensor<1x73440xf32>

    // CHECK:  return [[FC]] : tensor<1x73440xf32>
}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<4, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

// CHECK: !qElemType = !quant.quantile<i4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:1.000000e+00:100>

// CHECK-LABEL: @DynamicScaleStaticShiftDequantizationForNF4Weights
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<16x16x1x1x!QuantileFloat.quantileFloat<4, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<1x16x1x1xf16>
func.func @DynamicScaleStaticShiftDequantizationForNF4Weights(%input: tensor<1x16x16x16xf16>, %weights: tensor<16x16x1x1x!quantileFloatType>, %scale: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %zp = const.Declare tensor<1x16x1x1xsi4> = dense<100.0> : tensor<1x16x1x1xf16>,
              [#const.CastElemType<si4>]
    %zp_f16 = IE.Convert(%zp) { dstElemType = f16 } : tensor<1x16x1x1xsi4> -> tensor<1x16x1x1xf16>

    %weights_f16 = IE.Convert(%weights) { dstElemType = f16 } : tensor<16x16x1x1x!quantileFloatType> -> tensor<16x16x1x1xf16>

    %subtract = IE.Subtract(%weights_f16, %zp_f16) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>
    %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

    %conv = IE.Convolution(%input, %multiply) {
              dilations = [1, 1],
              pads_begin = [0, 0],
              pads_end = [0, 0],
              strides = [1, 1]
          } : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>
              -> tensor<1x16x16x16xf16>

    return %conv : tensor<1x16x16x16xf16>

    // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType}
    // CHECK-SAME:     : tensor<16x16x1x1x!QuantileFloat.quantileFloat<4, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<16x16x1x1x!qElemType>

    // CHECK:  [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANT_CAST]], [[SCALE]]) {dstElemType = f16}
    // CHECK-SAME:     : tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DYN_DEQUANT]])
    // CHECK-SAME:     {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:     : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    // CHECK:  return [[CONV]] : tensor<1x16x16x16xf16>
}
