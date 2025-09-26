//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
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

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

// CHECK: !quant.quantile<u4:f16:f32, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:5.000000e-01:100>

// CHECK-LABEL: @StaticScaleShiftNF4Dequantization
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @StaticScaleShiftNF4Dequantization(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3x!quantileFloatType>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>
  %shift = const.Declare tensor<1x1x1x1xf32> = dense<100.0> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%weights) {dstElemType = f32} : tensor<4x4x3x3x!quantileFloatType> -> tensor<4x4x3x3xf32>
  %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %conv : tensor<1x4x28x28xf32>

  // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f32} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf32>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  // CHECK: return [[CONV]]
}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

// CHECK: !qElemType = !quant.quantile<u4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:5.000000e-01>

// CHECK-LABEL: @StaticScaleDequantizationForFP16ActNF4WeightsWithQuantCast
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<16x16x1x1xui4>
func.func @StaticScaleDequantizationForFP16ActNF4WeightsWithQuantCast(%input: tensor<1x16x16x16xf16>, %weights: tensor<16x16x1x1xui4>) -> tensor<1x16x16x16xf16> {
    %scale = const.Declare tensor<1x16x1x1xf16> = dense<0.5> : tensor<1x16x1x1xf16>

    %quant_cast = IE.QuantizeCast(%weights) {dstElemType = !quantileFloatType} : tensor<16x16x1x1xui4> -> tensor<16x16x1x1x!quantileFloatType>

    %weights_f16 = IE.Convert(%quant_cast) { dstElemType = f16 } : tensor<16x16x1x1x!quantileFloatType> -> tensor<16x16x1x1xf16>

    %multiply = IE.Multiply(%weights_f16, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

    %conv = IE.Convolution(%input, %multiply) {
              dilations = [1, 1],
              pads_begin = [0, 0],
              pads_end = [0, 0],
              strides = [1, 1]
          } : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>
              -> tensor<1x16x16x16xf16>

    return %conv : tensor<1x16x16x16xf16>

    // CHECK: [[QUANTCAST_1:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>} : tensor<16x16x1x1xui4> -> tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>
    // CHECK: [[QUANTCAST_2:%.+]] = IE.QuantizeCast([[QUANTCAST_1]]) {dstElemType = !qElemType} : tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<16x16x1x1x!qElemType>
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QUANTCAST_2]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    // CHECK: return [[CONV]]
}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-6.875000e-01,-5.625000e-01,-4.062500e-01,-2.812500e-01,-1.718750e-01,-8.593750e-02,0.000000e+00,7.812500e-02,1.562500e-01,2.500000e-01,3.125000e-01,4.375000e-01,5.625000e-01,6.875000e-01,1.000000e+00}>

// CHECK: !qElemType = !quant.quantile<u4:f8E4M3FN:f16, {-1.000000e+00,-6.875000e-01,-5.625000e-01,-4.062500e-01,-2.812500e-01,-1.718750e-01,-8.593750e-02,0.000000e+00,7.812500e-02,1.562500e-01,2.500000e-01,3.125000e-01,4.375000e-01,5.625000e-01,6.875000e-01,1.000000e+00}:5.000000e-01>

// CHECK-LABEL: @StaticScaleDequantizationForF8E4M3FNActNF4WeightsWithQuantCast
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<16x16x1x1xui4>
func.func @StaticScaleDequantizationForF8E4M3FNActNF4WeightsWithQuantCast(%input: tensor<1x16x16x16xf16>, %weights: tensor<16x16x1x1xui4>) -> tensor<1x16x16x16xf16> {
    %scale = const.Declare tensor<1x16x1x1xf16> = dense<0.5> : tensor<1x16x1x1xf16>

    %quant_cast = IE.QuantizeCast(%weights) {dstElemType = !quantileFloatType} : tensor<16x16x1x1xui4> -> tensor<16x16x1x1x!quantileFloatType>

    %weights_f16 = IE.Convert(%quant_cast) { dstElemType = f16 } : tensor<16x16x1x1x!quantileFloatType> -> tensor<16x16x1x1xf16>

    %multiply = IE.Multiply(%weights_f16, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

    %conv = IE.Convolution(%input, %multiply) {
              dilations = [1, 1],
              pads_begin = [0, 0],
              pads_end = [0, 0],
              strides = [1, 1]
          } : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>
              -> tensor<1x16x16x16xf16>

    return %conv : tensor<1x16x16x16xf16>

    // CHECK: [[QUANTCAST_1:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-6.875000e-01,-5.625000e-01,-4.062500e-01,-2.812500e-01,-1.718750e-01,-8.593750e-02,0.000000e+00,7.812500e-02,1.562500e-01,2.500000e-01,3.125000e-01,4.375000e-01,5.625000e-01,6.875000e-01,1.000000e+00}>} : tensor<16x16x1x1xui4> -> tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-6.875000e-01,-5.625000e-01,-4.062500e-01,-2.812500e-01,-1.718750e-01,-8.593750e-02,0.000000e+00,7.812500e-02,1.562500e-01,2.500000e-01,3.125000e-01,4.375000e-01,5.625000e-01,6.875000e-01,1.000000e+00}>>
    // CHECK: [[QUANTCAST_2:%.+]] = IE.QuantizeCast([[QUANTCAST_1]]) {dstElemType = !qElemType} : tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-6.875000e-01,-5.625000e-01,-4.062500e-01,-2.812500e-01,-1.718750e-01,-8.593750e-02,0.000000e+00,7.812500e-02,1.562500e-01,2.500000e-01,3.125000e-01,4.375000e-01,5.625000e-01,6.875000e-01,1.000000e+00}>> -> tensor<16x16x1x1x!qElemType>
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QUANTCAST_2]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    // CHECK: return [[CONV]]
}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f8E5M2, {-5.000000e-01,-2.187500e-01,-1.562500e-01,-7.812500e-02,-3.906250e-02,-0.013671875,-0.00341796875,0.000000e+00,0.0029296875,0.01171875,3.125000e-02,4.687500e-02,9.375000e-02,1.562500e-01,2.187500e-01,5.000000e-01}>

// CHECK: !qElemType = !quant.quantile<u4:f8E5M2:f16, {-5.000000e-01,-2.187500e-01,-1.562500e-01,-7.812500e-02,-3.906250e-02,-0.013671875,-0.00341796875,0.000000e+00,0.0029296875,0.01171875,3.125000e-02,4.687500e-02,9.375000e-02,1.562500e-01,2.187500e-01,5.000000e-01}:5.000000e-01>

// CHECK-LABEL: @StaticScaleDequantizationForF8E5M2ActNF4WeightsWithQuantCast
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<16x16x1x1xui4>
func.func @StaticScaleDequantizationForF8E5M2ActNF4WeightsWithQuantCast(%input: tensor<1x16x16x16xf16>, %weights: tensor<16x16x1x1xui4>) -> tensor<1x16x16x16xf16> {
    %scale = const.Declare tensor<1x16x1x1xf16> = dense<0.5> : tensor<1x16x1x1xf16>

    %quant_cast = IE.QuantizeCast(%weights) {dstElemType = !quantileFloatType} : tensor<16x16x1x1xui4> -> tensor<16x16x1x1x!quantileFloatType>

    %weights_f16 = IE.Convert(%quant_cast) { dstElemType = f16 } : tensor<16x16x1x1x!quantileFloatType> -> tensor<16x16x1x1xf16>

    %multiply = IE.Multiply(%weights_f16, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

    %conv = IE.Convolution(%input, %multiply) {
              dilations = [1, 1],
              pads_begin = [0, 0],
              pads_end = [0, 0],
              strides = [1, 1]
          } : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>
              -> tensor<1x16x16x16xf16>

    return %conv : tensor<1x16x16x16xf16>

    // CHECK: [[QUANTCAST_1:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !QuantileFloat.quantileFloat<ui4:f8E5M2, {-5.000000e-01,-2.187500e-01,-1.562500e-01,-7.812500e-02,-3.906250e-02,-0.013671875,-0.00341796875,0.000000e+00,0.0029296875,0.01171875,3.125000e-02,4.687500e-02,9.375000e-02,1.562500e-01,2.187500e-01,5.000000e-01}>} : tensor<16x16x1x1xui4> -> tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f8E5M2, {-5.000000e-01,-2.187500e-01,-1.562500e-01,-7.812500e-02,-3.906250e-02,-0.013671875,-0.00341796875,0.000000e+00,0.0029296875,0.01171875,3.125000e-02,4.687500e-02,9.375000e-02,1.562500e-01,2.187500e-01,5.000000e-01}>>
    // CHECK: [[QUANTCAST_2:%.+]] = IE.QuantizeCast([[QUANTCAST_1]]) {dstElemType = !qElemType} : tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f8E5M2, {-5.000000e-01,-2.187500e-01,-1.562500e-01,-7.812500e-02,-3.906250e-02,-0.013671875,-0.00341796875,0.000000e+00,0.0029296875,0.01171875,3.125000e-02,4.687500e-02,9.375000e-02,1.562500e-01,2.187500e-01,5.000000e-01}>> -> tensor<16x16x1x1x!qElemType>
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QUANTCAST_2]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    // CHECK: return [[CONV]]
}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

// CHECK: !quant.quantile<u4:f16:f32:0, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:{0.20000000298023224:100,0.30000001192092896:100,0.40000000596046448:100,5.000000e-01:100}>

// CHECK-LABEL: @StaticMultiScaleShiftNF4Dequantization
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @StaticMultiScaleShiftNF4Dequantization(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3x!quantileFloatType>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<4x1x1x1xf32> = dense<[0.2, 0.3, 0.4, 0.5]> : tensor<4xf32>, [#const.Reshape<[4, 1, 1, 1]>]
  %shift = const.Declare tensor<1x1x1x1xf32> = dense<100.0> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%weights) {dstElemType = f32} : tensor<4x4x3x3x!quantileFloatType> -> tensor<4x4x3x3xf32>
  %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<4x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %conv : tensor<1x4x28x28xf32>

  // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f32} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf32>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  // CHECK: return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u2:f16:0, {0.199951171875:2,0.300048828125:2,0.39990234375:2,5.000000e-01:2}>

// CHECK-LABEL: @StaticScaleShiftU2DequantizationWithSymmetricZP
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui2>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @StaticScaleShiftU2DequantizationWithSymmetricZP(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xui2>) -> tensor<1x4x28x28xf16> {
  %scale = const.Declare tensor<4x1x1x1xf16> = dense<[0.2, 0.3, 0.4, 0.5]> : tensor<4xf16>, [#const.Reshape<[4, 1, 1, 1]>]
  %shift = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>

  %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3xf16>
  %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>

  // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANTCAST]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>
  // CHECK:  return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u2:f16:0, {0.199951171875:1,0.300048828125:1,0.39990234375:1,5.000000e-01:1}>

// CHECK-LABEL: @StaticScaleShiftU2DequantizationWithAsymmetricZP
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui2>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @StaticScaleShiftU2DequantizationWithAsymmetricZP(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xui2>) -> tensor<1x4x28x28xf16> {
  %scale = const.Declare tensor<4x1x1x1xf16> = dense<[0.2, 0.3, 0.4, 0.5]> : tensor<4xf16>, [#const.Reshape<[4, 1, 1, 1]>]
  %shift = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>

  %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3xf16>
  %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>

  // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANTCAST]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>
  // CHECK:  return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<i2:f16:0, {0.199951171875,0.300048828125,0.39990234375,5.000000e-01}>

// CHECK-LABEL: @StaticScaleShiftI2Dequantization
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xsi2>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @StaticScaleShiftI2Dequantization(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xsi2>) -> tensor<1x4x28x28xf16> {
  %scale = const.Declare tensor<4x1x1x1xf16> = dense<[0.2, 0.3, 0.4, 0.5]> : tensor<4xf16>, [#const.Reshape<[4, 1, 1, 1]>]

  %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xsi2> -> tensor<4x4x3x3xf16>
  %multiply = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>

  // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xsi2> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANTCAST]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>
  // CHECK:  return [[CONV]]
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
  // CHECK-SAME{LITERAL}: = dense<[[[[1.000000e-01]], [[2.000000e-01]], [[3.000000e-01]], [[4.000000e-01]]]]>
  // CHECK:  [[SHIFT:%.+]] = const.Declare tensor<1x4x1x1xf32>
  // CHECK-SAME{LITERAL}: = dense<[[[[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]]]]>

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

// CHECK-LABEL: @DynamicScaleShiftDequantizationOnInvalidShiftType
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<16x16x1x1xsi4>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<1x16x1x1xf16>,
// CHECK-SAME:     [[SHIFT:%.+]]: tensor<1x16x1x1xsi4>
func.func @DynamicScaleShiftDequantizationOnInvalidShiftType(%input: tensor<1x16x16x16xf16>, %weights: tensor<16x16x1x1xsi4>, %scale: tensor<1x16x1x1xf16>, %zp: tensor<1x16x1x1xsi4>) -> tensor<1x16x16x16xf16> {
    %weights_f16 = IE.Convert(%weights) { dstElemType = f16 } : tensor<16x16x1x1xsi4> -> tensor<16x16x1x1xf16>
    %zp_f16 = IE.Convert(%zp) { dstElemType = f16 } : tensor<1x16x1x1xsi4> -> tensor<1x16x1x1xf16>
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

    // CHECK:  [[WEIGHTS_CONVERT:%.+]] = IE.Convert([[WEIGHTS]])
    // CHECK:  [[SHIFT_CONVERT:%.+]] = IE.Convert([[SHIFT]])
    // CHECK:  [[SUBTRACT:%.+]] = IE.Subtract([[WEIGHTS_CONVERT]], [[SHIFT_CONVERT]])
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[SUBTRACT]], [[SCALE]])
    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[MULTIPLY]])

    // CHECK:  return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<i4:f32, 1.000000e+00>

// CHECK-LABEL: @DynamicScaleDequantizationWithTranspose
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x1x128xf32>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<28x512x128xsi4>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<28x1x512xf32>
func.func @DynamicScaleDequantizationWithTranspose(%input: tensor<1x1x128xf32>, %weights: tensor<28x512x128xsi4>, %scale: tensor<28x1x512xf32>) -> tensor<28x1x512xf32> {
    %weights_f32 = IE.Convert(%weights) {dstElemType = f32} : tensor<28x512x128xsi4> -> tensor<28x512x128xf32>
    %transpose = IE.Transpose(%weights_f32) {order_value = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : tensor<28x512x128xf32> -> tensor<28x128x512xf32>
    %multiply = IE.Multiply(%transpose, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<28x128x512xf32>, tensor<28x1x512xf32> -> tensor<28x128x512xf32>
    %matmul = IE.MatMul(%input, %multiply) : tensor<1x1x128xf32>, tensor<28x128x512xf32> -> tensor<28x1x512xf32>

    return %matmul : tensor<28x1x512xf32>

    // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<28x512x128xsi4> -> tensor<28x512x128x!qElemType>
    // CHECK:  [[TRANSPOSE:%.+]] = IE.Transpose([[QUANT_CAST]]) {order_value = #map} : tensor<28x512x128x!qElemType> -> tensor<28x128x512x!qElemType>
    // CHECK:  [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[TRANSPOSE]], [[SCALE]]) {dstElemType = f32} : tensor<28x128x512x!qElemType>, tensor<28x1x512xf32> -> tensor<28x128x512xf32>
    // CHECK:  [[MATMUL:%.+]] = IE.MatMul([[INPUT]], [[DYN_DEQUANT]]) : tensor<1x1x128xf32>, tensor<28x128x512xf32> -> tensor<28x1x512xf32>
    // CHECK:  return [[MATMUL]] : tensor<28x1x512xf32>
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

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

// CHECK: !qElemType = !quant.quantile<u4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:1.000000e+00>

// CHECK-LABEL: @DynamicScaleDequantizationForNF4Weights
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<1x16x1x1xf16>
func.func @DynamicScaleDequantizationForNF4Weights(%input: tensor<1x16x16x16xf16>, %weights: tensor<16x16x1x1x!quantileFloatType>, %scale: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %weights_f16 = IE.Convert(%weights) { dstElemType = f16 } : tensor<16x16x1x1x!quantileFloatType> -> tensor<16x16x1x1xf16>
    %multiply = IE.Multiply(%weights_f16, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
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
    // CHECK-SAME:     : tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<16x16x1x1x!qElemType>

    // CHECK:  [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANT_CAST]], [[SCALE]]) {dstElemType = f16}
    // CHECK-SAME:     : tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DYN_DEQUANT]])
    // CHECK-SAME:     {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:     : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    // CHECK:  return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

// CHECK: !quant.quantile<u4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:1.000000e+00>

// CHECK-LABEL: @DynamicScaleDequantizationForFP16ActNF4WeightsWithQuantCast
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<16x16x1x1xui4>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<1x16x1x1xf16>
func.func @DynamicScaleDequantizationForFP16ActNF4WeightsWithQuantCast(%input: tensor<1x16x16x16xf16>, %weights: tensor<16x16x1x1xui4>, %scale: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %quant_cast = IE.QuantizeCast(%weights) {dstElemType = !quantileFloatType} : tensor<16x16x1x1xui4> -> tensor<16x16x1x1x!quantileFloatType>

    %weights_f16 = IE.Convert(%quant_cast) { dstElemType = f16 } : tensor<16x16x1x1x!quantileFloatType> -> tensor<16x16x1x1xf16>

    %multiply = IE.Multiply(%weights_f16, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

    %conv = IE.Convolution(%input, %multiply) {
              dilations = [1, 1],
              pads_begin = [0, 0],
              pads_end = [0, 0],
              strides = [1, 1]
          } : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>
              -> tensor<1x16x16x16xf16>

    return %conv : tensor<1x16x16x16xf16>

    // CHECK: [[QUANTCAST_1:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>} : tensor<16x16x1x1xui4> -> tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>
    // CHECK: [[QUANTCAST_2:%.+]] = IE.QuantizeCast([[QUANTCAST_1]]) {dstElemType = !qElemType} : tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<16x16x1x1x!qElemType>
    // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANTCAST_2]], [[SCALE]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DYN_DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    // CHECK: return [[CONV]]
}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

// CHECK: !quant.quantile<u4:f8E4M3FN:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:1.000000e+00>

// CHECK-LABEL: @DynamicScaleDequantizationForF8E4M3FNActNF4WeightsWithQuantCast
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<16x16x1x1xui4>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<1x16x1x1xf16>
func.func @DynamicScaleDequantizationForF8E4M3FNActNF4WeightsWithQuantCast(%input: tensor<1x16x16x16xf16>, %weights: tensor<16x16x1x1xui4>, %scale: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %quant_cast = IE.QuantizeCast(%weights) {dstElemType = !quantileFloatType} : tensor<16x16x1x1xui4> -> tensor<16x16x1x1x!quantileFloatType>

    %weights_f16 = IE.Convert(%quant_cast) { dstElemType = f16 } : tensor<16x16x1x1x!quantileFloatType> -> tensor<16x16x1x1xf16>

    %multiply = IE.Multiply(%weights_f16, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

    %conv = IE.Convolution(%input, %multiply) {
              dilations = [1, 1],
              pads_begin = [0, 0],
              pads_end = [0, 0],
              strides = [1, 1]
          } : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>
              -> tensor<1x16x16x16xf16>

    return %conv : tensor<1x16x16x16xf16>

    // CHECK: [[QUANTCAST_1:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>} : tensor<16x16x1x1xui4> -> tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>
    // CHECK: [[QUANTCAST_2:%.+]] = IE.QuantizeCast([[QUANTCAST_1]]) {dstElemType = !qElemType} : tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f8E4M3FN, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<16x16x1x1x!qElemType>
    // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANTCAST_2]], [[SCALE]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DYN_DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    // CHECK: return [[CONV]]
}

// -----

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f8E5M2, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

// CHECK: !quant.quantile<u4:f8E5M2:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:1.000000e+00>

// CHECK-LABEL: @DynamicScaleDequantizationForF8E5M2ActNF4WeightsWithQuantCast
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<16x16x1x1xui4>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<1x16x1x1xf16>
func.func @DynamicScaleDequantizationForF8E5M2ActNF4WeightsWithQuantCast(%input: tensor<1x16x16x16xf16>, %weights: tensor<16x16x1x1xui4>, %scale: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %quant_cast = IE.QuantizeCast(%weights) {dstElemType = !quantileFloatType} : tensor<16x16x1x1xui4> -> tensor<16x16x1x1x!quantileFloatType>

    %weights_f16 = IE.Convert(%quant_cast) { dstElemType = f16 } : tensor<16x16x1x1x!quantileFloatType> -> tensor<16x16x1x1xf16>

    %multiply = IE.Multiply(%weights_f16, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

    %conv = IE.Convolution(%input, %multiply) {
              dilations = [1, 1],
              pads_begin = [0, 0],
              pads_end = [0, 0],
              strides = [1, 1]
          } : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>
              -> tensor<1x16x16x16xf16>

    return %conv : tensor<1x16x16x16xf16>

    // CHECK: [[QUANTCAST_1:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !QuantileFloat.quantileFloat<ui4:f8E5M2, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>} : tensor<16x16x1x1xui4> -> tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f8E5M2, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>
    // CHECK: [[QUANTCAST_2:%.+]] = IE.QuantizeCast([[QUANTCAST_1]]) {dstElemType = !qElemType} : tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f8E5M2, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<16x16x1x1x!qElemType>
    // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANTCAST_2]], [[SCALE]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DYN_DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    // CHECK: return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u2:f16:0, {0.0999755859375,0.199951171875,0.300048828125,0.39990234375}>

// CHECK-LABEL: @StaticScaleU2Dequantization
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui2>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @StaticScaleU2Dequantization(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xui2>) -> tensor<1x4x28x28xf16> {
  %scale = const.Declare tensor<4x1x1x1xf16> = dense<[0.1, 0.2, 0.3, 0.4]> : tensor<4xf16>, [#const.Reshape<[4, 1, 1, 1]>]

  %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3xf16>
  %multiply = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>

  // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANTCAST]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>
  // CHECK:  return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u2:f16, 1.000000e+00>

// CHECK-LABEL: @StaticScaleU2MultiAxisDequantization
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui2>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @StaticScaleU2MultiAxisDequantization(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xui2>) -> tensor<1x4x28x28xf16> {
  %scale = const.Declare tensor<4x4x1x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<16xf16>, [#const.Reshape<[4, 4, 1, 1]>]

  %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3xf16>
  %multiply = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<4x4x1x1xf16> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>
  // CHECK:  [[SCALE:%.+]] = const.Declare tensor<4x4x1x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01, 9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<16xf16>, [#const.Reshape<[4, 4, 1, 1]>]
  // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.DynamicDequantize([[QUANTCAST]], [[SCALE]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType>, tensor<4x4x1x1xf16> -> tensor<4x4x3x3xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>
  // CHECK:  return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u2:f16, 0.1500244140625:2>

// CHECK-LABEL: @StaticScaleShiftU2DequantizationWithSplatZP
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui2>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @StaticScaleShiftU2DequantizationWithSplatZP(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xui2>) -> tensor<1x4x28x28xf16> {
  %scale = const.Declare tensor<4x1x1x1xf16> = dense<0.15> : tensor<4x1x1x1xf16>
  %shift = const.Declare tensor<4x1x1x1xf16> = dense_resource<blob> : tensor<4x1x1x1xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f16>]

  %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3xf16>
  %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>

  // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANTCAST]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>
  // CHECK:  return [[CONV]]
}

{-#
  dialect_resources: {
    builtin: {
      // Note: 0b_10_10_10_10 => [2, 2, 2, 2]
      blob: "0x01000000AA"
    }
  }
#-}

// -----

// CHECK: !qElemType = !quant.uniform<u2:f16, 1.000000e+00>

// CHECK-LABEL: @StaticScaleShiftU2DequantizationWithNonSplatZP
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui2>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @StaticScaleShiftU2DequantizationWithNonSplatZP(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xui2>) -> tensor<1x4x28x28xf16> {
  %scale = const.Declare tensor<4x1x1x1xf16> = dense<1.0> : tensor<4x1x1x1xf16>
  %shift = const.Declare tensor<4x1x1x1xf16> = dense_resource<blob> : tensor<4x1x1x1xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f16>]

  %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3xf16>
  %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>


  // CHECK:  [[SHIFT:%.+]] = const.Declare tensor<4x1x1x1xui2> = dense_resource<blob> : tensor<4x1x1x1xui2>
  // CHECK:  [[SCALE:%.+]] = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00> : tensor<4x1x1x1xf16>
  // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.DynamicDequantize([[QUANTCAST]], [[SCALE]], [[SHIFT]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType>, tensor<4x1x1x1xf16>, tensor<4x1x1x1xui2> -> tensor<4x4x3x3xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>
  // CHECK:  return [[CONV]]
}

{-#
  dialect_resources: {
    builtin: {
      // Note: 0b_00_01_10_10 => [2, 2, 1, 0]
      blob: "0x010000001A"
    }
  }
#-}

// -----

// CHECK: !qElemType = !quant.uniform<u2:f16, 1.000000e+00>

// CHECK-LABEL: @DynamicScaleU2Dequantization
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui2>
// CHECK-SAME:      [[SCALE:%.+]]: tensor<4x1x1x1xf16>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @DynamicScaleU2Dequantization(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xui2>, %scale: tensor<4x1x1x1xf16>) -> tensor<1x4x28x28xf16> {
  %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3xf16>
  %multiply = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>

  // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.DynamicDequantize([[QUANTCAST]], [[SCALE]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>
  // CHECK:  return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u2:f16, 1.000000e+00>

// CHECK-LABEL: @DynamicScaleShiftU2Dequantization
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui2>
// CHECK-SAME:      [[SCALE:%.+]]: tensor<4x1x1x1xf16>
// CHECK-SAME:      [[SHIFT:%.+]]: tensor<4x1x1x1xui2>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @DynamicScaleShiftU2Dequantization(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xui2>, %scale: tensor<4x1x1x1xf16>, %shift: tensor<4x1x1x1xui2>) -> tensor<1x4x28x28xf16> {
  %weights_convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3xf16>
  %shift_convert = IE.Convert(%shift) {dstElemType = f16} : tensor<4x1x1x1xui2> -> tensor<4x1x1x1xf16>
  %subtract = IE.Subtract(%weights_convert, %shift_convert) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>

  // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xui2> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.DynamicDequantize([[QUANTCAST]], [[SCALE]], [[SHIFT]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType>, tensor<4x1x1x1xf16>, tensor<4x1x1x1xui2> -> tensor<4x4x3x3xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>
  // CHECK:  return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<i2:f16, 1.000000e+00>

// CHECK-LABEL: @DynamicScaleI2Dequantization
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xsi2>
// CHECK-SAME:      [[SCALE:%.+]]: tensor<4x1x1x1xf16>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @DynamicScaleI2Dequantization(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xsi2>, %scale: tensor<4x1x1x1xf16>) -> tensor<1x4x28x28xf16> {
  %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xsi2> -> tensor<4x4x3x3xf16>
  %multiply = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %conv : tensor<1x4x28x28xf16>

  // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xsi2> -> tensor<4x4x3x3x!qElemType>
  // CHECK:  [[DEQUANT:%.+]] = IE.DynamicDequantize([[QUANTCAST]], [[SCALE]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType>, tensor<4x1x1x1xf16> -> tensor<4x4x3x3xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>
  // CHECK:  return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f32, 1.000000e+00>

// CHECK-LABEL: @DynamicScaleDequantizationScaleOnInput1
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<256x2048xsi8>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<256x1xf32>,
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x2048xf32>
func.func @DynamicScaleDequantizationScaleOnInput1(%weights: tensor<256x2048xsi8>, %scale: tensor<256x1xf32>, %input: tensor<1x2048xf32>) -> tensor<1x256xf32> {
    %weights_f32 = IE.Convert(%weights) {dstElemType = f32} : tensor<256x2048xsi8> -> tensor<256x2048xf32>
    %multiply = IE.Multiply(%scale, %weights_f32) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<256x1xf32>, tensor<256x2048xf32> -> tensor<256x2048xf32>

    %fc = IE.FullyConnected(%input, %multiply) : tensor<1x2048xf32>, tensor<256x2048xf32> -> tensor<1x256xf32>

    return %fc: tensor<1x256xf32>

    // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<256x2048xsi8> -> tensor<256x2048x!qElemType>
    // CHECK:  [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANT_CAST]], [[SCALE]]) {dstElemType = f32} : tensor<256x2048x!qElemType>, tensor<256x1xf32> -> tensor<256x2048xf32>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT]], [[DYN_DEQUANT]]) : tensor<1x2048xf32>, tensor<256x2048xf32> -> tensor<1x256xf32>

    // CHECK:  return [[FC]] : tensor<1x256xf32>
}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f32, 1.000000e+00>

// CHECK-LABEL: @DynamicScaleDequantizationForScaleWithConvert
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<1792x2048xsi8>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<1x2048xf16>,
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1024x1792xf32>
func.func @DynamicScaleDequantizationForScaleWithConvert(%weights: tensor<1792x2048xsi8>, %scale: tensor<1x2048xf16>, %input: tensor<1024x1792xf32>) -> tensor<1024x2048xf32> {
    %weights_f32 = IE.Convert(%weights) {dstElemType = f32} : tensor<1792x2048xsi8> -> tensor<1792x2048xf32>
    %scale_f32 = IE.Convert(%scale) {dstElemType = f32} : tensor<1x2048xf16> -> tensor<1x2048xf32>
    %multiply = IE.Multiply(%weights_f32, %scale_f32) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1792x2048xf32>, tensor<1x2048xf32> -> tensor<1792x2048xf32>
    %reshape0 = IE.AffineReshape(%multiply) {dim_mapping = [[0], [1, 2]], shape_value = [1792, 8, 256]} : tensor<1792x2048xf32> -> tensor<1792x8x256xf32>
    %transpose = IE.Transpose(%reshape0) {order_value = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : tensor<1792x8x256xf32> -> tensor<8x256x1792xf32>
    %reshape1 = IE.AffineReshape(%transpose) {dim_mapping = [[0], [0], [1]], shape_value = [2048, 1792]} : tensor<8x256x1792xf32> -> tensor<2048x1792xf32>

    %fc = IE.FullyConnected(%input, %reshape1) : tensor<1024x1792xf32>, tensor<2048x1792xf32> -> tensor<1024x2048xf32>

    return %fc: tensor<1024x2048xf32>

    // CHECK:  [[CONVERT:%.+]] = IE.Convert([[SCALE]]) {dstElemType = f32} : tensor<1x2048xf16> -> tensor<1x2048xf32>
    // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<1792x2048xsi8> -> tensor<1792x2048x!qElemType>
    // CHECK:  [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANT_CAST]], [[CONVERT]]) {dstElemType = f32} : tensor<1792x2048x!qElemType>, tensor<1x2048xf32> -> tensor<1792x2048xf32>
    // CHECK:  [[RESHAPE_0:%.+]] = IE.AffineReshape([[DYN_DEQUANT]])
    // CHECK-SAME{LITERAL}:       {dim_mapping = [[0], [1, 2]], shape_value = [1792, 8, 256]} : tensor<1792x2048xf32> -> tensor<1792x8x256xf32>
    // CHECK:  [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE_0]]) {order_value = #HWC} : tensor<1792x8x256xf32> -> tensor<8x256x1792xf32>
    // CHECK:  [[RESHAPE_1:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:       {dim_mapping = [[0], [0], [1]], shape_value = [2048, 1792]} : tensor<8x256x1792xf32> -> tensor<2048x1792xf32>

    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT]], [[RESHAPE_1]]) : tensor<1024x1792xf32>, tensor<2048x1792xf32> -> tensor<1024x2048xf32>

    // CHECK:  return [[FC]] : tensor<1024x2048xf32>
}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f32, 1.000000e+00>

// CHECK-LABEL: @DynamicScaleDequantizationForScaleWithConvertOnInput1
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<1792x2048xsi8>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<1x2048xf16>,
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1024x1792xf32>
func.func @DynamicScaleDequantizationForScaleWithConvertOnInput1(%weights: tensor<1792x2048xsi8>, %scale: tensor<1x2048xf16>, %input: tensor<1024x1792xf32>) -> tensor<1024x2048xf32> {
    %weights_f32 = IE.Convert(%weights) {dstElemType = f32} : tensor<1792x2048xsi8> -> tensor<1792x2048xf32>
    %scale_f32 = IE.Convert(%scale) {dstElemType = f32} : tensor<1x2048xf16> -> tensor<1x2048xf32>
    %multiply = IE.Multiply(%scale_f32, %weights_f32) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2048xf32>, tensor<1792x2048xf32> -> tensor<1792x2048xf32>
    %reshape0 = IE.AffineReshape(%multiply) {dim_mapping = [[0], [1, 2]], shape_value = [1792, 8, 256]} : tensor<1792x2048xf32> -> tensor<1792x8x256xf32>
    %transpose = IE.Transpose(%reshape0) {order_value = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : tensor<1792x8x256xf32> -> tensor<8x256x1792xf32>
    %reshape1 = IE.AffineReshape(%transpose) {dim_mapping = [[0], [0], [1]], shape_value = [2048, 1792]} : tensor<8x256x1792xf32> -> tensor<2048x1792xf32>

    %fc = IE.FullyConnected(%input, %reshape1) : tensor<1024x1792xf32>, tensor<2048x1792xf32> -> tensor<1024x2048xf32>

    return %fc: tensor<1024x2048xf32>

    // CHECK:  [[CONVERT:%.+]] = IE.Convert([[SCALE]]) {dstElemType = f32} : tensor<1x2048xf16> -> tensor<1x2048xf32>
    // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<1792x2048xsi8> -> tensor<1792x2048x!qElemType>
    // CHECK:  [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANT_CAST]], [[CONVERT]]) {dstElemType = f32} : tensor<1792x2048x!qElemType>, tensor<1x2048xf32> -> tensor<1792x2048xf32>
    // CHECK:  [[RESHAPE_0:%.+]] = IE.AffineReshape([[DYN_DEQUANT]])
    // CHECK-SAME{LITERAL}:       {dim_mapping = [[0], [1, 2]], shape_value = [1792, 8, 256]} : tensor<1792x2048xf32> -> tensor<1792x8x256xf32>
    // CHECK:  [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE_0]]) {order_value = #HWC} : tensor<1792x8x256xf32> -> tensor<8x256x1792xf32>
    // CHECK:  [[RESHAPE_1:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:       {dim_mapping = [[0], [0], [1]], shape_value = [2048, 1792]} : tensor<8x256x1792xf32> -> tensor<2048x1792xf32>

    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT]], [[RESHAPE_1]]) : tensor<1024x1792xf32>, tensor<2048x1792xf32> -> tensor<1024x2048xf32>

    // CHECK:  return [[FC]] : tensor<1024x2048xf32>
}

// -----

// CHECK-LABEL: @NotConvertToDequantizeForSignlessType
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<4x4x3x3xi8>, [[INPUT2:%.+]]: tensor<4x4x3x3xf32>

func.func @NotConvertToDequantizeForSignlessType(%arg0: tensor<4x4x3x3xi8>, %arg1: tensor<4x4x3x3xf32>) -> tensor<4x4x3x3xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>
  %convert = IE.Convert(%arg0) {dstElemType = f32} : tensor<4x4x3x3xi8> -> tensor<4x4x3x3xf32>
  %multiply = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>

  return %multiply : tensor<4x4x3x3xf32>

  // CHECK-DAG:  [[CONST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01> : tensor<1x1x1x1xf32>
  // CHECK:  [[CONVERT:%.+]] = IE.Convert
  // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply

  // CHECK: return [[MULTIPLY]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u2:f16, 1.000000e+00>

// CHECK-LABEL: @WaCStaticScaleShiftU2Dequantization
// CHECK-SAME:      [[ACT:%.+]]: tensor<1x3x4xf32>
// CHECK-SAME: -> tensor<1x3x4xf32>
func.func @WaCStaticScaleShiftU2Dequantization(%act : tensor<1x3x4xf32>) -> tensor<1x3x4xf32> {
  %weights = const.Declare tensor<4x2x2xf16> = dense_resource<weights_blob> : tensor<4x2x2xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f16>]
  %scale = const.Declare tensor<4x2x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  %shift = const.Declare tensor<4x2x1xf16> = dense_resource<shift_blob> : tensor<4x2x1xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f16>]

  %1 = IE.Subtract(%weights, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  %5 = IE.Reshape(%act) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  %6 = IE.FullyConnected(%5, %4) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  %7 = IE.Reshape(%6) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  return %7 : tensor<1x3x4xf32>

  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x2x!qElemType> = dense_resource<weights_blob> : tensor<4x2x2xui2>, [#const.CastElemType<!qElemType>]
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  // CHECK-DAG: [[SHIFT:%.+]] = const.Declare tensor<4x2x1xui2> = dense_resource<shift_blob> : tensor<4x2x1xui2>
  // CHECK:     [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[SHIFT]]) {dstElemType = f16} : tensor<4x2x2x!qElemType>, tensor<4x2x1xf16>, tensor<4x2x1xui2> -> tensor<4x2x2xf16>
  // CHECK:     [[RESHAPE_0:%.+]] = IE.AffineReshape([[DYN_DEQUANT]])
  // CHECK-SAME{LITERAL}:           {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  // CHECK:     [[CONVERT:%.+]] = IE.Convert([[RESHAPE_0]]) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  // CHECK:     [[RESHAPE_1:%.+]] = IE.Reshape([[ACT]]) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  // CHECK:     [[FC:%.+]] = IE.FullyConnected([[RESHAPE_1]], [[CONVERT]]) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  // CHECK:     [[RESHAPE_2:%.+]] = IE.Reshape([[FC]]) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  // CHECK:     return [[RESHAPE_2]]
}

{-#
  dialect_resources: {
    builtin: {
      weights_blob: "0x010000001AA12345",
      shift_blob: "0x010000001AA1"
    }
  }
#-}

// -----

// CHECK-LABEL: @NotConvertToDequantizeForUnsupportedU2Case
// CHECK-SAME:      [[ACT:%.+]]:  tensor<1x48xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<3072x48x1xui2>
// CHECK-SAME:      [[SCALE:%.+]]: tensor<3072x48x1xf16>
// CHECK-SAME: -> tensor<1x3072xf16>
func.func @NotConvertToDequantizeForUnsupportedU2Case(%act: tensor<1x48xf16>, %weights: tensor<3072x48x1xui2>, %scale: tensor<3072x48x1xf16>) -> tensor<1x3072xf16> {
  %0 = IE.Convert(%weights) {dstElemType = f16} : tensor<3072x48x1xui2> -> tensor<3072x48x1xf16>
  %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3072x48x1xf16>, tensor<3072x48x1xf16> -> tensor<3072x48x1xf16>
  %2 = IE.Reshape(%1) {shape_value = [3072, 48]} : tensor<3072x48x1xf16> -> tensor<3072x48xf16>
  %3 = IE.FullyConnected(%act, %2) : tensor<1x48xf16>, tensor<3072x48xf16> -> tensor<1x3072xf16>
  return %3 : tensor<1x3072xf16>

  // CHECK: [[CONVERT:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f16} : tensor<3072x48x1xui2> -> tensor<3072x48x1xf16>
  // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3072x48x1xf16>, tensor<3072x48x1xf16> -> tensor<3072x48x1xf16>
  // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[MULTIPLY]]) {shape_value = [3072, 48]} : tensor<3072x48x1xf16> -> tensor<3072x48xf16>
  // CHECK: [[FC:%.+]] = IE.FullyConnected([[ACT]], [[RESHAPE]]) : tensor<1x48xf16>, tensor<3072x48xf16> -> tensor<1x3072xf16>
  // CHECK: return [[FC]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x010000001A"
    }
  }
#-}
