//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-weights-to-u8 --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
!qElemType1 = !quant.uniform<i8:f32, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
// CHECK: !qElemType1 = !quant.uniform<i8:f32, 1.000000e+00>
// CHECK:      func.func @KeepI8ForSI8WeightsAsInputs
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x256x2048xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<2048x2048xsi8>
func.func @KeepI8ForSI8WeightsAsInputs(%arg0: tensor<1x256x2048xf32>, %arg1: tensor<2048x2048xsi8>) -> tensor<256x2048x1x1xf16> {
  %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
  %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
  %3 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>
  %4 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048x!qElemType> -> tensor<256x2048x1x1x!qElemType>
  %5 = IE.AffineReshape(%3) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048x1x1x!qElemType1>
  %6 = IE.Convolution(%4, %5) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<2048x2048x1x1x!qElemType1> -> tensor<256x2048x1x1xf16>

  return %6 : tensor<256x2048x1x1xf16>

  // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
  // CHECK:       [[CONVERT:%.+]] = IE.Convert([[RESHAPE_0]]) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>
  // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[QUANTIZE]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048x!qElemType> -> tensor<256x2048x1x1x!qElemType>
  // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[QUANTIZE_CAST]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048x1x1x!qElemType1>
  // CHECK:       [[CONV_RESULT:%.+]] = IE.Convolution([[RESHAPE_1]], [[RESHAPE_2]])
  // CHECK-SAME:            {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<2048x2048x1x1x!qElemType1> -> tensor<256x2048x1x1xf16>

  // CHECK:       return [[CONV_RESULT]] : tensor<256x2048x1x1xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
!qElemType1 = !quant.uniform<i8:f32, 1.000000e+00>

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK: !qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
// CHECK: !qElemType1 = !quant.uniform<i8:f32, 1.000000e+00>
// CHECK:      func.func @KeepGroupConvAsI8ForChildConvHasSI8WeightsAsInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x256x2048x1xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x2048xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<2048x2048xsi8>
func.func @KeepGroupConvAsI8ForChildConvHasSI8WeightsAsInputs(%arg0: tensor<1x256x2048x1xf16>, %arg1: tensor<1x1x2048xf16>, %arg2: tensor<2048x2048xsi8>) -> tensor<256x2048x1x1xf16> {
  %0 = IE.Transpose(%arg0) {order_value = #NHCW} : tensor<1x256x2048x1xf16> -> tensor<1x2048x256x1xf16>
  %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0], [0, 1, 2, 3]], shape_value = [2048, 1, 1, 1]} : tensor<1x1x2048xf16> -> tensor<2048x1x1x1xf16>
  %2 = IE.GroupConvolution(%0, %1) {dilations = [1, 1], groups = 2048 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2048x256x1xf16>, tensor<2048x1x1x1xf16> -> tensor<1x2048x256x1x!qElemType>
  %3 = IE.Transpose(%2) {order_value = #NHCW} : tensor<1x2048x256x1x!qElemType> -> tensor<1x256x2048x1x!qElemType>
  %4 = IE.QuantizeCast(%arg2) {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>
  %5 = IE.AffineReshape(%3) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x256x2048x1x!qElemType> -> tensor<256x2048x1x1x!qElemType>
  %6 = IE.AffineReshape(%4) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048x1x1x!qElemType1>
  %7 = IE.Convolution(%5, %6) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<2048x2048x1x1x!qElemType1> -> tensor<256x2048x1x1xf16>

  return %7 : tensor<256x2048x1x1xf16>

  // CHECK:       [[TRANSPOSE_0:%.+]] = IE.Transpose([[INPUT_0]])
  // CHECK-SAME{LITERAL}:   {order_value = #NHCW} : tensor<1x256x2048x1xf16> -> tensor<1x2048x256x1xf16>
  // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT_1]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0, 1, 2, 3]], shape_value = [2048, 1, 1, 1]} : tensor<1x1x2048xf16> -> tensor<2048x1x1x1xf16>
  // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[TRANSPOSE_0]], [[RESHAPE_0]])
  // CHECK-SAME:            {dilations = [1, 1], groups = 2048 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2048x256x1xf16>, tensor<2048x1x1x1xf16> -> tensor<1x2048x256x1x!qElemType>
  // CHECK:       [[TRANSPOSE_1:%.+]] = IE.Transpose([[GROUP_CONV]])
  // CHECK-SAME{LITERAL}:   {order_value = #NHCW} : tensor<1x2048x256x1x!qElemType> -> tensor<1x256x2048x1x!qElemType>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]])
  // CHECK-SAME{LITERAL}:   {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>
  // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[TRANSPOSE_1]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x256x2048x1x!qElemType> -> tensor<256x2048x1x1x!qElemType>
  // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[QUANTIZE_CAST]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048x1x1x!qElemType1>
  // CHECK:       [[CONV_RESULT:%.+]] = IE.Convolution([[RESHAPE_1]], [[RESHAPE_2]])
  // CHECK-SAME:            {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<2048x2048x1x1x!qElemType1> -> tensor<256x2048x1x1xf16>

  // CHECK:       return [[CONV_RESULT]] : tensor<256x2048x1x1xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
!qElemType1 = !quant.uniform<i8:f32, 1.000000e+00>

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK: !qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
// CHECK: !qElemType1 = !quant.uniform<i8:f32, 1.000000e+00>
// CHECK:      func.func @KeepI8ForAllUsersHaveSI8WeightsAsInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x256x2048xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x2048xf16>
// CHECK-SAME:      [[WEIGHTS_0:%.+]]: tensor<2048x2048xsi8>
// CHECK-SAME:      [[WEIGHTS_1:%.+]]: tensor<1024x2048xsi8>
func.func @KeepI8ForAllUsersHaveSI8WeightsAsInputs(%arg0: tensor<1x1x256x2048xf16>, %arg1: tensor<1x1x2048xf16>, %arg2: tensor<2048x2048xsi8>, %arg3: tensor<1024x2048xsi8>) -> (tensor<256x2048x1x1xf16>, tensor<256x1024x1x1xf16>) {
  %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 256, 2048, 1]} : tensor<1x1x256x2048xf16> -> tensor<1x256x2048x1xf16>
  %1 = IE.Transpose(%0) {order_value = #NHCW} : tensor<1x256x2048x1xf16> -> tensor<1x2048x256x1xf16>
  %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0], [0, 1, 2, 3]], shape_value = [2048, 1, 1, 1]} : tensor<1x1x2048xf16> -> tensor<2048x1x1x1xf16>
  %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 2048 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2048x256x1xf16>, tensor<2048x1x1x1xf16> -> tensor<1x2048x256x1xf16>
  %4 = IE.Transpose(%3) {order_value = #NHCW} : tensor<1x2048x256x1xf16> -> tensor<1x256x2048x1xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType} : tensor<1x256x2048x1xf16> -> tensor<1x256x2048x1x!qElemType>
  %6 = IE.QuantizeCast(%arg2) {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>
  %7 = IE.AffineReshape(%5) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x256x2048x1x!qElemType> -> tensor<256x2048x1x1x!qElemType>
  %8 = IE.AffineReshape(%6) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048x1x1x!qElemType1>
  %9 = IE.Convolution(%7, %8) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<2048x2048x1x1x!qElemType1> -> tensor<256x2048x1x1xf16>

  %10 = IE.QuantizeCast(%arg3) {dstElemType = !qElemType1} : tensor<1024x2048xsi8> -> tensor<1024x2048x!qElemType1>
  %11 = IE.AffineReshape(%10) {dim_mapping = [[0], [1, 2, 3]], shape_value = [1024, 2048, 1, 1]} : tensor<1024x2048x!qElemType1> -> tensor<1024x2048x1x1x!qElemType1>
  %12 = IE.Convolution(%7, %11) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<1024x2048x1x1x!qElemType1> -> tensor<256x1024x1x1xf16>

  return %9, %12 : tensor<256x2048x1x1xf16>, tensor<256x1024x1x1xf16>

  // CHECK: [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT_0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 256, 2048, 1]} : tensor<1x1x256x2048xf16> -> tensor<1x256x2048x1xf16>
  // CHECK: [[TRANSPOSE_0:%.+]] = IE.Transpose([[RESHAPE_0]])
  // CHECK-SAME{LITERAL}: {order_value = #NHCW} : tensor<1x256x2048x1xf16> -> tensor<1x2048x256x1xf16>
  // CHECK: [[RESHAPE_1:%.+]] = IE.AffineReshape([[INPUT_1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0, 1, 2, 3]], shape_value = [2048, 1, 1, 1]} : tensor<1x1x2048xf16> -> tensor<2048x1x1x1xf16>
  // CHECK: [[GROUP_CONV:%.+]] = IE.GroupConvolution([[TRANSPOSE_0]], [[RESHAPE_1]])
  // CHECK-SAME: {dilations = [1, 1], groups = 2048 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2048x256x1xf16>, tensor<2048x1x1x1xf16> -> tensor<1x2048x256x1xf16>
  // CHECK: [[TRANSPOSE_1:%.+]] = IE.Transpose([[GROUP_CONV]])
  // CHECK-SAME{LITERAL}: {order_value = #NHCW} : tensor<1x2048x256x1xf16> -> tensor<1x256x2048x1xf16>
  // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[TRANSPOSE_1]])
  // CHECK-SAME: {dstElemType = !qElemType} : tensor<1x256x2048x1xf16> -> tensor<1x256x2048x1x!qElemType>
  // CHECK: [[QUANTIZE_CAST_0:%.+]] = IE.QuantizeCast([[WEIGHTS_0]])
  // CHECK-SAME: {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>
  // CHECK: [[RESHAPE_2:%.+]] = IE.AffineReshape([[QUANTIZE]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x256x2048x1x!qElemType> -> tensor<256x2048x1x1x!qElemType>
  // CHECK: [[RESHAPE_3:%.+]] = IE.AffineReshape([[QUANTIZE_CAST_0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048x1x1x!qElemType1>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[RESHAPE_2]], [[RESHAPE_3]])
  // CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<2048x2048x1x1x!qElemType1> -> tensor<256x2048x1x1xf16>

  // CHECK: [[QUANTIZE_CAST_1:%.+]] = IE.QuantizeCast([[WEIGHTS_1]])
  // CHECK-SAME: {dstElemType = !qElemType1} : tensor<1024x2048xsi8> -> tensor<1024x2048x!qElemType1>
  // CHECK: [[RESHAPE_4:%.+]] = IE.AffineReshape([[QUANTIZE_CAST_1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [1024, 2048, 1, 1]} : tensor<1024x2048x!qElemType1> -> tensor<1024x2048x1x1x!qElemType1>
  // CHECK: [[CONV_12:%.+]] = IE.Convolution([[RESHAPE_2]], [[RESHAPE_4]])
  // CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<1024x2048x1x1x!qElemType1> -> tensor<256x1024x1x1xf16>

  // CHECK: return [[CONV]], [[CONV_12]] : tensor<256x2048x1x1xf16>, tensor<256x1024x1x1xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
!qElemType1 = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL:   @KeepI4ForSI4WeightsAsInputs
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x256x2048xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<2048x2048xsi4>
func.func @KeepI4ForSI4WeightsAsInputs(%arg0: tensor<1x256x2048xf32>, %arg1: tensor<2048x2048xsi4>) -> tensor<256x2048x1x1xf16> {
  %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
  %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
  %3 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType1} : tensor<2048x2048xsi4> -> tensor<2048x2048x!qElemType1>
  %4 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048x!qElemType> -> tensor<256x2048x1x1x!qElemType>
  %5 = IE.AffineReshape(%3) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048x1x1x!qElemType1>
  %6 = IE.Convolution(%4, %5) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<2048x2048x1x1x!qElemType1> -> tensor<256x2048x1x1xf16>

  return %6 : tensor<256x2048x1x1xf16>

  // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
  // CHECK:       [[CONVERT:%.+]] = IE.Convert([[RESHAPE_0]]) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType1} : tensor<2048x2048xsi4> -> tensor<2048x2048x!qElemType1>
  // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[QUANTIZE]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048x!qElemType> -> tensor<256x2048x1x1x!qElemType>
  // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[QUANTIZE_CAST]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048x1x1x!qElemType1>
  // CHECK:       [[CONV_RESULT:%.+]] = IE.Convolution([[RESHAPE_1]], [[RESHAPE_2]])
  // CHECK-SAME:            {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<2048x2048x1x1x!qElemType1> -> tensor<256x2048x1x1xf16>

  // CHECK:       return [[CONV_RESULT]] : tensor<256x2048x1x1xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
!qElemType1 = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL:   @KeepI4ForSlicedSI4WeightsAsInputs
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x256x2048xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4096x2048xsi4>
func.func @KeepI4ForSlicedSI4WeightsAsInputs(%arg0: tensor<1x256x2048xf32>, %arg1: tensor<4096x2048xsi4>) -> tensor<256x2048x1x1xf16> {
  %reshape_0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
  %convert = IE.Convert(%reshape_0) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
  %quantize = IE.Quantize(%convert) {dstElemType = !qElemType} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
  %slice = IE.Slice %arg1 [0, 0] [2048, 2048] : tensor<4096x2048xsi4> to tensor<2048x2048xsi4>
  %quantize_cast = IE.QuantizeCast(%slice) {dstElemType = !qElemType1} : tensor<2048x2048xsi4> -> tensor<2048x2048x!qElemType1>
  %reshape_1 = IE.AffineReshape(%quantize) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048x!qElemType> -> tensor<256x2048x1x1x!qElemType>
  %reshape_2 = IE.AffineReshape(%quantize_cast) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048x1x1x!qElemType1>
  %conv = IE.Convolution(%reshape_1, %reshape_2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<2048x2048x1x1x!qElemType1> -> tensor<256x2048x1x1xf16>

  return %conv : tensor<256x2048x1x1xf16>

  // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
  // CHECK:       [[CONVERT:%.+]] = IE.Convert([[RESHAPE_0]]) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
  // CHECK:       [[SLICE:%.+]] = IE.Slice [[WEIGHTS]] [0, 0] [2048, 2048] : tensor<4096x2048xsi4> to tensor<2048x2048xsi4>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[SLICE]]) {dstElemType = !qElemType1} : tensor<2048x2048xsi4> -> tensor<2048x2048x!qElemType1>
  // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[QUANTIZE]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048x!qElemType> -> tensor<256x2048x1x1x!qElemType>
  // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[QUANTIZE_CAST]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048x1x1x!qElemType1>
  // CHECK:       [[CONV_RESULT:%.+]] = IE.Convolution([[RESHAPE_1]], [[RESHAPE_2]])
  // CHECK-SAME:            {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<2048x2048x1x1x!qElemType1> -> tensor<256x2048x1x1xf16>

  // CHECK:       return [[CONV_RESULT]] : tensor<256x2048x1x1xf16>
}
