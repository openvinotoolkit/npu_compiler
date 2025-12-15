//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-fake-quant %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK: !qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
// CHECK: !qElemType1 = !quant.uniform<i8:f32, 1.000000e+00>
// CHECK-LABEL: @SplitFakeQuantToI8ForSI8WeightsAsInputs
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x256x2048xf32>
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<2048x2048xsi8>
  func.func @SplitFakeQuantToI8ForSI8WeightsAsInputs(%arg0: tensor<1x256x2048xf32>, %arg1: tensor<2048x2048xsi8>) -> tensor<256x2048x1x1xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
    %2 = IE.FakeQuantize(%1, %cst_3, %cst_2, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x256x2048xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x2048xf16>
    %3 = IE.QuantizeCast(%arg1) {dstElemType = !quant.uniform<i8:f32, 1.000000e+00>} : tensor<2048x2048xsi8> -> tensor<2048x2048x!quant.uniform<i8:f32, 1.000000e+00>>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<2048x2048x!quant.uniform<i8:f32, 1.000000e+00>> -> tensor<2048x2048xf16>
    %5 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048xf16> -> tensor<256x2048x1x1xf16>
    %6 = IE.AffineReshape(%4) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
    %7 = IE.Convolution(%5, %6) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

    return %7 : tensor<256x2048x1x1xf16>

    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[RESHAPE0]]) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
    // CHECK: [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x1x256x2048x!qElemType> -> tensor<1x1x256x2048xf16>
    // CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[INPUT1]]) {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>
    // CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048xf16>
    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[DEQUANTIZE0]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048xf16> -> tensor<256x2048x1x1xf16>
    // CHECK: [[RESHAPE2:%.+]] = IE.AffineReshape([[DEQUANTIZE1]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[RESHAPE1]], [[RESHAPE2]])
    // CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

    // CHECK: return [[CONV]] : tensor<256x2048x1x1xf16>
  }

// -----

// CHECK: !qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
// CHECK: !qElemType1 = !quant.uniform<i8:f32, 1.000000e+00>
// CHECK-LABEL: @SplitFakeQuantToI8ForAllUsersHaveSI8WeightsAsInputs
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x256x2048x1xf16>
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<2048x2048xsi8>
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<1024x2048xsi8>
  func.func @SplitFakeQuantToI8ForAllUsersHaveSI8WeightsAsInputs(%arg0: tensor<1x256x2048x1xf16>, %arg1: tensor<2048x2048xsi8>, %arg2: tensor<1024x2048xsi8>) -> (tensor<256x2048x1x1xf16>, tensor<256x1024x1x1xf16>) {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.FakeQuantize(%arg0, %cst_3, %cst_2, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x256x2048x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x256x2048x1xf16>
    %1 = IE.QuantizeCast(%arg1) {dstElemType = !quant.uniform<i8:f32, 1.000000e+00>} : tensor<2048x2048xsi8> -> tensor<2048x2048x!quant.uniform<i8:f32, 1.000000e+00>>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<2048x2048x!quant.uniform<i8:f32, 1.000000e+00>> -> tensor<2048x2048xf16>
    %3 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x256x2048x1xf16> -> tensor<256x2048x1x1xf16>
    %4 = IE.AffineReshape(%2) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
    %5 = IE.Convolution(%3, %4) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

    %6 = IE.QuantizeCast(%arg2) {dstElemType = !quant.uniform<i8:f32, 1.000000e+00>} : tensor<1024x2048xsi8> -> tensor<1024x2048x!quant.uniform<i8:f32, 1.000000e+00>>
    %7 = IE.Dequantize(%6) {dstElemType = f16} : tensor<1024x2048x!quant.uniform<i8:f32, 1.000000e+00>> -> tensor<1024x2048xf16>
    %8 = IE.AffineReshape(%7) {dim_mapping = [[0], [1, 2, 3]], shape_value = [1024, 2048, 1, 1]} : tensor<1024x2048xf16> -> tensor<1024x2048x1x1xf16>
    %9 = IE.Convolution(%3, %8) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<1024x2048x1x1xf16> -> tensor<256x1024x1x1xf16>

    return %5, %9 : tensor<256x2048x1x1xf16>, tensor<256x1024x1x1xf16>

    // CHECK: [[QUANTIZE0:%.+]] = IE.Quantize([[INPUT0]])
    // CHECK-SAME: {dstElemType = !qElemType} : tensor<1x256x2048x1xf16> -> tensor<1x256x2048x1x!qElemType>
    // CHECK: [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZE0]])
    // CHECK-SAME: {dstElemType = f16} : tensor<1x256x2048x1x!qElemType> -> tensor<1x256x2048x1xf16>
    // CHECK: [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT1]])
    // CHECK-SAME: {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>
    // CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[QUANTIZECAST0]])
    // CHECK-SAME: {dstElemType = f16} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048xf16>
    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[DEQUANTIZE0]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x256x2048x1xf16> -> tensor<256x2048x1x1xf16>
    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[DEQUANTIZE1]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[RESHAPE0]], [[RESHAPE1]])
    // CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

    // CHECK: [[QUANTIZECAST1:%.+]] = IE.QuantizeCast([[INPUT2]])
    // CHECK-SAME: {dstElemType = !qElemType1} : tensor<1024x2048xsi8> -> tensor<1024x2048x!qElemType1>
    // CHECK: [[DEQUANTIZE2:%.+]] = IE.Dequantize([[QUANTIZECAST1]])
    // CHECK-SAME: {dstElemType = f16} : tensor<1024x2048x!qElemType1> -> tensor<1024x2048xf16>
    // CHECK: [[RESHAPE2:%.+]] = IE.AffineReshape([[DEQUANTIZE2]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2, 3]], shape_value = [1024, 2048, 1, 1]} : tensor<1024x2048xf16> -> tensor<1024x2048x1x1xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[RESHAPE0]], [[RESHAPE2]])
    // CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<1024x2048x1x1xf16> -> tensor<256x1024x1x1xf16>

    // CHECK: return [[CONV0]], [[CONV1]] : tensor<256x2048x1x1xf16>, tensor<256x1024x1x1xf16>
  }

// -----

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.060064338235294119:128>
// CHECK: !qElemType1 = !quant.uniform<i8:f32, 1.000000e+00>
// CHECK-LABEL: @SplitFakeQuantToU8ForNotAllUsersHaveSI8WeightsAsInputs
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x256x2048x1xf16>
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<2048x2048xsi8>
  func.func @SplitFakeQuantToU8ForNotAllUsersHaveSI8WeightsAsInputs(%arg0: tensor<1x256x2048x1xf16>, %arg1: tensor<2048x2048xsi8>) -> (tensor<256x2048x1x1xf16>, tensor<256x2048x1x1xf16>) {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.FakeQuantize(%arg0, %cst_3, %cst_2, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x256x2048x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x256x2048x1xf16>
    %1 = IE.QuantizeCast(%arg1) {dstElemType = !quant.uniform<i8:f32, 1.000000e+00>} : tensor<2048x2048xsi8> -> tensor<2048x2048x!quant.uniform<i8:f32, 1.000000e+00>>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<2048x2048x!quant.uniform<i8:f32, 1.000000e+00>> -> tensor<2048x2048xf16>
    %3 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x256x2048x1xf16> -> tensor<256x2048x1x1xf16>
    %4 = IE.AffineReshape(%2) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
    %5 = IE.Convolution(%3, %4) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

    %6 = IE.SoftMax(%3) {axisInd = 0} : tensor<256x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

    return %5, %6 : tensor<256x2048x1x1xf16>, tensor<256x2048x1x1xf16>

    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[INPUT0]])
    // CHECK-SAME: {dstElemType = !qElemType} : tensor<1x256x2048x1xf16> -> tensor<1x256x2048x1x!qElemType>

    // CHECK: [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZE]])
    // CHECK-SAME: {dstElemType = f16} : tensor<1x256x2048x1x!qElemType> -> tensor<1x256x2048x1xf16>

    // CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[INPUT1]])
    // CHECK-SAME: {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>

    // CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[QUANTIZECAST]])
    // CHECK-SAME: {dstElemType = f16} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048xf16>

    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[DEQUANTIZE0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x256x2048x1xf16> -> tensor<256x2048x1x1xf16>

    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[DEQUANTIZE1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[RESHAPE1]])
    // CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[RESHAPE0]])
    // CHECK-SAME: {axisInd = 0 : i64} : tensor<256x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

    // CHECK: return [[CONV]], [[SOFTMAX]] : tensor<256x2048x1x1xf16>, tensor<256x2048x1x1xf16>
  }

// -----

!qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
!qElemType1 = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @SplitFakeQuantToI8ForSI4WeightsAsInputs
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x256x2048xf32>
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<2048x2048xsi4>
func.func @SplitFakeQuantToI8ForSI4WeightsAsInputs(%arg0: tensor<1x256x2048xf32>, %arg1: tensor<2048x2048xsi4>) -> tensor<256x2048x1x1xf16> {
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

  %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
  %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
  %2 = IE.FakeQuantize(%1, %cst_3, %cst_2, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x256x2048xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x2048xf16>
  %3 = IE.Convert(%arg1) {dstElemType = f16} : tensor<2048x2048xsi4> -> tensor<2048x2048xf16>
  %4 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048xf16> -> tensor<256x2048x1x1xf16>
  %5 = IE.AffineReshape(%3) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
  %6 = IE.Convolution(%4, %5) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

  return %6 : tensor<256x2048x1x1xf16>

  // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
  // CHECK:       [[CONVERT0:%.+]] = IE.Convert([[RESHAPE0]]) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT0]]) {dstElemType = !qElemType} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
  // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x1x256x2048x!qElemType> -> tensor<1x1x256x2048xf16>
  // CHECK:       [[CONVERT1:%.+]] = IE.Convert([[INPUT1]]) {dstElemType = f16} : tensor<2048x2048xsi4> -> tensor<2048x2048xf16>
  // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[DEQUANTIZE0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048xf16> -> tensor<256x2048x1x1xf16>
  // CHECK:       [[RESHAPE2:%.+]] = IE.AffineReshape([[CONVERT1]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE1]], [[RESHAPE2]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

  // CHECK:       return [[CONV]] : tensor<256x2048x1x1xf16>
}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
// CHECK: !qElemType1 = !quant.uniform<i8:f32, 1.000000e+00>
// CHECK-LABEL: @SplitFakeQuantToI8ForSlicedIntWeightsAsInputs
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x256x2048xf32>
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<4096x2048xsi8>
func.func @SplitFakeQuantToI8ForSlicedIntWeightsAsInputs(%arg0: tensor<1x256x2048xf32>, %arg1: tensor<4096x2048xsi8>) -> tensor<256x2048x1x1xf16> {
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32> isSplat, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

  %reshape0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
  %convert = IE.Convert(%reshape0) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
  %quantize = IE.FakeQuantize(%convert, %cst_3, %cst_2, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x256x2048xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x2048xf16>
  %slice = IE.Slice %arg1 [0, 0] [2048, 2048] : tensor<4096x2048xsi8> to tensor<2048x2048xsi8>
  %quantizecast = IE.QuantizeCast(%slice) {dstElemType = !quant.uniform<i8:f32, 1.000000e+00>} : tensor<2048x2048xsi8> -> tensor<2048x2048x!quant.uniform<i8:f32, 1.000000e+00>>
  %dequantize1 = IE.Dequantize(%quantizecast) {dstElemType = f16} : tensor<2048x2048x!quant.uniform<i8:f32, 1.000000e+00>> -> tensor<2048x2048xf16>
  %reshape1 = IE.AffineReshape(%quantize) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048xf16> -> tensor<256x2048x1x1xf16>
  %reshape2 = IE.AffineReshape(%dequantize1) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
  %conv = IE.Convolution(%reshape1, %reshape2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

  return %conv : tensor<256x2048x1x1xf16>

  // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
  // CHECK: [[CONVERT:%.+]] = IE.Convert([[RESHAPE0]]) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
  // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
  // CHECK: [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x1x256x2048x!qElemType> -> tensor<1x1x256x2048xf16>
  // CHECK: [[SLICE:%.+]] = IE.Slice [[INPUT1]] [0, 0] [2048, 2048]
  // CHECK-SAME: tensor<4096x2048xsi8> to tensor<2048x2048xsi8>
  // CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[SLICE]]) {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>
  // CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048xf16>
  // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[DEQUANTIZE0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048xf16> -> tensor<256x2048x1x1xf16>
  // CHECK: [[RESHAPE2:%.+]] = IE.AffineReshape([[DEQUANTIZE1]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[RESHAPE1]], [[RESHAPE2]])
  // CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

  // CHECK: return [[CONV]] : tensor<256x2048x1x1xf16>
}
