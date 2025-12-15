//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --process-asymmetric-zero-points-for-convolution --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @FixZeroPointForConvolution

!qElemType = !quant.uniform<i8:f16, 1.600000e+01>
func.func @FixZeroPointForConvolution(%arg0: tensor<1x4x4x4xf32>) -> tensor<1x4x2x2xf32> {
  %cst = const.Declare tensor<4x4x3x3xf16> = dense<1.280000e+02> : tensor<4x4x3x3xf32>, [#const.CastElemType<f16>]
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<-2.400000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<2.700000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %0 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %1 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x4x4x4xf32> -> tensor<1x4x4x4xf16>
  %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x4x4xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x2x2xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x4x2x2xf16> -> tensor<1x4x2x2xf32>
  return %3 : tensor<1x4x2x2xf32>

  // CHECK:   [[CST:%.*]] = const.Declare tensor<4x4x3x3x!qElemType> = dense<1.000000e+00> : tensor<4x4x3x3xf16>, [#const.CastElemType<!qElemType>]
  // CHECK:   [[CST0:%.*]] = const.Declare tensor<4x4x3x3xf16> = dense<1.280000e+02> : tensor<4x4x3x3xf32>, [#const.CastElemType<f16>]
  // CHECK:   [[CST1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK:   [[CST2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK:   [[CST3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.400000e+02> :
  // CHECK-SAME:      tensor<1x1x1x1xf32>, [#const.CastElemType<f16>, #const.Add<-1.600000e+01 : f64>]
  // CHECK:   [[CST4:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.700000e+02> :
  // CHECK-SAME:      tensor<1x1x1x1xf32>, [#const.CastElemType<f16>, #const.Add<-1.600000e+01 : f64>]

  // CHECK:   [[FQ:%.*]] = IE.FakeQuantize([[CST0]], [[CST1]], [[CST2]], [[CST3]], [[CST4]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
  // CHECK-SAME:      tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16>,
  // CHECK-SAME:      tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>,
  // CHECK-SAME:      tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>

  // CHECK:   [[CONVERT:%.*]] = IE.Convert([[ARG0:%.*]]) {dstElemType = f16} : tensor<1x4x4x4xf32> -> tensor<1x4x4x4xf16>
  // CHECK:   [[CONV1:%.*]] = IE.Convolution([[CONVERT]], [[FQ]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
  // CHECK-SAME:      tensor<1x4x4x4xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x2x2xf16>
  // CHECK:   [[CONV2:%.*]] = IE.Convolution([[CONVERT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
  // CHECK-SAME:      tensor<1x4x4x4xf16>, tensor<4x4x3x3x!qElemType> -> tensor<1x4x2x2xf16>
  // CHECK:   [[ADD:%.*]] = IE.Add([[CONV1]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
  // CHECK-SAME:      tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16> -> tensor<1x4x2x2xf16>
  // CHECK:   [[RESULT:%.*]] = IE.Convert([[ADD]]) {dstElemType = f32} : tensor<1x4x2x2xf16> -> tensor<1x4x2x2xf32>

  // CHECK:   return [[RESULT]] : tensor<1x4x2x2xf32>
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.600000e+01>
func.func @FixZeroPointForConvolutionNoFQPattern(%arg0: tensor<1x4x4x4xf32>) -> tensor<1x4x2x2xf32> {
  %cst = const.Declare tensor<4x4x3x3xf16> = dense<1.280000e+02> : tensor<4x4x3x3xf32>, [#const.CastElemType<f16>]

  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x4x4x4xf32> -> tensor<1x4x4x4xf16>
  %1 = IE.Convolution(%0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x4x4xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x2x2xf16>
  %2 = IE.Convert(%1) {dstElemType = f32} : tensor<1x4x2x2xf16> -> tensor<1x4x2x2xf32>
  return %2 : tensor<1x4x2x2xf32>

  // CHECK:   [[CST:%.*]] = const.Declare tensor<4x4x3x3xf16> = dense<1.280000e+02> : tensor<4x4x3x3xf32>, [#const.CastElemType<f16>]
  // CHECK:   [[CONVERT:%.*]] = IE.Convert([[ARG0:%.*]]) {dstElemType = f16} : tensor<1x4x4x4xf32> -> tensor<1x4x4x4xf16>
  // CHECK:   [[CONV1:%.*]] = IE.Convolution([[CONVERT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
  // CHECK-SAME:      tensor<1x4x4x4xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x2x2xf16>
  // CHECK:   [[RESULT:%.*]] = IE.Convert([[CONV1]]) {dstElemType = f32} : tensor<1x4x2x2xf16> -> tensor<1x4x2x2xf32>

  // CHECK:   return [[RESULT]] : tensor<1x4x2x2xf32>
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.600000e+01>
func.func @CorrectZeroPointForConvolution(%arg0: tensor<1x4x4x4xf32>) -> tensor<1x4x2x2xf32> {
  %cst = const.Declare tensor<4x4x3x3xf16> = dense<1.280000e+02> : tensor<4x4x3x3xf32>, [#const.CastElemType<f16>]
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<-2.560000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<2.540000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %0 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %1 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x4x4x4xf32> -> tensor<1x4x4x4xf16>
  %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x4x4xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x2x2xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x4x2x2xf16> -> tensor<1x4x2x2xf32>
  return %3 : tensor<1x4x2x2xf32>


  // CHECK:   [[CST:%.*]] = const.Declare tensor<4x4x3x3xf16> = dense<1.280000e+02> : tensor<4x4x3x3xf32>, [#const.CastElemType<f16>]
  // CHECK:   [[CST0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK:   [[CST1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK:   [[CST2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.560000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

  // CHECK:   [[CST3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.540000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK:   [[FQ:%.*]] = IE.FakeQuantize([[CST]], [[CST0]], [[CST1]], [[CST2]], [[CST3]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
  // CHECK-SAME:      tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16>,
  // CHECK-SAME:      tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>,
  // CHECK-SAME:      tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
  // CHECK:   [[CONVERT:%.*]] = IE.Convert([[ARG0:%.*]]) {dstElemType = f16} : tensor<1x4x4x4xf32> -> tensor<1x4x4x4xf16>
  // CHECK:   [[CONV1:%.*]] = IE.Convolution([[CONVERT]], [[FQ]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
  // CHECK-SAME:      tensor<1x4x4x4xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x2x2xf16>
  // CHECK:   [[RESULT:%.*]] = IE.Convert([[CONV1]]) {dstElemType = f32} : tensor<1x4x2x2xf16> -> tensor<1x4x2x2xf32>

  // CHECK:   return [[RESULT]] : tensor<1x4x2x2xf32>
}
