//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --process-asymmetric-zero-points-for-matmul="matmul-mixed-precision-decomposition-ratio=0.5" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!qElemType = !quant.uniform<i8:f16, 2.000000e+00>
// CHECK-LABEL: @FixZeroPointForMatmul
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<2x16x32xf16>
func.func @FixZeroPointForMatmul(%arg0: tensor<2x16x32xf16>) -> tensor<2x16x64xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-2.520000e+02> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.580000e+02> : tensor<1x1x1x1xf16>
  %cst_3 = const.Declare tensor<1x1x32x64xf16> = dense<1.0> : tensor<32x64xf16>, [#const.Reshape<[1, 1, 32, 64]>]
  %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x32x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x32x64xf16>
  %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x32x64xf16> -> tensor<1x1x64x32xf16>
  %2 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [32, 32, 1, 1]} : tensor<2x16x32xf16> -> tensor<32x32x1x1xf16>
  %3 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [64, 32, 1, 1]} : tensor<1x1x64x32xf16> -> tensor<64x32x1x1xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x32x1x1xf16>, tensor<64x32x1x1xf16> -> tensor<32x64x1x1xf16>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [1], [1]], shape_value = [32, 64]} : tensor<32x64x1x1xf16> -> tensor<32x64xf16>
  %6 = IE.AffineReshape(%5) {dim_mapping = [[0, 1], [2]], shape_value = [2, 16, 64]} : tensor<32x64xf16> -> tensor<2x16x64xf16>
  return %6 : tensor<2x16x64xf16>

  // CHECK:   [[CST_DIFF:%.+]] = const.Declare tensor<1x64x1x1xf16> = dense<4.000000e+00> : tensor<1x64x1x1xf16>

  // CHECK:   [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
  // CHECK:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.560000e+02> : tensor<1x1x1x1xf16>
  // CHECK:   [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.540000e+02> : tensor<1x1x1x1xf16>
  // CHECK:   [[CST4:%.+]] = const.Declare tensor<1x1x32x64xf16> = dense<1.000000e+00> : tensor<32x64xf16>, [#const.Reshape<[1, 1, 32, 64]>]

  // CHECK:   [[FQ:%.+]] = IE.FakeQuantize([[CST4]], [[CST0]], [[CST1]], [[CST2]], [[CST3]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
  // CHECK-SAME:      tensor<1x1x32x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
  // CHECK-SAME:       -> tensor<1x1x32x64xf16>

  // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[FQ]]) {order_value = #NCWH} : tensor<1x1x32x64xf16> -> tensor<1x1x64x32xf16>
  // CHECK:   [[RESHAPE1:%.+]] = IE.AffineReshape([[ARG0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [32, 32, 1, 1]} : tensor<2x16x32xf16> -> tensor<32x32x1x1xf16>
  // CHECK:   [[RESHAPE2:%.+]] = IE.AffineReshape([[TRANSPOSE]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [64, 32, 1, 1]} : tensor<1x1x64x32xf16> -> tensor<64x32x1x1xf16>
  // CHECK:   [[TRANSPOSE2:%.+]] = IE.Transpose([[RESHAPE1]]) {order_value = #map} : tensor<32x32x1x1xf16> -> tensor<1x32x32x1xf16>
  // CHECK:   [[RESHAPE3:%.+]] = IE.AffineReshape([[TRANSPOSE2]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 32, 8, 4]} : tensor<1x32x32x1xf16> -> tensor<1x32x8x4xf16>

  // CHECK:   [[REDUCE_SUM:%.+]] = IE.ReduceSum([[RESHAPE3]]) {axes_value = [1], keep_dims} : tensor<1x32x8x4xf16> -> tensor<1x1x8x4xf16>
  // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[REDUCE_SUM]], [[CST_DIFF]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x8x4xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x8x4xf16>
  // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE3]], [[RESHAPE2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
  // CHECK-SAME{LITERAL}:  strides = [1, 1]} : tensor<1x32x8x4xf16>, tensor<64x32x1x1xf16> -> tensor<1x64x8x4xf16>
  // CHECK:   [[FIXED_CONV:%.+]] = IE.Add([[CONV]], [[MULTIPLY]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x8x4xf16>, tensor<1x64x8x4xf16> -> tensor<1x64x8x4xf16>

  // CHECK: [[RESHAPE4:%.+]] = IE.AffineReshape([[FIXED_CONV]])
  // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 32, 1]} : tensor<1x64x8x4xf16> -> tensor<1x64x32x1xf16>
  // CHECK: [[TRANSPOSE3:%.+]] = IE.Transpose([[RESHAPE4]]) {order_value = #map} : tensor<1x64x32x1xf16> -> tensor<32x64x1x1xf16>
  // CHECK: [[RESHAPE5:%.+]] = IE.AffineReshape([[TRANSPOSE3]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [1]], shape_value = [32, 64]} : tensor<32x64x1x1xf16> -> tensor<32x64xf16>
  // CHECK: [[RESHAPE6:%.+]] = IE.AffineReshape([[RESHAPE5]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2]], shape_value = [2, 16, 64]} : tensor<32x64xf16> -> tensor<2x16x64xf16>

  // CHECK:   return [[RESHAPE6]] :  tensor<2x16x64xf16>
}

// -----

// CHECK-LABEL: @FixZeroPointForMatmulPerChannelQuantized
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x2048xf16>
func.func @FixZeroPointForMatmulPerChannelQuantized(%arg0: tensor<1x1x2048xf16>) -> tensor<1x1x2xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
  %cst_1 = const.Declare tensor<1x2x1x1xf16> = dense<[[-5.000000e+00], [-1.019530e+01]]> : tensor<2x1xf16>, [#const.Reshape<[1, 2, 1, 1]>]
  %cst_2 = const.Declare tensor<1x2x1x1xf16> = dense<[[2.050000e+01], [4.078130e+01]]> : tensor<2x1xf16>, [#const.Reshape<[1, 2, 1, 1]>]
  %cst_3 = const.Declare tensor<1x2x1x2048xf16> = dense<226> : tensor<2x2048xui8>, [#const.Reshape<[1, 2, 1, 2048]>, #const.CastElemType<f16>]
  %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x1x2048xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x1x2048xf16>
  %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [1, 2048, 1, 1]} : tensor<1x1x2048xf16> -> tensor<1x2048x1x1xf16>
  %2 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [2, 2048, 1, 1]} : tensor<1x2x1x2048xf16> -> tensor<2x2048x1x1xf16>
  %3 = IE.Convolution(%1, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2048x1x1xf16>, tensor<2x2048x1x1xf16> -> tensor<1x2x1x1xf16>
  %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 2]} : tensor<1x2x1x1xf16> -> tensor<1x2xf16>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 2]} : tensor<1x2xf16> -> tensor<1x1x2xf16>
  return %5 : tensor<1x1x2xf16>

  // CHECK:   [[CST_DIFF:%.+]] = const.Declare tensor<1x2x1x1xf16> =
  // CHECK-SAME{LITERAL}: dense<[[[[7.800780e+00]], [[1.539060e+01]]]]> : tensor<1x2x1x1xf16>

  // CHECK:   [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
  // CHECK:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
  // CHECK:   [[CST2:%.+]] = const.Declare tensor<1x2x1x1xf16> =
  // CHECK-SAME{LITERAL}:    dense<[[[[-1.279690e+01]], [[-2.559380e+01]]]]> : tensor<1x2x1x1xf16>
  // CHECK:   [[CST3:%.+]] = const.Declare tensor<1x2x1x1xf16> =
  // CHECK-SAME{LITERAL}:    dense<[[[[1.270310e+01]], [[2.539060e+01]]]]> : tensor<1x2x1x1xf16>
  // CHECK:   [[CST4:%.+]] = const.Declare tensor<1x2x1x2048xf16> = dense<226> : tensor<2x2048xui8>, [#const.Reshape<[1, 2, 1, 2048]>, #const.CastElemType<f16>]

  // CHECK:   [[FQ:%.+]] = IE.FakeQuantize([[CST4]], [[CST0]], [[CST1]], [[CST2]], [[CST3]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
  // CHECK-SAME:      tensor<1x2x1x2048xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16>
  // CHECK-SAME:       -> tensor<1x2x1x2048xf16>

  // CHECK:   [[RESHAPE1:%.+]] = IE.AffineReshape([[ARG0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [1, 2048, 1, 1]} : tensor<1x1x2048xf16> -> tensor<1x2048x1x1xf16>
  // CHECK:   [[RESHAPE2:%.+]] = IE.AffineReshape([[FQ]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [2, 2048, 1, 1]} : tensor<1x2x1x2048xf16> -> tensor<2x2048x1x1xf16>

  // CHECK:   [[REDUCE_SUM:%.+]] = IE.ReduceSum([[RESHAPE1]]) {axes_value = [1], keep_dims} : tensor<1x2048x1x1xf16> -> tensor<1x1x1x1xf16>
  // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[REDUCE_SUM]], [[CST_DIFF]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x1x1xf16>
  // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE1]], [[RESHAPE2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
  // CHECK-SAME{LITERAL}:  strides = [1, 1]} : tensor<1x2048x1x1xf16>, tensor<2x2048x1x1xf16> -> tensor<1x2x1x1xf16>
  // CHECK:   [[FIXED_CONV:%.+]] = IE.Add([[CONV]], [[MULTIPLY]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x1x1xf16>

  // CHECK:   [[RESHAPE3:%.+]] = IE.AffineReshape([[FIXED_CONV]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 2]} : tensor<1x2x1x1xf16> -> tensor<1x2xf16>
  // CHECK:   [[RESHAPE4:%.+]] = IE.AffineReshape([[RESHAPE3]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 2]} : tensor<1x2xf16> -> tensor<1x1x2xf16>

  // CHECK:   return [[RESHAPE4]] : tensor<1x1x2xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 2.000000e+00>
// CHECK-LABEL: @FixZeroPointForMatmulFQToWeightsPattern
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<2x16x32xf16>
func.func @FixZeroPointForMatmulFQToWeightsPattern(%arg0: tensor<2x16x32xf16>) -> tensor<2x16x64xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-2.520000e+02> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.580000e+02> : tensor<1x1x1x1xf16>
  %cst_3 = const.Declare tensor<64x32x1x1xf16> = dense<1.0> : tensor<64x32xf16>, [#const.Reshape<[64, 32, 1, 1]>]
  %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    : tensor<64x32x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
    -> tensor<64x32x1x1xf16>
  %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [32, 32, 1, 1]}
    : tensor<2x16x32xf16> -> tensor<32x32x1x1xf16>
  %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    : tensor<32x32x1x1xf16>, tensor<64x32x1x1xf16> -> tensor<32x64x1x1xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1], [1]], shape_value = [32, 64]}
    : tensor<32x64x1x1xf16> -> tensor<32x64xf16>
  %4 = IE.AffineReshape(%3) {dim_mapping = [[0, 1], [2]], shape_value = [2, 16, 64]}
    : tensor<32x64xf16> -> tensor<2x16x64xf16>
  return %4 : tensor<2x16x64xf16>

  // CHECK:   [[CST_DIFF:%.+]] = const.Declare tensor<1x64x1x1xf16> = dense<4.000000e+00> : tensor<1x64x1x1xf16>

  // CHECK:   [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
  // CHECK:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.560000e+02> : tensor<1x1x1x1xf16>
  // CHECK:   [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.540000e+02> : tensor<1x1x1x1xf16>
  // CHECK:   [[CST4:%.+]] = const.Declare tensor<64x32x1x1xf16> = dense<1.000000e+00> : tensor<64x32xf16>, [#const.Reshape<[64, 32, 1, 1]>]

  // CHECK:   [[FQ:%.+]] = IE.FakeQuantize([[CST4]], [[CST0]], [[CST1]], [[CST2]], [[CST3]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
  // CHECK-SAME:      tensor<64x32x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
  // CHECK-SAME:       -> tensor<64x32x1x1xf16>

  // CHECK:   [[RESHAPE1:%.+]] = IE.AffineReshape([[ARG0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [32, 32, 1, 1]} : tensor<2x16x32xf16> -> tensor<32x32x1x1xf16>

  // CHECK:   [[TRANSPOSE2:%.+]] = IE.Transpose([[RESHAPE1]]) {order_value = #map} : tensor<32x32x1x1xf16> -> tensor<1x32x32x1xf16>
  // CHECK:   [[RESHAPE3:%.+]] = IE.AffineReshape([[TRANSPOSE2]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 32, 8, 4]} : tensor<1x32x32x1xf16> -> tensor<1x32x8x4xf16>

  // CHECK:   [[REDUCE_SUM:%.+]] = IE.ReduceSum([[RESHAPE3]]) {axes_value = [1], keep_dims} : tensor<1x32x8x4xf16> -> tensor<1x1x8x4xf16>
  // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[REDUCE_SUM]], [[CST_DIFF]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x8x4xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x8x4xf16>

  // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE3]], [[FQ]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
  // CHECK-SAME{LITERAL}:  strides = [1, 1]} : tensor<1x32x8x4xf16>, tensor<64x32x1x1xf16> -> tensor<1x64x8x4xf16>
  // CHECK:   [[FIXED_CONV:%.+]] = IE.Add([[CONV]], [[MULTIPLY]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x8x4xf16>, tensor<1x64x8x4xf16> -> tensor<1x64x8x4xf16>

  // CHECK: [[RESHAPE4:%.+]] = IE.AffineReshape([[FIXED_CONV]])
  // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 32, 1]} : tensor<1x64x8x4xf16> -> tensor<1x64x32x1xf16>
  // CHECK: [[TRANSPOSE3:%.+]] = IE.Transpose([[RESHAPE4]]) {order_value = #map} : tensor<1x64x32x1xf16> -> tensor<32x64x1x1xf16>
  // CHECK: [[RESHAPE5:%.+]] = IE.AffineReshape([[TRANSPOSE3]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [1]], shape_value = [32, 64]} : tensor<32x64x1x1xf16> -> tensor<32x64xf16>
  // CHECK: [[RESHAPE6:%.+]] = IE.AffineReshape([[RESHAPE5]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2]], shape_value = [2, 16, 64]} : tensor<32x64xf16> -> tensor<2x16x64xf16>

  // CHECK:   return [[RESHAPE6]] :  tensor<2x16x64xf16>
}

// -----


!qElemType = !quant.uniform<i8:f16, 2.000000e+00>
// CHECK-LABEL: @DontFixZeroPointForNon1x1Kernel
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<2x16x32xf16>
func.func @DontFixZeroPointForNon1x1Kernel(%arg0: tensor<2x16x32xf16>) -> tensor<2x16x64xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-2.520000e+02> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.580000e+02> : tensor<1x1x1x1xf16>
  %cst_3 = const.Declare tensor<3x3x32x64xf16> = dense<1.0> : tensor<3x3x32x64xf16>
  %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<3x3x32x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<3x3x32x64xf16>
  %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<3x3x32x64xf16> -> tensor<3x3x64x32xf16>
  %2 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [32, 32, 1, 1]} : tensor<2x16x32xf16> -> tensor<32x32x1x1xf16>
  %3 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [64, 32, 3, 3]} : tensor<3x3x64x32xf16> -> tensor<64x32x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<32x32x1x1xf16>, tensor<64x32x3x3xf16> -> tensor<32x64x1x1xf16>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [1], [1]], shape_value = [32, 64]} : tensor<32x64x1x1xf16> -> tensor<32x64xf16>
  %6 = IE.AffineReshape(%5) {dim_mapping = [[0, 1], [2]], shape_value = [2, 16, 64]} : tensor<32x64xf16> -> tensor<2x16x64xf16>
  return %6 : tensor<2x16x64xf16>

  // CHECK:   [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
  // CHECK:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.520000e+02> : tensor<1x1x1x1xf16>
  // CHECK:   [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.580000e+02> : tensor<1x1x1x1xf16>
  // CHECK:   [[CST4:%.+]] = const.Declare tensor<3x3x32x64xf16> = dense<1.000000e+00> : tensor<3x3x32x64xf16>

  // CHECK:   [[FQ:%.+]] = IE.FakeQuantize([[CST4]], [[CST0]], [[CST1]], [[CST2]], [[CST3]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
  // CHECK-SAME:      tensor<3x3x32x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
  // CHECK-SAME:       -> tensor<3x3x32x64xf16>

  // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[FQ]]) {order_value = #NCWH} : tensor<3x3x32x64xf16> -> tensor<3x3x64x32xf16>
  // CHECK:   [[RESHAPE1:%.+]] = IE.AffineReshape([[ARG0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [32, 32, 1, 1]} : tensor<2x16x32xf16> -> tensor<32x32x1x1xf16>
  // CHECK:   [[RESHAPE2:%.+]] = IE.AffineReshape([[TRANSPOSE]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [64, 32, 3, 3]} : tensor<3x3x64x32xf16> -> tensor<64x32x3x3xf16>

  // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE1]], [[RESHAPE2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1],
  // CHECK-SAME{LITERAL}:  strides = [1, 1]} : tensor<32x32x1x1xf16>, tensor<64x32x3x3xf16> -> tensor<32x64x1x1xf16>

  // CHECK:   [[RESHAPE3:%.+]] = IE.AffineReshape([[CONV]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [1]], shape_value = [32, 64]} : tensor<32x64x1x1xf16> -> tensor<32x64xf16>
  // CHECK:   [[RESHAPE4:%.+]] = IE.AffineReshape([[RESHAPE3]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2]], shape_value = [2, 16, 64]} : tensor<32x64xf16> -> tensor<2x16x64xf16>

  // CHECK:   return [[RESHAPE4]] : tensor<2x16x64xf16>
}
