//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --process-asymmetric-zero-points-for-matmul="matmul-mixed-precision-decomposition-ratio=250" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Test for large filter scenario (>100MB) - typical for LLM vocabulary embedding layers
// For LLM models (e.g., 0.5B parameters), the vocabulary dictionary in FP16 precision
// typically exceeds 100MB, making decomposition always beneficial regardless of overhead

// CHECK-LABEL: @FixZeroPointForMatmulWithLargeFilter
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x2048xf32>
func.func @FixZeroPointForMatmulWithLargeFilter(%arg0: tensor<1x8x2048xf32>) -> tensor<1x8x32000xf32> {
  %cst = const.Declare tensor<1x32000x1x2048xf16> = dense<1.0> : tensor<32000x2048xf16>, [#const.Reshape<[1, 32000, 1, 2048]>]
  %cst_0 = const.Declare tensor<1x32000x1x1xf16> = dense<-5.0> : tensor<32000x1xf16>, [#const.Reshape<[1, 32000, 1, 1]>]
  %cst_1 = const.Declare tensor<1x32000x1x1xf16> = dense<20.5> : tensor<32000x1xf16>, [#const.Reshape<[1, 32000, 1, 1]>]
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
  %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 8, 2048]} : tensor<1x8x2048xf32> -> tensor<1x1x8x2048xf32>
  %1 = IE.FakeQuantize(%cst, %cst_3, %cst_2, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32000x1x2048xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x32000x1x1xf16>, tensor<1x32000x1x1xf16> -> tensor<1x32000x1x2048xf16>
  %2 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x8x2048xf32> -> tensor<1x1x8x2048xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [8, 2048, 1, 1]} : tensor<1x1x8x2048xf16> -> tensor<8x2048x1x1xf16>
  %4 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [32000, 2048, 1, 1]} : tensor<1x32000x1x2048xf16> -> tensor<32000x2048x1x1xf16>
  %5 = IE.Convolution(%3, %4) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<8x2048x1x1xf16>, tensor<32000x2048x1x1xf16> -> tensor<8x32000x1x1xf16>
  %6 = IE.AffineReshape(%5) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 8, 32000]} : tensor<8x32000x1x1xf16> -> tensor<1x1x8x32000xf16>
  %7 = IE.Convert(%6) {dstElemType = f32} : tensor<1x1x8x32000xf16> -> tensor<1x1x8x32000xf32>
  %8 = IE.AffineReshape(%7) {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 8, 32000]} : tensor<1x1x8x32000xf32> -> tensor<1x8x32000xf32>
  return %8 : tensor<1x8x32000xf32>

  // CHECK-DAG:   [[CST_DIFF:%.+]] = const.Declare tensor<1x32000x1x1xf16> =
  // CHECK-SAME:      dense<7.800780e+00> : tensor<1x32000x1x1xf16>

  // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x32000x1x2048xf16> = dense<1.000000e+00> : tensor<32000x2048xf16>, [#const.Reshape<[1, 32000, 1, 2048]>]
  // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x32000x1x1xf16> = dense<-1.279690e+01> : tensor<1x32000x1x1xf16>
  // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x32000x1x1xf16> = dense<1.270310e+01> : tensor<1x32000x1x1xf16>
  // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
  // CHECK-DAG:   [[CST4:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]

  // CHECK:   [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG0]])
  // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 8, 2048]} : tensor<1x8x2048xf32> -> tensor<1x1x8x2048xf32>

  // CHECK:   [[FQ:%.+]] = IE.FakeQuantize([[CST0]], [[CST4]], [[CST3]], [[CST1]], [[CST2]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
  // CHECK-SAME:      tensor<1x32000x1x2048xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x32000x1x1xf16>, tensor<1x32000x1x1xf16>
  // CHECK-SAME:       -> tensor<1x32000x1x2048xf16>

  // CHECK:   [[CONVERT1:%.+]] = IE.Convert([[RESHAPE0]]) {dstElemType = f16} : tensor<1x1x8x2048xf32> -> tensor<1x1x8x2048xf16>
  // CHECK:   [[RESHAPE1:%.+]] = IE.AffineReshape([[CONVERT1]])
  // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [8, 2048, 1, 1]} : tensor<1x1x8x2048xf16> -> tensor<8x2048x1x1xf16>
  // CHECK:   [[RESHAPE2:%.+]] = IE.AffineReshape([[FQ]])
  // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [32000, 2048, 1, 1]} : tensor<1x32000x1x2048xf16> -> tensor<32000x2048x1x1xf16>

  // CHECK:   [[TRANSPOSE1:%.+]] = IE.Transpose([[RESHAPE1]]) {order_value = #map} : tensor<8x2048x1x1xf16> -> tensor<1x2048x8x1xf16>
  // CHECK:   [[RESHAPE3:%.+]] = IE.AffineReshape([[TRANSPOSE1]])
  // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 2048, 2, 4]} : tensor<1x2048x8x1xf16> -> tensor<1x2048x2x4xf16>

  // CHECK:   [[REDUCE_SUM:%.+]] = IE.ReduceSum([[RESHAPE3]]) {axes_value = [1], keep_dims} : tensor<1x2048x2x4xf16> -> tensor<1x1x2x4xf16>
  // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[REDUCE_SUM]], [[CST_DIFF]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x4xf16>, tensor<1x32000x1x1xf16> -> tensor<1x32000x2x4xf16>
  // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE3]], [[RESHAPE2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
  // CHECK-SAME{LITERAL}:  strides = [1, 1]} : tensor<1x2048x2x4xf16>, tensor<32000x2048x1x1xf16> -> tensor<1x32000x2x4xf16>
  // CHECK:   [[FIXED_CONV:%.+]] = IE.Add([[CONV]], [[MULTIPLY]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32000x2x4xf16>, tensor<1x32000x2x4xf16> -> tensor<1x32000x2x4xf16>

  // CHECK:   [[RESHAPE4:%.+]] = IE.AffineReshape([[FIXED_CONV]])
  // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 32000, 8, 1]} : tensor<1x32000x2x4xf16> -> tensor<1x32000x8x1xf16>
  // CHECK:   [[TRANSPOSE2:%.+]] = IE.Transpose([[RESHAPE4]]) {order_value = #map} : tensor<1x32000x8x1xf16> -> tensor<8x32000x1x1xf16>
  // CHECK:   [[RESHAPE5:%.+]] = IE.AffineReshape([[TRANSPOSE2]])
  // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 8, 32000]} : tensor<8x32000x1x1xf16> -> tensor<1x1x8x32000xf16>
  // CHECK:   [[CONVERT2:%.+]] = IE.Convert([[RESHAPE5]]) {dstElemType = f32} : tensor<1x1x8x32000xf16> -> tensor<1x1x8x32000xf32>
  // CHECK:   [[RESHAPE6:%.+]] = IE.AffineReshape([[CONVERT2]])
  // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 8, 32000]} : tensor<1x1x8x32000xf32> -> tensor<1x8x32000xf32>

  // CHECK:   return [[RESHAPE6]] : tensor<1x8x32000xf32>
}
