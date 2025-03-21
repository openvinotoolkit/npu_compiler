//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --align-scales="se-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @AlignConcatScalesInterpolate
func.func @AlignConcatScalesInterpolate(%arg0: tensor<1x16x4x4xf16>, %arg1: tensor<1x8x8x8xf16>) -> tensor<1x16x5x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<3.068850e-01> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  %0 = IE.FakeQuantize(%arg0, %cst_1, %cst, %cst_1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x4x4xf16>
  %1 = IE.FakeQuantize(%arg1, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x8x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x8x8x8xf16>
  %2 = IE.Interpolate(%0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [8, 8]
         } : tensor<1x16x4x4xf16> -> tensor<1x16x8x8xf16>
  %3 = IE.Reshape(%1) { shape_value = [1, 16, 1, 8] } : tensor<1x8x8x8xf16> -> tensor<1x16x1x8xf16>
  %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  %5 = IE.Convolution(%4, %cst_2) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  return %5 : tensor<1x16x5x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.755859375> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_3:%.*]] = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  // CHECK: [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x4x4xf16>
  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize(%arg1, [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x8x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x8x8x8xf16>
  // CHECK: [[CLAMP:%.*]] = IE.Clamp([[FQ_0]]) {max = 0.306884765625 : f64, min = 0.000000e+00 : f64} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16>
  // CHECK: [[INTERPOLATE:%.*]] = IE.Interpolate([[FQ]]) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [8, 8]} : tensor<1x16x4x4xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[RESHAPE:%.*]] = IE.Reshape([[CLAMP]]) {shape_value = [1, 16, 1, 8]} : tensor<1x8x8x8xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[INTERPOLATE]], [[RESHAPE]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  // CHECK: [[CONV:%.*]] = IE.Convolution([[CONCAT]], [[CST_3]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  // CHECK: return [[CONV]] : tensor<1x16x5x8xf16>

}

// -----

// CHECK-LABEL: @DoNotAlignConcatScalesInterpolateBicubic
func.func @DoNotAlignConcatScalesInterpolateBicubic(%arg0: tensor<1x16x4x4xf16>, %arg1: tensor<1x8x8x8xf16>) -> tensor<1x16x5x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<3.068850e-01> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  %0 = IE.FakeQuantize(%arg0, %cst_1, %cst, %cst_1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x4x4xf16>
  %1 = IE.FakeQuantize(%arg1, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x8x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x8x8x8xf16>
  %2 = IE.Interpolate(%0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <CUBIC>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [8, 8]
         } : tensor<1x16x4x4xf16> -> tensor<1x16x8x8xf16>
  %3 = IE.Reshape(%1) { shape_value = [1, 16, 1, 8] } : tensor<1x8x8x8xf16> -> tensor<1x16x1x8xf16>
  %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  %5 = IE.Convolution(%4, %cst_2) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  return %5 : tensor<1x16x5x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<3.068850e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  // CHECK: [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST_1]], [[CST]], [[CST_1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x4x4xf16>
  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize(%arg1, [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x8x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x8x8x8xf16>
  // CHECK: [[INTERPOLATE:%.*]] = IE.Interpolate([[FQ]]) {attr = #IE.Interpolate<mode = <CUBIC>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [8, 8]} : tensor<1x16x4x4xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[RESHAPE:%.*]] = IE.Reshape([[FQ_0]]) {shape_value = [1, 16, 1, 8]} : tensor<1x8x8x8xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[INTERPOLATE]], [[RESHAPE]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  // CHECK: [[CONV:%.*]] = IE.Convolution([[CONCAT]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  // CHECK: return [[CONV]] : tensor<1x16x5x8xf16>

}
