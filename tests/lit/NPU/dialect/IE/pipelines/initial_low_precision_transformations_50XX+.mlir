//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --initial-low-precision-transformations %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU50XX

// CHECK-LABEL: @ScalarArgumentsWeightsDequantizeToFakeQuantize
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x32x32xf32>
func.func @ScalarArgumentsWeightsDequantizeToFakeQuantize(%input: tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32> {
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158> : tensor<1x1x1x1xf32>
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %cst_weights = const.Declare tensor<16x1x1x1xf32> = dense<[[[[448.0]]], [[[352.0]]], [[[160.0]]], [[[1.375]]], [[[80.0]]], [[[32.0]]], [[[144.0]]], [[[2.25]]], [[[0.625]]], [[[104.0]]], [[[56.0]]], [[[176.0]]], [[[1.75]]], [[[0.3125]]], [[[1.125]]], [[[1.5]]]]> : tensor<16x1x1x1xf8E4M3FN>, [#const.CastElemType<f32>]
  %cst_3 = const.Declare tensor<f32> = dense<2.500000e+01> : tensor<f32>
  %cst_4 = const.Declare tensor<f32> = dense<0.0566197559> : tensor<f32>
  %0 = IE.FakeQuantize(%input, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x16x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x32x32xf32>
  %1 = IE.Subtract(%cst_weights, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<f32> -> tensor<16x1x1x1xf32>
  %2 = IE.Multiply(%1, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<f32> -> tensor<16x1x1x1xf32>
  %3 = IE.GroupConvolution(%0, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x32xf32>
  return %3 : tensor<1x16x32x32xf32>

  // CHECK-DAG:   [[CST_INPUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_INPUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_WEIGHTS_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_WEIGHTS_IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_WEIGHTS_OUT_LOW:%.+]] = const.Declare tensor<1xf32> = dense<-26.7811451> : tensor<1xf32>
  // CHECK-DAG:   [[CST_WEIGHTS_OUT_HIGH:%.+]] = const.Declare tensor<1xf32> = dense<23.9501572> : tensor<1xf32>
  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK-SAME{LITERAL}: = dense<[[[[4.480000e+02]]], [[[3.520000e+02]]], [[[1.600000e+02]]], [[[1.375000e+00]]], [[[8.000000e+01]]], [[[3.200000e+01]]], [[[1.440000e+02]]], [[[2.250000e+00]]], [[[6.250000e-01]]], [[[1.040000e+02]]], [[[5.600000e+01]]], [[[1.760000e+02]]], [[[1.750000e+00]]], [[[3.125000e-01]]], [[[1.125000e+00]]], [[[1.500000e+00]]]]> : tensor<16x1x1x1xf8E4M3FN>, [#const.CastElemType<f32>]

  // CHECK:       [[WT_FQ:%.+]] = IE.FakeQuantize([[CST_WEIGHTS]], [[CST_WEIGHTS_IN_LOW]], [[CST_WEIGHTS_IN_HIGH]], [[CST_WEIGHTS_OUT_LOW]], [[CST_WEIGHTS_OUT_HIGH]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<16x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<16x1x1x1xf32>
  // CHECK:       [[ACT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST_INPUT_LOW]], [[CST_INPUT_HIGH]], [[CST_INPUT_LOW]], [[CST_INPUT_HIGH]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x16x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x32x32xf32>
  // CHECK:   [[GRUP_CONV:%.+]] = IE.GroupConvolution([[ACT_FQ]], [[WT_FQ]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x32xf32>

  // CHECK:   return [[GRUP_CONV]] : tensor<1x16x32x32xf32>
}

// -----

// CHECK-LABEL: @ScalarArgumentsFakeConvert
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x80x3000xf16>
func.func @ScalarArgumentsFakeConvert(%input: tensor<1x80x3000xf16>) -> tensor<1x80x3000xf16> {
    %scale = const.Declare tensor<f16> = dense<0.5> : tensor<f16>
    %shift = const.Declare tensor<f16> = dense<2.0> : tensor<f16>
    %0 = IE.FakeConvert(%input, %scale, %shift) {dst_type = f8E4M3FN} : tensor<1x80x3000xf16>, tensor<f16>, tensor<f16> -> tensor<1x80x3000xf16>

    return %0 : tensor<1x80x3000xf16>

    // CHECK-DAG:   [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-8.940000e+02> : tensor<1xf16>
    // CHECK-DAG:   [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<8.980000e+02> : tensor<1xf16>

    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:                   low_fp_type = f8E4M3FN
    // CHECK-SAME:               } : tensor<1x80x3000xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x80x3000xf16>

    // CHECK:   return [[FQ]] : tensor<1x80x3000xf16>
}
