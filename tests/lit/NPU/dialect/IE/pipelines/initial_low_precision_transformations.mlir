//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --initial-low-precision-transformations %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ScalarArgumentsWeightsDequantizeToFakeQuantize
func.func @ScalarArgumentsWeightsDequantizeToFakeQuantize(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32> {
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158> : tensor<1x1x1x1xf32>
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %cst_2 = const.Declare tensor<16x1x1x1xf32> = dense<[[[[27]]], [[[25]]], [[[39]]], [[[22]]], [[[27]]], [[[25]]], [[[21]]], [[[27]]], [[[31]]], [[[29]]], [[[42]]], [[[27]]], [[[27]]], [[[28]]], [[[33]]], [[[33]]]]> : tensor<16x1x1x1xsi8>, [#const.CastElemType<f32>]
  %cst_3 = const.Declare tensor<f32> = dense<2.500000e+01> : tensor<f32>
  %cst_4 = const.Declare tensor<f32> = dense<0.0566197559> : tensor<f32>
  %0 = IE.FakeQuantize(%arg0, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x32x32xf32>
  %1 = IE.Subtract(%cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<f32> -> tensor<16x1x1x1xf32>
  %2 = IE.Multiply(%1, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<f32> -> tensor<16x1x1x1xf32>
  %3 = IE.GroupConvolution(%0, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x32xf32>
  return %3 : tensor<1x16x32x32xf32>

  // CHECK: [[ACT_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158>
  // CHECK: [[ACT_LOW:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00>

  // CHECK: [[WT_IN_LOW:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02>
  // CHECK: [[WT_IN_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02>
  // CHECK: [[WT_OUT_LOW:%.*]] = const.Declare tensor<1xf32> = dense<-8.66282272>
  // CHECK: [[WT_OUT_HIGH:%.*]] = const.Declare tensor<1xf32> = dense<5.77521514>

  // CHECK: [[WT_DATA:%.*]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[27]]], [[[25]]], [[[39]]], [[[22]]], [[[27]]], [[[25]]], [[[21]]], [[[27]]], [[[31]]], [[[29]]], [[[42]]], [[[27]]], [[[27]]], [[[28]]], [[[33]]], [[[33]]]]>

  // CHECK-DAG: [[WT_FQ:%.*]] = IE.FakeQuantize([[WT_DATA]], [[WT_IN_LOW]], [[WT_IN_HIGH]], [[WT_OUT_LOW]], [[WT_OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<16x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<16x1x1x1xf32>
  // CHECK-DAG: [[ACT_FQ:%.*]] = IE.FakeQuantize(%arg0, [[ACT_LOW]], [[ACT_HIGH]], [[ACT_LOW]], [[ACT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x32x32xf32>
  // CHECK:   [[GRUP_CONV:%.*]] = IE.GroupConvolution([[ACT_FQ]], [[WT_FQ]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x32xf32>

  // CHECK:   return [[GRUP_CONV]] : tensor<1x16x32x32xf32>
}
