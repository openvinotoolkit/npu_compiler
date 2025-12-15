//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --handle-asymmetric-strides --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @HandleConvolutionWithAsymmetricStridesWithFQuantF8E4M3FN
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x16x64x1024xf16>
func.func @HandleConvolutionWithAsymmetricStridesWithFQuantF8E4M3FN(%input: tensor<1x16x64x1024xf16>) -> tensor<1x32x64x512xf16> {
  %weights = const.Declare tensor<32x16x1x3xf16> = dense<1.0> : tensor<32x16x1x3xf16>
  %low = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
  %high = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>
  %out_low = const.Declare tensor<32x1x1x1xf16> = dense<-1.0> : tensor<32x1x1x1xf16>
  %out_high = const.Declare tensor<32x1x1x1xf16> = dense<1.0> : tensor<32x1x1x1xf16>

  %0 = IE.FakeQuantize(%input, %low, %high, %low, %high) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
  } : tensor<1x16x64x1024xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x64x1024xf16>

  %1 = IE.FakeQuantize(%weights, %low, %high, %out_low, %out_high) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
  } : tensor<32x16x1x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16> -> tensor<32x16x1x3xf16>

  %2 = IE.Convolution(%0, %1) {
    dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 2]
  } : tensor<1x16x64x1024xf16>, tensor<32x16x1x3xf16> -> tensor<1x32x64x512xf16>

  %3 = IE.FakeQuantize(%2, %low, %high, %low, %high) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
  } : tensor<1x32x64x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x64x512xf16>

  return %3 : tensor<1x32x64x512xf16>

  // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x16x1x3xf16> = dense<1.000000e+00> : tensor<32x16x1x3xf16>
  // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<32x1x1x1xf16> = dense<-1.000000e+00> : tensor<32x1x1x1xf16>
  // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<32x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x1x1xf16>

  // CHECK:    [[INPUT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x16x64x1024xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x64x1024xf16>

  // CHECK:    [[WEIGHTS_FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[LOW]], [[HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<32x16x1x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16> -> tensor<32x16x1x3xf16>

  // CHECK:    [[CONV_0:%.+]] = IE.Convolution([[INPUT_FQ]], [[WEIGHTS_FQ]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [2, 2]} : tensor<1x16x64x1024xf16>, tensor<32x16x1x3xf16> -> tensor<1x32x32x512xf16>

  // CHECK:    [[SLICED_FQ_0:%.+]] = IE.FakeQuantize([[CONV_0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x32x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x32x512xf16>

  // CHECK:    [[SLICE:%.+]] = IE.Slice [[INPUT]] [0, 0, 1, 0] [1, 16, 63, 1024] : tensor<1x16x64x1024xf16> to tensor<1x16x63x1024xf16>

  // CHECK:    [[SLICED_FQ_1:%.+]] = IE.FakeQuantize([[SLICE]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x16x63x1024xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x63x1024xf16>

  // CHECK:    [[CONV_1:%.+]] = IE.Convolution([[SLICED_FQ_1]], [[WEIGHTS_FQ]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]} : tensor<1x16x63x1024xf16>, tensor<32x16x1x3xf16> -> tensor<1x32x32x512xf16>

  // CHECK:    [[SLICED_FQ_2:%.+]] = IE.FakeQuantize([[CONV_1]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x32x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x32x512xf16>

  // CHECK:    [[CONCAT:%.+]] = IE.Concat([[SLICED_FQ_0]], [[SLICED_FQ_2]])
  // CHECK-SAME:  {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x32x32x512xf16>, tensor<1x32x32x512xf16> -> tensor<1x32x64x512xf16>

  // CHECK:    [[FINAL_FQ:%.+]] = IE.FakeQuantize([[CONCAT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x64x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x64x512xf16>

  // CHECK:    return [[FINAL_FQ]] : tensor<1x32x64x512xf16>
}
