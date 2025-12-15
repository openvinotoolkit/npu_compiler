//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --handle-asymmetric-strides --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @HandleConvolutionWithAsymmetricStrides
func.func @HandleConvolutionWithAsymmetricStrides(%arg0: tensor<1x16x64x1024xf16>) -> tensor<1x32x64x512xf16> {
  %0 = const.Declare tensor<32x16x1x3xf16> = dense<1.0> : tensor<32x16x1x3xf16>

  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 2]} : tensor<1x16x64x1024xf16>, tensor<32x16x1x3xf16> -> tensor<1x32x64x512xf16>

  return %1 : tensor<1x32x64x512xf16>

  // CHECK-DAG:       %[[CST:.*]] = const.Declare

  // CHECK:       %[[SLICED_CONV0:.*]] = IE.Convolution(%arg0, %[[CST]])
  // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [2, 2]}

  // CHECK:       %[[SLICED_INPUT1:.*]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 16, 63, 1024]

  // CHECK:       %[[SLICED_CONV1:.*]] = IE.Convolution(%[[SLICED_INPUT1]], %[[CST]])
  // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]}
  // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[SLICED_CONV0]], %[[SLICED_CONV1]])
  // CHECK-SAME:    {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>}

  // CHECK:       return %[[CONCAT]]
}

// -----

// CHECK-LABEL: func.func @ConvWithAsymStridesEqualInputAndKernel
func.func @ConvWithAsymStridesEqualInputAndKernel(%arg0: tensor<1x256x4x1xf16>) -> tensor<1x256x2x1xf16> {
  %0 = const.Declare tensor<256x256x3x1xf16> = dense<1.0> : tensor<256x256x3x1xf16>

  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 0], strides = [2, 1]} : tensor<1x256x4x1xf16>, tensor<256x256x3x1xf16> -> tensor<1x256x2x1xf16>

  return %1 : tensor<1x256x2x1xf16>

  // CHECK-DAG:       %[[CST:.*]] = const.Declare

  // CHECK:       %[[CONV:.*]] = IE.Convolution(%arg0, %[[CST]])
  // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 0], strides = [2, 2]}

  // CHECK:       return %[[CONV]]
}

// -----

// CHECK-LABEL: @HandleConvolutionWithAsymmetricStridesWithFQ
func.func @HandleConvolutionWithAsymmetricStridesWithFQ(%arg0: tensor<1x16x64x1024xf16>) -> tensor<1x32x64x512xf16> {
  %0 = const.Declare tensor<32x16x1x3xf16> = dense<1.0> : tensor<32x16x1x3xf16>
  %1 = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
  %2 = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
  %3 = const.Declare tensor<32x1x1x1xf16> = dense<0.0> : tensor<32x1x1x1xf16>
  %4 = const.Declare tensor<32x1x1x1xf16> = dense<1.0> : tensor<32x1x1x1xf16>

  %5 = IE.FakeQuantize(%arg0, %1, %2, %1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x64x1024xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x64x1024xf16>
  %6 = IE.FakeQuantize(%0, %1, %2, %3, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x16x1x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16> -> tensor<32x16x1x3xf16>
  %7 = IE.Convolution(%5, %6) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 2]} : tensor<1x16x64x1024xf16>, tensor<32x16x1x3xf16> -> tensor<1x32x64x512xf16>
  %8 = IE.FakeQuantize(%7, %1, %2, %1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x64x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x64x512xf16>

  return %8 : tensor<1x32x64x512xf16>

  // CHECK-DAG:   [[CST0:%.*]] = const.Declare tensor<32x16x1x3xf16>
  // CHECK-DAG:   [[CST1:%.*]] = const.Declare tensor<1x1x1x1xf16>
  // CHECK-DAG:   [[CST2:%.*]] = const.Declare tensor<1x1x1x1xf16>
  // CHECK-DAG:   [[CST3:%.*]] = const.Declare tensor<32x1x1x1xf16>
  // CHECK-DAG:   [[CST4:%.*]] = const.Declare tensor<32x1x1x1xf16>

  // CHECK:       [[VAR5:%.*]] = IE.FakeQuantize
  // CHECK:       [[VAR6:%.*]] = IE.FakeQuantize

  // CHECK:       [[SLICED_CONV0:%.*]] = IE.Convolution([[VAR5]], [[VAR6]])
  // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [2, 2]}

  // CHECK:       [[SLICED_FQ0:%.*]] = IE.FakeQuantize([[SLICED_CONV0]], [[CST1]], [[CST2]], [[CST1]], [[CST2]])

  // CHECK:       [[SLICED_INPUT1:%.*]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 16, 63, 1024]

  // CHECK:       [[SLICED_FQ:%.*]] = IE.FakeQuantize([[SLICED_INPUT1]], [[CST1]], [[CST2]], [[CST1]], [[CST2]])

  // CHECK:       [[SLICED_CONV1:%.*]] = IE.Convolution([[SLICED_FQ]], [[VAR6]])
  // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]}

  // CHECK:       [[SLICED_FQ1:%.*]] = IE.FakeQuantize([[SLICED_CONV1]], [[CST1]], [[CST2]], [[CST1]], [[CST2]])

  // CHECK:       [[CONCAT:%.*]] = IE.Concat([[SLICED_FQ0]], [[SLICED_FQ1]])
  // CHECK-SAME:    {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>}

  // CHECK:       [[OUT_FQ:%.*]] = IE.FakeQuantize([[CONCAT]], [[CST1]], [[CST2]], [[CST1]], [[CST2]])

  // CHECK:       return [[OUT_FQ]]
}
