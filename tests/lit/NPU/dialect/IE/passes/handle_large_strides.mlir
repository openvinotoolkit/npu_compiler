//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --handle-large-strides --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @HandleLargeStridesPrimeStride
func.func @HandleLargeStridesPrimeStride(%arg0: tensor<1x16x28x28xf16>) -> tensor<1x32x3x3xf16> {
  %0 = const.Declare tensor<32x16x3x3xf16> = dense<1.0> : tensor<32x16x3x3xf16>

  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [11, 11]} : tensor<1x16x28x28xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x3x3xf16>

  // CHECK-DAG:       %[[CST:.*]] = const.Declare
  // CHECK:       %[[SLICED_INPUT0:.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV0:.*]] = IE.Convolution(%[[SLICED_INPUT0]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[SLICED_INPUT1:.*]] = IE.Slice %arg0 [0, 0, 0, 11] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV1:.*]] = IE.Convolution(%[[SLICED_INPUT1]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[SLICED_INPUT2:.*]] = IE.Slice %arg0 [0, 0, 0, 22] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV2:.*]] = IE.Convolution(%[[SLICED_INPUT2]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[CONCAT0:.*]] = IE.Concat(%[[SLICED_CONV0]], %[[SLICED_CONV1]], %[[SLICED_CONV2]])
  // CHECK-SAME{LITERAL}:       {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]}
  // CHECK-SAME:      : tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x1x3xf16>

  // CHECK:       %[[SLICED_INPUT3:.*]] = IE.Slice %arg0 [0, 0, 11, 0] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV3:.*]] = IE.Convolution(%[[SLICED_INPUT3]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[SLICED_INPUT4:.*]] = IE.Slice %arg0 [0, 0, 11, 11] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV4:.*]] = IE.Convolution(%[[SLICED_INPUT4]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[SLICED_INPUT5:.*]] = IE.Slice %arg0 [0, 0, 11, 22] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV5:.*]] = IE.Convolution(%[[SLICED_INPUT5]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[CONCAT1:.*]] = IE.Concat(%[[SLICED_CONV3]], %[[SLICED_CONV4]], %[[SLICED_CONV5]])
  // CHECK-SAME{LITERAL}:       {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]}
  // CHECK-SAME:      : tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x1x3xf16>

  // CHECK:       %[[SLICED_INPUT6:.*]] = IE.Slice %arg0 [0, 0, 22, 0] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV6:.*]] = IE.Convolution(%[[SLICED_INPUT6]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[SLICED_INPUT7:.*]] = IE.Slice %arg0 [0, 0, 22, 11] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV7:.*]] = IE.Convolution(%[[SLICED_INPUT7]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[SLICED_INPUT8:.*]] = IE.Slice %arg0 [0, 0, 22, 22] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV8:.*]] = IE.Convolution(%[[SLICED_INPUT8]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[CONCAT2:.*]] = IE.Concat(%[[SLICED_CONV6]], %[[SLICED_CONV7]], %[[SLICED_CONV8]])
  // CHECK-SAME{LITERAL}:       {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]}
  // CHECK-SAME:      : tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x1x3xf16>

  // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[CONCAT0]], %[[CONCAT1]], %[[CONCAT2]])
  // CHECK-SAME{LITERAL}:       {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]}
  // CHECK-SAME:      : tensor<1x32x1x3xf16>, tensor<1x32x1x3xf16>, tensor<1x32x1x3xf16> -> tensor<1x32x3x3xf16>

  return %1 : tensor<1x32x3x3xf16>
  // CHECK        return %[[CONCAT]]
}

// -----

// CHECK-LABEL: @HandleLargeStridesNonPrimeStride
func.func @HandleLargeStridesNonPrimeStride(%arg0: tensor<1x16x28x28xf16>) -> tensor<1x32x2x2xf16> {
  %0 = const.Declare tensor<32x16x11x11xf16> = dense<1.0> : tensor<32x16x11x11xf16>

  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [10, 10]} : tensor<1x16x28x28xf16>, tensor<32x16x11x11xf16> -> tensor<1x32x2x2xf16>

  // CHECK-DAG:       %[[CST:.*]] = const.Declare
  // CHECK:       %[[CONV0:.*]] = IE.Convolution(%arg0, %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [5, 5]}

  // CHECK:       %[[MAXPOOL:.*]] = IE.MaxPool(%[[CONV0]])
  // CHECK-SAME:  {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x32x4x4xf16> -> tensor<1x32x2x2xf16>

  return %1 : tensor<1x32x2x2xf16>
  // CHECK        return %[[MAXPOOL]]
}

// -----

// CHECK-LABEL: @HandleLargeStridesAvgPool
func.func @HandleLargeStridesAvgPool(%arg0: tensor<1x16x72x128xf16>) -> tensor<1x16x8x16xf16> {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [9, 8],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [9, 8]
    } : tensor<1x16x72x128xf16> -> tensor<1x16x8x16xf16>

    return %ave_pool : tensor<1x16x8x16xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [9, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [3, 8]
    // CHECK-SAME:      : tensor<1x16x72x128xf16> -> tensor<1x16x22x16xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [3, 1]
    // CHECK-SAME:      : tensor<1x16x22x16xf16> -> tensor<1x16x8x16xf16>
}

// -----

// CHECK-LABEL: @HandleLargeAsymmetricStridesAvgPool
func.func @HandleLargeAsymmetricStridesAvgPool(%arg0: tensor<1x16x40x20xf16>) -> tensor<1x16x8x2xf16> {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [5, 10],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [5, 10]
    } : tensor<1x16x40x20xf16> -> tensor<1x16x8x2xf16>

    return %ave_pool : tensor<1x16x8x2xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [5, 10]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [5, 5]
    // CHECK-SAME:      : tensor<1x16x40x20xf16> -> tensor<1x16x8x3xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 2]
    // CHECK-SAME:      : tensor<1x16x8x3xf16> -> tensor<1x16x8x2xf16>
}

// -----

// CHECK-LABEL: @HandleLargeStridesConvolution
func.func @HandleLargeStridesConvolution(%arg0: tensor<1x1x1x2176xf16>, %arg1: tensor<258x1x1x256xf16>) -> tensor<1x2x129x16xf16> {
  %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [128, 128]} : tensor<1x1x1x2176xf16>, tensor<258x1x1x256xf16> -> tensor<1x258x1x16xf16>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 129, 16]} : tensor<1x258x1x16xf16> -> tensor<1x2x129x16xf16>
  return %1 : tensor<1x2x129x16xf16>
  // CHECK:       %[[CONV0:.*]] = IE.Convolution(%arg0, %arg1)
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [8, 8]}

  // CHECK:       %[[MAXPOOL1:.*]] = IE.MaxPool(%[[CONV0]])
  // CHECK-SAME:  {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 8]} : tensor<1x258x1x241xf16> -> tensor<1x258x1x31xf16>

  // CHECK:       %[[MAXPOOL2:.*]] = IE.MaxPool(%[[MAXPOOL1]])
  // CHECK-SAME:  {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x258x1x31xf16> -> tensor<1x258x1x16xf16>

  // CHECK:       %[[AffineReshape3:.*]] = IE.AffineReshape(%[[MAXPOOL2]])
  // CHECK-SAME:  : tensor<1x258x1x16xf16> -> tensor<1x2x129x16xf16>

  // CHECK        return %[[AffineReshape3]]
}

// -----

// CHECK-LABEL: @HandleLargeStridesAvgPoolWithSameKernelSizeAndStride
func.func @HandleLargeStridesAvgPoolWithSameKernelSizeAndStride(%arg0: tensor<1x240x11x11xf16>) -> tensor<1x240x1x1xf16> {
    %avg_pool = IE.AvgPool(%arg0) {
        kernel_size = [11, 11],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [11, 11]
    } : tensor<1x240x11x11xf16> -> tensor<1x240x1x1xf16>

    return %avg_pool : tensor<1x240x1x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [11, 11]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x240x11x11xf16> -> tensor<1x240x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeStridesAvgPoolWithFQinput
func.func @HandleLargeStridesAvgPoolWithFQinput(%arg0: tensor<1x32x22x22xf16>) -> tensor<1x32x2x2xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<40.2778358> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-40.594986> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x22x22xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x22x22xf16>
    %avg_pool = IE.AvgPool(%0) {
        kernel_size = [11, 11],
        pads_begin = [0, 0],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [11, 11]
    } : tensor<1x32x22x22xf16> -> tensor<1x32x2x2xf16>
    %fq = IE.FakeQuantize(%avg_pool, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x2x2xf16>

    return %fq : tensor<1x32x2x2xf16>

    // CHECK: %cst = const.Declare tensor<1x1x1x1xf16> = dense<40.2778358> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<-40.594986> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[FQ0:%.*]] = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x22x22xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x22x22xf16>
    // CHECK: [[SLICE0:%.*]] = IE.Slice [[FQ0:%.*]] [0, 0, 0, 0] [1, 32, 11, 11] : tensor<1x32x22x22xf16> to tensor<1x32x11x11xf16>
    // CHECK: [[FQ1:%.*]] = IE.FakeQuantize([[SLICE0:%.*]], %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x11x11xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x11x11xf16>
    // CHECK: [[AVGPOOL0:%.*]] = IE.AvgPool([[FQ1:%.*]]) {kernel_size = [11, 11], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x11x11xf16> -> tensor<1x32x1x1xf16>
    // CHECK: [[FQX:%.*]] = IE.FakeQuantize([[AVGPOOL0:%.*]], %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1x1xf16>
    // CHECK: [[SLICE1:%.*]] = IE.Slice [[FQ0:%.*]] [0, 0, 0, 11] [1, 32, 11, 10] : tensor<1x32x22x22xf16> to tensor<1x32x11x10xf16>
    // CHECK: [[FQ2:%.*]] = IE.FakeQuantize([[SLICE1:%.*]], %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x11x10xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x11x10xf16>
    // CHECK: [[AVGPOOL1:%.*]] = IE.AvgPool([[FQ2:%.*]]) {kernel_size = [11, 11], pads_begin = [0, 0], pads_end = [0, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x11x10xf16> -> tensor<1x32x1x1xf16>
    // CHECK: [[FQY:%.*]] = IE.FakeQuantize([[AVGPOOL1:%.*]], %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1x1xf16>
    // CHECK: [[CONCAT1:%.*]] = IE.Concat([[FQX:%.*]], [[FQY:%.*]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 1]]} : tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x1x2xf16>
    // CHECK: [[SLICE2:%.*]] = IE.Slice [[FQ0:%.*]] [0, 0, 11, 0] [1, 32, 10, 11] : tensor<1x32x22x22xf16> to tensor<1x32x10x11xf16>
    // CHECK: [[FQ3:%.*]] = IE.FakeQuantize([[SLICE2:%.*]], %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x10x11xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x10x11xf16>
    // CHECK: [[AVGPOOL2:%.*]] = IE.AvgPool([[FQ3:%.*]]) {kernel_size = [11, 11], pads_begin = [0, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x10x11xf16> -> tensor<1x32x1x1xf16>
    // CHECK: [[FQV:%.*]] = IE.FakeQuantize([[AVGPOOL2:%.*]], %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1x1xf16>
    // CHECK: [[SLICE3:%.*]] = IE.Slice [[FQ0:%.*]] [0, 0, 11, 11] [1, 32, 10, 10] : tensor<1x32x22x22xf16> to tensor<1x32x10x10xf16>
    // CHECK: [[FQ4:%.*]] = IE.FakeQuantize([[SLICE3:%.*]], %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x10x10xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x10x10xf16>
    // CHECK: [[AVGPOOL3:%.*]] = IE.AvgPool([[FQ4:%.*]]) {kernel_size = [11, 11], pads_begin = [0, 0], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x10x10xf16> -> tensor<1x32x1x1xf16>
    // CHECK: [[FQW:%.*]] = IE.FakeQuantize([[AVGPOOL3:%.*]], %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1x1xf16>
    // CHECK: [[CONCAT2:%.*]] = IE.Concat([[FQV:%.*]], [[FQW:%.*]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 1]]} : tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x1x2xf16>
    // CHECK: [[CONCAT3:%.*]] = IE.Concat([[CONCAT1:%.*]], [[CONCAT2:%.*]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x32x1x2xf16>, tensor<1x32x1x2xf16> -> tensor<1x32x2x2xf16>
    // CHECK: [[FQ5:%.*]] = IE.FakeQuantize([[CONCAT3:%.*]], %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x2x2xf16>
    // CHECK return %[[FQ5]]
  }

// -----

// CHECK-LABEL: @HandleLargeStridesAvgPoolMpThenGeneral
func.func @HandleLargeStridesAvgPoolMpThenGeneral(%arg0: tensor<1x16x24x42xf16>) -> tensor<1x16x2x2xf16> {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [12, 22],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [12, 22]
    } : tensor<1x16x24x42xf16> -> tensor<1x16x2x2xf16>

    return %ave_pool : tensor<1x16x2x2xf16>

    // CHECK:       [[AVGPOOL:%.*]] = IE.AvgPool(%arg0) {kernel_size = [12, 22], pads_begin = [0, 1], pads_end = [0, 1],
    // CHECK-SAME:             rounding_type = #IE.rounding_type<FLOOR>, strides = [6, 2]} : tensor<1x16x24x42xf16> -> tensor<1x16x3x12xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[AVGPOOL]] [0, 0, 0, 0] [1, 16, 1, 1] : tensor<1x16x3x12xf16> to tensor<1x16x1x1xf16>
    // CHECK:       [[MAXPOOL1:%.*]] = IE.MaxPool([[SLICE1]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK:       [[SLICE2:%.*]] = IE.Slice [[AVGPOOL]] [0, 0, 0, 11] [1, 16, 1, 1] : tensor<1x16x3x12xf16> to tensor<1x16x1x1xf16>
    // CHECK:       [[MAXPOOL2:%.*]] = IE.MaxPool([[SLICE2]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[MAXPOOL1]], [[MAXPOOL2]])
    // CHECK-SAME{LITERAL}:             {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1]]} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x2xf16>
    // CHECK:       [[SLICE3:%.*]] = IE.Slice [[AVGPOOL]] [0, 0, 2, 0] [1, 16, 1, 1] : tensor<1x16x3x12xf16> to tensor<1x16x1x1xf16>
    // CHECK:       [[MAXPOOL3:%.*]] = IE.MaxPool([[SLICE3]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK:       [[SLICE4:%.*]] = IE.Slice [[AVGPOOL]] [0, 0, 2, 11] [1, 16, 1, 1] : tensor<1x16x3x12xf16> to tensor<1x16x1x1xf16>
    // CHECK:       [[MAXPOOL4:%.*]] = IE.MaxPool([[SLICE4]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK:       [[CONCAT2:%.*]] = IE.Concat([[MAXPOOL3]], [[MAXPOOL4]])
    // CHECK-SAME{LITERAL}:             {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1]]} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x2xf16>
    // CHECK:       [[CONCAT3:%.*]] = IE.Concat([[CONCAT1]], [[CONCAT2]])
    // CHECK-SAME{LITERAL}:             {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x16x1x2xf16>, tensor<1x16x1x2xf16> -> tensor<1x16x2x2xf16>

    // CHECK:       return      [[CONCAT3]] : tensor<1x16x2x2xf16>
}
