//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --align-scales %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @AlignConcatScales
func.func @AlignConcatScales(%arg0: tensor<16x8x8xf16>, %arg1: tensor<16x1x8xf16>) -> tensor<1x16x5x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<3.068850e-01> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  %0 = IE.Reshape(%arg0) { shape_value = [1, 16, 8, 8] } : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  %1 = IE.Reshape(%arg1) { shape_value = [1, 16, 1, 8] } : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  %2 = IE.FakeQuantize(%0, %cst_1, %cst, %cst_1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %3 = IE.FakeQuantize(%1, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  %5 = IE.Convolution(%4, %cst_2) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  return %5 : tensor<1x16x5x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.755859375> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  // CHECK: [[RESHAPE:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 8, 8]} : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[RESHAPE_0:%.*]] = IE.Reshape(%arg1) {shape_value = [1, 16, 1, 8]} : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[FQ:%.*]] = IE.FakeQuantize([[RESHAPE]], [[CST_2]], [[CST_3]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[RESHAPE_0]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[CLAMP:%.*]] = IE.Clamp([[FQ_0]]) {max = 0.306884765625 : f64, min = 0.000000e+00 : f64} : tensor<1x16x1x8xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[FQ]], [[CLAMP]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  // CHECK: [[CONV:%.*]] = IE.Convolution([[CONCAT]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  // CHECK: return [[CONV]] : tensor<1x16x5x8xf16>
}

// -----

// CHECK-LABEL: @AlignConcatMaxPool
func.func @AlignConcatMaxPool(%arg0: tensor<1x128x1x256xf16>, %arg1: tensor<1x128x1x256xf16>) -> tensor<1x128x1x128xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e+00> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<3.068850e+00> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-2.700000e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<10.756000e+00> : tensor<1x1x1x1xf16>
  %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<-3.643000e+00> : tensor<1x1x1x1xf16>
  %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x256xf16>, tensor<1x128x1x256xf16> -> tensor<1x128x1x256xf16>
  %1 = IE.Add(%arg1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x256xf16>, tensor<1x128x1x256xf16> -> tensor<1x128x1x256xf16>
  %2 = IE.FakeQuantize(%0, %cst_1, %cst, %cst_1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x128x1x256xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x1x256xf16>
  %3 = IE.FakeQuantize(%1, %cst_2, %cst_0, %cst_2, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x128x1x256xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x1x256xf16>
  %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x128x1x256xf16>, tensor<1x128x1x256xf16> -> tensor<1x128x2x256xf16>
  %5 = IE.FakeQuantize(%4, %cst_4, %cst_3, %cst_4, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x128x2x256xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x2x256xf16>
  %6 = IE.MaxPool(%5) { kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2] } : tensor<1x128x2x256xf16> -> tensor<1x128x1x128xf16>

  return %6 : tensor<1x128x1x128xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-3.67068768> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<10.7297029> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK: [[ADD:%.*]] = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x256xf16>, tensor<1x128x1x256xf16> -> tensor<1x128x1x256xf16>
  // CHECK: [[ADD_0:%.*]] = IE.Add(%arg1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x256xf16>, tensor<1x128x1x256xf16> -> tensor<1x128x1x256xf16>
  // CHECK: [[FQ:%.*]] = IE.FakeQuantize([[ADD]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x128x1x256xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x1x256xf16>
  // CHECK: [[CLAMP:%.*]] = IE.Clamp([[FQ]]) {max = 7.55859375 : f64, min = -2.69921875 : f64} : tensor<1x128x1x256xf16> -> tensor<1x128x1x256xf16>
  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[ADD_0]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x128x1x256xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x1x256xf16>
  // CHECK: [[CLAMP_0:%.*]] = IE.Clamp([[FQ_0]]) {max = 3.068359375 : f64, min = -1.000000e+00 : f64} : tensor<1x128x1x256xf16> -> tensor<1x128x1x256xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[CLAMP]], [[CLAMP_0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x128x1x256xf16>, tensor<1x128x1x256xf16> -> tensor<1x128x2x256xf16>
  // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize([[CONCAT]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x128x2x256xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x2x256xf16>
  // CHECK: [[CLAMP_1:%.*]] = IE.Clamp([[FQ_1]]) {max = 10.7578125 : f64, min = -3.642578125 : f64} : tensor<1x128x2x256xf16> -> tensor<1x128x2x256xf16>
  // CHECK: [[MAXPOOL:%.*]] = IE.MaxPool([[CLAMP_1]]) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x128x2x256xf16> -> tensor<1x128x1x128xf16>

  // CHECK: return [[MAXPOOL]] : tensor<1x128x1x128xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: @AlignInNotOutConcatReshapeFQAgnostic
func.func @AlignInNotOutConcatReshapeFQAgnostic(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<1x2x4x3xf16>) -> tensor<1x1x12x4xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<3.18300e+00> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<3.068850e+00> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-10.443000e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<10.756000e+00> : tensor<1x1x1x1xf16>
  %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<-3.643000e+00> : tensor<1x1x1x1xf16>
  %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
  %1 = IE.FakeQuantize(%0, %cst_1, %cst, %cst_1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  %2 = IE.FakeQuantize(%arg1, %cst_2, %cst_0, %cst_2, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x4x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x4x3xf16>
  %3 = IE.Transpose(%2) {order_value = #NCWH} : tensor<1x2x4x3xf16> -> tensor<1x2x3x4xf16>
  %4 = IE.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
  %5 = IE.Reshape(%4) {shape_value = [1, 1, 12, 4] } : tensor<1x4x3x4xf16> -> tensor<1x1x12x4xf16>
  %6 = IE.FakeQuantize(%5, %cst_4, %cst_3, %cst_4, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x12x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x12x4xf16>

  return %6 : tensor<1x1x12x4xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-10.4768381> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<10.7262869> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK: [[ADD:%.*]] = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[FQ:%.*]] = IE.FakeQuantize([[ADD]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CLAMP:%.*]] = IE.Clamp([[FQ]]) {max = 3.18359375 : f64, min = -10.4453125 : f64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize(%arg1, [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x4x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x4x3xf16>
  // CHECK: [[CLAMP_0:%.*]] = IE.Clamp([[FQ_0]]) {max = 3.068359375 : f64, min = 0.000000e+00 : f64} : tensor<1x2x4x3xf16> -> tensor<1x2x4x3xf16>
  // CHECK: [[TRANSPOSE:%.*]] = IE.Transpose([[CLAMP_0]]) {order_value = #NCWH} : tensor<1x2x4x3xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[CLAMP]], [[TRANSPOSE]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 2, 0, 0]]} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
  // CHECK: [[RESHAPE:%.*]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 1, 12, 4]} : tensor<1x4x3x4xf16> -> tensor<1x1x12x4xf16>
  // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize([[RESHAPE]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x12x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x12x4xf16>
  // CHECK: [[CLAMP_1:%.*]] = IE.Clamp([[FQ_1]]) {max = 10.7578125 : f64, min = -3.642578125 : f64} : tensor<1x1x12x4xf16> -> tensor<1x1x12x4xf16>

  // CHECK:   return [[CLAMP_1]] : tensor<1x1x12x4xf16>
}

// -----

// CHECK-LABEL: @NotAlignToDifferentFQRanges
func.func @NotAlignToDifferentFQRanges(%arg0: tensor<16x8x8xf16>, %arg1: tensor<16x1x8xf16>) -> tensor<1x16x5x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<-3.068850e-01> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e+01> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<-5.000000e+01> : tensor<1x1x1x1xf16>
  %cst_3 = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+01> : tensor<16x16x5x3xf16>
  %0 = IE.Reshape(%arg0) {shape_value = [1, 16, 8, 8]} : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  %1 = IE.Reshape(%arg1) {shape_value = [1, 16, 1, 8]} : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  %2 = IE.FakeQuantize(%0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %3 = IE.FakeQuantize(%1, %cst_2, %cst_1, %cst_2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  %5 = IE.Convolution(%4, %cst_3) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>
  return %5 : tensor<1x16x5x8xf16>

  // CHECK : [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  // CHECK : [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-3.068850e-01> : tensor<1x1x1x1xf16>
  // CHECK : [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e+01> : tensor<1x1x1x1xf16>
  // CHECK : [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-5.000000e+01> : tensor<1x1x1x1xf16>
  // CHECK : [[CST_3:%.*]] = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+01> : tensor<16x16x5x3xf16>
  // CHECK : [[RESHAPE:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 8, 8]} : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK : [[RESHAPE_0:%.*]] = IE.Reshape(%arg1) {shape_value = [1, 16, 1, 8]} : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  // CHECK : [[FQ:%.*]] = IE.FakeQuantize([[RESHAPE]], [[CST_0]], [[CST]], [[CST_0]]_0, [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK : [[FQ_0:%.*]] = IE.FakeQuantize([[RESHAPE_0]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  // CHECK : [[CONCAT:%.*]] = IE.Concat([[FQ]], [[FQ_0]]) {static_offsets = {{\[\[}0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  // CHECK : [[CONV:%.*]] = IE.Convolution([[CONCAT]], [[CST_3]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  // CHECK : return [[CONV]] : tensor<1x16x5x8xf16>
}

// -----

// CHECK-LABEL: @AlignOnlyFQAgnostic
func.func @AlignOnlyFQAgnostic(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<1x2x3x4xf16>, %arg2: tensor<1x2x3x4xf16>) -> (tensor<1x1x3x2xf16>, tensor<1x2x3x4xf16>) {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<-1.5000e+00> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<4.0000e+00> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-2.4000e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.5000e+00> : tensor<1x1x1x1xf16>
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-1.7500e+00> : tensor<1x1x1x1xf16>
  %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<3.7500e+00> : tensor<1x1x1x1xf16>
  %cst_5 = const.Declare tensor<1x1x1x1xf16> = dense<-6.2500e+00> : tensor<1x1x1x1xf16>
  %cst_6 = const.Declare tensor<1x1x1x1xf16> = dense<5.3000e+00> : tensor<1x1x1x1xf16>
  %cst_7 = const.Declare tensor<1x1x1x1xf16> = dense<-2.4500e+00> : tensor<1x1x1x1xf16>
  %cst_8 = const.Declare tensor<1x1x1x1xf16> = dense<6.7000e+00> : tensor<1x1x1x1xf16>
  %cst_9 = const.Declare tensor<1x1x1x1xf16> = dense<-2.200e+00> : tensor<1x1x1x1xf16>
  %cst_10 = const.Declare tensor<1x1x1x1xf16> = dense<5.7000e+00> : tensor<1x1x1x1xf16>
  %0 = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  %1:2 = IE.Split(%0) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x3x4xf16> -> tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>
  %2 = IE.AvgPool(%1#0) {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2] } : tensor<1x1x3x4xf16> -> tensor<1x1x3x2xf16>
  %3 = IE.FakeQuantize(%2, %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x3x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x2xf16>
  %4 = IE.FakeQuantize(%arg1, %cst_3, %cst_4, %cst_3, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  %5 = IE.FakeQuantize(%arg2, %cst_5, %cst_6, %cst_5, %cst_6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
  %7 = IE.ReduceSum(%6) {axes_value = [1], keep_dims} : tensor<1x4x3x4xf16> -> tensor<1x1x3x4xf16>
  %8 = IE.FakeQuantize(%7, %cst_9, %cst_10, %cst_9, %cst_10) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x4xf16>
  %9 = IE.Concat(%1#1, %8) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16> -> tensor<1x2x3x4xf16>
  %10 = IE.FakeQuantize(%9, %cst_7, %cst_8, %cst_7, %cst_8) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>

  return %3, %10 : tensor<1x1x3x2xf16>, tensor<1x2x3x4xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.4395833> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<6.7088542> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-6.25101137> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<5.29977036> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.500000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_4:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.400390e+00> : tensor<1x1x1x1xf16>
  // CHECK: [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CLAMP:%.*]] = IE.Clamp([[FQ]]) {max = 4.000000e+00 : f64, min = -1.500000e+00 : f64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[SPLIT:%.*]]:2 = IE.Split([[CLAMP]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x3x4xf16> -> tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>
  // CHECK: [[AVGPOOL:%.*]] = IE.AvgPool([[SPLIT]]#0) {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2]} : tensor<1x1x3x4xf16> -> tensor<1x1x3x2xf16>
  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[AVGPOOL]], [[CST_4]], [[CST_3]], [[CST_4]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x3x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x2xf16>
  // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize(%arg1, [[CST_1]], [[CST_2]], [[CST_1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CLAMP_0:%.*]] = IE.Clamp([[FQ_1]]) {max = 3.750000e+00 : f64, min = -1.750000e+00 : f64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[FQ_2:%.*]] = IE.FakeQuantize(%arg2, [[CST_1]], [[CST_2]], [[CST_1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CLAMP_1:%.*]] = IE.Clamp([[FQ_2]]) {max = 5.30078125 : f64, min = -6.250000e+00 : f64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[CLAMP_0]], [[CLAMP_1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 2, 0, 0]]} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
  // CHECK: [[REDUCESUM:%.*]] = IE.ReduceSum([[CONCAT]]) {axes_value = [1], keep_dims} : tensor<1x4x3x4xf16> -> tensor<1x1x3x4xf16>
  // CHECK: [[FQ_3:%.*]] = IE.FakeQuantize([[REDUCESUM]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x4xf16>
  // CHECK: [[CLAMP_2:%.*]] = IE.Clamp([[FQ_3]]) {max = 5.69921875 : f64, min = -2.19921875 : f64} : tensor<1x1x3x4xf16> -> tensor<1x1x3x4xf16>
  // CHECK: [[CONCAT_0:%.*]] = IE.Concat([[SPLIT]]#1, [[CLAMP_2]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[FQ_4:%.*]] = IE.FakeQuantize([[CONCAT_0]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CLAMP_3:%.*]] = IE.Clamp([[FQ_4]]) {max = 6.69921875 : f64, min = -2.44921875 : f64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>

  // CHECK: return [[FQ_0]], [[CLAMP_3]] : tensor<1x1x3x2xf16>, tensor<1x2x3x4xf16>
}

// -----

// CHECK-LABEL: @NotAlignSameFQRanges
func.func @NotAlignSameFQRanges(%arg0: tensor<16x7x8xf16>, %arg1: tensor<16x1x8xf16>, %arg2: tensor<16x1x8xf16>) -> tensor<1x16x5x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e-01> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<-0.000000e-01> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<15.000000e+01> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<-15.000000e+01> : tensor<1x1x1x1xf16>
  %cst_3 = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+01> : tensor<16x16x5x3xf16>
  %0 = IE.Reshape(%arg0) {shape_value = [1, 16, 7, 8]} : tensor<16x7x8xf16> -> tensor<1x16x7x8xf16>
  %1 = IE.Reshape(%arg1) {shape_value = [1, 16, 1, 8]} : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  %2 = IE.Reshape(%arg2) {shape_value = [1, 16, 1, 8]} : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  %3 = IE.FakeQuantize(%0, %cst_2, %cst_1, %cst_2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x7x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x7x8xf16>
  %4 = IE.FakeQuantize(%1, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  %5 = IE.FakeQuantize(%2, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  %6 = IE.Concat(%3, %4, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 8, 0]]} : tensor<1x16x7x8xf16>, tensor<1x16x1x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  %7 = IE.Convolution(%6, %cst_3) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>
  return %7 : tensor<1x16x5x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+01> : tensor<16x16x5x3xf16>
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.500000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.500000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.548830e-01> : tensor<1x1x1x1xf16>

  // CHECK: [[RESHAPE:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 7, 8]} : tensor<16x7x8xf16> -> tensor<1x16x7x8xf16>
  // CHECK: [[RESHAPE_0:%.*]] = IE.Reshape(%arg1) {shape_value = [1, 16, 1, 8]} : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[RESHAPE_1:%.*]] = IE.Reshape(%arg2) {shape_value = [1, 16, 1, 8]} : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[FQ:%.*]] = IE.FakeQuantize([[RESHAPE]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x7x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x7x8xf16>
  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[RESHAPE_0]], [[CST_2]], [[CST_3]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize([[RESHAPE_1]], [[CST_2]], [[CST_3]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[FQ]], [[FQ_0]], [[FQ_1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 8, 0]]}  : tensor<1x16x7x8xf16>, tensor<1x16x1x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  // CHECK: [[CONV:%.*]] = IE.Convolution([[CONCAT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  // CHECK: return [[CONV]] : tensor<1x16x5x8xf16>
}

// -----

// CHECK-LABEL: @AlignConsecutiveCocats
func.func @AlignConsecutiveCocats(%arg0: tensor<1x32x144x128xf16>, %arg1: tensor<1x32x144x128xf16>) -> tensor<1x32x144x384xf16> {
  %cst = const.Declare tensor<32x1x1x2xf16> = dense<1.000000e+00> : tensor<32x1x1x2xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<6.000000e+00> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<2.540000e+02> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<2.500000e+02> : tensor<1x1x1x1xf16>
  %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<-3.000000e+00> : tensor<1x1x1x1xf16>
  %cst_5 = const.Declare tensor<1x1x1x1xf16> = dense<1.20000e+02> : tensor<1x1x1x1xf16>
  %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x144x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x144x128xf16>
  %1 = IE.FakeQuantize(%arg1, %cst_0, %cst_3, %cst_0, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x144x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x144x128xf16>
  %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x32x144x128xf16>, tensor<1x32x144x128xf16> -> tensor<1x32x144x256xf16>
  %3 = IE.Slice %2 [0, 0, 0, 255] [1, 32, 144, 1] : tensor<1x32x144x256xf16> to tensor<1x32x144x1xf16>
  %4 = IE.Concat(%2, %3) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x32x144x256xf16>, tensor<1x32x144x1xf16> -> tensor<1x32x144x257xf16>
  %5 = IE.MaxPool(%1) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x144x128xf16> -> tensor<1x32x144x128xf16>
  %6 = IE.FakeQuantize(%cst, %cst_2, %cst_3, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x1x1x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<32x1x1x2xf16>
  %7 = IE.GroupConvolution(%4, %6) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x144x257xf16>, tensor<32x1x1x2xf16> -> tensor<1x32x144x256xf16>
  %8 = IE.FakeQuantize(%7, %cst_2, %cst_1, %cst_2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x144x256xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x144x256xf16>
  %9 = IE.Concat(%5, %7) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x32x144x128xf16>, tensor<1x32x144x256xf16> -> tensor<1x32x144x384xf16>
  %10 = IE.FakeQuantize(%9, %cst_4, %cst_5, %cst_4, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x144x384xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x144x384xf16>

  return %10 : tensor<1x32x144x384xf16>


  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-3.02352953> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<253.976471> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.500000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_3:%.*]] = const.Declare tensor<32x1x1x2xf16> = dense<1.000000e+00> : tensor<32x1x1x2xf16>
  // CHECK: [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x144x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x144x128xf16>
  // CHECK: [[CLAMP:%.*]] = IE.Clamp([[FQ]]) {max = 2.540000e+02 : f64, min = 6.000000e+00 : f64} : tensor<1x32x144x128xf16> -> tensor<1x32x144x128xf16>
  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize(%arg1, [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x144x128xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x144x128xf16>
  // CHECK: [[CLAMP_0:%.*]] = IE.Clamp([[FQ_0]]) {max = 2.500000e+02 : f64, min = 6.000000e+00 : f64} : tensor<1x32x144x128xf16> -> tensor<1x32x144x128xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[CLAMP]], [[CLAMP_0]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x32x144x128xf16>, tensor<1x32x144x128xf16> -> tensor<1x32x144x256xf16>
  // CHECK: [[SLICE:%.*]] = IE.Slice [[CONCAT]] [0, 0, 0, 255] [1, 32, 144, 1] : tensor<1x32x144x256xf16> to tensor<1x32x144x1xf16>
  // CHECK: [[CONCAT_0:%.*]] = IE.Concat([[CONCAT]], [[SLICE]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x32x144x256xf16>, tensor<1x32x144x1xf16> -> tensor<1x32x144x257xf16>
  // CHECK: [[MAXPOOL:%.*]] = IE.MaxPool([[CLAMP_0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x144x128xf16> -> tensor<1x32x144x128xf16>
  // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize([[CST_3]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x1x1x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<32x1x1x2xf16>
  // CHECK: [[GROUPCONV:%.*]] = IE.GroupConvolution([[CONCAT_0]], [[FQ_1]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x144x257xf16>, tensor<32x1x1x2xf16> -> tensor<1x32x144x256xf16>
  // CHECK: [[CONCAT_1:%.*]] = IE.Concat([[MAXPOOL]], [[GROUPCONV]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x32x144x128xf16>, tensor<1x32x144x256xf16> -> tensor<1x32x144x384xf16>
  // CHECK: [[FQ_2:%.*]] = IE.FakeQuantize([[CONCAT_1]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x144x384xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x144x384xf16>
  // CHECK: [[CLAMP_1:%.*]] = IE.Clamp([[FQ_2]]) {max = 1.200000e+02 : f64, min = -3.000000e+00 : f64} : tensor<1x32x144x384xf16> -> tensor<1x32x144x384xf16>

  // CHECK:  return [[CLAMP_1]] : tensor<1x32x144x384xf16>
}

// -----

// CHECK-LABEL: @AlignDifferentInOutWithClampOnSameBranch
func.func @AlignDifferentInOutWithClampOnSameBranch(%arg0: tensor<16x8x8xf16>, %arg1: tensor<16x1x8xf16>) -> tensor<1x16x5x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<3.068850e-01> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<3.779295e-01> : tensor<1x1x1x1xf16>
  %0 = IE.Reshape(%arg0) { shape_value = [1, 16, 8, 8] } : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  %1 = IE.Reshape(%arg1) { shape_value = [1, 16, 1, 8] } : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  %2 = IE.FakeQuantize(%0, %cst_1, %cst_3, %cst_1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %3 = IE.FakeQuantize(%1, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  %5 = IE.Convolution(%4, %cst_2) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  return %5 : tensor<1x16x5x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.755859375> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_3:%.*]] = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  // CHECK-DAG: [[CST_4:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<3.779300e-01> : tensor<1x1x1x1xf16>
  // CHECK: [[RESHAPE:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 8, 8]} : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[RESHAPE_0:%.*]] = IE.Reshape(%arg1) {shape_value = [1, 16, 1, 8]} : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>

  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[RESHAPE]], [[CST_2]], [[CST_4]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize([[FQ_0]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[FQ_2:%.*]] = IE.FakeQuantize([[RESHAPE_0]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[CLAMP:%.*]] = IE.Clamp([[FQ_2]]) {max = 0.306884765625 : f64, min = 0.000000e+00 : f64} : tensor<1x16x1x8xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[FQ_1]], [[CLAMP]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  // CHECK: [[CONV:%.*]] = IE.Convolution([[CONCAT]], [[CST_3]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  // CHECK: return [[CONV]] : tensor<1x16x5x8xf16>
}


// -----

// CHECK-LABEL: @AlignDifferentInOutWithClampOnDifferentBranch
func.func @AlignDifferentInOutWithClampOnDifferentBranch(%arg0: tensor<16x8x8xf16>, %arg1: tensor<16x1x8xf16>) -> tensor<1x16x5x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<8.068850e-01> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<3.779295e-01> : tensor<1x1x1x1xf16>
  %0 = IE.Reshape(%arg0) { shape_value = [1, 16, 8, 8] } : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  %1 = IE.Reshape(%arg1) { shape_value = [1, 16, 1, 8] } : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  %2 = IE.FakeQuantize(%0, %cst_1, %cst_3, %cst_1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %3 = IE.FakeQuantize(%1, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  %5 = IE.Convolution(%4, %cst_2) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  return %5 : tensor<1x16x5x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.807128906> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<8.071280e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_4:%.*]] = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  // CHECK-DAG: [[CST_5:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<3.779300e-01> : tensor<1x1x1x1xf16>
  // CHECK: [[RESHAPE:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 8, 8]} : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[RESHAPE_0:%.*]] = IE.Reshape(%arg1) {shape_value = [1, 16, 1, 8]} : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>

  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[RESHAPE]], [[CST_3]], [[CST_5]], [[CST_3]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize([[FQ_0]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[CLAMP:%.*]] = IE.Clamp([[FQ_1]]) {max = 0.755859375 : f64, min = 0.000000e+00 : f64} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[FQ_2:%.*]] = IE.FakeQuantize([[RESHAPE_0]], [[CST_3]], [[CST_2]], [[CST_3]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[CLAMP]], [[FQ_2]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  // CHECK: [[CONV:%.*]] = IE.Convolution([[CONCAT]], [[CST_4]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  // CHECK: return [[CONV]] : tensor<1x16x5x8xf16>
}


// -----

// CHECK-LABEL: @NotAlignDifferentInOutWithRangeOutOfRequest
func.func @NotAlignDifferentInOutWithRangeOutOfRequest(%arg0: tensor<16x8x8xf16>, %arg1: tensor<16x1x8xf16>) -> tensor<1x16x9x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<3.068850e-01> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %0 = IE.Reshape(%arg0) { shape_value = [1, 16, 8, 8] } : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  %1 = IE.Reshape(%arg1) { shape_value = [1, 16, 1, 8] } : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  %2 = IE.FakeQuantize(%0, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %3 = IE.FakeQuantize(%1, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  %5 = IE.FakeQuantize(%4, %cst_1, %cst_0, %cst_1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x9x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x9x8xf16>
  %6 = IE.MaxPool(%5) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1] } : tensor<1x16x9x8xf16> -> tensor<1x16x9x8xf16>

  return %6 : tensor<1x16x9x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<3.068850e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK: [[RESHAPE:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 8, 8]} : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[RESHAPE_0:%.*]] = IE.Reshape(%arg1) {shape_value = [1, 16, 1, 8]} : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>

  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[RESHAPE]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize([[RESHAPE_0]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[FQ_0]], [[FQ_1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  // CHECK: [[FQ_2:%.*]] = IE.FakeQuantize([[CONCAT]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x9x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x9x8xf16>
  // CHECK: [[POOLING:%.*]] = IE.MaxPool([[FQ_2]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x9x8xf16> -> tensor<1x16x9x8xf16>
  // CHECK: return [[POOLING]] : tensor<1x16x9x8xf16>
}


// -----

// CHECK-LABEL: @PartialAlignDifferentInOut
func.func @PartialAlignDifferentInOut(%arg0: tensor<16x8x8xf16>, %arg1: tensor<16x1x8xf16>) -> tensor<1x16x5x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<8.068850e-01> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<3.778080e-02> : tensor<1x1x1x1xf16>
  %0 = IE.Reshape(%arg0) { shape_value = [1, 16, 8, 8] } : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  %1 = IE.Reshape(%arg1) { shape_value = [1, 16, 1, 8] } : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  %2 = IE.FakeQuantize(%0, %cst_1, %cst_3, %cst_1, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %3 = IE.MaxPool(%2) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1] } : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>
  %4 = IE.FakeQuantize(%3, %cst_1, %cst_3, %cst_1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %5 = IE.FakeQuantize(%1, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  %7 = IE.Convolution(%6, %cst_2) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  return %7 : tensor<1x16x5x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.807128906> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<8.071280e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_4:%.*]] = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  // CHECK-DAG: [[CST_5:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<3.778080e-02> : tensor<1x1x1x1xf16>
  // CHECK: [[RESHAPE:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 8, 8]} : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[RESHAPE_0:%.*]] = IE.Reshape(%arg1) {shape_value = [1, 16, 1, 8]} : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[RESHAPE]], [[CST_3]], [[CST_5]], [[CST_3]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[MAXPOOL:%.*]] = IE.MaxPool([[FQ_0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize([[MAXPOOL]], [[CST_3]], [[CST_5]], [[CST_3]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[FQ_2:%.*]] = IE.FakeQuantize([[FQ_1]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[CLAMP:%.*]] = IE.Clamp([[FQ_2]]) {max = 0.755859375 : f64, min = 0.000000e+00 : f64} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[FQ_3:%.*]] = IE.FakeQuantize([[RESHAPE_0]], [[CST_3]], [[CST_2]], [[CST_3]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[CLAMP]], [[FQ_3]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  // CHECK: [[CONV:%.*]] = IE.Convolution([[CONCAT]], [[CST_4]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  // CHECK: return [[CONV]] : tensor<1x16x5x8xf16>
}


// -----

// CHECK-LABEL: @PartialAlignDifferentInOutRangeWithMultiUser
func.func @PartialAlignDifferentInOutRangeWithMultiUser(%arg0: tensor<16x8x8xf16>, %arg1: tensor<16x1x8xf16>) -> (tensor<1x16x5x8xf16>, tensor<1x16x8x8xf16>) {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<8.068850e-01> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<3.778080e-02> : tensor<1x1x1x1xf16>
  %0 = IE.Reshape(%arg0) { shape_value = [1, 16, 8, 8] } : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  %1 = IE.Reshape(%arg1) { shape_value = [1, 16, 1, 8] } : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  %2 = IE.FakeQuantize(%0, %cst_1, %cst_3, %cst_1, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %3 = IE.MaxPool(%2) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1] } : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>
  %4 = IE.FakeQuantize(%3, %cst_1, %cst_3, %cst_1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %5 = IE.FakeQuantize(%1, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>

  %8 = IE.MaxPool(%4) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1] } : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>
  %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  %7 = IE.Convolution(%6, %cst_2) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  return %7, %8 : tensor<1x16x5x8xf16>, tensor<1x16x8x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.807128906> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<8.071280e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_4:%.*]] = const.Declare tensor<16x16x5x3xf16> = dense<1.000000e+00> : tensor<16x16x5x3xf16>
  // CHECK-DAG: [[CST_5:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<3.778080e-02> : tensor<1x1x1x1xf16>
  // CHECK: [[RESHAPE:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 8, 8]} : tensor<16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[RESHAPE_0:%.*]] = IE.Reshape(%arg1) {shape_value = [1, 16, 1, 8]} : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[RESHAPE]], [[CST_3]], [[CST_5]], [[CST_3]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[MAXPOOL:%.*]] = IE.MaxPool([[FQ_0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize([[MAXPOOL]], [[CST_3]], [[CST_5]], [[CST_3]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[FQ_2:%.*]] = IE.FakeQuantize([[FQ_1]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[CLAMP:%.*]] = IE.Clamp([[FQ_2]]) {max = 0.755859375 : f64, min = 0.000000e+00 : f64} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[FQ_3:%.*]] = IE.FakeQuantize([[RESHAPE_0]], [[CST_3]], [[CST_2]], [[CST_3]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[MAXPOOL_0:%.*]] = IE.MaxPool([[FQ_1]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[CLAMP]], [[FQ_3]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>
  // CHECK: [[CONV:%.*]] = IE.Convolution([[CONCAT]], [[CST_4]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x9x8xf16>, tensor<16x16x5x3xf16> -> tensor<1x16x5x8xf16>

  // CHECK: return [[CONV]], [[MAXPOOL_0]] : tensor<1x16x5x8xf16>, tensor<1x16x8x8xf16>
}


// -----

// CHECK-LABEL: @AlignSliceWithOutClamp
func.func @AlignSliceWithOutClamp(%arg0: tensor<1x16x8x8xf16>) -> tensor<1x10x8x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<13.068850e-01> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %0 = IE.FakeQuantize(%arg0, %cst_1, %cst, %cst_1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %1 = IE.Slice %0 [0, 0, 0, 0] [1, 10, 8, 8] : tensor<1x16x8x8xf16> to tensor<1x10x8x8xf16>
  %2 = IE.FakeQuantize(%1, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x10x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x10x8x8xf16>

  return %2 : tensor<1x10x8x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_0:%.*]] =  const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK:     [[FQ_0:%.*]] = IE.FakeQuantize(%arg0, [[CST_0]], [[CST]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK:     [[SLICE:%.*]] = IE.Slice [[FQ_0]] [0, 0, 0, 0] [1, 10, 8, 8] : tensor<1x16x8x8xf16> to tensor<1x10x8x8xf16>
  // CHECK:     [[FQ_1:%.*]] = IE.FakeQuantize([[SLICE]], [[CST_0]], [[CST]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x10x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x10x8x8xf16>

  // CHECK: return [[FQ_1]] : tensor<1x10x8x8xf16>
}

// -----

// CHECK-LABEL: @AlignSliceWithClamp
func.func @AlignSliceWithClamp(%arg0: tensor<1x16x8x8xf16>) -> tensor<1x10x8x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e+00> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<5.068850e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
  %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %1 = IE.Slice %0 [0, 0, 0, 0] [1, 10, 8, 8] : tensor<1x16x8x8xf16> to tensor<1x10x8x8xf16>
  %2 = IE.FakeQuantize(%1, %cst_2, %cst_1, %cst_2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x10x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x10x8x8xf16>

  return %2 : tensor<1x10x8x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_0:%.*]] =  const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK:     [[FQ_0:%.*]] = IE.FakeQuantize(%arg0, [[CST_0]], [[CST]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  // CHECK:     [[SLICE:%.*]] = IE.Slice [[FQ_0]] [0, 0, 0, 0] [1, 10, 8, 8] : tensor<1x16x8x8xf16> to tensor<1x10x8x8xf16>
  // CHECK:     [[FQ_1:%.*]] = IE.FakeQuantize([[SLICE]], [[CST_0]], [[CST]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x10x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x10x8x8xf16>
  // CHECK:     [[CLAMP:%.*]] = IE.Clamp([[FQ_1]]) {max = 5.0703125 : f64, min = 1.000000e+00 : f64} : tensor<1x10x8x8xf16> -> tensor<1x10x8x8xf16>
  // CHECK: return [[CLAMP]] : tensor<1x10x8x8xf16>
}


// -----

// CHECK-LABEL: @NotAlignSliceWithBigRange
func.func @NotAlignSliceWithBigRange(%arg0: tensor<1x16x8x8xf16>) -> tensor<1x10x8x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e+01> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<5.068850e+00> : tensor<1x1x1x1xf16>
  %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %1 = IE.Slice %0 [0, 0, 0, 0] [1, 10, 8, 8] : tensor<1x16x8x8xf16> to tensor<1x10x8x8xf16>
  %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x10x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x10x8x8xf16>

  return %2 : tensor<1x10x8x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.556250e+01> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<5.070310e+00> : tensor<1x1x1x1xf16>

  // CHECK:     [[FQ_0:%.*]] = IE.FakeQuantize(%arg0, [[CST_0]], [[CST]], [[CST_0]], [[CST]])
  // CHECK:     [[SLICE:%.*]] = IE.Slice [[FQ_0]] [0, 0, 0, 0] [1, 10, 8, 8]
  // CHECK:     [[FQ_1:%.*]] = IE.FakeQuantize([[SLICE]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]])

  // CHECK: return [[FQ_1]] : tensor<1x10x8x8xf16>
}

// -----

// CHECK-LABEL: @NotAlignSliceWithNoOverlap
func.func @NotAlignSliceWithNoOverlap(%arg0: tensor<1x16x8x8xf16>) -> tensor<1x10x8x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e+00> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<6.000000e+00> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<5.068850e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
  %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %1 = IE.Slice %0 [0, 0, 0, 0] [1, 10, 8, 8] : tensor<1x16x8x8xf16> to tensor<1x10x8x8xf16>
  %2 = IE.FakeQuantize(%1, %cst_2, %cst_1, %cst_2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x10x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x10x8x8xf16>

  return %2 : tensor<1x10x8x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<6.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<5.070310e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

  // CHECK:     [[FQ_0:%.*]] = IE.FakeQuantize(%arg0, [[CST_0]], [[CST]], [[CST_0]], [[CST]])
  // CHECK:     [[SLICE:%.*]] = IE.Slice [[FQ_0]] [0, 0, 0, 0] [1, 10, 8, 8]
  // CHECK:     [[FQ_1:%.*]] = IE.FakeQuantize([[SLICE]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]])

  // CHECK: return [[FQ_1]] : tensor<1x10x8x8xf16>
}


// -----

// CHECK-LABEL: @NotAlignSliceWithDifferentInOutRange
func.func @NotAlignSliceWithDifferentInOutRange(%arg0: tensor<1x16x8x8xf16>) -> tensor<1x10x8x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e+00> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<5.068850e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf16>
  %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %1 = IE.Slice %0 [0, 0, 0, 0] [1, 10, 8, 8] : tensor<1x16x8x8xf16> to tensor<1x10x8x8xf16>
  %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x10x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x10x8x8xf16>

  return %2 : tensor<1x10x8x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<5.070310e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK:     [[FQ_0:%.*]] = IE.FakeQuantize(%arg0, [[CST_0]], [[CST]], [[CST_0]], [[CST]])
  // CHECK:     [[SLICE:%.*]] = IE.Slice [[FQ_0]] [0, 0, 0, 0] [1, 10, 8, 8]
  // CHECK:     [[FQ_1:%.*]] = IE.FakeQuantize([[SLICE]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_2]])

  // CHECK: return [[FQ_1]] : tensor<1x10x8x8xf16>
}


// -----

// CHECK-LABEL: @NotAlignSliceWithPerTensorFq
func.func @NotAlignSliceWithPerTensorFq(%arg0: tensor<1x16x8x8xf16>) -> tensor<1x10x8x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e+00> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x10x1x1xf16> = dense<[[[[2.88798332]], [[13.5615578]],
        [[12.3310165]], [[3.609375]], [[2.681180e+00]], [[3.0952239]], [[3.04886699]], [[1.556250e+01]],
        [[2.70967746]], [[8.63315773]]]]> : tensor<1x10x1x1xf16>
  %cst_2 = const.Declare tensor<1x10x1x1xf16> = dense<[[[[4.88798332]], [[14.5615578]],
        [[13.3310165]], [[4.609375]], [[4.681180e+00]], [[4.0952239]], [[4.04886699]], [[2.556250e+01]],
        [[4.70967746]], [[9.63315773]]]]> : tensor<1x10x1x1xf16>

  %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %1 = IE.Slice %0 [0, 0, 0, 0] [1, 10, 8, 8] : tensor<1x16x8x8xf16> to tensor<1x10x8x8xf16>
  %2 = IE.FakeQuantize(%1, %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x10x8x8xf16>, tensor<1x10x1x1xf16>, tensor<1x10x1x1xf16>, tensor<1x10x1x1xf16>, tensor<1x10x1x1xf16> -> tensor<1x10x8x8xf16>

  return %2 : tensor<1x10x8x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x10x1x1xf16>
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<1x10x1x1xf16>
  // CHECK:     [[FQ_0:%.*]] = IE.FakeQuantize(%arg0, [[CST_0]], [[CST]], [[CST_0]], [[CST]])
  // CHECK:     [[SLICE:%.*]] = IE.Slice [[FQ_0]] [0, 0, 0, 0] [1, 10, 8, 8]
  // CHECK:     [[FQ_1:%.*]] = IE.FakeQuantize([[SLICE]], [[CST_1]], [[CST_2]], [[CST_1]], [[CST_2]])

  // CHECK: return [[FQ_1]] : tensor<1x10x8x8xf16>

}

// -----

// CHECK-LABEL: @AlignConcatWithSlice
func.func @AlignConcatWithSlice(%arg0: tensor<32x8x8xf16>, %arg1: tensor<16x1x8xf16>) -> tensor<1x16x9x8xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e+00> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<8.068850e+00> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<5.558590e+00> : tensor<1x1x1x1xf16>
  %0 = IE.Reshape(%arg0) { shape_value = [1, 32, 8, 8] } : tensor<32x8x8xf16> -> tensor<1x32x8x8xf16>
  %1 = IE.Reshape(%arg1) { shape_value = [1, 16, 1, 8] } : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  %2 = IE.FakeQuantize(%0, %cst_1, %cst, %cst_1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x8x8xf16>
  %3 = IE.Slice %2 [0, 0, 0, 0] [1, 16, 8, 8] : tensor<1x32x8x8xf16> to tensor<1x16x8x8xf16>
  %4 = IE.FakeQuantize(%3, %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
  %5 = IE.FakeQuantize(%1, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>

  %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>

  return %6 : tensor<1x16x9x8xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<8.0703125> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<8.070310e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>

  // CHECK: [[RESHAPE:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 32, 8, 8]} : tensor<32x8x8xf16> -> tensor<1x32x8x8xf16>
  // CHECK: [[RESHAPE_0:%.*]] = IE.Reshape(%arg1) {shape_value = [1, 16, 1, 8]} : tensor<16x1x8xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[FQ:%.*]] = IE.FakeQuantize([[RESHAPE]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x8x8xf16>
  // CHECK: [[CLAMP:%.*]] =  IE.Clamp([[FQ]]) {max = 7.55859375 : f64, min = 0.000000e+00 : f64} : tensor<1x32x8x8xf16> -> tensor<1x32x8x8xf16>
  // CHECK: [[SLICE:%.*]] = IE.Slice [[CLAMP]] [0, 0, 0, 0] [1, 16, 8, 8] : tensor<1x32x8x8xf16> to tensor<1x16x8x8xf16>

  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[SLICE]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>

  // CHECK: [[CLAMP_0:%.*]] = IE.Clamp([[FQ_0]]) {max = 5.55859375 : f64, min = 0.000000e+00 : f64} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>
  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[RESHAPE_0]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x8xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[CLAMP_0]], [[FQ_0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 8, 0]]} : tensor<1x16x8x8xf16>, tensor<1x16x1x8xf16> -> tensor<1x16x9x8xf16>

  // CHECK: return [[CONCAT]] : tensor<1x16x9x8xf16>
}

// -----

// CHECK-LABEL: @AlignAllFQ
func.func @AlignAllFQ(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<1x2x3x4xf16>, %arg2: tensor<1x2x3x4xf16>) -> (tensor<1x1x3x2xf16>, tensor<1x2x3x4xf16>) {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<-1.5000e+00> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<4.0000e+00> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-2.4000e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.5000e+00> : tensor<1x1x1x1xf16>
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-1.7500e+00> : tensor<1x1x1x1xf16>
  %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<3.7500e+00> : tensor<1x1x1x1xf16>
  %cst_5 = const.Declare tensor<1x1x1x1xf16> = dense<-6.2500e+00> : tensor<1x1x1x1xf16>
  %cst_6 = const.Declare tensor<1x1x1x1xf16> = dense<5.3000e+00> : tensor<1x1x1x1xf16>
  %cst_7 = const.Declare tensor<1x1x1x1xf16> = dense<-2.4500e+00> : tensor<1x1x1x1xf16>
  %cst_8 = const.Declare tensor<1x1x1x1xf16> = dense<6.7000e+00> : tensor<1x1x1x1xf16>
  %0 = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  %1:2 = IE.Split(%0) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x3x4xf16> -> tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>
  %2 = IE.MaxPool(%1#0) {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2] } : tensor<1x1x3x4xf16> -> tensor<1x1x3x2xf16>
  %3 = IE.FakeQuantize(%2, %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x3x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x2xf16>
  %4 = IE.FakeQuantize(%arg1, %cst_3, %cst_4, %cst_3, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  %5 = IE.FakeQuantize(%arg2, %cst_5, %cst_6, %cst_5, %cst_6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
  %7 = IE.ReduceMax(%6) {axes_value = [1], keep_dims} : tensor<1x4x3x4xf16> -> tensor<1x1x3x4xf16>
  %8 = IE.Concat(%7, %1#1) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16> -> tensor<1x2x3x4xf16>
  %9 = IE.FakeQuantize(%8, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>

  return %3, %9 : tensor<1x1x3x2xf16>, tensor<1x2x3x4xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-6.25101137> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<5.29977036> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK: [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CLAMP:%.*]] = IE.Clamp([[FQ]]) {max = 4.000000e+00 : f64, min = -1.500000e+00 : f64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[SPLIT:%.*]]:2 = IE.Split([[CLAMP]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x3x4xf16> -> tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>
  // CHECK: [[MAXPOOL:%.*]] = IE.MaxPool([[SPLIT]]#0) {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2]} : tensor<1x1x3x4xf16> -> tensor<1x1x3x2xf16>
  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[MAXPOOL]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x3x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x2xf16>
  // CHECK: [[CLAMP_0:%.*]] = IE.Clamp([[FQ_0]]) {max = 2.500000e+00 : f64, min = -2.400390625 : f64} : tensor<1x1x3x2xf16> -> tensor<1x1x3x2xf16>
  // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize(%arg1, [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CLAMP_1:%.*]] = IE.Clamp([[FQ_1]]) {max = 3.750000e+00 : f64, min = -1.750000e+00 : f64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[FQ_2:%.*]] = IE.FakeQuantize(%arg2, [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CLAMP_2:%.*]] = IE.Clamp([[FQ_2]]) {max = 5.30078125 : f64, min = -6.250000e+00 : f64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[CLAMP_1]], [[CLAMP_2]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 2, 0, 0]]} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
  // CHECK: [[REDUCEMAX:%.*]] = IE.ReduceMax([[CONCAT]]) {axes_value = [1], keep_dims} : tensor<1x4x3x4xf16> -> tensor<1x1x3x4xf16>
  // CHECK: [[CONCAT_0:%.*]] = IE.Concat([[REDUCEMAX]], [[SPLIT]]#1) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[FQ_3:%.*]] = IE.FakeQuantize([[CONCAT_0]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CLAMP_3:%.*]] = IE.Clamp([[FQ_3]]) {max = 4.000000e+00 : f64, min = -2.400390625 : f64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>

  // CHECK: return [[CLAMP_0]], [[CLAMP_3]]
}
