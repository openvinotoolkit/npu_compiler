//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-reduce-to-pooling %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertReduceMeanToPooling4D
func.func @ConvertReduceMeanToPooling4D(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

  // CHECK-NOT:   ReduceMean
  // CHECK:       [[RESHAPE1:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 1, 10, 5]}
  // CHECK-SAME:      : tensor<1x1x1x50xf16> -> tensor<1x1x10x5xf16>
  // CHECK:       [[POOL:%.*]] = IE.AvgPool([[RESHAPE1]])
  // CHECK-SAME:      {exclude_pads, kernel_size = [10, 5], pads_begin = [0, 0], pads_end = [0, 0]
  // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
  // CHECK-SAME:      : tensor<1x1x10x5xf16> -> tensor<1x1x1x1xf16>
}

func.func @ConvertReduceMeanToPoolingWithLargeSize(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
  return %0 : tensor<1x32x112x1xf16>

  // CHECK:       IE.AvgPool(%arg0) {
  // CHECK:           exclude_pads, kernel_size = [1, 112], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
  // CHECK-NOT:   ReduceMean
}

// CHECK-LABEL: @ConvertReduceMeanToPoolingReduceDimOneKeepDim
func.func @ConvertReduceMeanToPoolingReduceDimOneKeepDim(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x50xf16> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [0], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x50xf16>
  return %0 : tensor<1x1x1x50xf16>

  // CHECK-NOT:   ReduceMean
}

// CHECK-LABEL: @ConvertReduceMeanToPoolingReduceDimOne
func.func @ConvertReduceMeanToPoolingReduceDimOne(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x50xf16> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [0]}: tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  return %0 : tensor<1x1x50xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 1, 50]} : tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  // CHECK-NOT:   ReduceMean
}

// CHECK-LABEL: @ConvertReduceMaxToPooling4D
func.func @ConvertReduceMaxToPooling4D(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

  // CHECK-NOT:   ReduceMax
  // CHECK:       [[RESHAPE1:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 1, 10, 5]}
  // CHECK-SAME:      : tensor<1x1x1x50xf16> -> tensor<1x1x10x5xf16>
  // CHECK:       [[POOL:%.*]] = IE.MaxPool([[RESHAPE1]])
  // CHECK-SAME:      {kernel_size = [10, 5], pads_begin = [0, 0], pads_end = [0, 0]
  // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
  // CHECK-SAME:      : tensor<1x1x10x5xf16> -> tensor<1x1x1x1xf16>
}

// CHECK-LABEL: @ConvertReduceMaxToPooling3D
func.func @ConvertReduceMaxToPooling3D(%arg0: tensor<256x7x7xf16>) -> tensor<256x1x7xf16> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [1], keep_dims} : tensor<256x7x7xf16> -> tensor<256x1x7xf16>
  return %0 : tensor<256x1x7xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 256, 7, 7]} : tensor<256x7x7xf16> -> tensor<1x256x7x7xf16>
  // CHECK-NOT:   ReduceMax
  // CHECK:       %1 = IE.MaxPool(%0) {kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x256x7x7xf16> -> tensor<1x256x1x7xf16>
  // CHECK:       %2 = IE.Reshape(%1) {shape_value = [256, 1, 7]} : tensor<1x256x1x7xf16> -> tensor<256x1x7xf16>
}

// CHECK-LABEL: @ConvertReduceMaxToPoolingReduceDimOneKeepDim
func.func @ConvertReduceMaxToPoolingReduceDimOneKeepDim(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x50xf16> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [0], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x50xf16>
  return %0 : tensor<1x1x1x50xf16>

  // CHECK-NOT:   ReduceMax
}

// CHECK-LABEL: @ConvertReduceMaxToPoolingReduceDimOne
func.func @ConvertReduceMaxToPoolingReduceDimOne(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x50xf16> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [0]}: tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  return %0 : tensor<1x1x50xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 1, 50]} : tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  // CHECK-NOT:   ReduceMax
}

// CHECK-LABEL: @ConvertReduceSumToPooling4D
func.func @ConvertReduceSumToPooling4D(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

  // CHECK-NOT:   ReduceSum
  // CHECK:       [[RESHAPE1:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 1, 10, 5]}
  // CHECK-SAME:      : tensor<1x1x1x50xf16> -> tensor<1x1x10x5xf16>
  // CHECK:       [[POOL:%.*]] = IE.AvgPool([[RESHAPE1]])
  // CHECK-SAME:      {exclude_pads, kernel_size = [10, 5], pads_begin = [0, 0], pads_end = [0, 0]
  // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
  // CHECK-SAME:      : tensor<1x1x10x5xf16> -> tensor<1x1x1x1xf16>
  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1xf16> = dense<5.000000e+01> : tensor<1xf16>
  // CHECK:       [[MUL:%.*]] = IE.Multiply([[POOL]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<1x1x1x1xf16>, tensor<1xf16> -> tensor<1x1x1x1xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPooling3D
func.func @ConvertReduceSumToPooling3D(%arg0: tensor<256x7x7xf16>) -> tensor<256x1x7xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<256x7x7xf16> -> tensor<256x1x7xf16>
  return %0 : tensor<256x1x7xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 256, 7, 7]} : tensor<256x7x7xf16> -> tensor<1x256x7x7xf16>
  // CHECK-NOT:   ReduceSum
  // CHECK:       %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x256x7x7xf16> -> tensor<1x256x1x7xf16>
  // CHECK-DAG:   %cst = const.Declare tensor<1xf16> = dense<7.000000e+00> : tensor<1xf16>
  // CHECK:       %2 = IE.Multiply(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x7xf16>, tensor<1xf16> -> tensor<1x256x1x7xf16>
  // CHECK:       %3 = IE.Reshape(%2) {shape_value = [256, 1, 7]} : tensor<1x256x1x7xf16> -> tensor<256x1x7xf16>
}

// CHECK-LABEL: @ConvertBatchedReduceSumToPooling
// CHECK-SAME: [[INPUT:%.+]]: tensor<8x2x1x256xf16>
func.func @ConvertBatchedReduceSumToPooling(%arg0: tensor<8x2x1x256xf16>) -> tensor<8x1x256xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1]} : tensor<8x2x1x256xf16> -> tensor<8x1x256xf16>
  return %0 : tensor<8x1x256xf16>

  // CHECK-NOT:   ReduceSum
  // CHECK:    [[RESHAPED_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [8, 2, 16, 16]} : tensor<8x2x1x256xf16> -> tensor<8x2x16x16xf16>
  // CHECK:    [[TRANSPOSED_IN:%.+]] = IE.Transpose([[RESHAPED_IN]]) {order_value = #NWCH} : tensor<8x2x16x16xf16> -> tensor<8x16x2x16xf16>
  // CHECK:    [[RESHAPED_IN2:%.+]] = IE.Reshape([[TRANSPOSED_IN]]) {shape_value = [1, 128, 2, 16]} : tensor<8x16x2x16xf16> -> tensor<1x128x2x16xf16>
  // CHECK:    [[AVG_POOL:%.+]] = IE.AvgPool([[RESHAPED_IN2]]) {exclude_pads, kernel_size = [2, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x2x16xf16> -> tensor<1x128x1x16xf16>
  // CHECK:    [[CST:%.+]] = const.Declare tensor<1xf16> = dense<2.000000e+00> : tensor<1xf16>
  // CHECK:    [[MUL:%.+]] = IE.Multiply([[AVG_POOL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x16xf16>, tensor<1xf16> -> tensor<1x128x1x16xf16>
  // CHECK:    [[RESHAPED_OUT2:%.+]] = IE.Reshape([[MUL]]) {shape_value = [8, 16, 1, 16]} : tensor<1x128x1x16xf16> -> tensor<8x16x1x16xf16>
  // CHECK:    [[TRANSPOSED_OUT:%.+]] = IE.Transpose([[RESHAPED_OUT2]]) {order_value = #NHWC} : tensor<8x16x1x16xf16> -> tensor<8x1x16x16xf16>
  // CHECK:    [[RESHAPED_OUT:%.+]] = IE.Reshape([[TRANSPOSED_OUT]]) {shape_value = [8, 1, 256]} : tensor<8x1x16x16xf16> -> tensor<8x1x256xf16>
  // CHECK:    return [[RESHAPED_OUT]] : tensor<8x1x256xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPoolingReduceDimOneKeepDim
func.func @ConvertReduceSumToPoolingReduceDimOneKeepDim(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x50xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x50xf16>
  return %0 : tensor<1x1x1x50xf16>

  // CHECK-NOT:   ReduceSum
}

// CHECK-LABEL: @ConvertReduceSumToPoolingReduceDimOne
func.func @ConvertReduceSumToPoolingReduceDimOne(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x50xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0]}: tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  return %0 : tensor<1x1x50xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 1, 50]} : tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  // CHECK-NOT:   ReduceSum
}

// CHECK-LABEL: @ConvertReduceSumToPoolingAvoidingExpand
func.func @ConvertReduceSumToPoolingAvoidingExpand(%arg0: tensor<1x12x368x480xf16>) -> tensor<1x1x368x480xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x12x368x480xf16> -> tensor<1x1x368x480xf16>
  return %0 : tensor<1x1x368x480xf16>

  // CHECK-NOT:   ReduceSum
  // CHECK:       [[RESHAPE0:%.*]] = IE.Reshape({{[^:]+}}) {shape_value = [1, 12, 11040, 16]} : tensor<1x12x368x480xf16> -> tensor<1x12x11040x16xf16>
  // CHECK:       [[TRANSPOSE0:%.*]] = IE.Transpose([[RESHAPE0]]) {order_value = #NWCH} : tensor<1x12x11040x16xf16> -> tensor<1x16x12x11040xf16>
  // CHECK:       [[AVGPOOL0:%.*]] = IE.AvgPool([[TRANSPOSE0]]) {exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0],
  // CHECK-SAME:    rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x12x11040xf16> -> tensor<1x16x1x11040xf16>
  // CHECK:       [[MULTIPLY0:%.*]] = IE.Multiply([[AVGPOOL0]], {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x11040xf16>, tensor<1xf16> -> tensor<1x16x1x11040xf16>
  // CHECK:       [[TRANSPOSE1:%.*]] = IE.Transpose([[MULTIPLY0]]) {order_value = #NHWC} : tensor<1x16x1x11040xf16> -> tensor<1x1x11040x16xf16>
  // CHECK:       [[RESHAPE1:%.*]] = IE.Reshape([[TRANSPOSE1]]) {shape_value = [1, 1, 368, 480]} : tensor<1x1x11040x16xf16> -> tensor<1x1x368x480xf16>
  // CHECK:       return [[RESHAPE1]] : tensor<1x1x368x480xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPoolingAvoidingExpand2
func.func @ConvertReduceSumToPoolingAvoidingExpand2(%arg0: tensor<1x12x44x44xf16>) -> tensor<1x1x44x44xf16> {
  %1 = IE.ReduceMax(%arg0) {axes_value = [1], keep_dims} : tensor<1x12x44x44xf16> -> tensor<1x1x44x44xf16>
  return %1 : tensor<1x1x44x44xf16>

  // CHECK-NOT:   ReduceMax
  // CHECK:       [[RESHAPE0:%.*]] = IE.Reshape({{[^:]+}}) {shape_value = [1, 12, 121, 16]} : tensor<1x12x44x44xf16> -> tensor<1x12x121x16xf16>
  // CHECK:       [[TRANSPOSE0:%.*]] = IE.Transpose([[RESHAPE0]]) {order_value = #NWCH} : tensor<1x12x121x16xf16> -> tensor<1x16x12x121xf16>
  // CHECK:       [[MAXPOOL0:%.*]] = IE.MaxPool([[TRANSPOSE0]]) {kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0],
  // CHECK-SAME:    rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x12x121xf16> -> tensor<1x16x1x121xf16>
  // CHECK:       [[TRANSPOSE1:%.*]] = IE.Transpose([[MAXPOOL0]]) {order_value = #NHWC} : tensor<1x16x1x121xf16> -> tensor<1x1x121x16xf16>
  // CHECK:       [[RESHAPE1:%.*]] = IE.Reshape([[TRANSPOSE1]]) {shape_value = [1, 1, 44, 44]} : tensor<1x1x121x16xf16> -> tensor<1x1x44x44xf16>
  // CHECK:       return [[RESHAPE1]] : tensor<1x1x44x44xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPoolingNegativeAxis
func.func @ConvertReduceSumToPoolingNegativeAxis(%arg0: tensor<1x10x10x40x40xf16>) -> tensor<1x10x10xf16> {
  %1 = IE.ReduceSum(%arg0) {axes_value = [-1, -2]}: tensor<1x10x10x40x40xf16> -> tensor<1x10x10xf16>
  return %1 : tensor<1x10x10xf16>

  // CHECK:       [[RESHAPE_0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 100, 40, 40]} : tensor<1x10x10x40x40xf16> -> tensor<1x100x40x40xf16>
  // CHECK-NOT:   ReduceSum
  // CHECK:       [[AVGPOOL_0:%.*]] = IE.AvgPool([[RESHAPE_0]]) {exclude_pads, kernel_size = [40, 40], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x100x40x40xf16> -> tensor<1x100x1x1xf16>
  // CHECK-DAG:       [[CST_0:%.*]] = const.Declare tensor<1xf16> = dense<1.600000e+03> : tensor<1xf16>
  // CHECK:       [[MULTIPLY_0:%.*]] = IE.Multiply([[AVGPOOL_0]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x100x1x1xf16>, tensor<1xf16> -> tensor<1x100x1x1xf16>
  // CHECK:       [[RESHAPE_1:%.*]] = IE.Reshape([[MULTIPLY_0]]) {shape_value = [1, 10, 10]} : tensor<1x100x1x1xf16> -> tensor<1x10x10xf16>
}

// CHECK-LABEL: @ConvertReduceMinToPoolingKernelSize
func.func @ConvertReduceMinToPoolingKernelSize(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
  return %0 : tensor<1x32x112x1xf16>

  // CHECK:       [[NEGATIVE_0:%.*]] = IE.Negative(%arg0) : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
  // CHECK:       [[MAXPOOL:%.*]] = IE.MaxPool([[NEGATIVE_0]]) {
  // CHECK-DAG:       kernel_size = [1, 112], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
  // CHECK:       [[NEGATIVE_1:%.*]] = IE.Negative([[MAXPOOL]]) : tensor<1x32x112x1xf16> -> tensor<1x32x112x1xf16>
}

// CHECK-LABEL: @ConvertReduceMinToPoolingKernelSizeWithMultiAxis
func.func @ConvertReduceMinToPoolingKernelSizeWithMultiAxis(%arg0: tensor<1x2x4x9xf16>) -> tensor<1x1x1x1xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2, 3], keep_dims} : tensor<1x2x4x9xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

  // CHECK:       [[RESHAPE:%.*]] = IE.Reshape(%arg0) {
  // CHECK-DAG:       shape_value = [1, 1, 9, 8]} : tensor<1x2x4x9xf16> -> tensor<1x1x9x8xf16>
  // CHECK:       [[NEGATIVE_0:%.*]] = IE.Negative([[RESHAPE]]) : tensor<1x1x9x8xf16> -> tensor<1x1x9x8xf16>
  // CHECK:       [[MAXPOOL:%.*]] = IE.MaxPool([[NEGATIVE_0]]) {
  // CHECK-DAG:       kernel_size = [9, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x9x8xf16> -> tensor<1x1x1x1xf16>
  // CHECK:       [[NEGATIVE_1:%.*]] = IE.Negative([[MAXPOOL]]) : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
}

// CHECK-LABEL: @DoNotConvertReduceMinToPoolingKernelSizeWithMultiAxis
func.func @DoNotConvertReduceMinToPoolingKernelSizeWithMultiAxis(%arg0: tensor<1x8x4x76xf16>) -> tensor<1x1x1x1xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2, 3], keep_dims} : tensor<1x8x4x76xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

  // CHECK:       [[OUTPUT:%.*]] = IE.ReduceMin(%arg0) {
  // CHECK-DAG:       axes_value = [0, 1, 2, 3], keep_dims} : tensor<1x8x4x76xf16> -> tensor<1x1x1x1xf16>
  // CHECK:       return [[OUTPUT:%.*]] : tensor<1x1x1x1xf16>
}

// CHECK-LABEL: @DoNotConvertReduceMinToPoolingF32
func.func @DoNotConvertReduceMinToPoolingF32(%arg0: tensor<1x256x7x7xf32>) -> tensor<1x256x1x1xf32> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [2, 3], keep_dims} : tensor<1x256x7x7xf32> -> tensor<1x256x1x1xf32>
  return %0 : tensor<1x256x1x1xf32>

  // CHECK:       [[OUTPUT:%.*]] = IE.ReduceMin(%arg0) {
  // CHECK-DAG:       axes_value = [2, 3], keep_dims} : tensor<1x256x7x7xf32> -> tensor<1x256x1x1xf32>
  // CHECK:       return [[OUTPUT:%.*]] : tensor<1x256x1x1xf32>
}

// CHECK-LABEL: @ConvertReduceMaxwithLargeChannelDim
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x42840x12xf16>)
func.func @ConvertReduceMaxwithLargeChannelDim(%arg0: tensor<1x42840x12xf16>) -> tensor<1x42840x1xf16> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [2], keep_dims} : tensor<1x42840x12xf16> -> tensor<1x42840x1xf16>
  return %0 : tensor<1x42840x1xf16>

  // CHECK:       [[RESHAPE1:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [1, 42840, 12, 1]} : tensor<1x42840x12xf16> -> tensor<1x42840x12x1xf16>
  // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[RESHAPE1]]) {
  // CHECK-DAG:       kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x42840x12x1xf16> -> tensor<1x42840x1x1xf16>
  // CHECK:       [[RESHAPE2:%.+]] = IE.Reshape([[MAXPOOL]]) {shape_value = [1, 42840, 1]} : tensor<1x42840x1x1xf16> -> tensor<1x42840x1xf16>
  // CHECK:       return [[RESHAPE2:%.*]] : tensor<1x42840x1xf16>
}
