//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --convert-reduce-to-pooling %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ConvertReduceMeanToPooling4D
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x1x50xf16>
func.func @ConvertReduceMeanToPooling4D(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

  // CHECK-NOT:   ReduceMean
  // CHECK:       [[RESHAPE1:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 10, 5]}
  // CHECK-SAME:      : tensor<1x1x1x50xf16> -> tensor<1x1x10x5xf16>
  // CHECK:       [[POOL:%.+]] = IE.AvgPool([[RESHAPE1]])
  // CHECK-SAME:      {exclude_pads, kernel_size = [10, 5], pads_begin = [0, 0], pads_end = [0, 0]
  // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
  // CHECK-SAME:      : tensor<1x1x10x5xf16> -> tensor<1x1x1x1xf16>
  // CHECK:       return  [[POOL]] : tensor<1x1x1x1xf16>
}

// CHECK-LABEL: @ConvertReduceMeanToPoolingWithLargeSize
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x32x112x112xf16>
func.func @ConvertReduceMeanToPoolingWithLargeSize(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
  return %0 : tensor<1x32x112x1xf16>

  // CHECK:       IE.AvgPool([[INPUT]]) {
  // CHECK:           exclude_pads, kernel_size = [1, 112], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
  // CHECK-NOT:   ReduceMean
}

// CHECK-LABEL: @ConvertReduceMeanToPoolingReduceDimOneKeepDim
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x1x50xf16>
func.func @ConvertReduceMeanToPoolingReduceDimOneKeepDim(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x50xf16> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [0], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x50xf16>
  return %0 : tensor<1x1x1x50xf16>

  // CHECK-NOT:   ReduceMean
}

// CHECK-LABEL: @ConvertReduceMeanToPoolingReduceDimOne
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x1x50xf16>
func.func @ConvertReduceMeanToPoolingReduceDimOne(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x50xf16> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [0]}: tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  return %0 : tensor<1x1x50xf16>

  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 50]} : tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  // CHECK-NOT:   ReduceMean
}

// CHECK-LABEL: @ConvertReduceMaxToPooling4D
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x1x50xf16>
func.func @ConvertReduceMaxToPooling4D(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

  // CHECK-NOT:   ReduceMax
  // CHECK:       [[RESHAPE1:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 10, 5]}
  // CHECK-SAME:      : tensor<1x1x1x50xf16> -> tensor<1x1x10x5xf16>
  // CHECK:       [[POOL:%.+]] = IE.MaxPool([[RESHAPE1]])
  // CHECK-SAME:      {kernel_size = [10, 5], pads_begin = [0, 0], pads_end = [0, 0]
  // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
  // CHECK-SAME:      : tensor<1x1x10x5xf16> -> tensor<1x1x1x1xf16>
  // CHECK:       return  [[POOL]] : tensor<1x1x1x1xf16>
}

// CHECK-LABEL: @ConvertReduceMaxToPooling3D
// CHECK-SAME: [[INPUT:%.+]]: tensor<256x7x7xf16>
func.func @ConvertReduceMaxToPooling3D(%arg0: tensor<256x7x7xf16>) -> tensor<256x1x7xf16> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [1], keep_dims} : tensor<256x7x7xf16> -> tensor<256x1x7xf16>
  return %0 : tensor<256x1x7xf16>

  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 256, 7, 7]} : tensor<256x7x7xf16> -> tensor<1x256x7x7xf16>
  // CHECK-NOT:   ReduceMax
  // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[RESHAPE]]) {kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x256x7x7xf16> -> tensor<1x256x1x7xf16>
  // CHECK:       [[RESHAPE_OUT:%.+]] = IE.Reshape([[MAXPOOL]]) {shape_value = [256, 1, 7]} : tensor<1x256x1x7xf16> -> tensor<256x1x7xf16>
  // CHECK:       return  [[RESHAPE_OUT]] : tensor<256x1x7xf16>
}

// CHECK-LABEL: @ConvertReduceMaxToPoolingReduceDimOneKeepDim
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x1x50xf16>
func.func @ConvertReduceMaxToPoolingReduceDimOneKeepDim(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x50xf16> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [0], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x50xf16>
  return %0 : tensor<1x1x1x50xf16>

  // CHECK-NOT:   ReduceMax
}

// CHECK-LABEL: @ConvertReduceMaxToPoolingReduceDimOne
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x1x50xf16>
func.func @ConvertReduceMaxToPoolingReduceDimOne(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x50xf16> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [0]}: tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  return %0 : tensor<1x1x50xf16>

  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 50]} : tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  // CHECK-NOT:   ReduceMax
}

// CHECK-LABEL: @ConvertReduceSumToPooling4D
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x1x50xf16>
func.func @ConvertReduceSumToPooling4D(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

  // CHECK-NOT:   ReduceSum
  // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1xf16> = dense<5.000000e+01> : tensor<1xf16>
  // CHECK:       [[RESHAPE1:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 10, 5]}
  // CHECK-SAME:      : tensor<1x1x1x50xf16> -> tensor<1x1x10x5xf16>
  // CHECK:       [[POOL:%.+]] = IE.AvgPool([[RESHAPE1]])
  // CHECK-SAME:      {exclude_pads, kernel_size = [10, 5], pads_begin = [0, 0], pads_end = [0, 0]
  // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
  // CHECK-SAME:      : tensor<1x1x10x5xf16> -> tensor<1x1x1x1xf16>
  // CHECK:       [[MUL:%.+]] = IE.Multiply([[POOL]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<1x1x1x1xf16>, tensor<1xf16> -> tensor<1x1x1x1xf16>
  // CHECK:       return [[MUL]] : tensor<1x1x1x1xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPooling3D
// CHECK-SAME: [[INPUT:%.+]]: tensor<256x7x7xf16>
func.func @ConvertReduceSumToPooling3D(%arg0: tensor<256x7x7xf16>) -> tensor<256x1x7xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<256x7x7xf16> -> tensor<256x1x7xf16>
  return %0 : tensor<256x1x7xf16>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1xf16> = dense<7.000000e+00> : tensor<1xf16>
  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 256, 7, 7]} : tensor<256x7x7xf16> -> tensor<1x256x7x7xf16>
  // CHECK-NOT:   ReduceSum
  // CHECK:       [[AVGPOOL:%.+]] = IE.AvgPool([[RESHAPE]]) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x256x7x7xf16> -> tensor<1x256x1x7xf16>
  // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[AVGPOOL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x7xf16>, tensor<1xf16> -> tensor<1x256x1x7xf16>
  // CHECK:       [[RESHAPE_OUT:%.+]] = IE.Reshape([[MULTIPLY]]) {shape_value = [256, 1, 7]} : tensor<1x256x1x7xf16> -> tensor<256x1x7xf16>
  // CHECK:       return [[RESHAPE_OUT]] : tensor<256x1x7xf16>
}

// CHECK-LABEL: @ConvertBatchedReduceSumToPooling
// CHECK-SAME: [[INPUT:%.+]]: tensor<8x2x1x256xf16>
func.func @ConvertBatchedReduceSumToPooling(%arg0: tensor<8x2x1x256xf16>) -> tensor<8x1x256xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1]} : tensor<8x2x1x256xf16> -> tensor<8x1x256xf16>
  return %0 : tensor<8x1x256xf16>

  // CHECK-NOT:   ReduceSum
  // CHECK:    [[CST:%.+]] = const.Declare tensor<1xf16> = dense<2.000000e+00> : tensor<1xf16>
  // CHECK:    [[RESHAPED_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [8, 2, 16, 16]} : tensor<8x2x1x256xf16> -> tensor<8x2x16x16xf16>
  // CHECK:    [[TRANSPOSED_IN:%.+]] = IE.Transpose([[RESHAPED_IN]]) {order_value = #NWCH} : tensor<8x2x16x16xf16> -> tensor<8x16x2x16xf16>
  // CHECK:    [[RESHAPED_IN2:%.+]] = IE.Reshape([[TRANSPOSED_IN]]) {shape_value = [1, 128, 2, 16]} : tensor<8x16x2x16xf16> -> tensor<1x128x2x16xf16>
  // CHECK:    [[AVG_POOL:%.+]] = IE.AvgPool([[RESHAPED_IN2]]) {exclude_pads, kernel_size = [2, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x2x16xf16> -> tensor<1x128x1x16xf16>
  // CHECK:    [[MUL:%.+]] = IE.Multiply([[AVG_POOL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x16xf16>, tensor<1xf16> -> tensor<1x128x1x16xf16>
  // CHECK:    [[RESHAPED_OUT2:%.+]] = IE.Reshape([[MUL]]) {shape_value = [8, 16, 1, 16]} : tensor<1x128x1x16xf16> -> tensor<8x16x1x16xf16>
  // CHECK:    [[TRANSPOSED_OUT:%.+]] = IE.Transpose([[RESHAPED_OUT2]]) {order_value = #NHWC} : tensor<8x16x1x16xf16> -> tensor<8x1x16x16xf16>
  // CHECK:    [[RESHAPED_OUT:%.+]] = IE.Reshape([[TRANSPOSED_OUT]]) {shape_value = [8, 1, 256]} : tensor<8x1x16x16xf16> -> tensor<8x1x256xf16>
  // CHECK:    return [[RESHAPED_OUT]] : tensor<8x1x256xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPoolingReduceDimOneKeepDim
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x1x50xf16>
func.func @ConvertReduceSumToPoolingReduceDimOneKeepDim(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x50xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x50xf16>
  return %0 : tensor<1x1x1x50xf16>

  // CHECK-NOT:   ReduceSum
}

// CHECK-LABEL: @ConvertReduceSumToPoolingReduceDimOne
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x1x50xf16>
func.func @ConvertReduceSumToPoolingReduceDimOne(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x50xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0]}: tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  return %0 : tensor<1x1x50xf16>

  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 50]} : tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  // CHECK-NOT:   ReduceSum
}

// CHECK-LABEL: @ConvertReduceSumToPoolingAvoidingExpand
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x12x368x480xf16>
func.func @ConvertReduceSumToPoolingAvoidingExpand(%arg0: tensor<1x12x368x480xf16>) -> tensor<1x1x368x480xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x12x368x480xf16> -> tensor<1x1x368x480xf16>
  return %0 : tensor<1x1x368x480xf16>

  // CHECK-NOT:   ReduceSum
  // CHECK:       [[CST:%.+]] = const.Declare tensor<1xf16> = dense<1.200000e+01> : tensor<1xf16>
  // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 12, 11040, 16]} : tensor<1x12x368x480xf16> -> tensor<1x12x11040x16xf16>
  // CHECK:       [[TRANSPOSE0:%.+]] = IE.Transpose([[RESHAPE0]]) {order_value = #NWCH} : tensor<1x12x11040x16xf16> -> tensor<1x16x12x11040xf16>
  // CHECK:       [[AVGPOOL0:%.+]] = IE.AvgPool([[TRANSPOSE0]]) {exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0],
  // CHECK-SAME:    rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x12x11040xf16> -> tensor<1x16x1x11040xf16>
  // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[AVGPOOL0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x11040xf16>, tensor<1xf16> -> tensor<1x16x1x11040xf16>
  // CHECK:       [[TRANSPOSE1:%.+]] = IE.Transpose([[MULTIPLY0]]) {order_value = #NHWC} : tensor<1x16x1x11040xf16> -> tensor<1x1x11040x16xf16>
  // CHECK:       [[RESHAPE1:%.+]] = IE.Reshape([[TRANSPOSE1]]) {shape_value = [1, 1, 368, 480]} : tensor<1x1x11040x16xf16> -> tensor<1x1x368x480xf16>
  // CHECK:       return [[RESHAPE1]] : tensor<1x1x368x480xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPoolingAvoidingExpand2
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x12x44x44xf16>
func.func @ConvertReduceSumToPoolingAvoidingExpand2(%arg0: tensor<1x12x44x44xf16>) -> tensor<1x1x44x44xf16> {
  %1 = IE.ReduceMax(%arg0) {axes_value = [1], keep_dims} : tensor<1x12x44x44xf16> -> tensor<1x1x44x44xf16>
  return %1 : tensor<1x1x44x44xf16>

  // CHECK-NOT:   ReduceMax
  // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 12, 121, 16]} : tensor<1x12x44x44xf16> -> tensor<1x12x121x16xf16>
  // CHECK:       [[TRANSPOSE0:%.+]] = IE.Transpose([[RESHAPE0]]) {order_value = #NWCH} : tensor<1x12x121x16xf16> -> tensor<1x16x12x121xf16>
  // CHECK:       [[MAXPOOL0:%.+]] = IE.MaxPool([[TRANSPOSE0]]) {kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0],
  // CHECK-SAME:    rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x12x121xf16> -> tensor<1x16x1x121xf16>
  // CHECK:       [[TRANSPOSE1:%.+]] = IE.Transpose([[MAXPOOL0]]) {order_value = #NHWC} : tensor<1x16x1x121xf16> -> tensor<1x1x121x16xf16>
  // CHECK:       [[RESHAPE1:%.+]] = IE.Reshape([[TRANSPOSE1]]) {shape_value = [1, 1, 44, 44]} : tensor<1x1x121x16xf16> -> tensor<1x1x44x44xf16>
  // CHECK:       return [[RESHAPE1]] : tensor<1x1x44x44xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPoolingTwoAxis
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x10x10x40x40xf16>
func.func @ConvertReduceSumToPoolingTwoAxis(%arg0: tensor<1x10x10x40x40xf16>) -> tensor<1x10x10xf16> {
  %1 = IE.ReduceSum(%arg0) {axes_value = [3, 4]}: tensor<1x10x10x40x40xf16> -> tensor<1x10x10xf16>
  return %1 : tensor<1x10x10xf16>

  // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1xf16> = dense<1.600000e+03> : tensor<1xf16>
  // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 100, 40, 40]} : tensor<1x10x10x40x40xf16> -> tensor<1x100x40x40xf16>
  // CHECK-NOT:   ReduceSum
  // CHECK:       [[AVGPOOL_0:%.+]] = IE.AvgPool([[RESHAPE_0]]) {exclude_pads, kernel_size = [40, 40], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x100x40x40xf16> -> tensor<1x100x1x1xf16>
  // CHECK:       [[MULTIPLY_0:%.+]] = IE.Multiply([[AVGPOOL_0]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x100x1x1xf16>, tensor<1xf16> -> tensor<1x100x1x1xf16>
  // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[MULTIPLY_0]]) {shape_value = [1, 10, 10]} : tensor<1x100x1x1xf16> -> tensor<1x10x10xf16>
  // CHECK:       return [[RESHAPE_1]] : tensor<1x10x10xf16>
}

// CHECK-LABEL: @ConvertReduceMinToPoolingKernelSize
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x32x112x112xf16>
func.func @ConvertReduceMinToPoolingKernelSize(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
  return %0 : tensor<1x32x112x1xf16>

  // CHECK:       [[NEGATIVE_0:%.+]] = IE.Negative([[INPUT]]) : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
  // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[NEGATIVE_0]]) {
  // CHECK-DAG:       kernel_size = [1, 112], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
  // CHECK:       [[NEGATIVE_1:%.+]] = IE.Negative([[MAXPOOL]]) : tensor<1x32x112x1xf16> -> tensor<1x32x112x1xf16>
  // CHECK:       return [[NEGATIVE_1]] : tensor<1x32x112x1xf16>
}

// CHECK-LABEL: @ConvertReduceMinToPoolingKernelSizeWithMultiAxis
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x2x4x9xf16>
func.func @ConvertReduceMinToPoolingKernelSizeWithMultiAxis(%arg0: tensor<1x2x4x9xf16>) -> tensor<1x1x1x1xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2, 3], keep_dims} : tensor<1x2x4x9xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {
  // CHECK-DAG:       shape_value = [1, 1, 9, 8]} : tensor<1x2x4x9xf16> -> tensor<1x1x9x8xf16>
  // CHECK:       [[NEGATIVE_0:%.+]] = IE.Negative([[RESHAPE]]) : tensor<1x1x9x8xf16> -> tensor<1x1x9x8xf16>
  // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[NEGATIVE_0]]) {
  // CHECK-DAG:       kernel_size = [9, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x9x8xf16> -> tensor<1x1x1x1xf16>
  // CHECK:       [[NEGATIVE_1:%.+]] = IE.Negative([[MAXPOOL]]) : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
  // CHECK:       return [[NEGATIVE_1]] : tensor<1x1x1x1xf16>
}

// CHECK-LABEL: @DoNotConvertReduceMinToPoolingKernelSizeWithMultiAxis
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x8x4x76xf16>
func.func @DoNotConvertReduceMinToPoolingKernelSizeWithMultiAxis(%arg0: tensor<1x8x4x76xf16>) -> tensor<1x1x1x1xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2, 3], keep_dims} : tensor<1x8x4x76xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

  // CHECK:       [[OUTPUT:%.+]] = IE.ReduceMin([[INPUT]]) {
  // CHECK-DAG:       axes_value = [0, 1, 2, 3], keep_dims} : tensor<1x8x4x76xf16> -> tensor<1x1x1x1xf16>
  // CHECK:       return [[OUTPUT]] : tensor<1x1x1x1xf16>
}

// CHECK-LABEL: @DoNotConvertReduceMinToPoolingF32
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x256x7x7xf32>
func.func @DoNotConvertReduceMinToPoolingF32(%arg0: tensor<1x256x7x7xf32>) -> tensor<1x256x1x1xf32> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [2, 3], keep_dims} : tensor<1x256x7x7xf32> -> tensor<1x256x1x1xf32>
  return %0 : tensor<1x256x1x1xf32>

  // CHECK:       [[OUTPUT:%.+]] = IE.ReduceMin([[INPUT]]) {
  // CHECK-DAG:       axes_value = [2, 3], keep_dims} : tensor<1x256x7x7xf32> -> tensor<1x256x1x1xf32>
  // CHECK:       return [[OUTPUT]] : tensor<1x256x1x1xf32>
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
  // CHECK:       return [[RESHAPE2]] : tensor<1x42840x1xf16>
}


// -----

// CHECK-LABEL: @ConvertReduceSumToPoolingOnBatchDim
// CHECK-SAME: [[INPUT:%.+]]: tensor<8x1x4x256xf16>
func.func @ConvertReduceSumToPoolingOnBatchDim(%arg0: tensor<8x1x4x256xf16>) -> tensor<1x4x256xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0]} : tensor<8x1x4x256xf16> -> tensor<1x4x256xf16>
  return %0 : tensor<1x4x256xf16>

  // CHECK-NOT:   ReduceSum
  // CHECK:    [[CST:%.+]] = const.Declare tensor<1xf16> = dense<8.000000e+00> : tensor<1xf16>
  // CHECK:    [[RESHAPED_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 8, 4, 256]} : tensor<8x1x4x256xf16> -> tensor<1x8x4x256xf16>
  // CHECK:    [[RESHAPED_IN_1:%.+]] = IE.Reshape([[RESHAPED_IN]]) {shape_value = [1, 8, 64, 16]} : tensor<1x8x4x256xf16> -> tensor<1x8x64x16xf16>
  // CHECK:    [[TRANSPOSED_IN:%.+]] = IE.Transpose([[RESHAPED_IN_1]]) {order_value = #NWCH} : tensor<1x8x64x16xf16> -> tensor<1x16x8x64xf16>
  // CHECK:    [[AVG_POOL:%.+]] = IE.AvgPool([[TRANSPOSED_IN]]) {exclude_pads, kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x8x64xf16> -> tensor<1x16x1x64xf16>
  // CHECK:    [[MUL:%.+]] = IE.Multiply([[AVG_POOL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x64xf16>, tensor<1xf16> -> tensor<1x16x1x64xf16>
  // CHECK:    [[TRANSPOSED_OUT:%.+]] = IE.Transpose([[MUL]]) {order_value = #NHWC} : tensor<1x16x1x64xf16> -> tensor<1x1x64x16xf16>
  // CHECK:    [[RESHAPED_OUT:%.+]] = IE.Reshape([[TRANSPOSED_OUT]]) {shape_value = [1, 4, 256]} : tensor<1x1x64x16xf16> -> tensor<1x4x256xf16>

  // CHECK:    return [[RESHAPED_OUT]] : tensor<1x4x256xf16>
}

// -----

// CHECK-LABEL: @ConvertReduceMeanToPoolingWithNonConsecutiveAxes
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x3136x64xf16>
func.func @ConvertReduceMeanToPoolingWithNonConsecutiveAxes(%arg0: tensor<1x3136x64xf16>) -> tensor<3136xf16> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [0, 2]} : tensor<1x3136x64xf16> -> tensor<3136xf16>
  return %0 : tensor<3136xf16>

  // CHECK-NOT:   ReduceMean
  // CHECK:    [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 3136, 8, 8]} : tensor<1x3136x64xf16> -> tensor<1x3136x8x8xf16>
  // CHECK:    [[AVGPOOL:%.+]] = IE.AvgPool([[RESHAPE_IN]]) {exclude_pads, kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3136x8x8xf16> -> tensor<1x3136x1x1xf16>
  // CHECK:    [[RESHAPE_OUT:%.+]] = IE.Reshape([[AVGPOOL]]) {shape_value = [1, 3136, 1]} : tensor<1x3136x1x1xf16> -> tensor<1x3136x1xf16>
  // CHECK:    [[RESHAPE_OUT_1:%.+]] = IE.Reshape([[RESHAPE_OUT]]) {shape_value = [3136]} : tensor<1x3136x1xf16> -> tensor<3136xf16>
  // CHECK:    return [[RESHAPE_OUT_1]] : tensor<3136xf16>
}

// -----

// CHECK-LABEL: @ConvertReduceSumToPoolingWithNonConsecutiveAxes
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x3136x64xf16>
func.func @ConvertReduceSumToPoolingWithNonConsecutiveAxes(%arg0: tensor<1x3136x64xf16>) -> tensor<3136xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0, 2]} : tensor<1x3136x64xf16> -> tensor<3136xf16>
  return %0 : tensor<3136xf16>

  // CHECK-NOT: ReduceSum

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf16> = dense<6.400000e+01> : tensor<1xf16>
  // CHECK:     [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 3136, 8, 8]} : tensor<1x3136x64xf16> -> tensor<1x3136x8x8xf16>
  // CHECK:     [[AVGPOOL:%.+]] = IE.AvgPool([[RESHAPE_IN]]) {exclude_pads, kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3136x8x8xf16> -> tensor<1x3136x1x1xf16>
  // CHECK:     [[MUL:%.+]] = IE.Multiply([[AVGPOOL]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3136x1x1xf16>, tensor<1xf16> -> tensor<1x3136x1x1xf16>
  // CHECK:     [[RESHAPE_OUT:%.+]] = IE.Reshape([[MUL]]) {shape_value = [1, 3136, 1]} : tensor<1x3136x1x1xf16> -> tensor<1x3136x1xf16>
  // CHECK:     [[RESHAPE_OUT_1:%.+]] = IE.Reshape([[RESHAPE_OUT]]) {shape_value = [3136]} : tensor<1x3136x1xf16> -> tensor<3136xf16>
  // CHECK:     return [[RESHAPE_OUT_1]] : tensor<3136xf16>
}

// -----

// CHECK-LABEL: @ConvertReduceMinToPoolingWithNonConsecutiveAxes
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x3136x64xf16>
func.func @ConvertReduceMinToPoolingWithNonConsecutiveAxes(%arg0: tensor<1x3136x64xf16>) -> tensor<3136xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [0, 2]} : tensor<1x3136x64xf16> -> tensor<3136xf16>
  return %0 : tensor<3136xf16>

  // CHECK-NOT:   ReduceMin

  // CHECK:    [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 3136, 8, 8]} : tensor<1x3136x64xf16> -> tensor<1x3136x8x8xf16>
  // CHECK:    [[NEG:%.+]] = IE.Negative([[RESHAPE_IN]]) : tensor<1x3136x8x8xf16> -> tensor<1x3136x8x8xf16>
  // CHECK:    [[MAX_POOL:%.+]] = IE.MaxPool([[NEG]]) {kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3136x8x8xf16> -> tensor<1x3136x1x1xf16>
  // CHECK:    [[NEG_OUT:%.+]] = IE.Negative([[MAX_POOL]]) : tensor<1x3136x1x1xf16> -> tensor<1x3136x1x1xf16>
  // CHECK:    [[RESHAPE_OUT:%.+]] = IE.Reshape([[NEG_OUT]]) {shape_value = [1, 3136, 1]} : tensor<1x3136x1x1xf16> -> tensor<1x3136x1xf16>
  // CHECK:    [[RESHAPE_OUT_1:%.+]] = IE.Reshape([[RESHAPE_OUT]]) {shape_value = [3136]} : tensor<1x3136x1xf16> -> tensor<3136xf16>
  // CHECK:    return [[RESHAPE_OUT_1]] : tensor<3136xf16>
}

// -----

// CHECK-LABEL: @ConvertReduceMaxToPoolingWithNonConsecutiveAxes
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x3136x64xf16>
func.func @ConvertReduceMaxToPoolingWithNonConsecutiveAxes(%arg0: tensor<1x3136x64xf16>) -> tensor<3136xf16> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [0, 2]} : tensor<1x3136x64xf16> -> tensor<3136xf16>
  return %0 : tensor<3136xf16>

  // CHECK-NOT:   ReduceMax

  // CHECK:    [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 3136, 8, 8]} : tensor<1x3136x64xf16> -> tensor<1x3136x8x8xf16>
  // CHECK:    [[MAX_POOL:%.+]] = IE.MaxPool([[RESHAPE_IN]]) {kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3136x8x8xf16> -> tensor<1x3136x1x1xf16>
  // CHECK:    [[RESHAPE_OUT:%.+]] = IE.Reshape([[MAX_POOL]]) {shape_value = [1, 3136, 1]} : tensor<1x3136x1x1xf16> -> tensor<1x3136x1xf16>
  // CHECK:    [[RESHAPE_OUT_1:%.+]] = IE.Reshape([[RESHAPE_OUT]]) {shape_value = [3136]} : tensor<1x3136x1xf16> -> tensor<3136xf16>
  // CHECK:    return [[RESHAPE_OUT_1]] : tensor<3136xf16>
}

// -----

// Reduce N axis with H=1 and W channel-aligned: Reshape + Transpose (NWCH) + AvgPool +
// Multiply + Transpose (NHWC) + Reshape avoids Expand/Slice on the channel dim.

// CHECK-LABEL: @ConvertReduceSumBatchAxisAlignedW
// CHECK-SAME: [[INPUT:%.+]]: tensor<4x32x1x16xf16>
func.func @ConvertReduceSumBatchAxisAlignedW(%arg0: tensor<4x32x1x16xf16>) -> tensor<32x1x16xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0]} : tensor<4x32x1x16xf16> -> tensor<32x1x16xf16>
  return %0 : tensor<32x1x16xf16>

  // CHECK-NOT:   ReduceSum
  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1xf16> = dense<4.000000e+00> : tensor<1xf16>
  // CHECK:       [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 4, 32, 16]} : tensor<4x32x1x16xf16> -> tensor<1x4x32x16xf16>
  // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[RESHAPE_IN]]) {order_value = #NWCH} : tensor<1x4x32x16xf16> -> tensor<1x16x4x32xf16>
  // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[TRANSPOSE_IN]]) {exclude_pads, kernel_size = [4, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x4x32xf16> -> tensor<1x16x1x32xf16>
  // CHECK:       [[MUL:%.+]] = IE.Multiply([[AVG_POOL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x32xf16>, tensor<1xf16> -> tensor<1x16x1x32xf16>
  // CHECK:       [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[MUL]]) {order_value = #NHWC} : tensor<1x16x1x32xf16> -> tensor<1x1x32x16xf16>
  // CHECK:       [[RESHAPE_OUT:%.+]] = IE.Reshape([[TRANSPOSE_OUT]]) {shape_value = [32, 1, 16]} : tensor<1x1x32x16xf16> -> tensor<32x1x16xf16>
  // CHECK:       return [[RESHAPE_OUT]] : tensor<32x1x16xf16>
}

// -----

// Same as above with keep_dims: the final Reshape restores the kept N=1 dim.

// CHECK-LABEL: @ConvertReduceSumBatchAxisAlignedWKeepDims
// CHECK-SAME: [[INPUT:%.+]]: tensor<4x32x1x16xf16>
func.func @ConvertReduceSumBatchAxisAlignedWKeepDims(%arg0: tensor<4x32x1x16xf16>) -> tensor<1x32x1x16xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0], keep_dims} : tensor<4x32x1x16xf16> -> tensor<1x32x1x16xf16>
  return %0 : tensor<1x32x1x16xf16>

  // CHECK-NOT:   ReduceSum
  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1xf16> = dense<4.000000e+00> : tensor<1xf16>
  // CHECK:       [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 4, 32, 16]} : tensor<4x32x1x16xf16> -> tensor<1x4x32x16xf16>
  // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[RESHAPE_IN]]) {order_value = #NWCH} : tensor<1x4x32x16xf16> -> tensor<1x16x4x32xf16>
  // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[TRANSPOSE_IN]]) {exclude_pads, kernel_size = [4, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x4x32xf16> -> tensor<1x16x1x32xf16>
  // CHECK:       [[MUL:%.+]] = IE.Multiply([[AVG_POOL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x32xf16>, tensor<1xf16> -> tensor<1x16x1x32xf16>
  // CHECK:       [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[MUL]]) {order_value = #NHWC} : tensor<1x16x1x32xf16> -> tensor<1x1x32x16xf16>
  // CHECK:       [[RESHAPE_OUT:%.+]] = IE.Reshape([[TRANSPOSE_OUT]]) {shape_value = [1, 32, 1, 16]} : tensor<1x1x32x16xf16> -> tensor<1x32x1x16xf16>
  // CHECK:       return [[RESHAPE_OUT]] : tensor<1x32x1x16xf16>
}

// -----

// ReduceMean on N axis with H=1 and W channel-aligned: no Multiply step compared to ReduceSum.

// CHECK-LABEL: @ConvertReduceMeanBatchAxisAlignedW
// CHECK-SAME: [[INPUT:%.+]]: tensor<4x32x1x16xf16>
func.func @ConvertReduceMeanBatchAxisAlignedW(%arg0: tensor<4x32x1x16xf16>) -> tensor<32x1x16xf16> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [0]} : tensor<4x32x1x16xf16> -> tensor<32x1x16xf16>
  return %0 : tensor<32x1x16xf16>

  // CHECK-NOT:   ReduceMean
  // CHECK:       [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 4, 32, 16]} : tensor<4x32x1x16xf16> -> tensor<1x4x32x16xf16>
  // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[RESHAPE_IN]]) {order_value = #NWCH} : tensor<1x4x32x16xf16> -> tensor<1x16x4x32xf16>
  // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[TRANSPOSE_IN]]) {exclude_pads, kernel_size = [4, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x4x32xf16> -> tensor<1x16x1x32xf16>
  // CHECK-NOT:   IE.Multiply
  // CHECK:       [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[AVG_POOL]]) {order_value = #NHWC} : tensor<1x16x1x32xf16> -> tensor<1x1x32x16xf16>
  // CHECK:       [[RESHAPE_OUT:%.+]] = IE.Reshape([[TRANSPOSE_OUT]]) {shape_value = [32, 1, 16]} : tensor<1x1x32x16xf16> -> tensor<32x1x16xf16>
  // CHECK:       return [[RESHAPE_OUT]] : tensor<32x1x16xf16>
}

// -----

// H != 1: new path is not applicable; falls back to the generic batch-dim reshape path.

// CHECK-LABEL: @DoNotConvertReduceSumBatchAxisNonUnitH
// CHECK-SAME: [[INPUT:%.+]]: tensor<4x32x2x16xf16>
func.func @DoNotConvertReduceSumBatchAxisNonUnitH(%arg0: tensor<4x32x2x16xf16>) -> tensor<32x2x16xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0]} : tensor<4x32x2x16xf16> -> tensor<32x2x16xf16>
  return %0 : tensor<32x2x16xf16>

  // CHECK-NOT:   ReduceSum
  // New path would produce shape_value = [1, 4, 32, 16]; old path produces [1, 1, 4, 1024].
  // CHECK:       [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 4, 1024]} : tensor<4x32x2x16xf16> -> tensor<1x1x4x1024xf16>
  // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[RESHAPE_IN]]) {exclude_pads, kernel_size = [4, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x4x1024xf16> -> tensor<1x1x1x1024xf16>
}

// -----

// W not channel-aligned: new path is not applicable; falls back to the generic batch-dim reshape path.

// CHECK-LABEL: @DoNotConvertReduceSumBatchAxisUnalignedW
// CHECK-SAME: [[INPUT:%.+]]: tensor<4x32x1x12xf16>
func.func @DoNotConvertReduceSumBatchAxisUnalignedW(%arg0: tensor<4x32x1x12xf16>) -> tensor<32x1x12xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0]} : tensor<4x32x1x12xf16> -> tensor<32x1x12xf16>
  return %0 : tensor<32x1x12xf16>

  // CHECK-NOT:   ReduceSum
  // New path would produce shape_value = [1, 4, 32, 12]; old path produces [1, 1, 4, 384].
  // CHECK:       [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 4, 384]} : tensor<4x32x1x12xf16> -> tensor<1x1x4x384xf16>
  // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[RESHAPE_IN]]) {exclude_pads, kernel_size = [4, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x4x384xf16> -> tensor<1x1x1x384xf16>
}

// -----

// 3D ReduceSum on axis 0 (batch), W=2816 is channel-aligned: takes the Reshape+Transpose path.

// CHECK-LABEL: @ConvertReduceSum3DBatchAxisAlignedW
// CHECK-SAME: [[INPUT:%.+]]: tensor<8x1024x2816xf16>
func.func @ConvertReduceSum3DBatchAxisAlignedW(%arg0: tensor<8x1024x2816xf16>) -> tensor<1024x2816xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0]} : tensor<8x1024x2816xf16> -> tensor<1024x2816xf16>
  return %0 : tensor<1024x2816xf16>

  // CHECK-NOT:   IE.ReduceSum
  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1xf16> = dense<8.000000e+00> : tensor<1xf16>
  // CHECK:       [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 8, 1024, 2816]} : tensor<8x1024x2816xf16> -> tensor<1x8x1024x2816xf16>
  // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[RESHAPE_IN]]) {order_value = #NWCH} : tensor<1x8x1024x2816xf16> -> tensor<1x2816x8x1024xf16>
  // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[TRANSPOSE_IN]]) {exclude_pads, kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x2816x8x1024xf16> -> tensor<1x2816x1x1024xf16>
  // CHECK:       [[MUL:%.+]] = IE.Multiply([[AVG_POOL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2816x1x1024xf16>, tensor<1xf16> -> tensor<1x2816x1x1024xf16>
  // CHECK:       [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[MUL]]) {order_value = #NHWC} : tensor<1x2816x1x1024xf16> -> tensor<1x1x1024x2816xf16>
  // CHECK:       [[RESHAPE_OUT:%.+]] = IE.Reshape([[TRANSPOSE_OUT]]) {shape_value = [1024, 2816]} : tensor<1x1x1024x2816xf16> -> tensor<1024x2816xf16>
  // CHECK:       return [[RESHAPE_OUT]] : tensor<1024x2816xf16>
}

// -----

// Same 3D case with keep_dims: final Reshape restores the N=1 dim.

// CHECK-LABEL: @ConvertReduceSum3DBatchAxisAlignedWKeepDims
// CHECK-SAME: [[INPUT:%.+]]: tensor<8x1024x2816xf16>
func.func @ConvertReduceSum3DBatchAxisAlignedWKeepDims(%arg0: tensor<8x1024x2816xf16>) -> tensor<1x1024x2816xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0], keep_dims} : tensor<8x1024x2816xf16> -> tensor<1x1024x2816xf16>
  return %0 : tensor<1x1024x2816xf16>

  // CHECK-NOT:   IE.ReduceSum
  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1xf16> = dense<8.000000e+00> : tensor<1xf16>
  // CHECK:       [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 8, 1024, 2816]} : tensor<8x1024x2816xf16> -> tensor<1x8x1024x2816xf16>
  // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[RESHAPE_IN]]) {order_value = #NWCH} : tensor<1x8x1024x2816xf16> -> tensor<1x2816x8x1024xf16>
  // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[TRANSPOSE_IN]]) {exclude_pads, kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x2816x8x1024xf16> -> tensor<1x2816x1x1024xf16>
  // CHECK:       [[MUL:%.+]] = IE.Multiply([[AVG_POOL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2816x1x1024xf16>, tensor<1xf16> -> tensor<1x2816x1x1024xf16>
  // CHECK:       [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[MUL]]) {order_value = #NHWC} : tensor<1x2816x1x1024xf16> -> tensor<1x1x1024x2816xf16>
  // CHECK:       [[RESHAPE_OUT:%.+]] = IE.Reshape([[TRANSPOSE_OUT]]) {shape_value = [1, 1024, 2816]} : tensor<1x1x1024x2816xf16> -> tensor<1x1024x2816xf16>
  // CHECK:       return [[RESHAPE_OUT]] : tensor<1x1024x2816xf16>
}

// -----

// 3D ReduceMean on axis 0, W channel-aligned: same Reshape+Transpose path without Multiply.

// CHECK-LABEL: @ConvertReduceMean3DBatchAxisAlignedW
// CHECK-SAME: [[INPUT:%.+]]: tensor<8x1024x2816xf16>
func.func @ConvertReduceMean3DBatchAxisAlignedW(%arg0: tensor<8x1024x2816xf16>) -> tensor<1024x2816xf16> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [0]} : tensor<8x1024x2816xf16> -> tensor<1024x2816xf16>
  return %0 : tensor<1024x2816xf16>

  // CHECK-NOT:   IE.ReduceMean
  // CHECK:       [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 8, 1024, 2816]} : tensor<8x1024x2816xf16> -> tensor<1x8x1024x2816xf16>
  // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[RESHAPE_IN]]) {order_value = #NWCH} : tensor<1x8x1024x2816xf16> -> tensor<1x2816x8x1024xf16>
  // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[TRANSPOSE_IN]]) {exclude_pads, kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x2816x8x1024xf16> -> tensor<1x2816x1x1024xf16>
  // CHECK-NOT:   IE.Multiply
  // CHECK:       [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[AVG_POOL]]) {order_value = #NHWC} : tensor<1x2816x1x1024xf16> -> tensor<1x1x1024x2816xf16>
  // CHECK:       [[RESHAPE_OUT:%.+]] = IE.Reshape([[TRANSPOSE_OUT]]) {shape_value = [1024, 2816]} : tensor<1x1x1024x2816xf16> -> tensor<1024x2816xf16>
  // CHECK:       return [[RESHAPE_OUT]] : tensor<1024x2816xf16>
}

// -----

// 3D ReduceSum axis 0 with W not channel-aligned: new path not applicable, falls back to
// the generic non-spatial reshape path.

// CHECK-LABEL: @DoNotConvert3DReduceSumBatchAxisUnalignedW
// CHECK-SAME: [[INPUT:%.+]]: tensor<8x1024x100xf16>
func.func @DoNotConvert3DReduceSumBatchAxisUnalignedW(%arg0: tensor<8x1024x100xf16>) -> tensor<1024x100xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [0]} : tensor<8x1024x100xf16> -> tensor<1024x100xf16>
  return %0 : tensor<1024x100xf16>

  // CHECK-NOT:   IE.ReduceSum
  // 100 is not a multiple of 16, so new path is skipped; generic path flattens dims.
  // CHECK:       [[RESHAPE_IN:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 8, 102400]} : tensor<8x1024x100xf16> -> tensor<1x1x8x102400xf16>
  // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[RESHAPE_IN]]) {exclude_pads, kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x8x102400xf16> -> tensor<1x1x1x102400xf16>
}
