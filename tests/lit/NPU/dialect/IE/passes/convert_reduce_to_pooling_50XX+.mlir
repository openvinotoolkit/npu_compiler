//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW enable-is-reduce-supported=true" --convert-reduce-to-pooling %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK-LABEL:    func.func @DoNotConvertReduceMeanToPoolingOnChannelAxis
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<256x7x7xf16>)
func.func @DoNotConvertReduceMeanToPoolingOnChannelAxis(%arg0: tensor<256x7x7xf16>) -> tensor<256x1x7xf16> {
    %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims} : tensor<256x7x7xf16> -> tensor<256x1x7xf16>
    return %0 : tensor<256x1x7xf16>

    // CHECK-NOT: IE.AvgPool
    // CHECK: [[MEAN:%.+]] = IE.ReduceMean([[INPUT]]) {axes_value = [1], keep_dims} : tensor<256x7x7xf16> -> tensor<256x1x7xf16>
}

// -----

// CHECK-LABEL:    func.func @DoNotConvertReduceSumToPoolingOnChannelAxis
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<256x7x7xf16>)
func.func @DoNotConvertReduceSumToPoolingOnChannelAxis(%arg0: tensor<256x7x7xf16>) -> tensor<256x1x7xf16> {
    %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<256x7x7xf16> -> tensor<256x1x7xf16>
    return %0 : tensor<256x1x7xf16>

    // CHECK-NOT: IE.AvgPool
    // CHECK: [[MEAN:%.+]] = IE.ReduceSum([[INPUT]]) {axes_value = [1], keep_dims} : tensor<256x7x7xf16> -> tensor<256x1x7xf16>
}

// -----

// CHECK-LABEL: @ConvertReduceMeanToPoolingOnSpatialDimension
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x1x1x50xf16>)
func.func @ConvertReduceMeanToPoolingOnSpatialDimension(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

  // CHECK-NOT:   ReduceMean
  // CHECK:       [[RESHAPE1:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 10, 5]}
  // CHECK-SAME:      : tensor<1x1x1x50xf16> -> tensor<1x1x10x5xf16>
  // CHECK:       [[POOL:%.+]] = IE.AvgPool([[RESHAPE1]])
  // CHECK-SAME:      {exclude_pads, kernel_size = [10, 5], pads_begin = [0, 0], pads_end = [0, 0]
  // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
  // CHECK-SAME:      : tensor<1x1x10x5xf16> -> tensor<1x1x1x1xf16>
}

// -----

// CHECK-LABEL: @ConvertReduceSumToPoolingOnSpatialDimension
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x1x1x50xf16>)
func.func @ConvertReduceSumToPoolingOnSpatialDimension(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

  // CHECK-NOT:   ReduceSum
  // CHECK:       [[RESHAPE1:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 10, 5]}
  // CHECK-SAME:      : tensor<1x1x1x50xf16> -> tensor<1x1x10x5xf16>
  // CHECK:       [[POOL:%.+]] = IE.AvgPool([[RESHAPE1]])
  // CHECK-SAME:      {exclude_pads, kernel_size = [10, 5], pads_begin = [0, 0], pads_end = [0, 0]
  // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
  // CHECK-SAME:      : tensor<1x1x10x5xf16> -> tensor<1x1x1x1xf16>
}
