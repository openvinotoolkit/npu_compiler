//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-logsoftmax-variants --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @FuseLogSoftmaxTopK_ConvertPattern
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x151x7049xf32>)
func.func @FuseLogSoftmaxTopK_ConvertPattern(%arg0: tensor<1x1x151x7049xf32>) -> (tensor<1x1x1x151xsi64>, tensor<1x1x151x7049xf32>) {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1x151x7049xf32> -> tensor<1x1x151x7049xf16>
  %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7056xf16>
  %2 = IE.LogSoftmax(%1) {axisInd = 3 : i64, padSize = 7 : i64} : tensor<1x1x151x7056xf16> -> tensor<1x1x151x7056xf16>
  %3 = IE.Slice %2 [0, 0, 0, 0] [1, 1, 151, 7049] : tensor<1x1x151x7056xf16> to tensor<1x1x151x7049xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7049xf32>
  %output_values, %target_shape = IE.TopK(%0) {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x1xf16>, tensor<1x1x151x1xsi32>
  %5 = IE.AffineReshape(%target_shape) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x1x151x1xsi32> -> tensor<1x1x1x151xsi32>
  %6 = IE.Convert(%5) {dstElemType = si64} : tensor<1x1x1x151xsi32> -> tensor<1x1x1x151xsi64>

  return %6, %4 : tensor<1x1x1x151xsi64>, tensor<1x1x151x7049xf32>

  // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x1x151x7049xf32> -> tensor<1x1x151x7049xf16>
  // CHECK: [[EXPAND:%.+]] = IE.Expand([[CONVERT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7056xf16>
  // CHECK: [[OUTPUT:%.+]], [[TOPK_OUTPUT:%.+]] = IE.LogSoftmaxTopK([[EXPAND]]) {axisInd = 3 : i64, dstElemType = f32, padSize = 7 : i64} : tensor<1x1x151x7056xf16> -> tensor<1x1x151x7056xf32>, tensor<1x1x151x1xsi64>
  // CHECK: [[SLICE:%.+]] = IE.Slice [[OUTPUT]] [0, 0, 0, 0] [1, 1, 151, 7049] : tensor<1x1x151x7056xf32> to tensor<1x1x151x7049xf32>
  // CHECK: [[RESHAPE_TOPK:%.+]] = IE.AffineReshape([[TOPK_OUTPUT]]) {dim_mapping = {{\[\[}}0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x1x151x1xsi64> -> tensor<1x1x1x151xsi64>
  // CHECK: return [[RESHAPE_TOPK]], [[SLICE]] : tensor<1x1x1x151xsi64>, tensor<1x1x151x7049xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

// CHECK-LABEL: @FuseLogSoftmaxTopK_PermuteCastPattern
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x7049x151x1xf16, {order = #NHWC}>)
func.func @FuseLogSoftmaxTopK_PermuteCastPattern(%arg0: tensor<1x7049x151x1xf16, {order = #NHWC}>) -> (tensor<1x1x1x151xsi64>, tensor<1x1x151x7049xf32>) {
  %0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #map2} : tensor<1x7049x151x1xf16, {order = #NHWC}> -> tensor<151x7049x1x1xf16>
  %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 7, 0, 0]} : tensor<151x7049x1x1xf16> -> tensor<151x7056x1x1xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 151, 7056]} : tensor<151x7056x1x1xf16> -> tensor<1x1x151x7056xf16>
  %3 = IE.LogSoftmax(%2) {axisInd = 3 : i64, padSize = 7 : i64} : tensor<1x1x151x7056xf16> -> tensor<1x1x151x7056xf16>
  %4 = IE.Slice %3 [0, 0, 0, 0] [1, 1, 151, 7049] : tensor<1x1x151x7056xf16> to tensor<1x1x151x7049xf16>
  %5 = IE.Convert(%4) {dstElemType = f32} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7049xf32>
  %6 = IE.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 151, 7049]} : tensor<151x7049x1x1xf16> -> tensor<1x1x151x7049xf16>
  %output_values, %target_shape = IE.TopK(%6) {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x1xf16>, tensor<1x1x151x1xsi32>
  %7 = IE.AffineReshape(%target_shape) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x1x151x1xsi32> -> tensor<1x1x1x151xsi32>
  %8 = IE.Convert(%7) {dstElemType = si64} : tensor<1x1x1x151xsi32> -> tensor<1x1x1x151xsi64>

  return %8, %5 : tensor<1x1x1x151xsi64>, tensor<1x1x151x7049xf32>

  // CHECK: [[PERMUTE:%.+]] = IE.PermuteCast([[ARG0]]) {dst_order = #NCHW, mem_perm = #map} : tensor<1x7049x151x1xf16, {order = #NHWC}> -> tensor<151x7049x1x1xf16>
  // CHECK: [[EXPAND:%.+]] = IE.Expand([[PERMUTE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 7, 0, 0]} : tensor<151x7049x1x1xf16> -> tensor<151x7056x1x1xf16>
  // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[EXPAND]]) {dim_mapping = {{\[\[}}0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 151, 7056]} : tensor<151x7056x1x1xf16> -> tensor<1x1x151x7056xf16>
  // CHECK: [[OUTPUT:%.+]], [[TOPK_OUTPUT:%.+]] = IE.LogSoftmaxTopK([[RESHAPE1]]) {axisInd = 3 : i64, dstElemType = f32, padSize = 7 : i64} : tensor<1x1x151x7056xf16> -> tensor<1x1x151x7056xf32>, tensor<1x1x151x1xsi64>
  // CHECK: [[SLICE:%.+]] = IE.Slice [[OUTPUT]] [0, 0, 0, 0] [1, 1, 151, 7049] : tensor<1x1x151x7056xf32> to tensor<1x1x151x7049xf32>
  // CHECK: [[RESHAPE_TOPK:%.+]] = IE.AffineReshape([[TOPK_OUTPUT]]) {dim_mapping = {{\[\[}}0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x1x151x1xsi64> -> tensor<1x1x1x151xsi64>
  // CHECK: return [[RESHAPE_TOPK]], [[SLICE]] : tensor<1x1x1x151xsi64>, tensor<1x1x151x7049xf32>
}

// -----

// CHECK-LABEL: @NotFuseLogSoftmaxTopK_NonInnerAxis
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x151x7049xf32>)
func.func @NotFuseLogSoftmaxTopK_NonInnerAxis(%arg0: tensor<1x1x151x7049xf32>) -> (tensor<1x1x1x7049xsi64>, tensor<1x1x151x7049xf32>) {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1x151x7049xf32> -> tensor<1x1x151x7049xf16>
  %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 7, 0]} : tensor<1x1x151x7049xf16> -> tensor<1x1x158x7049xf16>
  %2 = IE.LogSoftmax(%1) {axisInd = 2 : i64, padSize = 7 : i64} : tensor<1x1x158x7049xf16> -> tensor<1x1x158x7049xf16>
  %3 = IE.Slice %2 [0, 0, 0, 0] [1, 1, 151, 7049] : tensor<1x1x158x7049xf16> to tensor<1x1x151x7049xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7049xf32>
  %output_values, %target_shape = IE.TopK(%0) {axis = 2 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x1x151x7049xf16> -> tensor<1x1x1x7049xf16>, tensor<1x1x1x7049xsi32>
  %5 = IE.Convert(%target_shape) {dstElemType = si64} : tensor<1x1x1x7049xsi32> -> tensor<1x1x1x7049xsi64>

  return %5, %4 : tensor<1x1x1x7049xsi64>, tensor<1x1x151x7049xf32>

  // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x1x151x7049xf32> -> tensor<1x1x151x7049xf16>
  // CHECK: [[EXPAND:%.+]] = IE.Expand([[CONVERT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 7, 0]} : tensor<1x1x151x7049xf16> -> tensor<1x1x158x7049xf16>
  // CHECK: [[LOGSOFTMAX:%.+]] = IE.LogSoftmax([[EXPAND]]) {axisInd = 2 : i64, padSize = 7 : i64} : tensor<1x1x158x7049xf16> -> tensor<1x1x158x7049xf16>
  // CHECK: [[SLICE:%.+]] = IE.Slice [[LOGSOFTMAX]] [0, 0, 0, 0] [1, 1, 151, 7049] : tensor<1x1x158x7049xf16> to tensor<1x1x151x7049xf16>
  // CHECK: [[CONVERT_OUT:%.+]] = IE.Convert([[SLICE]]) {dstElemType = f32} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7049xf32>
  // CHECK: [[OUTPUT_VALUES:%.+]], [[TARGET_SHAPE:%.+]] = IE.TopK([[CONVERT]]) {axis = 2 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x1x151x7049xf16> -> tensor<1x1x1x7049xf16>, tensor<1x1x1x7049xsi32>
  // CHECK: [[CONVERT_TOPK:%.+]] = IE.Convert([[TARGET_SHAPE]]) {dstElemType = si64} : tensor<1x1x1x7049xsi32> -> tensor<1x1x1x7049xsi64>
  // CHECK: return [[CONVERT_TOPK]], [[CONVERT_OUT]] : tensor<1x1x1x7049xsi64>, tensor<1x1x151x7049xf32>
  // CHECK-NOT: IE.LogSoftmaxTopK
}

// -----

// CHECK-LABEL: @NotFuseLogSoftmaxTopK_MaxValuesOutputUsed
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x151x7049xf32>)
func.func @NotFuseLogSoftmaxTopK_MaxValuesOutputUsed(%arg0: tensor<1x1x151x7049xf32>) -> (tensor<1x1x1x151xsi64>, tensor<1x1x151x7049xf32>, tensor<1x1x151x1xf16>) {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1x151x7049xf32> -> tensor<1x1x151x7049xf16>
  %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7056xf16>
  %2 = IE.LogSoftmax(%1) {axisInd = 3 : i64, padSize = 7 : i64} : tensor<1x1x151x7056xf16> -> tensor<1x1x151x7056xf16>
  %3 = IE.Slice %2 [0, 0, 0, 0] [1, 1, 151, 7049] : tensor<1x1x151x7056xf16> to tensor<1x1x151x7049xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7049xf32>
  %output_values, %target_shape = IE.TopK(%0) {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x1xf16>, tensor<1x1x151x1xsi32>
  %5 = IE.AffineReshape(%target_shape) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x1x151x1xsi32> -> tensor<1x1x1x151xsi32>
  %6 = IE.Convert(%5) {dstElemType = si64} : tensor<1x1x1x151xsi32> -> tensor<1x1x1x151xsi64>

  return %6, %4, %output_values : tensor<1x1x1x151xsi64>, tensor<1x1x151x7049xf32>, tensor<1x1x151x1xf16>

  // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x1x151x7049xf32> -> tensor<1x1x151x7049xf16>
  // CHECK: [[EXPAND:%.+]] = IE.Expand([[CONVERT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7056xf16>
  // CHECK: [[LOGSOFTMAX:%.+]] = IE.LogSoftmax([[EXPAND]]) {axisInd = 3 : i64, padSize = 7 : i64} : tensor<1x1x151x7056xf16> -> tensor<1x1x151x7056xf16>
  // CHECK: [[SLICE:%.+]] = IE.Slice [[LOGSOFTMAX]] [0, 0, 0, 0] [1, 1, 151, 7049] : tensor<1x1x151x7056xf16> to tensor<1x1x151x7049xf16>
  // CHECK: [[CONVERT_OUT:%.+]] = IE.Convert([[SLICE]]) {dstElemType = f32} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7049xf32>
  // CHECK: [[OUTPUT_VALUES:%.+]], [[TARGET_SHAPE:%.+]] = IE.TopK([[CONVERT]]) {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x1xf16>, tensor<1x1x151x1xsi32>
  // CHECK: [[RESHAPE_TOPK:%.+]] = IE.AffineReshape([[TARGET_SHAPE]]) {dim_mapping = {{\[\[}}0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x1x151x1xsi32> -> tensor<1x1x1x151xsi32>
  // CHECK: [[CONVERT_TOPK:%.+]] = IE.Convert([[RESHAPE_TOPK]]) {dstElemType = si64} : tensor<1x1x1x151xsi32> -> tensor<1x1x1x151xsi64>
  // CHECK: return [[CONVERT_TOPK]], [[CONVERT_OUT]], [[OUTPUT_VALUES]] : tensor<1x1x1x151xsi64>, tensor<1x1x151x7049xf32>, tensor<1x1x151x1xf16>
  // CHECK-NOT: IE.LogSoftmaxTopK
}

// -----

// CHECK-LABEL: @FuseLogSoftmaxPeak_ConvertPattern
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x151x7049xf16>)
func.func @FuseLogSoftmaxPeak_ConvertPattern(%arg0: tensor<1x1x151x7049xf16>) -> (tensor<1x1x1x151xsi64>, tensor<1x1x1x151xf32>) {
  %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7056xf16>
  %1 = IE.LogSoftmax(%0) {axisInd = 3 : i64, padSize = 7 : i64} : tensor<1x1x151x7056xf16> -> tensor<1x1x151x7056xf16>
  %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 151, 7049] : tensor<1x1x151x7056xf16> to tensor<1x1x151x7049xf16>
  %output_values, %target_shape = IE.TopK(%arg0) {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x1xf16>, tensor<1x1x151x1xsi32>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 151, 7049, 1]} : tensor<1x1x151x7049xf16> -> tensor<1x151x7049x1xf16>
  %4 = IE.AffineReshape(%target_shape) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 151, 1, 1]} : tensor<1x1x151x1xsi32> -> tensor<1x151x1x1xsi32>
  %5 = IE.GatherElements(%3, %4) {axis = 2 : i64} : tensor<1x151x7049x1xf16>, tensor<1x151x1x1xsi32> -> tensor<1x151x1x1xf16>
  %6 = IE.AffineReshape(%5) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x151x1x1xf16> -> tensor<1x1x1x151xf16>
  %7 = IE.AffineReshape(%target_shape) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x1x151x1xsi32> -> tensor<1x1x1x151xsi32>
  %8 = IE.Convert(%7) {dstElemType = si64} : tensor<1x1x1x151xsi32> -> tensor<1x1x1x151xsi64>
  %9 = IE.Convert(%6) {dstElemType = f32} : tensor<1x1x1x151xf16> -> tensor<1x1x1x151xf32>
  return %8, %9 : tensor<1x1x1x151xsi64>, tensor<1x1x1x151xf32>

  // CHECK: [[EXPAND:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7056xf16>
  // CHECK: [[OUTPUT:%.+]], [[TOPK_OUTPUT:%.+]] = IE.LogSoftmaxPeak([[EXPAND]]) {axisInd = 3 : i64, dstElemType = f32, padSize = 7 : i64} : tensor<1x1x151x7056xf16> -> tensor<1x1x151x1xf32>, tensor<1x1x151x1xsi64>
  // CHECK: [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[OUTPUT]]) {dim_mapping = {{\[\[}}0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x1x151x1xf32> -> tensor<1x1x1x151xf32>
  // CHECK: [[RESHAPE_TOPK:%.+]] = IE.AffineReshape([[TOPK_OUTPUT]]) {dim_mapping = {{\[\[}}0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x1x151x1xsi64> -> tensor<1x1x1x151xsi64>
  // CHECK: return [[RESHAPE_TOPK]], [[RESHAPE_OUT]] : tensor<1x1x1x151xsi64>, tensor<1x1x1x151xf32>
}

// -----

// CHECK-LABEL: @FuseLogSoftmaxPeak_WithAffineReshapeBeforeLogSoftmax
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<151x7049x1x1xf16>)
func.func @FuseLogSoftmaxPeak_WithAffineReshapeBeforeLogSoftmax(%arg0: tensor<151x7049x1x1xf16>) -> (tensor<1x1x1x151xsi64>, tensor<1x1x1x151xf32>) {
  %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 151, 7049]} : tensor<151x7049x1x1xf16> -> tensor<1x1x151x7049xf16>
  %1 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 7, 0, 0]} : tensor<151x7049x1x1xf16> -> tensor<151x7056x1x1xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 151, 7056]} : tensor<151x7056x1x1xf16> -> tensor<1x1x151x7056xf16>
  %3 = IE.LogSoftmax(%2) {axisInd = 3 : i64, padSize = 7 : i64} : tensor<1x1x151x7056xf16> -> tensor<1x1x151x7056xf16>
  %4 = IE.Slice %3 [0, 0, 0, 0] [1, 1, 151, 7049] : tensor<1x1x151x7056xf16> to tensor<1x1x151x7049xf16>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 151, 7049, 1]} : tensor<1x1x151x7049xf16> -> tensor<1x151x7049x1xf16>
  %output_values, %target_shape = IE.TopK(%0) {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x1xf16>, tensor<1x1x151x1xsi32>
  %6 = IE.AffineReshape(%target_shape) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 151, 1, 1]} : tensor<1x1x151x1xsi32> -> tensor<1x151x1x1xsi32>
  %7 = IE.AffineReshape(%target_shape) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x1x151x1xsi32> -> tensor<1x1x1x151xsi32>
  %8 = IE.GatherElements(%5, %6) {axis = 2 : i64} : tensor<1x151x7049x1xf16>, tensor<1x151x1x1xsi32> -> tensor<1x151x1x1xf16>
  %9 = IE.AffineReshape(%8) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x151x1x1xf16> -> tensor<1x1x1x151xf16>
  %10 = IE.Convert(%9) {dstElemType = f32} : tensor<1x1x1x151xf16> -> tensor<1x1x1x151xf32>
  %11 = IE.Convert(%7) {dstElemType = si64} : tensor<1x1x1x151xsi32> -> tensor<1x1x1x151xsi64>
  return %11, %10 : tensor<1x1x1x151xsi64>, tensor<1x1x1x151xf32>

  // CHECK: [[EXPAND:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 7, 0, 0]} : tensor<151x7049x1x1xf16> -> tensor<151x7056x1x1xf16>
  // CHECK: [[RESHAPE_EXPAND:%.+]] = IE.AffineReshape([[EXPAND]]) {dim_mapping = {{\[\[}}0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 151, 7056]} : tensor<151x7056x1x1xf16> -> tensor<1x1x151x7056xf16>
  // CHECK: [[OUTPUT:%.+]], [[TOPK_OUTPUT:%.+]] = IE.LogSoftmaxPeak([[RESHAPE_EXPAND]]) {axisInd = 3 : i64, dstElemType = f32, padSize = 7 : i64} : tensor<1x1x151x7056xf16> -> tensor<1x1x151x1xf32>, tensor<1x1x151x1xsi64>
  // CHECK: [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[OUTPUT]]) {dim_mapping = {{\[\[}}0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x1x151x1xf32> -> tensor<1x1x1x151xf32>
  // CHECK: [[RESHAPE_TOPK:%.+]] = IE.AffineReshape([[TOPK_OUTPUT]]) {dim_mapping = {{\[\[}}0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x1x151x1xsi64> -> tensor<1x1x1x151xsi64>
  // CHECK: return [[RESHAPE_TOPK]], [[RESHAPE_OUT]] : tensor<1x1x1x151xsi64>, tensor<1x1x1x151xf32>
}

// -----

// CHECK-LABEL: @NotFuseLogSoftmaxPeak_NonInnerAxis
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x151x7049x1xf16>)
func.func @NotFuseLogSoftmaxPeak_NonInnerAxis(%arg0: tensor<1x151x7049x1xf16>) -> (tensor<1x151xsi64>, tensor<1x151xf32>) {
  %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 7, 0]} : tensor<1x151x7049x1xf16> -> tensor<1x151x7056x1xf16>
  %1 = IE.LogSoftmax(%0) {axisInd = 2 : i64, padSize = 7 : i64} : tensor<1x151x7056x1xf16> -> tensor<1x151x7056x1xf16>
  %2 = IE.Slice %1 [0, 0, 0, 0] [1, 151, 7049, 1] : tensor<1x151x7056x1xf16> to tensor<1x151x7049x1xf16>
  %output_values, %target_shape = IE.TopK(%arg0) {axis = 2 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x151x7049x1xf16> -> tensor<1x151x1x1xf16>, tensor<1x151x1x1xsi32>
  %3 = IE.GatherElements(%2, %target_shape) {axis = 2 : i64} : tensor<1x151x7049x1xf16>, tensor<1x151x1x1xsi32> -> tensor<1x151x1x1xf16>
  %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 151]} : tensor<1x151x1x1xf16> -> tensor<1x151xf16>
  %5 = IE.AffineReshape(%target_shape) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 151]} : tensor<1x151x1x1xsi32> -> tensor<1x151xsi32>
  %6 = IE.Convert(%5) {dstElemType = si64} : tensor<1x151xsi32> -> tensor<1x151xsi64>
  %7 = IE.Convert(%4) {dstElemType = f32} : tensor<1x151xf16> -> tensor<1x151xf32>
  return %6, %7 : tensor<1x151xsi64>, tensor<1x151xf32>

  // CHECK-NOT: IE.LogSoftmaxPeak
  // CHECK: [[EXPAND:%.+]] = IE.Expand([[ARG0]])
  // CHECK: [[LOGSOFTMAX:%.+]] = IE.LogSoftmax([[EXPAND]]) {axisInd = 2 : i64, padSize = 7 : i64}
  // CHECK: return
}

// -----

// CHECK-LABEL: @NotFuseLogSoftmaxPeak_TopKValuesUsed
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x151x7049xf16>)
func.func @NotFuseLogSoftmaxPeak_TopKValuesUsed(%arg0: tensor<1x1x151x7049xf16>) -> (tensor<1x1x1x151xsi64>, tensor<1x1x1x151xf32>, tensor<1x1x151x1xf16>) {
  %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7056xf16>
  %1 = IE.LogSoftmax(%0) {axisInd = 3 : i64, padSize = 7 : i64} : tensor<1x1x151x7056xf16> -> tensor<1x1x151x7056xf16>
  %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 151, 7049] : tensor<1x1x151x7056xf16> to tensor<1x1x151x7049xf16>
  %output_values, %target_shape = IE.TopK(%arg0) {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x1x151x7049xf16> -> tensor<1x1x151x1xf16>, tensor<1x1x151x1xsi32>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 151, 7049, 1]} : tensor<1x1x151x7049xf16> -> tensor<1x151x7049x1xf16>
  %4 = IE.AffineReshape(%target_shape) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 151, 1, 1]} : tensor<1x1x151x1xsi32> -> tensor<1x151x1x1xsi32>
  %5 = IE.GatherElements(%3, %4) {axis = 2 : i64} : tensor<1x151x7049x1xf16>, tensor<1x151x1x1xsi32> -> tensor<1x151x1x1xf16>
  %6 = IE.AffineReshape(%5) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x151x1x1xf16> -> tensor<1x1x1x151xf16>
  %7 = IE.AffineReshape(%target_shape) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 151]} : tensor<1x1x151x1xsi32> -> tensor<1x1x1x151xsi32>
  %8 = IE.Convert(%7) {dstElemType = si64} : tensor<1x1x1x151xsi32> -> tensor<1x1x1x151xsi64>
  %9 = IE.Convert(%6) {dstElemType = f32} : tensor<1x1x1x151xf16> -> tensor<1x1x1x151xf32>
  return %8, %9, %output_values : tensor<1x1x1x151xsi64>, tensor<1x1x1x151xf32>, tensor<1x1x151x1xf16>

  // CHECK-NOT: IE.LogSoftmaxPeak
  // CHECK: [[EXPAND:%.+]] = IE.Expand([[ARG0]])
  // CHECK: [[LOGSOFTMAX:%.+]] = IE.LogSoftmax([[EXPAND]]) {axisInd = 3 : i64, padSize = 7 : i64}
  // CHECK: [[TOPK_VALUES:%.+]], [[TOPK_INDICES:%.+]] = IE.TopK([[ARG0]])
  // CHECK: return {{%.+}}, {{%.+}}, [[TOPK_VALUES]]
}
