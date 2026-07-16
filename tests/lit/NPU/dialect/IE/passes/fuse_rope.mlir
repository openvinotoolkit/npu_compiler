//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-rope --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FuseRoPE
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x1024x128xf32>, [[ARG1:%.+]]: tensor<1x1x1024x128xf32>, [[ARG2:%.+]]: tensor<1x1x1024x128xf32>)
func.func @FuseRoPE(%arg0: tensor<1x32x1024x128xf32>, %arg1: tensor<1x1x1024x128xf32>, %arg2: tensor<1x1x1024x128xf32>) -> tensor<1x32x1024x128xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x128xf32>, tensor<1x1x1024x128xf32> -> tensor<1x32x1024x128xf32>
    %1 = IE.StridedSlice(%arg0) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 64], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [1, 32, 1024, 128], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<1x32x1024x128xf32> -> tensor<1x32x1024x64xf32>
    %2 = IE.Multiply(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x64xf32>
    %3 = IE.StridedSlice(%arg0) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [1, 32, 1024, 64], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<1x32x1024x128xf32> -> tensor<1x32x1024x64xf32>
    %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 64]]} : tensor<1x32x1024x64xf32>, tensor<1x32x1024x64xf32> -> tensor<1x32x1024x128xf32>
    %5 = IE.Multiply(%4, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x128xf32>, tensor<1x1x1024x128xf32> -> tensor<1x32x1024x128xf32>
    %6 = IE.Add(%0, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x128xf32>, tensor<1x32x1024x128xf32> -> tensor<1x32x1024x128xf32>
    return %6 : tensor<1x32x1024x128xf32>

    // CHECK: [[RoPE:%.+]] = IE.RoPE([[ARG0]], [[ARG1]], [[ARG2]]) {mode = #IE.rope_mode<SPLIT_HALF>} : tensor<1x32x1024x128xf32>, tensor<1x1x1024x128xf32>, tensor<1x1x1024x128xf32> -> tensor<1x32x1024x128xf32>
    // CHECK: return [[RoPE]] : tensor<1x32x1024x128xf32>

}

// -----

// CHECK-LABEL: @FuseRoPEWithDifferentChannel
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x64x1x64xf32>, [[ARG1:%.+]]: tensor<1x64x1x64xf32>, [[ARG2:%.+]]: tensor<1x64x1x64xf32>)
func.func @FuseRoPEWithDifferentChannel(%arg0: tensor<1x64x1x64xf32>, %arg1: tensor<1x64x1x64xf32>, %arg2: tensor<1x64x1x64xf32>) -> tensor<1x64x1x64xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x1x64xf32>, tensor<1x64x1x64xf32> -> tensor<1x64x1x64xf32>
    %1 = IE.Slice %arg0 [0, 0, 0, 32] [1, 64, 1, 32] : tensor<1x64x1x64xf32> to tensor<1x64x1x32xf32>
    %2 = IE.Multiply(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x1x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x1x32xf32>
    %3 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 1, 32] : tensor<1x64x1x64xf32> to tensor<1x64x1x32xf32>
    %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32]]} : tensor<1x64x1x32xf32>, tensor<1x64x1x32xf32> -> tensor<1x64x1x64xf32>
    %5 = IE.Multiply(%4, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x1x64xf32>, tensor<1x64x1x64xf32> -> tensor<1x64x1x64xf32>
    %6 = IE.Add(%0, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x1x64xf32>, tensor<1x64x1x64xf32> -> tensor<1x64x1x64xf32>
    return %6 : tensor<1x64x1x64xf32>

    // CHECK: [[RoPE:%.+]] = IE.RoPE([[ARG0]], [[ARG1]], [[ARG2]]) {mode = #IE.rope_mode<SPLIT_HALF>} : tensor<1x64x1x64xf32>, tensor<1x64x1x64xf32>, tensor<1x64x1x64xf32> -> tensor<1x64x1x64xf32>
    // CHECK: return [[RoPE]] : tensor<1x64x1x64xf32>
}

// -----

// CHECK-LABEL: @FuseRoPEWithReshape
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x64x1x64xf32>, [[ARG1:%.+]]: tensor<1x64x1x64xf32>, [[ARG2:%.+]]: tensor<1x1x64x64xf32>)
func.func @FuseRoPEWithReshape(%arg0: tensor<1x64x1x64xf32>, %arg1: tensor<1x64x1x64xf32>, %arg2: tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x1x64xf32>, tensor<1x64x1x64xf32> -> tensor<1x64x1x64xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 64, 64]} : tensor<1x64x1x64xf32> -> tensor<1x1x64x64xf32>
    %2 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 64, 64]} : tensor<1x64x1x64xf32> -> tensor<1x1x64x64xf32>
    %3 = IE.Slice %2 [0, 0, 0, 32] [1, 1, 64, 32] : tensor<1x1x64x64xf32> to tensor<1x1x64x32xf32>
    %4 = IE.Multiply(%3, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x32xf32>
    %5 = IE.Slice %2 [0, 0, 0, 0] [1, 1, 64, 32] : tensor<1x1x64x64xf32> to tensor<1x1x64x32xf32>
    %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32]]} : tensor<1x1x64x32xf32>, tensor<1x1x64x32xf32> -> tensor<1x1x64x64xf32>
    %7 = IE.Multiply(%6, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32> -> tensor<1x1x64x64xf32>
    %8 = IE.Add(%1, %7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32> -> tensor<1x1x64x64xf32>
    return %8 : tensor<1x1x64x64xf32>

    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 64, 64]} : tensor<1x64x1x64xf32> -> tensor<1x1x64x64xf32>
    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[ARG1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 64, 64]} : tensor<1x64x1x64xf32> -> tensor<1x1x64x64xf32>
    // CHECK: [[RoPE:%.+]] = IE.RoPE([[RESHAPE0]], [[RESHAPE1]], [[ARG2]]) {mode = #IE.rope_mode<SPLIT_HALF>} : tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32> -> tensor<1x1x64x64xf32>
    // CHECK: return [[RoPE]] : tensor<1x1x64x64xf32>
}

// -----

// CHECK-LABEL: @FuseRoPEWithHeightOne
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x2048xf32>, [[ARG1:%.+]]: tensor<1x1x1x128xf32>, [[ARG2:%.+]]: tensor<1x1x1x128xf32>)
func.func @FuseRoPEWithHeightOne(%arg0: tensor<1x1x2048xf32>, %arg1: tensor<1x1x1x128xf32>, %arg2: tensor<1x1x1x128xf32>) -> tensor<1x16x1x128xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [1, 16, 1, 128]} : tensor<1x1x2048xf32> -> tensor<1x16x1x128xf32>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x128xf32>, tensor<1x1x1x128xf32> -> tensor<1x16x1x128xf32>
    %2 = IE.StridedSlice(%0) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 64], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [1, 16, 1, 128], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<1x16x1x128xf32> -> tensor<1x16x1x64xf32>
    %3 = IE.Multiply(%2, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x1x64xf32>
    %4 = IE.StridedSlice(%0) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [1, 16, 1, 64], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<1x16x1x128xf32> -> tensor<1x16x1x64xf32>
    %5 = IE.Concat(%3, %4) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 64]]} : tensor<1x16x1x64xf32>, tensor<1x16x1x64xf32> -> tensor<1x16x1x128xf32>
    %6 = IE.Multiply(%5, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x128xf32>, tensor<1x1x1x128xf32> -> tensor<1x16x1x128xf32>
    %7 = IE.Add(%1, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x128xf32>, tensor<1x16x1x128xf32> -> tensor<1x16x1x128xf32>
    return %7 : tensor<1x16x1x128xf32>

    // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [1, 16, 1, 128]} : tensor<1x1x2048xf32> -> tensor<1x16x1x128xf32>
    // CHECK: [[ROPE:%.+]] = IE.RoPE([[RESHAPE]], [[ARG1]], [[ARG2]]) {mode = #IE.rope_mode<SPLIT_HALF>} : tensor<1x16x1x128xf32>, tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32> -> tensor<1x16x1x128xf32>
    // CHECK: return [[ROPE]] : tensor<1x16x1x128xf32>
}

// -----

// CHECK-LABEL: @FuseRoPEInterleaved
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x256x64xf32>, [[ARG1:%.+]]: tensor<1x1x256x64xf32>, [[ARG2:%.+]]: tensor<1x1x256x64xf32>)
func.func @FuseRoPEInterleaved(%arg0: tensor<1x1x256x64xf32>, %arg1: tensor<1x1x256x64xf32>, %arg2: tensor<1x1x256x64xf32>) ->tensor<1x1x256x64xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>
  %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x64xf32>, tensor<1x1x256x64xf32> -> tensor<1x1x256x64xf32>
  %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 1, 256, 32, 2]} : tensor<1x1x256x64xf32> -> tensor<1x1x256x32x2xf32>
  %2:2 = IE.Split(%1) {axis_value = 4 : i64, num_splits = 2 : i64} : tensor<1x1x256x32x2xf32> -> tensor<1x1x256x32x1xf32>, tensor<1x1x256x32x1xf32>
  %3 = IE.AffineReshape(%2#1) {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 1, 256, 32]} : tensor<1x1x256x32x1xf32> -> tensor<1x1x256x32xf32>
  %4 = IE.Multiply(%3, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x256x32xf32>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 1, 256, 32, 1]} : tensor<1x1x256x32xf32> -> tensor<1x1x256x32x1xf32>
  %6 = IE.Concat(%5, %2#0) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]} : tensor<1x1x256x32x1xf32>, tensor<1x1x256x32x1xf32> -> tensor<1x1x256x32x2xf32>
  %7 = IE.AffineReshape(%6) {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 1, 256, 64]} : tensor<1x1x256x32x2xf32> -> tensor<1x1x256x64xf32>
  %8 = IE.Multiply(%7, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x64xf32>, tensor<1x1x256x64xf32> -> tensor<1x1x256x64xf32>
  %9 = IE.Add(%0, %8) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x64xf32>, tensor<1x1x256x64xf32> -> tensor<1x1x256x64xf32>
  return %9 : tensor<1x1x256x64xf32>

    // CHECK: [[RoPE:%.+]] = IE.RoPE([[ARG0]], [[ARG1]], [[ARG2]]) {mode = #IE.rope_mode<INTERLEAVED>} : tensor<1x1x256x64xf32>, tensor<1x1x256x64xf32>, tensor<1x1x256x64xf32> -> tensor<1x1x256x64xf32>
    // CHECK: return [[RoPE]] : tensor<1x1x256x64xf32>
}

// -----

// CHECK-LABEL: @FuseRoPEInterleavedWithUnsqueeze
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x16x1024x128xf32>, [[ARG1:%.+]]: tensor<1x1x1024x128xf32>, [[ARG2:%.+]]: tensor<1x1x1024x128xf32>)
func.func @FuseRoPEInterleavedWithUnsqueeze(%arg0: tensor<1x16x1024x128xf32>, %arg1: tensor<1x1x1024x128xf32>, %arg2: tensor<1x1x1024x128xf32>) -> tensor<1x16x1024x128xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>
    %mul_cos = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1024x128xf32>, tensor<1x1x1024x128xf32> -> tensor<1x16x1024x128xf32>
    %slice_odd = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 16, 1024, 128], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x16x1024x128xf32> -> tensor<1x16x1024x64xf32>
    %neg = IE.Multiply(%slice_odd, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1024x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x1024x64xf32>
    %unsq_neg = IE.AffineReshape(%neg) {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 16, 1024, 64, 1]} : tensor<1x16x1024x64xf32> -> tensor<1x16x1024x64x1xf32>
    %slice_even = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 16, 1024, 128], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x16x1024x128xf32> -> tensor<1x16x1024x64xf32>
    %unsq_even = IE.AffineReshape(%slice_even) {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 16, 1024, 64, 1]} : tensor<1x16x1024x64xf32> -> tensor<1x16x1024x64x1xf32>
    %stack = IE.Concat(%unsq_neg, %unsq_even) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]} : tensor<1x16x1024x64x1xf32>, tensor<1x16x1024x64x1xf32> -> tensor<1x16x1024x64x2xf32>
    %flat = IE.AffineReshape(%stack) {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 16, 1024, 128]} : tensor<1x16x1024x64x2xf32> -> tensor<1x16x1024x128xf32>
    %mul_sin = IE.Multiply(%flat, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1024x128xf32>, tensor<1x1x1024x128xf32> -> tensor<1x16x1024x128xf32>
    %add = IE.Add(%mul_cos, %mul_sin) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1024x128xf32>, tensor<1x16x1024x128xf32> -> tensor<1x16x1024x128xf32>
    return %add : tensor<1x16x1024x128xf32>

    // CHECK: [[ROPE:%.+]] = IE.RoPE([[ARG0]], [[ARG1]], [[ARG2]]) {mode = #IE.rope_mode<INTERLEAVED>} : tensor<1x16x1024x128xf32>, tensor<1x1x1024x128xf32>, tensor<1x1x1024x128xf32> -> tensor<1x16x1024x128xf32>
    // CHECK: return [[ROPE]] : tensor<1x16x1024x128xf32>
}

// -----

// CHECK-LABEL: @FuseRoPEPairwiseFusion
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x256x2x256xf32>, [[ARG1:%.+]]: tensor<1x256x1x128xf32>, [[ARG2:%.+]]: tensor<1x256x1x128xf32>)
func.func @FuseRoPEPairwiseFusion(%arg0: tensor<1x256x2x256xf32>, %arg1: tensor<1x256x1x128xf32>, %arg2: tensor<1x256x1x128xf32>) -> tensor<1x256x2x256xf32> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 256, 2, 128] : tensor<1x256x2x256xf32> to tensor<1x256x2x128xf32>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x128xf32>, tensor<1x256x1x128xf32> -> tensor<1x256x2x128xf32>
    %2 = IE.Slice %arg0 [0, 0, 0, 128] [1, 256, 2, 128] : tensor<1x256x2x256xf32> to tensor<1x256x2x128xf32>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x128xf32>, tensor<1x256x1x128xf32> -> tensor<1x256x2x128xf32>
    %4 = IE.Subtract(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x128xf32>, tensor<1x256x2x128xf32> -> tensor<1x256x2x128xf32>
    %5 = IE.Multiply(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x128xf32>, tensor<1x256x1x128xf32> -> tensor<1x256x2x128xf32>
    %6 = IE.Multiply(%2, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x128xf32>, tensor<1x256x1x128xf32> -> tensor<1x256x2x128xf32>
    %7 = IE.Add(%5, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x128xf32>, tensor<1x256x2x128xf32> -> tensor<1x256x2x128xf32>
    %8 = IE.Concat(%4, %7) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 128]]} : tensor<1x256x2x128xf32>, tensor<1x256x2x128xf32> -> tensor<1x256x2x256xf32>
    return %8 : tensor<1x256x2x256xf32>

    // CHECK: [[RoPE:%.+]] = IE.RoPE([[ARG0]], [[ARG1]], [[ARG2]]) {mode = #IE.rope_mode<PAIRWISE>} : tensor<1x256x2x256xf32>, tensor<1x256x1x128xf32>, tensor<1x256x1x128xf32> -> tensor<1x256x2x256xf32>
    // CHECK: return [[RoPE]] : tensor<1x256x2x256xf32>
}

// -----

// Partial-width RoPE: RoPE applies to the first 96 of 128 head dimensions.
// The sin-path slices ([0:48], [48:96]) are taken directly from the full 128-dim
// tensor, so stridedSliceOp1->getOperand(0) has width 128 != cosWidth 96.
// FuseRoPE must fall back to mulOp1.getOperand(0) (= Slice[0:96](Q)) as the
// canonical RoPE input, append the untouched trailing 32 dims via Concat.
//
// CHECK-LABEL: @FuseRoPEPartialWidth
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x4x128xf32>, [[ARG1:%.+]]: tensor<1x1x4x96xf32>, [[ARG2:%.+]]: tensor<1x1x4x96xf32>)
func.func @FuseRoPEPartialWidth(%arg0: tensor<1x8x4x128xf32>, %arg1: tensor<1x1x4x96xf32>, %arg2: tensor<1x1x4x96xf32>) -> tensor<1x8x4x128xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>
    // Cos path: slice the RoPE region [0:96], multiply by cos.
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 4, 96] : tensor<1x8x4x128xf32> to tensor<1x8x4x96xf32>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x4x96xf32>, tensor<1x1x4x96xf32> -> tensor<1x8x4x96xf32>
    // Sin path: both half-slices come directly from the full 128-dim tensor (not from %0).
    %2 = IE.Slice %arg0 [0, 0, 0, 48] [1, 8, 4, 48] : tensor<1x8x4x128xf32> to tensor<1x8x4x48xf32>
    %3 = IE.Multiply(%2, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x4x48xf32>, tensor<1x1x1x1xf32> -> tensor<1x8x4x48xf32>
    %4 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 4, 48] : tensor<1x8x4x128xf32> to tensor<1x8x4x48xf32>
    %5 = IE.Concat(%3, %4) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 48]]} : tensor<1x8x4x48xf32>, tensor<1x8x4x48xf32> -> tensor<1x8x4x96xf32>
    %6 = IE.Multiply(%5, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x4x96xf32>, tensor<1x1x4x96xf32> -> tensor<1x8x4x96xf32>
    %7 = IE.Add(%1, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x4x96xf32>, tensor<1x8x4x96xf32> -> tensor<1x8x4x96xf32>
    // Append the untouched trailing 32 dims to restore the full head dimension.
    %8 = IE.Slice %arg0 [0, 0, 0, 96] [1, 8, 4, 32] : tensor<1x8x4x128xf32> to tensor<1x8x4x32xf32>
    %9 = IE.Concat(%7, %8) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 96]]} : tensor<1x8x4x96xf32>, tensor<1x8x4x32xf32> -> tensor<1x8x4x128xf32>
    return %9 : tensor<1x8x4x128xf32>

    // CHECK: [[ROPE_IN:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 8, 4, 96] : tensor<1x8x4x128xf32> to tensor<1x8x4x96xf32>
    // CHECK: [[ROPE:%.+]] = IE.RoPE([[ROPE_IN]], [[ARG1]], [[ARG2]]) {mode = #IE.rope_mode<SPLIT_HALF>} : tensor<1x8x4x96xf32>, tensor<1x1x4x96xf32>, tensor<1x1x4x96xf32> -> tensor<1x8x4x96xf32>
    // CHECK: [[TAIL:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 96] [1, 8, 4, 32] : tensor<1x8x4x128xf32> to tensor<1x8x4x32xf32>
    // CHECK: [[RESULT:%.+]] = IE.Concat([[ROPE]], [[TAIL]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 96]]} : tensor<1x8x4x96xf32>, tensor<1x8x4x32xf32> -> tensor<1x8x4x128xf32>
    // CHECK: return [[RESULT]] : tensor<1x8x4x128xf32>
}

// -----

// CHECK-LABEL: @FuseRoPEPairwiseInterleavedFusion
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x256x2x256xf32>, [[ARG1:%.+]]: tensor<1x256x1x128xf32>, [[ARG2:%.+]]: tensor<1x256x1x128xf32>)
func.func @FuseRoPEPairwiseInterleavedFusion(%arg0: tensor<1x256x2x256xf32>, %arg1: tensor<1x256x1x128xf32>, %arg2: tensor<1x256x1x128xf32>) -> tensor<1x256x2x256xf32> {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [1, 256, 2, 256], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 2]} : tensor<1x256x2x256xf32> -> tensor<1x256x2x128xf32>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x128xf32>, tensor<1x256x1x128xf32> -> tensor<1x256x2x128xf32>
    %2 = IE.StridedSlice(%arg0) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [1, 256, 2, 256], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 2]} : tensor<1x256x2x256xf32> -> tensor<1x256x2x128xf32>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x128xf32>, tensor<1x256x1x128xf32> -> tensor<1x256x2x128xf32>
    %4 = IE.Subtract(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x128xf32>, tensor<1x256x2x128xf32> -> tensor<1x256x2x128xf32>
    %5 = IE.Multiply(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x128xf32>, tensor<1x256x1x128xf32> -> tensor<1x256x2x128xf32>
    %6 = IE.Multiply(%2, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x128xf32>, tensor<1x256x1x128xf32> -> tensor<1x256x2x128xf32>
    %7 = IE.Add(%5, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x128xf32>, tensor<1x256x2x128xf32> -> tensor<1x256x2x128xf32>
    %8 = IE.Reshape(%4) {shape_value = [1, 256, 2, 128, 1]} : tensor<1x256x2x128xf32> -> tensor<1x256x2x128x1xf32>
    %9 = IE.Reshape(%7) {shape_value = [1, 256, 2, 128, 1]} : tensor<1x256x2x128xf32> -> tensor<1x256x2x128x1xf32>
    %10 = IE.Concat(%8, %9) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]} : tensor<1x256x2x128x1xf32>, tensor<1x256x2x128x1xf32> -> tensor<1x256x2x128x2xf32>
    %11 = IE.Reshape(%10) {shape_value = [1, 256, 2, 256]} : tensor<1x256x2x128x2xf32> -> tensor<1x256x2x256xf32>
    return %11 : tensor<1x256x2x256xf32>

    // CHECK: [[RoPE:%.+]] = IE.RoPE([[ARG0]], [[ARG1]], [[ARG2]]) {mode = #IE.rope_mode<PAIRWISE_INTERLEAVED>} : tensor<1x256x2x256xf32>, tensor<1x256x1x128xf32>, tensor<1x256x1x128xf32> -> tensor<1x256x2x256xf32>
    // CHECK: return [[RoPE]] : tensor<1x256x2x256xf32>
}

// -----

// CHECK-LABEL: @FuseRoPEPairwiseWithChannelSlice
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x60x8x128xf32>, [[ARG1:%.+]]: tensor<1x10x1x64xf32>, [[ARG2:%.+]]: tensor<1x10x1x64xf32>)
func.func @FuseRoPEPairwiseWithChannelSlice(%arg0: tensor<1x60x8x128xf32>, %arg1: tensor<1x10x1x64xf32>, %arg2: tensor<1x10x1x64xf32>) -> tensor<1x10x8x128xf32> {
    %0 = IE.Slice %arg0 [0, 40, 0,  0] [1, 10, 8, 64] : tensor<1x60x8x128xf32> to tensor<1x10x8x64xf32>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x8x64xf32>, tensor<1x10x1x64xf32> -> tensor<1x10x8x64xf32>
    %2 = IE.Slice %arg0 [0, 40, 0, 64] [1, 10, 8, 64] : tensor<1x60x8x128xf32> to tensor<1x10x8x64xf32>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x8x64xf32>, tensor<1x10x1x64xf32> -> tensor<1x10x8x64xf32>
    %4 = IE.Subtract(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x8x64xf32>, tensor<1x10x8x64xf32> -> tensor<1x10x8x64xf32>
    %5 = IE.Multiply(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x8x64xf32>, tensor<1x10x1x64xf32> -> tensor<1x10x8x64xf32>
    %6 = IE.Multiply(%2, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x8x64xf32>, tensor<1x10x1x64xf32> -> tensor<1x10x8x64xf32>
    %7 = IE.Add(%5, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x8x64xf32>, tensor<1x10x8x64xf32> -> tensor<1x10x8x64xf32>
    %8 = IE.Concat(%4, %7) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 64]]} : tensor<1x10x8x64xf32>, tensor<1x10x8x64xf32> -> tensor<1x10x8x128xf32>
    return %8 : tensor<1x10x8x128xf32>

    // CHECK: [[MERGED:%.+]] = IE.Slice [[ARG0]] [0, 40, 0, 0] [1, 10, 8, 128] : tensor<1x60x8x128xf32> to tensor<1x10x8x128xf32>
    // CHECK: [[ROPE:%.+]] = IE.RoPE([[MERGED]], [[ARG1]], [[ARG2]]) {mode = #IE.rope_mode<PAIRWISE>} : tensor<1x10x8x128xf32>, tensor<1x10x1x64xf32>, tensor<1x10x1x64xf32> -> tensor<1x10x8x128xf32>
    // CHECK: return [[ROPE]] : tensor<1x10x8x128xf32>
}
