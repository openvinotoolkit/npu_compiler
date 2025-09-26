//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-rope --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @FuseRoPE
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x1024x128xf32>, [[ARG1:%.+]]: tensor<1x1x1024x128xf32>, [[ARG2:%.+]]: tensor<1x1x1024x128xf32>)
func.func @FuseRoPE(%arg0: tensor<1x32x1024x128xf32>, %arg1: tensor<1x1x1024x128xf32>, %arg2: tensor<1x1x1024x128xf32>) -> tensor<1x32x1024x128xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x128xf32>, tensor<1x1x1024x128xf32> -> tensor<1x32x1024x128xf32>
    %1 = IE.StridedSlice(%arg0) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 64], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [1, 32, 1024, 128], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<1x32x1024x128xf32> -> tensor<1x32x1024x64xf32>
    %2 = IE.Multiply(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x64xf32>
    %3 = IE.StridedSlice(%arg0) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [1, 32, 1024, 64], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<1x32x1024x128xf32> -> tensor<1x32x1024x64xf32>
    %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 64]]} : tensor<1x32x1024x64xf32>, tensor<1x32x1024x64xf32> -> tensor<1x32x1024x128xf32>
    %5 = IE.Multiply(%4, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x128xf32>, tensor<1x1x1024x128xf32> -> tensor<1x32x1024x128xf32>
    %6 = IE.Add(%0, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x128xf32>, tensor<1x32x1024x128xf32> -> tensor<1x32x1024x128xf32>
    return %6 : tensor<1x32x1024x128xf32>

    // CHECK: [[RoPE:%.+]] = IE.RoPE([[ARG0]], [[ARG1]], [[ARG2]]) : tensor<1x32x1024x128xf32>, tensor<1x1x1024x128xf32>, tensor<1x1x1024x128xf32> -> tensor<1x32x1024x128xf32>
    // CHECK: return [[RoPE]] : tensor<1x32x1024x128xf32>

}

// CHECK-LABEL: @FuseRoPEWithDifferentChannel
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x64x1x64xf32>, [[ARG1:%.+]]: tensor<1x64x1x64xf32>, [[ARG2:%.+]]: tensor<1x64x1x64xf32>)
func.func @FuseRoPEWithDifferentChannel(%arg0: tensor<1x64x1x64xf32>, %arg1: tensor<1x64x1x64xf32>, %arg2: tensor<1x64x1x64xf32>) -> tensor<1x64x1x64xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x1x64xf32>, tensor<1x64x1x64xf32> -> tensor<1x64x1x64xf32>
    %1 = IE.Slice %arg0 [0, 0, 0, 32] [1, 64, 1, 32] : tensor<1x64x1x64xf32> to tensor<1x64x1x32xf32>
    %2 = IE.Multiply(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x1x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x1x32xf32>
    %3 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 1, 32] : tensor<1x64x1x64xf32> to tensor<1x64x1x32xf32>
    %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32]]} : tensor<1x64x1x32xf32>, tensor<1x64x1x32xf32> -> tensor<1x64x1x64xf32>
    %5 = IE.Multiply(%4, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x1x64xf32>, tensor<1x64x1x64xf32> -> tensor<1x64x1x64xf32>
    %6 = IE.Add(%0, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x1x64xf32>, tensor<1x64x1x64xf32> -> tensor<1x64x1x64xf32>
    return %6 : tensor<1x64x1x64xf32>

    // CHECK: [[RoPE:%.+]] = IE.RoPE([[ARG0]], [[ARG1]], [[ARG2]]) : tensor<1x64x1x64xf32>, tensor<1x64x1x64xf32>, tensor<1x64x1x64xf32> -> tensor<1x64x1x64xf32>
    // CHECK: return [[RoPE]] : tensor<1x64x1x64xf32>
}

// CHECK-LABEL: @FuseRoPEWithReshape
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x64x1x64xf32>, [[ARG1:%.+]]: tensor<1x64x1x64xf32>, [[ARG2:%.+]]: tensor<1x1x64x64xf32>)
func.func @FuseRoPEWithReshape(%arg0: tensor<1x64x1x64xf32>, %arg1: tensor<1x64x1x64xf32>, %arg2: tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32> isSplat
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
    // CHECK: [[RoPE:%.+]] = IE.RoPE([[RESHAPE0]], [[RESHAPE1]], [[ARG2]]) : tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32> -> tensor<1x1x64x64xf32>
    // CHECK: return [[RoPE]] : tensor<1x1x64x64xf32>
}

// CHECK-LABEL: @FuseRoPEWithHeightOne
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x2048xf32>, [[ARG1:%.+]]: tensor<1x1x1x128xf32>, [[ARG2:%.+]]: tensor<1x1x1x128xf32>)
func.func @FuseRoPEWithHeightOne(%arg0: tensor<1x1x2048xf32>, %arg1: tensor<1x1x1x128xf32>, %arg2: tensor<1x1x1x128xf32>) -> tensor<1x16x1x128xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32> isSplat
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
    // CHECK: [[ROPE:%.+]] = IE.RoPE([[RESHAPE]], [[ARG1]], [[ARG2]]) : tensor<1x16x1x128xf32>, tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32> -> tensor<1x16x1x128xf32>
    // CHECK: return [[ROPE]] : tensor<1x16x1x128xf32>
}
