//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
