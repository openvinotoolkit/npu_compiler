//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-reduce-ops-with-mem-permute="enable-fuse-reduce-minmax-to-dpu=true" %s | FileCheck %s
// REQUIRES: platform-NPU6010

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotInsertMemPermuteForReduceMaxWithDPUParentSameLayout
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x8x8x8xf16>)
func.func @DoNotInsertMemPermuteForReduceMaxWithDPUParentSameLayout(%arg0: tensor<1x8x8x8xf16>) -> tensor<1x1x8x8xf16> {
    %weights = const.Declare tensor<8x8x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<8x8x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
                : tensor<1x8x8x8xf16>, tensor<8x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16>
    %0 = IE.ReduceMax(%conv) {axes_value = [1], keep_dims} : tensor<1x8x8x8xf16> -> tensor<1x1x8x8xf16>
    return %0 : tensor<1x1x8x8xf16>

    // DPU parent has the same input/output layout (both NCHW); the channel-axis fusion
    // opportunity is preserved, so no MemPermute is inserted even though axis 1 is not
    // the innermost memory dimension in NCHW.
    // CHECK:     [[CONV:%.+]] = IE.Convolution
    // CHECK-NOT: IE.MemPermute
    // CHECK:     IE.ReduceMax([[CONV]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotInsertMemPermuteForReduceMinWithDPUParentSameLayout
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x8x8x8xf16>)
func.func @DoNotInsertMemPermuteForReduceMinWithDPUParentSameLayout(%arg0: tensor<1x8x8x8xf16>) -> tensor<1x1x8x8xf16> {
    %weights = const.Declare tensor<8x8x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<8x8x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
                : tensor<1x8x8x8xf16>, tensor<8x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16>
    %0 = IE.ReduceMin(%conv) {axes_value = [1], keep_dims} : tensor<1x8x8x8xf16> -> tensor<1x1x8x8xf16>
    return %0 : tensor<1x1x8x8xf16>

    // DPU parent has the same input/output layout (both NCHW); the channel-axis fusion
    // opportunity is preserved, so no MemPermute is inserted even though axis 1 is not
    // the innermost memory dimension in NCHW.
    // CHECK:     [[CONV:%.+]] = IE.Convolution
    // CHECK-NOT: IE.MemPermute
    // CHECK:     IE.ReduceMin([[CONV]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @InsertMemPermuteForReduceMaxWithDPUParentPermutingOutput
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x8x8x8xf16, {order = #NHWC}>)
func.func @InsertMemPermuteForReduceMaxWithDPUParentPermutingOutput(%arg0: tensor<1x8x8x8xf16, {order = #NHWC}>) -> tensor<1x1x8x8xf16> {
    %weights = const.Declare tensor<8x8x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<8x8x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
                : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<8x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16>
    %0 = IE.ReduceMax(%conv) {axes_value = [1], keep_dims} : tensor<1x8x8x8xf16> -> tensor<1x1x8x8xf16>
    return %0 : tensor<1x1x8x8xf16>

    // DPU parent produces NCHW output from NHWC input (permuting layout).
    // isChannelAxisReductionWithDPUParent returns false; axis 1 is not innermost in NCHW,
    // so MemPermute is inserted to move the reduce axis to the innermost memory dimension.
    // CHECK:     [[CONV:%.+]] = IE.Convolution
    // CHECK:     [[IN_PERM:%.+]] = IE.MemPermute([[CONV]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:     [[REDUCE:%.+]] = IE.ReduceMax([[IN_PERM]]) {axes_value = [1], keep_dims} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x1x8x8xf16, {order = #NHWC}>
    // CHECK:     IE.MemPermute([[REDUCE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x8x8xf16, {order = #NHWC}> -> tensor<1x1x8x8xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @InsertMemPermuteForReduceMinWithDPUParentPermutingOutput
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x8x8x8xf16, {order = #NHWC}>)
func.func @InsertMemPermuteForReduceMinWithDPUParentPermutingOutput(%arg0: tensor<1x8x8x8xf16, {order = #NHWC}>) -> tensor<1x1x8x8xf16> {
    %weights = const.Declare tensor<8x8x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<8x8x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
                : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<8x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16>
    %0 = IE.ReduceMin(%conv) {axes_value = [1], keep_dims} : tensor<1x8x8x8xf16> -> tensor<1x1x8x8xf16>
    return %0 : tensor<1x1x8x8xf16>

    // DPU parent produces NCHW output from NHWC input (permuting layout).
    // isChannelAxisReductionWithDPUParent returns false; axis 1 is not innermost in NCHW,
    // so MemPermute is inserted to move the reduce axis to the innermost memory dimension.
    // CHECK:     [[CONV:%.+]] = IE.Convolution
    // CHECK:     [[IN_PERM:%.+]] = IE.MemPermute([[CONV]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:     [[REDUCE:%.+]] = IE.ReduceMin([[IN_PERM]]) {axes_value = [1], keep_dims} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x1x8x8xf16, {order = #NHWC}>
    // CHECK:     IE.MemPermute([[REDUCE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x8x8xf16, {order = #NHWC}> -> tensor<1x1x8x8xf16>
}
