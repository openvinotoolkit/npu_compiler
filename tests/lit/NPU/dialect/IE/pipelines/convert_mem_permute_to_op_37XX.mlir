//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-mem-permute-to-op --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @BigMemPermute
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x16x19320xf16>
func.func @BigMemPermute(%arg0: tensor<1x4x16x19320xf16>) -> tensor<1x16x4x19320xf16, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCWH} :
        tensor<1x4x16x19320xf16> -> tensor<1x16x4x19320xf16, {order = #NHWC}>

    return %MEM_PERMUTE : tensor<1x16x4x19320xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 8]} : tensor<1x4x16x19320xf16> -> tensor<1x4x16x19328xf16>
    // CHECK:       [[PERMUTE_CAST_IN:%.+]] = IE.PermuteCast([[EXPAND]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x16x19328xf16> -> tensor<1x19328x4x16xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTE_CAST_IN]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x19328x4x16xf16, {order = #NHWC}> -> tensor<1x19328x4x16xf16, {order = #NHCW}>
    // CHECK:       [[PERMUTE_CAST_OUT:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x19328x4x16xf16, {order = #NHCW}> -> tensor<1x16x4x19328xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[PERMUTE_CAST_OUT]] [0, 0, 0, 0] [1, 16, 4, 19320] : tensor<1x16x4x19328xf16, {order = #NHWC}> to tensor<1x16x4x19320xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE]] : tensor<1x16x4x19320xf16, {order = #NHWC}>
}
