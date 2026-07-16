//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-mem-permute-to-op --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @BigMemPermute
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x16x19320xf16>
func.func @BigMemPermute(%arg0: tensor<1x4x16x19320xf16>) -> tensor<1x16x4x19320xf16, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCWH} :
        tensor<1x4x16x19320xf16> -> tensor<1x16x4x19320xf16, {order = #NHWC}>

    return %MEM_PERMUTE : tensor<1x16x4x19320xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x4x16x19320xf16> -> tensor<1x16x4x19320xf16, {order = #NHWC}>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x16x4x19320xf16, {order = #NHWC}>
}
