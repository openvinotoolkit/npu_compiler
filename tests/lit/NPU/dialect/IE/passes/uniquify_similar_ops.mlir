//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --uniquify-similar-ops %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @UniquifyMemPermute
func.func @UniquifyMemPermute(%arg0: tensor<1x16x2x3xf16>) ->
        (tensor<1x3x16x2xf16>, tensor<1x3x16x2xf16>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x16x2x3xf16> -> tensor<1x3x16x2xf16>
    %1 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x16x2x3xf16> -> tensor<1x3x16x2xf16>

    return %0, %1 : tensor<1x3x16x2xf16>, tensor<1x3x16x2xf16>

    // CHECK:     [[PERMUTE:%.+]] = IE.MemPermute({{[^:]+}})
    // CHECK-NOT: IE.MemPermute
    // CHECK:     return [[PERMUTE]], [[PERMUTE]] : tensor<1x3x16x2xf16>, tensor<1x3x16x2xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL:   @UniquifyMemPermuteForTheSameMergedPermutation
func.func @UniquifyMemPermuteForTheSameMergedPermutation(%arg0: tensor<1x1x512x1500xf16, {order = #NHWC}>) ->
        (tensor<1x1x1500x512xf16>, tensor<1x1x1500x512xf16, {order = #NHWC}>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} :
        tensor<1x1x512x1500xf16, {order = #NHWC}> -> tensor<1x1x1500x512xf16>

    %1 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHCW} :
        tensor<1x1x512x1500xf16, {order = #NHWC}> -> tensor<1x1x1500x512xf16, {order = #NHWC}>

    return %0, %1 : tensor<1x1x1500x512xf16>, tensor<1x1x1500x512xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE:%.+]] = IE.MemPermute({{[^:]+}})
    // CHECK-SAME:  {dst_order = #NCHW, mem_perm = #NWHC} :
    // CHECK-SAME:  tensor<1x1x512x1500xf16, {order = #NHWC}> -> tensor<1x1x1500x512xf16>

    // CHECK:   [[PERMUTECAST:%.+]] = IE.PermuteCast([[PERMUTE]])
    // CHECK-SAME:  {dst_order = #NHWC, mem_perm = #NHWC} :
    // CHECK-SAME:  tensor<1x1x1500x512xf16> -> tensor<1x1x1500x512xf16, {order = #NHWC}>

    // CHECK:     return [[PERMUTE]], [[PERMUTECAST]] : tensor<1x1x1500x512xf16>, tensor<1x1x1500x512xf16, {order = #NHWC}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NDHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>
#perm1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>
#perm2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK:  #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>

// CHECK-LABEL:   @UniquifyMemPermuteForTheSame3DMergedPermutation
func.func @UniquifyMemPermuteForTheSame3DMergedPermutation(%arg0: tensor<1x2x3x4x1xf16, {order = #NCDHW}>) ->
        (tensor<1x2x4x3x1xf16>, tensor<1x3x2x4x1xf16, {order = #NDHWC}>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCDHW, mem_perm = #perm1} :
        tensor<1x2x3x4x1xf16, {order = #NCDHW}> -> tensor<1x2x4x3x1xf16>

    %1 = IE.MemPermute(%arg0) {dst_order = #NDHWC, mem_perm = #perm2} :
        tensor<1x2x3x4x1xf16, {order = #NCDHW}> -> tensor<1x3x2x4x1xf16, {order = #NDHWC}>

    return %0, %1 : tensor<1x2x4x3x1xf16>, tensor<1x3x2x4x1xf16, {order = #NDHWC}>

    // CHECK:   [[PERMUTE:%.+]] = IE.MemPermute({{[^:]+}})
    // CHECK-SAME:  {dst_order = #NCDHW, mem_perm = #map} :
    // CHECK-SAME:  tensor<1x2x3x4x1xf16, {order = #NCDHW}> -> tensor<1x2x4x3x1xf16>

    // CHECK:   [[PERMUTECAST:%.+]] = IE.PermuteCast([[PERMUTE]])
    // CHECK-SAME:  {dst_order = #NDHWC, mem_perm = #map1} :
    // CHECK-SAME:  tensor<1x2x4x3x1xf16> -> tensor<1x3x2x4x1xf16, {order = #NDHWC}>

    // CHECK:     return [[PERMUTE]], [[PERMUTECAST]] : tensor<1x2x4x3x1xf16>, tensor<1x3x2x4x1xf16, {order = #NDHWC}>
}
