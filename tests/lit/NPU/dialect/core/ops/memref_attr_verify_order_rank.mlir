//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Positive: order rank matches shape rank.
func.func @OrderRankMatchesShape(%arg0: memref<1x2x3x4xf32, {order = #NHWC}>)
        -> memref<1x2x3x4xf32, {order = #NHWC}> {
    return %arg0 : memref<1x2x3x4xf32, {order = #NHWC}>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// Negative: 3-D (CHW) order used with a 4-D shape.
// expected-error@+1 {{Dims order 'affine_map<(d0, d1, d2) -> (d0, d1, d2)>' rank does not match shape '[4, 2, 3, 4]' rank}}
func.func @OrderRankSmallerThanShape(%arg0: memref<4x2x3x4xf32, {order = #CHW}>)
        -> memref<4x2x3x4xf32, {order = #CHW}> {
    return %arg0 : memref<4x2x3x4xf32, {order = #CHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Negative: 4-D (NCHW) order used with a 3-D shape.
// expected-error@+1 {{Dims order 'affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>' rank does not match shape '[2, 3, 4]' rank}}
func.func @OrderRankLargerThanShape(%arg0: memref<2x3x4xf32, {order = #NCHW}>)
        -> memref<2x3x4xf32, {order = #NCHW}> {
    return %arg0 : memref<2x3x4xf32, {order = #NCHW}>
}
