//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
 
//CHECK: func.func @DepthToSpaceDynamicHeightAndWidth([[ARG0:%.*]]: tensor<1x144x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 144, 3, 3]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 9, 9]> : tensor<4xsi64>, order = #NHWC}> {
func.func @DepthToSpaceDynamicHeightAndWidth(%input: tensor<1x144x?x?xf16, {bounds = #const.OpaqueI64Elements<[1,144,3,3]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1,16,9,9]> : tensor<4xsi64>, order = #NCHW}> {
    %d2s = IE.DepthToSpace(%input) {
        block_size = 3 : i64,
        mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    } : tensor<1x144x?x?xf16, {bounds = #const.OpaqueI64Elements<[1,144,3,3]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1,16,9,9]> : tensor<4xsi64>, order = #NCHW}>

    return %d2s : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1,16,9,9]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK: [[VAL0:%.*]] = IE.DepthToSpace([[ARG0]]) {block_size = 3 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x144x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 144, 3, 3]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 9, 9]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: return [[VAL0]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 9, 9]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
 
//CHECK: func.func @DepthToSpaceStaticHeightAndWidth([[ARG0:%.*]]: tensor<1x144x3x3xf16, {order = #NHWC}>) -> tensor<1x16x9x9xf16, {order = #NHWC}> {
func.func @DepthToSpaceStaticHeightAndWidth(%input: tensor<1x144x3x3xf16, {order = #NCHW}>) -> tensor<1x16x9x9xf16, {order = #NCHW}> {
    %d2s = IE.DepthToSpace(%input) {
        block_size = 3 : i64,
        mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    } : tensor<1x144x3x3xf16, {order = #NCHW}> -> tensor<1x16x9x9xf16, {order = #NCHW}>

    return %d2s : tensor<1x16x9x9xf16, {order = #NCHW}>
    //CHECK: [[VAL0:%.*]] = IE.DepthToSpace([[ARG0]]) {block_size = 3 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x144x3x3xf16, {order = #NHWC}> -> tensor<1x16x9x9xf16, {order = #NHWC}>
    //CHECK: return [[VAL0]] : tensor<1x16x9x9xf16, {order = #NHWC}>
}
