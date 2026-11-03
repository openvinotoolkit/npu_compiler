//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% allow-custom-values=true power-optimized=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --incremental-pipeline="vertical-fusion=true workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DepthToSpaceConvDepthToSpace
func.func @DepthToSpaceConvDepthToSpace(%input: tensor<1x128x400x640xf16, {order = #NHWC}>)
        -> tensor<1x8x1600x2560xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %d2s0 = VPU.DepthToSpace(%input) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
        : tensor<1x128x400x640xf16, {order = #NHWC}> -> tensor<1x32x800x1280xf16, {order = #NHWC}>
    %conv = VPU.NCE.Convolution(%d2s0, %weights) rawFilterShape [32, 32, 3, 3] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]
    } : tensor<1x32x800x1280xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
        -> tensor<1x32x800x1280xf16, {order = #NHWC}>
    %d2s1 = VPU.DepthToSpace(%conv) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
        : tensor<1x32x800x1280xf16, {order = #NHWC}> -> tensor<1x8x1600x2560xf16, {order = #NHWC}>
    return %d2s1 : tensor<1x8x1600x2560xf16, {order = #NHWC}>

    // The identity pool separates the incompatible D2S alignment and convolution halo requirements.
    // All ops will be fused into one VF region.
    // CHECK:      VPU.DepthToSpace
    // CHECK-SAME: loop_tile_coord = {axes = array<i64: 2, 3>, indices = array<i64: 0, 0>}
    // CHECK-SAME: vf_loop_index = 0 : i64
    // CHECK-SAME: vf_loop_tile_index = 0 : i64
    // CHECK:      VPU.NCE.AveragePool
    // CHECK-SAME: kernel_size = [1, 1]
    // CHECK-SAME: loop_tile_coord = {axes = array<i64: 2, 3>, indices = array<i64: 0, 0>}
    // CHECK-SAME: strides = [1, 1]
    // CHECK-SAME: vf_loop_index = 0 : i64
    // CHECK-SAME: vf_loop_tile_index = 0 : i64
    // CHECK:      VPU.NCE.Convolution
    // CHECK-SAME: rawFilterShape [32, 32, 3, 3]
    // CHECK-SAME: loop_tile_coord = {axes = array<i64: 2, 3>, indices = array<i64: 0, 0>}
    // CHECK-SAME: vf_loop_index = 0 : i64
    // CHECK-SAME: vf_loop_tile_index = 0 : i64
    // CHECK:      VPU.DepthToSpace
    // CHECK-SAME: loop_tile_coord = {axes = array<i64: 2, 3>, indices = array<i64: 0, 0>}
    // CHECK-SAME: vf_loop_index = 0 : i64
    // CHECK-SAME: vf_loop_tile_index = 0 : i64
}
