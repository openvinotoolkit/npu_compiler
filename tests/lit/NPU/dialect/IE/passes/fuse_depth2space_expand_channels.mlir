//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --fuse-depth-to-space-expand-channels %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

func.func @FuseDepth2SpaceExpandChannels(%arg0: tensor<1x4x512x8xf16>) -> tensor<1x16x1024x16xf16> {
    // CHECK-LABEL: @FuseDepth2SpaceExpandChannels
    // CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x4x512x8xf16>)
    %0 = IE.DepthToSpace(%arg0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x512x8xf16> -> tensor<1x1x1024x16xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1024x16xf16> -> tensor<1x16x1024x16xf16>

    return %1 : tensor<1x16x1024x16xf16>

    // CHECK:           [[VAL0:%.*]] = IE.DepthToSpace([[INPUT]])
    // CHECK-SAME:      {block_size = 2 : i64,
    // CHECK-SAME:      mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    // CHECK-SAME:      padded_channels = #IE.ChannelPadding<input = 0 : i64, output = 15 : i64>
    // CHECK-NOT:       IE.Expand
    // CHECK:           return [[VAL0]] : tensor<1x16x1024x16xf16>
}

// -----

func.func @NotFuseDepth2SpaceExpandChannels(%arg0: tensor<1x4x512x8xf16>) -> tensor<1x32x1024x16xf16> {
    // CHECK-LABEL: @NotFuseDepth2SpaceExpandChannels
    // CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x4x512x8xf16>)
    %0 = IE.DepthToSpace(%arg0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, padded_channels = #IE.ChannelPadding<input = 0 : i64, output = 15 : i64>} : tensor<1x4x512x8xf16> -> tensor<1x16x1024x16xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 16, 0, 0]} : tensor<1x16x1024x16xf16> -> tensor<1x32x1024x16xf16>

    return %1 : tensor<1x32x1024x16xf16>

    // CHECK:           [[VAL0:%.*]] = IE.DepthToSpace([[INPUT]])
    // CHECK:           [[VAL1:%.*]] = IE.Expand([[VAL0]])
    // CHECK:           return [[VAL1]] : tensor<1x32x1024x16xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NotFuseDepth2SpaceExpandChannelsMultipleUsers(%arg0: tensor<1x4x512x8xf16>) -> (tensor<1x16x1024x16xf16>, tensor<1x1x1024x16xf16, {order = #NHWC}>) {
    // CHECK-LABEL: @NotFuseDepth2SpaceExpandChannelsMultipleUsers
    // CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x4x512x8xf16>)
    %0 = IE.DepthToSpace(%arg0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x512x8xf16> -> tensor<1x1x1024x16xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1024x16xf16> -> tensor<1x16x1024x16xf16>
    %2 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x1x1024x16xf16> -> tensor<1x1x1024x16xf16, {order = #NHWC}>
    return %1, %2 : tensor<1x16x1024x16xf16>, tensor<1x1x1024x16xf16, {order = #NHWC}>

    // CHECK:           [[VAL0:%.*]] = IE.DepthToSpace([[INPUT]])
    // CHECK:           [[VAL1:%.*]] = IE.Expand([[VAL0]])
    // CHECK:           [[VAL2:%.*]] = IE.Reorder([[VAL0]])
    // CHECK:           return [[VAL1]], [[VAL2]] : tensor<1x16x1024x16xf16>, tensor<1x1x1024x16xf16, {order = #NHWC}>
}

// -----

func.func @NotFuseDepth2SpaceExpandNotChannels(%arg0: tensor<1x4x512x8xf16>) -> tensor<1x1x1024x32xf16> {
    // CHECK-LABEL: @NotFuseDepth2SpaceExpandNotChannels
    // CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x4x512x8xf16>)
    %0 = IE.DepthToSpace(%arg0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x512x8xf16> -> tensor<1x1x1024x16xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 16]} : tensor<1x1x1024x16xf16> -> tensor<1x1x1024x32xf16>

    return %1 : tensor<1x1x1024x32xf16>

    // CHECK:           [[VAL0:%.*]] = IE.DepthToSpace([[INPUT]])
    // CHECK:           [[VAL1:%.*]] = IE.Expand([[VAL0]])
    // CHECK:           return [[VAL1]] : tensor<1x1x1024x32xf16>
}
