//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% num-of-dpu-groups=1" --fuse-scale --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @DoNotFuseRoPEQueryScaleOnSingleTile
func.func @DoNotFuseRoPEQueryScaleOnSingleTile(%arg0: tensor<1x2x4x3xf16>, %cos: tensor<1x1x4x3xf16>, %sin: tensor<1x1x4x3xf16>,
                                               %w0: tensor<4x3x1x1xf16>, %w1: tensor<4x3x1x1xf16>)
        -> (tensor<4x4x1x1xf16>, tensor<4x4x1x1xf16>) {
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<0.135327876> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    %rope = IE.RoPE(%arg0, %cos, %sin) {mode = #IE.rope_mode<SPLIT_HALF>}
        : tensor<1x2x4x3xf16>, tensor<1x1x4x3xf16>, tensor<1x1x4x3xf16> -> tensor<1x2x4x3xf16>

    %mul = IE.Multiply(%rope, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x2x4x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x4x3xf16>

    %slice0 = IE.Slice %mul [0, 0, 0, 0] [1, 1, 4, 3] : tensor<1x2x4x3xf16> to tensor<1x1x4x3xf16>
    %slice1 = IE.Slice %mul [0, 1, 0, 0] [1, 1, 4, 3] : tensor<1x2x4x3xf16> to tensor<1x1x4x3xf16>

    %reshape0 = IE.AffineReshape(%slice0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 3, 1, 1]}
        : tensor<1x1x4x3xf16> -> tensor<4x3x1x1xf16>
    %reshape1 = IE.AffineReshape(%slice1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 3, 1, 1]}
        : tensor<1x1x4x3xf16> -> tensor<4x3x1x1xf16>

    %conv0 = IE.Convolution(%reshape0, %w0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<4x3x1x1xf16>, tensor<4x3x1x1xf16> -> tensor<4x4x1x1xf16>
    %conv1 = IE.Convolution(%reshape1, %w1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<4x3x1x1xf16>, tensor<4x3x1x1xf16> -> tensor<4x4x1x1xf16>

    return %conv0, %conv1 : tensor<4x4x1x1xf16>, tensor<4x4x1x1xf16>

    // CHECK:     IE.RoPE
    // CHECK:     IE.Multiply
    // CHECK:     IE.Convolution
    // CHECK-NOT: static_scale
    // CHECK:     IE.Convolution
    // CHECK-NOT: static_scale
    // CHECK:     return
}
