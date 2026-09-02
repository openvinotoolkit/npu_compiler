//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-tile-op %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @NotFuseGroupConvWithBiasAddDynamicFilterUnsupportedPlatform
func.func @NotFuseGroupConvWithBiasAddDynamicFilterUnsupportedPlatform(%input: tensor<1x4x4x4xf16>, %filter: tensor<4x1x1x1xf16>) -> tensor<1x4x4x4xf16> {
    %bias = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01> : tensor<1x4x1x1xf16>
    %0 = IE.GroupConvolution(%input, %filter) {
            dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x4x4x4xf16>, tensor<4x1x1x1xf16> -> tensor<1x4x4x4xf16>
    %1 = IE.Tile(%bias) {repeats_values = [1, 1, 4, 4]} : tensor<1x4x1x1xf16> -> tensor<1x4x4x4xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x4x4xf16>, tensor<1x4x4x4xf16> -> tensor<1x4x4x4xf16>
    return %2 : tensor<1x4x4x4xf16>

    // Dynamic (non-const) filter with a static bias, on a platform that does not support the new
    // weight-table bias format for dynamic-weight GroupConv (see
    // MPEEngineConfig::isNewWeightTableFormatSupportedWithDwOps): FuseGroupConvWithBiasAdd must not
    // fuse; Tile and Add remain.
    // CHECK:       IE.GroupConvolution
    // CHECK:       IE.Tile
    // CHECK:       IE.Add
}
