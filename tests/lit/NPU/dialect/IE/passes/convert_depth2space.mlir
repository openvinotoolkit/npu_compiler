//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-depthToSpace %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @Depth2SpaceConvertToNNDMAs
func.func @Depth2SpaceConvertToNNDMAs(%arg0: tensor<1x4x512x8xf16>) -> tensor<1x1x1024x16xf16> {
    %0 = IE.DepthToSpace(%arg0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x512x8xf16> -> tensor<1x1x1024x16xf16>

    return %0 : tensor<1x1x1024x16xf16>

    // CHECK-NOT:   IE.DepthToSpace
    // CHECK:       [[VAL0:%.*]] = IE.DepthToSpace(%arg0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x512x8xf16> -> tensor<1x1x1024x16xf16>
    // CHECK:       return [[VAL0]] : tensor<1x1x1024x16xf16>
}
