//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-spaceToDepth %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

// Don't convert to reshape -> transpose -> reshape pattern if can convert to DMA or DPU instead
// CHECK-LABEL: @noConvertSpaceToDepth_BLOCKS_FIRST
func.func @noConvertSpaceToDepth_BLOCKS_FIRST(%arg0: tensor<1x3x512x512xf16>) -> tensor<1x48x128x128xf16> {
    %0 = IE.SpaceToDepthOp(%arg0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x3x512x512xf16> -> tensor<1x48x128x128xf16>

    return %0 : tensor<1x48x128x128xf16>

    //CHECK: [[SPACETODEPTH:%.*]] = IE.SpaceToDepthOp(%arg0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x3x512x512xf16> -> tensor<1x48x128x128xf16>
    //CHECK: return [[SPACETODEPTH]] : tensor<1x48x128x128xf16>
}
// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5, d2, d4)>

// CHECK-LABEL: @noConvertSpaceToDepth_DEPTH_FIRST
func.func @noConvertSpaceToDepth_DEPTH_FIRST(%arg0: tensor<1x3x512x512xf16>) -> tensor<1x48x128x128xf16> {
    %0 = IE.SpaceToDepthOp(%arg0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x3x512x512xf16> -> tensor<1x48x128x128xf16>

    return %0 : tensor<1x48x128x128xf16>

    //CHECK: [[SPACETODEPTH:%.*]] = IE.SpaceToDepthOp(%arg0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x3x512x512xf16> -> tensor<1x48x128x128xf16>
    //CHECK: return [[SPACETODEPTH]] : tensor<1x48x128x128xf16>
}
// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

// CHECK-LABEL: @noConvertSpaceToDepth_BLOCKS_FIRST
func.func @noConvertSpaceToDepth_BLOCKS_FIRST(%arg0: tensor<1x3x520x520xf16>) -> tensor<1x202800x2x2xf16> {
    %0 = IE.SpaceToDepthOp(%arg0) {block_size = 260 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x3x520x520xf16> -> tensor<1x202800x2x2xf16>

    return %0 : tensor<1x202800x2x2xf16>

    //CHECK: [[SPACETODEPTH:%.*]] = IE.SpaceToDepthOp(%arg0) {block_size = 260 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x3x520x520xf16> -> tensor<1x202800x2x2xf16>
    //CHECK: return [[SPACETODEPTH]] : tensor<1x202800x2x2xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5, d2, d4)>

// CHECK-LABEL: @noConvertSpaceToDepth_DEPTH_FIRST
func.func @noConvertSpaceToDepth_DEPTH_FIRST(%arg0: tensor<1x3x520x520xf16>) -> tensor<1x202800x2x2xf16> {
    %0 = IE.SpaceToDepthOp(%arg0) {block_size = 260 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x3x520x520xf16> -> tensor<1x202800x2x2xf16>

    return %0 : tensor<1x202800x2x2xf16>

    //CHECK: [[SPACETODEPTH:%.*]] = IE.SpaceToDepthOp(%arg0) {block_size = 260 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x3x520x520xf16> -> tensor<1x202800x2x2xf16>
    //CHECK: return [[SPACETODEPTH]] : tensor<1x202800x2x2xf16>
}
