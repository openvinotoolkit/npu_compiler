//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-spaceToDepth %s | FileCheck %s
// REQUIRES: arch-NPU37XX
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

// CHECK-LABEL: @convertSpaceToDepth_BLOCKS_FIRST
func.func @convertSpaceToDepth_BLOCKS_FIRST(%arg0: tensor<1x3x520x520xf16>) -> tensor<1x202800x2x2xf16> {
    %0 = IE.SpaceToDepthOp(%arg0) {block_size = 260 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x3x520x520xf16> -> tensor<1x202800x2x2xf16>

    return %0 : tensor<1x202800x2x2xf16>

    //CHECK-NOT: IE.SpaceToDepthOp
    //CHECK: [[RESHAPE0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 3, 2, 260, 2, 260]} : tensor<1x3x520x520xf16> -> tensor<1x3x2x260x2x260xf16>
    //CHECK: [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE0]]) {order_value = #map} : tensor<1x3x2x260x2x260xf16> -> tensor<1x260x260x3x2x2xf16>
    //CHECK: [[RESHAPE1:%.*]] = IE.Reshape([[TRANSPOSE]]) {shape_value = [1, 202800, 2, 2]} : tensor<1x260x260x3x2x2xf16> -> tensor<1x202800x2x2xf16>
    //CHECK: return [[RESHAPE1]] : tensor<1x202800x2x2xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5, d2, d4)>

// CHECK-LABEL: @convertSpaceToDepth_DEPTH_FIRST
func.func @convertSpaceToDepth_DEPTH_FIRST(%arg0: tensor<1x3x520x520xf16>) -> tensor<1x202800x2x2xf16> {
    %0 = IE.SpaceToDepthOp(%arg0) {block_size = 260 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x3x520x520xf16> -> tensor<1x202800x2x2xf16>

    return %0 : tensor<1x202800x2x2xf16>

    //CHECK-NOT: IE.SpaceToDepthOp
    //CHECK: [[RESHAPE0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 3, 2, 260, 2, 260]} : tensor<1x3x520x520xf16> -> tensor<1x3x2x260x2x260xf16>
    //CHECK: [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE0]]) {order_value = #map} : tensor<1x3x2x260x2x260xf16> -> tensor<1x3x260x260x2x2xf16>
    //CHECK: [[RESHAPE1:%.*]] = IE.Reshape([[TRANSPOSE]]) {shape_value = [1, 202800, 2, 2]} : tensor<1x3x260x260x2x2xf16> -> tensor<1x202800x2x2xf16>
    //CHECK: return [[RESHAPE1]] : tensor<1x202800x2x2xf16>
}
