//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tile-copies %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @LegalizeCopy(
        %arg0: memref<1x256x2048x8192xf16, #NCHW>,
        %arg1: memref<1x256x2048x8192xf16, #NCHW>)
        -> memref<1x256x2048x8192xf16, #NCHW> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x256x2048x8192xf16, #NCHW>)
                   outputs(%arg1 : memref<1x256x2048x8192xf16, #NCHW>)
                   -> memref<1x256x2048x8192xf16, #NCHW>

    return %0 : memref<1x256x2048x8192xf16, #NCHW>

    // Currently, large Copy nodes are tiled C-wise

    // CHECK: [[SUBVIEW_SRC_0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 127, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16>
    // CHECK-SAME:   to memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK: [[SUBVIEW_DST_0:%.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 127, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16>
    // CHECK-SAME:   to memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK: [[COPY_RET_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_0]] : memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_0]] : memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:        -> memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>

    // CHECK: [[SUBVIEW_SRC_1:%.*]] = VPUIP.SubView %arg0 [0, 127, 0, 0] [1, 127, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16>
    // CHECK-SAME:   to memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK: [[SUBVIEW_DST_1:%.*]] = VPUIP.SubView %arg1 [0, 127, 0, 0] [1, 127, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16>
    // CHECK-SAME:   to memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK: [[COPY_RET_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_1]] : memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_1]] : memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:        -> memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>

    // CHECK: [[SUBVIEW_SRC_2:%.*]] = VPUIP.SubView %arg0 [0, 254, 0, 0] [1, 1, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16>
    // CHECK-SAME:   to memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK: [[SUBVIEW_DST_2:%.*]] = VPUIP.SubView %arg1 [0, 254, 0, 0] [1, 1, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16>
    // CHECK-SAME:   to memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK: [[COPY_RET_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_2]] : memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_2]] : memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:        -> memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>

    // CHECK: [[SUBVIEW_SRC_3:%.*]] = VPUIP.SubView %arg0 [0, 255, 0, 0] [1, 1, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16>
    // CHECK-SAME:   to memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK: [[SUBVIEW_DST_3:%.*]] = VPUIP.SubView %arg1 [0, 255, 0, 0] [1, 1, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16>
    // CHECK-SAME:   to memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK: [[COPY_RET_3:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_3]] : memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_3]] : memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:        -> memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>

    // CHECK: [[CONCAT_VIEW_0:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_RET_0]], [[COPY_RET_1]], [[COPY_RET_2]], [[COPY_RET_3]] :
    // CHECK-SAME:        memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK-SAME:        memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK-SAME:        memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK-SAME:        memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x256x2048x8192xf16>)
    // CHECK-SAME:        -> memref<1x256x2048x8192xf16>

    // CHECK: return [[CONCAT_VIEW_0]] : memref<1x256x2048x8192xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @LegalizeStridedCopy(
        %arg0: memref<1x256x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>,
        %arg1: memref<1x256x2048x8192xf16, #NCHW>)
        -> memref<1x256x2048x8192xf16, #NCHW> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x256x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>)
                   outputs(%arg1 : memref<1x256x2048x8192xf16, #NCHW>)
                   -> memref<1x256x2048x8192xf16, #NCHW>

    return %0 : memref<1x256x2048x8192xf16, #NCHW>

    // Currently, large Copy nodes are tiled C-wise
    // If the Copy is strided, the strides should be preserved

    // CHECK: [[SUBVIEW_SRC_0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 127, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>
    // CHECK-SAME:   to memref<1x127x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>
    // CHECK: [[SUBVIEW_DST_0:%.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 127, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16>
    // CHECK-SAME:   to memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK: [[COPY_RET_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_0]] : memref<1x127x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_0]] : memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:        -> memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>

    // CHECK: [[SUBVIEW_SRC_1:%.*]] = VPUIP.SubView %arg0 [0, 127, 0, 0] [1, 127, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>
    // CHECK-SAME:   to memref<1x127x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>
    // CHECK: [[SUBVIEW_DST_1:%.*]] = VPUIP.SubView %arg1 [0, 127, 0, 0] [1, 127, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16>
    // CHECK-SAME:   to memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK: [[COPY_RET_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_1]] : memref<1x127x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_1]] : memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:        -> memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>

    // CHECK: [[SUBVIEW_SRC_2:%.*]] = VPUIP.SubView %arg0 [0, 254, 0, 0] [1, 1, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>
    // CHECK-SAME:   to memref<1x1x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>
    // CHECK: [[SUBVIEW_DST_2:%.*]] = VPUIP.SubView %arg1 [0, 254, 0, 0] [1, 1, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16>
    // CHECK-SAME:   to memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK: [[COPY_RET_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_2]] : memref<1x1x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_2]] : memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:        -> memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>

    // CHECK: [[SUBVIEW_SRC_3:%.*]] = VPUIP.SubView %arg0 [0, 255, 0, 0] [1, 1, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>
    // CHECK-SAME:   to memref<1x1x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>
    // CHECK: [[SUBVIEW_DST_3:%.*]] = VPUIP.SubView %arg1 [0, 255, 0, 0] [1, 1, 2048, 8192] :
    // CHECK-SAME:      memref<1x256x2048x8192xf16>
    // CHECK-SAME:   to memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK: [[COPY_RET_3:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_3]] : memref<1x1x2048x8192xf16, {order = #NCHW, strides = [8589934592, 16777216, 8192, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_3]] : memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:        -> memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>

    // CHECK: [[CONCAT_VIEW_0:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_RET_0]], [[COPY_RET_1]], [[COPY_RET_2]], [[COPY_RET_3]] :
    // CHECK-SAME:        memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK-SAME:        memref<1x127x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK-SAME:        memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>
    // CHECK-SAME:        memref<1x1x2048x8192xf16, {order = #NCHW, strides = [4294967296, 16777216, 8192, 1]}>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x256x2048x8192xf16>)
    // CHECK-SAME:        -> memref<1x256x2048x8192xf16>

    // CHECK: return [[CONCAT_VIEW_0]] : memref<1x256x2048x8192xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @DoNotLegalizeCopy(
        %arg0: memref<1x127x2048x8192xf16, #NCHW>,
        %arg1: memref<1x127x2048x8192xf16, #NCHW>)
        -> memref<1x127x2048x8192xf16, #NCHW> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x127x2048x8192xf16, #NCHW>)
                   outputs(%arg1 : memref<1x127x2048x8192xf16, #NCHW>)
                   -> memref<1x127x2048x8192xf16, #NCHW>

    return %0 : memref<1x127x2048x8192xf16, #NCHW>

    // Small enough Copy nodes should not be affected by the pass

    // CHECK: [[COPY_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x127x2048x8192xf16>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x127x2048x8192xf16>)
    // CHECK-SAME:        -> memref<1x127x2048x8192xf16>
    // CHECK: return [[COPY_0]] : memref<1x127x2048x8192xf16>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// [17179869184, 33554432, 1]
func.func @SplitByHeight3D(
        %arg0: memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>,
        %arg1: memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>)
        -> memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>)
                   outputs(%arg1 : memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>)
                   -> memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>

    return %0 : memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0] [1, 64, 16777216] :
    // CHECK-SAME:              memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>
    // CHECK-SAME:           to memref<1x64x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0] [1, 64, 16777216] :
    // CHECK-SAME:              memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>
    // CHECK-SAME:           to memref<1x64x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x64x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x64x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>)
    // CHECK-SAME:  -> memref<1x64x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 64, 0] [1, 64, 16777216] :
    // CHECK-SAME:              memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>
    // CHECK-SAME:           to memref<1x64x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 64, 0] [1, 64, 16777216] :
    // CHECK-SAME:              memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>
    // CHECK-SAME:           to memref<1x64x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x64x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x64x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>)
    // CHECK-SAME:  -> memref<1x64x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>

    // CHECK: %[[CONCAT_VIEW_0:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x64x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>,
    // CHECK-SAME:      memref<1x64x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>)
    // CHECK-SAME:  -> memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>

    // CHECK: return %[[CONCAT_VIEW_0]] : memref<1x128x16777216xf16, {order = #CHW, strides = [8589934592, 33554432, 1]}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @SplitByDepth5D(
        %arg0: memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>,
        %arg1: memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>)
        -> memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>)
                   outputs(%arg1 : memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>)
                   -> memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>

    return %0 : memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0, 0] [1, 1, 2048, 256, 2048] :
    // CHECK-SAME:              memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>
    // CHECK-SAME:           to memref<1x1x2048x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0, 0] [1, 1, 2048, 256, 2048] :
    // CHECK-SAME:              memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>
    // CHECK-SAME:           to memref<1x1x2048x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x1x2048x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x1x2048x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>)
    // CHECK-SAME:  -> memref<1x1x2048x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 0, 2048, 0, 0] [1, 1, 2048, 256, 2048] :
    // CHECK-SAME:              memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>
    // CHECK-SAME:           to memref<1x1x2048x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 0, 2048, 0, 0] [1, 1, 2048, 256, 2048] :
    // CHECK-SAME:              memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>
    // CHECK-SAME:           to memref<1x1x2048x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x1x2048x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x1x2048x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>)
    // CHECK-SAME:  -> memref<1x1x2048x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>

    // CHECK: %[[CONCAT_VIEW_0:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x1x2048x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>,
    // CHECK-SAME:      memref<1x1x2048x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>)
    // CHECK-SAME:  -> memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>

    // CHECK: return %[[CONCAT_VIEW_0]] : memref<1x1x4096x256x2048xf16, {order = #NCDHW, strides = [4294967296, 4294967296, 1048576, 4096, 1]}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @DoNotSplit5DCopyWithLevel3Striding(
        %arg0: memref<1x1x4095x256x2048xf16, {order = #NCDHW, strides = [17175674880, 17175674880, 2097152, 4096, 1]}>,
        %arg1: memref<1x1x4095x256x2048xf16, {order = #NCDHW, strides = [17175674880, 17175674880, 2097152, 4096, 1]}>)
        -> memref<1x1x4095x256x2048xf16, {order = #NCDHW, strides = [17175674880, 17175674880, 2097152, 4096, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x1x4095x256x2048xf16, {order = #NCDHW, strides = [17175674880, 17175674880, 2097152, 4096, 1]}>)
                   outputs(%arg1 : memref<1x1x4095x256x2048xf16, {order = #NCDHW, strides = [17175674880, 17175674880, 2097152, 4096, 1]}>)
                   -> memref<1x1x4095x256x2048xf16, {order = #NCDHW, strides = [17175674880, 17175674880, 2097152, 4096, 1]}>

    return %0 : memref<1x1x4095x256x2048xf16, {order = #NCDHW, strides = [17175674880, 17175674880, 2097152, 4096, 1]}>

    // Level 3 striding Copy is supported on LNL and should not be affected by the pass

    // CHECK: [[COPY_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x1x4095x256x2048xf16, {order = #NCDHW, strides = [17175674880, 17175674880, 2097152, 4096, 1]}>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x1x4095x256x2048xf16, {order = #NCDHW, strides = [17175674880, 17175674880, 2097152, 4096, 1]}>)
    // CHECK-SAME:        -> memref<1x1x4095x256x2048xf16, {order = #NCDHW, strides = [17175674880, 17175674880, 2097152, 4096, 1]}>

    // CHECK: return [[COPY_0]] : memref<1x1x4095x256x2048xf16, {order = #NCDHW, strides = [17175674880, 17175674880, 2097152, 4096, 1]}>
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputType = memref<3x512x2048x8192x!quant.uniform<u8:f16, 1.2608227898092832:124>, #NHWC, @DDR>
!OutputType = memref<3x512x2048x8192x!quant.uniform<u8:f16, 1.2608227898092832:124>, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

func.func @RecursiveSplitLargeCopy(
        %arg0: !InputType,
        %arg1: !OutputType)
        -> !OutputType {

    // CHECK-LABEL: @RecursiveSplitLargeCopy
    %0 = VPUIP.Copy inputs(%arg0 : !InputType)
                   outputs(%arg1 : !OutputType)
                   -> !OutputType

    return %0 : !OutputType
    // First split across N axis
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 512, 2048, 8192]
    // CHECK-SAME:         memref<3x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 512, 2048, 8192]
    // CHECK-SAME:         memref<3x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // Next 4 splits across H axis
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 0] [1, 512, 1023, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>
    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 0, 0] [1, 512, 1023, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_3]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 1023, 0] [1, 512, 1023, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>
    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 1023, 0] [1, 512, 1023, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_5]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_6:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 2046, 0] [1, 512, 1, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>
    // CHECK:       [[SUBVIEW_7:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 2046, 0] [1, 512, 1, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_6]] : memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_7]] : memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_8:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 2047, 0] [1, 512, 1, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>
    // CHECK:       [[SUBVIEW_9:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 2047, 0] [1, 512, 1, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_8]] : memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_9]] : memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // Concat of splits across H axis for first N split
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]], [[SUBVIEW_1]]0, [[SUBVIEW_1]]3 : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>, memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>, memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>, memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_1]] : memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>


    // CHECK:       [[SUBVIEW_10:%.+]] = VPUIP.SubView %arg0 [1, 0, 0, 0] [1, 512, 2048, 8192]
    // CHECK-SAME:         memref<3x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_11:%.+]] = VPUIP.SubView %arg1 [1, 0, 0, 0] [1, 512, 2048, 8192]
    // CHECK-SAME:         memref<3x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_12:%.+]] = VPUIP.SubView [[SUBVIEW_10]] [0, 0, 0, 0] [1, 512, 1023, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>
    // CHECK:       [[SUBVIEW_13:%.+]] = VPUIP.SubView [[SUBVIEW_11]] [0, 0, 0, 0] [1, 512, 1023, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK:       [[COPY_4:%.+]] = VPUIP.Copy inputs([[SUBVIEW_12]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_13]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_14:%.+]] = VPUIP.SubView [[SUBVIEW_10]] [0, 0, 1023, 0] [1, 512, 1023, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>
    // CHECK:       [[SUBVIEW_15:%.+]] = VPUIP.SubView [[SUBVIEW_11]] [0, 0, 1023, 0] [1, 512, 1023, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK:       [[COPY_5:%.+]] = VPUIP.Copy inputs([[SUBVIEW_14]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_15]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_16:%.+]] = VPUIP.SubView [[SUBVIEW_10]] [0, 0, 2046, 0] [1, 512, 1, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>
    // CHECK:       [[SUBVIEW_17:%.+]] = VPUIP.SubView [[SUBVIEW_11]] [0, 0, 2046, 0] [1, 512, 1, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK:       [[COPY_6:%.+]] = VPUIP.Copy inputs([[SUBVIEW_16]] : memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_17]] : memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_18:%.+]] = VPUIP.SubView [[SUBVIEW_10]] [0, 0, 2047, 0] [1, 512, 1, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>
    // CHECK:       [[SUBVIEW_19:%.+]] = VPUIP.SubView [[SUBVIEW_11]] [0, 0, 2047, 0] [1, 512, 1, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK:       [[COPY_7:%.+]] = VPUIP.Copy inputs([[SUBVIEW_18]] : memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_19]] : memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_4]], [[COPY_5]], [[COPY_6]], [[COPY_7]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>, memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>, memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>, memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_11]] : memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>


    // CHECK:       [[SUBVIEW_20:%.+]] = VPUIP.SubView %arg0 [2, 0, 0, 0] [1, 512, 2048, 8192] 
    // CHECK-SAME:         memref<3x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_21:%.+]] = VPUIP.SubView %arg1 [2, 0, 0, 0] [1, 512, 2048, 8192] 
    // CHECK-SAME:         memref<3x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_22:%.+]] = VPUIP.SubView [[SUBVIEW_20]] [0, 0, 0, 0] [1, 512, 1023, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>
    // CHECK:       [[SUBVIEW_23:%.+]] = VPUIP.SubView [[SUBVIEW_21]] [0, 0, 0, 0] [1, 512, 1023, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK:       [[COPY_8:%.+]] = VPUIP.Copy inputs([[SUBVIEW_22]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_23]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_24:%.+]] = VPUIP.SubView [[SUBVIEW_20]] [0, 0, 1023, 0] [1, 512, 1023, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>
    // CHECK:       [[SUBVIEW_25:%.+]] = VPUIP.SubView [[SUBVIEW_21]] [0, 0, 1023, 0] [1, 512, 1023, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK:       [[COPY_9:%.+]] = VPUIP.Copy inputs([[SUBVIEW_24]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_25]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_26:%.+]] = VPUIP.SubView [[SUBVIEW_20]] [0, 0, 2046, 0] [1, 512, 1, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>
    // CHECK:       [[SUBVIEW_27:%.+]] = VPUIP.SubView [[SUBVIEW_21]] [0, 0, 2046, 0] [1, 512, 1, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK:       [[COPY_10:%.+]] = VPUIP.Copy inputs([[SUBVIEW_26]] : memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_27]] : memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_28:%.+]] = VPUIP.SubView [[SUBVIEW_20]] [0, 0, 2047, 0] [1, 512, 1, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:      to memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>
    // CHECK:       [[SUBVIEW_29:%.+]] = VPUIP.SubView [[SUBVIEW_21]] [0, 0, 2047, 0] [1, 512, 1, 8192]
    // CHECK-SAME:         memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK-SAME:      to memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK:       [[COPY_11:%.+]] = VPUIP.Copy inputs([[SUBVIEW_28]] : memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [8589934592, 1, 4194304, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_29]] : memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[CONCATVIEW_2:%.+]] = VPUIP.ConcatView inputs([[COPY_8]], [[COPY_9]], [[COPY_10]], [[COPY_11]] : memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>, memref<1x512x1023x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>, memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>, memref<1x512x1x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_21]] : memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>

    // CHECK:       [[CONCATVIEW_3:%.+]] = VPUIP.ConcatView inputs([[CONCATVIEW_0]], [[CONCATVIEW_1]], [[CONCATVIEW_2]] : memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>, memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>, memref<1x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:         outputs(%arg1 : memref<3x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<3x512x2048x8192x!qElemType, {order = #NHWC, strides = [34359738368, 1, 16777216, 512]}, @DDR>
    // CHECK:       return [[CONCATVIEW_3]]
}
