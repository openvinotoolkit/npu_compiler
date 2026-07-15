//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --tile-copies %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @SplitDoubleStrideCopyByChannels
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>, [[ARG_1:%[^:]+]]: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
func.func @SplitDoubleStrideCopyByChannels(
        %arg0: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
        %arg1: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
        -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
                   outputs(%arg1 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
                   -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    return %0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: [[ARG_0_TILE_0:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 160, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: [[ARG_1_TILE_0:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 0, 0] [1, 160, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: [[COPY_TILE_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_0]] : memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_0]] : memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: [[ARG_0_TILE_1:%.+]] = VPUIP.SubView [[ARG_0]] [0, 160, 0, 0] [1, 160, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: [[ARG_1_TILE_1:%.+]] = VPUIP.SubView [[ARG_1]] [0, 160, 0, 0] [1, 160, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: [[COPY_TILE_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_1]] : memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_1]] : memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[COPY_TILE_0]], [[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:      memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1]] : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: return [[CONCAT]] : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @SplitOutputDoubleStrideCopyByChannels
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>, [[ARG_1:%[^:]+]]: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
func.func @SplitOutputDoubleStrideCopyByChannels(
        %arg0: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>,
        %arg1: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
        -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>)
                   outputs(%arg1 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
                   -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    return %0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: [[ARG_0_TILE_0:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 160, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>
    // CHECK-SAME:           to memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>

    // CHECK: [[ARG_1_TILE_0:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 0, 0] [1, 160, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: [[COPY_TILE_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_0]] : memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_0]] : memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: [[ARG_0_TILE_1:%.+]] = VPUIP.SubView [[ARG_0]] [0, 160, 0, 0] [1, 160, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>
    // CHECK-SAME:           to memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>

    // CHECK: [[ARG_1_TILE_1:%.+]] = VPUIP.SubView [[ARG_1]] [0, 160, 0, 0] [1, 160, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: [[COPY_TILE_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_1]] : memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_1]] : memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[COPY_TILE_0]], [[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:      memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1]] : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: return [[CONCAT]] : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @LegalizeCopy
// CHECK-SAME:      ([[ARG_0:%[^:]+]]: memref<1x64x512x512xf16>,
// CHECK-SAME:      [[ARG_1:%[^:]+]]: memref<1x64x512x512xf16>)
func.func @LegalizeCopy(
        %arg0: memref<1x64x512x512xf16>,
        %arg1: memref<1x64x512x512xf16>)
        -> memref<1x64x512x512xf16> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x64x512x512xf16>)
                   outputs(%arg1 : memref<1x64x512x512xf16>)
                   -> memref<1x64x512x512xf16>

    return %0 : memref<1x64x512x512xf16>

    // Currently, large Copy nodes are tiled C-wise

    // Cut first tile:
    // CHECK: [[SUBVIEW_SRC_1:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_1:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[COPY_RET_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_1]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_1]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        -> memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the second tile:
    // CHECK: [[SUBVIEW_SRC_2:%.+]] = VPUIP.SubView [[ARG_0]] [0, 31, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_2:%.+]] = VPUIP.SubView [[ARG_1]] [0, 31, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[COPY_RET_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_2]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_2]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        -> memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the third tile:
    // CHECK: [[SUBVIEW_SRC_3:%.+]] = VPUIP.SubView [[ARG_0]] [0, 62, 0, 0] [1, 1, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_3:%.+]] = VPUIP.SubView [[ARG_1]] [0, 62, 0, 0] [1, 1, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[COPY_RET_3:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_3]] : memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_3]] : memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:          memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the 4th tile:
    // CHECK: [[SUBVIEW_SRC_4:%.+]] = VPUIP.SubView [[ARG_0]] [0, 63, 0, 0] [1, 1, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_4:%.+]] = VPUIP.SubView [[ARG_1]] [0, 63, 0, 0] [1, 1, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[COPY_RET_4:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_4]] : memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_4]] : memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:          memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Concatenate the resulting output tiles:
    // CHECK: [[VAR6:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_RET_1]], [[COPY_RET_2]], [[COPY_RET_3]], [[COPY_RET_4]] :
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[ARG_1]] : memref<1x64x512x512xf16>)
    // CHECK-SAME:        -> memref<1x64x512x512xf16>
    // CHECK: return [[VAR6]] : memref<1x64x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @LegalizeStridedCopy
// CHECK-SAME:      ([[ARG_0:%[^:]+]]: memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>,
// CHECK-SAME:      [[ARG_1:%[^:]+]]: memref<1x64x512x512xf16>)
func.func @LegalizeStridedCopy(
        %arg0: memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>,
        %arg1: memref<1x64x512x512xf16>)
        -> memref<1x64x512x512xf16> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
                   outputs(%arg1 : memref<1x64x512x512xf16>)
                   -> memref<1x64x512x512xf16>

    return %0 : memref<1x64x512x512xf16>

    // Currently, large Copy nodes are tiled C-wise
    // If the Copy is strided, the strides should be preserved

    // Cut the first tile:
    // CHECK: [[SUBVIEW_SRC_1:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK-SAME:      memref<1x31x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_1:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // The Copy-tile preserves the original strides:
    // CHECK: [[COPY_RET_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_1]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_1]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the second tile:
    // CHECK: [[SUBVIEW_SRC_2:%.+]] = VPUIP.SubView [[ARG_0]] [0, 31, 0, 0] [1, 31, 512, 512]
    // CHECK-SAME:      memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK-SAME:      memref<1x31x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_2:%.+]] = VPUIP.SubView [[ARG_1]] [0, 31, 0, 0] [1, 31, 512, 512]
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // The Copy-tile preserves the original strides:
    // CHECK: [[COPY_RET_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_2]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_2]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the third tile:
    // CHECK: [[SUBVIEW_SRC_3:%.+]] = VPUIP.SubView [[ARG_0]] [0, 62, 0, 0] [1, 1, 512, 512]
    // CHECK-SAME:          memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK-SAME:          memref<1x1x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_3:%.+]] = VPUIP.SubView [[ARG_1]] [0, 62, 0, 0] [1, 1, 512, 512]
    // CHECK-SAME:          memref<1x64x512x512xf16>
    // CHECK-SAME:          memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // CHECK: [[COPY_RET_3:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[SUBVIEW_SRC_3]] : memref<1x1x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
    // CHECK-SAME:          outputs([[SUBVIEW_DST_3]] : memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:            memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the 4th tile:
    // CHECK: [[SUBVIEW_SRC_4:%.+]] = VPUIP.SubView [[ARG_0]] [0, 63, 0, 0] [1, 1, 512, 512]
    // CHECK-SAME:          memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK-SAME:          memref<1x1x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_4:%.+]] = VPUIP.SubView [[ARG_1]] [0, 63, 0, 0] [1, 1, 512, 512]
    // CHECK-SAME:          memref<1x64x512x512xf16>
    // CHECK-SAME:          memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // CHECK: [[COPY_RET_4:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[SUBVIEW_SRC_4]] : memref<1x1x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
    // CHECK-SAME:          outputs([[SUBVIEW_DST_4]] : memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:            memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>


    // Concatenate the resulting output tiles:
    // CHECK: [[RESULT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_RET_1]], [[COPY_RET_2]], [[COPY_RET_3]], [[COPY_RET_4]] :
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x1x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[ARG_1]] : memref<1x64x512x512xf16>)
    // CHECK-SAME:        memref<1x64x512x512xf16>

    // CHECK: return [[RESULT]] : memref<1x64x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @DoNotLegalizeCopy
// CHECK-SAME:      ([[ARG_0:%[^:]+]]: memref<1x315x221x241xi8>,
// CHECK-SAME:      [[ARG_1:%[^:]+]]: memref<1x315x221x241xi8>)
func.func @DoNotLegalizeCopy(
        %arg0: memref<1x315x221x241xi8>,
        %arg1: memref<1x315x221x241xi8>)
        -> memref<1x315x221x241xi8> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x315x221x241xi8>)
                   outputs(%arg1 : memref<1x315x221x241xi8>)
                   -> memref<1x315x221x241xi8>

    return %0 : memref<1x315x221x241xi8>

    // Small enough Copy nodes (those with transaction volume less than (16MB - 1Byte)) should not be affected by the pass

    // CHECK: [[VAR0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG_0]] : memref<1x315x221x241xi8>)
    // CHECK-SAME:      outputs([[ARG_1]] : memref<1x315x221x241xi8>)
    // CHECK-SAME:        -> memref<1x315x221x241xi8>
    // CHECK: return [[VAR0]] : memref<1x315x221x241xi8>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @SplitDoubleStrideCopy
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>, [[ARG_1:%[^:]+]]: memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
func.func  @SplitDoubleStrideCopy(
        %arg0: memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>,
        %arg1: memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
        -> memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
                   outputs(%arg1 : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
                   -> memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    return %0 : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK: [[ARG_0_TILE_0:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 128, 16, 1] :
    // CHECK-SAME:              memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK-SAME:           to memref<1x128x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: [[ARG_1_TILE_0:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 0, 0] [1, 128, 16, 1] :
    // CHECK-SAME:              memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK-SAME:           to memref<1x128x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: [[COPY_TILE_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_0]] : memref<1x128x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_0]] : memref<1x128x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x128x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: [[ARG_0_TILE_1:%.+]] = VPUIP.SubView [[ARG_0]] [0, 128, 0, 0] [1, 128, 16, 1] :
    // CHECK-SAME:              memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK-SAME:           to memref<1x128x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: [[ARG_1_TILE_1:%.+]] = VPUIP.SubView [[ARG_1]] [0, 128, 0, 0] [1, 128, 16, 1] :
    // CHECK-SAME:              memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK-SAME:           to memref<1x128x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: [[COPY_TILE_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_1]] : memref<1x128x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_1]] : memref<1x128x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x128x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[COPY_TILE_0]], [[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x128x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>,
    // CHECK-SAME:      memref<1x128x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1]] : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: return [[CONCAT]] : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!SparseType =  !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
                                       sparsity_map=memref<1x320x32x16xi1, {order = #NCHW, strides = [655360, 2048, 32, 1]}>>

// CHECK-LABEL: func.func @SplitByChannelsSparseBuffers
// CHECK-SAME:      ([[ARG_0:%[^:]+]]: !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16,
// CHECK-SAME:      [[ARG_1:%[^:]+]]: !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16,
func.func @SplitByChannelsSparseBuffers(%arg0: !SparseType, %arg1: !SparseType) -> !SparseType {
    %0 = VPUIP.Copy inputs(%arg0 : !SparseType) outputs(%arg1 : !SparseType)-> !SparseType
    return %0 : !SparseType

    // CHECK:       [[ARG_0_TILE_0:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 160, 32, 16] :
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x320x32x16xi1, {order = #NCHW, strides = [655360, 2048, 32, 1]}>> to
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x160x32x16xi1, {order = #NCHW, strides = [655360, 2048, 32, 1]}>>

    // CHECK:       [[ARG_1_TILE_0:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 0, 0] [1, 160, 32, 16] :
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x320x32x16xi1, {order = #NCHW, strides = [655360, 2048, 32, 1]}>> to
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x160x32x16xi1, {order = #NCHW, strides = [655360, 2048, 32, 1]}>>

    // CHECK:       [[COPY_TILE_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG_0_TILE_0]]
    // CHECK-SAME:      outputs([[ARG_1_TILE_0]]

    // CHECK:       [[ARG_0_TILE_1:%.+]] = VPUIP.SubView [[ARG_0]] [0, 160, 0, 0] [1, 160, 32, 16] :
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x320x32x16xi1, {order = #NCHW, strides = [655360, 2048, 32, 1]}>> to
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x160x32x16xi1, {order = #NCHW, strides = [655360, 2048, 32, 1]}>>

    // CHECK:       [[ARG_1_TILE_1:%.+]] = VPUIP.SubView [[ARG_1]] [0, 160, 0, 0] [1, 160, 32, 16] :
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x320x32x16xi1, {order = #NCHW, strides = [655360, 2048, 32, 1]}>> to
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x160x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x160x32x16xi1, {order = #NCHW, strides = [655360, 2048, 32, 1]}>>

    // CHECK:       [[COPY_TILE_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG_0_TILE_1]]

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_TILE_0]], [[COPY_TILE_1]]
    // CHECK-SAME:      outputs([[ARG_1]]
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x320x32x16xi1, {order = #NCHW, strides = [655360, 2048, 32, 1]}>>

    // CHECK:       return [[CONCAT]]
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @SplitByHeight3D
// CHECK-SAME:      ([[ARG_0:%[^:]+]]: memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>,
// CHECK-SAME:      [[ARG_1:%[^:]+]]: memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
func.func @SplitByHeight3D(
        %arg0: memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>,
        %arg1: memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
        -> memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
                   outputs(%arg1 : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
                   -> memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    return %0 : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: [[ARG_0_TILE_0:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0] [1, 130, 64000] :
    // CHECK-SAME:              memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
    // CHECK-SAME:           to memref<1x130x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: [[ARG_1_TILE_0:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 0] [1, 130, 64000] :
    // CHECK-SAME:              memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
    // CHECK-SAME:           to memref<1x130x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: [[COPY_TILE_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_0]] : memref<1x130x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_0]] : memref<1x130x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  -> memref<1x130x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: [[ARG_0_TILE_1:%.+]] = VPUIP.SubView [[ARG_0]] [0, 130, 0] [1, 130, 64000] :
    // CHECK-SAME:              memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
    // CHECK-SAME:           to memref<1x130x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: [[ARG_1_TILE_1:%.+]] = VPUIP.SubView [[ARG_1]] [0, 130, 0] [1, 130, 64000] :
    // CHECK-SAME:              memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
    // CHECK-SAME:           to memref<1x130x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: [[COPY_TILE_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_1]] : memref<1x130x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_1]] : memref<1x130x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  -> memref<1x130x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[COPY_TILE_0]], [[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x130x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>,
    // CHECK-SAME:      memref<1x130x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1]] : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  -> memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: return [[CONCAT]] : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

// CHECK-LABEL: func.func @SplitByChannels5D
// CHECK-SAME:      ([[ARG_0:%[^:]+]]: memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>,
// CHECK-SAME:      [[ARG_1:%[^:]+]]: memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
func.func @SplitByChannels5D(
        %arg0: memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>,
        %arg1: memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
        -> memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
                   outputs(%arg1 : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
                   -> memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    return %0 : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: [[ARG_0_TILE_0:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0, 0] [1, 160, 32, 16, 1] :
    // CHECK-SAME:              memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
    // CHECK-SAME:           to memref<1x160x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: [[ARG_1_TILE_0:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 0, 0, 0] [1, 160, 32, 16, 1] :
    // CHECK-SAME:              memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
    // CHECK-SAME:           to memref<1x160x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: [[COPY_TILE_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_0]] : memref<1x160x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_0]] : memref<1x160x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  -> memref<1x160x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: [[ARG_0_TILE_1:%.+]] = VPUIP.SubView [[ARG_0]] [0, 160, 0, 0, 0] [1, 160, 32, 16, 1] :
    // CHECK-SAME:              memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
    // CHECK-SAME:           to memref<1x160x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: [[ARG_1_TILE_1:%.+]] = VPUIP.SubView [[ARG_1]] [0, 160, 0, 0, 0] [1, 160, 32, 16, 1] :
    // CHECK-SAME:              memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
    // CHECK-SAME:           to memref<1x160x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: [[COPY_TILE_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_1]] : memref<1x160x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_1]] : memref<1x160x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  -> memref<1x160x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[COPY_TILE_0]], [[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x160x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>,
    // CHECK-SAME:      memref<1x160x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1]] : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  -> memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: return [[CONCAT]] : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

// CHECK-LABEL: func.func @SplitByDepth5D
// CHECK-SAME:      ([[ARG_0:%[^:]+]]: memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>,
// CHECK-SAME:      [[ARG_1:%[^:]+]]: memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
func.func @SplitByDepth5D(
        %arg0: memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>,
        %arg1: memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
        -> memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
                   outputs(%arg1 : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
                   -> memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    return %0 : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: [[ARG_0_TILE_0:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0, 0] [1, 1, 160, 192, 160] :
    // CHECK-SAME:              memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
    // CHECK-SAME:           to memref<1x1x160x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: [[ARG_1_TILE_0:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 0, 0, 0] [1, 1, 160, 192, 160] :
    // CHECK-SAME:              memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
    // CHECK-SAME:           to memref<1x1x160x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: [[COPY_TILE_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_0]] : memref<1x1x160x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_0]] : memref<1x1x160x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  -> memref<1x1x160x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: [[ARG_0_TILE_1:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 160, 0, 0] [1, 1, 160, 192, 160] :
    // CHECK-SAME:              memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
    // CHECK-SAME:           to memref<1x1x160x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: [[ARG_1_TILE_1:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 160, 0, 0] [1, 1, 160, 192, 160] :
    // CHECK-SAME:              memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
    // CHECK-SAME:           to memref<1x1x160x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: [[COPY_TILE_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_1]] : memref<1x1x160x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_1]] : memref<1x1x160x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  -> memref<1x1x160x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[COPY_TILE_0]], [[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x1x160x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>,
    // CHECK-SAME:      memref<1x1x160x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  outputs([[ARG_1]] : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  -> memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: return [[CONCAT]] : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
}

//
// -----
//


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputType = memref<3x32x1080x1920x!quant.uniform<u8:f16, 1.2608227898092832:124>, {order = #NHWC}, @DDR>
!OutputType = memref<3x32x1080x1920x!quant.uniform<u8:f16, 1.2608227898092832:124>, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>

// CHECK-LABEL: @RecursiveSplitLargeCopy
// CHECK-SAME:      ([[ARG_0:%[^:]+]]: memref<3x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>,
// CHECK-SAME:      [[ARG_1:%[^:]+]]: memref<3x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
func.func @RecursiveSplitLargeCopy(
        %arg0: !InputType,
        %arg1: !OutputType)
        -> !OutputType {

    %0 = VPUIP.Copy inputs(%arg0 : !InputType)
                   outputs(%arg1 : !OutputType)
                   -> !OutputType

    return %0 : !OutputType
    // First split across N axis
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 32, 1080, 1920]
    // CHECK-SAME:         memref<3x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 0, 0] [1, 32, 1080, 1920]
    // CHECK-SAME:         memref<3x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>

    // Next 5 splits across H axis
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 0, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_3]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 255, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 255, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_5]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_6:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 510, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_7:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 510, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_6]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_7]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_8:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 765, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_9:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 765, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_8]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_9]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_10:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 1020, 0] [1, 32, 30, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_11:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 1020, 0] [1, 32, 30, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_4:%.+]] = VPUIP.Copy inputs([[SUBVIEW_10]] : memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_11]] : memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_10_1:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 1050, 0] [1, 32, 30, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_11_1:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 1050, 0] [1, 32, 30, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_4_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_10_1]] : memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_11_1]] : memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>

    // Concat of splits across H axis for first N split
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]], [[COPY_2]], [[COPY_3]], [[COPY_4]], [[COPY_4_1]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_1]] : memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>


    // CHECK:       [[SUBVIEW_12:%.+]] = VPUIP.SubView [[ARG_0]] [1, 0, 0, 0] [1, 32, 1080, 1920]
    // CHECK-SAME:         memref<3x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK:       [[SUBVIEW_13:%.+]] = VPUIP.SubView [[ARG_1]] [1, 0, 0, 0] [1, 32, 1080, 1920]
    // CHECK-SAME:         memref<3x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_14:%.+]] = VPUIP.SubView [[SUBVIEW_12]] [0, 0, 0, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_15:%.+]] = VPUIP.SubView [[SUBVIEW_13]] [0, 0, 0, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_5:%.+]] = VPUIP.Copy inputs([[SUBVIEW_14]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_15]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_16:%.+]] = VPUIP.SubView [[SUBVIEW_12]] [0, 0, 255, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_17:%.+]] = VPUIP.SubView [[SUBVIEW_13]] [0, 0, 255, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_6:%.+]] = VPUIP.Copy inputs([[SUBVIEW_16]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_17]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_18:%.+]] = VPUIP.SubView [[SUBVIEW_12]] [0, 0, 510, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_19:%.+]] = VPUIP.SubView [[SUBVIEW_13]] [0, 0, 510, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_7:%.+]] = VPUIP.Copy inputs([[SUBVIEW_18]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_19]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_20:%.+]] = VPUIP.SubView [[SUBVIEW_12]] [0, 0, 765, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_21:%.+]] = VPUIP.SubView [[SUBVIEW_13]] [0, 0, 765, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_8:%.+]] = VPUIP.Copy inputs([[SUBVIEW_20]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_21]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_22:%.+]] = VPUIP.SubView [[SUBVIEW_12]] [0, 0, 1020, 0] [1, 32, 30, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_23:%.+]] = VPUIP.SubView [[SUBVIEW_13]] [0, 0, 1020, 0] [1, 32, 30, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_9:%.+]] = VPUIP.Copy inputs([[SUBVIEW_22]] : memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_23]] : memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_22_1:%.+]] = VPUIP.SubView [[SUBVIEW_12]] [0, 0, 1050, 0] [1, 32, 30, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_23_1:%.+]] = VPUIP.SubView [[SUBVIEW_13]] [0, 0, 1050, 0] [1, 32, 30, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_9_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_22_1]] : memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_23_1]] : memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_5]], [[COPY_6]], [[COPY_7]], [[COPY_8]], [[COPY_9]], [[COPY_9_1]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_13]] : memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_24:%.+]] = VPUIP.SubView [[ARG_0]] [2, 0, 0, 0] [1, 32, 1080, 1920]
    // CHECK-SAME:         memref<3x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK:       [[SUBVIEW_25:%.+]] = VPUIP.SubView [[ARG_1]] [2, 0, 0, 0] [1, 32, 1080, 1920]
    // CHECK-SAME:         memref<3x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_26:%.+]] = VPUIP.SubView [[SUBVIEW_24]] [0, 0, 0, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_27:%.+]] = VPUIP.SubView [[SUBVIEW_25]] [0, 0, 0, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_10:%.+]] = VPUIP.Copy inputs([[SUBVIEW_26]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_27]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_28:%.+]] = VPUIP.SubView [[SUBVIEW_24]] [0, 0, 255, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_29:%.+]] = VPUIP.SubView [[SUBVIEW_25]] [0, 0, 255, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_11:%.+]] = VPUIP.Copy inputs([[SUBVIEW_28]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_29]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_30:%.+]] = VPUIP.SubView [[SUBVIEW_24]] [0, 0, 510, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_31:%.+]] = VPUIP.SubView [[SUBVIEW_25]] [0, 0, 510, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_12:%.+]] = VPUIP.Copy inputs([[SUBVIEW_30]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_31]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_32:%.+]] = VPUIP.SubView [[SUBVIEW_24]] [0, 0, 765, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_33:%.+]] = VPUIP.SubView [[SUBVIEW_25]] [0, 0, 765, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_13:%.+]] = VPUIP.Copy inputs([[SUBVIEW_32]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_33]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_34:%.+]] = VPUIP.SubView [[SUBVIEW_24]] [0, 0, 1020, 0] [1, 32, 30, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_35:%.+]] = VPUIP.SubView [[SUBVIEW_25]] [0, 0, 1020, 0] [1, 32, 30, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_14:%.+]] = VPUIP.Copy inputs([[SUBVIEW_34]] : memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_35]] : memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_34_1:%.+]] = VPUIP.SubView [[SUBVIEW_24]] [0, 0, 1050, 0] [1, 32, 30, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC}, @DDR>
    // CHECK-SAME:         to memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_35_1:%.+]] = VPUIP.SubView [[SUBVIEW_25]] [0, 0, 1050, 0] [1, 32, 30, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_14_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_34_1]] : memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_35_1]] : memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[CONCATVIEW_2:%.+]] = VPUIP.ConcatView inputs([[COPY_10]], [[COPY_11]], [[COPY_12]], [[COPY_13]], [[COPY_14]], [[COPY_14_1]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x30x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_25]] : memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>

    // Resulting concat
    // CHECK:       [[CONCATVIEW_3:%.+]] = VPUIP.ConcatView inputs([[CONCATVIEW_0]], [[CONCATVIEW_1]], [[CONCATVIEW_2]] : memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[ARG_1]] : memref<3x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<3x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       return [[CONCATVIEW_3]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SplitDoubleStrideCopyByChannelsWithLowestDimStrided
// CHECK-SAME:      ([[ARG_0:%[^:]+]]: memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>,
// CHECK-SAME:      [[ARG_1:%[^:]+]]: memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>)
func.func @SplitDoubleStrideCopyByChannelsWithLowestDimStrided(
        %arg0: memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>,
        %arg1: memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>)
        -> memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>)
                   outputs(%arg1 : memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>)
                   -> memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>

    return %0 : memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>

    // CHECK: [[ARG_0_TILE_0:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 129, 1, 17] :
    // CHECK-SAME:  memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>
    // CHECK-SAME:  to memref<1x129x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>

    // CHECK: [[ARG_1_TILE_0:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 0, 0] [1, 129, 1, 17] :
    // CHECK-SAME:  memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>
    // CHECK-SAME:  to memref<1x129x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>

    // CHECK: [[COPY_TILE_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_0]] : memref<1x129x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_0]] : memref<1x129x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>)
    // CHECK-SAME:    -> memref<1x129x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>

    // CHECK: [[ARG_0_TILE_1:%.+]] = VPUIP.SubView [[ARG_0]] [0, 129, 0, 0] [1, 129, 1, 17] :
    // CHECK-SAME:  memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>
    // CHECK-SAME:  to memref<1x129x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>

    // CHECK: [[ARG_1_TILE_1:%.+]] = VPUIP.SubView [[ARG_1]] [0, 129, 0, 0] [1, 129, 1, 17] :
    // CHECK-SAME:  memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>
    // CHECK-SAME:  to memref<1x129x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>

    // CHECK: [[COPY_TILE_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[ARG_0_TILE_1]] : memref<1x129x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>)
    // CHECK-SAME:  outputs([[ARG_1_TILE_1]] : memref<1x129x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>)
    // CHECK-SAME:    -> memref<1x129x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs([[COPY_TILE_0]], [[COPY_TILE_1]] :
    // CHECK-SAME:    memref<1x129x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>,
    // CHECK-SAME:    memref<1x129x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>)
    // CHECK-SAME:  outputs([[ARG_1]] : memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>)
    // CHECK-SAME:    -> memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>

    // CHECK: return [[CONCAT]] : memref<1x258x1x17xf16, {order = #NCHW, strides = [660222, 2559, 2559, 128]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistrType = !VPUIP.DistributedBuffer<
    1x16x38x640xf16,
    #NHWC, @CMX_NN, {
        mode = "DUPLICATED",
        num_tiles = [1, 2, 1, 1],
        num_clusters = 2 : i64}>

// CHECK-LABEL: func.func @SplitSingleOneStrideCopyWithConcatUser
// CHECK-SAME:    ([[INPUT0:%[a-z0-9]+]]: !VPUIP.DistributedBuffer<1x16x38x640xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
// CHECK-SAME:    [[INPUT1:%.+]]: !VPUIP.DistributedBuffer<1x16x38x640xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
// CHECK-SAME:    [[INPUT2:%.+]]: memref<1x16x76x640xf16, {order = #NHWC}, @DDR>)
func.func @SplitSingleOneStrideCopyWithConcatUser(
        %arg0: !DistrType,
        %arg1: !DistrType,
        %arg2: memref<1x16x76x640xf16, {order = #NHWC}, @DDR>)
        -> memref<1x32x76x640xf16, {order = #NHWC}, @DDR> {
    %0 = memref.alloc() : memref<1x32x76x640xf16, {order = #NHWC}, @DDR>

    %1 = VPUIP.SubView %0 [0, 16, 0, 0] [1, 16, 38, 640] : memref<1x32x76x640xf16, {order = #NHWC}, @DDR> to memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>
    %2 = VPUIP.Copy
        inputs(%arg0 : !DistrType)
        outputs(%1 : memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>)  ->  memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 16, 38, 0] [1, 16, 38, 640] : memref<1x32x76x640xf16, {order = #NHWC}, @DDR> to memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>
    %4 = VPUIP.Copy
        inputs(%arg1 : !DistrType)
        outputs(%3 : memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>)  ->  memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>

    %5 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 76, 640] : memref<1x32x76x640xf16, {order = #NHWC}, @DDR> to memref<1x16x76x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>
    %6 = VPUIP.Copy inputs(%arg2 : memref<1x16x76x640xf16, {order = #NHWC}, @DDR>) outputs(%5 : memref<1x16x76x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>) -> memref<1x16x76x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>

    %7 = VPUIP.ConcatView inputs(%6, %2, %4 : memref<1x16x76x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>, memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>, memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>) outputs(%0 : memref<1x32x76x640xf16, {order = #NHWC}, @DDR>) -> memref<1x32x76x640xf16, {order = #NHWC}, @DDR>

    return %7 : memref<1x32x76x640xf16, {order = #NHWC}, @DDR>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x32x76x640xf16, {order = #NHWC}, @DDR>

    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 16, 0, 0] [1, 16, 38, 640] :
    // CHECK-SAME:             memref<1x32x76x640xf16, {order = #NHWC}, @DDR>
    // CHECK-SAME:             to memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>
    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:              inputs([[INPUT0]]
    // CHECK-SAME:              outputs([[SUBVIEW_1]]

    // CHECK: [[SUBVIEW_2:%.+]] = VPUIP.SubView [[ALLOC]] [0, 16, 38, 0] [1, 16, 38, 640] :
    // CHECK-SAME:             memref<1x32x76x640xf16, {order = #NHWC}, @DDR>
    // CHECK-SAME:             to memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>
    // CHECK: [[COPY_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:              inputs([[INPUT1]]
    // CHECK-SAME:              outputs([[SUBVIEW_2]]

    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 16, 76, 640] :
    // CHECK-SAME:             memref<1x32x76x640xf16, {order = #NHWC}, @DDR>
    // CHECK-SAME:             to memref<1x16x76x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>

    // CHECK: [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[INPUT2]] [0, 0, 0, 0] [1, 16, 38, 640] :
    // CHECK-SAME:             memref<1x16x76x640xf16, {order = #NHWC}, @DDR>
    // CHECK-SAME:             to memref<1x16x38x640xf16, {order = #NHWC, strides = [778240, 1, 10240, 16]}, @DDR>
    // CHECK: [[CONCAT0_SUBVIEW_0_0:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 0] [1, 16, 38, 640] :
    // CHECK-SAME:             memref<1x16x76x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>
    // CHECK-SAME:             to memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>
    // CHECK: [[COPY_0_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[SUBVIEW_0_0]] : memref<1x16x38x640xf16, {order = #NHWC, strides = [778240, 1, 10240, 16]}, @DDR>)
    // CHECK-SAME:           outputs([[CONCAT0_SUBVIEW_0_0]] : memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>) -> memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>

    // CHECK: [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[INPUT2]] [0, 0, 38, 0] [1, 16, 38, 640] :
    // CHECK-SAME:             memref<1x16x76x640xf16, {order = #NHWC}, @DDR>
    // CHECK-SAME:             to memref<1x16x38x640xf16, {order = #NHWC, strides = [778240, 1, 10240, 16]}, @DDR>
    // CHECK: [[CONCAT0_SUBVIEW_0_1:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 38, 0] [1, 16, 38, 640] :
    // CHECK-SAME:             memref<1x16x76x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>
    // CHECK-SAME:             to memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>
    // CHECK: [[COPY_0_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[SUBVIEW_0_1]] : memref<1x16x38x640xf16, {order = #NHWC, strides = [778240, 1, 10240, 16]}, @DDR>)
    // CHECK-SAME:           outputs([[CONCAT0_SUBVIEW_0_1]] : memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>) -> memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>

    // CHECK: [[CONCAT0:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:           inputs([[COPY_0_0]], [[COPY_0_1]] : memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>, memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>)
    // CHECK-SAME:           outputs([[SUBVIEW_0]] : memref<1x16x76x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>) -> memref<1x16x76x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>

    // CHECK: [[CONCAT1:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:           inputs([[CONCAT0]], [[COPY_1]], [[COPY_2]] : memref<1x16x76x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>, memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>, memref<1x16x38x640xf16, {order = #NHWC, strides = [1556480, 1, 20480, 32]}, @DDR>)
    // CHECK-SAME:           outputs([[ALLOC]] : memref<1x32x76x640xf16, {order = #NHWC}, @DDR>) -> memref<1x32x76x640xf16, {order = #NHWC}, @DDR>

    // CHECK: return [[CONCAT1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NotSplitSingleOneStrideCopyWithConcatUser
// CHECK-SAME:    ([[INPUT0:%.+]]: memref<1x15x1x1xf16, {order = #NHWC}, @CMX_NN>, [[INPUT1:%.+]]: memref<1x1x1x1xf16, {order = #NHWC}, @DDR>)
func.func @NotSplitSingleOneStrideCopyWithConcatUser(
        %arg0: memref<1x15x1x1xf16, {order = #NHWC}, @CMX_NN>,
        %arg1: memref<1x1x1x1xf16, {order = #NHWC}, @DDR>)
        -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR> {
    %0 = memref.alloc() : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    %1 = VPUIP.SubView %0 [0, 1, 0, 0] [1, 15, 1, 1] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR> to memref<1x15x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>
    %2 = VPUIP.Copy
        inputs(%arg0 : memref<1x15x1x1xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%1 : memref<1x15x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>)  ->  memref<1x15x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 1, 1, 1] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR> to memref<1x1x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>
    %4 = VPUIP.Copy inputs(%arg1 : memref<1x1x1x1xf16, {order = #NHWC}, @DDR>) outputs(%3 : memref<1x1x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>) -> memref<1x1x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%4, %2 : memref<1x1x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>, memref<1x15x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>)
        outputs(%0 : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    return %5 : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 1, 0, 0] [1, 15, 1, 1] :
    // CHECK-SAME:             memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK-SAME:             to memref<1x15x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>
    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[INPUT0]]
    // CHECK-SAME:           outputs([[SUBVIEW_1]]

    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 1, 1, 1] :
    // CHECK-SAME:             memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK-SAME:             to memref<1x1x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>
    // CHECK: [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[INPUT1]] : memref<1x1x1x1xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:           outputs([[SUBVIEW_0]] : memref<1x1x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>) -> memref<1x1x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_0]], [[COPY_1]] : memref<1x1x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>, memref<1x15x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>
    // CHECK-SAME:     outputs([[ALLOC]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK: return [[CONCAT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @RecursiveSplitCopyWithLargeSinglePlaneSize
// CHECK-SAME:      [[INPUT:%.+]]: memref<2x64x288x576xf16, @DDR>
// CHECK-SAME:      [[OUTPUT:%.+]]: memref<2x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>
func.func @RecursiveSplitCopyWithLargeSinglePlaneSize(%arg0: memref<2x64x288x576xf16, @DDR>,
        %arg1: memref<2x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>)
        -> memref<2x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<2x64x288x576xf16, @DDR>)
                    outputs(%arg1 : memref<2x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>)
                    -> memref<2x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>


    return %0 : memref<2x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>

    // CHECK: [[IN_SLICE_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 64, 288, 576] :
    // CHECK-SAME:             memref<2x64x288x576xf16, @DDR>
    // CHECK-SAME:             to memref<1x64x288x576xf16, @DDR>
    // CHECK: [[OUT_SLICE_0:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 64, 288, 576] :
    // CHECK-SAME:             memref<2x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>
    // CHECK-SAME:             to memref<1x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>

    // CHECK: [[IN_SLICE_0_0:%.+]] = VPUIP.SubView [[IN_SLICE_0]] [0, 0, 0, 0] [1, 32, 288, 576] :
    // CHECK-SAME:             memref<1x64x288x576xf16, @DDR>
    // CHECK-SAME:             to memref<1x32x288x576xf16, {order = #NCHW, strides = [10616832, 165888, 576, 1]}, @DDR>
    // CHECK: [[OUT_SLICE_0_0:%.+]] = VPUIP.SubView [[OUT_SLICE_0]] [0, 0, 0, 0] [1, 32, 288, 576] :
    // CHECK-SAME:             memref<1x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>
    // CHECK-SAME:             to memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>
    // CHECK: [[COPY_0_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[IN_SLICE_0_0]] : memref<1x32x288x576xf16, {order = #NCHW, strides = [10616832, 165888, 576, 1]}, @DDR>)
    // CHECK-SAME:           outputs([[OUT_SLICE_0_0]] : memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>) -> memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>

    // CHECK: [[IN_SLICE_0_1:%.+]] = VPUIP.SubView [[IN_SLICE_0]] [0, 32, 0, 0] [1, 32, 288, 576] :
    // CHECK-SAME:             memref<1x64x288x576xf16, @DDR>
    // CHECK-SAME:             to memref<1x32x288x576xf16, {order = #NCHW, strides = [10616832, 165888, 576, 1]}, @DDR>
    // CHECK: [[OUT_SLICE_0_1:%.+]] = VPUIP.SubView [[OUT_SLICE_0]] [0, 32, 0, 0] [1, 32, 288, 576] :
    // CHECK-SAME:             memref<1x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>
    // CHECK-SAME:             to memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>
    // CHECK: [[COPY_0_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[IN_SLICE_0_1]] : memref<1x32x288x576xf16, {order = #NCHW, strides = [10616832, 165888, 576, 1]}, @DDR>)
    // CHECK-SAME:           outputs([[OUT_SLICE_0_1]] : memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>) -> memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>

    // CHECK: [[CONCAT_0:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:           inputs([[COPY_0_0]], [[COPY_0_1]] : memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>, memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>)
    // CHECK-SAME:           outputs([[OUT_SLICE_0]] : memref<1x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>) -> memref<1x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>

    // CHECK: [[IN_SLICE_1:%.+]] = VPUIP.SubView [[INPUT]] [1, 0, 0, 0] [1, 64, 288, 576] :
    // CHECK-SAME:             memref<2x64x288x576xf16, @DDR>
    // CHECK-SAME:             to memref<1x64x288x576xf16, @DDR>
    // CHECK: [[OUT_SLICE_1:%.+]] = VPUIP.SubView [[OUTPUT]] [1, 0, 0, 0] [1, 64, 288, 576] :
    // CHECK-SAME:             memref<2x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>
    // CHECK-SAME:             to memref<1x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>

    // CHECK: [[IN_SLICE_1_0:%.+]] = VPUIP.SubView [[IN_SLICE_1]] [0, 0, 0, 0] [1, 32, 288, 576] :
    // CHECK-SAME:             memref<1x64x288x576xf16, @DDR>
    // CHECK-SAME:             to memref<1x32x288x576xf16, {order = #NCHW, strides = [10616832, 165888, 576, 1]}, @DDR>
    // CHECK: [[OUT_SLICE_1_0:%.+]] = VPUIP.SubView [[OUT_SLICE_1]] [0, 0, 0, 0] [1, 32, 288, 576] :
    // CHECK-SAME:             memref<1x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>
    // CHECK-SAME:             to memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>
    // CHECK: [[COPY_1_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[IN_SLICE_1_0]] : memref<1x32x288x576xf16, {order = #NCHW, strides = [10616832, 165888, 576, 1]}, @DDR>)
    // CHECK-SAME:           outputs([[OUT_SLICE_1_0]] : memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>) -> memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>

    // CHECK: [[IN_SLICE_1_1:%.+]] = VPUIP.SubView [[IN_SLICE_1]] [0, 32, 0, 0] [1, 32, 288, 576] :
    // CHECK-SAME:             memref<1x64x288x576xf16, @DDR>
    // CHECK-SAME:             to memref<1x32x288x576xf16, {order = #NCHW, strides = [10616832, 165888, 576, 1]}, @DDR>
    // CHECK: [[OUT_SLICE_1_1:%.+]] = VPUIP.SubView [[OUT_SLICE_1]] [0, 32, 0, 0] [1, 32, 288, 576] :
    // CHECK-SAME:             memref<1x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>
    // CHECK-SAME:             to memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>
    // CHECK: [[COPY_1_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[IN_SLICE_1_1]] : memref<1x32x288x576xf16, {order = #NCHW, strides = [10616832, 165888, 576, 1]}, @DDR>)
    // CHECK-SAME:           outputs([[OUT_SLICE_1_1]] : memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>) -> memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>

    // CHECK: [[CONCAT_1:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:           inputs([[COPY_1_0]], [[COPY_1_1]] : memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>, memref<1x32x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>)
    // CHECK-SAME:           outputs([[OUT_SLICE_1]] : memref<1x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>) -> memref<1x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>

    // CHECK: [[CONCAT_2:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:           inputs([[CONCAT_0]], [[CONCAT_1]] : memref<1x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>, memref<1x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>)
    // CHECK-SAME:           outputs([[OUTPUT]] : memref<2x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>) -> memref<2x64x288x576xf16, {order = #NCHW, strides = [22560768, 331776, 1152, 1]}, @DDR>

    // CHECK: return [[CONCAT_2]]
}
