//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-depth-to-space-dma  %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>

func.func @UnrollDepthToSpaceDMABlockFirstHWCWithSplit(%arg0: memref<16x4x6xf16, #HWC>, %arg1: memref<16x2x12xf16, #HWC>) -> memref<16x2x12xf16, #HWC> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x4x6xf16, #HWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<16x2x12xf16, #HWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<16x4x6xf16, #HWC>) outputs(%2 : memref<16x4x6xf16, #HWC, [@CMX_NN, 0]>) -> memref<16x4x6xf16, #HWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%2 : memref<16x4x6xf16, #HWC, [@CMX_NN, 0]>)
                outputs(%3 : memref<16x2x12xf16, #HWC, [@CMX_NN, 0]>) -> memref<16x2x12xf16, #HWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%3 : memref<16x2x12xf16, #HWC, [@CMX_NN, 0]>) outputs(%arg1 :  memref<16x2x12xf16, #HWC>) -> memref<16x2x12xf16, #HWC>
    }

    return %arg1: memref<16x2x12xf16, #HWC>

    //CHECK:    #HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
    //CHECK:    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    //CHECK:    #map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>

    //CHECK:    [[INPUT:%.+]]: memref<16x4x6xf16, #HWC>
    //CHECK:    [[OUTPUT:%.+]]: memref<16x2x12xf16, #HWC>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x4x6xf16, #HWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x4x3xf16, {order = #HWC, strides = [1, 96, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <96> -> memref<16x4x3xf16, {order = #HWC, strides = [1, 96, 16]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<16x2x12xf16, #HWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<16x2x6xf16, {order = #HWC, strides = [1, 192, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1216> -> memref<16x2x6xf16, {order = #HWC, strides = [1, 192, 16]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<16x2x2x3xf16, {order = #map, strides = [1, 192, 96, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<16x2x3x2xf16, {order = #map, strides = [1, 192, 32, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #NHWC
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<16x2x2x3xf16, {order = #map, strides = [1, 192, 96, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<16x2x3x2xf16, {order = #map, strides = [1, 192, 32, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #NHWC
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_1]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUTPUT_BUFFER]]
    //CHECK-SAME:       outputs([[OUTPUT]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UnrollDepthToSpaceDMABlockFirstNHWCWithSplit(%arg0: memref<1x16x3x6xf16, #NHWC>, %arg1: memref<1x4x6x12xf16, #NHWC>) -> memref<1x4x6x12xf16, #NHWC> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<1x16x3x6xf16, #NHWC>) outputs(%2 : memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%2 : memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%3 : memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%3 : memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 :  memref<1x4x6x12xf16, #NHWC>) -> memref<1x4x6x12xf16, #NHWC>
    }

    return %arg1: memref<1x4x6x12xf16, #NHWC>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d1, d5, d2)>

    //CHECK:    [[INPUT:%.+]]: memref<1x16x3x6xf16, #NHWC>
    //CHECK:    [[OUTPUT:%.+]]: memref<1x4x6x12xf16, #NHWC>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x6xf16, {order = #NHWC, strides = [288, 1, 96, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <192> -> memref<1x16x2x6xf16, {order = #NHWC, strides = [288, 1, 96, 16]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x4x2x12xf16, {order = #NHWC, strides = [288, 1, 48, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1216> -> memref<1x4x4x12xf16, {order = #NHWC, strides = [288, 1, 48, 4]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x4x1x6xf16, {order = #map, strides = [288, 8, 4, 1, 96, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x4x1x2x6x2xf16, {order = #map1, strides = [288, 1, 96, 48, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x4x2x6xf16, {order = #map, strides = [288, 8, 4, 1, 96, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x4x2x2x6x2xf16, {order = #map1, strides = [288, 1, 96, 48, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_1]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUTPUT_BUFFER]]
    //CHECK-SAME:       outputs([[OUTPUT]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UnrollDepthToSpaceDMABlockFirstNHWCNoSplit(%arg0: memref<1x16x1x6xf16, #NHWC>, %arg1: memref<1x4x2x12xf16, #NHWC>) -> memref<1x4x2x12xf16, #NHWC> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x6xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x4x2x12xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<1x16x1x6xf16, #NHWC>) outputs(%2 : memref<1x16x1x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x1x6xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%2 : memref<1x16x1x6xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%3 : memref<1x4x2x12xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x2x12xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%3 : memref<1x4x2x12xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 :  memref<1x4x2x12xf16, #NHWC>) -> memref<1x4x2x12xf16, #NHWC>
    }

    return %arg1: memref<1x4x2x12xf16, #NHWC>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d1, d5, d2)>

    //CHECK:    [[INPUT:%.+]]: memref<1x16x1x6xf16, #NHWC>
    //CHECK:    [[OUTPUT:%.+]]: memref<1x4x2x12xf16, #NHWC>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x4x2x12xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x4x1x6xf16, #map, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x4x1x2x6x2xf16, #map1, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_1]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUTPUT_BUFFER]]
    //CHECK-SAME:       outputs([[OUTPUT]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UnrollDepthToSpaceDMABlockFirstNHWC(%arg0: memref<1x8x2x3xf16, #NHWC>, %arg1: memref<1x2x4x6xf16, #NHWC>) -> memref<1x2x4x6xf16, #NHWC> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<1x8x2x3xf16, #NHWC>) outputs(%3 : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%3 : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%4 : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%4 : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 :  memref<1x2x4x6xf16, #NHWC>) -> memref<1x2x4x6xf16, #NHWC>
    }

    return %arg1: memref<1x2x4x6xf16, #NHWC>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d1, d5, d2)>

    //CHECK:    [[INPUT:%.+]]: memref<1x8x2x3xf16, #NHWC>
    //CHECK:    [[OUTPUT:%.+]]: memref<1x2x4x6xf16, #NHWC>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x1x3xf16, {order = #NHWC, strides = [48, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <48> -> memref<1x8x1x3xf16, {order = #NHWC, strides = [48, 1, 24, 8]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x2x2x6xf16, {order = #NHWC, strides = [48, 1, 12, 2]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1072> -> memref<1x2x2x6xf16, {order = #NHWC, strides = [48, 1, 12, 2]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x2x1x3xf16, {order = #map, strides = [48, 4, 2, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x1x2x3x2xf16, {order = #map1, strides = [48, 1, 24, 12, 4, 2]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x2x1x3xf16, {order = #map, strides = [48, 4, 2, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x1x2x3x2xf16, {order = #map1, strides = [48, 1, 24, 12, 4, 2]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_1]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUTPUT_BUFFER]]
    //CHECK-SAME:       outputs([[OUTPUT]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UnrollDepthToSpaceDMADepthFirstNHWC(%arg0: memref<1x8x2x1xf16, #NHWC>, %arg1: memref<1x2x4x2xf16, #NHWC>) -> memref<1x2x4x2xf16, #NHWC> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x2x1xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x2x4x2xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<1x8x2x1xf16, #NHWC>) outputs(%3 : memref<1x8x2x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, output_channel = 2 : i64, output_width = 2 : i64}
                inputs(%3 : memref<1x8x2x1xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%4 : memref<1x2x4x2xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x2xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%4 : memref<1x2x4x2xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 :  memref<1x2x4x2xf16, #NHWC>) -> memref<1x2x4x2xf16, #NHWC>
    }

    return %arg1: memref<1x2x4x2xf16, #NHWC>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d5, d3)>

    //CHECK:    [[INPUT:%.+]]: memref<1x8x2x1xf16, #NHWC>
    //CHECK:    [[OUTPUT:%.+]]: memref<1x2x4x2xf16, #NHWC>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x1x1xf16, {order = #NHWC, strides = [16, 1, 8, 8]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <16> -> memref<1x8x1x1xf16, {order = #NHWC, strides = [16, 1, 8, 8]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x2x4x2xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 4, 2]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1040> -> memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 4, 2]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x2x1x1xf16, {order = #map, strides = [16, 4, 2, 1, 8, 8]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x1x2x1x2xf16, {order = #map1, strides = [16, 1, 8, 4, 4, 2]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x2x1x1xf16, {order = #map, strides = [16, 4, 2, 1, 8, 8]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x1x2x1x2xf16, {order = #map1, strides = [16, 1, 8, 4, 4, 2]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_1]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUTPUT_BUFFER]]
    //CHECK-SAME:       outputs([[OUTPUT]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x3x4x6x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// Case 1: Unroll ClusterD2SDMA with single-cluster input and multi-cluster(SEGMENTED) output
func.func @UnrollSegmentedClusterDepthToSpaceDMACase1() -> !OutputDistributed {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x12x2x3x!qElemType, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> <1024> -> !OutputDistributed

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
              inputs(%2 : memref<1x12x2x3x!qElemType, #NHWC, [@CMX_NN, 0]>)
              outputs(%3 : !OutputDistributed) -> !OutputDistributed
    }

    return %3: !OutputDistributed

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d1, d5, d2)>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x12x1x3x!qElemType, {order = #NHWC, strides = [72, 1, 36, 12]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <36> -> memref<1x12x1x3x!qElemType, {order = #NHWC, strides = [72, 1, 36, 12]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> <1024> -> !VPUIP.DistributedBuffer<1x3x4x6x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x3x2x6x!qElemType, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <1024> -> memref<1x3x2x6x!qElemType, #NHWC, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x3x1x3x!qElemType, {order = #map, strides = [72, 6, 3, 1, 36, 12]}, [@CMX_NN, 0]
    //CHECK-SAME:       outputType = memref<1x3x1x2x3x2x!qElemType, #map1, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x3x1x3x!qElemType, {order = #map, strides = [72, 6, 3, 1, 36, 12]}, [@CMX_NN, 0]
    //CHECK-SAME:       outputType = memref<1x3x1x2x3x2x!qElemType, #map1, [@CMX_NN, 1]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT_BUFFER]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x12x2x3x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// Case 2: Unroll ClusterD2SDMA with multi-cluster(SEGMENTED) input and single-cluster output
func.func @UnrollSegmentedClusterDepthToSpaceDMACase2() -> memref<1x3x4x6x!qElemType, #NHWC, [@CMX_NN, 0]> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x3x4x6x!qElemType, #NHWC, [@CMX_NN, 0]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
              inputs(%2 :  !InputDistributed)
              outputs(%3 : memref<1x3x4x6x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x3x4x6x!qElemType, #NHWC, [@CMX_NN, 0]>
    }

    return %3: memref<1x3x4x6x!qElemType, #NHWC, [@CMX_NN, 0]>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d1, d5, d2)>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x12x1x3x!qElemType, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x12x1x3x!qElemType, #NHWC, [@CMX_NN, 1]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x3x4x6x!qElemType, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x3x2x6x!qElemType, {order = #NHWC, strides = [72, 1, 18, 3]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1060> -> memref<1x3x2x6x!qElemType, {order = #NHWC, strides = [72, 1, 18, 3]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x3x1x3x!qElemType, #map, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x3x1x2x3x2x!qElemType, {order = #map1, strides = [72, 1, 36, 18, 6, 3]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x3x1x3x!qElemType, #map, [@CMX_NN, 1]>
    //CHECK-SAME:       outputType = memref<1x3x1x2x3x2x!qElemType, {order = #map1, strides = [72, 1, 36, 18, 6, 3]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT_BUFFER]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x12x2x3x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x3x4x6x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// Case 3: Unroll ClusterD2SDMA with multi-cluster(SEGMENTED) input and multi-cluster(SEGMENTED) output
func.func @UnrollSegmentedClusterDepthToSpaceDMACase3() -> !OutputDistributed {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %3 = VPURT.DeclareBuffer <CMX_NN> <1024> -> !OutputDistributed

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
              inputs(%2 :  !InputDistributed)
              outputs(%3 : !OutputDistributed) -> !OutputDistributed
    }

    return %3: !OutputDistributed

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d1, d5, d2)>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x12x1x3x!qElemType, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x12x1x3x!qElemType, #NHWC, [@CMX_NN, 1]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> <1024> -> !VPUIP.DistributedBuffer<1x3x4x6x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x3x2x6x!qElemType, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <1024> -> memref<1x3x2x6x!qElemType, #NHWC, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x3x1x3x!qElemType, #map, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x3x1x2x3x2x!qElemType, #map1, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x3x1x3x!qElemType, #map, [@CMX_NN, 1]>
    //CHECK-SAME:       outputType = memref<1x3x1x2x3x2x!qElemType, #map1, [@CMX_NN, 1]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT_BUFFER]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x128x128x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x256x256x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// Case 4: Unroll padded ClusterD2SDMA with multi-cluster(SEGMENTED) input and multi-cluster(SEGMENTED) output
func.func @UnrollSegmentedClusterDepthToSpaceDMACase4() -> !OutputDistributed {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %3 = VPURT.DeclareBuffer <CMX_NN> <524288> -> !OutputDistributed

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64, padded_channels = #IE.ChannelPadding<input = 4: i64, output = 1: i64>}
              inputs(%2 :  !InputDistributed)
              outputs(%3 : !OutputDistributed) -> !OutputDistributed
    }

    return %3: !OutputDistributed

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d1, d5, d2)>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x12x32x128x!qElemType, {order = #NHWC, strides = [131072, 1, 2048, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <65536> -> memref<1x12x32x128x!qElemType, {order = #NHWC, strides = [131072, 1, 2048, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x12x32x128x!qElemType, {order = #NHWC, strides = [131072, 1, 2048, 16]}, [@CMX_NN, 1]>
    //CHECK:    [[INPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <65536> -> memref<1x12x32x128x!qElemType, {order = #NHWC, strides = [131072, 1, 2048, 16]}, [@CMX_NN, 1]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> <524288> -> !VPUIP.DistributedBuffer<1x4x256x256x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <524288> -> memref<1x3x64x256x!qElemType, {order = #NHWC, strides = [131072, 1, 1024, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <589824> -> memref<1x3x64x256x!qElemType, {order = #NHWC, strides = [131072, 1, 1024, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <524288> -> memref<1x3x64x256x!qElemType, {order = #NHWC, strides = [131072, 1, 1024, 4]}, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <589824> -> memref<1x3x64x256x!qElemType, {order = #NHWC, strides = [131072, 1, 1024, 4]}, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x3x32x128x!qElemType, {order = #map, strides = [131072, 6, 3, 1, 2048, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x3x32x2x128x2x!qElemType, {order = #map1, strides = [131072, 1, 2048, 1024, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x3x32x128x!qElemType, {order = #map, strides = [131072, 6, 3, 1, 2048, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x3x32x2x128x2x!qElemType, {order = #map1, strides = [131072, 1, 2048, 1024, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x3x32x128x!qElemType, {order = #map, strides = [131072, 6, 3, 1, 2048, 16]}, [@CMX_NN, 1]>
    //CHECK-SAME:       outputType = memref<1x3x32x2x128x2x!qElemType, {order = #map1, strides = [131072, 1, 2048, 1024, 8, 4]}, [@CMX_NN, 1]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_2]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_2]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x3x32x128x!qElemType, {order = #map, strides = [131072, 6, 3, 1, 2048, 16]}, [@CMX_NN, 1]>
    //CHECK-SAME:       outputType = memref<1x3x32x2x128x2x!qElemType, {order = #map1, strides = [131072, 1, 2048, 1024, 8, 4]}, [@CMX_NN, 1]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_3]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_3]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT_BUFFER]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UnrollDepthToSpaceWithBlockFirstLargeH() -> memref<1x1x534x34xf16, #NHWC, [@CMX_NN, 0]> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x267x17xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <65536> -> memref<1x1x534x34xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 1 : i64}
            inputs(%2 : memref<1x4x267x17xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%3 : memref<1x1x534x34xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x534x34xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %3: memref<1x1x534x34xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d1, d5, d2)>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x133x17xf16, {order = #NHWC, strides = [18156, 1, 68, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <18088> -> memref<1x4x134x17xf16, {order = #NHWC, strides = [18156, 1, 68, 4]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <65536> -> memref<1x1x534x34xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <65536> -> memref<1x1x266x34xf16, {order = #NHWC, strides = [18156, 1, 34, 1]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <83624> -> memref<1x1x268x34xf16, {order = #NHWC, strides = [18156, 1, 34, 1]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x1x133x17xf16, {order = #map, strides = [18156, 2, 1, 1, 68, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x1x133x2x17x2xf16, {order = #map1, strides = [18156, 1, 68, 34, 2, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x2x1x134x17xf16, {order = #map, strides = [18156, 2, 1, 1, 68, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x1x134x2x17x2xf16, {order = #map1, strides = [18156, 1, 68, 34, 2, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT_BUFFER]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UnrollDepthToSpaceWithDepthFirstLargeH() -> memref<1x1x534x34xf16, #NHWC, [@CMX_NN, 0]> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x267x17xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <65536> -> memref<1x1x534x34xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, port = 1 : i64}
            inputs(%2 : memref<1x4x267x17xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>)
            outputs(%3 : memref<1x1x534x34xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) -> memref<1x1x534x34xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
    }

    return %3: memref<1x1x534x34xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d5, d3)>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x133x17xf16, {order = #NHWC, strides = [18156, 1, 68, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <18088> -> memref<1x4x134x17xf16, {order = #NHWC, strides = [18156, 1, 68, 4]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <65536> -> memref<1x1x534x34xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <65536> -> memref<1x1x266x34xf16, {order = #NHWC, strides = [18156, 1, 34, 1]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <83624> -> memref<1x1x268x34xf16, {order = #NHWC, strides = [18156, 1, 34, 1]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType =  memref<1x1x2x2x133x17xf16, {order = #map, strides = [18156, 4, 2, 1, 68, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x1x133x2x17x2xf16, {order = #map1, strides = [18156, 1, 68, 34, 2, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.DepthToSpaceDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x1x2x2x134x17xf16, {order = #map, strides = [18156, 4, 2, 1, 68, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x1x134x2x17x2xf16, {order = #map1, strides = [18156, 1, 68, 34, 2, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT_BUFFER]]
}
