//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-space-to-depth-dma %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>

func.func @UnrollSpaceToDepthDMABlockFirstHWCWithSplit(%arg0: memref<16x2x12xf16, #HWC>, %arg1: memref<16x4x6xf16, #HWC>) -> memref<16x4x6xf16, #HWC> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x2x12xf16, #HWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<16x4x6xf16, #HWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<16x2x12xf16, #HWC>) outputs(%2 : memref<16x2x12xf16, #HWC, [@CMX_NN, 0]>) -> memref<16x2x12xf16, #HWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%2 : memref<16x2x12xf16, #HWC, [@CMX_NN, 0]>)
                outputs(%3 : memref<16x4x6xf16, #HWC, [@CMX_NN, 0]>) -> memref<16x4x6xf16, #HWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%3 : memref<16x4x6xf16, #HWC, [@CMX_NN, 0]>) outputs(%arg1 :  memref<16x4x6xf16, #HWC>) -> memref<16x4x6xf16, #HWC>
    }

    return %arg1: memref<16x4x6xf16, #HWC>

    //CHECK:    #HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
    //CHECK:    #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)
    //CHECK:    #map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>

    //CHECK:    [[INPUT:%.+]]: memref<16x2x12xf16, #HWC>
    //CHECK:    [[OUTPUT:%.+]]: memref<16x4x6xf16, #HWC>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x2x12xf16, #HWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x2x6xf16, {order = #HWC, strides = [1, 192, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <192> -> memref<16x2x6xf16, {order = #HWC, strides = [1, 192, 16]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<16x4x6xf16, #HWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<16x4x3xf16, {order = #HWC, strides = [1, 96, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1120> -> memref<16x4x3xf16, {order = #HWC, strides = [1, 96, 16]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<16x2x3x2xf16, {order = #map, strides = [1, 192, 32, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<16x2x2x3xf16, {order = #map, strides = [1, 192, 96, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #NWCH
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<16x2x3x2xf16, {order = #map, strides = [1, 192, 32, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<16x2x2x3xf16, {order = #map, strides = [1, 192, 96, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #NWCH
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
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

func.func @UnrollSpaceToDepthDMABlockFirstNHWCWithSplit(%arg0: memref<1x4x6x12xf16, #NHWC>, %arg1: memref<1x16x3x6xf16, #NHWC>) -> memref<1x16x3x6xf16, #NHWC> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<1x4x6x12xf16, #NHWC>) outputs(%2 : memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%2 : memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%3 : memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%3 : memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 :  memref<1x16x3x6xf16, #NHWC>) -> memref<1x16x3x6xf16, #NHWC>
    }

    return %arg1: memref<1x16x3x6xf16, #NHWC>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

    //CHECK:    [[INPUT:%.+]]: memref<1x4x6x12xf16, #NHWC>
    //CHECK:    [[OUTPUT:%.+]]: memref<1x16x3x6xf16, #NHWC>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x2x12xf16, {order = #NHWC, strides = [288, 1, 48, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <192> -> memref<1x4x4x12xf16, {order = #NHWC, strides = [288, 1, 48, 4]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x16x1x6xf16, {order = #NHWC, strides = [288, 1, 96, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1216> -> memref<1x16x2x6xf16, {order = #NHWC, strides = [288, 1, 96, 16]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x1x2x6x2xf16, {order = #map, strides = [288, 1, 96, 48, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x1x6xf16, {order = #map1, strides = [288, 8, 4, 1, 96, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x2x2x6x2xf16, {order = #map, strides = [288, 1, 96, 48, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x2x6xf16, {order = #map1, strides = [288, 8, 4, 1, 96, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
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

func.func @UnrollSpaceToDepthDMABlockFirstNHWCNoSplit(%arg0: memref<1x4x2x12xf16, #NHWC>, %arg1: memref<1x16x1x6xf16, #NHWC>) -> memref<1x16x1x6xf16, #NHWC> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x2x12xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x16x1x6xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<1x4x2x12xf16, #NHWC>) outputs(%2 : memref<1x4x2x12xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x2x12xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%2 : memref<1x4x2x12xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%3 : memref<1x16x1x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x1x6xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%3 : memref<1x16x1x6xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 :  memref<1x16x1x6xf16, #NHWC>) -> memref<1x16x1x6xf16, #NHWC>
    }

    return %arg1: memref<1x16x1x6xf16, #NHWC>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

    //CHECK:    [[INPUT:%.+]]: memref<1x4x2x12xf16, #NHWC>
    //CHECK:    [[OUTPUT:%.+]]: memref<1x16x1x6xf16, #NHWC>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x2x12xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x16x1x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x1x2x6x2xf16, #map, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x1x6xf16, #map1, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
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

func.func @UnrollSpaceToDepthDMABlockFirstNHWC(%arg0: memref<1x2x4x6xf16, #NHWC>, %arg1: memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<1x2x4x6xf16, #NHWC>) outputs(%2 : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%2 : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%4 : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%4 : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 :  memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC>
    }

    return %arg1: memref<1x8x2x3xf16, #NHWC>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

    //CHECK:    [[INPUT:%.+]]: memref<1x2x4x6xf16, #NHWC>
    //CHECK:    [[OUTPUT:%.+]]: memref<1x8x2x3xf16, #NHWC>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x2x6xf16, {order = #NHWC, strides = [48, 1, 12, 2]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <48> -> memref<1x2x2x6xf16, {order = #NHWC, strides = [48, 1, 12, 2]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x1x3xf16, {order = #NHWC, strides = [48, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1072> -> memref<1x8x1x3xf16, {order = #NHWC, strides = [48, 1, 24, 8]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x1x2x3x2xf16, {order = #map, strides = [48, 1, 24, 12, 4, 2]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x2x1x3xf16, {order = #map1, strides = [48, 4, 2, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x1x2x3x2xf16, {order = #map, strides = [48, 1, 24, 12, 4, 2]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x2x1x3xf16, {order = #map1, strides = [48, 4, 2, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
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

func.func @UnrollSpaceToDepthDMABlockFirstNCHW(%arg0: memref<1x2x4x6xf16>, %arg1: memref<1x8x2x3xf16>) -> memref<1x8x2x3xf16> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x2x3xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<1x2x4x6xf16>) outputs(%2 : memref<1x2x4x6xf16, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%2 : memref<1x2x4x6xf16, [@CMX_NN, 0]>)
                outputs(%4 : memref<1x8x2x3xf16, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%4 : memref<1x8x2x3xf16, [@CMX_NN, 0]>) outputs(%arg1 :  memref<1x8x2x3xf16>) -> memref<1x8x2x3xf16>
    }

    return %arg1: memref<1x8x2x3xf16>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

    //CHECK:    [[INPUT:%.+]]: memref<1x2x4x6xf16>
    //CHECK:    [[OUTPUT:%.+]]: memref<1x8x2x3xf16>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x2x6xf16, {order = #NCHW, strides = [48, 24, 6, 1]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<1x2x2x6xf16, {order = #NCHW, strides = [48, 24, 6, 1]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x2x3xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x1x3xf16, {order = #NCHW, strides = [48, 6, 3, 1]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1030> -> memref<1x8x1x3xf16, {order = #NCHW, strides = [48, 6, 3, 1]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x1x2x3x2xf16, {order = #map, strides = [48, 24, 12, 6, 2, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x2x1x3xf16, {order = #map, strides = [48, 24, 12, 6, 3, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map1
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x1x2x3x2xf16, {order = #map, strides = [48, 24, 12, 6, 2, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x2x1x3xf16, {order = #map, strides = [48, 24, 12, 6, 3, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map1
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
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

func.func @UnrollSpaceToDepthDMADepthFirstNHWC(%arg0: memref<1x2x4x6xf16, #NHWC>, %arg1: memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<1x2x4x6xf16, #NHWC>) outputs(%2 : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%2 : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%4 : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%4 : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 :  memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC>
    }

    return %arg1: memref<1x8x2x3xf16, #NHWC>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5, d2, d4)>

    //CHECK:    [[INPUT:%.+]]: memref<1x2x4x6xf16, #NHWC>
    //CHECK:    [[OUTPUT:%.+]]: memref<1x8x2x3xf16, #NHWC>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x2x6xf16, {order = #NHWC, strides = [48, 1, 12, 2]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <48> -> memref<1x2x2x6xf16, {order = #NHWC, strides = [48, 1, 12, 2]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x1x3xf16, {order = #NHWC, strides = [48, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1072> -> memref<1x8x1x3xf16, {order = #NHWC, strides = [48, 1, 24, 8]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x1x2x3x2xf16, {order = #map, strides = [48, 1, 24, 12, 4, 2]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x2x1x3xf16, {order = #map1, strides = [48, 4, 2, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<DEPTH_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x1x2x3x2xf16, {order = #map, strides = [48, 1, 24, 12, 4, 2]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x2x1x3xf16, {order = #map1, strides = [48, 4, 2, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<DEPTH_FIRST>
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

func.func @UnrollSpaceToDepthDMADepthFirstNCHW(%arg0: memref<1x2x4x6xf16>, %arg1: memref<1x8x2x3xf16>) -> memref<1x8x2x3xf16> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x2x3xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<1x2x4x6xf16>) outputs(%2 : memref<1x2x4x6xf16, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%2 : memref<1x2x4x6xf16, [@CMX_NN, 0]>)
                outputs(%4 : memref<1x8x2x3xf16, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%4 : memref<1x8x2x3xf16, [@CMX_NN, 0]>) outputs(%arg1 :  memref<1x8x2x3xf16>) -> memref<1x8x2x3xf16>
    }

    return %arg1: memref<1x8x2x3xf16>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5, d2, d4)>

    //CHECK:    [[INPUT:%.+]]: memref<1x2x4x6xf16>
    //CHECK:    [[OUTPUT:%.+]]: memref<1x8x2x3xf16>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x2x6xf16, {order = #NCHW, strides = [48, 24, 6, 1]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<1x2x2x6xf16, {order = #NCHW, strides = [48, 24, 6, 1]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x2x3xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x1x3xf16, {order = #NCHW, strides = [48, 6, 3, 1]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1030> -> memref<1x8x1x3xf16, {order = #NCHW, strides = [48, 6, 3, 1]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x1x2x3x2xf16, {order = #map, strides = [48, 24, 12, 6, 2, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x2x1x3xf16, {order = #map, strides = [48, 24, 12, 6, 3, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map1
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<DEPTH_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x1x2x3x2xf16, {order = #map, strides = [48, 24, 12, 6, 2, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x2x1x3xf16, {order = #map, strides = [48, 24, 12, 6, 3, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map1
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<DEPTH_FIRST>
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

func.func @UnrollSpaceToDepthDMABlockFirstNCHWToNHWC(%arg0: memref<1x2x4x6xf16>, %arg1: memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<1x2x4x6xf16>) outputs(%2 : memref<1x2x4x6xf16, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%2 : memref<1x2x4x6xf16, [@CMX_NN, 0]>)
                outputs(%4 : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%4 : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 :  memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC>
    }

    return %arg1: memref<1x8x2x3xf16, #NHWC>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

    //CHECK:    [[INPUT:%.+]]: memref<1x2x4x6xf16>
    //CHECK:    [[OUTPUT:%.+]]: memref<1x8x2x3xf16, #NHWC>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x2x6xf16, {order = #NCHW, strides = [48, 24, 6, 1]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<1x2x2x6xf16, {order = #NCHW, strides = [48, 24, 6, 1]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x1x3xf16, {order = #NHWC, strides = [48, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1072> -> memref<1x8x1x3xf16, {order = #NHWC, strides = [48, 1, 24, 8]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x1x2x3x2xf16, {order = #map, strides = [48, 24, 12, 6, 2, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x2x1x3xf16, {order = #map1, strides = [48, 4, 2, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x1x2x3x2xf16, {order = #map, strides = [48, 24, 12, 6, 2, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x2x1x3xf16, {order = #map1, strides = [48, 4, 2, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
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

func.func @UnrollSpaceToDepthDMADepthFirstNCHWToNHWC(%arg0: memref<1x2x4x6xf16>, %arg1: memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%arg0 : memref<1x2x4x6xf16>) outputs(%2 : memref<1x2x4x6xf16, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%2 : memref<1x2x4x6xf16, [@CMX_NN, 0]>)
                outputs(%4 : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%4 : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 :  memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC>
    }

    return %arg1: memref<1x8x2x3xf16, #NHWC>

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5, d2, d4)>

    //CHECK:    [[INPUT:%.+]]: memref<1x2x4x6xf16>
    //CHECK:    [[OUTPUT:%.+]]: memref<1x8x2x3xf16, #NHWC>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x2x6xf16, {order = #NCHW, strides = [48, 24, 6, 1]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<1x2x2x6xf16, {order = #NCHW, strides = [48, 24, 6, 1]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x1x3xf16, {order = #NHWC, strides = [48, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1072> -> memref<1x8x1x3xf16, {order = #NHWC, strides = [48, 1, 24, 8]}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   updates([[BARRIER_0]]
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:       inputs([[INPUT]]
    //CHECK-SAME:       outputs([[INPUT_BUFFER]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x1x2x3x2xf16, {order = #map, strides = [48, 24, 12, 6, 2, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x2x1x3xf16, {order = #map1, strides = [48, 4, 2, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<DEPTH_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x2x1x2x3x2xf16, {order = #map, strides = [48, 24, 12, 6, 2, 1]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x2x1x3xf16, {order = #map1, strides = [48, 4, 2, 1, 24, 8]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<DEPTH_FIRST>
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
    1x12x2x3x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

func.func @UnrollSegmentedClusterSpaceToDepthDMA() -> !OutputDistributed {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x4x6x!qElemType, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> <1024> -> !OutputDistributed

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>}
              inputs(%2 : memref<1x3x4x6x!qElemType, #NHWC, [@CMX_NN, 0]>)
              outputs(%3 : !OutputDistributed) -> !OutputDistributed
    }

    return %3: !OutputDistributed

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x2x6x!qElemType, {order = #NHWC, strides = [72, 1, 18, 3]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <36> -> memref<1x3x2x6x!qElemType, {order = #NHWC, strides = [72, 1, 18, 3]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> <1024> -> !VPUIP.DistributedBuffer<1x12x2x3x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x12x1x3x!qElemType, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <1024> -> memref<1x12x1x3x!qElemType, #NHWC, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x3x1x2x3x2x!qElemType, {order = #map, strides = [72, 1, 36, 18, 6, 3]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x3x1x3x!qElemType, #map1, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x3x1x2x3x2x!qElemType, {order = #map, strides = [72, 1, 36, 18, 6, 3]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x3x1x3x!qElemType, #map1, [@CMX_NN, 1]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT_BUFFER]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x24x24x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 1, top = 0, bottom = 1>,
    strides = [1, 1],
    num_clusters = 2
}>

func.func @UnrollOverlappedClusterSpaceToDepthDMA() -> !OutputDistributed {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x48x48x!qElemType, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> <65536> -> !OutputDistributed

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>}
              inputs(%2 : memref<1x4x48x48x!qElemType, #NHWC, [@CMX_NN, 0]>)
              outputs(%3 : !OutputDistributed) -> !OutputDistributed
    }

    return %3: !OutputDistributed

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x14x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x4x14x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4608> -> memref<1x4x12x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6912> -> memref<1x4x12x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> <65536> -> !VPUIP.DistributedBuffer<1x16x24x24x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <65536> -> memref<1x16x7x24x!qElemType, {order = #NHWC, strides = [5376, 1, 384, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <68224> -> memref<1x16x7x24x!qElemType, {order = #NHWC, strides = [5376, 1, 384, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <65536> -> memref<1x16x6x24x!qElemType, {order = #NHWC, strides = [4608, 1, 384, 16]}, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <67840> -> memref<1x16x6x24x!qElemType, {order = #NHWC, strides = [4608, 1, 384, 16]}, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x7x2x24x2x!qElemType, {order = #map, strides = [9216, 1, 384, 192, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x7x24x!qElemType, {order = #map1, strides = [5376, 8, 4, 1, 384, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x7x2x24x2x!qElemType, {order = #map, strides = [9216, 1, 384, 192, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x7x24x!qElemType, {order = #map1, strides = [5376, 8, 4, 1, 384, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x6x2x24x2x!qElemType, {order = #map, strides = [9216, 1, 384, 192, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x6x24x!qElemType, {order = #map1, strides = [4608, 8, 4, 1, 384, 16]}, [@CMX_NN, 1]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_2]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_2]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x6x2x24x2x!qElemType, {order = #map, strides = [9216, 1, 384, 192, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x6x24x!qElemType, {order = #map1, strides = [4608, 8, 4, 1, 384, 16]}, [@CMX_NN, 1]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_3]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_3]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT_BUFFER]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x24x24x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 1, top = 0, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>

func.func @UnrollOverlappedClusterSpaceToDepthDMA() -> !OutputDistributed {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x48x48x!qElemType, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> <65536> -> !OutputDistributed

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>}
              inputs(%2 : memref<1x4x48x48x!qElemType, #NHWC, [@CMX_NN, 0]>)
              outputs(%3 : !OutputDistributed) -> !OutputDistributed
    }

    return %3: !OutputDistributed

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x8x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1536> -> memref<1x4x8x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2304> -> memref<1x4x8x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <3840> -> memref<1x4x8x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4608> -> memref<1x4x8x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_5:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6144> -> memref<1x4x8x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_6:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6912> -> memref<1x4x6x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_7:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8064> -> memref<1x4x6x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> <65536> -> !VPUIP.DistributedBuffer<1x16x24x24x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <65536> -> memref<1x16x4x24x!qElemType, {order = #NHWC, strides = [3072, 1, 384, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <67072> -> memref<1x16x4x24x!qElemType, {order = #NHWC, strides = [3072, 1, 384, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <65536> -> memref<1x16x4x24x!qElemType, {order = #NHWC, strides = [3072, 1, 384, 16]}, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <67072> -> memref<1x16x4x24x!qElemType, {order = #NHWC, strides = [3072, 1, 384, 16]}, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT_BUFFER_4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <65536> -> memref<1x16x4x24x!qElemType, {order = #NHWC, strides = [3072, 1, 384, 16]}, [@CMX_NN, 2]>
    //CHECK:    [[OUTPUT_BUFFER_5:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <67072> -> memref<1x16x4x24x!qElemType, {order = #NHWC, strides = [3072, 1, 384, 16]}, [@CMX_NN, 2]>
    //CHECK:    [[OUTPUT_BUFFER_6:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <65536> -> memref<1x16x3x24x!qElemType, {order = #NHWC, strides = [2304, 1, 384, 16]}, [@CMX_NN, 3]>
    //CHECK:    [[OUTPUT_BUFFER_7:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <66688> -> memref<1x16x3x24x!qElemType, {order = #NHWC, strides = [2304, 1, 384, 16]}, [@CMX_NN, 3]>

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x4x2x24x2x!qElemType, {order = #map, strides = [9216, 1, 384, 192, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x4x24x!qElemType, {order = #map1, strides = [3072, 8, 4, 1, 384, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x4x2x24x2x!qElemType, {order = #map, strides = [9216, 1, 384, 192, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x4x24x!qElemType, {order = #map1, strides = [3072, 8, 4, 1, 384, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x4x2x24x2x!qElemType, {order = #map, strides = [9216, 1, 384, 192, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x4x24x!qElemType, {order = #map1, strides = [3072, 8, 4, 1, 384, 16]}, [@CMX_NN, 1]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_2]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_2]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x4x2x24x2x!qElemType, {order = #map, strides = [9216, 1, 384, 192, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x4x24x!qElemType, {order = #map1, strides = [3072, 8, 4, 1, 384, 16]}, [@CMX_NN, 1]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_3]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_3]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x4x2x24x2x!qElemType, {order = #map, strides = [9216, 1, 384, 192, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x4x24x!qElemType, {order = #map1, strides = [3072, 8, 4, 1, 384, 16]}, [@CMX_NN, 2]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_4]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_4]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x4x2x24x2x!qElemType, {order = #map, strides = [9216, 1, 384, 192, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x4x24x!qElemType, {order = #map1, strides = [3072, 8, 4, 1, 384, 16]}, [@CMX_NN, 2]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_5]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_5]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x3x2x24x2x!qElemType, {order = #map, strides = [9216, 1, 384, 192, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x3x24x!qElemType, {order = #map1, strides = [2304, 8, 4, 1, 384, 16]}, [@CMX_NN, 3]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       inputs([[INPUT_BUFFER_6]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_6]]
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK-SAME:   waits([[BARRIER_0]]
    //CHECK-SAME:   updates([[BARRIER_1]]
    //CHECK:        VPUIP.SpaceToDepthDMA
    //CHECK-SAME:       block_size = 2 : i64,
    //CHECK-SAME:       #VPUIP.InternalDataFlowAttr
    //CHECK-SAME:       inputType = memref<1x4x3x2x24x2x!qElemType, {order = #map, strides = [9216, 1, 384, 192, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:       outputType = memref<1x2x2x4x3x24x!qElemType, {order = #map1, strides = [2304, 8, 4, 1, 384, 16]}, [@CMX_NN, 3]>
    //CHECK-SAME:       mappingOrder = #map2
    //CHECK-SAME:       loopOrder = #map
    //CHECK-SAME:       mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       inputs([[INPUT_BUFFER_7]]
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_7]]
    //CHECK:    }

    //CHECK:    return [[OUTPUT_BUFFER]]
}
