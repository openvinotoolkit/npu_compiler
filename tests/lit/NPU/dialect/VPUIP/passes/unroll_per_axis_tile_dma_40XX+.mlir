//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-per-axis-tile-dma %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x512x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @UnrollPerAxisTileDMAWrapInClusterDUPLICATED
func.func @UnrollPerAxisTileDMAWrapInClusterDUPLICATED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <DDR> <64> -> memref<1x1x1x1xf16, #NHWC, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      VPUIP.PerAxisTileDMA {axis = 1 : i64, tiles = 512 : i64}
            inputs(%0 : memref<1x1x1x1xf16, #NHWC, @DDR>)
            outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1 : !OutputDistributed

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <DDR> <64> -> [[INPUT_TYPE_0:.+]]

    //CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> [[OUTPUT_TYPE:.+]]

    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> [[OUTPUT_TYPE_0:.+]]


    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA
    //CHECK-SAME:       axis = 1
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       tiles = 512
    //CHECK:                inputs([[INPUT_BUFFER_0]] : [[INPUT_TYPE_0]]
    //CHECK:                outputs([[OUTPUT_BUFFER_0]] : [[OUTPUT_TYPE_0]]

    //CHECK:    return [[OUTPUT]] : [[OUTPUT_TYPE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x512x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters =  2 : i64,
    compute_shapes = [[1, 512, 1, 1], [1, 512, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 512, 1, 1], [1, 512, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @UnrollPerAxisTileDMAWrapInClusterDUPLICATEDExplicitDistribution
func.func @UnrollPerAxisTileDMAWrapInClusterDUPLICATEDExplicitDistribution() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <DDR> <64> -> memref<1x1x1x1xf16, #NHWC, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      VPUIP.PerAxisTileDMA {axis = 1 : i64, tiles = 512 : i64}
            inputs(%0 : memref<1x1x1x1xf16, #NHWC, @DDR>)
            outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1 : !OutputDistributed

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <DDR> <64> -> [[INPUT_TYPE_0:.+]]

    //CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> [[OUTPUT_TYPE:.+]]

    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> [[OUTPUT_TYPE_0:.+]]


    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA
    //CHECK-SAME:       axis = 1
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       tiles = 512
    //CHECK:                inputs([[INPUT_BUFFER_0]] : [[INPUT_TYPE_0]]
    //CHECK:                outputs([[OUTPUT_BUFFER_0]] : [[OUTPUT_TYPE_0]]

    //CHECK:    return [[OUTPUT]] : [[OUTPUT_TYPE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x35x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64,
    alignment = [1, 1, 2, 1]
}>

// CHECK-LABEL: @UnrollPerAxisTileDMAWrapInClusterSEGMENTED
func.func @UnrollPerAxisTileDMAWrapInClusterSEGMENTED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <DDR> <64> -> memref<1x2x35x16xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      VPUIP.PerAxisTileDMA {axis = 1 : i64, port = 0 : i64, tiles = 8 : i64}
            inputs(%input : memref<1x2x35x16xf16, #NHWC, @DDR>)
            outputs(%output : !OutputDistributed) -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <DDR> <64> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <DDR> <1216> -> [[INPUT_TYPE_1:.+]]

    //CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> [[OUTPUT_TYPE:.+]]

    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> [[OUTPUT_TYPE_1:.+]]

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA {
    //CHECK-SAME:       axis = 1
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       tiles = 8
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]] : [[INPUT_TYPE_0]])
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]] : [[OUTPUT_TYPE_0]])
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA {
    //CHECK-SAME:       axis = 1
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       tiles = 8
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]] : [[INPUT_TYPE_1]])
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]] : [[OUTPUT_TYPE_1]])
    //CHECK:    }

    //CHECK:    return [[OUTPUT]] : [[OUTPUT_TYPE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x34x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64,
    alignment = [1, 1, 2, 1],
    compute_shapes = [[1, 16, 18, 16], [1, 16, 16, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0]],
    memory_shapes = [[1, 16, 18, 16], [1, 16, 16, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0]]
}>

// CHECK-LABEL: @UnrollPerAxisTileDMAExplicitSEGMENTED
func.func @UnrollPerAxisTileDMAExplicitSEGMENTED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <DDR> <64> -> memref<1x2x35x16xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      VPUIP.PerAxisTileDMA {axis = 1 : i64, port = 0 : i64, tiles = 8 : i64}
            inputs(%input : memref<1x2x35x16xf16, #NHWC, @DDR>)
            outputs(%output : !OutputDistributed) -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <DDR> <64> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <DDR> <1216> -> [[INPUT_TYPE_1:.+]]

    //CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> [[OUTPUT_TYPE:.+]]

    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> [[OUTPUT_TYPE_1:.+]]

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA {
    //CHECK-SAME:       axis = 1
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       tiles = 8
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]] : [[INPUT_TYPE_0]])
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]] : [[OUTPUT_TYPE_0]])
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA {
    //CHECK-SAME:       axis = 1
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       tiles = 8
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]] : [[INPUT_TYPE_1]])
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]] : [[OUTPUT_TYPE_1]])
    //CHECK:    }

    //CHECK:    return [[OUTPUT]] : [[OUTPUT_TYPE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x240x240xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 1, top = 0, bottom = 1>,
    strides = [2, 2],
    num_clusters = 2
}>

// CHECK-LABEL: @UnrollPerAxisTileDMAWrapInClusterOVERLAPPED
func.func @UnrollPerAxisTileDMAWrapInClusterOVERLAPPED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x240x120xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      VPUIP.PerAxisTileDMA {axis = 3 : i64, port = 0 : i64, tiles = 2 : i64}
            inputs(%input : memref<1x3x240x120xf16, #NHWC, @DDR>)
            outputs(%output : !OutputDistributed) -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <115200> -> [[INPUT_TYPE_1:.+]]

    //CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> [[OUTPUT_TYPE:.+]]

    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> [[OUTPUT_TYPE_1:.+]]

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA {
    //CHECK-SAME:       axis = 3
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       tiles = 2
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]] : [[INPUT_TYPE_0]])
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]] : [[OUTPUT_TYPE_0]])
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA {
    //CHECK-SAME:       axis = 3
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       tiles = 2
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]] : [[INPUT_TYPE_1]])
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]] : [[OUTPUT_TYPE_1]])
    //CHECK:    }

    //CHECK:    return [[OUTPUT]] : [[OUTPUT_TYPE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x240x240xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 4, 120, 240], [1, 4, 120, 240]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 120, 0]],
    memory_shapes = [[1, 4, 122, 240], [1, 4, 123, 240]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 117, 0]]
}>

// CHECK-LABEL: @UnrollPerAxisTileDMAExplicitOVERLAPPED
func.func @UnrollPerAxisTileDMAExplicitOVERLAPPED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x240x120xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      VPUIP.PerAxisTileDMA {axis = 3 : i64, port = 0 : i64, tiles = 2 : i64}
            inputs(%input : memref<1x3x240x120xf16, #NHWC, @DDR>)
            outputs(%output : !OutputDistributed) -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> [[INPUT_TYPE_0:.+]]
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <112320> -> [[INPUT_TYPE_1:.+]]

    //CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> [[OUTPUT_TYPE:.+]]

    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[OUTPUT_TYPE_0:.+]]
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> [[OUTPUT_TYPE_1:.+]]

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA {
    //CHECK-SAME:       axis = 3
    //CHECK-SAME:       port = 0
    //CHECK-SAME:       tiles = 2
    //CHECK-SAME:       inputs([[INPUT_BUFFER_0]] : [[INPUT_TYPE_0]])
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_0]] : [[OUTPUT_TYPE_0]])
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA {
    //CHECK-SAME:       axis = 3
    //CHECK-SAME:       port = 1
    //CHECK-SAME:       tiles = 2
    //CHECK-SAME:       inputs([[INPUT_BUFFER_1]] : [[INPUT_TYPE_1]])
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER_1]] : [[OUTPUT_TYPE_1]])
    //CHECK:    }

    //CHECK:    return [[OUTPUT]] : [[OUTPUT_TYPE]]
}
