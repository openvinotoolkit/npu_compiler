//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-per-axis-tile-dma %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

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

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT:%.+]] = VPURT.DeclareBuffer <DDR> <64> -> memref<1x1x1xf16, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x512x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x512x1xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PerAxisTileDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 1024 : i64, srcWidth = 2 : i64, srcStride = 0 : i64, srcPlaneStride = 2 : i64, dstWidth = 1024 : i64, dstStride = 1024 : i64, dstPlaneStride = 1024 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT]] : memref<1x1x1xf16, @DDR>
    //CHECK:                outputs([[OUTPUT]] : !VPUIP.DistributedBuffer<1x512x1xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

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

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT:%.+]] = VPURT.DeclareBuffer <DDR> <64> -> memref<1x1x1xf16, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0>
    //CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x512x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 512, 1, 1], [1, 512, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 512, 1, 1], [1, 512, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0>
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x512x1xf16, #CHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 512, 1], [1, 512, 1]], compute_offsets = [[0, 0, 0], [0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 512, 1], [1, 512, 1]], memory_offsets = [[0, 0, 0], [0, 0, 0]]}>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PerAxisTileDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:       numPlanes = 1 : i64, len = 1024 : i64,
    //CHECK-SAME:       srcWidth = 2 : i64, srcStride = 0 : i64, srcPlaneStride = 2 : i64,
    //CHECK-SAME:       dstWidth = 1024 : i64, dstStride = 1024 : i64, dstPlaneStride = 1024 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT]] : memref<1x1x1xf16, @DDR>
    //CHECK:                outputs([[OUTPUT]] : !VPUIP.DistributedBuffer<1x512x1xf16, #CHW, @CMX_NN,
    //CHECK-SAME:                                 {mode = "DUPLICATED", num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:                         compute_shapes = [[1, 512, 1], [1, 512, 1]],
    //CHECK-SAME{LITERAL}:                         compute_offsets = [[0, 0, 0], [0, 0, 0]],
    //CHECK-SAME{LITERAL}:                         memory_shapes = [[1, 512, 1], [1, 512, 1]],
    //CHECK-SAME{LITERAL}:                         memory_offsets = [[0, 0, 0], [0, 0, 0]]}>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer
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

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_0:%.+]] = VPURT.DeclareBuffer <DDR> <1088> -> memref<32x2x1xf16, @DDR>
    //CHECK:    [[INPUT_1:%.+]] = VPURT.DeclareBuffer <DDR> <64> -> memref<256x2x1xf16, @DDR>
    //CHECK:    [[INPUT_2:%.+]] = VPURT.DeclareBuffer <DDR> <2240> -> memref<16x2x1xf16, @DDR>
    //CHECK:    [[INPUT_3:%.+]] = VPURT.DeclareBuffer <DDR> <1216> -> memref<256x2x1xf16, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x35x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>
    //CHECK:    [[OUTPUT_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<32x16x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<256x16x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <8192> -> memref<16x16x1xf16, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<256x16x1xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PerAxisTileDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 256 : i64, len = 32 : i64, srcWidth = 4 : i64, srcStride = 0 : i64, srcPlaneStride = 4 : i64, dstWidth = 32 : i64, dstStride = 32 : i64, dstPlaneStride = 32 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT_1]] : memref<256x2x1xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_1]] : memref<256x16x1xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PerAxisTileDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 32 : i64, len = 32 : i64, srcWidth = 4 : i64, srcStride = 0 : i64, srcPlaneStride = 4 : i64, dstWidth = 32 : i64, dstStride = 32 : i64, dstPlaneStride = 32 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT_0]] : memref<32x2x1xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_0]] : memref<32x16x1xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PerAxisTileDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 256 : i64, len = 32 : i64, srcWidth = 4 : i64, srcStride = 0 : i64, srcPlaneStride = 4 : i64, dstWidth = 32 : i64, dstStride = 32 : i64, dstPlaneStride = 32 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT_3]] : memref<256x2x1xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_3]] : memref<256x16x1xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PerAxisTileDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 32 : i64, srcWidth = 4 : i64, srcStride = 0 : i64, srcPlaneStride = 4 : i64, dstWidth = 32 : i64, dstStride = 32 : i64, dstPlaneStride = 32 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT_2]] : memref<16x2x1xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_2]] : memref<16x16x1xf16, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer
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

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_0:%.+]] = VPURT.DeclareBuffer <DDR> <1088> -> memref<32x2x1xf16, @DDR>
    //CHECK:    [[INPUT_1:%.+]] = VPURT.DeclareBuffer <DDR> <64> -> memref<256x2x1xf16, @DDR>
    //CHECK:    [[INPUT_2:%.+]] = VPURT.DeclareBuffer <DDR> <1216> -> memref<256x2x1xf16, @DDR>

    //CHECK:    [[OUTDISTRIBUTION:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0>
    //CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x16x34x16xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1],
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 18, 16], [1, 16, 16, 16]], compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 18, 16], [1, 16, 16, 16]], memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0]]}>

    //CHECK:    [[OUTPUT_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<32x16x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<256x16x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<256x16x1xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PerAxisTileDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:       numPlanes = 256 : i64, len = 32 : i64,
    //CHECK-SAME:       srcWidth = 4 : i64, srcStride = 0 : i64, srcPlaneStride = 4 : i64,
    //CHECK-SAME:       dstWidth = 32 : i64, dstStride = 32 : i64, dstPlaneStride = 32 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT_1]] : memref<256x2x1xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_1]] : memref<256x16x1xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PerAxisTileDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:       numPlanes = 32 : i64, len = 32 : i64,
    //CHECK-SAME:       srcWidth = 4 : i64, srcStride = 0 : i64, srcPlaneStride = 4 : i64,
    //CHECK-SAME:       dstWidth = 32 : i64, dstStride = 32 : i64, dstPlaneStride = 32 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT_0]] : memref<32x2x1xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_0]] : memref<32x16x1xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PerAxisTileDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:       numPlanes = 256 : i64, len = 32 : i64,
    //CHECK-SAME:       srcWidth = 4 : i64, srcStride = 0 : i64, srcPlaneStride = 4 : i64,
    //CHECK-SAME:       dstWidth = 32 : i64, dstStride = 32 : i64, dstPlaneStride = 32 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT_2]] : memref<256x2x1xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_2]] : memref<256x16x1xf16, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer
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

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<121x120x4xf16, @DDR>
    //CHECK:    [[INPUT_1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <115200> -> memref<120x120x4xf16, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x4x240x240xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<121x240x4xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<120x240x4xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PerAxisTileDMA {
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:               numPlanes = 121 : i64, len = 1920 : i64,
    //CHECK-SAME:               srcWidth = 960 : i64, srcStride = 0 : i64, srcPlaneStride = 960 : i64,
    //CHECK-SAME:               dstWidth = 1920 : i64, dstStride = 1920 : i64, dstPlaneStride = 1920 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT_0]] : memref<121x120x4xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_0]] : memref<121x240x4xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PerAxisTileDMA {
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:               numPlanes = 120 : i64, len = 1920 : i64,
    //CHECK-SAME:               srcWidth = 960 : i64, srcStride = 0 : i64, srcPlaneStride = 960 : i64,
    //CHECK-SAME:               dstWidth = 1920 : i64, dstStride = 1920 : i64, dstPlaneStride = 1920 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT_1]] : memref<120x120x4xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_1]] : memref<120x240x4xf16, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer
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

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<122x120x4xf16, @DDR>
    //CHECK:    [[INPUT_1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <112320> -> memref<123x120x4xf16, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0>
    //CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x4x240x240xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 4, 120, 240], [1, 4, 120, 240]], compute_offsets = [[0, 0, 0, 0], [0, 0, 120, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 4, 122, 240], [1, 4, 123, 240]], memory_offsets = [[0, 0, 0, 0], [0, 0, 117, 0]]}>

    //CHECK:    [[OUTPUT_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<122x240x4xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<123x240x4xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PerAxisTileDMA {
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:               numPlanes = 122 : i64, len = 1920 : i64,
    //CHECK-SAME:               srcWidth = 960 : i64, srcStride = 0 : i64, srcPlaneStride = 960 : i64,
    //CHECK-SAME:               dstWidth = 1920 : i64, dstStride = 1920 : i64, dstPlaneStride = 1920 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT_0]] : memref<122x120x4xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_0]] : memref<122x240x4xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PerAxisTileDMA {
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:               numPlanes = 123 : i64, len = 1920 : i64,
    //CHECK-SAME:               srcWidth = 960 : i64, srcStride = 0 : i64, srcPlaneStride = 960 : i64,
    //CHECK-SAME:               dstWidth = 1920 : i64, dstStride = 1920 : i64, dstPlaneStride = 1920 : i64>, port = 1 : i64}
    //CHECK:                inputs([[INPUT_1]] : memref<123x120x4xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_1]] : memref<123x240x4xf16, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer
}
