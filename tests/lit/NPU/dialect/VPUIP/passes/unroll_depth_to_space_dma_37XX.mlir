//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-depth-to-space-dma  %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollDepthToSpaceDMABlockFirstNHWC
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x8x2x3xf16, {order = #NHWC}>, [[ARG_1:%.+]]: memref<1x2x4x6xf16, {order = #NHWC}>)
func.func @UnrollDepthToSpaceDMABlockFirstNHWC(%input: memref<1x8x2x3xf16, {order = #NHWC}>, %output: memref<1x2x4x6xf16, {order = #NHWC}>) -> memref<1x2x4x6xf16, {order = #NHWC}> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %inBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %outBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%input : memref<1x8x2x3xf16, {order = #NHWC}>) outputs(%inBuffer : memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.DepthToSpaceDMA {output_channel = 2 : i64, output_width = 6 : i64} <{block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}>
                inputs(%inBuffer : memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                outputs(%outBuffer : memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%outBuffer : memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%output :  memref<1x2x4x6xf16, {order = #NHWC}>) -> memref<1x2x4x6xf16, {order = #NHWC}>
    }

    return %output: memref<1x2x4x6xf16, {order = #NHWC}>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8> -> memref<1x4x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <152> -> memref<1x4x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x4x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs([[ARG_0]] : memref<1x8x2x3xf16, {order = #NHWC}>)
    //CHECK:            outputs([[INPUT_BUFFER]] : memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.DepthToSpaceDMA <{block_size = 2 : i64,
    //CHECK:            dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 24 : i64, srcWidth = 8 : i64, srcStride = 16 : i64, srcPlaneStride = 48 : i64, dstWidth = 24 : i64, dstStride = 2 : i64, dstPlaneStride = 48 : i64>,
    //CHECK:            mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64}>
    //CHECK:            inputs([[INPUT_0]] : memref<1x4x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_0]] : memref<1x4x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x4x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.DepthToSpaceDMA <{block_size = 2 : i64,
    //CHECK:            dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 24 : i64, srcWidth = 8 : i64, srcStride = 16 : i64, srcPlaneStride = 48 : i64, dstWidth = 24 : i64, dstStride = 2 : i64, dstPlaneStride = 48 : i64>,
    //CHECK:            mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 1 : i64}>
    //CHECK:            inputs([[INPUT_1]] : memref<1x4x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_1]] : memref<1x4x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x4x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs([[OUTPUT_BUFFER]] : memref<1x2x4x6xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:            outputs([[ARG_1]] : memref<1x2x4x6xf16, {order = #NHWC}>) -> memref<1x2x4x6xf16, {order = #NHWC}>
    //CHECK:    }

    //CHECK:    return [[ARG_1]] : memref<1x2x4x6xf16, {order = #NHWC}>
}

// CHECK-LABEL: @UnrollDepthToSpaceDMADepthFirstNHWC
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x8x2x1xf16, {order = #NHWC}>, [[ARG_1:%.+]]: memref<1x2x4x2xf16, {order = #NHWC}>)
func.func @UnrollDepthToSpaceDMADepthFirstNHWC(%input: memref<1x8x2x1xf16, {order = #NHWC}>, %output: memref<1x2x4x2xf16, {order = #NHWC}>) -> memref<1x2x4x2xf16, {order = #NHWC}> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %inBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %outBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x2x4x2xf16, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%input : memref<1x8x2x1xf16, {order = #NHWC}>) outputs(%inBuffer : memref<1x8x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x8x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.DepthToSpaceDMA {output_channel = 2 : i64, output_width = 2 : i64} <{block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}>
                inputs(%inBuffer : memref<1x8x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                outputs(%outBuffer : memref<1x2x4x2xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x4x2xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%outBuffer : memref<1x2x4x2xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%output :  memref<1x2x4x2xf16, {order = #NHWC}>) -> memref<1x2x4x2xf16, {order = #NHWC}>
    }

    return %output: memref<1x2x4x2xf16, {order = #NHWC}>

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4> -> memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8> -> memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <12> -> memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x2x4x2xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <136> -> memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <130> -> memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <138> -> memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs([[ARG_0]] : memref<1x8x2x1xf16, {order = #NHWC}>)
    //CHECK:            outputs([[INPUT_BUFFER]] : memref<1x8x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x8x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.DepthToSpaceDMA <{block_size = 2 : i64,
    //CHECK:            dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 4 : i64, srcWidth = 4 : i64, srcStride = 16 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 4 : i64, dstPlaneStride = 16 : i64>,
    //CHECK:            mode = #IE.depth_to_space_mode<DEPTH_FIRST>, port = 0 : i64}>
    //CHECK:            inputs([[INPUT_0]] : memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_0]] : memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.DepthToSpaceDMA <{block_size = 2 : i64,
    //CHECK:            dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 4 : i64, srcWidth = 4 : i64, srcStride = 16 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 4 : i64, dstPlaneStride = 16 : i64>,
    //CHECK:            mode = #IE.depth_to_space_mode<DEPTH_FIRST>, port = 1 : i64}>
    //CHECK:            inputs([[INPUT_1]] : memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_1]] : memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    }

   //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.DepthToSpaceDMA <{block_size = 2 : i64,
    //CHECK:            dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 4 : i64, srcWidth = 4 : i64, srcStride = 16 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 4 : i64, dstPlaneStride = 16 : i64>,
    //CHECK:            mode = #IE.depth_to_space_mode<DEPTH_FIRST>, port = 0 : i64}>
    //CHECK:            inputs([[INPUT_2]] : memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_2]] : memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.DepthToSpaceDMA <{block_size = 2 : i64,
    //CHECK:            dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 4 : i64, srcWidth = 4 : i64, srcStride = 16 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 4 : i64, dstPlaneStride = 16 : i64>,
    //CHECK:            mode = #IE.depth_to_space_mode<DEPTH_FIRST>, port = 1 : i64}>
    //CHECK:            inputs([[INPUT_3]] : memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_3]] : memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x2x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs([[OUTPUT_BUFFER]] : memref<1x2x4x2xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:            outputs([[ARG_1]] : memref<1x2x4x2xf16, {order = #NHWC}>) -> memref<1x2x4x2xf16, {order = #NHWC}>
    //CHECK:    }

    //CHECK:    return [[ARG_1]] : memref<1x2x4x2xf16, {order = #NHWC}>
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
// CHECK-LABEL: @UnrollSegmentedClusterDepthToSpaceDMACase1
func.func @UnrollSegmentedClusterDepthToSpaceDMACase1() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x12x2x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}>
              inputs(%0 : memref<1x12x2x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[INPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <42> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <36> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTDISTRIBUTION:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x3x4x6x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-DAG:    [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <18> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <18> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUTPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 18 : i64, srcWidth = 6 : i64, srcStride = 12 : i64, srcPlaneStride = 36 : i64, dstWidth = 18 : i64, dstStride = 1 : i64, dstPlaneStride = 36 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64}>
    //CHECK:                inputs([[INPUT1]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT1]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 18 : i64, srcWidth = 6 : i64, srcStride = 12 : i64, srcPlaneStride = 36 : i64, dstWidth = 18 : i64, dstStride = 1 : i64, dstPlaneStride = 36 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 1 : i64}>
    //CHECK:                inputs([[INPUT0]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT0]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 18 : i64, srcWidth = 6 : i64, srcStride = 12 : i64, srcPlaneStride = 36 : i64, dstWidth = 18 : i64, dstStride = 1 : i64, dstPlaneStride = 36 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64}>
    //CHECK:                inputs([[INPUT3]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT3]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 18 : i64, srcWidth = 6 : i64, srcStride = 12 : i64, srcPlaneStride = 36 : i64, dstWidth = 18 : i64, dstStride = 1 : i64, dstPlaneStride = 36 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 1 : i64}>
    //CHECK:                inputs([[INPUT2]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT2]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x3x4x6x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
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
// CHECK-LABEL: @UnrollSegmentedClusterDepthToSpaceDMACase2
func.func @UnrollSegmentedClusterDepthToSpaceDMACase2() -> memref<1x3x4x6x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x4x6x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}>
              inputs(%0 :  !InputDistributed)
              outputs(%1 : memref<1x3x4x6x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x3x4x6x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    }
    return %1: memref<1x3x4x6x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[INPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <6> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[INPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUTSINGLECMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x4x6x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <18> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <54> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <36> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 18 : i64, srcWidth = 6 : i64, srcStride = 12 : i64, srcPlaneStride = 36 : i64, dstWidth = 18 : i64, dstStride = 1 : i64, dstPlaneStride = 36 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64}>
    //CHECK:                inputs([[INPUT1]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT1]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 18 : i64, srcWidth = 6 : i64, srcStride = 12 : i64, srcPlaneStride = 36 : i64, dstWidth = 18 : i64, dstStride = 1 : i64, dstPlaneStride = 36 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 1 : i64}>
    //CHECK:                inputs([[INPUT0]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT0]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 18 : i64, srcWidth = 6 : i64, srcStride = 12 : i64, srcPlaneStride = 36 : i64, dstWidth = 18 : i64, dstStride = 1 : i64, dstPlaneStride = 36 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64}>
    //CHECK:                inputs([[INPUT3]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:                outputs([[OUTPUT3]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 18 : i64, srcWidth = 6 : i64, srcStride = 12 : i64, srcPlaneStride = 36 : i64, dstWidth = 18 : i64, dstStride = 1 : i64, dstPlaneStride = 36 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 1 : i64}>
    //CHECK:                inputs([[INPUT2]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:                outputs([[OUTPUT2]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    return [[OUTSINGLECMX]] : memref<1x3x4x6x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
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
// CHECK-LABEL: @UnrollSegmentedClusterDepthToSpaceDMACase3
func.func @UnrollSegmentedClusterDepthToSpaceDMACase3() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}>
              inputs(%0 :  !InputDistributed)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[INPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <6> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[INPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUTDISTRIBUTION:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x3x4x6x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-DAG:    [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <18> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <18> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUTPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 18 : i64, srcWidth = 6 : i64, srcStride = 12 : i64, srcPlaneStride = 36 : i64, dstWidth = 18 : i64, dstStride = 1 : i64, dstPlaneStride = 36 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64}>
    //CHECK:                inputs([[INPUT1]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT1]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 18 : i64, srcWidth = 6 : i64, srcStride = 12 : i64, srcPlaneStride = 36 : i64, dstWidth = 18 : i64, dstStride = 1 : i64, dstPlaneStride = 36 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 1 : i64}>
    //CHECK:                inputs([[INPUT0]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT0]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 18 : i64, srcWidth = 6 : i64, srcStride = 12 : i64, srcPlaneStride = 36 : i64, dstWidth = 18 : i64, dstStride = 1 : i64, dstPlaneStride = 36 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64}>
    //CHECK:                inputs([[INPUT3]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:                outputs([[OUTPUT3]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 18 : i64, srcWidth = 6 : i64, srcStride = 12 : i64, srcPlaneStride = 36 : i64, dstWidth = 18 : i64, dstStride = 1 : i64, dstPlaneStride = 36 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 1 : i64}>
    //CHECK:                inputs([[INPUT2]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:                outputs([[OUTPUT2]] : memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x6x1x3x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x3x4x6x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
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
// CHECK-LABEL: @UnrollSegmentedClusterDepthToSpaceDMACase4
func.func @UnrollSegmentedClusterDepthToSpaceDMACase4() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64, padded_channels = #IE.ChannelPadding<input = 4: i64, output = 1: i64>}>
              inputs(%0 :  !InputDistributed)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[INPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6> -> memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <6> -> memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[INPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUTDISTRIBUTION:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x4x256x256x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-DAG:    [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <1024> -> memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUTPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 64 : i64, len = 768 : i64, srcWidth = 6 : i64, srcStride = 16 : i64, srcPlaneStride = 2048 : i64, dstWidth = 3 : i64, dstStride = 4 : i64, dstPlaneStride = 2048 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64}>
    //CHECK:                inputs([[INPUT1]] : memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT1]] : memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 64 : i64, len = 768 : i64, srcWidth = 6 : i64, srcStride = 16 : i64, srcPlaneStride = 2048 : i64, dstWidth = 3 : i64, dstStride = 4 : i64, dstPlaneStride = 2048 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 1 : i64}>
    //CHECK:                inputs([[INPUT0]] : memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT0]] : memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 64 : i64, len = 768 : i64, srcWidth = 6 : i64, srcStride = 16 : i64, srcPlaneStride = 2048 : i64, dstWidth = 3 : i64, dstStride = 4 : i64, dstPlaneStride = 2048 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64}>
    //CHECK:                inputs([[INPUT3]] : memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:                outputs([[OUTPUT3]] : memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 64 : i64, len = 768 : i64, srcWidth = 6 : i64, srcStride = 16 : i64, srcPlaneStride = 2048 : i64, dstWidth = 3 : i64, dstStride = 4 : i64, dstPlaneStride = 2048 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 1 : i64}>
    //CHECK:                inputs([[INPUT2]] : memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:                outputs([[OUTPUT2]] : memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x8x64x128x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x4x256x256x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}



// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollDepthToSpaceWithBlockFirstLargeH
func.func @UnrollDepthToSpaceWithBlockFirstLargeH() -> memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x267x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <36352> -> memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %6 = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 1 : i64}>
    inputs(%0 : memref<1x4x267x17xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    outputs(%1 : memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
    return %1: memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]>


    //CHECK:       [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:       [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:       [[INPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <18092> -> memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[INPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <18088> -> memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[INPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4> -> memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[INPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <36352> -> memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <54508> -> memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <54440> -> memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[OUTPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <36420> -> memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[OUTPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <36352> -> memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:            [[DTS_DMA:%.+]] = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 133 : i64, len = 68 : i64, srcWidth = 4 : i64, srcStride = 8 : i64, srcPlaneStride = 136 : i64, dstWidth = 68 : i64, dstStride = 2 : i64, dstPlaneStride = 136 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64}> inputs([[INPUT3]] : memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUTPUT3]] : memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:         }

    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:        [[DTS_DMA:%.+]] = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 133 : i64, len = 68 : i64, srcWidth = 4 : i64, srcStride = 8 : i64, srcPlaneStride = 136 : i64, dstWidth = 68 : i64, dstStride = 2 : i64, dstPlaneStride = 136 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 1 : i64}> inputs([[INPUT2]] : memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUTPUT2]] : memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:        [[DTS_DMA:%.+]] = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 134 : i64, len = 68 : i64, srcWidth = 4 : i64, srcStride = 8 : i64, srcPlaneStride = 136 : i64, dstWidth = 68 : i64, dstStride = 2 : i64, dstPlaneStride = 136 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64}> inputs([[INPUT1]] : memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUTPUT1]] : memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:        [[DTS_DMA:%.+]] = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 134 : i64, len = 68 : i64, srcWidth = 4 : i64, srcStride = 8 : i64, srcPlaneStride = 136 : i64, dstWidth = 68 : i64, dstStride = 2 : i64, dstPlaneStride = 136 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 1 : i64}> inputs([[INPUT0]] : memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUTPUT0]] : memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        }

    //CHECK:        return [[OUTPUT]] : memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollDepthToSpaceWithDepthFirstLargeH
func.func @UnrollDepthToSpaceWithDepthFirstLargeH() -> memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x267x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <36352> -> memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %6 = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, port = 1 : i64}>
    inputs(%0 : memref<1x4x267x17xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    outputs(%1 : memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    return %1: memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:       [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:       [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:       [[INPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <18092> -> memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[INPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <18088> -> memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[INPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4> -> memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[INPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <36352> -> memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <54508> -> memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <54440> -> memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[OUTPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <36420> -> memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:       [[OUTPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <36352> -> memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:        [[DTS_DMA:%.+]] = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 133 : i64, len = 68 : i64, srcWidth = 4 : i64, srcStride = 8 : i64, srcPlaneStride = 136 : i64, dstWidth = 2 : i64, dstStride = 2 : i64, dstPlaneStride = 136 : i64>, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, port = 0 : i64}> inputs([[INPUT3]] : memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUTPUT3]] : memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:        [[DTS_DMA:%.+]] = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 133 : i64, len = 68 : i64, srcWidth = 4 : i64, srcStride = 8 : i64, srcPlaneStride = 136 : i64, dstWidth = 2 : i64, dstStride = 2 : i64, dstPlaneStride = 136 : i64>, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, port = 1 : i64}> inputs([[INPUT2]] : memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUTPUT2]] : memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x133x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        }
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:        [[DTS_DMA:%.+]] = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 134 : i64, len = 68 : i64, srcWidth = 4 : i64, srcStride = 8 : i64, srcPlaneStride = 136 : i64, dstWidth = 2 : i64, dstStride = 2 : i64, dstPlaneStride = 136 : i64>, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, port = 0 : i64}> inputs([[INPUT1]] : memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUTPUT1]] : memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:        [[DTS_DMA:%.+]] = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 134 : i64, len = 68 : i64, srcWidth = 4 : i64, srcStride = 8 : i64, srcPlaneStride = 136 : i64, dstWidth = 2 : i64, dstStride = 2 : i64, dstPlaneStride = 136 : i64>, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, port = 1 : i64}> inputs([[INPUT0]] : memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUTPUT0]] : memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x2x134x17xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        }

    //CHECK:        return [[OUTPUT]] : memref<1x1x534x34xf16, {order = #NHWC}, [@CMX_NN, 0]>
}
