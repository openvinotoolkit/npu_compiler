//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dma-unrolling %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

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

// CHECK-LABEL: @UnrollS2DepthAndUpsamplingDMA
func.func @UnrollS2DepthAndUpsamplingDMA() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %inputToS2D = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x48x48x!qElemType, #NHWC, [@CMX_NN, 0]>
    %outputOfS2D = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>}
              inputs(%inputToS2D : memref<1x4x48x48x!qElemType, #NHWC, [@CMX_NN, 0]>)
              outputs(%outputOfS2D : !OutputDistributed) -> !OutputDistributed
    }


    %inputOfUpsDMA = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x30x1xf16, @DDR>
    %outputOfUpsDMA = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x60x1xf16, @DDR>
    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3 : !VPURT.Barrier) {
        %upsDMA = VPUIP.UpsamplingDMAOp {expand = [0, 13, 0, 0], port = 1 : i64, upsampling_factor = [1, 1, 2, 1]}
          inputs(%inputOfUpsDMA : memref<1x3x30x1xf16, @DDR>)
          outputs(%outputOfUpsDMA : memref<1x16x60x1xf16, @DDR>) -> memref<1x16x60x1xf16, @DDR>
    }


    return %outputOfS2D: !OutputDistributed

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2304> -> memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4608> -> memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6912> -> memref<1x4x12x48x!qElemType, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTDISTRIBUTION:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x24x24x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
    //CHECK:    [[OUTPUT0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 2]>
    //CHECK:    [[OUTPUT3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <0> -> memref<1x4x12x48x!qElemType, #NHWC, [@CMX_NN, 3]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 1536 : i64, srcWidth = 192 : i64, srcStride = 384 : i64, srcPlaneStride = 192 : i64, dstWidth = 8 : i64, dstStride = 16 : i64, dstPlaneStride = 8 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT0]] : memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 1536 : i64, srcWidth = 192 : i64, srcStride = 384 : i64, srcPlaneStride = 192 : i64, dstWidth = 8 : i64, dstStride = 16 : i64, dstPlaneStride = 8 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 1 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT1]] : memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 1]>) -> memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 1536 : i64, srcWidth = 192 : i64, srcStride = 384 : i64, srcPlaneStride = 192 : i64, dstWidth = 8 : i64, dstStride = 16 : i64, dstPlaneStride = 8 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64}
    //CHECK:                inputs([[INPUT2]] : memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT2]] : memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 2]>) -> memref<1x4x16x48x!qElemType, #NHWC, [@CMX_NN, 2]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 1152 : i64, srcWidth = 192 : i64, srcStride = 384 : i64, srcPlaneStride = 192 : i64, dstWidth = 8 : i64, dstStride = 16 : i64, dstPlaneStride = 8 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 1 : i64}
    //CHECK:                inputs([[INPUT3]] : memref<1x4x12x48x!qElemType, #NHWC, [@CMX_NN, 0]>)
    //CHECK:                outputs([[OUTPUT3]] : memref<1x4x12x48x!qElemType, #NHWC, [@CMX_NN, 3]>) -> memref<1x4x12x48x!qElemType, #NHWC, [@CMX_NN, 3]>

    // CHECK:   [[INPUT:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x90x1xf16, [@DDR, 0]>
    // CHECK:   [[OUTPUT4:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x60x1xf16, @DDR>
    // CHECK:       VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:          numPlanes = 90 : i64,
    // CHECK-SAME:          len = 2 : i64,
    // CHECK-SAME:          srcWidth = 2 : i64, srcStride = 2 : i64, srcPlaneStride = 2 : i64,
    // CHECK-SAME:          dstWidth = 2 : i64, dstStride = 2 : i64, dstPlaneStride = 4 : i64>,
    // CHECK-SAME:          expand = [0, 13, 0, 0],
    // CHECK-SAME:          port = 0 : i64,
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 1]
    // CHECK-SAME:      }
    // CHECK-SAME:        inputs([[INPUT]] : memref<1x1x90x1xf16, [@DDR, 0]>)
    // CHECK-SAME:        outputs([[OUTPUT4]] : memref<1x16x60x1xf16, @DDR>) -> memref<1x16x60x1xf16, @DDR>
    // CHECK:       }

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x16x24x24x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
}
