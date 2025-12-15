//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dma-unrolling %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

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

    //CHECK:    #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
    //CHECK:    #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
    //CHECK:    #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

    //CHECK-LABEL:  @UnrollS2DepthAndUpsamplingDMA

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x8x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1536> -> memref<1x4x8x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2304> -> memref<1x4x8x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <3840> -> memref<1x4x8x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4608> -> memref<1x4x8x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_5:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6144> -> memref<1x4x8x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_6:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6912> -> memref<1x4x6x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_7:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8064> -> memref<1x4x6x48x!qElemType, {order = #NHWC, strides = [9216, 1, 192, 4]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTDISTRIBUTION:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x24x24x!qElemType, #NHWC, @CMX_NN
    //CHECK-SAME:   {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
    //CHECK-SAME:   strides = [1, 1], num_clusters = 4 : i64}>

    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x24x!qElemType, {order = #NHWC, strides = [3072, 1, 384, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1536> -> memref<1x16x4x24x!qElemType, {order = #NHWC, strides = [3072, 1, 384, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x4x24x!qElemType, {order = #NHWC, strides = [3072, 1, 384, 16]}, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT_BUFFER_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <1536> -> memref<1x16x4x24x!qElemType, {order = #NHWC, strides = [3072, 1, 384, 16]}, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT_BUFFER_4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x16x4x24x!qElemType, {order = #NHWC, strides = [3072, 1, 384, 16]}, [@CMX_NN, 2]>
    //CHECK:    [[OUTPUT_BUFFER_5:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <1536> -> memref<1x16x4x24x!qElemType, {order = #NHWC, strides = [3072, 1, 384, 16]}, [@CMX_NN, 2]>
    //CHECK:    [[OUTPUT_BUFFER_6:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <0> -> memref<1x16x3x24x!qElemType, {order = #NHWC, strides = [2304, 1, 384, 16]}, [@CMX_NN, 3]>
    //CHECK:    [[OUTPUT_BUFFER_7:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <1152> -> memref<1x16x3x24x!qElemType, {order = #NHWC, strides = [2304, 1, 384, 16]}, [@CMX_NN, 3]>

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

    //CHECK:    [[INPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x90x1xf16, [@DDR, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x60x1xf16, @DDR>
    //CHECK:    VPURT.Task waits([[BARRIER_2]] : !VPURT.Barrier) updates([[BARRIER_3]] : !VPURT.Barrier)
    //CHECK:        VPUIP.UpsamplingDMAOp {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:       numPlanes = 90 : i64,
    //CHECK-SAME:       len = 2 : i64,
    //CHECK-SAME:       srcWidth = 2 : i64, srcStride = 2 : i64, srcPlaneStride = 2 : i64,
    //CHECK-SAME:       dstWidth = 2 : i64, dstStride = 2 : i64, dstPlaneStride = 4 : i64>,
    //CHECK-SAME:       expand = [0, 13, 0, 0],
    //CHECK-SAME:       port = 0 : i64,
    //CHECK-SAME:       upsampling_factor = [1, 1, 2, 1]
    //CHECK-SAME:   }
    //CHECK-SAME:   inputs([[INPUT_BUFFER]] : memref<1x1x90x1xf16, [@DDR, 0]>)
    //CHECK-SAME:   outputs([[OUTPUT_BUFFER]] : memref<1x16x60x1xf16, @DDR>) -> memref<1x16x60x1xf16, @DDR>
    //CHECK:    }

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x16x24x24x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
}
