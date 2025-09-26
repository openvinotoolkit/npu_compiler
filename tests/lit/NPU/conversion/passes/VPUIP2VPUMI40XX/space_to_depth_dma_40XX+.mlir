//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

module @spaceToDepth {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x4x6x12xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x3x6xf16>
  }

  func.func @main(%arg0: memref<1x4x6x12xf16, #NHWC>, %arg1: memref<1x16x3x6xf16, #NHWC>) -> memref<1x16x3x6xf16, #NHWC> {
    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %2 = VPURT.ConfigureBarrier<2> <{isFinalBarrier}> -> !VPURT.Barrier

    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x2x12xf16, {order = #NHWC, strides = [288, 1, 48, 4]}, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <192> -> memref<1x4x4x12xf16, {order = #NHWC, strides = [288, 1, 48, 4]}, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x16x1x6xf16, {order = #NHWC, strides = [288, 1, 96, 16]}, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <1216> -> memref<1x16x2x6xf16, {order = #NHWC, strides = [288, 1, 96, 16]}, [@CMX_NN, 0]>

    //CHECK:    [[INPUT:%.+]]: memref<1x4x6x12xf16, #NHWC>
    //CHECK:    [[OUTPUT:%.+]]: memref<1x16x3x6xf16, #NHWC>

    //CHECK:    [[BARRIER_0:%.+]] = VPUMI40XX.ConfigureBarrier
    //CHECK:    [[BARRIER_1:%.+]] = VPUMI40XX.ConfigureBarrier
    //CHECK:    [[BARRIER_2:%.+]] = VPUMI40XX.ConfigureBarrier

    //CHECK:    [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x2x12xf16, {order = #NHWC, strides = [288, 1, 48, 4]}, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <192> -> memref<1x4x4x12xf16, {order = #NHWC, strides = [288, 1, 48, 4]}, [@CMX_NN, 0]>

    //CHECK:    [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x16x1x6xf16, {order = #NHWC, strides = [288, 1, 96, 16]}, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1216> -> memref<1x16x2x6xf16, {order = #NHWC, strides = [288, 1, 96, 16]}, [@CMX_NN, 0]>

    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %9 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x4x6x12xf16, #NHWC>) outputs(%3 : memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>
    }

    //CHECK-NOT:  VPUIP.NNDMA
    //CHECK:      [[IN_DMA:%.+]] = VPUMI40XX.NNDMA
    //CHECK-SAME: port = 0 : i64
    //CHECK-SAME: inputs([[INPUT]]
    //CHECK-SAME: outputs([[INPUT_BUFFER_0]]
    //CHECK-SAME: updates([[BARRIER_0]]
    //CHECK-SAME: start_after(0)
    //CHECK-SAME: clean_after(0)
    //CHECK-SAME: acceleration_mode(<DISABLE>)
    //CHECK-SAME: dma_transaction
    //CHECK-SAME:     #VPUMI40XX.NNDMATransaction
    //CHECK-SAME:         inputType = memref<1x4x6x12xf16, #NHWC>
    //CHECK-SAME:         outputType = memref<1x4x6x12xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-SAME: -> !VPURegMapped.Index<0:0:0>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %9 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, internalDataFlow = #VPUIP.InternalDataFlowAttr<inputType = memref<1x4x1x2x6x2xf16, {order = #map, strides = [288, 1, 96, 48, 8, 4]}, [@CMX_NN, 0]>, outputType = memref<1x2x2x4x1x6xf16, {order = #map1, strides = [288, 8, 4, 1, 96, 16]}, [@CMX_NN, 0]>, mappingOrder = #map2, loopOrder = #map>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64} inputs(%4 : memref<1x4x2x12xf16, {order = #NHWC, strides = [288, 1, 48, 4]}, [@CMX_NN, 0]>) outputs(%7 : memref<1x16x1x6xf16, {order = #NHWC, strides = [288, 1, 96, 16]}, [@CMX_NN, 0]>) -> memref<1x16x1x6xf16, {order = #NHWC, strides = [288, 1, 96, 16]}, [@CMX_NN, 0]>
    }

    //CHECK-NOT:  VPUIP.NNDMA
    //CHECK:      [[S2D_DMA_0:%.+]] = VPUMI40XX.NNDMA
    //CHECK-SAME: allow_different_in_out_shapes
    //CHECK-SAME: port = 0 : i64
    //CHECK-SAME: inputs([[INPUT_BUFFER_1]]
    //CHECK-SAME: outputs([[OUTPUT_BUFFER_1]]
    //CHECK-SAME: waits([[BARRIER_0]]
    //CHECK-SAME: updates([[BARRIER_1]]
    //CHECK-SAME: start_after(0)
    //CHECK-SAME: clean_after(0)
    //CHECK-SAME: acceleration_mode(<DISABLE>)
    //CHECK-SAME: dma_transaction
    //CHECK-SAME:     #VPUMI40XX.PermuteDMATransaction
    //CHECK-SAME:         inputType = memref<1x4x1x2x6x2xf16, {order = #map, strides = [288, 1, 96, 48, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:         outputType = memref<1x2x2x4x1x6xf16, {order = #map1, strides = [288, 8, 4, 1, 96, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:         mappingOrder = #map2
    //CHECK-SAME:         loopOrder = #map
    //CHECK-SAME: -> !VPURegMapped.Index<0:1:0>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %9 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, internalDataFlow = #VPUIP.InternalDataFlowAttr<inputType = memref<1x4x2x2x6x2xf16, {order = #map, strides = [288, 1, 96, 48, 8, 4]}, [@CMX_NN, 0]>, outputType = memref<1x2x2x4x2x6xf16, {order = #map1, strides = [288, 8, 4, 1, 96, 16]}, [@CMX_NN, 0]>, mappingOrder = #map2, loopOrder = #map>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 1 : i64} inputs(%5 : memref<1x4x4x12xf16, {order = #NHWC, strides = [288, 1, 48, 4]}, [@CMX_NN, 0]>) outputs(%8 : memref<1x16x2x6xf16, {order = #NHWC, strides = [288, 1, 96, 16]}, [@CMX_NN, 0]>) -> memref<1x16x2x6xf16, {order = #NHWC, strides = [288, 1, 96, 16]}, [@CMX_NN, 0]>
    }

    //CHECK-NOT:  VPUIP.NNDMA
    //CHECK:      [[S2D_DMA_1:%.+]] = VPUMI40XX.NNDMA
    //CHECK-SAME: port = 1 : i64
    //CHECK-SAME: inputs([[INPUT_BUFFER_2]]
    //CHECK-SAME: outputs([[OUTPUT_BUFFER_2]]
    //CHECK-SAME: waits([[BARRIER_0]]
    //CHECK-SAME: updates([[BARRIER_1]]
    //CHECK-SAME: start_after(0)
    //CHECK-SAME: clean_after(0)
    //CHECK-SAME: acceleration_mode(<DISABLE>)
    //CHECK-SAME: dma_transaction
    //CHECK-SAME:     #VPUMI40XX.PermuteDMATransaction
    //CHECK-SAME:         inputType = memref<1x4x2x2x6x2xf16, {order = #map, strides = [288, 1, 96, 48, 8, 4]}, [@CMX_NN, 0]>
    //CHECK-SAME:         outputType = memref<1x2x2x4x2x6xf16, {order = #map1, strides = [288, 8, 4, 1, 96, 16]}, [@CMX_NN, 0]>
    //CHECK-SAME:         mappingOrder = #map2
    //CHECK-SAME:         loopOrder = #map
    //CHECK-SAME: -> !VPURegMapped.Index<1:1:0>

    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) {
      %9 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x3x6xf16, #NHWC>) -> memref<1x16x3x6xf16, #NHWC>
    }

    //CHECK-NOT:  VPUIP.NNDMA
    //CHECK:      [[OUT_DMA:%.+]] = VPUMI40XX.NNDMA
    //CHECK-SAME: port = 0 : i64
    //CHECK-SAME: inputs([[OUTPUT_BUFFER_0]]
    //CHECK-SAME: outputs([[OUTPUT]]
    //CHECK-SAME: previousDMA([[S2D_DMA_0]]
    //CHECK-SAME: waits([[BARRIER_1]]
    //CHECK-SAME: updates([[BARRIER_2]]
    //CHECK-SAME: start_after(0)
    //CHECK-SAME: clean_after(0)
    //CHECK-SAME: acceleration_mode(<DISABLE>)
    //CHECK-SAME: dma_transaction
    //CHECK-SAME:     #VPUMI40XX.NNDMATransaction
    //CHECK-SAME:         inputType = memref<1x16x3x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-SAME:         outputType = memref<1x16x3x6xf16, #NHWC>
    //CHECK-SAME: -> !VPURegMapped.Index<0:1:1>

    return %arg1 : memref<1x16x3x6xf16, #NHWC>
  }
}
