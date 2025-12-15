//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --unroll-gather-dma  %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!IndicesDistributed = !VPUIP.DistributedBuffer<
    1x1x1024x1xi64, #NCHW, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1024, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1024, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x1x1024x2048xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1024, 683], [1, 1, 1024, 683], [1, 1, 1024, 682]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 683], [0, 0, 0, 1366]],
    memory_shapes = [[1, 1, 1024, 683], [1, 1, 1024, 683], [1, 1, 1024, 682]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 683], [0, 0, 0, 1366]]
}>

// Case 1: Unroll ClusterD2SDMA with single-cluster input and multi-cluster(SEGMENTED) output
func.func @UnrollGatherDMA() -> !OutputDistributed {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x128256x2048xf16, @DDR>
    %3 = VPURT.DeclareBuffer <CMX_NN> <16768> -> !IndicesDistributed
    %4 = VPURT.DeclareBuffer <CMX_NN> <24960> -> !OutputDistributed

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
        VPUIP.GatherDMA {channelType = 0 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}
            inputs(%2 : memref<1x1x128256x2048xf16, @DDR>)
            indices(%3 : !IndicesDistributed)
            outputs(%4 : !OutputDistributed) -> !OutputDistributed
    }

    return %4: !OutputDistributed

    //CHECK:    [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFF_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x128256x683xf16, {order = #NCHW, strides = [262668288, 262668288, 2048, 1]}, @DDR>
    //CHECK:    [[INPUT_BUFF_1:%.+]] = VPURT.DeclareBuffer <DDR> <1366> -> memref<1x1x128256x683xf16, {order = #NCHW, strides = [262668288, 262668288, 2048, 1]}, @DDR>
    //CHECK:    [[INPUT_BUFF_2:%.+]] = VPURT.DeclareBuffer <DDR> <2732> -> memref<1x1x128256x682xf16, {order = #NCHW, strides = [262668288, 262668288, 2048, 1]}, @DDR>

    //CHECK:    [[INDICES_BUFF_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <16768> -> memref<1x1x1024x1xi64, [@CMX_NN, 0]>
    //CHECK:    [[INDICES_BUFF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <16768> -> memref<1x1x1024x1xi64, [@CMX_NN, 1]>
    //CHECK:    [[INDICES_BUFF_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <16768> -> memref<1x1x1024x1xi64, [@CMX_NN, 2]>

    //CHECK:    [[OUTPUT_BUFF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <24960>
    //CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x1x1024x2048xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:       {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 1, 1024, 683], [1, 1, 1024, 683], [1, 1, 1024, 682]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 683], [0, 0, 0, 1366]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 1, 1024, 683], [1, 1, 1024, 683], [1, 1, 1024, 682]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 683], [0, 0, 0, 1366]]}>

    //CHECK:    [[OUTPUT_BUFF_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <24960> -> memref<1x1x1024x683xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <24960> -> memref<1x1x1024x683xf16, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT_BUFF_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <24960> -> memref<1x1x1024x682xf16, [@CMX_NN, 2]>

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.GatherDMA {elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}
    //CHECK-SAME:   inputs([[INPUT_BUFF_0]] : memref<1x1x128256x683xf16, {order = #NCHW, strides = [262668288, 262668288, 2048, 1]}, @DDR>)
    //CHECK-SAME:   indices([[INDICES_BUFF_0]] : memref<1x1x1024x1xi64, [@CMX_NN, 0]>)
    //CHECK-SAME:   outputs([[OUTPUT_BUFF_0]] : memref<1x1x1024x683xf16, [@CMX_NN, 0]>)
    //CHECK-SAME:   -> memref<1x1x1024x683xf16, [@CMX_NN, 0]>
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.GatherDMA {elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}
    //CHECK-SAME:   inputs([[INPUT_BUFF_1]] : memref<1x1x128256x683xf16, {order = #NCHW, strides = [262668288, 262668288, 2048, 1]}, @DDR>)
    //CHECK-SAME:   indices([[INDICES_BUFF_1]] : memref<1x1x1024x1xi64, [@CMX_NN, 1]>)
    //CHECK-SAME:   outputs([[OUTPUT_BUFF_1]] : memref<1x1x1024x683xf16, [@CMX_NN, 1]>)
    //CHECK-SAME:   -> memref<1x1x1024x683xf16, [@CMX_NN, 1]>
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.GatherDMA {elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}
    //CHECK-SAME:   inputs([[INPUT_BUFF_2]] : memref<1x1x128256x682xf16, {order = #NCHW, strides = [262668288, 262668288, 2048, 1]}, @DDR>)
    //CHECK-SAME:   indices([[INDICES_BUFF_2]] : memref<1x1x1024x1xi64, [@CMX_NN, 2]>)
    //CHECK-SAME:   outputs([[OUTPUT_BUFF_2]] : memref<1x1x1024x682xf16, [@CMX_NN, 2]>)
    //CHECK-SAME:   -> memref<1x1x1024x682xf16, [@CMX_NN, 2]>
    //CHECK:    }

    //CHECK:    return [[OUTPUT_BUFF]]
}
