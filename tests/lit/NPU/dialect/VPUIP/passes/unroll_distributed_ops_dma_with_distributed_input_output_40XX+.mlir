//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-distributed-ops --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.012266390931372549:116>

!SubviewDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x31x!qElemType, {order = #NHWC, strides = [819200, 1, 1024, 32]}, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 267, 31], [1, 32, 267, 31], [1, 32, 266, 31]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]],
    memory_shapes = [[1, 32, 268, 31], [1, 32, 269, 31], [1, 32, 267, 31]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]
}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x31x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 267, 31], [1, 32, 267, 31], [1, 32, 266, 31]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]],
    memory_shapes = [[1, 32, 268, 31], [1, 32, 269, 31], [1, 32, 267, 31]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]
}>

// CHECK-LABEL: @UnrollDistributedOpsDMADistributedInputOutput
func.func @UnrollDistributedOpsDMADistributedInputOutput() {
    %input_cmx_buffer = VPURT.DeclareBuffer <CMX_NN> <0> -> !SubviewDistributedType
    %output_cmx_buffer = VPURT.DeclareBuffer <CMX_NN> <266848> -> !OutputDistributedType

    %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%input_cmx_buffer : !SubviewDistributedType) outputs(%output_cmx_buffer : !OutputDistributedType) -> !OutputDistributedType
    }

    return

    // CHECK:    [[INPUT_BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x268x31x!qElemType, {order = #NHWC, strides = [274432, 1, 1024, 32]}, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x32x269x31x!qElemType, {order = #NHWC, strides = [275456, 1, 1024, 32]}, [@CMX_NN, 1]>
    // CHECK:    [[INPUT_BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x32x267x31x!qElemType, {order = #NHWC, strides = [273408, 1, 1024, 32]}, [@CMX_NN, 2]>
    // CHECK:    [[OUTPUT_BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <266848> -> memref<1x32x268x31x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <266848> -> memref<1x32x269x31x!qElemType, #NHWC, [@CMX_NN, 1]>
    // CHECK:    [[OUTPUT_BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <266848> -> memref<1x32x267x31x!qElemType, #NHWC, [@CMX_NN, 2]>
    // CHECK:    [[BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:      [[DMA0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[INPUT_BUF0]] : memref<1x32x268x31x!qElemType, {order = #NHWC, strides = [274432, 1, 1024, 32]}, [@CMX_NN, 0]>)
    // CHECK-SAME: outputs([[OUTPUT_BUF0]] : memref<1x32x268x31x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x268x31x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:      [[DMA1:%.+]] = VPUIP.NNDMA {port = 1 : i64} inputs([[INPUT_BUF1]] : memref<1x32x269x31x!qElemType, {order = #NHWC, strides = [275456, 1, 1024, 32]}, [@CMX_NN, 1]>)
    // CHECK-SAME: outputs([[OUTPUT_BUF1]] : memref<1x32x269x31x!qElemType, #NHWC, [@CMX_NN, 1]>) -> memref<1x32x269x31x!qElemType, #NHWC, [@CMX_NN, 1]>
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:      [[DMA2:%.+]] = VPUIP.NNDMA {port = 0 : i64, split_candidate} inputs([[INPUT_BUF2]] : memref<1x32x267x31x!qElemType, {order = #NHWC, strides = [273408, 1, 1024, 32]}, [@CMX_NN, 2]>)
    // CHECK-SAME: outputs([[OUTPUT_BUF2]] : memref<1x32x267x31x!qElemType, #NHWC, [@CMX_NN, 2]>) -> memref<1x32x267x31x!qElemType, #NHWC, [@CMX_NN, 2]>
    // CHECK:    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.012266390931372549:116>

!SubviewDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x29x!qElemType, {order = #NHWC, strides = [896000, 1, 1120, 32]}, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 268, 29], [1, 32, 269, 29], [1, 32, 267, 29]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]],
    memory_shapes = [[1, 32, 268, 29], [1, 32, 269, 29], [1, 32, 267, 29]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]
}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x29x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 267, 29], [1, 32, 267, 29], [1, 32, 266, 29]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]],
    memory_shapes = [[1, 32, 267, 29], [1, 32, 267, 29], [1, 32, 266, 29]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]]
}>

// CHECK-LABEL: @UnrollDistributedOpsDMADistributedInputOutputWithDifferentMemoryView
func.func @UnrollDistributedOpsDMADistributedInputOutputWithDifferentMemoryView() {
    %input_cmx_buffer = VPURT.DeclareBuffer <CMX_NN> <0> -> !SubviewDistributedType
    %output_cmx_buffer = VPURT.DeclareBuffer <CMX_NN> <266848> -> !OutputDistributedType

    %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%input_cmx_buffer : !SubviewDistributedType) outputs(%output_cmx_buffer : !OutputDistributedType) -> !OutputDistributedType
    }

    return

    // CHECK:    [[INPUT_BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x267x29x!qElemType, {order = #NHWC, strides = [299040, 1, 1120, 32]}, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <1120> -> memref<1x32x267x29x!qElemType, {order = #NHWC, strides = [299040, 1, 1120, 32]}, [@CMX_NN, 1]>
    // CHECK:    [[INPUT_BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <1120> -> memref<1x32x266x29x!qElemType, {order = #NHWC, strides = [297920, 1, 1120, 32]}, [@CMX_NN, 2]>
    // CHECK:    [[OUTPUT_BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <266848> -> memref<1x32x267x29x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <266848> -> memref<1x32x267x29x!qElemType, #NHWC, [@CMX_NN, 1]>
    // CHECK:    [[OUTPUT_BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <266848> -> memref<1x32x266x29x!qElemType, #NHWC, [@CMX_NN, 2]>
    // CHECK:    [[BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:      [[DMA0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[INPUT_BUF0]] : memref<1x32x267x29x!qElemType, {order = #NHWC, strides = [299040, 1, 1120, 32]}, [@CMX_NN, 0]>)
    // CHECK-SAME: outputs([[OUTPUT_BUF0]] : memref<1x32x267x29x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x267x29x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:      [[DMA1:%.+]] = VPUIP.NNDMA {port = 1 : i64} inputs([[INPUT_BUF1]] : memref<1x32x267x29x!qElemType, {order = #NHWC, strides = [299040, 1, 1120, 32]}, [@CMX_NN, 1]>)
    // CHECK-SAME: outputs([[OUTPUT_BUF1]] : memref<1x32x267x29x!qElemType, #NHWC, [@CMX_NN, 1]>) -> memref<1x32x267x29x!qElemType, #NHWC, [@CMX_NN, 1]>
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:      [[DMA2:%.+]] = VPUIP.NNDMA {port = 0 : i64, split_candidate} inputs([[INPUT_BUF2]] : memref<1x32x266x29x!qElemType, {order = #NHWC, strides = [297920, 1, 1120, 32]}, [@CMX_NN, 2]>)
    // CHECK-SAME: outputs([[OUTPUT_BUF2]] : memref<1x32x266x29x!qElemType, #NHWC, [@CMX_NN, 2]>) -> memref<1x32x266x29x!qElemType, #NHWC, [@CMX_NN, 2]>
    // CHECK:    }
}
