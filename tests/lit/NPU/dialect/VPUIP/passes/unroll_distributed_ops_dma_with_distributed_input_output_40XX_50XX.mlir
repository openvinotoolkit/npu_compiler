//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-distributed-ops --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

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
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%input_cmx_buffer : !SubviewDistributedType) outputs(%output_cmx_buffer : !OutputDistributedType) -> !OutputDistributedType
    }

    return

    // CHECK:    [[INPUT_BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x268x31x!qElemType, {order = #NHWC, strides = [819200, 1, 1024, 32]}, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x32x269x31x!qElemType, {order = #NHWC, strides = [819200, 1, 1024, 32]}, [@CMX_NN, 1]>
    // CHECK:    [[INPUT_BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x32x267x31x!qElemType, {order = #NHWC, strides = [819200, 1, 1024, 32]}, [@CMX_NN, 2]>
    // CHECK:    [[OUTPUT_BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <266848> -> memref<1x32x268x31x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <266848> -> memref<1x32x269x31x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    // CHECK:    [[OUTPUT_BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <266848> -> memref<1x32x267x31x!qElemType, {order = #NHWC}, [@CMX_NN, 2]>
    // CHECK:    [[BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:      [[DMA0:%.+]] = VPUIP.NNDMA {unrollIdx = 0 : i64} <{port = 0 : i64}> inputs([[INPUT_BUF0]] : memref<1x32x268x31x!qElemType, {order = #NHWC, strides = [819200, 1, 1024, 32]}, [@CMX_NN, 0]>)
    // CHECK-SAME: outputs([[OUTPUT_BUF0]] : memref<1x32x268x31x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x268x31x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:      [[DMA1:%.+]] = VPUIP.NNDMA {unrollIdx = 0 : i64} <{port = 1 : i64}> inputs([[INPUT_BUF1]] : memref<1x32x269x31x!qElemType, {order = #NHWC, strides = [819200, 1, 1024, 32]}, [@CMX_NN, 1]>)
    // CHECK-SAME: outputs([[OUTPUT_BUF1]] : memref<1x32x269x31x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x32x269x31x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:      [[DMA2:%.+]] = VPUIP.NNDMA {unrollIdx = 0 : i64} <{port = 0 : i64}> inputs([[INPUT_BUF2]] : memref<1x32x267x31x!qElemType, {order = #NHWC, strides = [819200, 1, 1024, 32]}, [@CMX_NN, 2]>)
    // CHECK-SAME: outputs([[OUTPUT_BUF2]] : memref<1x32x267x31x!qElemType, {order = #NHWC}, [@CMX_NN, 2]>) -> memref<1x32x267x31x!qElemType, {order = #NHWC}, [@CMX_NN, 2]>
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
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%input_cmx_buffer : !SubviewDistributedType) outputs(%output_cmx_buffer : !OutputDistributedType) -> !OutputDistributedType
    }

    return

    // CHECK:    [[INPUT_BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x267x29x!qElemType, {order = #NHWC, strides = [896000, 1, 1120, 32]}, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <1120> -> memref<1x32x267x29x!qElemType, {order = #NHWC, strides = [896000, 1, 1120, 32]}, [@CMX_NN, 1]>
    // CHECK:    [[INPUT_BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <1120> -> memref<1x32x266x29x!qElemType, {order = #NHWC, strides = [896000, 1, 1120, 32]}, [@CMX_NN, 2]>
    // CHECK:    [[OUTPUT_BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <266848> -> memref<1x32x267x29x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <266848> -> memref<1x32x267x29x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    // CHECK:    [[OUTPUT_BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <266848> -> memref<1x32x266x29x!qElemType, {order = #NHWC}, [@CMX_NN, 2]>
    // CHECK:    [[BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:      [[DMA0:%.+]] = VPUIP.NNDMA {unrollIdx = 0 : i64} <{port = 0 : i64}> inputs([[INPUT_BUF0]] : memref<1x32x267x29x!qElemType, {order = #NHWC, strides = [896000, 1, 1120, 32]}, [@CMX_NN, 0]>)
    // CHECK-SAME: outputs([[OUTPUT_BUF0]] : memref<1x32x267x29x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x32x267x29x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:      [[DMA1:%.+]] = VPUIP.NNDMA {unrollIdx = 0 : i64} <{port = 1 : i64}> inputs([[INPUT_BUF1]] : memref<1x32x267x29x!qElemType, {order = #NHWC, strides = [896000, 1, 1120, 32]}, [@CMX_NN, 1]>)
    // CHECK-SAME: outputs([[OUTPUT_BUF1]] : memref<1x32x267x29x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x32x267x29x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    // CHECK:      [[DMA2:%.+]] = VPUIP.NNDMA {unrollIdx = 0 : i64} <{port = 0 : i64}> inputs([[INPUT_BUF2]] : memref<1x32x266x29x!qElemType, {order = #NHWC, strides = [896000, 1, 1120, 32]}, [@CMX_NN, 2]>)
    // CHECK-SAME: outputs([[OUTPUT_BUF2]] : memref<1x32x266x29x!qElemType, {order = #NHWC}, [@CMX_NN, 2]>) -> memref<1x32x266x29x!qElemType, {order = #NHWC}, [@CMX_NN, 2]>
    // CHECK:    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    12x4x3x3xf16,    {
        order = #NCHW, strides = [144, 9, 3, 1] // the most outer stride is 144 to emulate slicing of the buffer 12x16x3x3xf16
    }, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[12, 4, 1, 3], [12, 4, 1, 3], [12, 4, 1, 3]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]],
        memory_shapes = [[12, 4, 1, 3], [12, 4, 1, 3], [12, 4, 1, 3]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]
}>

!Output_DDR = memref<12x4x3x3xf16, @DDR>

//CHECK-LABEL: @UnrollDistributedOpsDMADistributedInputNCHW
func.func @UnrollDistributedOpsDMADistributedInputNCHW() -> !Output_DDR {
    // test case also emulates slicing when previous SW/HW op produces bigger output, for example
    // %output_sw = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<12x16x3x3xf16, ...

    // same CMX offset, but "sliced" shape: 12x4x3x3xf16
    %input = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %output = VPURT.DeclareBuffer <DDR> <0> -> !Output_DDR
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%input : !InputDistributed) outputs(%output : !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    // CHECK: [[INPUT_CMX0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<12x4x1x3xf16, {order = #NCHW, strides = [48, 3, 3, 1]}, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<12x4x1x3xf16, {order = #NCHW, strides = [48, 3, 3, 1]}, [@CMX_NN, 1]>
    // CHECK: [[INPUT_CMX2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<12x4x1x3xf16, {order = #NCHW, strides = [48, 3, 3, 1]}, [@CMX_NN, 2]>

    // CHECK: [[OUTPUT_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<12x4x3x3xf16, @DDR>
    // CHECK: [[OUTPUT_DDR0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<12x4x1x3xf16, {order = #NCHW, strides = [36, 9, 3, 1]}, @DDR>
    // CHECK: [[OUTPUT_DDR1:%.+]] = VPURT.DeclareBuffer <DDR> <6> -> memref<12x4x1x3xf16, {order = #NCHW, strides = [36, 9, 3, 1]}, @DDR>
    // CHECK: [[OUTPUT_DDR2:%.+]] = VPURT.DeclareBuffer <DDR> <12> -> memref<12x4x1x3xf16, {order = #NCHW, strides = [36, 9, 3, 1]}, @DDR>
    // CHECK: [[BAR:%.+]] = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR]] : !VPURT.Barrier)  {
    // CHECK:   VPUIP.NNDMA {unrollIdx = 0 : i64} <{port = 0 : i64}> inputs([[INPUT_CMX0]] : memref<12x4x1x3xf16, {order = #NCHW, strides = [48, 3, 3, 1]}, [@CMX_NN, 0]>)
    // CHECK:           outputs([[OUTPUT_DDR0]] : memref<12x4x1x3xf16, {order = #NCHW, strides = [36, 9, 3, 1]}, @DDR>) -> memref<12x4x1x3xf16, {order = #NCHW, strides = [36, 9, 3, 1]}, @DDR>

    // CHECK: VPURT.Task updates([[BAR]] : !VPURT.Barrier)  {
    // CHECK:   VPUIP.NNDMA {unrollIdx = 0 : i64} <{port = 1 : i64}> inputs([[INPUT_CMX1]] : memref<12x4x1x3xf16, {order = #NCHW, strides = [48, 3, 3, 1]}, [@CMX_NN, 1]>)
    // CHECK:           outputs([[OUTPUT_DDR1]] : memref<12x4x1x3xf16, {order = #NCHW, strides = [36, 9, 3, 1]}, @DDR>) -> memref<12x4x1x3xf16, {order = #NCHW, strides = [36, 9, 3, 1]}, @DDR>

    // CHECK: VPURT.Task updates([[BAR]] : !VPURT.Barrier)  {
    // CHECK:   VPUIP.NNDMA {unrollIdx = 0 : i64} <{port = 0 : i64}> inputs([[INPUT_CMX2]] : memref<12x4x1x3xf16, {order = #NCHW, strides = [48, 3, 3, 1]}, [@CMX_NN, 2]>)
    // CHECK:           outputs([[OUTPUT_DDR2]] : memref<12x4x1x3xf16, {order = #NCHW, strides = [36, 9, 3, 1]}, @DDR>) -> memref<12x4x1x3xf16, {order = #NCHW, strides = [36, 9, 3, 1]}, @DDR>

    // CHECK: return [[OUTPUT_DDR]] : memref<12x4x3x3xf16, @DDR>
}
