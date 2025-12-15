//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-expand-dma %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x240x320xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @UnrollExpandDMAWithSEGMENTED
func.func @UnrollExpandDMAWithSEGMENTED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x240x320xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
                inputs(%input : memref<1x3x240x320xf16, #NHWC, @DDR>)
                outputs(%output : !OutputDistributed) -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x120x320xf16, #NHWC, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <230400> -> memref<1x3x120x320xf16, #NHWC, @DDR>
    //CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x4x240x320xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 230400 : i64, srcWidth = 230400 : i64, srcStride = 230400 : i64, srcPlaneStride = 0 : i64, dstWidth = 6 : i64, dstStride = 8 : i64, dstPlaneStride = 0 : i64>,
    //CHECK:                pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<1x3x120x320xf16, #NHWC, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 230400 : i64, srcWidth = 230400 : i64, srcStride = 230400 : i64, srcPlaneStride = 0 : i64, dstWidth = 6 : i64, dstStride = 8 : i64, dstPlaneStride = 0 : i64>,
    //CHECK:                pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 1 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<1x3x120x320xf16, #NHWC, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:    return [[OUTPUT]] : !VPUIP.DistributedBuffer<1x4x240x320xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x240x320xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64,
    compute_shapes = [[1, 4, 120, 320], [1, 4, 120, 320]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 120, 0]],
    memory_shapes = [[1, 4, 120, 320], [1, 4, 120, 320]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 120, 0]]
}>

// CHECK-LABEL: @UnrollExpandDMAWithSEGMENTEDExplicit
func.func @UnrollExpandDMAWithSEGMENTEDExplicit() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x240x320xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !OutputDistributed

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
                inputs(%input : memref<1x3x240x320xf16, #NHWC, @DDR>)
                outputs(%output : !OutputDistributed) -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x120x320xf16, #NHWC, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <230400> -> memref<1x3x120x320xf16, #NHWC, @DDR>
    //CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0>
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x4x240x320xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 4, 120, 320], [1, 4, 120, 320]], compute_offsets = [[0, 0, 0, 0], [0, 0, 120, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 4, 120, 320], [1, 4, 120, 320]], memory_offsets = [[0, 0, 0, 0], [0, 0, 120, 0]]}>

    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {
    //CHECK-SAME:       pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 0 : i64}
    //CHECK-SAME:       inputs([[INPUT0]] : memref<1x3x120x320xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[OUTPUT0]] : memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 230400 : i64, srcWidth = 230400 : i64, srcStride = 230400 : i64, srcPlaneStride = 0 : i64, dstWidth = 6 : i64, dstStride = 8 : i64, dstPlaneStride = 0 : i64>,
    //CHECK-SAME:       pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 1 : i64}
    //CHECK-SAME:       inputs([[INPUT1]] : memref<1x3x120x320xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[OUTPUT1]] : memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:    return [[OUTPUT]] : !VPUIP.DistributedBuffer<1x4x240x320xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                   {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 4, 120, 320], [1, 4, 120, 320]], compute_offsets = [[0, 0, 0, 0], [0, 0, 120, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 4, 120, 320], [1, 4, 120, 320]], memory_offsets = [[0, 0, 0, 0], [0, 0, 120, 0]]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4432x1x2xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

// CHECK-LABEL: @UnrollExpandDMAWithDUPLICATED
func.func @UnrollExpandDMAWithDUPLICATED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4420x1x2xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
                inputs(%input : memref<1x4420x1x2xf16, #NHWC, @DDR>)
                outputs(%output : !OutputDistributed) -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4420x1x2xf16, #NHWC, @DDR>
    //CHECK:    [[RETURN:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x4432x1x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x4432x1x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 17680 : i64, srcWidth = 17680 : i64, srcStride = 17680 : i64, srcPlaneStride = 0 : i64, dstWidth = 8840 : i64, dstStride = 8864 : i64, dstPlaneStride = 0 : i64>,
    //CHECK:                pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0], port = 0 : i64}
    //CHECK:                inputs([[INPUT]] : memref<1x4420x1x2xf16, #NHWC, @DDR>)
    //CHECK:                outputs([[OUTPUT]] : !VPUIP.DistributedBuffer<1x4432x1x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:                    -> !VPUIP.DistributedBuffer<1x4432x1x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:    return [[RETURN]] : !VPUIP.DistributedBuffer<1x4432x1x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4432x1x2xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 4432, 1, 2], [1, 4432, 1, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 4432, 1, 2], [1, 4432, 1, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @UnrollExpandDMAWithDUPLICATEDExplicit
func.func @UnrollExpandDMAWithDUPLICATEDExplicit() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4420x1x2xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
                inputs(%input : memref<1x4420x1x2xf16, #NHWC, @DDR>)
                outputs(%output : !OutputDistributed) -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4420x1x2xf16, #NHWC, @DDR>
    //CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0>
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x4432x1x2xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 4432, 1, 2], [1, 4432, 1, 2]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 4432, 1, 2], [1, 4432, 1, 2]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 17680 : i64, srcWidth = 17680 : i64, srcStride = 17680 : i64, srcPlaneStride = 0 : i64, dstWidth = 8840 : i64, dstStride = 8864 : i64, dstPlaneStride = 0 : i64>,
    //CHECK-SAME:       pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0], port = 0 : i64}
    //CHECK-SAME:       inputs([[INPUT]] : memref<1x4420x1x2xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[OUTPUT]] : !VPUIP.DistributedBuffer<1x4432x1x2xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                             {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:                     compute_shapes = [[1, 4432, 1, 2], [1, 4432, 1, 2]],
    //CHECK-SAME{LITERAL}:                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:                     memory_shapes = [[1, 4432, 1, 2], [1, 4432, 1, 2]],
    //CHECK-SAME{LITERAL}:                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK:                    -> !VPUIP.DistributedBuffer<1x4432x1x2xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                   {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:           compute_shapes = [[1, 4432, 1, 2], [1, 4432, 1, 2]],
    //CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:           memory_shapes = [[1, 4432, 1, 2], [1, 4432, 1, 2]],
    //CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x240x320xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 1, top = 0, bottom = 1>,
    strides = [2, 2],
    num_clusters = 2
}>

// CHECK-LABEL: @UnrollExpandDMAWithOVERLAPPED
func.func @UnrollExpandDMAWithOVERLAPPED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x240x320xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
                inputs(%input : memref<1x3x240x320xf16, #NHWC, @DDR>)
                outputs(%output : !OutputDistributed) -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x121x320xf16, #NHWC, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <230400> -> memref<1x3x120x320xf16, #NHWC, @DDR>
    //CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x4x240x320xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64}>

    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x121x320xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 232320 : i64, srcWidth = 232320 : i64, srcStride = 232320 : i64, srcPlaneStride = 0 : i64, dstWidth = 6 : i64, dstStride = 8 : i64, dstPlaneStride = 0 : i64>,
    //CHECK:                pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<1x3x121x320xf16, #NHWC, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<1x4x121x320xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x121x320xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 230400 : i64, srcWidth = 230400 : i64, srcStride = 230400 : i64, srcPlaneStride = 0 : i64, dstWidth = 6 : i64, dstStride = 8 : i64, dstPlaneStride = 0 : i64>,
    //CHECK:                pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 1 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<1x3x120x320xf16, #NHWC, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:    return [[OUTPUT]] : !VPUIP.DistributedBuffer<1x4x240x320xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x240x320xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 4, 120, 320], [1, 4, 120, 320]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 120, 0]],
    memory_shapes = [[1, 4, 122, 320], [1, 4, 123, 320]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 117, 0]]
}>

// CHECK-LABEL: @UnrollExpandDMAWithOVERLAPPEDExplicit
func.func @UnrollExpandDMAWithOVERLAPPEDExplicit() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x240x320xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !OutputDistributed

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
                inputs(%input : memref<1x3x240x320xf16, #NHWC, @DDR>)
                outputs(%output : !OutputDistributed) -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x122x320xf16, #NHWC, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <224640> -> memref<1x3x123x320xf16, #NHWC, @DDR>
    //CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0>
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x4x240x320xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 4, 120, 320], [1, 4, 120, 320]], compute_offsets = [[0, 0, 0, 0], [0, 0, 120, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 4, 122, 320], [1, 4, 123, 320]], memory_offsets = [[0, 0, 0, 0], [0, 0, 117, 0]]}>

    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x122x320xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x4x123x320xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 234240 : i64, srcWidth = 234240 : i64, srcStride = 234240 : i64, srcPlaneStride = 0 : i64, dstWidth = 6 : i64, dstStride = 8 : i64, dstPlaneStride = 0 : i64>,
    //CHECK-SAME:       pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 0 : i64}
    //CHECK-SAME:       inputs([[INPUT0]] : memref<1x3x122x320xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[OUTPUT0]] : memref<1x4x122x320xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       -> memref<1x4x122x320xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 236160 : i64, srcWidth = 236160 : i64, srcStride = 236160 : i64, srcPlaneStride = 0 : i64, dstWidth = 6 : i64, dstStride = 8 : i64, dstPlaneStride = 0 : i64>,
    //CHECK-SAME:       pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 1 : i64}
    //CHECK-SAME:       inputs([[INPUT1]] : memref<1x3x123x320xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[OUTPUT1]] : memref<1x4x123x320xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       -> memref<1x4x123x320xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:    return [[OUTPUT]] : !VPUIP.DistributedBuffer<1x4x240x320xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    //CHECK-SAME{LITERAL}:            compute_shapes = [[1, 4, 120, 320], [1, 4, 120, 320]], compute_offsets = [[0, 0, 0, 0], [0, 0, 120, 0]],
    //CHECK-SAME{LITERAL}:            memory_shapes = [[1, 4, 122, 320], [1, 4, 123, 320]], memory_offsets = [[0, 0, 0, 0], [0, 0, 117, 0]]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0040670955882352944:128>

// CHECK-LABEL: @UnrollExpandDMAWithInOutStridesOnExpandAxis
func.func @UnrollExpandDMAWithInOutStridesOnExpandAxis() -> memref<1x16x40x40x!qElemType, {order = #NHWC, strides = [76800, 1, 1920, 48]}, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <DDR> <0> -> memref<1x8x40x40x!qElemType, {order = #NHWC, strides = [64000, 1, 1600, 40]}, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <18432000> -> memref<1x16x40x40x!qElemType, {order = #NHWC, strides = [76800, 1, 1920, 48]}, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        %0 = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]}
                inputs(%input : memref<1x8x40x40x!qElemType, {order = #NHWC, strides = [64000, 1, 1600, 40]}, @DDR>)
                outputs(%output : memref<1x16x40x40x!qElemType, {order = #NHWC, strides = [76800, 1, 1920, 48]}, @DDR>)
                -> memref<1x16x40x40x!qElemType, {order = #NHWC, strides = [76800, 1, 1920, 48]}, @DDR>
    }

    return %output: memref<1x16x40x40x!qElemType, {order = #NHWC, strides = [76800, 1, 1920, 48]}, @DDR>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x8x40x40x!qElemType, {order = #NHWC, strides = [64000, 1, 1600, 40]}, @DDR>
    //CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <DDR> <18432000> -> memref<1x16x40x40x!qElemType, {order = #NHWC, strides = [76800, 1, 1920, 48]}, @DDR>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 12800 : i64, srcWidth = 8 : i64, srcStride = 40 : i64, srcPlaneStride = 0 : i64, dstWidth = 8 : i64, dstStride = 48 : i64, dstPlaneStride = 0 : i64>,
    //CHECK-SAME:     pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0], port = 0 : i64}
    //CHECK:                inputs([[INPUT]] : memref<1x8x40x40x!qElemType, {order = #NHWC, strides = [64000, 1, 1600, 40]}, @DDR>)
    //CHECK:                outputs([[OUTPUT]] : memref<1x16x40x40x!qElemType, {order = #NHWC, strides = [76800, 1, 1920, 48]}, @DDR>)
    //CHECK-SAME:       -> memref<1x16x40x40x!qElemType, {order = #NHWC, strides = [76800, 1, 1920, 48]}, @DDR>

    //CHECK:    return [[OUTPUT]] : memref<1x16x40x40x!qElemType, {order = #NHWC, strides = [76800, 1, 1920, 48]}, @DDR>
}

