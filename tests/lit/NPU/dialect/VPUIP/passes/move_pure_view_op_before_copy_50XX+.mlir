//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --move-pure-view-op-before-copy %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!InputOverlapped = !VPUIP.DistributedBuffer<
    1x16x240x200xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 60, 0], [0, 0, 120, 0], [0, 0, 180, 0]],
    memory_shapes = [[1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 60, 0], [0, 0, 120, 0], [0, 0, 180, 0]]
}>

// CHECK-LABEL: @MoveShapeCastBeforeCopyOverlappedReshapeMultiUses
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x240x200xf16, #NHWC, @CMX_NN,
func.func @MoveShapeCastBeforeCopyOverlappedReshapeMultiUses(%arg0: !InputOverlapped) -> (!InputOverlapped, memref<1x1x1500x512xf16, {order = #NHWC}, @DDR>) {
    %alloc_0 = memref.alloc() : memref<1x16x240x200xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy inputs(%arg0 : !InputOverlapped)
        outputs(%alloc_0 : memref<1x16x240x200xf16, {order = #NHWC}, @DDR>)
        -> memref<1x16x240x200xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.ShapeCast {shape = [1, 1, 1500, 512]}
        inputs(%0 : memref<1x16x240x200xf16, {order = #NHWC}, @DDR>)
        -> memref<1x1x1500x512xf16, {order = #NHWC}, @DDR>

    return %arg0, %1 : !InputOverlapped, memref<1x1x1500x512xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 1, 1500, 512]}
    // CHECK-SAME:      inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x240x200xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200]], compute_offsets = [[0, 0, 0, 0], [0, 0, 60, 0], [0, 0, 120, 0], [0, 0, 180, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200]], memory_offsets = [[0, 0, 0, 0], [0, 0, 60, 0], [0, 0, 120, 0], [0, 0, 180, 0]]}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x1x1500x512xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 375, 512], [1, 1, 375, 512], [1, 1, 375, 512], [1, 1, 375, 512]], compute_offsets = [[0, 0, 0, 0], [0, 0, 375, 0], [0, 0, 750, 0], [0, 0, 1125, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 375, 512], [1, 1, 375, 512], [1, 1, 375, 512], [1, 1, 375, 512]], memory_offsets = [[0, 0, 0, 0], [0, 0, 375, 0], [0, 0, 750, 0], [0, 0, 1125, 0]]}>
    // CHECK:       [[ALLOC0:%.+]] = memref.alloc() : memref<1x1x1500x512xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy inputs([[SHAPECAST]] : !VPUIP.DistributedBuffer<1x1x1500x512xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      outputs([[ALLOC0]] : memref<1x1x1500x512xf16, {order = #NHWC}, @DDR>)
    // CHECK:       return [[ARG_0]], [[COPY0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputOverlapped = !VPUIP.DistributedBuffer<
    1x16x240x200xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 60, 0], [0, 0, 120, 0], [0, 0, 180, 0]],
    memory_shapes = [[1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 60, 0], [0, 0, 120, 0], [0, 0, 180, 0]]
}>

// CHECK-LABEL: @MoveGenericReshapeBeforeCopySegmentedLikeOverlapped
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x240x200xf16, #NHWC, @CMX_NN,
func.func @MoveGenericReshapeBeforeCopySegmentedLikeOverlapped(%arg0: !InputOverlapped) -> memref<1x1500x512x1xf16, {order = #NHWC}, @DDR> {
    %alloc_0 = memref.alloc() : memref<1x16x240x200xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy inputs(%arg0 : !InputOverlapped)
        outputs(%alloc_0 : memref<1x16x240x200xf16, {order = #NHWC}, @DDR>)
        -> memref<1x16x240x200xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x16x240x200xf16, {order = #NHWC}, @DDR>)
        -> memref<1x1500x512x1xf16, {order = #NHWC}, @DDR>

    return %1 : memref<1x1500x512x1xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[GENERICRESHAPE:%.+]] = VPUIP.GenericReshape
    // CHECK-SAME:      inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x240x200xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200]], compute_offsets = [[0, 0, 0, 0], [0, 0, 60, 0], [0, 0, 120, 0], [0, 0, 180, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200], [1, 16, 60, 200]], memory_offsets = [[0, 0, 0, 0], [0, 0, 60, 0], [0, 0, 120, 0], [0, 0, 180, 0]]}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x1500x512x1xf16
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1500, 128, 1], [1, 1500, 128, 1], [1, 1500, 128, 1], [1, 1500, 128, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 128, 0], [0, 0, 256, 0], [0, 0, 384, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1500, 128, 1], [1, 1500, 128, 1], [1, 1500, 128, 1], [1, 1500, 128, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 128, 0], [0, 0, 256, 0], [0, 0, 384, 0]]}>
    // CHECK:       [[ALLOC0:%.+]] = memref.alloc() : memref<1x1500x512x1xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy inputs([[GENERICRESHAPE]] : !VPUIP.DistributedBuffer<1x1500x512x1xf16
    // CHECK-SAME:      outputs([[ALLOC0]] : memref<1x1500x512x1xf16, {order = #NHWC}, @DDR>)
    // CHECK:       return [[COPY0]] : memref<1x1500x512x1xf16, {order = #NHWC}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputOverlapped = !VPUIP.DistributedBuffer<
    1x1x1500x512xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 375, 512], [1, 1, 375, 512], [1, 1, 375, 512], [1, 1, 375, 512]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 375, 0], [0, 0, 750, 0], [0, 0, 1125, 0]],
    memory_shapes = [[1, 1, 375, 512], [1, 1, 375, 512], [1, 1, 375, 512], [1, 1, 375, 512]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 375, 0], [0, 0, 750, 0], [0, 0, 1125, 0]]
}>

// CHECK-LABEL: @MoveGenericReshapeBeforeCopySegmentedLikeOverlappedChangedAxis
// CHECK-SAME:  [[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x1x1500x512xf16, #NCHW, @CMX_NN,
func.func @MoveGenericReshapeBeforeCopySegmentedLikeOverlappedChangedAxis(%arg0: !InputOverlapped) -> memref<1x1500x512x1xf16, @DDR> {
    %alloc_0 = memref.alloc() : memref<1x1x1500x512xf16, @DDR>
    %0 = VPUIP.Copy inputs(%arg0 : !InputOverlapped)
        outputs(%alloc_0 : memref<1x1x1500x512xf16, @DDR>)
        -> memref<1x1x1500x512xf16, @DDR>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x1x1500x512xf16, @DDR>)
        -> memref<1x1500x512x1xf16, @DDR>

    return %1 : memref<1x1500x512x1xf16, @DDR>

    // CHECK: [[GENERICRESHAPE:%.+]] = VPUIP.GenericReshape
    // CHECK-SAME:      inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x1x1500x512xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 375, 512], [1, 1, 375, 512], [1, 1, 375, 512], [1, 1, 375, 512]], compute_offsets = [[0, 0, 0, 0], [0, 0, 375, 0], [0, 0, 750, 0], [0, 0, 1125, 0]]
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 375, 512], [1, 1, 375, 512], [1, 1, 375, 512], [1, 1, 375, 512]], memory_offsets = [[0, 0, 0, 0], [0, 0, 375, 0], [0, 0, 750, 0], [0, 0, 1125, 0]]}>)
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x1500x512x1xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME       mode = "OVERLAPPED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 375, 512, 1], [1, 375, 512, 1], [1, 375, 512, 1], [1, 375, 512, 1]], compute_offsets = [[0, 0, 0, 0], [0, 375, 0, 0], [0, 750, 0, 0], [0, 1125, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 375, 512, 1], [1, 375, 512, 1], [1, 375, 512, 1], [1, 375, 512, 1]], memory_offsets = [[0, 0, 0, 0], [0, 375, 0, 0], [0, 750, 0, 0], [0, 1125, 0, 0]]}>
    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x1500x512x1xf16, @DDR>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[GENERICRESHAPE]] : !VPUIP.DistributedBuffer<1x1500x512x1xf16
    // CHECK:           outputs([[ALLOC0]] : memref<1x1500x512x1xf16, @DDR>)
    // CHECK: return [[COPY0]]
}
