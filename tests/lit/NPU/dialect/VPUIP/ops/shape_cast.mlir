//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x3x96x160xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 3, 49, 160], [1, 3, 49, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]],
    memory_shapes = [[1, 3, 49, 160], [1, 3, 49, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x96x160xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 49, 160], [1, 16, 49, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]],
    memory_shapes = [[1, 16, 49, 160], [1, 16, 49, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]
}>

// CHECK-LABEL: ShapeCastWithExplicitOverlappedDistributedBuffType
func.func @ShapeCastWithExplicitOverlappedDistributedBuffType(%arg0: !InputDistributed)
    -> !OutputDistributed {

    %0 = VPUIP.ShapeCast {shape = [1, 16, 96, 160]} inputs(%arg0 : !InputDistributed) -> !OutputDistributed
    return %0 : !OutputDistributed

    // CHECK:        [[SHAPECAST:%.*]] = VPUIP.ShapeCast {shape = [1, 16, 96, 160]}
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x3x96x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x16x96x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 49, 160], [1, 16, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 49, 160], [1, 16, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x8x10xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 4, 10], [1, 16, 4, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    memory_shapes = [[1, 16, 4, 10], [1, 16, 4, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x40x2xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 20, 2], [1, 16, 20, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    memory_shapes = [[1, 16, 20, 2], [1, 16, 20, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
}>

// CHECK-LABEL: ShapeCastWithExplicitSegmentedDistributedBuffTypeWithClusteringAxisChange
func.func @ShapeCastWithExplicitSegmentedDistributedBuffTypeWithClusteringAxisChange(%arg0: !InputDistributed)
    -> !OutputDistributed {

    %0 = VPUIP.ShapeCast {shape = [1, 16, 40, 2]} inputs(%arg0 : !InputDistributed) -> !OutputDistributed
    return %0 : !OutputDistributed

    // CHECK:        [[SHAPECAST:%.*]] = VPUIP.ShapeCast {shape = [1, 16, 40, 2]}
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x16x8x10xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x16x40x2xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 20, 2], [1, 16, 20, 2]], compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 20, 2], [1, 16, 20, 2]], memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x88x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|MULTICASTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 44, 128], [1, 16, 44, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 44, 0]],
    memory_shapes = [[1, 16, 88, 128], [1, 16, 88, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x128x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 128, 128], [1, 16, 128, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 128, 128], [1, 16, 128, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: ShapeCastWithExplicitSegmentedMulticastedInputDistributedTensorType
func.func @ShapeCastWithExplicitSegmentedMulticastedInputDistributedTensorType(%arg0: !InputDistributed)
    -> !OutputDistributed {

    %0 = VPUIP.ShapeCast {shape = [1, 16, 128, 128]} inputs(%arg0 : !InputDistributed) -> !OutputDistributed
    return %0 : !OutputDistributed

    // CHECK:        [[SHAPECAST:%.*]] = VPUIP.ShapeCast {shape = [1, 16, 128, 128]}
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x16x88x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 128, 128], [1, 16, 128, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 128, 128], [1, 16, 128, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ShapeCastWithInStrideDimAtNAndNotSplit(%arg0: memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
            -> memref<1x48x192x128xf16, {order = #NHWC, strides = [2359296, 1, 6144, 48]}, @DDR> {

    %0 = VPUIP.ShapeCast {shape = [1, 48, 192, 128]}
                inputs(%arg0 : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
                -> memref<1x48x192x128xf16, {order = #NHWC, strides = [2359296, 1, 6144, 48]}, @DDR>
    return %0 : memref<1x48x192x128xf16, {order = #NHWC, strides = [2359296, 1, 6144, 48]}, @DDR>

    // CHECK:        [[SHAPECAST:%.*]] = VPUIP.ShapeCast {shape = [1, 48, 192, 128]}
    // CHECK-SAME:         inputs(%arg0 : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK-SAME:                     -> memref<1x48x192x128xf16, {order = #NHWC, strides = [2359296, 1, 6144, 48]}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ShapeCastWithInStrideDimAtWAndNotSplit(%arg0: memref<1x16x32x512xf16, {order = #NHWC, strides = [524288, 1, 16384, 32]}, @DDR>)
            -> memref<2x16x16x512xf16, {order = #NHWC, strides = [262144, 1, 16384, 32]}, @DDR> {

    %0 = VPUIP.ShapeCast {shape = [2, 16, 16, 512]}
                inputs(%arg0 : memref<1x16x32x512xf16, {order = #NHWC, strides = [524288, 1, 16384, 32]}, @DDR>)
                -> memref<2x16x16x512xf16, {order = #NHWC, strides = [262144, 1, 16384, 32]}, @DDR>
    return %0 : memref<2x16x16x512xf16, {order = #NHWC, strides = [262144, 1, 16384, 32]}, @DDR>

    // CHECK:        [[SHAPECAST:%.*]] = VPUIP.ShapeCast {shape = [2, 16, 16, 512]}
    // CHECK-SAME:         inputs(%arg0 : memref<1x16x32x512xf16, {order = #NHWC, strides = [524288, 1, 16384, 32]}, @DDR>)
    // CHECK-SAME:                     -> memref<2x16x16x512xf16, {order = #NHWC, strides = [262144, 1, 16384, 32]}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ShapeCastWithInStrideDimAtWAndSplitIntoAfterDim(%arg0: memref<1x1x512x256xf16, {order = #NHWC, strides = [262144, 1, 512, 2]}, @DDR>)
            -> memref<1x16x512x16xf16, {order = #NHWC, strides = [262144, 2, 512, 32]}, @DDR> {

    %0 = VPUIP.ShapeCast {shape = [1, 16, 512, 16]}
                inputs(%arg0 : memref<1x1x512x256xf16, {order = #NHWC, strides = [262144, 1, 512, 2]}, @DDR>)
                -> memref<1x16x512x16xf16, {order = #NHWC, strides = [262144, 2, 512, 32]}, @DDR>
    return %0 : memref<1x16x512x16xf16, {order = #NHWC, strides = [262144, 2, 512, 32]}, @DDR>

    // CHECK:        [[SHAPECAST:%.*]] = VPUIP.ShapeCast {shape = [1, 16, 512, 16]}
    // CHECK-SAME:         inputs(%arg0 : memref<1x1x512x256xf16, {order = #NHWC, strides = [262144, 1, 512, 2]}, @DDR>)
    // CHECK-SAME:                     -> memref<1x16x512x16xf16, {order = #NHWC, strides = [262144, 2, 512, 32]}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ShapeCastWithInStrideDimAtWAndSplitIntoFrontDim(%arg0: memref<1x512x1x256xf16, {order = #NHWC, strides = [262144, 1, 262144, 1024]}, @DDR>)
            -> memref<1x512x16x16xf16, {order = #NHWC, strides = [262144, 1, 16384,  1024]}, @DDR> {

    %0 = VPUIP.ShapeCast {shape = [1, 512, 16, 16]}
                inputs(%arg0 : memref<1x512x1x256xf16, {order = #NHWC, strides = [262144, 1, 262144, 1024]}, @DDR>)
                -> memref<1x512x16x16xf16, {order = #NHWC, strides = [262144, 1, 16384,  1024]}, @DDR>
    return %0 : memref<1x512x16x16xf16, {order = #NHWC, strides = [262144, 1, 16384,  1024]}, @DDR>

    // CHECK:        [[SHAPECAST:%.*]] = VPUIP.ShapeCast {shape = [1, 512, 16, 16]}
    // CHECK-SAME:         inputs(%arg0 : memref<1x512x1x256xf16, {order = #NHWC, strides = [262144, 1, 262144, 1024]}, @DDR>)
    // CHECK-SAME:                     -> memref<1x512x16x16xf16, {order = #NHWC, strides = [262144, 1, 16384, 1024]}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x2x800x2xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 1, 16, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    2x40x1x40xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

// CHECK-LABEL: ShapeCastWithSegmentedDuplicatedInputDistributedTensorType
func.func @ShapeCastWithSegmentedDuplicatedInputDistributedTensorType(%arg0: !InputDistributed)
    -> !OutputDistributed {

    %0 = VPUIP.ShapeCast {shape = [2, 40, 1, 40]} inputs(%arg0 : !InputDistributed) -> !OutputDistributed
    return %0 : !OutputDistributed

    // CHECK:        [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [2, 40, 1, 40]}
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x2x800x2xf16, #NCHW, @CMX_NN
    // CHECK-SAME:       mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 16, 1]
    // CHECK-SAME: -> !VPUIP.DistributedBuffer<2x40x1x40xf16, #NCHW, @CMX_NN
    // CHECK-SAME:       mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK: return [[SHAPECAST]] : !VPUIP.DistributedBuffer<2x40x1x40xf16, #NCHW, @CMX_NN
    // CHECK-SAME:       mode = "DUPLICATED", num_clusters = 2 : i64
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    2x80x1x40xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x2x80x40xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: ShapeCastWithSegmentedInputDistributedTensorType
func.func @ShapeCastWithSegmentedInputDistributedTensorType(%arg0: !InputDistributed)
    -> !OutputDistributed {

    %0 = VPUIP.ShapeCast {shape = [1, 2, 80, 40]} inputs(%arg0 : !InputDistributed) -> !OutputDistributed
    return %0 : !OutputDistributed

    // CHECK:        [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 2, 80, 40]}
    // CHECK-SAME:    !VPUIP.DistributedBuffer<2x80x1x40xf16, #NCHW, @CMX_NN
    // CHECK-SAME:       mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64
    // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x2x80x40xf16, #NCHW, @CMX_NN
    // CHECK-SAME:       mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    // CHECK: return [[SHAPECAST]] : !VPUIP.DistributedBuffer<1x2x80x40xf16, #NCHW, @CMX_NN
    // CHECK-SAME:       mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    240x10x1x1xf16, {order = #NHWC, strides = [16, 1, 10, 10]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    240x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

// CHECK-LABEL: ShapeCastWithSegmentedInputAndOutputHaveDifferentSize
func.func @ShapeCastWithSegmentedInputAndOutputHaveDifferentSize(%arg0: !InputDistributed)
    -> !OutputDistributed {

    %0 = VPUIP.ShapeCast {shape = [240, 16, 1, 1]} inputs(%arg0 : !InputDistributed) -> !OutputDistributed
    return %0 : !OutputDistributed

    // CHECK:        [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [240, 16, 1, 1]}
    // CHECK-SAME:    !VPUIP.DistributedBuffer<240x10x1x1xf16, {order = #NHWC, strides = [16, 1, 10, 10]}, @CMX_NN
    // CHECK-SAME:       mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]
    // CHECK-SAME: -> !VPUIP.DistributedBuffer<240x16x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:       mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]
    // CHECK: return [[SHAPECAST]] : !VPUIP.DistributedBuffer<240x16x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:       mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]
}
