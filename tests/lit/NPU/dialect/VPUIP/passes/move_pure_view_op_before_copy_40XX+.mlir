//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --move-pure-view-op-before-copy %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConvertInput = memref<1x16x4x4xf32, {order = #NHWC}, @CMX_NN>
!ConvertOutputDistributed = !VPUIP.DistributedBuffer<
    1x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments
}>

// CHECK-LABEL: @DoNotMoveViewLikeBeforeCopyWithDistributedConvertDMAInput
// CHECK-SAME:  [[ARG_0:%[^:]+]]: memref<1x16x4x4xf32, {order = #NHWC}, @CMX_NN>
func.func @DoNotMoveViewLikeBeforeCopyWithDistributedConvertDMAInput(%arg0: !ConvertInput)
        -> memref<1x4x16x4xf16, {order = #NHWC}, @DDR> {
    %convertAlloc = VPURT.AllocDistributed -> !ConvertOutputDistributed
    %convert = VPUIP.ConvertDMA inputs(%arg0 : !ConvertInput)
        outputs(%convertAlloc : !ConvertOutputDistributed) -> !ConvertOutputDistributed
    %copyAlloc = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, @DDR>
    %copy = VPUIP.Copy inputs(%convert : !ConvertOutputDistributed)
        outputs(%copyAlloc : memref<1x16x4x4xf16, {order = #NHWC}, @DDR>) -> memref<1x16x4x4xf16, {order = #NHWC}, @DDR>
    %shapeCast = VPUIP.ShapeCast {shape = [1, 4, 16, 4]} inputs(%copy : memref<1x16x4x4xf16, {order = #NHWC}, @DDR>)
        -> memref<1x4x16x4xf16, {order = #NHWC}, @DDR>

    return %shapeCast : memref<1x4x16x4xf16, {order = #NHWC}, @DDR>

    // CHECK: [[CONVERT_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x4x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"
    // CHECK: [[CONVERT:%.+]] = VPUIP.ConvertDMA inputs([[ARG_0]] : memref<1x16x4x4xf32, {order = #NHWC}, @CMX_NN>) outputs([[CONVERT_ALLOC]] : !VPUIP.DistributedBuffer<1x16x4x4xf16, #NHWC, @CMX_NN
    // CHECK: [[COPY_ALLOC:%.+]] = memref.alloc() : memref<1x16x4x4xf16, {order = #NHWC}, @DDR>
    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[CONVERT]] : !VPUIP.DistributedBuffer<1x16x4x4xf16, #NHWC, @CMX_NN
    // CHECK-SAME: outputs([[COPY_ALLOC]] : memref<1x16x4x4xf16, {order = #NHWC}, @DDR>) -> memref<1x16x4x4xf16, {order = #NHWC}, @DDR>
    // CHECK: [[SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 4, 16, 4]} inputs([[COPY]] : memref<1x16x4x4xf16, {order = #NHWC}, @DDR>) -> memref<1x4x16x4xf16, {order = #NHWC}, @DDR>
    // CHECK: return [[SHAPE_CAST]] : memref<1x4x16x4xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x128x192x24xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 128, 96, 24], [1, 128, 96, 24]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]],
    memory_shapes = [[1, 128, 96, 24], [1, 128, 96, 24]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]]
}>

// CHECK-LABEL: @MoveShapeCastBeforeTilingCopyOverlapped
// CHECK-SAME:  [[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x128x192x24xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED"
func.func @MoveShapeCastBeforeTilingCopyOverlapped(%arg0: !InputDistributed) -> memref<1x16x192x192xf16, {order = #NHWC}, @DDR> {
    %alloc = memref.alloc() : memref<1x128x192x24xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%alloc : memref<1x128x192x24xf16, {order = #NHWC}, @DDR>) ->  memref<1x128x192x24xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.ShapeCast {shape = [1, 16, 192, 192]} inputs(%0 : memref<1x128x192x24xf16, {order = #NHWC}, @DDR>) -> memref<1x16x192x192xf16, {order = #NHWC}, @DDR>

    return %1 : memref<1x16x192x192xf16, {order = #NHWC}, @DDR>

    //CHECK:               [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 192, 192]}
    //CHECK-SAME:              inputs([[INPUT]] : !VPUIP.DistributedBuffer<1x128x192x24xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:          compute_shapes = [[1, 128, 96, 24], [1, 128, 96, 24]], compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]],
    //CHECK-SAME{LITERAL}:          memory_shapes = [[1, 128, 96, 24], [1, 128, 96, 24]], memory_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]]}>)
    //CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x16x192x192xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:          compute_shapes = [[1, 16, 96, 192], [1, 16, 96, 192]], compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]],
    //CHECK-SAME{LITERAL}:          memory_shapes = [[1, 16, 96, 192], [1, 16, 96, 192]], memory_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]]}>
    //CHECK:               [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x192x192xf16, {order = #NHWC}, @DDR>
    // CHECK:              [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[SHAPECAST]] : !VPUIP.DistributedBuffer<1x16x192x192xf16, #NHWC, @CMX_NN
    // CHECK-SAME:           outputs([[OUTBUFF]] : memref<1x16x192x192xf16, {order = #NHWC}, @DDR>) -> memref<1x16x192x192xf16, {order = #NHWC}, @DDR>
    //CHECK:               return [[COPY]] : memref<1x16x192x192xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x3x128x128xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 3, 64, 128], [1, 3, 64, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    memory_shapes = [[1, 3, 64, 128], [1, 3, 64, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]
}>

// CHECK-LABEL: @NotMoveShapeCastBeforeTilingCopyOverlapped
// CHECK-SAME:  [[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x3x128x128xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED"
func.func @NotMoveShapeCastBeforeTilingCopyOverlapped(%arg0: !InputDistributed) -> memref<1x48x32x32xf16, {order = #NHWC}, @DDR> {
    %alloc = memref.alloc() : memref<1x3x128x128xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%alloc : memref<1x3x128x128xf16, {order = #NHWC}, @DDR>) ->  memref<1x3x128x128xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.ShapeCast {shape = [1, 48, 32, 32]} inputs(%0 : memref<1x3x128x128xf16, {order = #NHWC}, @DDR>) -> memref<1x48x32x32xf16, {order = #NHWC}, @DDR>

    return %1 : memref<1x48x32x32xf16, {order = #NHWC}, @DDR>

    //CHECK:               [[OUTBUFF:%.+]] = memref.alloc() : memref<1x3x128x128xf16, {order = #NHWC}, @DDR>
    // CHECK:              [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[INPUT]]
    // CHECK-SAME:           outputs([[OUTBUFF]] : memref<1x3x128x128xf16, {order = #NHWC}, @DDR>) -> memref<1x3x128x128xf16, {order = #NHWC}, @DDR>
    //CHECK:               [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 48, 32, 32]}
    //CHECK-SAME:              inputs([[COPY]] : memref<1x3x128x128xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:              -> memref<1x48x32x32xf16, {order = #NHWC}, @DDR>
    //CHECK:               return [[SHAPECAST]] : memref<1x48x32x32xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x256x128x8xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 256, 64, 8], [1, 256, 64, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    memory_shapes = [[1, 256, 64, 8], [1, 256, 64, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]
}>

// CHECK-LABEL: @MoveGenericReshapeBeforeTilingCopyOverlapped
// CHECK-SAME:  [[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x256x128x8xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED"
func.func @MoveGenericReshapeBeforeTilingCopyOverlapped(%arg0: !InputDistributed) -> memref<1x16x128x128xf16, {order = #NHWC}, @DDR> {
    %alloc = memref.alloc() : memref<1x256x128x8xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%alloc : memref<1x256x128x8xf16, {order = #NHWC}, @DDR>) ->  memref<1x256x128x8xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x256x128x8xf16, {order = #NHWC}, @DDR>) -> memref<1x16x128x128xf16, {order = #NHWC}, @DDR>

    return %1 : memref<1x16x128x128xf16, {order = #NHWC}, @DDR>

    //CHECK:               [[GENERICRESHAPE:%.+]] = VPUIP.GenericReshape
    //CHECK-SAME:              inputs([[INPUT]] : !VPUIP.DistributedBuffer<1x256x128x8xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:          compute_shapes = [[1, 256, 64, 8], [1, 256, 64, 8]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    //CHECK-SAME{LITERAL}:          memory_shapes = [[1, 256, 64, 8], [1, 256, 64, 8]], memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]}>)
    //CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:          compute_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    //CHECK-SAME{LITERAL}:          memory_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]}>
    //CHECK:               [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x128x128xf16, {order = #NHWC}, @DDR>
    //CHECK:              [[COPY:%.+]] = VPUIP.Copy
    //CHECK-SAME:           inputs([[GENERICRESHAPE]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           outputs([[OUTBUFF]] : memref<1x16x128x128xf16, {order = #NHWC}, @DDR>) -> memref<1x16x128x128xf16, {order = #NHWC}, @DDR>
    //CHECK:               return [[COPY]] : memref<1x16x128x128xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x128x192x24xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 128, 96, 24], [1, 128, 96, 24]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]],
    memory_shapes = [[1, 128, 98, 24], [1, 128, 98, 24]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 94, 0]]
}>

// CHECK-LABEL: @MoveShapeCastBeforeTilingCopyOverlappedWithOverlap
// CHECK-SAME:  [[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x128x192x24xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED"
func.func @MoveShapeCastBeforeTilingCopyOverlappedWithOverlap(%arg0: !InputDistributed) -> memref<1x16x192x192xf16, {order = #NHWC}, @DDR> {
    %alloc = memref.alloc() : memref<1x128x192x24xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%alloc : memref<1x128x192x24xf16, {order = #NHWC}, @DDR>) ->  memref<1x128x192x24xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.ShapeCast {shape = [1, 16, 192, 192]} inputs(%0 : memref<1x128x192x24xf16, {order = #NHWC}, @DDR>) -> memref<1x16x192x192xf16, {order = #NHWC}, @DDR>

    return %1 : memref<1x16x192x192xf16, {order = #NHWC}, @DDR>

    //CHECK:               [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 192, 192]}
    //CHECK-SAME:              inputs([[INPUT]] : !VPUIP.DistributedBuffer<1x128x192x24xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:          compute_shapes = [[1, 128, 96, 24], [1, 128, 96, 24]], compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]],
    //CHECK-SAME{LITERAL}:          memory_shapes = [[1, 128, 98, 24], [1, 128, 98, 24]], memory_offsets = [[0, 0, 0, 0], [0, 0, 94, 0]]}>)
    //CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x16x192x192xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:          compute_shapes = [[1, 16, 96, 192], [1, 16, 96, 192]], compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]],
    //CHECK-SAME{LITERAL}:          memory_shapes = [[1, 16, 98, 192], [1, 16, 98, 192]], memory_offsets = [[0, 0, 0, 0], [0, 0, 94, 0]]}>
    //CHECK:               [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x192x192xf16, {order = #NHWC}, @DDR>
    //CHECK:               [[COPY:%.+]] = VPUIP.Copy
    //CHECK-SAME:   inputs([[SHAPECAST]] : !VPUIP.DistributedBuffer<1x16x192x192xf16, #NHWC, @CMX_NN
    //CHECK-SAME:            outputs([[OUTBUFF]] : memref<1x16x192x192xf16, {order = #NHWC}, @DDR>) -> memref<1x16x192x192xf16, {order = #NHWC}, @DDR>
    //CHECK:               return [[COPY]] : memref<1x16x192x192xf16, {order = #NHWC}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x192x16x48xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 6, 1, 1],
    num_clusters = 6,
    uniform_distributed_segments
}>

// CHECK-LABEL: @MovePermuteCastBeforeTilingCopy
// CHECK-SAME:  [[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x192x16x48xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
func.func @MovePermuteCastBeforeTilingCopy(%arg0: !InputDistributed) -> memref<1x48x192x16xf16, {order = #NHWC}, @DDR> {
    %0 = memref.alloc() : memref<1x192x16x48xf16, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%0 : memref<1x192x16x48xf16, @DDR>) ->  memref<1x192x16x48xf16, @DDR>
    %2 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%1: memref<1x192x16x48xf16, @DDR>) -> memref<1x48x192x16xf16, {order = #NHWC}, @DDR>
    return %2 : memref<1x48x192x16xf16, {order = #NHWC}, @DDR>

    //CHECK:              [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs([[INPUT]] : !VPUIP.DistributedBuffer<1x192x16x48xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>)
    //CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x48x192x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments}>
    //CHECK:              [[ALLOC:%.+]] = memref.alloc() : memref<1x48x192x16xf16, {order = #NHWC}, @DDR>
    // CHECK:             [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[PERMUTECAST]] : !VPUIP.DistributedBuffer<1x48x192x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments}>)
    // CHECK-SAME:          outputs([[ALLOC]] : memref<1x48x192x16xf16, {order = #NHWC}, @DDR>) -> memref<1x48x192x16xf16, {order = #NHWC}, @DDR>
    //CHECK:              return [[COPY]] : memref<1x48x192x16xf16, {order = #NHWC}, @DDR>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x192x16x48xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 6, 1, 1],
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 16, 48], [1, 32, 16, 48], [1, 32, 16, 48], [1, 32, 16, 48], [1, 32, 16, 48], [1, 32, 16, 48]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0], [0, 128, 0, 0], [0, 160, 0, 0]],
    memory_shapes = [[1, 32, 16, 48], [1, 32, 16, 48], [1, 32, 16, 48], [1, 32, 16, 48], [1, 32, 16, 48], [1, 32, 16, 48]],
    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0], [0, 128, 0, 0], [0, 160, 0, 0]]
}>

// CHECK-LABEL: @MovePermuteCastBeforeTilingCopyWithExplicitAttr
// CHECK-SAME:  [[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x192x16x48xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
func.func @MovePermuteCastBeforeTilingCopyWithExplicitAttr(%arg0: !InputDistributed) -> memref<1x48x192x16xf16, {order = #NHWC}, @DDR> {
    %0 = memref.alloc() : memref<1x192x16x48xf16, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%0 : memref<1x192x16x48xf16, @DDR>) ->  memref<1x192x16x48xf16, @DDR>
    %2 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%1: memref<1x192x16x48xf16, @DDR>) -> memref<1x48x192x16xf16, {order = #NHWC}, @DDR>
    return %2 : memref<1x48x192x16xf16, {order = #NHWC}, @DDR>


    //CHECK:              [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs([[INPUT]] : !VPUIP.DistributedBuffer<1x192x16x48xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
    //CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x48x192x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:           compute_shapes = [[1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16]],
    //CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0], [0, 0, 128, 0], [0, 0, 160, 0]],
    //CHECK-SAME{LITERAL}:           memory_shapes = [[1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16]],
    //CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0], [0, 0, 128, 0], [0, 0, 160, 0]]}>
    //CHECK:              [[ALLOC:%.+]] = memref.alloc() : memref<1x48x192x16xf16, {order = #NHWC}, @DDR>
    // CHECK:             [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[PERMUTECAST]]
    // CHECK-SAME:          outputs([[ALLOC]] : memref<1x48x192x16xf16, {order = #NHWC}, @DDR>)
    //CHECK:              return [[COPY]] : memref<1x48x192x16xf16, {order = #NHWC}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x48x192x16xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0], [0, 0, 128, 0], [0, 0, 160, 0]],
    memory_shapes = [[1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16], [1, 48, 32, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0], [0, 0, 128, 0], [0, 0, 160, 0]]
}>


// CHECK-LABEL: @MoveOverlappedPermuteCastBeforeTilingCopyWithExplicitAttr
// CHECK-SAME:  [[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x48x192x16xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED"
func.func @MoveOverlappedPermuteCastBeforeTilingCopyWithExplicitAttr(%arg0: !InputDistributed) -> memref<1x192x16x48xf16, {order = #NWCH}, @DDR> {
    %0 = memref.alloc() : memref<1x48x192x16xf16, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%0 : memref<1x48x192x16xf16, @DDR>) ->  memref<1x48x192x16xf16, @DDR>
    %2 = VPUIP.PermuteCast {dst_order = #NWCH, mem_perm = #NCHW} inputs(%1: memref<1x48x192x16xf16, @DDR>) -> memref<1x192x16x48xf16, {order = #NWCH}, @DDR>
    return %2 : memref<1x192x16x48xf16, {order = #NWCH}, @DDR>

    //CHECK:              [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NWCH, mem_perm = #NCHW} inputs([[INPUT]] : !VPUIP.DistributedBuffer<1x48x192x16xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x192x16x48xf16, #NWCH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK:              [[ALLOC:%.+]] = memref.alloc() : memref<1x192x16x48xf16, {order = #NWCH}, @DDR>
    //CHECK:              [[COPY:%.+]] = VPUIP.Copy
    //CHECK-SAME:           inputs([[PERMUTECAST]]
    //CHECK-SAME:           outputs([[ALLOC]] : memref<1x192x16x48xf16, {order = #NWCH}, @DDR>) -> memref<1x192x16x48xf16, {order = #NWCH}, @DDR>
    //CHECK:              return [[COPY]] : memref<1x192x16x48xf16, {order = #NWCH}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x48x192x16xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[1, 48, 34, 16], [1, 48, 34, 16], [1, 48, 34, 16], [1, 48, 34, 16], [1, 48, 34, 16], [1, 48, 34, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 30, 0], [0, 0, 62, 0], [0, 0, 94, 0], [0, 0, 126, 0], [0, 0, 158, 0]],
    memory_shapes = [[1, 48, 34, 16], [1, 48, 34, 16], [1, 48, 34, 16], [1, 48, 34, 16], [1, 48, 34, 16], [1, 48, 34, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 30, 0], [0, 0, 62, 0], [0, 0, 94, 0], [0, 0, 126, 0], [0, 0, 158, 0]]
}>


// CHECK-LABEL: @NotMoveOverlappedPermuteCastBeforeTilingCopyWithExplicitAttr
// CHECK-SAME:  [[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x48x192x16xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED"
func.func @NotMoveOverlappedPermuteCastBeforeTilingCopyWithExplicitAttr(%arg0: !InputDistributed) -> memref<1x192x16x48xf16, {order = #NWCH}, @DDR> {
    %0 = memref.alloc() : memref<1x48x192x16xf16, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%0 : memref<1x48x192x16xf16, @DDR>) ->  memref<1x48x192x16xf16, @DDR>
    %2 = VPUIP.PermuteCast {dst_order = #NWCH, mem_perm = #NCHW} inputs(%1: memref<1x48x192x16xf16, @DDR>) -> memref<1x192x16x48xf16, {order = #NWCH}, @DDR>
    return %2 : memref<1x192x16x48xf16, {order = #NWCH}, @DDR>

    //CHECK:              [[ALLOC:%.+]] = memref.alloc() : memref<1x48x192x16xf16, @DDR>
    // CHECK:             [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[INPUT]]
    // CHECK-SAME:          outputs([[ALLOC]] : memref<1x48x192x16xf16, @DDR>) -> memref<1x48x192x16xf16, @DDR>
    //CHECK:              [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NWCH, mem_perm = #NCHW} inputs([[COPY]] : memref<1x48x192x16xf16, @DDR>)
    //CHECK-SAME:              -> memref<1x192x16x48xf16, {order = #NWCH}, @DDR>

    //CHECK:              return [[PERMUTECAST]] : memref<1x192x16x48xf16, {order = #NWCH}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x64x64x64xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 16, 64], [1, 64, 16, 64], [1, 64, 16, 64], [1, 64, 16, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0]],
    memory_shapes = [[1, 64, 16, 64], [1, 64, 16, 64], [1, 64, 16, 64], [1, 64, 16, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0]]
}>

// CHECK-LABEL: @NotMoveOverlappedPermuteCastBeforeTilingCopyWithNonTrivialReorder
// CHECK-SAME:  [[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED"
func.func @NotMoveOverlappedPermuteCastBeforeTilingCopyWithNonTrivialReorder(%arg0: !InputDistributed) -> memref<1x64x64x64xf16> {
    %alloc = memref.alloc() : memref<1x64x64x64xf16, {order = #NHWC}>
    %cluster_copy = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%alloc : memref<1x64x64x64xf16, {order = #NHWC}>) ->  memref<1x64x64x64xf16, {order = #NHWC}>
    %permute_cast = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%cluster_copy : memref<1x64x64x64xf16, {order = #NHWC}>) -> memref<1x64x64x64xf16>

    return %permute_cast : memref<1x64x64x64xf16>

    // CHECK:              [[ALLOC:%.+]] = memref.alloc() : memref<1x64x64x64xf16, {order = #NHWC}>
    // CHECK:              [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[INPUT]]
    // CHECK-SAME:           outputs([[ALLOC]] : memref<1x64x64x64xf16, {order = #NHWC}>) -> memref<1x64x64x64xf16, {order = #NHWC}>
    // CHECK:              [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs([[COPY]] : memref<1x64x64x64xf16, {order = #NHWC}>)
    // CHECK-SAME:              -> memref<1x64x64x64xf16>
    // CHECK:              return [[PERMUTECAST]] : memref<1x64x64x64xf16>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @ChangeForStridedCopy
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x2048x8192xui8, {order = #CHW, strides = [33554432, 16384, 1]}, @DDR>)
func.func @ChangeForStridedCopy(
        %arg0: memref<1x2048x8192xui8, {order = #CHW, strides = [33554432, 16384, 1]}, @DDR>) -> memref<1x2048x64x128x!qElemType, {order = #NHWC}, @DDR> {

    %0 = memref.alloc() : memref<1x2048x8192xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x2048x8192xui8, {order = #CHW, strides = [33554432, 16384, 1]}, @DDR>)
        outputs(%0 : memref<1x2048x8192xui8, @DDR>)
        -> memref<1x2048x8192xui8, @DDR>

    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x2048x8192xui8, @DDR>) -> memref<1x1x2048x8192xui8, @DDR>

    %3 = VPUIP.QuantizeCast inputs(%2 : memref<1x1x2048x8192xui8, @DDR>) -> memref<1x1x2048x8192x!qElemType, @DDR>

    %4 = VPUIP.ShapeCast {shape = [1, 2048, 64, 128]} inputs(%3 : memref<1x1x2048x8192x!qElemType, @DDR>) -> memref<1x2048x64x128x!qElemType, @DDR>

    %5 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
        inputs(%4 : memref<1x2048x64x128x!qElemType, @DDR>)
        -> memref<1x2048x64x128x!qElemType, {order = #NHWC}, @DDR>

    return %5 : memref<1x2048x64x128x!qElemType, {order = #NHWC}, @DDR>

    // CHECK:   [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]] : memref<1x2048x8192xui8, {order = #CHW, strides = [33554432, 16384, 1]}, @DDR>) -> memref<1x1x2048x8192xui8, {order = #NCHW, strides = [33554432, 33554432, 16384, 1]}, @DDR>
    // CHECK:   [[QUANT:%.+]] = VPUIP.QuantizeCast inputs([[RESHAPE]] : memref<1x1x2048x8192xui8, {order = #NCHW, strides = [33554432, 33554432, 16384, 1]}, @DDR>) -> memref<1x1x2048x8192x!qElemType, {order = #NCHW, strides = [33554432, 33554432, 16384, 1]}, @DDR>
    // CHECK:   [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 2048, 64, 128]} inputs([[QUANT]] : memref<1x1x2048x8192x!qElemType, {order = #NCHW, strides = [33554432, 33554432, 16384, 1]}, @DDR>) -> memref<1x2048x64x128x!qElemType, {order = #NCHW, strides = [33554432, 16384, 128, 1]}, @DDR>
    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<1x2048x64x128x!qElemType, @DDR>
    // CHECK:   [[COPY:%.+]] = VPUIP.Copy inputs([[SHAPECAST]] : memref<1x2048x64x128x!qElemType, {order = #NCHW, strides = [33554432, 16384, 128, 1]}, @DDR>) outputs([[ALLOC]] : memref<1x2048x64x128x!qElemType, @DDR>) -> memref<1x2048x64x128x!qElemType, @DDR>
    // CHECK:   [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[COPY]] : memref<1x2048x64x128x!qElemType, @DDR>) -> memref<1x2048x64x128x!qElemType, {order = #NHWC}, @DDR>
    // CHECK:   return [[PERMUTE]] : memref<1x2048x64x128x!qElemType, {order = #NHWC}, @DDR>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @ChangeWithNotEffectiveStridesGenericReshape
func.func @ChangeWithNotEffectiveStridesGenericReshape(
    %arg0: memref<1x16x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) -> memref<1x1x64xui8, @DDR> {
    %0 = memref.alloc() : memref<1x16x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) outputs(%0 : memref<1x16x4xui8, @DDR>) -> memref<1x16x4xui8, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x16x4xui8, @DDR>) -> memref<1x1x64xui8, @DDR>
    return %2 : memref<1x1x64xui8, @DDR>
    // CHECK:   VPUIP.GenericReshape
    // CHECK:   VPUIP.Copy
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @NoChangeWithEffectiveStridesGenericReshape
func.func @NoChangeWithEffectiveStridesGenericReshape(
    %arg0: memref<1x16x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) -> memref<1x1x64xui8, @DDR> {
    %0 = memref.alloc() : memref<1x16x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) outputs(%0 : memref<1x16x4xui8, @DDR>) -> memref<1x16x4xui8, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x16x4xui8, @DDR>) -> memref<1x1x64xui8, @DDR>
    return %2 : memref<1x1x64xui8, @DDR>
    // CHECK:   VPUIP.Copy
    // CHECK:   VPUIP.GenericReshape
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @ChangeWithNotEffectiveStridesPermuteCast
func.func @ChangeWithNotEffectiveStridesPermuteCast(
    %arg0: memref<1x16x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) -> memref<1x16x4xui8, {order = #map}, @DDR> {
    %0 = memref.alloc() : memref<1x16x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) outputs(%0 : memref<1x16x4xui8, @DDR>) -> memref<1x16x4xui8, @DDR>
    %2 = VPUIP.PermuteCast {dst_order = #map, mem_perm = #map} inputs(%1 : memref<1x16x4xui8, @DDR>) -> memref<1x16x4xui8, {order = #map}, @DDR>
    return %2 : memref<1x16x4xui8, {order = #map}, @DDR>
    // CHECK:   VPUIP.PermuteCast
    // CHECK:   VPUIP.Copy
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @NoChangeWithEffectiveStridesPermuteCast
func.func @NoChangeWithEffectiveStridesPermuteCast(
    %arg0: memref<1x16x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) -> memref<1x16x4xui8, {order = #map}, @DDR> {
    %0 = memref.alloc() : memref<1x16x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) outputs(%0 : memref<1x16x4xui8, @DDR>) -> memref<1x16x4xui8, @DDR>
    %2 = VPUIP.PermuteCast {dst_order = #map, mem_perm = #map} inputs(%1 : memref<1x16x4xui8, @DDR>) -> memref<1x16x4xui8, {order = #map}, @DDR>
    return %2 : memref<1x16x4xui8, {order = #map}, @DDR>
    // CHECK:   VPUIP.Copy
    // CHECK:   VPUIP.PermuteCast
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @ChangeWithNotEffectiveStridesQuantizeCast
func.func @ChangeWithNotEffectiveStridesQuantizeCast(
    %arg0: memref<1x16x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) -> memref<1x16x4x!qElemType, @DDR> {
    %0 = memref.alloc() : memref<1x16x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) outputs(%0 : memref<1x16x4xui8, @DDR>) -> memref<1x16x4xui8, @DDR>
    %2 = VPUIP.QuantizeCast inputs(%1 : memref<1x16x4xui8, @DDR>) -> memref<1x16x4x!qElemType, @DDR>
    return %2 : memref<1x16x4x!qElemType, @DDR>
    // CHECK:   VPUIP.QuantizeCast
    // CHECK:   VPUIP.Copy
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NoChangeWithEffectiveStridesQuantizeCast
func.func @NoChangeWithEffectiveStridesQuantizeCast(
    %arg0: memref<1x16x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) -> memref<1x16x4x!qElemType, @DDR> {
    %0 = memref.alloc() : memref<1x16x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) outputs(%0 : memref<1x16x4xui8, @DDR>) -> memref<1x16x4xui8, @DDR>
    %2 = VPUIP.QuantizeCast inputs(%1 : memref<1x16x4xui8, @DDR>) -> memref<1x16x4x!qElemType, @DDR>
    return %2 : memref<1x16x4x!qElemType, @DDR>
    // CHECK:   VPUIP.Copy
    // CHECK:   VPUIP.QuantizeCast
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @ChangeWithNotEffectiveStridesShapeCast
func.func @ChangeWithNotEffectiveStridesShapeCast(
    %arg0: memref<1x16x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) -> memref<1x64x1xui8, @DDR> {
    %0 = memref.alloc() : memref<1x16x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) outputs(%0 : memref<1x16x4xui8, @DDR>) -> memref<1x16x4xui8, @DDR>
    %2 = VPUIP.ShapeCast {shape = [1, 64, 1]} inputs(%1 : memref<1x16x4xui8, @DDR>) -> memref<1x64x1xui8, @DDR>
    return %2 : memref<1x64x1xui8, @DDR>
    // CHECK:   VPUIP.ShapeCast
    // CHECK:   VPUIP.Copy
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @NoChangeWithEffectiveStridesShapeCast
func.func @NoChangeWithEffectiveStridesShapeCast(
    %arg0: memref<1x16x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) -> memref<1x64x1xui8, @DDR> {
    %0 = memref.alloc() : memref<1x16x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) outputs(%0 : memref<1x16x4xui8, @DDR>) -> memref<1x16x4xui8, @DDR>
    %2 = VPUIP.ShapeCast {shape = [1, 64, 1]} inputs(%1 : memref<1x16x4xui8, @DDR>) -> memref<1x64x1xui8, @DDR>
    return %2 : memref<1x64x1xui8, @DDR>
    // CHECK:   VPUIP.Copy
    // CHECK:   VPUIP.ShapeCast
}
