//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-concat-view-copies %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!ResultT = !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]
}>

!Distributed0 = !VPUIP.DistributedBuffer<1x32x128x1xf32, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]
}>

!Distributed1 = !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]
}>

!Arg0T = memref<1x32x128x1023xf16, @DDR>
!Arg1T = memref<1x32x128x1xf32, @DDR>

// CHECK-LABEL: func.func @SplitUnbalancedConcatOnDifferentAxisBranchInputIsDistributedOverlappedWithConvertDMARightBranchFromDDR
// CHECK-SAME:  ([[LEFT_INPUT_ARG:%.+]]: memref<1x32x128x1023xf16, @DDR>, [[RIGHT_INPUT_ARG:%.+]]: memref<1x32x128x1xf32, @DDR>)
func.func @SplitUnbalancedConcatOnDifferentAxisBranchInputIsDistributedOverlappedWithConvertDMARightBranchFromDDR(%arg0 : !Arg0T, %arg1 : !Arg1T) -> (!ResultT, !ResultT) {
    %alloc = memref.alloc() : memref<1x32x128x1024xf16, @DDR>
    // Left branch
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 128, 1023] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x128x1023xf16, @DDR>) outputs(%0 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // Right branch
    %2 = VPURT.AllocDistributed -> !Distributed0
    %3 = VPUIP.Copy inputs(%arg1 : memref<1x32x128x1xf32, @DDR>) outputs(%2 : !Distributed0) -> !Distributed0
    %4 = VPURT.AllocDistributed -> !Distributed1
    %5 = VPUIP.ConvertDMA inputs(%3 : !Distributed0) outputs(%4 : !Distributed1) -> !Distributed1
    %6 = VPUIP.SubView %alloc [0, 0, 0, 1023] [1, 32, 128, 1] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %7 = VPUIP.Copy inputs(%5 : !Distributed1) outputs(%6 : memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %8 = VPUIP.ConcatView
        inputs(%1, %7 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>, memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x128x1024xf16, @DDR>) -> memref<1x32x128x1024xf16, @DDR>
    %9 = VPUIP.GenericReshape inputs(%8 : memref<1x32x128x1024xf16, @DDR>) -> memref<4096x1024x1x1xf16, @DDR>
    %10 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%9 : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, #NHWC, @DDR>
    %11 = VPUIP.SubView %10 [0, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %12 = VPUIP.SubView %10 [128, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %13 = VPURT.AllocDistributed -> !ResultT
    %14 = VPUIP.Copy inputs(%11 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%13 : !ResultT) -> !ResultT
    %15 = VPURT.AllocDistributed -> !ResultT
    %16 = VPUIP.Copy inputs(%12 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%15 : !ResultT) -> !ResultT

    return %14, %16 : !ResultT, !ResultT

    // CHECK:                   [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[LEFT_INPUT_ARG]] : memref<1x32x128x1023xf16, @DDR>) -> memref<4096x1023x1x1xf16, @DDR>
    // CHECK:                   [[PERMUTECAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<4096x1023x1x1xf16, @DDR>) -> memref<4096x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[GENERICRESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[RIGHT_INPUT_ARG]] : memref<1x32x128x1xf32, @DDR>) -> memref<4096x1x1x1xf32, @DDR>
    // CHECK:                   [[PERMUTECAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_1]] : memref<4096x1x1x1xf32, @DDR>) -> memref<4096x1x1x1xf32, #NHWC, @DDR>
    // CHECK:                   [[DISTRIBUTED_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [0, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[DISTRIBUTED_0]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<128x1023x1x1xf16, #NHWC, @DDR>) 
    // CHECK-SAME:                    outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [0, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf32, #NHWC, @DDR> to memref<128x1x1x1xf32, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_3:%.+]] = VPUIP.SubView [[DISTRIBUTED_0]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONVERTDMA_0:%.+]] = VPUIP.ConvertDMA inputs([[SUBVIEW_2]] : memref<128x1x1x1xf32, #NHWC, @DDR>) 
    // CHECK-SAME:                    outputs([[SUBVIEW_3]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[CONVERTDMA_0]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) 
    // CHECK-SAME:                    outputs([[DISTRIBUTED_0]] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[DISTRIBUTED_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_4:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [128, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_5:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<128x1023x1x1xf16, #NHWC, @DDR>) 
    // CHECK-SAME:                    outputs([[SUBVIEW_5]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_6:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [128, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf32, #NHWC, @DDR> to memref<128x1x1x1xf32, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_7:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONVERTDMA_1:%.+]] = VPUIP.ConvertDMA inputs([[SUBVIEW_6]] : memref<128x1x1x1xf32, #NHWC, @DDR>) 
    // CHECK-SAME:                    outputs([[SUBVIEW_7]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_1]], [[CONVERTDMA_1]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) 
    // CHECK-SAME:                    outputs([[DISTRIBUTED_1]] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   return [[CONCATVIEW_0]], [[CONCATVIEW_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!ResultT = !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]
}>

!Distributed0 = !VPUIP.DistributedBuffer<1x32x128x1xf32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]
}>

!Distributed1 = !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]
}>

!Arg0T = memref<1x32x128x1023xf16, @DDR>
!Arg1T = memref<1x32x128x1xf32, @DDR>

// CHECK-LABEL: func.func @SplitUnbalancedConcatOnDifferentAxisBranchInputIsDistributedDuplicatedWithConvertDMARightBranchFromCMX
// CHECK-SAME:  ([[INPUT_ARG:%.+]]: memref<1x32x128x1023xf16, @DDR>)
func.func @SplitUnbalancedConcatOnDifferentAxisBranchInputIsDistributedDuplicatedWithConvertDMARightBranchFromCMX(%arg0 : !Arg0T) -> (!ResultT, !ResultT) {
    %alloc = memref.alloc() : memref<1x32x128x1024xf16, @DDR>
    // Left branch
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 128, 1023] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x128x1023xf16, @DDR>) outputs(%0 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // Right branch
    %2 = VPURT.AllocDistributed -> !Distributed0
    %3 = VPURT.AllocDistributed -> !Distributed1
    %4 = VPUIP.ConvertDMA inputs(%2 : !Distributed0) outputs(%3 : !Distributed1) -> !Distributed1
    %5 = VPUIP.SubView %alloc [0, 0, 0, 1023] [1, 32, 128, 1] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %6 = VPUIP.Copy inputs(%4 : !Distributed1) outputs(%5 : memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %7 = VPUIP.ConcatView
        inputs(%1, %6 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>, memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x128x1024xf16, @DDR>) -> memref<1x32x128x1024xf16, @DDR>
    %8 = VPUIP.GenericReshape inputs(%7 : memref<1x32x128x1024xf16, @DDR>) -> memref<4096x1024x1x1xf16, @DDR>
    %9 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%8 : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, #NHWC, @DDR>
    %10 = VPUIP.SubView %9 [0, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %11 = VPUIP.SubView %9 [128, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %12 = VPURT.AllocDistributed -> !ResultT
    %13 = VPUIP.Copy inputs(%10 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%12 : !ResultT) -> !ResultT
    %14 = VPURT.AllocDistributed -> !ResultT
    %15 = VPUIP.Copy inputs(%11 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%14 : !ResultT) -> !ResultT

    return %13, %15 : !ResultT, !ResultT

    // CHECK:                   [[DISTRIBUTED_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x128x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>
    // CHECK:                   [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[INPUT_ARG]] : memref<1x32x128x1023xf16, @DDR>) -> memref<4096x1023x1x1xf16, @DDR>
    // CHECK:                   [[PERMUTECAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<4096x1023x1x1xf16, @DDR>) -> memref<4096x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[ALLOC:%.+]] = memref.alloc() : memref<1x32x128x1xf32, @DDR>
    // CHECK:                   [[COPY_0:%.+]] = VPUIP.Copy inputs([[DISTRIBUTED_0]] : !VPUIP.DistributedBuffer<1x32x128x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>) outputs(
    // CHECK:                   [[ALLOC]] : memref<1x32x128x1xf32, @DDR>) -> memref<1x32x128x1xf32, @DDR>
    // CHECK:                   [[GENERICRESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[COPY_0]] : memref<1x32x128x1xf32, @DDR>) -> memref<4096x1x1x1xf32, @DDR>
    // CHECK:                   [[PERMUTECAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_1]] : memref<4096x1x1x1xf32, @DDR>) -> memref<4096x1x1x1xf32, #NHWC, @DDR>
    // CHECK:                   [[DISTRIBUTED_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [0, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<128x1023x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [0, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf32, #NHWC, @DDR> to memref<128x1x1x1xf32, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_3:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONVERTDMA_0:%.+]] = VPUIP.ConvertDMA inputs([[SUBVIEW_2]] : memref<128x1x1x1xf32, #NHWC, @DDR>) outputs([[SUBVIEW_3]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_1]], [[CONVERTDMA_0]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) outputs(
    // CHECK:                   [[DISTRIBUTED_1]] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[DISTRIBUTED_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_4:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [128, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_5:%.+]] = VPUIP.SubView [[DISTRIBUTED_2]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<128x1023x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_5]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_6:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [128, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf32, #NHWC, @DDR> to memref<128x1x1x1xf32, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_7:%.+]] = VPUIP.SubView [[DISTRIBUTED_2]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONVERTDMA_1:%.+]] = VPUIP.ConvertDMA inputs([[SUBVIEW_6]] : memref<128x1x1x1xf32, #NHWC, @DDR>) outputs([[SUBVIEW_7]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_2]], [[CONVERTDMA_1]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) outputs(
    // CHECK:                   [[DISTRIBUTED_2]] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, 
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], 
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   return [[CONCATVIEW_0]], [[CONCATVIEW_1]]
}

