//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x256x40x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    memory_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
}>

!InputOutputDistributed = !VPUIP.DistributedBuffer<
    1x40x1x256xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 20, 1, 256], [1, 20, 1, 256]],
    compute_offsets = [[0, 0, 0, 0], [0, 20, 0, 0]],
    memory_shapes = [[1, 20, 1, 256], [1, 20, 1, 256]],
    memory_offsets = [[0, 0, 0, 0], [0, 20, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x256x1x40xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 256, 1, 20], [1, 256, 1, 20]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 20]],
    memory_shapes = [[1, 256, 1, 20], [1, 256, 1, 20]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 20]]
}>

// CHECK-LABEL: PermuteCastDistributed
func.func @PermuteCastDistributed(%arg0: !InputDistributed) -> !OutputDistributed {

    %0 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%arg0: !InputDistributed)
            -> !InputOutputDistributed

    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHCW} inputs(%0 : !InputOutputDistributed)
            -> !OutputDistributed

    return %1 : !OutputDistributed

    // CHECK:        [[PERMUTECAST0:%.*]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x256x40x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x40x1x256xf16, #NCHW, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 20, 1, 256], [1, 20, 1, 256]], compute_offsets = [[0, 0, 0, 0], [0, 20, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 20, 1, 256], [1, 20, 1, 256]], memory_offsets = [[0, 0, 0, 0], [0, 20, 0, 0]]

    // CHECK:        [[PERMUTECAST1:%.*]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHCW}
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x40x1x256xf16, #NCHW, @CMX_NN
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x256x1x40xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 256, 1, 20], [1, 256, 1, 20]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 20]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 256, 1, 20], [1, 256, 1, 20]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 20]]

    // return [[PERMUTECAST1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x256x40x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    memory_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x256x40x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    memory_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
}>

// CHECK-LABEL: PermuteCastDistributedWithOnlyOrderChanged
func.func @PermuteCastDistributedWithOnlyOrderChanged(%arg0: !InputDistributed) -> !OutputDistributed {

    %0 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%arg0: !InputDistributed)
            -> !OutputDistributed

    return %0 : !OutputDistributed

    // CHECK:        [[PERMUTECAST:%.*]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x256x40x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x256x40x1xf16, #NCHW, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]

    // return [[PERMUTECAST]]
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

func.func @PermuteCastMemPermute() -> memref<1x2xf32, #CN> {
    %cst = const.Declare memref<1x2xf32> = dense<[[1.0, 2.0]]> : memref<1x2xf32>
    %permute_cast = VPUIP.PermuteCast {dst_order = #CN, mem_perm = #CN} inputs(%cst : memref<1x2xf32>) -> memref<1x2xf32, #CN>
    return %permute_cast : memref<1x2xf32, #CN>
}

// CHECK: func.func @PermuteCastMemPermute() -> memref<1x2xf32, #CN> {
// CHECK:    [[CST:%.+]] = const.Declare memref<1x2xf32, #CN> = dense<{{\[\[}}1.000000e+00, 2.000000e+00]]> : memref<1x2xf32>, [#const.MemPermute<#CN, #CN>]
// CHECK:    return [[CST]] : memref<1x2xf32, #CN>
// CHECK: }

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#CN = affine_map<(d0, d1) -> (d1, d0)>

func.func @PermuteCastNoOp() -> memref<1x2xf32> {
    %cst = const.Declare memref<1x2xf32> = dense<[[1.0, 2.0]]> : memref<1x2xf32>
    %permute_cast_0 = VPUIP.PermuteCast {dst_order = #NC, mem_perm = #NC} inputs(%cst : memref<1x2xf32>) -> memref<1x2xf32>
    return %permute_cast_0 : memref<1x2xf32>
}

// CHECK: func.func @PermuteCastNoOp() -> memref<1x2xf32> {
// CHECK:     [[CST:%.+]] = const.Declare memref<1x2xf32> = dense<{{\[\[}}1.000000e+00, 2.000000e+00]]> : memref<1x2xf32>
// CHECK:     return [[CST]] : memref<1x2xf32>
// CHECK: }
