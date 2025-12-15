//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>
!qElemType1 = !quant.uniform<u8:f16, 0.01293658088235294:64>

!InDistributedTensor = !VPU.DistributedTensor<
    1x128x16x16x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 8, 16], [1, 128, 8, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    memory_shapes = [[1, 128, 8, 16], [1, 128, 8, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]
}>

!OutDistributedTensor = !VPU.DistributedTensor<
    1x128x16x16x!qElemType1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 8, 16], [1, 128, 8, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    memory_shapes = [[1, 128, 8, 16], [1, 128, 8, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]
}>

// CHECK-LABEL: @QuantizeCastOverlapped
func.func @QuantizeCastOverlapped(%arg0: !InDistributedTensor) -> !OutDistributedTensor {
    %0 = VPU.QuantizeCast(%arg0) {dstElemType = !qElemType1}
        : !InDistributedTensor -> !OutDistributedTensor
    return %0 : !OutDistributedTensor

    // CHECK:       VPU.QuantizeCast
    // CHECK-SAME:      : !VPU.DistributedTensor<1x128x16x16x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED"
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x128x16x16x!qElemType1, #NHWC, @CMX_NN, {mode = "OVERLAPPED"
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPU.DistributedTensor<
    1x1x256x40xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 1, 256, 20], [1, 1, 256, 20]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 20]],
    memory_shapes = [[1, 1, 256, 22], [1, 1, 256, 22]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 18]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x256x40x1xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    memory_shapes = [[1, 256, 22, 1], [1, 256, 22, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0]]
}>

// CHECK-LABEL: PermuteCastOverlappedDistributed
func.func @PermuteCastOverlappedDistributed(%arg0: !InputDistributed) -> !OutputDistributed {

    %0 = VPU.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW}
        : !InputDistributed -> !OutputDistributed

    return %0 : !OutputDistributed

    // CHECK:        VPU.PermuteCast
    // CHECK-SAME:      {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:         !VPU.DistributedTensor<1x1x256x40xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 256, 20], [1, 1, 256, 20]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 20]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 256, 22], [1, 1, 256, 22]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 18]]
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x256x40x1xf16, #NCHW, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 256, 22, 1], [1, 256, 22, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

!InputDistributed = !VPU.DistributedTensor<
    1x256x40x1xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    memory_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x40x1x256xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 20, 1, 256], [1, 20, 1, 256]],
    compute_offsets = [[0, 0, 0, 0], [0, 20, 0, 0]],
    memory_shapes = [[1, 20, 1, 256], [1, 20, 1, 256]],
    memory_offsets = [[0, 0, 0, 0], [0, 20, 0, 0]]
}>

// CHECK-LABEL: PermuteCastOverlappedToSegmentedDistributed
func.func @PermuteCastOverlappedToSegmentedDistributed(%arg0: !InputDistributed) -> !OutputDistributed {

    %0 = VPU.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW}
            : !InputDistributed -> !OutputDistributed

    return %0 : !OutputDistributed

    // CHECK:        VPU.PermuteCast
    // CHECK-SAME:      {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:         !VPU.DistributedTensor<1x256x40x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x40x1x256xf16, #NCHW, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 20, 1, 256], [1, 20, 1, 256]], compute_offsets = [[0, 0, 0, 0], [0, 20, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 20, 1, 256], [1, 20, 1, 256]], memory_offsets = [[0, 0, 0, 0], [0, 20, 0, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!InDistributedTensor = !VPU.DistributedTensor<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutDistributedTensor = !VPU.DistributedTensor<
    1x128x1x256xf16, #NWCH, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

// CHECK-LABEL: @AffineReshapeDuplicated
func.func @AffineReshapeDuplicated(%arg0: !InDistributedTensor) -> !OutDistributedTensor {
    %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 128, 1, 256]}
        : !InDistributedTensor -> !OutDistributedTensor
    return %0 : !OutDistributedTensor

    // CHECK:       VPU.AffineReshape
    // CHECK-SAME:      !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED"
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x128x1x256xf16, #NWCH, @CMX_NN, {mode = "DUPLICATED"
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

!InputDistributed = !VPU.DistributedTensor<
    1x256x40x1xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    memory_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x256x40x1xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    memory_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
}>

// CHECK-LABEL: LayoutCastDistributed
func.func @LayoutCastDistributed(%arg0: !InputDistributed) -> !OutputDistributed {

    %0 = VPU.LayoutCast(%arg0) {dst_order = #NCHW}
            : !InputDistributed -> !OutputDistributed

    return %0 : !OutputDistributed

    // CHECK:        VPU.LayoutCast
    // CHECK-SAME:      {dst_order = #NCHW}
    // CHECK-SAME:         !VPU.DistributedTensor<1x256x40x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x256x40x1xf16, #NCHW, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 256, 20, 1], [1, 256, 20, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
}
