//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x1x96x160xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]],
    memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x1x49x160xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]],
    memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]
}>

// CHECK-LABEL: SliceWithExplicitOverlappedDistributedTensorType
func.func @SliceWithExplicitOverlappedDistributedTensorType(%arg0: !InputDistributed)
    -> (!OutputDistributed, !OutputDistributed) {

    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 1, 49, 160] : !InputDistributed to !OutputDistributed
    %1 = VPU.Slice %arg0 [0, 0, 47, 0] [1, 1, 49, 160] : !InputDistributed to !OutputDistributed
    return %0, %1 : !OutputDistributed, !OutputDistributed

    // CHECK:        [[SLICE0:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 1, 49, 160]
    // CHECK-SAME:         !VPU.DistributedTensor<1x1x96x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         to !VPU.DistributedTensor<1x1x49x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]

    // CHECK:        [[SLICE1:%.*]] = VPU.Slice %arg0 [0, 0, 47, 0] [1, 1, 49, 160]
    // CHECK-SAME:         !VPU.DistributedTensor<1x1x96x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         to !VPU.DistributedTensor<1x1x49x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x16x88x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 44, 128], [1, 16, 44, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 44, 0]],
    memory_shapes = [[1, 16, 44, 128], [1, 16, 44, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 44, 0]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x16x44x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]],
    memory_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]
}>

// CHECK-LABEL: SliceWithExplicitSegmentedDistributedTensorType
func.func @SliceWithExplicitSegmentedDistributedTensorType(%arg0: !InputDistributed)
    -> (!OutputDistributed, !OutputDistributed) {

    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 44, 128] : !InputDistributed to !OutputDistributed
    %1 = VPU.Slice %arg0 [0, 0, 44, 0] [1, 16, 44, 128] : !InputDistributed to !OutputDistributed
    return %0, %1 : !OutputDistributed, !OutputDistributed

    // CHECK:        [[SLICE0:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 44, 128]
    // CHECK-SAME:         !VPU.DistributedTensor<1x16x88x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         to !VPU.DistributedTensor<1x16x44x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]

    // CHECK:        [[SLICE1:%.*]] = VPU.Slice %arg0 [0, 0, 44, 0] [1, 16, 44, 128]
    // CHECK-SAME:         !VPU.DistributedTensor<1x16x88x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         to !VPU.DistributedTensor<1x16x44x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]

}
