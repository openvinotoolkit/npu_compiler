//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedTensor = !VPU.DistributedTensor<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

// CHECK-LABEL: @Fold
func.func @Fold(%arg0: tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> to !DistributedTensor
    %1 = VPU.DistributedCast(%0 : !DistributedTensor) -> !DistributedTensor
    %2 = builtin.unrealized_conversion_cast %1 : !DistributedTensor to tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    return %2 : tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK-NOT:  VPU.DistributedCast
    // CHECK:      return %arg0 : tensor<1x128x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InDistributedTensor = !VPU.DistributedTensor<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutDistributedTensor = !VPU.DistributedTensor<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2
}>

// CHECK-LABEL: @CompatibleDistributedCast
func.func @CompatibleDistributedCast(%arg0: !InDistributedTensor) -> !OutDistributedTensor {
    %0 = VPU.DistributedCast(%arg0 : !InDistributedTensor) -> !OutDistributedTensor
    return %0 : !OutDistributedTensor

    // CHECK:       VPU.DistributedCast
    // CHECK-SAME:      !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED"
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InDistributedTensor = !VPU.DistributedTensor<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 8, 16], [1, 128, 8, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    memory_shapes = [[1, 128, 8, 16], [1, 128, 8, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]
}>

!OutDistributedTensor = !VPU.DistributedTensor<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 8, 16], [1, 128, 8, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    memory_shapes = [[1, 128, 8, 16], [1, 128, 8, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]
}>

// CHECK-LABEL: @CompatibleDistributedCastWithExplicitDistribution
func.func @CompatibleDistributedCastWithExplicitDistribution(%arg0: !InDistributedTensor) -> !OutDistributedTensor {
    %0 = VPU.DistributedCast(%arg0 : !InDistributedTensor) -> !OutDistributedTensor
    return %0 : !OutDistributedTensor

    // CHECK:       VPU.DistributedCast
    // CHECK-SAME:      !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED"
}
