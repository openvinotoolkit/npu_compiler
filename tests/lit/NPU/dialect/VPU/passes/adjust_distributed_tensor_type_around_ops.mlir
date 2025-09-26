//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --adjust-distributed-tensor-around-ops %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed1 = !VPU.DistributedTensor<
    1x128x104x64xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 128, 52, 64], [1, 128, 52, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 52, 0]],
    memory_shapes = [[1, 128, 54, 64], [1, 128, 54, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 50, 0]]
 }>

!InputDistributed2 = !VPU.DistributedTensor<
    1x128x104x64xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 128, 52, 64], [1, 128, 52, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 52, 0]],
    memory_shapes = [[1, 128, 52, 64], [1, 128, 52, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 52, 0]]
}>

!WeightsDistributed = !VPU.DistributedTensor<
    64x128x1x1xi8, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[64, 128, 1, 1], [64, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 128, 1, 1], [64, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
 }>

 !WeightTableDistributed = !VPU.DistributedTensor<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x64x104x64xi8, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 52, 64], [1, 64, 52, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 52, 0]],
    memory_shapes = [[1, 64, 52, 64], [1, 64, 52, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 52, 0]]
}>

// CHECK-LABEL: @AdjustInputDistributedType
// CHECK-SAME:    ([[ARG0:%.+]]: !VPU.DistributedTensor<1x128x104x64xi8,
func.func @AdjustInputDistributedType(%arg0: !InputDistributed1) -> !OutputDistributed {
    %cst_0 = const.Declare tensor<64x128x1x1xi8, {order = #NHWC}> = dense<1> : tensor<64x128x1x1xi8>, [#const.Reorder<#NHWC>]
    %0 = VPU.Copy(%arg0) : !InputDistributed1 -> tensor<1x128x104x64xi8, {order = #NHWC}>
    %1 = VPU.Copy(%0) {out_mem_space = @CMX_NN} : tensor<1x128x104x64xi8, {order = #NHWC}> -> !InputDistributed2

    %2 = VPU.Copy(%cst_0) {out_mem_space = @CMX_NN} : tensor<64x128x1x1xi8, {order = #NHWC}> -> !WeightsDistributed
    %4 = VPU.NCE.Convolution(%1, %2) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} : !InputDistributed2, !WeightsDistributed -> !OutputDistributed
    return %4 : !OutputDistributed

    // CHECK:    [[WEIGHTS:%.+]] = const.Declare tensor<64x128x1x1xi8, {order = #NHWC}> = dense<1> : tensor<64x128x1x1xi8>, [#const.Reorder<#NHWC>]
    // CHECK:    [[INPUT_COPY0:%.+]] = VPU.Copy([[ARG0]]) :
    // CHECK-SAME{LITERAL}:      !VPU.DistributedTensor<1x128x104x64xi8, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 128, 52, 64], [1, 128, 52, 64]],
    // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 52, 0]],
    // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 128, 54, 64], [1, 128, 54, 64]],
    // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 50, 0]]}>
    // CHECK-SAME{LITERAL}:       -> tensor<1x128x104x64xi8, {order = #NHWC}>
    // CHECK:    [[INPUT_COPY1:%.+]] = VPU.Copy([[INPUT_COPY0]]) {out_mem_space = @CMX_NN} : tensor<1x128x104x64xi8, {order = #NHWC}> ->
    // CHECK-SAME:     !VPU.DistributedTensor<1x128x104x64xi8, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 128, 52, 64], [1, 128, 52, 64]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 52, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 128, 54, 64], [1, 128, 54, 64]],
    // CHECK-NOT{LITERAL}:       memory_shapes = [[1, 128, 52, 64], [1, 128, 52, 64]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 50, 0]]}>
    // CHECK-NOT{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 52, 0]]
    // CHECK:    [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN} : tensor<64x128x1x1xi8, {order = #NHWC}>
    // CHECK:    [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT_COPY1]], [[WEIGHTS_COPY]])
    // CHECK:    return [[CONV]]
}
