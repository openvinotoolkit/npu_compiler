//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x64x28x28xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x80x28x28xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!WeightsDistributed = !VPU.DistributedTensor<
    80x64x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!WeightsTableDistributed = !VPU.DistributedTensor<
    80x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

// CHECK:       func.func @CheckConv([[INPUT:%.+]]: !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
// CHECK-SAME:                                                                  {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
// CHECK-SAME:                                                  [[WEIGHTS:%.+]]: !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
// CHECK-SAME:                                                  [[WT:%.+]]: !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

func.func @CheckConv(%input: !InputDistributed, %weights: !WeightsDistributed,
                     %wt: !WeightsTableDistributed) -> !OutputDistributed {

    %convOut= VPU.NCE.Convolution(%input, %weights, %wt) rawFilterShape [80, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                                          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                                                          ppe = #VPU.PPEStub<>,
                                                           strides = [1, 1]} : !InputDistributed, !WeightsDistributed, !WeightsTableDistributed -> !OutputDistributed
    return %convOut : !OutputDistributed
}

//CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WT]]) rawFilterShape [80, 64, 3, 3] {
    // CHECK-SAME:                           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
//CHECK-SAME:                           pad = #VPU.Padding<left = 1 : i64, right = 1 : i64,
//CHECK-SAME:                                   top = 1 : i64, bottom = 1 : i64>,
//CHECK-SAME:                           ppe = #VPU.PPEStub<>,
// CHECK-SAME:                          strides = [1, 1]}
//CHECK-SAME:   -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

//CHECK:        return [[CONV]] : !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
