//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --make-ops-with-distributed-tensor="enable-explicit-distributed-attr=true" --make-distributed-copies %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @ConvEltwiseConcatNoOverlapped
func.func @ConvEltwiseConcatNoOverlapped(%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>)
    -> tensor<1x128x56x56xf16, {order = #NHWC}> {
    %w0 = const.Declare tensor<128x256x3x3xf16, {order = #NHWC}>
            = dense<1.0> : tensor<128x256x3x3xf16, {order = #NHWC}>
    %w1 = const.Declare tensor<128x256x3x3xf16, {order = #NHWC}>
            = dense<1.0> : tensor<128x256x3x3xf16, {order = #NHWC}>
    %w2 = const.Declare tensor<128x256x3x3xf16, {order = #NHWC}>
            = dense<1.0> : tensor<128x256x3x3xf16, {order = #NHWC}>
    %0 = VPU.NCE.Convolution(%arg0, %w0) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [128, 256, 3, 3],
        strides = [1, 1]}
            : tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<128x256x3x3xf16, {order = #NHWC}> -> tensor<1x128x56x56xf16, {order = #NHWC}>

    %1 = VPU.NCE.Eltwise(%0, %0) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>}
            -> tensor<1x128x56x56xf16, {order = #NHWC}>

    %2 = VPU.NCE.Convolution(%arg0, %w1) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [128, 256, 3, 3],
        strides = [1, 1]}
            : tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<128x256x3x3xf16, {order = #NHWC}> -> tensor<1x128x56x56xf16, {order = #NHWC}>

    %3 = VPU.NCE.Eltwise(%0, %0) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>}
            -> tensor<1x128x56x56xf16, {order = #NHWC}>

    %4 = VPU.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]}:
            tensor<1x128x56x56xf16, {order = #NHWC}>, tensor<1x128x56x56xf16, {order = #NHWC}>
                -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %5 = VPU.NCE.Convolution(%4, %w1) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [128, 256, 3, 3],
        strides = [1, 1]}
            : tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<128x256x3x3xf16, {order = #NHWC}> -> tensor<1x128x56x56xf16, {order = #NHWC}>

    return %5 : tensor<1x128x56x56xf16, {order = #NHWC}>

    // Check the output memory shapes are OVERLAPPED but has no actual overlapped region
    // CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME-LITERAL:      -> !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64,
    // CHECK-SAME-LITERAL:      uniform_distributed_segments, compute_shapes = [[1, 128, 19, 56], [1, 128, 19, 56], [1, 128, 18, 56]], compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0]],
    // CHECK-SAME-LITERAL:      memory_shapes = [[1, 128, 19, 56], [1, 128, 19, 56], [1, 128, 18, 56]], memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0]]}>
    // CHECK-NOT-LITERAL:      memory_shapes = [[1, 128, 20, 56], [1, 128, 21, 56], [1, 128, 19, 56]], memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0]]}>
    // CHECK:       [[ELTWISE0:%.+]] = VPU.NCE.Eltwise
    // CHECK-SAME-LITERAL:      -> !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64,
    // CHECK-SAME-LITERAL:      uniform_distributed_segments, compute_shapes = [[1, 128, 19, 56], [1, 128, 19, 56], [1, 128, 18, 56]], compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0]],
    // CHECK-SAME-LITERAL:      memory_shapes = [[1, 128, 19, 56], [1, 128, 19, 56], [1, 128, 18, 56]], memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0]]}>
    // CHECK-NOT-LITERAL:      memory_shapes = [[1, 128, 20, 56], [1, 128, 21, 56], [1, 128, 19, 56]], memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0]]}>
    // CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME-LITERAL:      -> !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64,
    // CHECK-SAME-LITERAL:      uniform_distributed_segments, compute_shapes = [[1, 128, 19, 56], [1, 128, 19, 56], [1, 128, 18, 56]], compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0]],
    // CHECK-SAME-LITERAL:      memory_shapes = [[1, 128, 19, 56], [1, 128, 19, 56], [1, 128, 18, 56]], memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0]]}>
    // CHECK-NOT-LITERAL:      memory_shapes = [[1, 128, 20, 56], [1, 128, 21, 56], [1, 128, 19, 56]], memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0]]}>
    // CHECK:       [[ELTWISE1:%.+]] = VPU.NCE.Eltwise
    // CHECK-SAME-LITERAL:      -> !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64,
    // CHECK-SAME-LITERAL:      uniform_distributed_segments, compute_shapes = [[1, 128, 19, 56], [1, 128, 19, 56], [1, 128, 18, 56]], compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0]],
    // CHECK-SAME-LITERAL:      memory_shapes = [[1, 128, 19, 56], [1, 128, 19, 56], [1, 128, 18, 56]], memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0]]}>
    // CHECK-NOT-LITERAL:      memory_shapes = [[1, 128, 20, 56], [1, 128, 21, 56], [1, 128, 19, 56]], memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0]]}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToDistributedOpHKSwitch
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @ConvToDistributedOpHKSwitch(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [80, 64, 3, 3],
        strides = [1, 1]}
      : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[ARG0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]}

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToDistributedOpSOK4Clusters
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x28x28xf16, {order = #NHWC}>
func.func @ConvToDistributedOpSOK4Clusters(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} : tensor<1x128x28x28xf16, {order = #NHWC}>, tensor<64x128x1x1xf16, {order = #NHWC}> -> tensor<1x64x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<64x128x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 28, 28], [1, 64, 28, 28], [1, 64, 28, 28], [1, 64, 28, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToDistributedOpSOB3Batches
// CHECK-SAME:      [[INPUT:%.+]]: tensor<3x1024x14x14xf16, {order = #NHWC}>
func.func @ConvToDistributedOpSOB3Batches(%arg0: tensor<3x1024x14x14xf16, {order = #NHWC}>) -> tensor<3x256x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 1024, 1, 1], strides = [1, 1]} : tensor<3x1024x14x14xf16, {order = #NHWC}>, tensor<256x1024x1x1xf16, {order = #NHWC}> -> tensor<3x256x14x14xf16, {order = #NHWC}>
    return %0 : tensor<3x256x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<3x1024x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<256x1024x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:           [[INPUT_CMX]]
    //CHECK-SAME:           [[WEIGHTS_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<3x256x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<3x256x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToDistributedOpSOB
// CHECK-SAME:      [[INPUT:%.+]]: tensor<6x1024x14x14xf16, {order = #NHWC}>
func.func @ConvToDistributedOpSOB(%arg0: tensor<6x1024x14x14xf16, {order = #NHWC}>) -> tensor<6x256x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 1024, 1, 1], strides = [1, 1]} : tensor<6x1024x14x14xf16, {order = #NHWC}>, tensor<256x1024x1x1xf16, {order = #NHWC}> -> tensor<6x256x14x14xf16, {order = #NHWC}>
    return %0 : tensor<6x256x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<6x1024x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<256x1024x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:           [[INPUT_CMX]]
    //CHECK-SAME:           [[WEIGHTS_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<6x256x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<6x256x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToDistributedOpClustering
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x14x14xf16, {order = #NHWC}>
func.func @ConvToDistributedOpClustering(%arg0: tensor<1x64x14x14xf16, {order = #NHWC}>) -> tensor<1x48x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]} : tensor<1x64x14x14xf16, {order = #NHWC}>, tensor<48x64x3x3xf16, {order = #NHWC}> -> tensor<1x48x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x48x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<48x64x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:             [[WEIGHTS_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x48x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x48x14x14xf16, {order = #NHWC}>
}

}
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DepthConvToDistributedOpSOHOverlapped
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>
func.func @DepthConvToDistributedOpSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 20, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 20, 112], [1, 32, 19, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 37, 0], [0, 0, 56, 0], [0, 0, 75, 0], [0, 0, 93, 0]]

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.DepthConvolution(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:             [[WEIGHTS_CMX]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DepthConvToDistributedOpHKSwitch
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
func.func @DepthConvToDistributedOpHKSwitch(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_0) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 1, 3, 3],
        strides = [1, 1]}
            -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 20, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 20, 112], [1, 32, 19, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 37, 0], [0, 0, 56, 0], [0, 0, 75, 0], [0, 0, 93, 0]]

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.DepthConvolution(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:             [[WEIGHTS_CMX]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DepthConvToDistributedOpSOHOverlappedNoAlign
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x14x14xf16, {order = #NHWC}>)
func.func @DepthConvToDistributedOpSOHOverlappedNoAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:      [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:      [[INPUT_CMX:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1],
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 4, 14], [1, 32, 5, 14], [1, 32, 4, 14], [1, 32, 4, 14], [1, 32, 4, 14], [1, 32, 3, 14]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 5, 0], [0, 0, 7, 0], [0, 0, 9, 0], [0, 0, 11, 0]]}>

    //CHECK:      [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.DepthConvolution(
    //CHECK-SAME:         [[INPUT_CMX]]
    //CHECK-SAME:         [[WEIGHTS_CMX]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DepthConvToDistributedOpSOK
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x56x56xf16, {order = #NHWC}>
func.func @DepthConvToDistributedOpSOK(%arg0: tensor<1x128x56x56xf16, {order = #NHWC}>) -> tensor<1x128x56x56xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x56x56xf16, {order = #NHWC}>
    return %0 : tensor<1x128x56x56xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 56, 56], [1, 32, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0], [0, 96, 0, 0], [0, 112, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 56, 56], [1, 32, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0], [0, 96, 0, 0], [0, 112, 0, 0]]

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<128x16x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0], [96, 0, 0, 0], [112, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0], [96, 0, 0, 0], [112, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.DepthConvolution(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:             [[WEIGHTS_CMX]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 56, 56], [1, 32, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0], [0, 96, 0, 0], [0, 112, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 56, 56], [1, 128, 56, 56], [1, 128, 56, 56], [1, 128, 56, 56], [1, 128, 56, 56], [1, 128, 56, 56]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x128x56x56xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DepthConvToDistributedOpClustering
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x14x14xf16, {order = #NHWC}>
func.func @DepthConvToDistributedOpClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.DepthConvolution(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:             [[WEIGHTS_CMX]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MaxPoolToDistributedOpSOHOverlapped
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>
func.func @MaxPoolToDistributedOpSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.MaxPool(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MaxPoolToDistributedOpHKSwitch
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
func.func @MaxPoolToDistributedOpHKSwitch(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.MaxPool(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MaxPoolToDistributedOpSOHOverlappedNoAlign
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x14x14xf16, {order = #NHWC}>)
func.func @MaxPoolToDistributedOpSOHOverlappedNoAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.MaxPool(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:        {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy(
    //CHECK-SAME:       [[OUT_CMX]]
    //CHECK-SAME:         -> tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MaxPoolToDistributedOpClustering
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x14x14xf16, {order = #NHWC}>
func.func @MaxPoolToDistributedOpClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.MaxPool(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:  func.func @MaxPoolToDistributedOpSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<6x32x14x14xf16, {order = #NHWC}>)
func.func @MaxPoolToDistributedOpSOB(%input: tensor<6x32x14x14xf16, {order = #NHWC}>) -> tensor<6x32x14x14xf16, {order = #NHWC}> {
    %maxpool = VPU.NCE.MaxPool(%input) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<6x32x14x14xf16, {order = #NHWC}>
    return %maxpool : tensor<6x32x14x14xf16, {order = #NHWC}>

    // CHECK:                [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<6x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]
    // CHECK-SAME:           }>

    // CHECK:                [[OUT_CMX:%.+]] = VPU.NCE.MaxPool([[INPUT_CMX]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<6x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED",  num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]
    // CHECK-SAME:           }>

    // CHECK:                [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:                return [[OUT]] : tensor<6x32x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:  func.func @MaxPoolToDistributedOpSOB3Batches
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<3x32x14x14xf16, {order = #NHWC}>)
func.func @MaxPoolToDistributedOpSOB3Batches(%input: tensor<3x32x14x14xf16, {order = #NHWC}>) -> tensor<3x32x14x14xf16, {order = #NHWC}> {
    %maxpool = VPU.NCE.MaxPool(%input) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<3x32x14x14xf16, {order = #NHWC}>
    return %maxpool : tensor<3x32x14x14xf16, {order = #NHWC}>

    // CHECK:                [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<3x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]
    // CHECK-SAME:           }>

    // CHECK:                [[OUT_CMX:%.+]] = VPU.NCE.MaxPool([[INPUT_CMX]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<3x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED",  num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]
    // CHECK-SAME:           }>

    // CHECK:                [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:                return [[OUT]] : tensor<3x32x14x14xf16, {order = #NHWC}>
}

}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MaxPoolToDistributedOpSOKWithNWCHOutput
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<1x128x16x64xf16, {order = #NHWC}>)
func.func @MaxPoolToDistributedOpSOKWithNWCHOutput(%arg0: tensor<1x128x16x64xf16, {order = #NHWC}>) -> tensor<1x128x16x64xf16, {order = #NWCH}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x128x16x64xf16, {order = #NWCH}>
    return %0 : tensor<1x128x16x64xf16, {order = #NWCH}>

    // CHECK:                [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x128x16x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 16, 64], [1, 32, 16, 64], [1, 16, 16, 64], [1, 16, 16, 64], [1, 16, 16, 64], [1, 16, 16, 64]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0], [0, 96, 0, 0], [0, 112, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 16, 64], [1, 32, 16, 64], [1, 16, 16, 64], [1, 16, 16, 64], [1, 16, 16, 64], [1, 16, 16, 64]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0], [0, 96, 0, 0], [0, 112, 0, 0]]
    // CHECK-SAME:           }>

    // CHECK:                [[OUT_CMX:%.+]] = VPU.NCE.MaxPool([[INPUT_CMX]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x128x16x64xf16, #NWCH, @CMX_NN, {
    // CHECK-SAME:               mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 16, 64], [1, 32, 16, 64], [1, 16, 16, 64], [1, 16, 16, 64], [1, 16, 16, 64], [1, 16, 16, 64]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0], [0, 96, 0, 0], [0, 112, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 128, 16, 64], [1, 128, 16, 64], [1, 128, 16, 64], [1, 128, 16, 64], [1, 128, 16, 64], [1, 128, 16, 64]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:           }>

    // CHECK:                [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:                return [[OUT]] : tensor<1x128x16x64xf16, {order = #NWCH}>
}

}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MaxPoolToDistributedOpSOKWithNWCHOutputAndLargeChannel
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<1x512x16x64xf16, {order = #NHWC}>)
func.func @MaxPoolToDistributedOpSOKWithNWCHOutputAndLargeChannel(%arg0: tensor<1x512x16x64xf16, {order = #NHWC}>) -> tensor<1x512x16x64xf16, {order = #NWCH}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x512x16x64xf16, {order = #NWCH}>
    return %0 : tensor<1x512x16x64xf16, {order = #NWCH}>

    // CHECK:                [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x512x16x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 96, 16, 64], [1, 96, 16, 64], [1, 80, 16, 64], [1, 80, 16, 64], [1, 80, 16, 64], [1, 80, 16, 64]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 96, 16, 64], [1, 96, 16, 64], [1, 80, 16, 64], [1, 80, 16, 64], [1, 80, 16, 64], [1, 80, 16, 64]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]]
    // CHECK-SAME:           }>

    // CHECK:                [[OUT_CMX:%.+]] = VPU.NCE.MaxPool([[INPUT_CMX]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x512x16x64xf16, #NWCH, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 96, 16, 64], [1, 96, 16, 64], [1, 80, 16, 64], [1, 80, 16, 64], [1, 80, 16, 64], [1, 80, 16, 64]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 96, 16, 64], [1, 96, 16, 64], [1, 80, 16, 64], [1, 80, 16, 64], [1, 80, 16, 64], [1, 80, 16, 64]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]]
    // CHECK-SAME:           }>

    // CHECK:                [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:                return [[OUT]] : tensor<1x512x16x64xf16, {order = #NWCH}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseAddToDistributedOpSOHOverlapped
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>
func.func @EltwiseAddToDistributedOpSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} :
         tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
         -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.+]] = VPU.Copy([[INPUT0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]

    //CHECK:        [[INPUT1_CMX:%.+]] = VPU.Copy([[INPUT1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Eltwise(
    //CHECK-SAME:       [[INPUT0_CMX]],
    //CHECK-SAME:       [[INPUT1_CMX]])
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseAddToDistributedOpHKSwitch
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>,
// CHECK-SAME:   [[ARG1:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>
func.func @EltwiseAddToDistributedOpHKSwitch(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} :
         tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
         -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]

    //CHECK:        [[INPUT1_CMX:%.+]] = VPU.Copy([[ARG1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Eltwise(
    //CHECK-SAME:       [[INPUT0_CMX]],
    //CHECK-SAME:       [[INPUT1_CMX]])
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:               {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseAddToDistributedOpClustering
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x32x14x14xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x14x14xf16, {order = #NHWC}>
func.func @EltwiseAddToDistributedOpClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>, %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} :
         tensor<1x32x14x14xf16, {order = #NHWC}>, tensor<1x32x14x14xf16, {order = #NHWC}>
         -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0: tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.+]] = VPU.Copy([[INPUT0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[INPUT1_CMX:%.+]] = VPU.Copy([[INPUT1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Eltwise(
    //CHECK-SAME:       [[INPUT0_CMX]],
    //CHECK-SAME:       [[INPUT1_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AvgPoolToDistributedOpSOHOverlapped
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>
func.func @AvgPoolToDistributedOpSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 20, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 20, 112], [1, 32, 19, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 37, 0], [0, 0, 56, 0], [0, 0, 75, 0], [0, 0, 93, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.AveragePool(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AvgPoolToDistributedOpHKSwitch
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
func.func @AvgPoolToDistributedOpHKSwitch(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 20, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 20, 112], [1, 32, 19, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 37, 0], [0, 0, 56, 0], [0, 0, 75, 0], [0, 0, 93, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.AveragePool(
    //CHECK-SAME:             [[INPUT_CMX]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AvgPoolToDistributedOpSOHOverlappedNoAlign
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x14x14xf16, {order = #NHWC}>)
func.func @AvgPoolToDistributedOpSOHOverlappedNoAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:      [[INPUT_CMX:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 4, 14], [1, 32, 5, 14], [1, 32, 4, 14], [1, 32, 4, 14], [1, 32, 4, 14], [1, 32, 3, 14]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 5, 0], [0, 0, 7, 0], [0, 0, 9, 0], [0, 0, 11, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.AveragePool(
    //CHECK-SAME:          [[INPUT_CMX]]
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:  func.func @AvgPoolToDistributedOpSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<6x32x14x14xf16, {order = #NHWC}>)
func.func @AvgPoolToDistributedOpSOB(%input: tensor<6x32x14x14xf16, {order = #NHWC}>) -> tensor<6x32x14x14xf16, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%input) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<6x32x14x14xf16, {order = #NHWC}>
    return %avgpool : tensor<6x32x14x14xf16, {order = #NHWC}>

    // CHECK:                [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<6x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]
    // CHECK-SAME:           }>

    // CHECK:                [[OUT_CMX:%.+]] = VPU.NCE.AveragePool([[INPUT_CMX]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<6x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED",  num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]
    // CHECK-SAME:           }>

    // CHECK:                [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:                return [[OUT]] : tensor<6x32x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:  func.func @AvgPoolToDistributedOpSOB3Batches
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<3x32x14x14xf16, {order = #NHWC}>)
func.func @AvgPoolToDistributedOpSOB3Batches(%input: tensor<3x32x14x14xf16, {order = #NHWC}>) -> tensor<3x32x14x14xf16, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%input) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<3x32x14x14xf16, {order = #NHWC}>
    return %avgpool : tensor<3x32x14x14xf16, {order = #NHWC}>

    // CHECK:                [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<3x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]
    // CHECK-SAME:           }>

    // CHECK:                [[OUT_CMX:%.+]] = VPU.NCE.AveragePool([[INPUT_CMX]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<3x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED",  num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]
    // CHECK-SAME:           }>

    // CHECK:                [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:                return [[OUT]] : tensor<3x32x14x14xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceL1SplitOverKernel
module @ReduceL1SplitOverKernel {

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceL1SplitOverKernel(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
func.func @ReduceL1SplitOverKernel(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
  %0 = VPU.ReduceL1(%arg0) {axes_value = [2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
  return %0 : tensor<1x1024x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]], memory_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}>

    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceL1(%[[VAL_1]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1024x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]], memory_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}>

    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]

    // CHECK:   return %[[VAL_7]] : tensor<1x1024x1x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceL2SplitOverKernel
module @ReduceL2SplitOverKernel {

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceL2SplitOverKernel(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
func.func @ReduceL2SplitOverKernel(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
  %0 = VPU.ReduceL2(%arg0) {axes_value = [2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
  return %0 : tensor<1x1024x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]], memory_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}>

    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceL2(%[[VAL_1]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1024x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]], memory_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}>

    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1024x1x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceLogicalAndClustering
module @ReduceLogicalAndClustering {

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceLogicalAndClustering(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
func.func @ReduceLogicalAndClustering(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
  %0 = VPU.ReduceLogicalAnd(%arg0) {axes_value = [1, 2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = {{\[\[}}1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceLogicalAnd(%[[VAL_1]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = {{\[\[}}1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1x1x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceLogicalOrClustering
module @ReduceLogicalOrClustering {

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceLogicalOrClustering(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
func.func @ReduceLogicalOrClustering(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
  %0 = VPU.ReduceLogicalOr(%arg0) {axes_value = [1, 2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = {{\[\[}}1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceLogicalOr(%[[VAL_1]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = {{\[\[}}1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1x1x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceMaxSplitOverHeight
module @ReduceMaxSplitOverHeight {

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceMaxSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceMaxSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceMax(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}>

    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceMax(%[[VAL_1]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}>

    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceMeanSplitOverHeight
module @ReduceMeanSplitOverHeight {

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceMeanSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceMeanSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceMean(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}>

    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceMean(%[[VAL_1]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}>

    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceProdSplitOverHeight
module @ReduceProdSplitOverHeight {

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceProdSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceProdSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceProd(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}>

    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceProd(%[[VAL_1]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}>

    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceSumSplitOverHeight
module @ReduceSumSplitOverHeight {

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceSumSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceSumSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.Copy(%[[VAL_0]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}>

    // CHECK:   %[[VAL_4:.*]] = VPU.ReduceSum(%[[VAL_1]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}>
    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_4]]
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SparseConvToDistributedOpSOHOverlapped
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x64x28x28xi1, {order = #NHWC}>
func.func @SparseConvToDistributedOpSOHOverlapped(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 01 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [80, 64, 3, 3],
            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>, !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<80x1x1x640xi1>, is_weights> -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT0]], [[INPUT1]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00>
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    // CHECK:       [[INPUT_SPARSE_CMX:%.+]] = VPU.Copy
    // CHECK-SAME:      ([[INPUT_SPARSE]]
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    // CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPU.Copy
    // CHECK-SAME:      ([[WEIGHTS_SPARSE]])
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<80x1x1x640xi1, #NCHW, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          is_weights>

     // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    // CHECK-SAME:      [[INPUT_SPARSE_CMX]],
    // CHECK-SAME:      [[WEIGHTS_SPARSE_CMX]])
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x80x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]

    // CHECK:       [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SparseConvToDistributedOpHKSwitch
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>
// CHECK-SAME:   [[ARG1:%.+]]: tensor<1x64x28x28xi1, {order = #NHWC}>
func.func @SparseConvToDistributedOpHKSwitch(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 01 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [80, 64, 3, 3],
            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>, !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<80x1x1x640xi1>, is_weights> -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[ARG1]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00>
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    // CHECK:       [[INPUT_SPARSE_CMX:%.+]] = VPU.Copy
    // CHECK-SAME:      ([[INPUT_SPARSE]]
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    // CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPU.Copy
    // CHECK-SAME:      ([[WEIGHTS_SPARSE]]
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<80x1x1x640xi1, #NCHW, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          is_weights>

     // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    // CHECK-SAME:      [[INPUT_SPARSE_CMX]],
    // CHECK-SAME:      [[WEIGHTS_SPARSE_CMX]])
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x80x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DontSetAlignmentForConvEltwiseChainCase1
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x16x22x22xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x16x22x22xf16, {order = #NHWC}>
func.func @DontSetAlignmentForConvEltwiseChainCase1(%arg0: tensor<1x16x22x22xf16, {order = #NHWC}>, %arg1: tensor<1x16x22x22xf16, {order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} : tensor<1x16x22x22xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%0, %1) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    return %2 : tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX_0:%.+]] = VPU.Copy([[INPUT0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 5, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 5, 22], [1, 16, 4, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 7, 0], [0, 0, 11, 0], [0, 0, 15, 0], [0, 0, 18, 0]]

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_CMX_0:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX_0]],
    //CHECK-SAME:             [[WEIGHTS_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_CMX_0]]

    //CHECK:        [[INPUT0_CMX_1:%.+]] = VPU.Copy([[INPUT0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 5, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 5, 22], [1, 16, 4, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 7, 0], [0, 0, 11, 0], [0, 0, 15, 0], [0, 0, 18, 0]]

    //CHECK:        [[INPUT1_CMX_1:%.+]] = VPU.Copy([[INPUT1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 5, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 5, 22], [1, 16, 4, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 7, 0], [0, 0, 11, 0], [0, 0, 15, 0], [0, 0, 18, 0]]

    //CHECK:        [[OUT_CMX_1:%.+]] = VPU.NCE.Eltwise(
    //CHECK-SAME:       [[INPUT0_CMX_1]],
    //CHECK-SAME:       [[INPUT1_CMX_1]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]

    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_CMX_1]]

    //CHECK:        [[OUT_2:%.+]] = VPU.NCE.Eltwise([[OUT_0]], [[OUT_1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        return [[OUT_2]] : tensor<1x16x22x22xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DontSetAlignmentForConvEltwiseChainCase2
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x16x22x22xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x16x22x22xf16, {order = #NHWC}>
func.func @DontSetAlignmentForConvEltwiseChainCase2(%arg0: tensor<1x16x22x22xf16, {order = #NHWC}>, %arg1: tensor<1x16x22x22xf16, {order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %2 = VPU.NCE.Convolution(%1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} : tensor<1x16x22x22xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    return %2 : tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX_0:%.+]] = VPU.Copy([[INPUT0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]

    //CHECK:        [[INPUT1_CMX_0:%.+]] = VPU.Copy([[INPUT1]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]

    //CHECK:        [[OUT_CMX_0:%.+]] = VPU.NCE.Eltwise(
    //CHECK-SAME:       [[INPUT0_CMX_0]],
    //CHECK-SAME:       [[INPUT1_CMX_0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_CMX_0]]

    //CHECK:        [[INPUT0_CMX_1:%.+]] = VPU.Copy([[OUT_0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]

    //CHECK:        [[INPUT1_CMX_1:%.+]] = VPU.Copy([[INPUT1]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]

    //CHECK:        [[OUT_CMX_1:%.+]] = VPU.NCE.Eltwise(
    //CHECK-SAME:       [[INPUT0_CMX_1]],
    //CHECK-SAME:       [[INPUT1_CMX_1]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 5, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 5, 22], [1, 16, 4, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 7, 0], [0, 0, 11, 0], [0, 0, 15, 0], [0, 0, 18, 0]]

    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_CMX_1]]

    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX_2:%.+]] = VPU.Copy([[OUT_1]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 5, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 5, 22], [1, 16, 4, 22]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 7, 0], [0, 0, 11, 0], [0, 0, 15, 0], [0, 0, 18, 0]]

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_CMX_2:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX_2]],
    //CHECK-SAME:             [[WEIGHTS_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]

    //CHECK:        [[OUT_2:%.+]] = VPU.Copy([[OUT_CMX_2]]

    //CHECK:        return [[OUT_2]] : tensor<1x16x22x22xf16, {order = #NHWC}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MVNToDistributedOpDuplicateBuffer
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x512x1xf16, {order = #NCWH}>
func.func @MVNToDistributedOpDuplicateBuffer(%arg0: tensor<1x4x512x1xf16, {order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>

    return %0: tensor<1x4x512x1xf16, {order = #NCWH}>

    //CHECK:            [[ClusterCopy:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:            [[RClusterMVN:%.+]] = VPU.MVN([[ClusterCopy]]
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK: [[OUT:%.+]] = VPU.Copy([[RClusterMVN]]

    //CHECK: return [[OUT]] : tensor<1x4x512x1xf16, {order = #NCWH}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MVNToDistributedOpSegmentedBuffer
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x12x512x1xf16, {order = #NCWH}>
func.func @MVNToDistributedOpSegmentedBuffer(%arg0: tensor<1x12x512x1xf16, {order = #NCWH}>) -> tensor<1x12x512x1xf16, {order = #NCWH}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x12x512x1xf16, {order = #NCWH}> -> tensor<1x12x512x1xf16, {order = #NCWH}>

    return %0: tensor<1x12x512x1xf16, {order = #NCWH}>

    //CHECK:            [[ClusterCopy:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x12x512x1xf16, #NCWH, @CMX_NN,
    //CHECK-SAME:               {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0], [0, 10, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0], [0, 10, 0, 0]]

    //CHECK:            [[RClusterMVN:%.+]] = VPU.MVN([[ClusterCopy]]
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x12x512x1xf16, #NCWH, @CMX_NN,
    //CHECK-SAME:               {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0], [0, 10, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1]],
    //CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0], [0, 10, 0, 0]]

    //CHECK:            [[OUT:%.+]] = VPU.Copy([[RClusterMVN]]

    //CHECK: return [[OUT]] : tensor<1x12x512x1xf16, {order = #NCWH}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MVNToDistributedOpSegmentedBufferReducedClusters
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x512x1xf16, {order = #NCWH}>
func.func @MVNToDistributedOpSegmentedBufferReducedClusters(%arg0: tensor<1x4x512x1xf16, {order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>

    return %0: tensor<1x4x512x1xf16, {order = #NCWH}>

    //CHECK:        [[ClusterCopy:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN,
    //CHECK-SAME:               {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]

    //CHECK:        [[RClusterMVN:%.+]] = VPU.MVN([[ClusterCopy]]
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN,
    //CHECK-SAME:               {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[RClusterMVN]]

    //CHECK: return [[OUT]] : tensor<1x4x512x1xf16, {order = #NCWH}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVN6SOK4
module @MVN6SOK4 {

config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @MVN6SOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x32x15x64xf16>)
func.func @MVN6SOK(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [2], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,

    //CHECK:        [[MVN:%.+]] = VPU.MVN6([[INPUT]])
    //CHECK-SAME:                   -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[MVN]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x15x64xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVN6SOH4
module @MVN6SOH4 {

config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @MVN6SOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x32x15x64xf16>)
func.func @MVN6SOH(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [1, 3], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,

    //CHECK:        [[MVN:%.+]] = VPU.MVN6([[INPUT]]
    //CHECK-SAME:                   -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[MVN]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x15x64xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PadSOH4
module @PadSOH4 {

config.Resources 4 of @NCE at 1.700000e+03 MHz
// CHECK-LABEL: func.func @PadSwSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x32x50xf16>)
func.func @PadSwSOH(%arg0: tensor<1x16x32x50xf16>) -> tensor<1x17x32x60xf16> {
    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 1, 0, 10]} : tensor<1x16x32x50xf16> -> tensor<1x17x32x60xf16>
    return %0 : tensor<1x17x32x60xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                     -> !VPU.DistributedTensor<1x16x32x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50]],
    //CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50]],
    //CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]}>

    //CHECK:        [[PAD:%.+]] = VPU.Pad([[INPUT]])
    //CHECK-SAME:                   -> !VPU.DistributedTensor<1x17x32x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60]],
    //CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60]],
    //CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[PAD]]
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PadSOK4
module @PadSOK4 {

config.Resources 4 of @NCE at 1.700000e+03 MHz
// CHECK-LABEL: func.func @PadSwSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x30x50xf16>)
func.func @PadSwSOK(%arg0: tensor<1x16x30x50xf16>) -> tensor<1x16x33x53xf16> {
    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 3, 3]} : tensor<1x16x30x50xf16> -> tensor<1x16x33x53xf16>
    return %0 : tensor<1x16x33x53xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                     -> !VPU.DistributedTensor<1x16x30x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50]],
    //CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    //CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50]],
    //CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    //CHECK:        [[PAD:%.+]] = VPU.Pad([[INPUT]])
    //CHECK-SAME:                   -> !VPU.DistributedTensor<1x16x33x53xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53]],
    //CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    //CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53]],
    //CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[PAD]]
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @UnrollSOKConvOutputSegmented
func.func @UnrollSOKConvOutputSegmented(%input: tensor<1x64x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv = VPU.NCE.Convolution(%input, %weights) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [64, 64, 1, 1], strides = [1, 1]}
                : tensor<1x64x64x64xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>
    %mvn = VPU.MVN(%conv) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true
            } : tensor<1x64x64x64xf16, {order = #NHWC}>
                -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %mvn : tensor<1x64x64x64xf16, {order = #NHWC}>

    // (DUP 4 CL) CONV (SEG 4 CL) -> (SEG 6 CL) MVN (SEG 6 CL)

    //CHECK:        [[CONV_IN:%.+]] = VPU.Copy(%arg0
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[SOK_CONV:%.+]] = VPU.NCE.Convolution([[CONV_IN]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]

    //CHECK:        [[MVN_IN:%.+]] = VPU.Copy(%3
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]

    //CHECK:        [[SOK_MVN:%.+]] = VPU.MVN([[MVN_IN]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @UnrollSOKDWConvInputOutputDuplicated
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x320x1xf16>
func.func @UnrollSOKDWConvInputOutputDuplicated(%input: tensor<1x1x320x1xf16>) -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<320x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x320xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 320]>, #const.Reshape<[1, 320, 1, 1]>, #const.Reshape<[320, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[320, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]

    %mvn = VPU.MVN(%input) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true}
            : tensor<1x1x320x1xf16> -> tensor<1x1x320x1xf16>

    %reshape = VPU.AffineReshape(%mvn) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]}
            : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>

    %cast = VPU.PermuteCast(%reshape) {dst_order = #NHWC, mem_perm = #NHWC}
            : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    %dwconv = VPU.NCE.DepthConvolution(%cast, %weights) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [320, 1, 1, 1], strides = [1, 1]}
                -> tensor<1x320x1x1xf16, {order = #NHWC}>

    %activation = VPU.Sigmoid(%dwconv) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
            : tensor<1x320x1x1xf16, {order = #NHWC}> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    return %activation : tensor<1x320x1x1xf16, {order = #NHWC}>

    // (DUP) MVN (DUP) -> (DUP) DWCONV (SEG) -> (SEG) Sigmoid (SEG)

    //CHECK:        [[MVN_COPY_IN:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[MVN:%.+]] = VPU.MVN([[MVN_COPY_IN]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[MVN_COPY_OUT:%.+]] = VPU.Copy([[MVN]])
    //CHECK-SAME:       -> tensor<1x1x320x1xf16>

    //CHECK:        [[RESHAPE:%.+]] = VPU.AffineReshape([[MVN_COPY_OUT]])
    //CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]} : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>
    //CHECK:        [[CAST:%.+]] = VPU.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    //CHECK:        [[DWCONV_INPUT_COPY_IN:%.+]] = VPU.Copy([[CAST]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[DWCONV_WEIGHTS_COPY_IN:%.+]] = VPU.Copy(%cst)
    //CHECK-SAME:       -> !VPU.DistributedTensor<320x16x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0], [176, 0, 0, 0], [224, 0, 0, 0], [272, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0], [176, 0, 0, 0], [224, 0, 0, 0], [272, 0, 0, 0]]}>

    //CHECK:        [[DWCONV:%.*]] = VPU.NCE.DepthConvolution([[DWCONV_INPUT_COPY_IN]],
    //CHECK-SAME:                                           [[DWCONV_WEIGHTS_COPY_IN]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>

    //CHECK:        [[DWCONV_COPY_OUT:%.+]] = VPU.Copy([[DWCONV]])
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}>

    //CHECK:        [[SIGMOID_COPY_IN:%.+]] = VPU.Copy([[DWCONV_COPY_OUT]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>

    //CHECK:        [[SIGMOID:%.+]] = VPU.Sigmoid([[SIGMOID_COPY_IN]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>

    //CHECK:        [[SIGMOID_COPY_OUT:%.+]] = VPU.Copy([[SIGMOID]])
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @UnrollSOKConvOutputSegmentedWithSlice
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x80x1x3000xf16, {order = #NHWC}>
func.func @UnrollSOKConvOutputSegmentedWithSlice(%input: tensor<1x80x1x3000xf16, {order = #NHWC}>) -> tensor<1x384x1x1500xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<384x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x80x1x3xf16>, [#const.Reorder<#NHWC>]
    %conv = VPU.NCE.Convolution(%input, %weights) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [384, 80, 1, 3], strides = [1, 1]}
               : tensor<1x80x1x3000xf16, {order = #NHWC}>, tensor<384x80x1x3xf16, {order = #NHWC}> -> tensor<1x384x1x3000xf16, {order = #NHWC}>
    %slice = VPU.Slice %conv [0, 0, 0, 0] [1, 384, 1, 1500] : tensor<1x384x1x3000xf16, {order = #NHWC}> to tensor<1x384x1x1500xf16, {order = #NHWC}>
    %gelu =  VPU.Gelu(%slice) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x384x1x1500xf16, {order = #NHWC}> -> tensor<1x384x1x1500xf16, {order = #NHWC}>

    return %gelu : tensor<1x384x1x1500xf16, {order = #NHWC}>

    // (DUP) CONV (SEG) -> SLICE -> (SEG) GELU (SEG)

    // CHECK:       [[CONV_IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x80x1x3000xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       [[SOK_CONV:%.+]] = VPU.NCE.Convolution([[CONV_IN]]
    // CHECK-SAME:    -> !VPU.DistributedTensor<1x384x1x3000xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 256, 0, 0], [0, 320, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 256, 0, 0], [0, 320, 0, 0]]

    // CHECK:       [[CONV_OUT:%.+]] = VPU.Copy([[SOK_CONV]])
    // CHECK-SAME:     -> tensor<1x384x1x3000xf16, {order = #NHWC}>

    // CHECK:       [[SLICE:%.+]] = VPU.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 384, 1, 1500] : tensor<1x384x1x3000xf16, {order = #NHWC}> to tensor<1x384x1x1500xf16, {order = #NHWC}>

    // CHECK:       [[GELU_IN:%.+]] = VPU.Copy([[SLICE]])
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x384x1x1500xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 256, 0, 0], [0, 320, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 256, 0, 0], [0, 320, 0, 0]]

    // CHECK:       [[SOK_GELU:%.+]] = VPU.Gelu([[GELU_IN]])
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x384x1x1500xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 256, 0, 0], [0, 320, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 256, 0, 0], [0, 320, 0, 0]]
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @UnrollSOKDWConvInputOutputSegmented
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x64x64xf16, {order = #NHWC}>
func.func @UnrollSOKDWConvInputOutputSegmented(%input: tensor<1x64x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %mvn = VPU.MVN(%input) {across_channels = false, eps = 1.0E-4 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true}
            : tensor<1x64x64x64xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>
    %dwconv = VPU.NCE.DepthConvolution(%mvn, %weights) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [64, 1, 1, 1], strides = [1, 1]}
                -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %dwconv : tensor<1x64x64x64xf16, {order = #NHWC}>

    // (SEG 6 CL) MVN (SEG 6 CL) -> (SEG 4 CL) DWCONV (SEG|DUP 4 CL)
    // DW is SEG|DUP since only consequent SW layer is compatible with SEG, in all other cases it is SEG|DUP

    //CHECK:        [[MVN_IN:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]

    //CHECK:        [[SOK_MVN:%.+]] = VPU.MVN([[MVN_IN]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]

    //CHECK:        [[DWCONV_IN:%.+]] = VPU.Copy(%2
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]

    //CHECK:        [[SOK_DWCONV:%.+]] = VPU.NCE.DepthConvolution
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsToNCEClusteringKHSwitch
func.func @ChainOpsToNCEClusteringKHSwitch(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x96x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} : tensor<1x128x28x28xf16, {order = #NHWC}>, tensor<96x128x3x3xf16, {order = #NHWC}> -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [96, 96, 5, 5], strides = [2, 2]} : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<96x96x5x5xf16, {order = #NHWC}> -> tensor<1x96x14x14xf16, {order = #NHWC}>
    return %1 : tensor<1x96x14x14xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS_0:%.+]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[WEIGHTS_0_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,

    //CHECK:        [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_0_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]])
    //CHECK-SAME:       -> tensor<1x96x28x28xf16, {order = #NHWC}>

    //CHECK:        [[OUT_0_COPYBACK:%.+]] = VPU.Copy([[OUT_0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[WEIGHTS_1_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x5x5xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK]],
    //CHECK-SAME:             [[WEIGHTS_1_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 3, 14], [1, 96, 3, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 3, 14], [1, 96, 3, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]

    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]]

    //CHECK:        return [[OUT_1]] : tensor<1x96x14x14xf16, {order = #NHWC}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsToNCEClusteringSOHOverlapped
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x28x28xf16, {order = #NHWC}>
func.func @ChainOpsToNCEClusteringSOHOverlapped(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x96x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} : tensor<1x128x28x28xf16, {order = #NHWC}>, tensor<96x128x3x3xf16, {order = #NHWC}> -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [96, 96, 5, 5], strides = [2, 2]} : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<96x96x5x5xf16, {order = #NHWC}> -> tensor<1x96x14x14xf16, {order = #NHWC}>
    return %1 : tensor<1x96x14x14xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS_0:%.+]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 4, 28], [1, 128, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 6, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 6, 28], [1, 128, 5, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    //CHECK:        [[WEIGHTS_0_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_0_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]])
    //CHECK-SAME:       -> tensor<1x96x28x28xf16, {order = #NHWC}>

    //CHECK:        [[OUT_0_COPYBACK:%.+]] = VPU.Copy([[OUT_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]

    //CHECK:        [[WEIGHTS_1_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x5x5xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK]],
    //CHECK-SAME:             [[WEIGHTS_1_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 3, 14], [1, 96, 3, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 3, 14], [1, 96, 3, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]

    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]])
    //CHECK-SAME:        -> tensor<1x96x14x14xf16, {order = #NHWC}>

    //CHECK:        return [[OUT_1]] : tensor<1x96x14x14xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainSparseOpsoDistributedOpSOHOverlapped
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x64x28x28xi1, {order = #NHWC}>
func.func @ChainSparseOpsoDistributedOpSOHOverlapped(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<64x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<64x1x1x640xi1> = dense<1.000000e+00> : tensor<64x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<64x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<64x1x1x640xi1>, is_weights>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [64, 64, 3, 3],
            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>, !VPU.SparseTensor<data=tensor<64x64x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x640xi1>, is_weights> -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 32, 32] <left = 1 , right = 1, top = 1, bottom = 1> #VPU.mpe_mode<VECTOR_FP16>
        }

    %1 = VPU.NCE.Convolution(%0, %weights_sparse) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [64, 64, 3, 3],
            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>, !VPU.SparseTensor<data=tensor<64x64x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x640xi1>, is_weights> -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 32, 32] <left = 1 , right = 1, top = 1, bottom = 1> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %1 : !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT0]], [[INPUT1]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<64x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<64x1x1x640xi1> = dense<1.000000e+00>
    // CHECK-DAG:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<64x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<64x1x1x640xi1>, is_weights>
    // CHECK:       [[INPUT_SPARSE_CMX:%.+]] = VPU.Copy
    // CHECK-SAME:      ([[INPUT_SPARSE]])
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    // CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPU.Copy
    // CHECK-SAME:      ([[WEIGHTS_SPARSE]]
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<64x64x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<64x1x1x640xi1, #NCHW, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          is_weights>

     // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    // CHECK-SAME:      [[INPUT_SPARSE_CMX]],
    // CHECK-SAME:      [[WEIGHTS_SPARSE_CMX]])
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    // CHECK:       [[OUT:%.+]] = VPU.Copy([[OUT_CMX]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[OUT_COPYBACK:%.+]] = VPU.Copy
    // CHECK-SAME:      ([[OUT]])
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    // CHECK:       [[WEIGHTS_1_SPARSE_CMX:%.+]] = VPU.Copy
    // CHECK-SAME:      ([[WEIGHTS_SPARSE]])
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<64x64x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<64x1x1x640xi1, #NCHW, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          is_weights>

     // CHECK:       [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    // CHECK-SAME:      [[OUT_COPYBACK]],
    // CHECK-SAME:      [[WEIGHTS_1_SPARSE_CMX]])
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]

    // CHECK:       [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK:       return [[OUT_1]] : !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsMultipleConsumersToNCEClusteringSOHOverlapped
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x28x28xf16, {order = #NHWC}>
func.func @ChainOpsMultipleConsumersToNCEClusteringSOHOverlapped(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>)
    -> (tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<1x96x28x28xf16, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} : tensor<1x128x28x28xf16, {order = #NHWC}>, tensor<96x128x3x3xf16, {order = #NHWC}> -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 96, 3, 3], strides = [1, 1]} : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<96x96x3x3xf16, {order = #NHWC}> -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [96, 96, 5, 5], strides = [1, 1]} : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<96x96x5x5xf16, {order = #NHWC}> -> tensor<1x96x28x28xf16, {order = #NHWC}>
    return %1, %2 : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<1x96x28x28xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS_0:%.+]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_2:%.+]] = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]

    // Conv producer

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 4, 28], [1, 128, 4, 28]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 128, 6, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 6, 28], [1, 128, 5, 28]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    //CHECK:        [[WEIGHTS_0_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_0_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 9, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 18, 0], [0, 0, 22, 0]

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]])
    //CHECK-SAME:       -> tensor<1x96x28x28xf16, {order = #NHWC}>

    // First conv comsumer

    //CHECK:        [[OUT_0_COPYBACK:%.+]] = VPU.Copy([[OUT_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 9, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 18, 0], [0, 0, 22, 0]]

    //CHECK:        [[WEIGHTS_1_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK]],
    //CHECK-SAME:             [[WEIGHTS_1_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]

    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]]

    // Second conv comsumer

    //CHECK:        [[OUT_0_COPYBACK_1:%.+]] = VPU.Copy([[OUT_0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 9, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 18, 0], [0, 0, 22, 0]

    //CHECK:        [[WEIGHTS_2_CMX:%.+]] = VPU.Copy([[WEIGHTS_2]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x5x5xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_2_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK_1]],
    //CHECK-SAME:             [[WEIGHTS_2_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]

    //CHECK:        [[OUT_2:%.+]] = VPU.Copy([[OUT_2_CMX]]

    //CHECK:        return [[OUT_1]], [[OUT_2]] : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<1x96x28x28xf16, {order = #NHWC}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedSiblingsMemViewUnion0
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x28x28xf16, {order = #NHWC}>
func.func @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedSiblingsMemViewUnion0(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>)
    -> (tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<1x96x14x14xf16, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} : tensor<1x128x28x28xf16, {order = #NHWC}>, tensor<96x128x3x3xf16, {order = #NHWC}> -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 96, 3, 3], strides = [1, 1]} : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<96x96x3x3xf16, {order = #NHWC}> -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [96, 96, 5, 5], strides = [2, 2]} : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<96x96x5x5xf16, {order = #NHWC}> -> tensor<1x96x14x14xf16, {order = #NHWC}>
    return %1, %2 : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<1x96x14x14xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS_0:%.+]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_2:%.+]] = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]

    // Conv producer

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 4, 28], [1, 128, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 6, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 6, 28], [1, 128, 5, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    //CHECK:        [[WEIGHTS_0_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_0_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]])
    //CHECK-SAME:       -> tensor<1x96x28x28xf16, {order = #NHWC}>

    // First conv consumer
    //
    // Requirements for this consumer only, w/o sibling:
    //  memory_shapes = [[1, 96, 6, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28], [1, 96, 5, 28]]
    //  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    //CHECK:        [[OUT_0_COPYBACK:%.+]] = VPU.Copy([[OUT_0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]

    //CHECK:        [[WEIGHTS_1_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK]],
    //CHECK-SAME:             [[WEIGHTS_1_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]

    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]])
    //CHECK-SAME:       -> tensor<1x96x28x28xf16, {order = #NHWC}>

    // Second conv comsumer
    //
    // Requirements for this consumer only, w/o sibling:
    //  memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28]]
    //  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]

    //CHECK:        [[OUT_0_COPYBACK_1:%.+]] = VPU.Copy([[OUT_0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]

    //CHECK:        [[WEIGHTS_2_CMX:%.+]] = VPU.Copy([[WEIGHTS_2]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x5x5xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_2_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK_1]],
    //CHECK-SAME:             [[WEIGHTS_2_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 3, 14], [1, 96, 3, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 3, 14], [1, 96, 3, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]

    //CHECK:        [[OUT_2:%.+]] = VPU.Copy([[OUT_2_CMX]]

    //CHECK:        return [[OUT_1]], [[OUT_2]] : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<1x96x14x14xf16, {order = #NHWC}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedSiblingsMemViewUnion1
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x28x28xf16, {order = #NHWC}>
func.func @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedSiblingsMemViewUnion1(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>)
    -> (tensor<1x96x26x26xf16, {order = #NHWC}>, tensor<1x96x27x27xf16, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<96x96x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x4x4xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} : tensor<1x128x28x28xf16, {order = #NHWC}>, tensor<96x128x3x3xf16, {order = #NHWC}> -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 3, 3], strides = [1, 1]} : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<96x96x3x3xf16, {order = #NHWC}> -> tensor<1x96x26x26xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 2 : i64, right = 0 : i64, top = 2 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 4, 4], strides = [1, 1]} : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<96x96x4x4xf16, {order = #NHWC}> -> tensor<1x96x27x27xf16, {order = #NHWC}>
    return %1, %2 : tensor<1x96x26x26xf16, {order = #NHWC}>, tensor<1x96x27x27xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS_0:%.+]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_2:%.+]] = const.Declare tensor<96x96x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x4x4xf16>, [#const.Reorder<#NHWC>]

    // Conv producer

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 4, 28], [1, 128, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 6, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 6, 28], [1, 128, 5, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    //CHECK:        [[WEIGHTS_0_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_0_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]]

    // First conv consumer
    //
    // Requirements for this consumer only, w/o sibling:
    //  memory_shapes = [[1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28], [1, 96, 6, 28], [1, 96, 6, 28], [1, 96, 6, 28]]
    //  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]

    //CHECK:        [[OUT_0_COPYBACK:%.+]] = VPU.Copy([[OUT_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]

    //CHECK:        [[WEIGHTS_1_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK]],
    //CHECK-SAME:             [[WEIGHTS_1_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x26x26xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 26], [1, 96, 5, 26], [1, 96, 4, 26], [1, 96, 4, 26], [1, 96, 4, 26], [1, 96, 4, 26]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 5, 26], [1, 96, 5, 26], [1, 96, 4, 26], [1, 96, 4, 26], [1, 96, 4, 26], [1, 96, 4, 26]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]

    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]]

    // Second conv consumer
    //
    // Requirements for this consumer only, w/o sibling:
    //  memory_shapes = [[1, 96, 6, 28], [1, 96, 8, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28]]
    //  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]

    //CHECK:        [[OUT_0_COPYBACK_1:%.+]] = VPU.Copy([[OUT_0]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]

    //CHECK:        [[WEIGHTS_2_CMX:%.+]] = VPU.Copy([[WEIGHTS_2]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x4x4xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_2_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK_1]],
    //CHECK-SAME:             [[WEIGHTS_2_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x27x27xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 27], [1, 96, 5, 27], [1, 96, 5, 27], [1, 96, 4, 27], [1, 96, 4, 27], [1, 96, 4, 27]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 19, 0], [0, 0, 23, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 5, 27], [1, 96, 5, 27], [1, 96, 5, 27], [1, 96, 4, 27], [1, 96, 4, 27], [1, 96, 4, 27]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    //CHECK:        [[OUT_2:%.+]] = VPU.Copy([[OUT_2_CMX]]

    //CHECK:        return [[OUT_1]], [[OUT_2]] : tensor<1x96x26x26xf16, {order = #NHWC}>, tensor<1x96x27x27xf16, {order = #NHWC}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedImproperSplitForOutputShape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x8x8xf16, {order = #NHWC}>
// Between the two conv siblings, even though one has the bigger kernel, the inferred output shapes per cluster
// don't fully satisfy H >= 1 for each tile.
func.func @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedImproperSplitForOutputShape(%arg0: tensor<1x128x8x8xf16, {order = #NHWC}>)
    -> (tensor<1x96x1x1xf16, {order = #NHWC}>, tensor<1x96x8x8xf16, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<96x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x8x8xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x8x8xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<96x96x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 128, 1, 1], strides = [1, 1]} : tensor<1x128x8x8xf16, {order = #NHWC}>, tensor<96x128x1x1xf16, {order = #NHWC}> -> tensor<1x96x8x8xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 8, 8], strides = [8, 8]} : tensor<1x96x8x8xf16, {order = #NHWC}>, tensor<96x96x8x8xf16, {order = #NHWC}> -> tensor<1x96x1x1xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 1, 1], strides = [1, 1]} : tensor<1x96x8x8xf16, {order = #NHWC}>, tensor<96x96x1x1xf16, {order = #NHWC}> -> tensor<1x96x8x8xf16, {order = #NHWC}>
    return %1, %2 : tensor<1x96x1x1xf16, {order = #NHWC}>, tensor<1x96x8x8xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS_0:%.+]] = const.Declare tensor<96x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<96x96x8x8xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x8x8xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_2:%.+]] = const.Declare tensor<96x96x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x1x1xf16>, [#const.Reorder<#NHWC>]

    // Conv producer

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x8x8xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 2, 8], [1, 128, 2, 8], [1, 128, 1, 8], [1, 128, 1, 8], [1, 128, 1, 8], [1, 128, 1, 8]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 2, 8], [1, 128, 2, 8], [1, 128, 1, 8], [1, 128, 1, 8], [1, 128, 1, 8], [1, 128, 1, 8]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]]

    //CHECK:        [[WEIGHTS_0_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_0_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 2, 8], [1, 96, 2, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 2, 8], [1, 96, 2, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]]

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]]

    // First conv comsumer
    //
    // Op is incompatible with SOH strategy, do not take into account when computin overlapped params for consumer or sibling.
    //

    //CHECK:        [[OUT_0_COPYBACK:%.+]] = VPU.Copy([[OUT_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[WEIGHTS_1_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x8x8xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]]

    //CHECK:        [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK]],
    //CHECK-SAME:             [[WEIGHTS_1_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 1, 1], [1, 96, 1, 1], [1, 96, 1, 1], [1, 96, 1, 1], [1, 96, 1, 1], [1, 96, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]]

    // Second conv comsumer

    //CHECK:        [[OUT_0_COPYBACK_1:%.+]] = VPU.Copy([[OUT_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 2, 8], [1, 96, 2, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 2, 8], [1, 96, 2, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]]

    //CHECK:        [[WEIGHTS_2_CMX:%.+]] = VPU.Copy([[WEIGHTS_2]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_2_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK_1]],
    //CHECK-SAME:             [[WEIGHTS_2_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 2, 8], [1, 96, 2, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 2, 8], [1, 96, 2, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]]

    //CHECK:        [[OUT_2:%.+]] = VPU.Copy([[OUT_2_CMX]]

    //CHECK:        return [[OUT_1]], [[OUT_2]] : tensor<1x96x1x1xf16, {order = #NHWC}>, tensor<1x96x8x8xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsToNCEClusteringSOHIncompatibleOutputOverlappedStart
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x65x65xf16, {order = #NHWC}>
func.func @ChainOpsToNCEClusteringSOHIncompatibleOutputOverlappedStart(%arg0: tensor<1x128x65x65xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} : tensor<1x128x65x65xf16, {order = #NHWC}>, tensor<96x128x3x3xf16, {order = #NHWC}> -> tensor<1x96x65x65xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 3, 3], strides = [2, 2]} : tensor<1x96x65x65xf16, {order = #NHWC}>, tensor<96x96x3x3xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>
    return %1 : tensor<1x96x32x32xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS_0:%.+]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x65x65xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 10, 65]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 12, 65], [1, 128, 13, 65], [1, 128, 13, 65], [1, 128, 13, 65], [1, 128, 13, 65], [1, 128, 11, 65]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 21, 0], [0, 0, 32, 0], [0, 0, 43, 0], [0, 0, 54, 0]]

    //CHECK:        [[WEIGHTS_0_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_0_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x65x65xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 10, 65]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 13, 65], [1, 96, 14, 65], [1, 96, 13, 65], [1, 96, 12, 65], [1, 96, 11, 65], [1, 96, 11, 65]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]]

    // Requirements for this consumer only, w/o producer compute view:
    //  memory_shapes = [[1, 96, 13, 65], [1, 96, 13, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65]]
    //  memory_offsets = [[0, 0, 0, 0], [0, 0, 12, 0], [0, 0, 24, 0], [0, 0, 34, 0], [0, 0, 44, 0], [0, 0, 54, 0]]

    //CHECK:        [[OUT_0_COPYBACK:%.+]] = VPU.Copy([[OUT_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x65x65xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 10, 65]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 13, 65], [1, 96, 14, 65], [1, 96, 13, 65], [1, 96, 12, 65], [1, 96, 11, 65], [1, 96, 11, 65]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]

    //CHECK:        [[WEIGHTS_1_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK]],
    //CHECK-SAME:             [[WEIGHTS_1_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x32x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 6, 32], [1, 96, 6, 32], [1, 96, 5, 32], [1, 96, 5, 32], [1, 96, 5, 32], [1, 96, 5, 32]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 6, 32], [1, 96, 6, 32], [1, 96, 5, 32], [1, 96, 5, 32], [1, 96, 5, 32], [1, 96, 5, 32]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]

    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]]

    //CHECK:        return [[OUT_1]] : tensor<1x96x32x32xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsToNCEClusteringSOHIncompatibleOutputOverlappedEnd
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x65x65xf16, {order = #NHWC}>
func.func @ChainOpsToNCEClusteringSOHIncompatibleOutputOverlappedEnd(%arg0: tensor<1x128x65x65xf16, {order = #NHWC}>) -> tensor<1x96x20x20xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x7x7xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} : tensor<1x128x65x65xf16, {order = #NHWC}>, tensor<96x128x3x3xf16, {order = #NHWC}> -> tensor<1x96x65x65xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 7, 7], strides = [3, 3]} : tensor<1x96x65x65xf16, {order = #NHWC}>, tensor<96x96x7x7xf16, {order = #NHWC}> -> tensor<1x96x20x20xf16, {order = #NHWC}>
    return %1 : tensor<1x96x20x20xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS_0:%.+]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<96x96x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x7x7xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x65x65xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 10, 65]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 12, 65], [1, 128, 13, 65], [1, 128, 13, 65], [1, 128, 13, 65], [1, 128, 13, 65], [1, 128, 11, 65]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 21, 0], [0, 0, 32, 0], [0, 0, 43, 0], [0, 0, 54, 0]]

    //CHECK:        [[WEIGHTS_0_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_0_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x65x65xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 10, 65]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 15, 65], [1, 96, 16, 65], [1, 96, 14, 65], [1, 96, 13, 65], [1, 96, 14, 65], [1, 96, 15, 65]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 41, 0], [0, 0, 50, 0]]

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]]

    // Requirements for this consumer only, w/o producer compute view:
    //  memory_shapes = [[1, 96, 15, 65], [1, 96, 16, 65], [1, 96, 13, 65], [1, 96, 13, 65], [1, 96, 13, 65], [1, 96, 13, 65]]
    //  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 23, 0], [0, 0, 32, 0], [0, 0, 41, 0], [0, 0, 50, 0]]

    //CHECK:        [[OUT_0_COPYBACK:%.+]] = VPU.Copy([[OUT_0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x65x65xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 10, 65]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 15, 65], [1, 96, 16, 65], [1, 96, 14, 65], [1, 96, 13, 65], [1, 96, 14, 65], [1, 96, 15, 65]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 41, 0], [0, 0, 50, 0]]

    //CHECK:        [[WEIGHTS_1_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x7x7xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[OUT_0_COPYBACK]],
    //CHECK-SAME:             [[WEIGHTS_1_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x20x20xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 4, 20], [1, 96, 4, 20], [1, 96, 3, 20], [1, 96, 3, 20], [1, 96, 3, 20], [1, 96, 3, 20]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 11, 0], [0, 0, 14, 0], [0, 0, 17, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 4, 20], [1, 96, 4, 20], [1, 96, 3, 20], [1, 96, 3, 20], [1, 96, 3, 20], [1, 96, 3, 20]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 11, 0], [0, 0, 14, 0], [0, 0, 17, 0]]

    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]]

    //CHECK:        return [[OUT_1]] : tensor<1x96x20x20xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ProducerConvType = tensor<1x16x28x28xf16, {order = #NHWC}>
!ConcatOutputType = tensor<1x48x28x28xf16, {order = #NHWC}>
!ConvConsumerOutput0 = tensor<1x16x26x26xf16, {order = #NHWC}>
!ConvConsumerOutput1 = tensor<1x16x27x27xf16, {order = #NHWC}>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConcatWithOverlappedInputsNCEConsumersMemViewUnion
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x28x28xf16, {order = #NHWC}>
func.func @ConcatWithOverlappedInputsNCEConsumersMemViewUnion(%arg0: !ProducerConvType) -> (!ConcatOutputType, !ConvConsumerOutput0, !ConvConsumerOutput1) {
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} : !ProducerConvType, tensor<16x16x3x3xf16, {order = #NHWC}> -> !ProducerConvType
    %1 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} : !ProducerConvType, tensor<16x16x3x3xf16, {order = #NHWC}> -> !ProducerConvType
    %2 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} : !ProducerConvType, tensor<16x16x3x3xf16, {order = #NHWC}> -> !ProducerConvType

    %3 = VPU.Concat(%0, %1, %2) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]} : !ProducerConvType, !ProducerConvType, !ProducerConvType -> !ConcatOutputType

    %4 = VPU.NCE.Convolution(%0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} : !ProducerConvType, tensor<16x16x3x3xf16, {order = #NHWC}> -> !ConvConsumerOutput0

    %5 = VPU.NCE.Convolution(%2, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 2 : i64, right = 0 : i64, top = 2 : i64, bottom = 0 : i64>, rawFilterShape = [16, 16, 4, 4], strides = [1, 1]} : !ProducerConvType, tensor<16x16x4x4xf16, {order = #NHWC}> -> !ConvConsumerOutput1

    return %3, %4, %5 : !ConcatOutputType, !ConvConsumerOutput0, !ConvConsumerOutput1

    //CHECK:    [[WEIGHTS_0:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_1:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_2:%.+]] = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}>

    //CONV 0

    //CHECK:           [[INPUT_CMX_0:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 6, 28], [1, 16, 5, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    //CHECK:           [[WEIGHTS_0_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:             [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:                  [[INPUT_CMX_0]],
    //CHECK-SAME:                  [[WEIGHTS_0_CMX]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 7, 28], [1, 16, 9, 28], [1, 16, 8, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]

    //CHECK:           [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]]

    // CONV 1

    //CHECK:           [[INPUT_CMX_1:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 6, 28], [1, 16, 5, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    //CHECK:           [[WEIGHTS_1_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:           [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:                [[INPUT_CMX_1]],
    //CHECK-SAME:                [[WEIGHTS_1_CMX]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:        mode = "OVERLAPPED"
    //CHECK-SAME:        num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:        num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 7, 28], [1, 16, 9, 28], [1, 16, 8, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]


    //CHECK:           [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]]

    // CONV 2

    //CHECK:           [[INPUT_CMX_2:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 6, 28], [1, 16, 5, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    //CHECK:           [[WEIGHTS_2_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:             [[OUT_2_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:                  [[INPUT_CMX_2]],
    //CHECK-SAME:                  [[WEIGHTS_2_CMX]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 7, 28], [1, 16, 9, 28], [1, 16, 8, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]


    //CHECK:           [[OUT_2:%.+]] = VPU.Copy([[OUT_2_CMX]]


    //CHECK:           [[CONCAT:%.+]] = VPU.Concat([[OUT_0]], [[OUT_1]], [[OUT_2]])
    //CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]
    //CHECK-SAME:               tensor<1x16x28x28xf16, {order = #NHWC}>, tensor<1x16x28x28xf16, {order = #NHWC}>, tensor<1x16x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:              -> tensor<1x48x28x28xf16, {order = #NHWC}>

    //CONV 3

    //CHECK:           [[INPUT_CMX_3:%.+]] = VPU.Copy([[OUT_0]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 7, 28], [1, 16, 9, 28], [1, 16, 8, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]

    //CHECK:           [[WEIGHTS_3_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:             [[OUT_3_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:                  [[INPUT_CMX_3]],
    //CHECK-SAME:                  [[WEIGHTS_3_CMX]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x26x26xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 26], [1, 16, 5, 26], [1, 16, 4, 26], [1, 16, 4, 26], [1, 16, 4, 26], [1, 16, 4, 26]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 5, 26], [1, 16, 5, 26], [1, 16, 4, 26], [1, 16, 4, 26], [1, 16, 4, 26], [1, 16, 4, 26]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]

    //CONV 4

    //CHECK:           [[INPUT_CMX_4:%.+]] = VPU.Copy([[OUT_2]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 7, 28], [1, 16, 9, 28], [1, 16, 8, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]

    //CHECK:           [[WEIGHTS_4_CMX:%.+]] = VPU.Copy([[WEIGHTS_2]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x4x4xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:             [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:                  [[INPUT_CMX_4]],
    //CHECK-SAME:                  [[WEIGHTS_4_CMX]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x27x27xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 27], [1, 16, 5, 27], [1, 16, 5, 27], [1, 16, 4, 27], [1, 16, 4, 27], [1, 16, 4, 27]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 5, 27], [1, 16, 5, 27], [1, 16, 5, 27], [1, 16, 4, 27], [1, 16, 4, 27], [1, 16, 4, 27]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ProducerConvType = tensor<1x16x32x32xf16, {order = #NHWC}>
!ConcatOutputType = tensor<1x48x32x32xf16, {order = #NHWC}>
!ConvConsumerOutput0 = tensor<1x16x32x32xf16, {order = #NHWC}>
!ConvConsumerOutput1 = tensor<1x16x32x32xf16, {order = #NHWC}>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConcatWithOverlappedInputsCompatibleNCEConsumers
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x32x32xf16, {order = #NHWC}>
func.func @ConcatWithOverlappedInputsCompatibleNCEConsumers(%arg0: !ProducerConvType) -> (!ConcatOutputType, !ConvConsumerOutput0, !ConvConsumerOutput1) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1]} : !ProducerConvType, tensor<16x16x1x1xf16, {order = #NHWC}> -> !ProducerConvType
    %1 = VPU.NCE.Convolution(%arg0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} : !ProducerConvType, tensor<16x16x3x3xf16, {order = #NHWC}> -> !ProducerConvType
    %2 = VPU.NCE.Convolution(%arg0, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [16, 16, 5, 5], strides = [1, 1]} : !ProducerConvType, tensor<16x16x5x5xf16, {order = #NHWC}> -> !ProducerConvType

    %3 = VPU.Concat(%0, %1, %2) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]} : !ProducerConvType, !ProducerConvType, !ProducerConvType -> !ConcatOutputType

    %4 = VPU.NCE.Convolution(%0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} : !ProducerConvType, tensor<16x16x3x3xf16, {order = #NHWC}> -> !ConvConsumerOutput0

    %5 = VPU.NCE.Convolution(%2, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [16, 16, 5, 5], strides = [1, 1]} : !ProducerConvType, tensor<16x16x5x5xf16, {order = #NHWC}> -> !ConvConsumerOutput1

    return %3, %4, %5 : !ConcatOutputType, !ConvConsumerOutput0, !ConvConsumerOutput1


    //CHECK:    [[WEIGHTS_0:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_1:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_2:%.+]] = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}>

    //CONV 0

    //CHECK:            [[INPUT_CMX_0:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "OVERLAPPED"
    //CHECK-SAME:           num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]

    //CHECK:            [[WEIGHTS_0_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:         -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "DUPLICATED"
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:              [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:                   [[INPUT_CMX_0]],
    //CHECK-SAME:                   [[WEIGHTS_0_CMX]])
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "OVERLAPPED"
    //CHECK-SAME:           num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]]

    // CONV 1

    //CHECK:            [[INPUT_CMX_1:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "OVERLAPPED"
    //CHECK-SAME:           num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]

    //CHECK:            [[WEIGHTS_1_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    //CHECK-SAME:         -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "DUPLICATED"
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:              [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:                   [[INPUT_CMX_1]],
    //CHECK-SAME:                   [[WEIGHTS_1_CMX]])
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "OVERLAPPED"
    //CHECK-SAME:           num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]


    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]]

    // CONV 2

    //CHECK:            [[INPUT_CMX_2:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "OVERLAPPED"
    //CHECK-SAME:           num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]

    //CHECK:            [[WEIGHTS_2_CMX:%.+]] = VPU.Copy([[WEIGHTS_2]]
    //CHECK-SAME:         -> !VPU.DistributedTensor<16x16x5x5xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "DUPLICATED"
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:              [[OUT_2_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:                   [[INPUT_CMX_2]],
    //CHECK-SAME:                   [[WEIGHTS_2_CMX]])
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "OVERLAPPED"
    //CHECK-SAME:           num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]


    //CHECK:        [[OUT_2:%.+]] = VPU.Copy([[OUT_2_CMX]]


    //CHECK:                VPU.Concat([[OUT_0]], [[OUT_1]], [[OUT_2]])
    //CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]
    //CHECK-SAME:               tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK-SAME:              -> tensor<1x48x32x32xf16, {order = #NHWC}>

    //CONV 3

    //CHECK:           [[INPUT_CMX_3:%.+]] = VPU.Copy([[OUT_0]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]

    //CHECK:           [[WEIGHTS_3_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:             [[OUT_3_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:                  [[INPUT_CMX_3]],
    //CHECK-SAME:                  [[WEIGHTS_3_CMX]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]

    //CONV 4

    //CHECK:           [[INPUT_CMX_4:%.+]] = VPU.Copy([[OUT_2]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]

    //CHECK:           [[WEIGHTS_4_CMX:%.+]] = VPU.Copy([[WEIGHTS_2]]
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x5x5xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:             [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:                  [[INPUT_CMX_4]],
    //CHECK-SAME:                  [[WEIGHTS_4_CMX]])
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @NCEInterpolateToDistributedOpClustering
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>
func.func @NCEInterpolateToDistributedOpClustering(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x2x2xi1> = dense<1> : tensor<1x16x2x2xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [16], dataShape = [1, 16, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>
    } -> tensor<1x1x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x2x2xi1>,
                           storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x2x2xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.+]] = const.Declare tensor<1x16x2x2xi1> = dense<true> : tensor<1x16x2x2xi1>
    // CHECK:        [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 1, 1],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [16]}
    // CHECK-SAME:       -> tensor<1x1x2x2xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_SPARSE]]
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x16x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x16x2x2xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x2x2xi32, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                      scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>

    // CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Interpolate(
    // CHECK-SAME:             [[INPUT_CMX]],
    // CHECK-SAME:             [[WEIGHTS_CMX]],
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x2x2xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:        return [[OUT]] : tensor<1x16x2x2xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @NCEInterpolateToDistributedOpSOK
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x5x10xf16, {order = #NHWC}>
func.func @NCEInterpolateToDistributedOpSOK(%arg0: tensor<1x64x5x10xf16, {order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x10x20xi1, {order = #NHWC}> = dense<1> : tensor<1x64x10x20xi1, {order = #NHWC}>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [64], dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>
    } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } -> !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x10x20xi1, {order = #NHWC}>,
                           storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [64, 64, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x10x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x10x20xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.+]] = const.Declare tensor<1x64x10x20xi1, {order = #NHWC}> = dense<true> : tensor<1x64x10x20xi1, {order = #NHWC}>
    // CHECK:        [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 5, 10],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [64]}
    // CHECK-SAME:       -> tensor<1x1x10x20xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_SPARSE]]
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x64x5x10xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 64, 5, 10], [1, 64, 5, 10], [1, 64, 5, 10], [1, 64, 5, 10]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 64, 5, 10], [1, 64, 5, 10], [1, 64, 5, 10], [1, 64, 5, 10]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x64x10x20xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 64, 10, 20], [1, 64, 10, 20], [1, 64, 10, 20], [1, 64, 10, 20]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 64, 10, 20], [1, 64, 10, 20], [1, 64, 10, 20], [1, 64, 10, 20]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x10x20xi32, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 10, 20], [1, 1, 10, 20], [1, 1, 10, 20], [1, 1, 10, 20]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 10, 20], [1, 1, 10, 20], [1, 1, 10, 20], [1, 1, 10, 20]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                     scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>

    // CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<64x64x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 64, 1, 1], [16, 64, 1, 1], [16, 64, 1, 1], [16, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 64, 1, 1], [16, 64, 1, 1], [16, 64, 1, 1], [16, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]

    // CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    // CHECK-SAME:     -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]

    // CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Interpolate(
    // CHECK-SAME:             [[INPUT_CMX]],
    // CHECK-SAME:             [[WEIGHTS_CMX]],
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x64x10x20xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 10, 20], [1, 16, 10, 20], [1, 16, 10, 20], [1, 16, 10, 20]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 64, 10, 20], [1, 64, 10, 20], [1, 64, 10, 20], [1, 64, 10, 20]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:        return [[OUT]] : tensor<1x64x10x20xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 2 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @BilinearNCEInterpolateToDistributedOpSOH
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x5x5xf16, {order = #NHWC}>
func.func @BilinearNCEInterpolateToDistributedOpSOH(%arg0: tensor<1x16x5x5xf16, {order = #NHWC}>) -> tensor<1x16x10x10xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x12x12xi1> = dense<1> : tensor<1x16x12x12xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [16], dataShape = [1, 16, 5, 5],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 12, 12]>
    } -> tensor<1x1x12x12xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 12, 12]>
    } -> !VPU.SparseTensor<data=tensor<1x16x5x5xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x12x12xi1>,
                           storage_element_table=tensor<1x1x12x12xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 12, 12]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>,
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        scales_attr = [1.0, 1.0, 2.0, 2.0]
    } -> tensor<1x16x10x10xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x10x10xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.+]] = const.Declare tensor<1x16x12x12xi1> = dense<true> : tensor<1x16x12x12xi1>
    // CHECK:        [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 5, 5],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 12, 12]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [16]}
    // CHECK-SAME:       -> tensor<1x1x12x12xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_SPARSE]])
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x16x5x5xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 3, 5], [1, 16, 3, 5]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes =  [[1, 16, 3, 5], [1, 16, 3, 5]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x16x12x12xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 6, 12], [1, 16, 6, 12]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes =  [[1, 16, 7, 12], [1, 16, 7, 12]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]]
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x12x12xi32, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 6, 12], [1, 1, 6, 12]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 7, 12], [1, 1, 7, 12]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]]
    // CHECK-SAME:               #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                     scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 12, 12]>>

    // CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Interpolate(
    // CHECK-SAME:             [[INPUT_CMX]],
    // CHECK-SAME:             [[WEIGHTS_CMX]],
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x16x10x10xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 10], [1, 16, 5, 10]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 5, 10], [1, 16, 5, 10]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]]

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:        return [[OUT]] : tensor<1x16x10x10xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 2 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @BilinearNCEInterpolateToDistributedOpSOHWithTiling
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x42x320xf16, {order = #NHWC}>)
func.func @BilinearNCEInterpolateToDistributedOpSOHWithTiling(%arg0: tensor<1x16x42x320xf16, {order = #NHWC}>) -> tensor<1x16x80x320xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x3x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x3x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x82x320xi1> = dense<1> : tensor<1x16x82x320xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [16], dataShape = [1, 16, 42, 320],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                    scale = [1.0, 1.0, 2.0, 1.0], nearest_mode = <FLOOR>,
                                    offsets = [0, 0, 2, 0], sizes = [1, 16, 82, 320],
                                    initial_input_shape = [1, 16, 160, 320], initial_output_shape = [1, 16, 320, 320]>
    } -> tensor<1x1x82x320xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 2.0, 1.0], nearest_mode = <FLOOR>,
            offsets = [0, 0, 2, 0], sizes = [1, 16, 82, 320],
            initial_input_shape = [1, 16, 160, 320], initial_output_shape = [1, 16, 320, 320]>
    } -> !VPU.SparseTensor<data=tensor<1x16x42x320xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x82x320xi1>,
                           storage_element_table=tensor<1x1x82x320xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                              scale = [1.0, 1.0, 2.0, 1.0], nearest_mode = <FLOOR>,
                                              offsets = [0, 0, 2, 0], sizes = [1, 16, 82, 320],
                                              initial_input_shape = [1, 16, 160, 320], initial_output_shape = [1, 16, 320, 320]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 3, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        ppe = #VPU.PPEStub<>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        scales_attr = [1.0, 1.0, 2.0, 1.0]
    } -> tensor<1x16x80x320xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x80x320xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.+]] = const.Declare tensor<1x16x82x320xi1> = dense<true> : tensor<1x16x82x320xi1>
    // CHECK:        [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 42, 320],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00], nearest_mode = <FLOOR>,
    // CHECK-SAME:                                   offsets = [0, 0, 2, 0], sizes = [1, 16, 82, 320], initial_input_shape = [1, 16, 160, 320], initial_output_shape = [1, 16, 320, 320]>,
    // CHECK-SAME:                                   seDepth = 1 : i64, seSize = [16]}
    // CHECK-SAME:       -> tensor<1x1x82x320xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_SPARSE]])
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x16x42x320xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 21, 320], [1, 16, 21, 320]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 21, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes =  [[1, 16, 22, 320], [1, 16, 22, 320]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x16x82x320xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 41, 320], [1, 16, 41, 320]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 41, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes =  [[1, 16, 42, 320], [1, 16, 42, 320]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 40, 0]]
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x82x320xi32, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 41, 320], [1, 1, 41, 320]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 41, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 42, 320], [1, 1, 42, 320]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 40, 0]]
    // CHECK-SAME:               #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                     scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00], nearest_mode = <FLOOR>,
    // CHECK-SAME:                     offsets = [0, 0, 2, 0], sizes = [1, 16, 82, 320],
    // CHECK-SAME:                     initial_input_shape = [1, 16, 160, 320], initial_output_shape = [1, 16, 320, 320]>>

    // CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x16x3x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 3, 1], [16, 16, 3, 1]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 3, 1], [16, 16, 3, 1]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Interpolate(
    // CHECK-SAME:             [[INPUT_CMX]]
    // CHECK-SAME:             [[WEIGHTS_CMX]]
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]]
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x16x80x320xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 40, 320], [1, 16, 40, 320]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 40, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 40, 320], [1, 16, 40, 320]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 40, 0]]

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:        return [[OUT]] : tensor<1x16x80x320xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 2 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @BilinearNCEInterpolateToDistributedOpSOK
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x5x5xf16, {order = #NHWC}>
func.func @BilinearNCEInterpolateToDistributedOpSOK(%arg0: tensor<1x32x5x5xf16, {order = #NHWC}>) -> tensor<1x32x10x10xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x4x4xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x32x22x22xi1> = dense<1> : tensor<1x32x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [32], dataShape = [1, 32, 5, 5],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x32x5x5xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x32x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [32, 32, 4, 4],
        strides = [2, 2],
        ppe = #VPU.PPEStub<>,
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        scales_attr = [1.0, 1.0, 2.0, 2.0]
    } -> tensor<1x32x10x10xf16, {order = #NHWC}>

    return %interpolate : tensor<1x32x10x10xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x32x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.+]] = const.Declare tensor<1x32x22x22xi1> = dense<true> : tensor<1x32x22x22xi1>
    // CHECK:        [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 5, 5],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [32]}
    // CHECK-SAME:       -> tensor<1x1x22x22xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_SPARSE]]
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x32x5x5xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 32, 5, 5], [1, 32, 5, 5]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes =  [[1, 32, 5, 5], [1, 32, 5, 5]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x32x22x22xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 32, 22, 22], [1, 32, 22, 22]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes =  [[1, 32, 22, 22], [1, 32, 22, 22]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x22x22xi32, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 22, 22], [1, 1, 22, 22]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 22, 22], [1, 1, 22, 22]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                     scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>>

    // CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x32x4x4xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 32, 4, 4], [16, 32, 4, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 32, 4, 4], [16, 32, 4, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0]]

    // CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0]]

    // CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Interpolate(
    // CHECK-SAME:             [[INPUT_CMX]],
    // CHECK-SAME:             [[WEIGHTS_CMX]],
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]])
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x32x10x10xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 10, 10], [1, 16, 10, 10]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 10, 10], [1, 32, 10, 10]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:        return [[OUT]] : tensor<1x32x10x10xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 2 of @NCE at 1.700000e+03 MHz

// Different memory offsets and shapes are generated for the output of the Convolution and the input of the Interpolate,
// which would force a spill to be preserved between them

// CHECK:       func.func @OverlappedConvToOverlappedSEPOp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x32x60x60xf16, {order = #NHWC}>
func.func @OverlappedConvToOverlappedSEPOp(%input: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x32x60x60xf16, {order = #NHWC}> {
    %conv_weights = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %conv = VPU.NCE.Convolution(%input, %conv_weights) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 16, 3, 3],
        strides = [1, 1]}
      : tensor<1x16x30x30xf16, {order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}> -> tensor<1x32x30x30xf16, {order = #NHWC}>

    %input_sparsity_map = const.Declare tensor<1x32x62x62xi1> = dense<1> : tensor<1x32x62x62xi1>
    %input_storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [32], dataShape = [1, 32, 30, 30],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 62, 62]>
    } -> tensor<1x1x62x62xi32, {order = #NHWC}>
    %input_sparse = VPU.GroupSparseTensor(%conv, %input_sparsity_map, %input_storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 62, 62]>
    } -> !VPU.SparseTensor<data=tensor<1x32x30x30xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x32x62x62xi1>,
                           storage_element_table=tensor<1x1x62x62xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 62, 62]>>

    %weights_interp = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table_interp = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %interpolate = VPU.NCE.Interpolate(%input_sparse, %weights_interp, %weights_table_interp) {
        rawFilterShape = [32, 32, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>,
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        scales_attr = [1.0, 1.0, 2.0, 2.0]
    } -> tensor<1x32x60x60xf16, {order = #NHWC}>

    return %interpolate : tensor<1x32x60x60xf16, {order = #NHWC}>

    // CHECK-DAG:    [[CONV_WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:        [[CONV_INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x30x30xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 16, 15, 30], [1, 16, 15, 30]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 16, 16, 30], [1, 16, 16, 30]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]}

    // CHECK:        [[CONV_WEIGHTS_CMX:%.+]] = VPU.Copy([[CONV_WEIGHTS]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x16x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[32, 16, 3, 3], [32, 16, 3, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[32, 16, 3, 3], [32, 16, 3, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}

    // CHECK:        [[CONV_CMX:%.+]] = VPU.NCE.Convolution(
    // CHECK-SAME:          [[CONV_INPUT_CMX]],
    // CHECK-SAME:          [[CONV_WEIGHTS_CMX]])
    // CHECK-SAME:        -> !VPU.DistributedTensor<1x32x30x30xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 15, 30], [1, 32, 15, 30]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0]],
    // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 15, 30], [1, 32, 15, 30]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0]]}

    // CHECK:        [[CONV_DDR:%.+]] = VPU.Copy([[CONV_CMX]]
    // CHECK-SAME:      -> tensor<1x32x30x30xf16, {order = #NHWC}>

    // CHECK-DAG:    [[INTERP_INPUT_SM:%.+]] = const.Declare tensor<1x32x62x62xi1> = dense<true> : tensor<1x32x62x62xi1>
    // CHECK:        [[INTERP_INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 30, 30],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 62, 62]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [32]}
    // CHECK-SAME:       -> tensor<1x1x62x62xi32, {order = #NHWC}>
    // CHECK:        [[INTERP_INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[CONV_DDR]], [[INTERP_INPUT_SM]], [[INTERP_INPUT_SE]])

    // CHECK-DAG:    [[INTERP_WEIGHTS:%.+]] = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[INTERP_WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    // CHECK:        [[INTER_INPUT_CMX:%.+]] = VPU.Copy([[INTERP_INPUT_SPARSE]]
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x32x30x30xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 32, 15, 30], [1, 32, 15, 30]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 32, 16, 30], [1, 32, 16, 30]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]}
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x32x62x62xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 32, 31, 62], [1, 32, 31, 62]], compute_offsets = [[0, 0, 0, 0], [0, 0, 31, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 32, 32, 62], [1, 32, 32, 62]], memory_offsets = [[0, 0, 0, 0], [0, 0, 30, 0]]}
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x62x62xi32, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 31, 62], [1, 1, 31, 62]], compute_offsets = [[0, 0, 0, 0], [0, 0, 31, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 32, 62], [1, 1, 32, 62]], memory_offsets = [[0, 0, 0, 0], [0, 0, 30, 0]]}
    // CHECK-SAME:               #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                     scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 62, 62]>>

    // CHECK:        [[INTERP_WEIGHTS_CMX:%.+]] = VPU.Copy([[INTERP_WEIGHTS]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x32x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 32, 3, 3], [32, 32, 3, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 32, 3, 3], [32, 32, 3, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}

    // CHECK:        [[INTERP_WEIGHTS_TABLE_CMX:%.+]] = VPU.Copy([[INTERP_WEIGHTS_TABLE]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}

    // CHECK:        [[INTERP_CMX:%.+]] = VPU.NCE.Interpolate(
    // CHECK-SAME:             [[INTER_INPUT_CMX]],
    // CHECK-SAME:             [[INTERP_WEIGHTS_CMX]],
    // CHECK-SAME:             [[INTERP_WEIGHTS_TABLE_CMX]]
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x32x60x60xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 30, 60], [1, 32, 30, 60]], compute_offsets = [[0, 0, 0, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 30, 60], [1, 32, 30, 60]], memory_offsets = [[0, 0, 0, 0], [0, 0, 30, 0]]}

    // CHECK:        [[INTERP_DDR:%.+]] = VPU.Copy([[INTERP_CMX]]

    // CHECK:        return [[INTERP_DDR]] : tensor<1x32x60x60xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 2 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SEPadToDistributedOpSOH
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x40x40xf16, {order = #NHWC}>
func.func @SEPadToDistributedOpSOH(%arg0: tensor<1x16x40x40xf16, {order = #NHWC}>) -> tensor<1x32x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %sparsity_map = const.Declare tensor<1x16x42x42xi1, {order = #NHWC}> = dense<1> : tensor<1x16x42x42xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]

    %storage_element = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 40, 40],
            seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>, seDepth = 1 : i64, seSize = [16]
        } -> tensor<1x1x42x42xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
            seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>
        } -> !VPU.SparseTensor<data=tensor<1x16x40x40xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x16x42x42xi1, {order = #NHWC}>,
                               storage_element_table=tensor<1x1x42x42xi32, {order = #NHWC}>,
                               #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>>

    %conv = VPU.NCE.Convolution(%input, %weights) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [32, 16, 3, 3], strides = [2, 2]
        } : !VPU.SparseTensor<data=tensor<1x16x40x40xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x42x42xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x42x42xi32, {order = #NHWC}>, #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>>, tensor<32x16x3x3xf16, {order = #NHWC}> -> tensor<1x32x20x20xf16, {order = #NHWC}>

    return %conv : tensor<1x32x20x20xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[INPUT_SM:%.+]] = const.Declare tensor<1x16x42x42xi1, {order = #NHWC}> = dense<1> : tensor<1x16x42x42xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:        [[INPUT_SE:%.+]] = VPU.StorageElementTable {
    // CHECK-SAME:                              dataElemType = f16, dataShape = [1, 16, 40, 40],
    // CHECK-SAME:                              seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>,
    // CHECK-SAME:                              seDepth = 1 : i64, seSize = [16]} -> tensor<1x1x42x42xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT_SPARSE]]
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x16x40x40xf16, #NHWC, @CMX_NN
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 20, 40], [1, 16, 20, 40]]
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 16, 20, 40], [1, 16, 21, 40]]
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]]
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x16x42x42xi1, #NHWC, @CMX_NN
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 21, 42], [1, 16, 21, 42]]
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 21, 0]]
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 16, 21, 42], [1, 16, 21, 42]]
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x42x42xi32, #NHWC, @CMX_NN
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 21, 42], [1, 1, 21, 42]]
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 21, 0]]
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 21, 42], [1, 1, 21, 42]]
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME:               #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>>

    // CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x16x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 16, 3, 3], [32, 16, 3, 3]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 16, 3, 3], [32, 16, 3, 3]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    // CHECK-SAME:             [[INPUT_CMX]]
    // CHECK-SAME:             [[WEIGHTS_CMX]])
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x32x20x20xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 10, 20], [1, 32, 10, 20]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 10, 20], [1, 32, 10, 20]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]

    // CHECK:       [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:        return [[OUT]] : tensor<1x32x20x20xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SliceConvConcatGeluSOK
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x80x1x3008xf16, {order = #NHWC}>
func.func @SliceConvConcatGeluSOK(%arg0: tensor<1x80x1x3008xf16, {order = #NHWC}>) -> tensor<1x512x1x3000xf16, {order = #NHWC}> {
    %weights_0 = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]
    %weights_1 = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 80, 1, 3000] : tensor<1x80x1x3008xf16, {order = #NHWC}> to tensor<1x80x1x3000xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %weights_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 80, 1, 3], strides = [1, 1]} : tensor<1x80x1x3000xf16, {order = #NHWC}>, tensor<256x80x1x3xf16, {order = #NHWC}> -> tensor<1x256x1x3000xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %weights_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 80, 1, 3], strides = [1, 1]} : tensor<1x80x1x3000xf16, {order = #NHWC}>, tensor<256x80x1x3xf16, {order = #NHWC}> -> tensor<1x256x1x3000xf16, {order = #NHWC}>

    %3 = VPU.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x1x3000xf16, {order = #NHWC}>, tensor<1x256x1x3000xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>

    %4 = VPU.Slice %3 [0, 0, 0, 0] [1, 512, 1, 1500] : tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>
    %5 = VPU.Gelu(%4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x1500xf16, {order = #NHWC}>

    %6 = VPU.Slice %3 [0, 0, 0, 1500] [1, 512, 1, 1500] : tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>
    %7 = VPU.Gelu(%6) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x1500xf16, {order = #NHWC}>

    %8 = VPU.Concat(%5, %7) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1500]]} : tensor<1x512x1x1500xf16, {order = #NHWC}>, tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>
    return %8 : tensor<1x512x1x3000xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK: [[CONV_INPUT:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 80, 1, 3000] : tensor<1x80x1x3008xf16, {order = #NHWC}> to tensor<1x80x1x3000xf16, {order = #NHWC}>
    // CHECK: [[CONV0_INPUT_CMX:%.+]] = VPU.Copy([[CONV_INPUT]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x80x1x3000xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK: [[CONV0_WEIGHT_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<256x80x1x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [32, 80, 1, 3], [32, 80, 1, 3]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [224, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [32, 80, 1, 3], [32, 80, 1, 3]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [224, 0, 0, 0]]

    // CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[CONV0_INPUT_CMX]],
    // CHECK:               [[CONV0_WEIGHT_CMX]])
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x256x1x3000xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 32, 1, 3000], [1, 32, 1, 3000]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0], [0, 192, 0, 0], [0, 224, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 32, 1, 3000], [1, 32, 1, 3000]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0], [0, 192, 0, 0], [0, 224, 0, 0]]

    // CHECK: [[CONV0_OUTPUT:%.+]] = VPU.Copy([[CONV0]]


    // CHECK: [[CONV1_INPUT_CMX:%.+]] = VPU.Copy([[CONV_INPUT]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x80x1x3000xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK: [[CONV1_WEIGHT_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<256x80x1x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [32, 80, 1, 3], [32, 80, 1, 3]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [224, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [32, 80, 1, 3], [32, 80, 1, 3]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [224, 0, 0, 0]]

    // CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV1_INPUT_CMX]],
    // CHECK:               [[CONV1_WEIGHT_CMX]])
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x256x1x3000xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 32, 1, 3000], [1, 32, 1, 3000]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0], [0, 192, 0, 0], [0, 224, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 32, 1, 3000], [1, 32, 1, 3000]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0], [0, 192, 0, 0], [0, 224, 0, 0]]

    // CHECK: [[CONV1_OUTPUT:%.+]] = VPU.Copy([[CONV1]]

    // CHECK: [[CONV_CONCAT:%.+]] = VPU.Concat([[CONV0_OUTPUT]], [[CONV1_OUTPUT]]) {static_offsets = [
    // CHECK-SAME:    [0, 0, 0, 0], [0, 256, 0, 0]
    // CHECK-SAME:  ]} :
    // CHECK-SAME: tensor<1x256x1x3000xf16, {order = #NHWC}>,
    // CHECK-SAME: tensor<1x256x1x3000xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>

    // CHECK: [[GELU_0_SLICE:%.+]] = VPU.Slice [[CONV_CONCAT]] [0, 0, 0, 0] [1, 512, 1, 1500] :
    // CHECK-SAME:                  tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>

    // CHECK: [[GELU_0_INPUT:%.+]] = VPU.Copy([[GELU_0_SLICE]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN,
    /// CHECK-SAME:             {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]]

    // CHECK: [[GELU_0:%.+]] = VPU.Gelu([[GELU_0_INPUT]])
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN,
    /// CHECK-SAME:             {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]]

    // CHECK: [[GELU_0_OUTPUT:%.+]] = VPU.Copy([[GELU_0]]
    // CHECK-SAME:          -> tensor<1x512x1x1500xf16, {order = #NHWC}>

    // CHECK: [[GELU_1_SLICE:%.+]] = VPU.Slice [[CONV_CONCAT]] [0, 0, 0, 1500] [1, 512, 1, 1500] :
    // CHECK-SAME:                  tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>

    // CHECK: [[GELU_1_INPUT:%.+]] = VPU.Copy([[GELU_1_SLICE]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]]

    // CHECK: [[GELU_1:%.+]] = VPU.Gelu([[GELU_1_INPUT]])
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]]

    // CHECK: [[GELU_1_OUTPUT:%.+]] = VPU.Copy([[GELU_1]]

    // CHECK: [[GELU_CONCAT:%.+]] = VPU.Concat([[GELU_0_OUTPUT]], [[GELU_1_OUTPUT]]) {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 0, 0, 1500]
    // CHECK:  ]} : tensor<1x512x1x1500xf16, {order = #NHWC}>, tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>

    // CHECK: return [[GELU_CONCAT]] : tensor<1x512x1x3000xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ProducerConvType = tensor<1x16x32x32xf16, {order = #NHWC}>
!ConcatOutputType = tensor<1x48x32x32xf16, {order = #NHWC}>
!ConvConsumerOutput0 = tensor<1x16x32x32xf16, {order = #NHWC}>
!ConvConsumerOutput1 = tensor<1x16x32x32xf16, {order = #NHWC}>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @OverlappedThroughConcatWithCompatibleNCEConsumers
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x16x32x32xf16, {order = #NHWC}>)
func.func @OverlappedThroughConcatWithCompatibleNCEConsumers(%arg0: !ProducerConvType) -> (!ConvConsumerOutput0, !ConvConsumerOutput1) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16>, [#const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<16x48x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x48x7x7xf16>, [#const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<16x48x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x48x5x5xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1]} : !ProducerConvType, tensor<16x16x1x1xf16, {order = #NHWC}> -> !ProducerConvType
    %1 = VPU.NCE.Convolution(%arg0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} : !ProducerConvType, tensor<16x16x3x3xf16, {order = #NHWC}> -> !ProducerConvType
    %2 = VPU.NCE.Convolution(%arg0, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [16, 16, 5, 5], strides = [1, 1]} : !ProducerConvType, tensor<16x16x5x5xf16, {order = #NHWC}> -> !ProducerConvType

    %3 = VPU.Concat(%0, %1, %2) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]} : !ProducerConvType, !ProducerConvType, !ProducerConvType -> !ConcatOutputType

    %4 = VPU.NCE.Convolution(%3, %cst_4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>, rawFilterShape = [16, 48, 7, 7], strides = [1, 1]} : !ConcatOutputType, tensor<16x48x7x7xf16, {order = #NHWC}> -> !ConvConsumerOutput0

    %5 = VPU.NCE.Convolution(%3, %cst_5) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [16, 48, 5, 5], strides = [1, 1]} : !ConcatOutputType, tensor<16x48x5x5xf16, {order = #NHWC}> -> !ConvConsumerOutput1

    return %4, %5 : !ConvConsumerOutput0, !ConvConsumerOutput1


    //CHECK:    [[WEIGHTS_0:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_1:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_2:%.+]] = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_3:%.+]] = const.Declare tensor<16x48x7x7xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_4:%.+]] = const.Declare tensor<16x48x5x5xf16, {order = #NHWC}>

    //CONV 0

    //CHECK:        [[INPUT_CMX_0:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]

    //CHECK:        [[WEIGHTS_0_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:      [[INPUT_CMX_0]],
    //CHECK-SAME:      [[WEIGHTS_0_CMX]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 9, 32], [1, 16, 12, 32], [1, 16, 11, 32], [1, 16, 11, 32], [1, 16, 11, 32], [1, 16, 8, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 24, 0]]

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]]

    // CONV 1

    //CHECK:        [[INPUT_CMX_1:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]

    //CHECK:        [[WEIGHTS_1_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX_1]],
    //CHECK-SAME:             [[WEIGHTS_1_CMX]])
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 9, 32], [1, 16, 12, 32], [1, 16, 11, 32], [1, 16, 11, 32], [1, 16, 11, 32], [1, 16, 8, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 24, 0]]


    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]]

    // CONV 2

    //CHECK:        [[INPUT_CMX_2:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]

    //CHECK:        [[WEIGHTS_2_CMX:%.+]] = VPU.Copy([[WEIGHTS_2]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x5x5xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_2_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX_2]],
    //CHECK-SAME:             [[WEIGHTS_2_CMX]])
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 9, 32], [1, 16, 12, 32], [1, 16, 11, 32], [1, 16, 11, 32], [1, 16, 11, 32], [1, 16, 8, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 24, 0]]


    //CHECK:        [[OUT_2:%.+]] = VPU.Copy([[OUT_2_CMX]]


    //CHECK:         [[CONCAT:%.+]] =  VPU.Concat([[OUT_0]], [[OUT_1]], [[OUT_2]])
    //CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]
    //CHECK-SAME:               tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK-SAME:              -> tensor<1x48x32x32xf16, {order = #NHWC}>

    //CONV 3

    //CHECK:        [[INPUT_CMX_3:%.+]] = VPU.Copy([[CONCAT]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 48, 6, 32], [1, 48, 6, 32], [1, 48, 5, 32], [1, 48, 5, 32], [1, 48, 5, 32], [1, 48, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 48, 9, 32], [1, 48, 12, 32], [1, 48, 11, 32], [1, 48, 11, 32], [1, 48, 11, 32], [1, 48, 8, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 24, 0]]

    //CHECK:        [[WEIGHTS_3_CMX:%.+]] = VPU.Copy([[WEIGHTS_3]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x48x7x7xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_3_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX_3]],
    //CHECK-SAME:             [[WEIGHTS_3_CMX]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]

    //CHECK:        [[OUT_3:%.+]] = VPU.Copy([[OUT_3_CMX]]

    //CONV 4

    //CHECK:        [[INPUT_CMX_4:%.+]] = VPU.Copy([[CONCAT]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 48, 6, 32], [1, 48, 6, 32], [1, 48, 5, 32], [1, 48, 5, 32], [1, 48, 5, 32], [1, 48, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 48, 9, 32], [1, 48, 12, 32], [1, 48, 11, 32], [1, 48, 11, 32], [1, 48, 11, 32], [1, 48, 8, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 24, 0]]

    //CHECK:        [[WEIGHTS_4_CMX:%.+]] = VPU.Copy([[WEIGHTS_4]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x48x5x5xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_4_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX_4]],
    //CHECK-SAME:             [[WEIGHTS_4_CMX]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]

    //CHECK:        [[OUT_4:%.+]] = VPU.Copy([[OUT_4_CMX]]

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ProducerConvType = tensor<1x16x32x32xf16, {order = #NHWC}>
!ConcatOutputType = tensor<1x16x96x32xf16, {order = #NHWC}>
!ConvConsumerOutput0 = tensor<1x16x96x32xf16, {order = #NHWC}>
!ConvConsumerOutput1 = tensor<1x16x96x32xf16, {order = #NHWC}>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @IncompatibleConcatOverlappedWithNCEConsumers
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x16x32x32xf16, {order = #NHWC}>)
func.func @IncompatibleConcatOverlappedWithNCEConsumers(%arg0: !ProducerConvType) -> (!ConvConsumerOutput0, !ConvConsumerOutput1) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16>, [#const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<16x16x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x7x7xf16>, [#const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1]} : !ProducerConvType, tensor<16x16x1x1xf16, {order = #NHWC}> -> !ProducerConvType
    %1 = VPU.NCE.Convolution(%arg0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} : !ProducerConvType, tensor<16x16x3x3xf16, {order = #NHWC}> -> !ProducerConvType
    %2 = VPU.NCE.Convolution(%arg0, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [16, 16, 5, 5], strides = [1, 1]} : !ProducerConvType, tensor<16x16x5x5xf16, {order = #NHWC}> -> !ProducerConvType

    %3 = VPU.Concat(%0, %1, %2) {static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0]]} : !ProducerConvType, !ProducerConvType, !ProducerConvType -> !ConcatOutputType

    %4 = VPU.NCE.Convolution(%3, %cst_4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>, rawFilterShape = [16, 16, 7, 7], strides = [1, 1]} : !ConcatOutputType, tensor<16x16x7x7xf16, {order = #NHWC}> -> !ConvConsumerOutput0

    %5 = VPU.NCE.Convolution(%3, %cst_5) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [16, 16, 5, 5], strides = [1, 1]} : !ConcatOutputType, tensor<16x16x5x5xf16, {order = #NHWC}> -> !ConvConsumerOutput1

    return %4, %5 : !ConvConsumerOutput0, !ConvConsumerOutput1


    //CHECK:    [[WEIGHTS_0:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_1:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_2:%.+]] = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_3:%.+]] = const.Declare tensor<16x16x7x7xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_4:%.+]] = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}>

    //CONV 0

    //CHECK:        [[INPUT_CMX_0:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]

    //CHECK:        [[WEIGHTS_0_CMX:%.+]] = VPU.Copy([[WEIGHTS_0]]
    //CHECK-SAME:     -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:      [[INPUT_CMX_0]],
    //CHECK-SAME:      [[WEIGHTS_0_CMX]])
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]

    //CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]]

    // CONV 1

    //CHECK:        [[INPUT_CMX_1:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]

    //CHECK:        [[WEIGHTS_1_CMX:%.+]] = VPU.Copy([[WEIGHTS_1]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_1_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX_1]],
    //CHECK-SAME:             [[WEIGHTS_1_CMX]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]


    //CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]]

    // CONV 2

    //CHECK:        [[INPUT_CMX_2:%.+]] = VPU.Copy([[ARG0]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]

    //CHECK:        [[WEIGHTS_2_CMX:%.+]] = VPU.Copy([[WEIGHTS_2]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x5x5xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_2_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX_2]],
    //CHECK-SAME:             [[WEIGHTS_2_CMX]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]


    //CHECK:        [[OUT_2:%.+]] = VPU.Copy([[OUT_2_CMX]]


    //CHECK:         [[CONCAT:%.+]] =  VPU.Concat([[OUT_0]], [[OUT_1]], [[OUT_2]])
    //CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0]]
    //CHECK-SAME:               tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK-SAME:              -> tensor<1x16x96x32xf16, {order = #NHWC}>

    //CONV 3

    //CHECK:        [[INPUT_CMX_3:%.+]] = VPU.Copy([[CONCAT]]
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x16x96x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0], [0, 0, 64, 0], [0, 0, 80, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 19, 32], [1, 16, 22, 32], [1, 16, 22, 32], [1, 16, 22, 32], [1, 16, 22, 32], [1, 16, 19, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 29, 0], [0, 0, 45, 0], [0, 0, 61, 0], [0, 0, 77, 0]]

    //CHECK:        [[WEIGHTS_3_CMX:%.+]] = VPU.Copy([[WEIGHTS_3]]
    //CHECK-SAME:     -> !VPU.DistributedTensor<16x16x7x7xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_3_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX_3]],
    //CHECK-SAME:             [[WEIGHTS_3_CMX]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x96x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0], [0, 0, 64, 0], [0, 0, 80, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0], [0, 0, 64, 0], [0, 0, 80, 0]]

    //CHECK:        [[OUT_3:%.+]] = VPU.Copy([[OUT_3_CMX]]

    //CONV 4

    //CHECK:        [[INPUT_CMX_4:%.+]] = VPU.Copy([[CONCAT]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x96x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0], [0, 0, 64, 0], [0, 0, 80, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 19, 32], [1, 16, 22, 32], [1, 16, 22, 32], [1, 16, 22, 32], [1, 16, 22, 32], [1, 16, 19, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 29, 0], [0, 0, 45, 0], [0, 0, 61, 0], [0, 0, 77, 0]]

    //CHECK:        [[WEIGHTS_4_CMX:%.+]] = VPU.Copy([[WEIGHTS_4]]
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x5x5xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_4_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX_4]],
    //CHECK-SAME:             [[WEIGHTS_4_CMX]])
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x96x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0], [0, 0, 64, 0], [0, 0, 80, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0], [0, 0, 64, 0], [0, 0, 80, 0]]

    //CHECK:        [[OUT_4:%.+]] = VPU.Copy([[OUT_4_CMX]]
    //CHECK-SAME:       -> tensor<1x16x96x32xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @CompressConvToDistributedOpSOB
// CHECK-SAME:      [[INPUT:%.+]]: tensor<6x4x224x224xf16, {order = #NHWC}>
func.func @CompressConvToDistributedOpSOB(%arg0: tensor<6x4x224x224xf16, {order = #NHWC}>) -> tensor<6x64x112x112xf16, {order = #NHWC}>  {
    %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x1x1x160xf16>, [#const.Reorder<#NHWC>]
    %compressConv = VPU.NCE.CompressConvolution(%arg0, %cst_0, %cst) {
                cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [64, 4, 7, 7], strides = [2, 2]}
        -> tensor<6x64x112x112xf16, {order = #NHWC}>

    return %compressConv : tensor<6x64x112x112xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x1x1x160xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<6x4x224x224xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 4, 224, 224], [1, 4, 224, 224], [1, 4, 224, 224], [1, 4, 224, 224], [1, 4, 224, 224], [1, 4, 224, 224]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 4, 224, 224], [1, 4, 224, 224], [1, 4, 224, 224], [1, 4, 224, 224], [1, 4, 224, 224], [1, 4, 224, 224]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<64x1x1x160xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[WEIGHTSTABLE_CMX:%.+]] = VPU.Copy([[WEIGHTSTABLE]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.CompressConvolution(
    //CHECK-SAME:           [[INPUT_CMX]]
    //CHECK-SAME:           [[WEIGHTS_CMX]],
    //CHECK-SAME:           [[WEIGHTSTABLE_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<6x64x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 112, 112], [1, 64, 112, 112], [1, 64, 112, 112], [1, 64, 112, 112], [1, 64, 112, 112], [1, 64, 112, 112]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 112, 112], [1, 64, 112, 112], [1, 64, 112, 112], [1, 64, 112, 112], [1, 64, 112, 112], [1, 64, 112, 112]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<6x64x112x112xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

module @Permute {
config.Resources 2 of @NCE at 1.300000e+03 MHz

// CHECK-LABEL: @NCEPermuteCompressConv
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x3x224x224xf16>
func.func @NCEPermuteCompressConv(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x16x112x112xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x1x1x48xf16>, [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Reorder<#NHWC>
        ]

    %WEIGHT_TABLE = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    %1 = VPU.NCE.CompressConvolution(%0, %WEIGHTS, %WEIGHT_TABLE) {
        cm_sp_pattern = 7 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [16, 4, 3, 3],
        strides = [2, 2]
    } -> tensor<1x16x112x112xf16, {order = #NHWC}>

    return %1 : tensor<1x16x112x112xf16, {order = #NHWC}>

    // CHECK:       [[COPY_INPUT:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[COPY_INPUT]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 4, 113, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>

    // CHECK:          -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 4, 113, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @UnrollSOKAveragePoolInputDuplicatedOutputSegmented
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x320x1xf16>
func.func @UnrollSOKAveragePoolInputDuplicatedOutputSegmented(%input: tensor<1x1x320x1xf16>) -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    %mvn = VPU.MVN(%input) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true}
            : tensor<1x1x320x1xf16> -> tensor<1x1x320x1xf16>

    %reshape = VPU.AffineReshape(%mvn) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]}
            : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>

    %cast = VPU.PermuteCast(%reshape) {dst_order = #NHWC, mem_perm = #NHWC}
            : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    %averagePool = VPU.NCE.AveragePool(%cast) {kernel_size = [1, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x320x1x1xf16, {order = #NHWC}>

    %activation = VPU.Sigmoid(%averagePool) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
            : tensor<1x320x1x1xf16, {order = #NHWC}> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    return %activation : tensor<1x320x1x1xf16, {order = #NHWC}>

    // (DUP) MVN (DUP) -> (DUP) AveragePool (SEG) -> (SEG) Sigmoid

    //CHECK:        [[MVN_COPY_IN:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[MVN:%.+]] = VPU.MVN([[MVN_COPY_IN]]
    //CHECK-SAME:       !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[MVN_COPY_OUT:%.+]] = VPU.Copy([[MVN]]
    //CHECK-SAME:       -> tensor<1x1x320x1xf16>

    //CHECK:        [[RESHAPE:%.+]] = VPU.AffineReshape([[MVN_COPY_OUT]])
    //CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]} : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>

    //CHECK:        [[CAST:%.+]] = VPU.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    //CHECK:        [[AVERAGEPOOL_INPUT_COPY_IN:%.+]] = VPU.Copy([[CAST]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[AVERAGEPOOL:%.+]]  = VPU.NCE.AveragePool([[AVERAGEPOOL_INPUT_COPY_IN]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>

    //CHECK:        [[AVERAGEPOOL_COPY_OUT:%.+]] = VPU.Copy([[AVERAGEPOOL]]
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}>

    //CHECK:        [[SIGMOID_COPY_IN:%.+]] = VPU.Copy([[AVERAGEPOOL_COPY_OUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>

    //CHECK:        [[SIGMOID:%.+]] = VPU.Sigmoid([[SIGMOID_COPY_IN]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>

    //CHECK:        [[SIGMOID_COPY_OUT:%.+]] = VPU.Copy([[SIGMOID]]
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NCEPermute
module @NCEPermute {

config.Resources 2 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @NCEPermute3x224x224
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x3x224x224xf16>
func.func @NCEPermute3x224x224(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       [[COPY_INPUT:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:          mode = "OVERLAPPED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          num_clusters = 2 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]
    // CHECK-SAME:      }


    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "OVERLAPPED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          num_clusters = 2 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]
    // CHECK-SAME:      }


    // CHECK:       [[COPY_OUTPUT:%.+]] = VPU.Copy
    // CHECK-SAME:      -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEPermuteWithSOK
module @NCEPermuteWithSOK {

config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @NCEPermuteSOK
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x32x64xf16>
func.func @NCEPermuteSOK(%arg0: tensor<1x128x32x64xf16>) -> tensor<1x128x32x64xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = f16,
        dstOrder = #NHWC,
        expandedChannels = 128 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x128x32x64xf16, {order = #NHWC}>

    return %0 : tensor<1x128x32x64xf16, {order = #NHWC}>

    // CHECK:       [[COPY_INPUT:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x128x32x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:          mode = "SEGMENTED",
    // CHECK-SAME:          num_tiles = [1, 4, 1, 1],
    // CHECK-SAME:          num_clusters = 4 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]
    // CHECK-SAME:      }


    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x128x32x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "SEGMENTED",
    // CHECK-SAME:          num_tiles = [1, 4, 1, 1],
    // CHECK-SAME:          num_clusters = 4 : i64,
    // CHECK-SAME:          alignment = [1, 16, 1, 1],
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]
    // CHECK-SAME:      }


    // CHECK:       [[COPY_OUTPUT:%.+]] = VPU.Copy
    // CHECK-SAME:      -> tensor<1x128x32x64xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NCEPermuteDepthwiseConv
module @NCEPermuteDepthwiseConv {

config.Resources 2 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @NCEPermuteDWCONV3x224x224
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x3x224x224xf16>
func.func @NCEPermuteDWCONV3x224x224(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x16x224x224x!qElemType, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Reorder<#NHWC>
        ]

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 16 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    %1 = VPU.NCE.DepthConvolution(%0, %WEIGHTS) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [16, 1, 1, 1],
        strides = [1, 1]
    } -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       [[COPY_INPUT:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:          mode = "OVERLAPPED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          num_clusters = 2 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]
    // CHECK-SAME:      }


    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "OVERLAPPED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          num_clusters = 2 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 112, 224], [1, 16, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 112, 224], [1, 16, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]
    // CHECK-SAME:      }


    // CHECK:       [[COPY_OUTPUT:%.+]] = VPU.Copy
    // CHECK-SAME:      -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NCEPermuteConv3x3
module @NCEPermuteConv3x3 {

config.Resources 2 of @NCE at 1.700000e+03 MHz

// CHECK: @NCEPermuteCONV3x3([[ARG0:%.+]]: tensor<1x3x224x224xf16>)
func.func @NCEPermuteCONV3x3(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x16x224x224x!qElemType, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<16x16x3x3x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Reorder<#NHWC>
        ]
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 16 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    %1 = VPU.NCE.Convolution(%0, %WEIGHTS) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [16, 16, 3, 3],
        strides = [1, 1]
    } : tensor<1x16x224x224x!qElemType, {order = #NHWC}>, tensor<16x16x3x3x!qElemType, {order = #NHWC}> -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       [[COPY_INPUT:%.+]] = VPU.Copy([[ARG0]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:          mode = "OVERLAPPED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          num_clusters = 2 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]
    // CHECK-SAME:      }

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "OVERLAPPED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          num_clusters = 2 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 112, 224], [1, 16, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 113, 224], [1, 16, 113, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 111, 0]]
    // CHECK-SAME:      }

    // CHECK:       [[COPY_OUTPUT:%.+]] = VPU.Copy
    // CHECK-SAME:      -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiDepthConv
func.func @MultiDepthConv(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x96x112x112xf16, {order = #NHWC}> {
    %weight_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %dwconv_1 = VPU.NCE.DepthConvolution(%arg0, %weight_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>

    %weight_2 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %dwconv_2 = VPU.NCE.DepthConvolution(%arg0, %weight_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>

    %weight_3 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %dwconv_3 = VPU.NCE.DepthConvolution(%arg0, %weight_3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>

    %concat = VPU.Concat(%dwconv_1, %dwconv_2, %dwconv_3) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]]} : tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x96x112x112xf16, {order = #NHWC}>
    return %concat: tensor<1x96x112x112xf16, {order = #NHWC}>

    // CHECK: [[TILING_COPY_1:%.*]] = VPU.Copy
    // CHECK: [[TILING_COPY_2:%.*]] = VPU.Copy
    // CHECK: [[DWCONV_1:%.*]] = VPU.NCE.DepthConvolution
    // CHECK: [[TILING_COPY_OUT_1:%.*]] = VPU.Copy

    // CHECK: [[TILING_COPY_4:%.*]] = VPU.Copy
    // CHECK: [[TILING_COPY_5:%.*]] = VPU.Copy
    // CHECK: [[DWCONV_2:%.*]] = VPU.NCE.DepthConvolution
    // CHECK: [[TILING_COPY_OUT_2:%.*]] = VPU.Copy

    // CHECK: [[TILING_COPY_7:%.*]] = VPU.Copy
    // CHECK: [[TILING_COPY_8:%.*]] = VPU.Copy
    // CHECK: [[DWCONV_3:%.*]] = VPU.NCE.DepthConvolution
    // CHECK: [[TILING_COPY_OUT_3:%.*]] = VPU.Copy

    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[TILING_COPY_OUT_1]], [[TILING_COPY_OUT_2]], [[TILING_COPY_OUT_3]])
    // CHECK:    {static_offsets = [
    // CHECK-SAME:   [0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]
    // CHECK-SAME:   ]} : tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
    // CHECK-SAME:    -> tensor<1x96x112x112xf16, {order = #NHWC}>
    // CHECK:   return [[CONCAT]] : tensor<1x96x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 2 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @InplaceEltwiseInferFromInput
func.func @InplaceEltwiseInferFromInput(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>

    %1 = VPU.NCE.AveragePool(%0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
            is_inplace = true,
            ppe = #VPU.PPEStub<>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>
        } -> tensor<1x32x112x112xf16, {order = #NHWC}>

    return %2 : tensor<1x32x112x112xf16, {order = #NHWC}>

    // CHECK: [[TILING_COPY_1:%.+]] = VPU.Copy
    // CHECK:       [[AVG_POOL_1:%.+]] = VPU.NCE.AveragePool
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 56, 112], [1, 32, 56, 112]],
    // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]],
    // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 57, 112], [1, 32, 57, 112]],
    // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 55, 0]]
    // CHECK: [[TILING_COPY_2:%.+]] = VPU.Copy

    // CHECK: [[TILING_COPY_3:%.+]] = VPU.Copy
    // CHECK:       [[AVG_POOL_2:%.+]] = VPU.NCE.AveragePool
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 56, 112], [1, 32, 56, 112]],
    // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]],
    // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 57, 112], [1, 32, 57, 112]],
    // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 55, 0]]
    // CHECK: [[TILING_COPY_4:%.+]] = VPU.Copy

    // CHECK: [[INPUT0_CMX:%.+]] = VPU.Copy
    // CHECK: [[INPUT1_CMX:%.+]] = VPU.Copy

    // CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Eltwise(
    // CHECK-SAME:       [[INPUT0_CMX]],
    // CHECK-SAME:       [[INPUT1_CMX]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 56, 112], [1, 32, 56, 112]],
    // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]],
    // CHECK-NOT{LITERAL}:        memory_shapes = [[1, 32, 56, 112], [1, 32, 56, 112]],
    // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 57, 112], [1, 32, 57, 112]],
    // CHECK-NOT{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]]
    // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 55, 0]]

    // CHECK: [[TILING_COPY_OUT_2:%.+]] = VPU.Copy
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @SubtractSWSOHTileAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x44xf16>
func.func @SubtractSWSOHTileAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x1x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x1x44xf16> -> tensor<1x32x44x44xf16>

    return %0 : tensor<1x32x44x44xf16>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[SUBTRACT:%.+]] = VPU.Subtract([[INPUT0]],
    //CHECK:                                        [[INPUT1]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[SUBTRACT]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @AddSWSOHTileNotAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x44x44xf16>
func.func @AddSWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x44x44xf16> -> tensor<1x32x44x44xf16>

    return %0 : tensor<1x32x44x44xf16>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[ADD:%.+]] = VPU.Add([[INPUT0]],
    //CHECK:                              [[INPUT1]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[ADD]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @AddSWSOHTileAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x44xf16>
func.func @AddSWSOHTileAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x1x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x1x44xf16> -> tensor<1x32x44x44xf16>

    return %0 : tensor<1x32x44x44xf16>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[ADD:%.+]] = VPU.Add([[INPUT0]],
    //CHECK:                              [[INPUT1]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[ADD]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @EqualSWSOHTileNotAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x44x44xf16>
func.func @EqualSWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x44x44xf16>) -> tensor<1x32x44x44xi8> {
    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x44x44xf16> -> tensor<1x32x44x44xi8>

    return %0 : tensor<1x32x44x44xi8>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[EQUAL:%.+]] = VPU.Equal([[INPUT0]],
    //CHECK:                                  [[INPUT1]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[EQUAL]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xi8>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @EqualSWSOHTileAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x44xf16>
func.func @EqualSWSOHTileAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x1x44xf16>) -> tensor<1x32x44x44xi8> {
    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x1x44xf16> -> tensor<1x32x44x44xi8>

    return %0 : tensor<1x32x44x44xi8>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[EQUAL:%.+]] = VPU.Equal([[INPUT0]],
    //CHECK:                                  [[INPUT1]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[EQUAL]]

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xi8>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @FloorSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x16x512xf16>)
func.func @FloorSWSOH(%arg0: tensor<1x16x16x512xf16>) -> tensor<1x16x16x512xf16> {

    %0 = VPU.Floor(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
          : tensor<1x16x16x512xf16> -> tensor<1x16x16x512xf16>

    return %0 : tensor<1x16x16x512xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>

    //CHECK:        [[FLOOR:%.+]] = VPU.Floor([[INPUT]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[FLOOR]]

    //CHECK:        return [[OUTPUT]] : tensor<1x16x16x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @FloorSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x1x513xf16>)
func.func @FloorSWSOK(%arg0: tensor<1x16x1x513xf16>) -> tensor<1x16x1x513xf16> {

    %0 = VPU.Floor(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
          : tensor<1x16x1x513xf16> -> tensor<1x16x1x513xf16>

    return %0 : tensor<1x16x1x513xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy({{[^:]+}}
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}>

    //CHECK:        [[FLOOR:%.+]] = VPU.Floor([[INPUT]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[FLOOR]]

    //CHECK:        return [[OUTPUT]] : tensor<1x16x1x513xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @FloorSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x1x513xf16>)
func.func @FloorSWClustering(%arg0: tensor<1x1x1x513xf16>) -> tensor<1x1x1x513xf16> {

    %0 = VPU.Floor(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
          : tensor<1x1x1x513xf16> -> tensor<1x1x1x513xf16>

    return %0 : tensor<1x1x1x513xf16>

    //CHECK:        [[INPUT:%.+]]  = VPU.Copy({{[^:]+}}
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[FLOOR:%.+]] = VPU.Floor([[INPUT]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[FLOOR]]

    //CHECK:        return [[OUTPUT]] : tensor<1x1x1x513xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @RoundSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x16x512xf16>)
func.func @RoundSWSOH(%arg0: tensor<1x16x16x512xf16>) -> tensor<1x16x16x512xf16> {

    %0 = VPU.Round(%arg0) {
            mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
          : tensor<1x16x16x512xf16> -> tensor<1x16x16x512xf16>

    return %0 : tensor<1x16x16x512xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>

    //CHECK:        [[ROUND:%.+]] = VPU.Round([[INPUT]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[ROUND]]

    //CHECK:        return [[OUTPUT]] : tensor<1x16x16x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @RoundSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x1x513xf16>)
func.func @RoundSWSOK(%arg0: tensor<1x16x1x513xf16>) -> tensor<1x16x1x513xf16> {

    %0 = VPU.Round(%arg0) {
            mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
          : tensor<1x16x1x513xf16> -> tensor<1x16x1x513xf16>

    return %0 : tensor<1x16x1x513xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy({{[^:]+}}
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}>

    //CHECK:        [[ROUND:%.+]] = VPU.Round([[INPUT]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[ROUND]]

    //CHECK:        return [[OUTPUT]] : tensor<1x16x1x513xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @RoundSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x1x513xf16>)
func.func @RoundSWClustering(%arg0: tensor<1x1x1x513xf16>) -> tensor<1x1x1x513xf16> {

    %0 = VPU.Round(%arg0) {
            mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
          : tensor<1x1x1x513xf16> -> tensor<1x1x1x513xf16>

    return %0 : tensor<1x1x1x513xf16>

    //CHECK:        [[INPUT:%.+]]  = VPU.Copy({{[^:]+}}
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[ROUND:%.+]] = VPU.Round([[INPUT]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[ROUND]]

    //CHECK:        return [[OUTPUT]] : tensor<1x1x1x513xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AccumulateClustering
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @AccumulateClustering(
    %LHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    } : tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   [[COPY_LHS:%.+]] = VPU.Copy([[LHS]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS:%.+]] = VPU.Copy([[RHS]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_LHS_SCALES:%.+]] = VPU.Copy([[LHS_SCALES]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS_SCALES:%.+]] = VPU.Copy([[RHS_SCALES]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[ACCUMULATE:%.+]] = VPU.Accumulate(
    // CHECK-SAME:      [[COPY_LHS]]
    // CHECK-SAME:      [[COPY_RHS]]
    // CHECK-SAME:      [[COPY_LHS_SCALES]]
    // CHECK-SAME:      [[COPY_RHS_SCALES]]
    // CHECK-SAME:    -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_OUT:%.+]] = VPU.Copy([[ACCUMULATE]]
    // CHECK-SAME:  -> tensor<1x64x16x1xf16, {order = #NHWC}>

    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AccumulateSplitOverHeight
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @AccumulateSplitOverHeight(
    %LHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   [[COPY_LHS:%.+]] = VPU.Copy([[LHS]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, {{6|3|4}}, 1],
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS:%.+]] = VPU.Copy([[RHS]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, {{6|3|4}}, 1],
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_LHS_SCALES:%.+]] = VPU.Copy([[LHS_SCALES]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS_SCALES:%.+]] = VPU.Copy([[RHS_SCALES]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[ACCUMULATE:%.+]] = VPU.Accumulate(
    // CHECK-SAME:      [[COPY_LHS]]
    // CHECK-SAME:      [[COPY_RHS]]
    // CHECK-SAME:      [[COPY_LHS_SCALES]]
    // CHECK-SAME:      [[COPY_RHS_SCALES]]
    // CHECK-SAME:    -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, {{6|3|4}}, 1],
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_OUT:%.+]] = VPU.Copy([[ACCUMULATE]]
    // CHECK-SAME:  -> tensor<1x64x16x1xf16, {order = #NHWC}>

    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AccumulateSplitOverKernel
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @AccumulateSplitOverKernel(
    %LHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    } : tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   [[COPY_LHS:%.+]] = VPU.Copy([[LHS]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, {{6|3|4}}, 1, 1],
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS:%.+]] = VPU.Copy([[RHS]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, {{6|3|4}}, 1, 1],
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_LHS_SCALES:%.+]] = VPU.Copy([[LHS_SCALES]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, {{6|3|4}}, 1, 1],
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS_SCALES:%.+]] = VPU.Copy([[RHS_SCALES]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, {{6|3|4}}, 1, 1],
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[ACCUMULATE:%.+]] = VPU.Accumulate(
    // CHECK-SAME:      [[COPY_LHS]]
    // CHECK-SAME:      [[COPY_RHS]]
    // CHECK-SAME:      [[COPY_LHS_SCALES]]
    // CHECK-SAME:      [[COPY_RHS_SCALES]]
    // CHECK-SAME:    -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, {{6|3|4}}, 1, 1],
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_OUT:%.+]] = VPU.Copy([[ACCUMULATE]]
    // CHECK-SAME:  -> tensor<1x64x16x1xf16, {order = #NHWC}>

    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AccumulateSplitOverWidth
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x32xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x32xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @AccumulateSplitOverWidth(
    %LHS: tensor<1x64x16x32xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x32xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x32xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>
    } : tensor<1x64x16x32xf16, {order = #NHWC}>,
        tensor<1x64x16x32xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x32xf16, {order = #NHWC}>

    // CHECK:   [[COPY_LHS:%.+]] = VPU.Copy([[LHS]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x32xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, {{6|3|4}}],
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS:%.+]] = VPU.Copy([[RHS]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x32xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, {{6|3|4}}],
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_LHS_SCALES:%.+]] = VPU.Copy([[LHS_SCALES]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_RHS_SCALES:%.+]] = VPU.Copy([[RHS_SCALES]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[ACCUMULATE:%.+]] = VPU.Accumulate(
    // CHECK-SAME:      [[COPY_LHS]]
    // CHECK-SAME:      [[COPY_RHS]]
    // CHECK-SAME:      [[COPY_LHS_SCALES]]
    // CHECK-SAME:      [[COPY_RHS_SCALES]]
    // CHECK-SAME:    -> !VPU.DistributedTensor<1x64x16x32xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, {{6|3|4}}],
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_OUT:%.+]] = VPU.Copy([[ACCUMULATE]]
    // CHECK-SAME:  -> tensor<1x64x16x32xf16, {order = #NHWC}>

    return %ACCUMULATE : tensor<1x64x16x32xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x64x16x32xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PoolingSplitOverWidth
// CHECK-SAME: ([[DATA:%arg[0-9]]]: tensor<1x16x32x64xf16, {order = #NHWC}>)
func.func @PoolingSplitOverWidth(
    %DATA: tensor<1x16x32x64xf16, {order = #NHWC}>
) -> tensor<1x16x32x64xf16, {order = #NHWC}> {
    %POOL = VPU.NCE.MaxPool(%DATA) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [3, 3]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    // CHECK:   [[COPY_INPUT:%.+]] = VPU.Copy([[DATA]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x16x32x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, {{6|3|4}}],
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[POOL:%.+]] = VPU.NCE.MaxPool(
    // CHECK-SAME:      [[COPY_INPUT]]
    // CHECK-SAME:    -> !VPU.DistributedTensor<1x16x32x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, {{6|3|4}}],
    // CHECK-SAME:      num_clusters = {{6|3|4}} : i64
    // CHECK-SAME:  }>

    // CHECK:   [[COPY_OUT:%.+]] = VPU.Copy([[POOL]]
    // CHECK-SAME:  -> tensor<1x16x32x64xf16, {order = #NHWC}>

    return %POOL : tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x16x32x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @FakeQuantizeSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x3x384x640xf16>)
func.func @FakeQuantizeSWSOH(%arg0: tensor<1x3x384x640xf16>) -> tensor<1x3x384x640xf16> {
    %inLow = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+01> : tensor<1x3x1x1xf16>
    %inHigh = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+01> : tensor<1x3x1x1xf16>
    %outLow = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %outHigh = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x3x384x640xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x384x640xf16>
    return %fq : tensor<1x3x384x640xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+01> : tensor<1x3x1x1xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+01> : tensor<1x3x1x1xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>

    //CHECK: [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x3x384x640xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, {{6|3|4}}, 1], num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[IN_LOW_COPY:%.+]] = VPU.Copy([[IN_LOW]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[IN_HIGH_COPY:%.+]]  = VPU.Copy([[IN_HIGH]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[OUT_LOW_COPY:%.+]] = VPU.Copy([[OUT_LOW]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[OUT_HIGH_COPY:%.+]] = VPU.Copy([[OUT_HIGH]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[FQ_CLUSTER:%.+]] = VPU.FakeQuantize([[INPUT_COPY]], [[IN_LOW_COPY]], [[IN_HIGH_COPY]], [[OUT_LOW_COPY]], [[OUT_HIGH_COPY]])
    //CHECK-SAME: -> !VPU.DistributedTensor<1x3x384x640xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, {{6|3|4}}, 1], num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[OUTPUT:%.+]] = VPU.Copy([[FQ_CLUSTER]]
    //CHECK: return [[OUTPUT]] : tensor<1x3x384x640xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @FakeQuantizeSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x128x1x512xf16>)
func.func @FakeQuantizeSWSOK(%arg0: tensor<1x128x1x512xf16>) -> tensor<1x128x1x512xf16> {
    %inLow = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %inHigh = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    %outLow = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    %outHigh = const.Declare tensor<1x128x1x1xf16> = dense<1.000000e+01> : tensor<1x128x1x1xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x128x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x1x512xf16>
    return %fq : tensor<1x128x1x512xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
        //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<1.000000e+01> : tensor<1x128x1x1xf16>

    //CHECK: [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x128x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, {{6|3|4}}, 1, 1], num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[IN_LOW_COPY:%.+]] = VPU.Copy([[IN_LOW]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[IN_HIGH_COPY:%.+]]  = VPU.Copy([[IN_HIGH]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[OUT_LOW_COPY:%.+]] = VPU.Copy([[OUT_LOW]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, {{6|3|4}}, 1, 1], num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[OUT_HIGH_COPY:%.+]] = VPU.Copy([[OUT_HIGH]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, {{6|3|4}}, 1, 1], num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[FQ_CLUSTER:%.+]] = VPU.FakeQuantize([[INPUT_COPY]], [[IN_LOW_COPY]], [[IN_HIGH_COPY]], [[OUT_LOW_COPY]], [[OUT_HIGH_COPY]])
    //CHECK-SAME: -> !VPU.DistributedTensor<1x128x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, {{6|3|4}}, 1, 1], num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[OUTPUT:%.+]] = VPU.Copy([[FQ_CLUSTER]]
    //CHECK: return [[OUTPUT]] : tensor<1x128x1x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @FakeQuantizeSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x1x512xf16>)
func.func @FakeQuantizeSWClustering(%arg0: tensor<1x1x1x512xf16>) -> tensor<1x1x1x512xf16> {
    %inLow = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    %inHigh = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>
    %outLow = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    %outHigh = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, levels = 256 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16>
    return %fq : tensor<1x1x1x512xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>

    //CHECK: [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[IN_LOW_COPY:%.+]] = VPU.Copy([[IN_LOW]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[IN_HIGH_COPY:%.+]] = VPU.Copy([[IN_HIGH]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[OUT_LOW_COPY:%.+]] = VPU.Copy([[OUT_LOW]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[OUT_HIGH_COPY:%.+]] = VPU.Copy([[OUT_HIGH]]
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[FQ_CLUSTER:%.+]] = VPU.FakeQuantize([[INPUT_COPY]], [[IN_LOW_COPY]], [[IN_HIGH_COPY]], [[OUT_LOW_COPY]], [[OUT_HIGH_COPY]])
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3|4}} : i64, uniform_distributed_segments

    //CHECK: [[OUTPUT:%.+]] = VPU.Copy([[FQ_CLUSTER]]
    //CHECK: return [[OUTPUT]] : tensor<1x1x1x512xf16>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @SelectSWSplitOverHeight
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x10x40x40xf16>, [[INPUT1:%.+]]: tensor<1x10x40x40xf16>)
func.func @SelectSWSplitOverHeight(%arg0: tensor<1x10x40x40xf16>, %arg1: tensor<1x10x40x40xf16>) -> tensor<1x10x40x40xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x10x40x40xf16>, tensor<1x1x1x1xf16>, tensor<1x10x40x40xf16> -> tensor<1x10x40x40xf16>
    return %0 : tensor<1x10x40x40xf16>

    //CHECK-DAG:    [[INPUT0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x10x40x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 6, 40], [1, 10, 6, 40]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0], [0, 0, 28, 0], [0, 0, 34, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 6, 40], [1, 10, 6, 40]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0], [0, 0, 28, 0], [0, 0, 34, 0]]}>

    //CHECK:        [[INPUT0_COPY:%.+]] = VPU.Copy([[INPUT0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[INPUT1_COPY:%.+]] = VPU.Copy([[INPUT1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x10x40x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 6, 40], [1, 10, 6, 40]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0], [0, 0, 28, 0], [0, 0, 34, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 6, 40], [1, 10, 6, 40]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0], [0, 0, 28, 0], [0, 0, 34, 0]]}>

    //CHECK:        [[SELECT:%.+]] = VPU.Select(
    //CHECK-SAME:           [[INPUT_COPY]],
    //CHECK-SAME:           [[INPUT0_COPY]],
    //CHECK-SAME:           [[INPUT1_COPY]]

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[SELECT]]

    //CHECK: return [[OUTPUT]] : tensor<1x10x40x40xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @SelectSWSplitOverKernel
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x40xf16>, [[INPUT1:%.+]]: tensor<1x16x1x40xf16>)
func.func @SelectSWSplitOverKernel(%arg0: tensor<1x16x1x40xf16>, %arg1: tensor<1x16x1x40xf16>) -> tensor<1x16x1x40xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        } : tensor<1x16x1x40xf16>, tensor<1x1x1x1xf16>, tensor<1x16x1x40xf16> -> tensor<1x16x1x40xf16>
    return %0 : tensor<1x16x1x40xf16>

    //CHECK-DAG:    [[INPUT0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x1x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 2, 1, 40], [1, 2, 1, 40]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 2, 1, 40], [1, 2, 1, 40]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}>

    //CHECK:        [[INPUT0_COPY:%.+]] = VPU.Copy([[INPUT0]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[INPUT1_COPY:%.+]] = VPU.Copy([[INPUT1]]
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x1x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 2, 1, 40], [1, 2, 1, 40]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 2, 1, 40], [1, 2, 1, 40]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}>

    //CHECK:        [[SELECT:%.+]] = VPU.Select(
    //CHECK-SAME:           [[INPUT_COPY]],
    //CHECK-SAME:           [[INPUT0_COPY]],
    //CHECK-SAME:           [[INPUT1_COPY]]

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[SELECT]]

    //CHECK: return [[OUTPUT]] : tensor<1x16x1x40xf16>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseInputsSameOffsets
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x128x72x72xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x128x72x72xf16, {order = #NHWC}>)
func.func @EltwiseInputsSameOffsets(%arg0: tensor<1x128x72x72xf16, {order = #NHWC}>, %arg1: tensor<1x128x72x72xf16, {order = #NHWC}>) -> tensor<1x128x72x72xf16> {
    %cst = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} : tensor<1x128x72x72xf16, {order = #NHWC}>, tensor<64x128x1x1xf16, {order = #NHWC}> -> tensor<1x64x72x72xf16, {order = #NHWC}>
    %1 = VPU.NCE.DepthConvolution(%0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [64, 1, 3, 3], strides = [1, 1]} -> tensor<1x64x72x72xf16, {order = #NHWC}>
    %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x72x72xf16, {order = #NHWC}>, tensor<1x64x72x72xf16, {order = #NHWC}> -> tensor<1x128x72x72xf16, {order = #NHWC}>

    %3 = VPU.NCE.Eltwise(%2, %arg1) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>
        } -> tensor<1x128x72x72xf16>

    return %3 : tensor<1x128x72x72xf16>

    // CHECK:                   [[TILING_COPY_0:%.+]] = VPU.Copy([[ARG0]]
    // CHECK:                   [[TILING_COPY_1:%.+]] = VPU.Copy
    // CHECK:                   [[TILING_CONV:%.+]] = VPU.NCE.Convolution([[TILING_COPY_0]]
    // CHECK-SAME{LITERAL}:         -> !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72]], compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]], memory_shapes = [[1, 64, 19, 72], [1, 64, 20, 72], [1, 64, 20, 72], [1, 64, 19, 72]], memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}>
    // CHECK:                   [[TILING_COPY_3:%.*]] = VPU.Copy([[TILING_CONV]]
    // CHECK:                   [[TILING_COPY_4:%.*]] = VPU.Copy([[TILING_COPY_3]]
    // CHECK:                   [[TILING_COPY_5:%.*]] = VPU.Copy
    // CHECK:                   [[TILING_DWCONV:%.*]] = VPU.NCE.DepthConvolution([[TILING_COPY_4]]
    // CHECK-SAME{LITERAL}:         -> !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72]], compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]], memory_shapes = [[1, 64, 19, 72], [1, 64, 20, 72], [1, 64, 20, 72], [1, 64, 19, 72]], memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}>
    // CHECK:                   [[TILING_COPY_7:%.+]] = VPU.Copy([[TILING_DWCONV]]
    // CHECK:                   [[CONCAT:%.+]] = VPU.Concat([[TILING_COPY_3]], [[TILING_COPY_7]])
    // CHECK-SAME{LITERAL}:         {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x72x72xf16, {order = #NHWC}>, tensor<1x64x72x72xf16, {order = #NHWC}> -> tensor<1x128x72x72xf16, {order = #NHWC}>
    // CHECK:                   [[TILING_COPY_8:%.+]] = VPU.Copy([[CONCAT]]
    // CHECK-SAME{LITERAL}:         -> !VPU.DistributedTensor<1x128x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]], compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]], memory_shapes = [[1, 128, 19, 72], [1, 128, 20, 72], [1, 128, 20, 72], [1, 128, 19, 72]], memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}>
    // CHECK:                   [[TILING_COPY_9:%.+]] = VPU.Copy([[ARG1]]
    // CHECK-SAME{LITERAL}:         -> !VPU.DistributedTensor<1x128x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]], compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]], memory_shapes = [[1, 128, 19, 72], [1, 128, 20, 72], [1, 128, 20, 72], [1, 128, 19, 72]], memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}>
    // CHECK:                   [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[TILING_COPY_8]], [[TILING_COPY_9]])
    // CHECK-SAME{LITERAL}:         -> !VPU.DistributedTensor<1x128x72x72xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]], compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]], memory_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]], memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]]}>
    // CHECK:                   [[TILING_COPY_9:%.+]] = VPU.Copy([[ELTWISE]]
    // CHECK:                   return  [[TILING_COPY_9]] : tensor<1x128x72x72xf16>
}

// -----

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AndClustering
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x32x32xf16>, [[INPUT1:%.+]]: tensor<1x1x32x32xf16>)
func.func @AndClustering(%arg0: tensor<1x16x32x32xf16>, %arg1: tensor<1x1x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = VPU.And(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    } : tensor<1x16x32x32xf16>, tensor<1x1x32x32xf16> -> tensor<1x16x32x32xf16>

    return %0 : tensor<1x16x32x32xf16>

    // CHECK:               [[INPUT0_COPY:%.+]] = VPU.Copy([[INPUT0]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:               [[INPUT1_COPY:%.+]] = VPU.Copy([[INPUT1]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:               [[AND:%.+]] = VPU.And([[INPUT0_COPY]], [[INPUT1_COPY]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:               [[RES:%.+]] = VPU.Copy([[AND]]

    // CHECK:               return [[RES]] : tensor<1x16x32x32xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @LogSoftmaxSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x16x512xf16>)
func.func @LogSoftmaxSWSOH(%arg0: tensor<1x16x16x512xf16>) -> tensor<1x16x16x512xf16> {

    %0 = VPU.LogSoftmax(%arg0) {
            axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
          : tensor<1x16x16x512xf16> -> tensor<1x16x16x512xf16>

    return %0 : tensor<1x16x16x512xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy([[INPUT_DATA]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>

    //CHECK:        [[LOG_SOFTMAX:%.+]] = VPU.LogSoftmax([[INPUT]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[LOG_SOFTMAX]]

    //CHECK:        return [[OUTPUT]] : tensor<1x16x16x512xf16>
}

// -----

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AndSplitOverHeight
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x32x32xf16>, [[INPUT1:%.+]]: tensor<1x1x32x32xf16>)
func.func @AndSplitOverHeight(%arg0: tensor<1x16x32x32xf16>, %arg1: tensor<1x1x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = VPU.And(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x16x32x32xf16>, tensor<1x1x32x32xf16> -> tensor<1x16x32x32xf16>

    return %0 : tensor<1x16x32x32xf16>

    // CHECK:               [[INPUT0_COPY:%.+]] = VPU.Copy([[INPUT0]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]], memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>

    // CHECK:               [[INPUT1_COPY:%.+]] = VPU.Copy([[INPUT1]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 6, 32], [1, 1, 6, 32], [1, 1, 5, 32], [1, 1, 5, 32], [1, 1, 5, 32], [1, 1, 5, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]], memory_shapes = [[1, 1, 6, 32], [1, 1, 6, 32], [1, 1, 5, 32], [1, 1, 5, 32], [1, 1, 5, 32], [1, 1, 5, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>

    // CHECK:               [[AND:%.+]] = VPU.And([[INPUT0_COPY]], [[INPUT1_COPY]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]], memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>

    // CHECK:               [[RES:%.+]] = VPU.Copy([[AND]]

    // CHECK:               return [[RES]] : tensor<1x16x32x32xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @LogSoftmaxSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x1x513xf16>)
func.func @LogSoftmaxSWSOK(%arg0: tensor<1x16x1x513xf16>) -> tensor<1x16x1x513xf16> {

    %0 = VPU.LogSoftmax(%arg0) {
            axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
          : tensor<1x16x1x513xf16> -> tensor<1x16x1x513xf16>

    return %0 : tensor<1x16x1x513xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.Copy({{[^:]+}}
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}>

    //CHECK:        [[LOG_SOFTMAX:%.+]] = VPU.LogSoftmax([[INPUT]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[LOG_SOFTMAX]]

    //CHECK:        return [[OUTPUT]] : tensor<1x16x1x513xf16>
}

// -----

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AndSplitOverKernel
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x64x32x32xf16>, [[INPUT1:%.+]]: tensor<1x1x32x32xf16>)
func.func @AndSplitOverKernel(%arg0: tensor<1x64x32x32xf16>, %arg1: tensor<1x1x32x32xf16>) -> tensor<1x64x32x32xf16> {
    %0 = VPU.And(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    } : tensor<1x64x32x32xf16>, tensor<1x1x32x32xf16> -> tensor<1x64x32x32xf16>

    return %0 : tensor<1x64x32x32xf16>

    // CHECK:               [[INPUT0_COPY:%.+]] = VPU.Copy([[INPUT0]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x64x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 10, 32, 32], [1, 10, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]], memory_shapes = [[1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 10, 32, 32], [1, 10, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]}>

    // CHECK:               [[INPUT1_COPY:%.+]] = VPU.Copy([[INPUT1]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x1x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:               [[AND:%.+]] = VPU.And([[INPUT0_COPY]], [[INPUT1_COPY]])
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x64x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 10, 32, 32], [1, 10, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]], memory_shapes = [[1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 10, 32, 32], [1, 10, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]}>

    // CHECK:               [[RES:%.+]] = VPU.Copy([[AND]]

    // CHECK:               return [[RES]] : tensor<1x64x32x32xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @LogSoftmaxSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x1x513xf16>)
func.func @LogSoftmaxSWClustering(%arg0: tensor<1x1x1x513xf16>) -> tensor<1x1x1x513xf16> {

    %0 = VPU.LogSoftmax(%arg0) {
            axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
          : tensor<1x1x1x513xf16> -> tensor<1x1x1x513xf16>

    return %0 : tensor<1x1x1x513xf16>

    //CHECK:        [[INPUT:%.+]]  = VPU.Copy({{[^:]+}}
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[LOG_SOFTMAX:%.+]] = VPU.LogSoftmax([[INPUT]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[LOG_SOFTMAX]]

    //CHECK:        return [[OUTPUT]] : tensor<1x1x1x513xf16>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @SinSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @SinSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Sin(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[SIN:%.+]] = VPU.Sin([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[SIN]]
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @SinSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @SinSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Sin(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[SIN:%.+]] = VPU.Sin([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[SIN]]
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @SinSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @SinSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Sin(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[SIN:%.+]] = VPU.Sin([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[SIN]]
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @CosSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @CosSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Cos(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[COS:%.+]] = VPU.Cos([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[COS]]
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @CosSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @CosSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Cos(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[COS:%.+]] = VPU.Cos([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[COS]]
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @CosSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @CosSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Cos(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[COS:%.+]] = VPU.Cos([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[COS]]
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @ExpSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @ExpSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Exp(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[EXP:%.+]] = VPU.Exp([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[EXP]]
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @ExpSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @ExpSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Exp(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[EXP:%.+]] = VPU.Exp([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[EXP]]
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @ExpSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @ExpSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Exp(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[EXP:%.+]] = VPU.Exp([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[EXP]]
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EnableChainOpsToNCEClusteringKHSwitchWithCompressedConv
// CHECK-SAME:  [[INPUT:%.+]]:  tensor<1x80x4x80xf16, {order = #NHWC}>
func.func @EnableChainOpsToNCEClusteringKHSwitchWithCompressedConv(%arg0: tensor<1x80x4x80xf16, {order = #NHWC}>) -> tensor<1x16x80x80xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x48xf16, {order = #NHWC}> = dense<0.100000e+00> : tensor<16x1x1x48xf16, {order = #NHWC}>
    %cst_0 = const.Declare tensor<16x1x1x4xsi32> = dense<8> : tensor<16x1x1x4xsi32>
    %0 = VPU.NCE.MaxPool(%arg0) {
            kernel_size = [1, 1],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } -> tensor<1x80x4x80xf16, {order = #NWCH}>
    %1 = VPU.PermuteCast(%0) {
            dst_order = #NHWC,
            mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        } : tensor<1x80x4x80xf16, {order = #NWCH}>
         -> tensor<1x4x80x80xf16, {order = #NHWC}>
    %2 = VPU.NCE.CompressConvolution(%1, %cst, %cst_0) {
            cm_sp_pattern = 15 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 4, 3, 3], strides = [1, 1]
        } -> tensor<1x16x80x80xf16, {order = #NHWC}>

    return %2 : tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<16x1x1x48xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<16x1x1x48xf16, {order = #NHWC}>
    // CHECK-DAG:    [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<8> : tensor<16x1x1x4xsi32>

    // CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x80x4x80xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]

    // CHECK:        [[MAX_POOL_CMX:%.+]] = VPU.NCE.MaxPool([[INPUT_CMX]])
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x80x4x80xf16, #NWCH, @CMX_NN,
    // CHECK-SAME:           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 4, 80], [1, 16, 4, 80], [1, 16, 4, 80], [1, 16, 4, 80]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 80, 4, 80], [1, 80, 4, 80], [1, 80, 4, 80], [1, 80, 4, 80]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:        [[MAX_POOL_DDR:%.+]] = VPU.Copy([[MAX_POOL_CMX]]
    // CHECK-SAME:       -> tensor<1x80x4x80xf16, {order = #NWCH}>

    // CHECK:        [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[MAX_POOL_DDR]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:           : tensor<1x80x4x80xf16, {order = #NWCH}> -> tensor<1x4x80x80xf16, {order = #NHWC}>

    // CHECK:        [[RESHAPED_OUTPUT_CMX:%.+]] = VPU.Copy([[PERMUTE_CAST]])
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x4x80x80xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 4 : i64,

    // CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]])
    // CHECK-SAME:       -> !VPU.DistributedTensor<16x1x1x48xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 4 : i64,

    // CHECK:        [[WEIGHTS_TABLE_CMX:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 4 : i64,

    // CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.CompressConvolution(
    // CHECK-SAME:             [[RESHAPED_OUTPUT_CMX]],
    // CHECK-SAME:             [[WEIGHTS_CMX]],
    // CHECK-SAME:             [[WEIGHTS_TABLE_CMX]])
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x16x80x80xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 20, 80], [1, 16, 20, 80], [1, 16, 20, 80], [1, 16, 20, 80]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 40, 0], [0, 0, 60, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 20, 80], [1, 16, 20, 80], [1, 16, 20, 80], [1, 16, 20, 80]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 40, 0], [0, 0, 60, 0]]

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    // CHECK:        return [[OUT]] : tensor<1x16x80x80xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DisableChainOpsToNCEClusteringKHSwitchWithCompressedConv
// CHECK-SAME:  [[INPUT:%.+]]:  tensor<1x16x288x68xf16, {order = #NHWC}>
func.func @DisableChainOpsToNCEClusteringKHSwitchWithCompressedConv(%arg0: tensor<1x16x288x68xf16, {order = #NHWC}>) -> tensor<1x16x288x544xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<0.200000e+00> : tensor<32x16x1x1xf16, {order = #NHWC}>
    %cst_1 = const.Declare tensor<16x1x1x64xf16, {order = #NHWC}> = dense<0.100000e+00> : tensor<16x1x1x64xf16, {order = #NHWC}>
    %cst_2 = const.Declare tensor<16x1x1x4xsi32> = dense<4> : tensor<16x1x1x4xsi32>
    %0 = VPU.NCE.Convolution(%arg0, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [32, 16, 1, 1], strides = [1, 1]
        } : tensor<1x16x288x68xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<1x32x288x68xf16, {order = #NHWC}>
    %1 = VPU.AffineReshape(%0) {
            dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 4, 288, 544]
        } : tensor<1x32x288x68xf16, {order = #NHWC}> -> tensor<1x4x288x544xf16, {order = #NHWC}>
    %2 = VPU.NCE.CompressConvolution(%1, %cst_1, %cst_2) {
            cm_sp_pattern = 3 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
            pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 2, 5, 5], strides = [1, 1]
        } -> tensor<1x16x288x544xf16, {order = #NHWC}>

    return %2 : tensor<1x16x288x544xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS_0:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.999510e-01> : tensor<32x16x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<16x1x1x64xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<16x1x1x64xf16, {order = #NHWC}>
    // CHECK-DAG:    [[WEIGHTS_TABLE_1:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<4> : tensor<16x1x1x4xsi32>

    // CHECK:        [[INPUT_CMX_0:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x16x288x68xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 4 : i64

    // CHECK:        [[WEIGHTS_CMX_0:%.+]] = VPU.Copy([[WEIGHTS_0]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 4 : i64
    // CHECK-SAME{LITERAL}:   compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],

    // CHECK:        [[OUT_0_CMX:%.+]] = VPU.NCE.Convolution(
    // CHECK-SAME:             [[INPUT_CMX_0]],
    // CHECK-SAME:             [[WEIGHTS_CMX_0]])
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x32x288x68xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 4 : i64

    // CHECK:        [[OUT_0:%.+]] = VPU.Copy([[OUT_0_CMX]]
    // CHECK-SAME:       -> tensor<1x32x288x68xf16, {order = #NHWC}>

    // CHECK:        [[RESHAPED_OUTPUT:%.+]] = VPU.AffineReshape([[OUT_0]]) {
    // CHECK-SAME{LITERAL}:         dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 4, 288, 544]}
    // CHECK-SAME:           : tensor<1x32x288x68xf16, {order = #NHWC}> -> tensor<1x4x288x544xf16, {order = #NHWC}>

    // CHECK:        [[INPUT_CMX_1:%.+]] = VPU.Copy([[RESHAPED_OUTPUT]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x4x288x544xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64,

    // CHECK:        [[WEIGHTS_CMX_1:%.+]] = VPU.Copy([[WEIGHTS_1]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<16x1x1x64xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 4 : i64,

    // CHECK:        [[WEIGHTS_TABLE_CMX_1:%.+]] = VPU.Copy([[WEIGHTS_TABLE_1]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 4 : i64,

    // CHECK:        [[OUT_1_CMX:%.+]] = VPU.NCE.CompressConvolution(
    // CHECK-SAME:             [[INPUT_CMX_1]],
    // CHECK-SAME:             [[WEIGHTS_CMX_1]],
    // CHECK-SAME:             [[WEIGHTS_TABLE_CMX_1]])
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x16x288x544xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64,

    // CHECK:        [[OUT_1:%.+]] = VPU.Copy([[OUT_1_CMX]]

    // CHECK:        return [[OUT_1]] : tensor<1x16x288x544xf16, {order = #NHWC}>
}

}


// -----

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @PReluSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x8x128x128xf16>)
func.func @PReluSWSOH(%arg0: tensor<1x8x128x128xf16>) -> tensor<1x8x128x128xf16> {

    %cst = const.Declare tensor<1x8x1x1xf16> = dense<-1.000000e+01> : tensor<1x8x1x1xf16>
    %0 = VPU.PRelu(%arg0, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
          : tensor<1x8x128x128xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x128x128xf16>

    return %0 : tensor<1x8x128x128xf16>

    // CHECK-DAG:    [[SLOPE:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<-1.000000e+01> : tensor<1x8x1x1xf16>
    // CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_DATA]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x8x128x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64

    // CHECK:        [[INPUT1:%.+]] = VPU.Copy([[SLOPE]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x8x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64

    // CHECK:        [[PRELU:%.+]] = VPU.PRelu([[INPUT0]],
    // CHECK:                                  [[INPUT1]])
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x8x128x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64

    // CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[PRELU]]

    // CHECK:        return [[OUTPUT]] : tensor<1x8x128x128xf16>
}
}

// -----

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @PReluSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x128x1x1xf16>)
func.func @PReluSWSOK(%arg0: tensor<1x128x1x1xf16>) -> tensor<1x128x1x1xf16> {

    %cst = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    %0 = VPU.PRelu(%arg0, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
          : tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf16>

    return %0 : tensor<1x128x1x1xf16>

    // CHECK-DAG:    [[SLOPE:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    // CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_DATA]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64

    // CHECK:        [[INPUT1:%.+]] = VPU.Copy([[SLOPE]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64

    // CHECK:        [[PRELU:%.+]] = VPU.PRelu([[INPUT0]],
    // CHECK:                                  [[INPUT1]])
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64

    // CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[PRELU]]

    // CHECK:        return [[OUTPUT]] : tensor<1x128x1x1xf16>
}
}

// -----

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @PReluSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x8x128x128xf16>)
func.func @PReluSWClustering(%arg0: tensor<1x8x128x128xf16>) -> tensor<1x8x128x128xf16> {

    %cst = const.Declare tensor<1x8x1x1xf16> = dense<-1.000000e+01> : tensor<1x8x1x1xf16>
    %0 = VPU.PRelu(%arg0, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
          : tensor<1x8x128x128xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x128x128xf16>

    return %0 : tensor<1x8x128x128xf16>

    // CHECK-DAG:    [[SLOPE:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<-1.000000e+01> : tensor<1x8x1x1xf16>
    // CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_DATA]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x8x128x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64

    // CHECK:        [[INPUT1:%.+]] = VPU.Copy([[SLOPE]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x8x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64

    // CHECK:        [[PRELU:%.+]] = VPU.PRelu([[INPUT0]],
    // CHECK:                                  [[INPUT1]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x8x128x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64

    // CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[PRELU]]

    // CHECK:        return [[OUTPUT]] : tensor<1x8x128x128xf16>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @RandomUniformSWSOHTile
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x1x1xf32>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf32>
func.func @RandomUniformSWSOHTile(%arg0: tensor<1x1x1x1xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x1x512x1024xf32> {
    %0 = VPU.RandomUniform(%arg0, %arg1) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, global_seed = 0 : i64, op_seed = 0 : i64, outputType = f32, output_shape = [1, 1, 512, 1024]
            } : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x1024xf32>

    return %0 : tensor<1x1x512x1024xf32>

    // CHECK:        [[MIN:%.+]] = VPU.Copy([[INPUT_0]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64
    // CHECK:        [[MAX:%.+]] = VPU.Copy([[INPUT_1]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64

    // CHECK:        [[RANDOMUNIFORM:%.+]] = VPU.RandomUniform([[MIN]], [[MAX]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x512x1024xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64

    // CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[RANDOMUNIFORM]]

    // CHECK:        return [[OUTPUT]] : tensor<1x1x512x1024xf32>
}
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @BitwiseOrSwWithSOH
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x1x1152x1xi8>, [[INPUT1:%.+]]: tensor<1x1x1152x1xi8>
func.func @BitwiseOrSwWithSOH(%arg0: tensor<1x1x1152x1xi8>, %arg1: tensor<1x1x1152x1xi8>) -> tensor<1x1x1152x1xi8> {
    %0 = VPU.BitwiseOr(%arg0, %arg1) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
            } : tensor<1x1x1152x1xi8>, tensor<1x1x1152x1xi8> -> tensor<1x1x1152x1xi8>

    return %0 : tensor<1x1x1152x1xi8>

    // CHECK:        [[IN0:%.+]] = VPU.Copy([[INPUT0]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1152x1xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 288, 1], [1, 1, 288, 1], [1, 1, 288, 1], [1, 1, 288, 1]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 288, 0], [0, 0, 576, 0], [0, 0, 864, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 288, 1], [1, 1, 288, 1], [1, 1, 288, 1], [1, 1, 288, 1]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 288, 0], [0, 0, 576, 0], [0, 0, 864, 0]]}>
    // CHECK:        [[IN1:%.+]] = VPU.Copy([[INPUT1]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1152x1xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 288, 1], [1, 1, 288, 1], [1, 1, 288, 1], [1, 1, 288, 1]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 288, 0], [0, 0, 576, 0], [0, 0, 864, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 288, 1], [1, 1, 288, 1], [1, 1, 288, 1], [1, 1, 288, 1]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 288, 0], [0, 0, 576, 0], [0, 0, 864, 0]]}>

    // CHECK:        [[BITWISEOR:%.+]] = VPU.BitwiseOr([[IN0]], [[IN1]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1152x1xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 288, 1], [1, 1, 288, 1], [1, 1, 288, 1], [1, 1, 288, 1]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 288, 0], [0, 0, 576, 0], [0, 0, 864, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 288, 1], [1, 1, 288, 1], [1, 1, 288, 1], [1, 1, 288, 1]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 288, 0], [0, 0, 576, 0], [0, 0, 864, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[BITWISEOR]]
    // CHECK-SAME:                       -> tensor<1x1x1152x1xi8>

    // CHECK:        return [[OUT]] : tensor<1x1x1152x1xi8>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @BitwiseOrSwWithSOK
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x1152x1x1xi8>, [[INPUT1:%.+]]: tensor<1x1152x1x1xi8>
func.func @BitwiseOrSwWithSOK(%arg0: tensor<1x1152x1x1xi8>, %arg1: tensor<1x1152x1x1xi8>) -> tensor<1x1152x1x1xi8> {
    %0 = VPU.BitwiseOr(%arg0, %arg1) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
            } : tensor<1x1152x1x1xi8>, tensor<1x1152x1x1xi8> -> tensor<1x1152x1x1xi8>

    return %0 : tensor<1x1152x1x1xi8>

    // CHECK:        [[IN0:%.+]] = VPU.Copy([[INPUT0]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1152x1x1xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 288, 0, 0], [0, 576, 0, 0], [0, 864, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 288, 0, 0], [0, 576, 0, 0], [0, 864, 0, 0]]}>
    // CHECK:        [[IN1:%.+]] = VPU.Copy([[INPUT1]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1152x1x1xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 288, 0, 0], [0, 576, 0, 0], [0, 864, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 288, 0, 0], [0, 576, 0, 0], [0, 864, 0, 0]]}>

    // CHECK:        [[BITWISEOR:%.+]] = VPU.BitwiseOr([[IN0]], [[IN1]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1152x1x1xi8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 288, 0, 0], [0, 576, 0, 0], [0, 864, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 288, 0, 0], [0, 576, 0, 0], [0, 864, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[BITWISEOR]]
    // CHECK-SAME:                       -> tensor<1x1152x1x1xi8>

    // CHECK:        return [[OUT]] : tensor<1x1152x1x1xi8>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @BitwiseOrSwWithClustering
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x1x1x513xi8>, [[INPUT1:%.+]]: tensor<1x1x1x513xi8>
func.func @BitwiseOrSwWithClustering(%arg0: tensor<1x1x1x513xi8>, %arg1: tensor<1x1x1x513xi8>) -> tensor<1x1x1x513xi8> {
    %0 = VPU.BitwiseOr(%arg0, %arg1) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
            } : tensor<1x1x1x513xi8>, tensor<1x1x1x513xi8> -> tensor<1x1x1x513xi8>

    return %0 : tensor<1x1x1x513xi8>

    // CHECK:        [[IN0:%.+]] = VPU.Copy([[INPUT0]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xi8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:        [[IN1:%.+]] = VPU.Copy([[INPUT1]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xi8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[BITWISEOR:%.+]] = VPU.BitwiseOr([[IN0]], [[IN1]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xi8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[BITWISEOR]]
    // CHECK-SAME:                       -> tensor<1x1x1x513xi8>

    // CHECK:        return [[OUT]] : tensor<1x1x1x513xi8>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @NotEqualSWSOHTileNotAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x44x44xf16>
func.func @NotEqualSWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x44x44xf16>) -> tensor<1x32x44x44xi8> {
    %0 = VPU.NotEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x44x44xf16> -> tensor<1x32x44x44xi8>

    return %0 : tensor<1x32x44x44xi8>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[EQUAL:%.+]] = VPU.NotEqual([[INPUT0]],
    //CHECK:                                  [[INPUT1]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[EQUAL]])

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xi8>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @NegativeSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @NegativeSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Negative(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[NEGATIVE:%.+]] = VPU.Negative([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[NEGATIVE]]
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @NegativeSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @NegativeSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Negative(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[NEGATIVE:%.+]] = VPU.Negative([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[NEGATIVE]]
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @NegativeSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @NegativeSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Negative(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[NEGATIVE:%.+]] = VPU.Negative([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[NEGATIVE]]
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @SoftPlusSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16, {order = #NHWC}>
func.func @SoftPlusSWWithSOH(%arg0: tensor<1x32x44x44xf16, {order = #NHWC}>) -> tensor<1x32x44x44xf16, {order = #NHWC}> {
    %0 = VPU.SoftPlus(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16, {order = #NHWC}> -> tensor<1x32x44x44xf16, {order = #NHWC}>
    return %0 : tensor<1x32x44x44xf16, {order = #NHWC}>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[SOFTPLUS:%.+]] = VPU.SoftPlus([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[SOFTPLUS]]
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @SoftPlusSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @SoftPlusSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.SoftPlus(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[SOFTPLUS:%.+]] = VPU.SoftPlus([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[SOFTPLUS]]
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @SoftPlusSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @SoftPlusSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.SoftPlus(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[SOFTPLUS:%.+]] = VPU.SoftPlus([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[SOFTPLUS]]
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @LessEqualSWSOHTileNotAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x44x44xf16>
func.func @LessEqualSWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x44x44xf16>) -> tensor<1x32x44x44xi8> {
    %0 = VPU.LessEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x44x44xf16> -> tensor<1x32x44x44xi8>

    return %0 : tensor<1x32x44x44xi8>

    //CHECK:        [[INPUT0:%.+]] = VPU.Copy([[INPUT_0]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[INPUT1:%.+]] = VPU.Copy([[INPUT_1]]
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[EQUAL:%.+]] = VPU.LessEqual([[INPUT0]],
    //CHECK:                                  [[INPUT1]])
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[EQUAL]])

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xi8>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @LogicalNotSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16, {order = #NHWC}>
func.func @LogicalNotSWWithSOH(%arg0: tensor<1x32x44x44xf16, {order = #NHWC}>) -> tensor<1x32x44x44xf16, {order = #NHWC}> {
    %0 = VPU.LogicalNot(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16, {order = #NHWC}> -> tensor<1x32x44x44xf16, {order = #NHWC}>
    return %0 : tensor<1x32x44x44xf16, {order = #NHWC}>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[LOGICALNOT:%.+]] = VPU.LogicalNot([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[LOGICALNOT]]
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @LogicalNotSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @LogicalNotSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.LogicalNot(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[LOGICALNOT:%.+]] = VPU.LogicalNot([[IN]])
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[LOGICALNOT]]
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @LogicalNotSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @LogicalNotSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.LogicalNot(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[LOGICALNOT:%.+]] = VPU.LogicalNot([[IN]]
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[LOGICALNOT]]
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 10.351000019148284:128>
!qElemType1 = !quant.uniform<u8:f16, 33.033453967524508:128>
!qElemType2 = !quant.uniform<u8:f16, 37.162151501225487:128>
!qElemType3 = !quant.uniform<u8:f16, 37.503749234068628:128>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseAddMulticlusterSOHOverlappedDepthConvolution
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x64x64x64x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[ARG1:%.+]]: tensor<1x64x64x64x!qElemType1, {order = #NHWC}>)
func.func @EltwiseAddMulticlusterSOHOverlappedDepthConvolution(%arg0: tensor<1x64x64x64x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x64x64x!qElemType1, {order = #NHWC}>) -> tensor<1x64x128x128x!qElemType2, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
        quant_mult = [27959], quant_shift = [29], quant_post_shift = 0 : i64, in1_quant_mult = [5299], in2_quant_mult = [16913], fp_prelu_alpha = 1.000000e+00 : f64>}
        -> tensor<1x64x64x64x!qElemType3, {order = #NHWC}>

    %1 = VPU.StorageElementTable {dataElemType = !qElemType3, dataShape = [1, 64, 64, 64],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>, seDepth = 1 : i64, seSize = [64]}
        -> tensor<1x1x130x130xi32, {order = #NHWC}>
    %cst_220 = const.Declare tensor<1x64x130x130xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1> : tensor<1x64x130x130xi8>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.CastElemType<i1>]
    %2 = VPU.GroupSparseTensor(%0, %cst_220, %1) {
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>}
        -> !VPU.SparseTensor<data=tensor<1x64x64x64x!qElemType3, {order = #NHWC}>,
        sparsity_map=tensor<1x64x130x130xi1, {order = #NHWC}>,
        storage_element_table=tensor<1x1x130x130xi32, {order = #NHWC}>,
        #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>>

    %cst_16 = const.Declare tensor<64x16x1x1x!quant.uniform<u8:f16, 6.250000e-02>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.000000e+00> : tensor<64x1x3x3xf32>, [#const.CastElemType<!quant.uniform<u8:f16, 6.250000e-02>>, #const.Reshape<[64, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
    %3 = VPU.NCE.DepthConvolution(%2, %cst_16) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 1, 3, 3], strides = [1, 1]}
        -> tensor<1x64x128x128x!qElemType2, {order = #NHWC}>
    return %3: tensor<1x64x128x128x!qElemType2, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.+]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x64x64x64x!qElemType, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x64x64x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>
// CHECK:               [[IN_CP1:%.+]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x64x64x64x!qElemType1, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x64x64x!qElemType1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>
// CHECK:               [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[IN_CP0]], [[IN_CP1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x64x64x!qElemType3, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 12, 64], [1, 64, 13, 64], [1, 64, 12, 64], [1, 64, 12, 64], [1, 64, 12, 64], [1, 64, 11, 64]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 21, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 53, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.Copy([[ELTWISE]]) : !VPU.DistributedTensor<1x64x64x64x!qElemType3, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 12, 64], [1, 64, 13, 64], [1, 64, 12, 64], [1, 64, 12, 64], [1, 64, 12, 64], [1, 64, 11, 64]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 21, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 53, 0]]}>
// CHECK-SAME{LITERAL}:                                    -> tensor<1x64x64x64x!qElemType3, {order = #NHWC}>
// CHECK:               [[GROUP_ST:%.+]] = VPU.GroupSparseTensor([[OUT_CP]]
// CHECK:               [[IN_DEPTH_CONV:%.+]] = VPU.Copy([[GROUP_ST]]
// CHECK:               [[DEPTH_CONV:%.+]] = VPU.NCE.DepthConvolution([[IN_DEPTH_CONV]]
// CHECK:               [[OUT_DEPTH_CONV:%.+]] = VPU.Copy([[DEPTH_CONV]]
// CHECK:               return [[OUT_DEPTH_CONV]] : tensor<1x64x128x128x!qElemType2, {order = #NHWC}>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 10.351000019148284:128>
!qElemType1 = !quant.uniform<u8:f16, 33.033453967524508:128>
!qElemType2 = !quant.uniform<u8:f16, 37.162151501225487:128>
!qElemType3 = !quant.uniform<u8:f16, 37.503749234068628:128>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseAddMulticlusterSOHOverlappedConvolution
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x64x64x64x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[ARG1:%.+]]: tensor<1x64x64x64x!qElemType1, {order = #NHWC}>)
func.func @EltwiseAddMulticlusterSOHOverlappedConvolution(%arg0: tensor<1x64x64x64x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x64x64x!qElemType1, {order = #NHWC}>) -> tensor<1x64x130x130x!qElemType2, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
        quant_mult = [27959], quant_shift = [29], quant_post_shift = 0 : i64, in1_quant_mult = [5299], in2_quant_mult = [16913], fp_prelu_alpha = 1.000000e+00 : f64>}
        -> tensor<1x64x64x64x!qElemType3, {order = #NHWC}>

    %1 = VPU.StorageElementTable {dataElemType = !qElemType3, dataShape = [1, 64, 64, 64],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>, seDepth = 1 : i64, seSize = [64]}
        -> tensor<1x1x130x130xi32, {order = #NHWC}>
    %cst_220 = const.Declare tensor<1x64x130x130xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1> : tensor<1x64x130x130xi8>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.CastElemType<i1>]
    %2 = VPU.GroupSparseTensor(%0, %cst_220, %1) {
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>}
        -> !VPU.SparseTensor<data=tensor<1x64x64x64x!qElemType3, {order = #NHWC}>,
        sparsity_map=tensor<1x64x130x130xi1, {order = #NHWC}>,
        storage_element_table=tensor<1x1x130x130xi32, {order = #NHWC}>,
        #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>>


    %cst = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<0.200000e+00> : tensor<64x64x1x1xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [64, 64, 1, 1], strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x64x64x64x!qElemType3, {order = #NHWC}>,
        sparsity_map=tensor<1x64x130x130xi1, {order = #NHWC}>,
        storage_element_table=tensor<1x1x130x130xi32, {order = #NHWC}>,
        #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>>, tensor<64x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x130x130x!qElemType2, {order = #NHWC}>
    return %3: tensor<1x64x130x130x!qElemType2, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.+]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x64x64x64x!qElemType, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x64x64x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>
// CHECK:               [[IN_CP1:%.+]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x64x64x64x!qElemType1, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x64x64x!qElemType1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>
// CHECK:               [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[IN_CP0]], [[IN_CP1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x64x64x!qElemType3, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.Copy([[ELTWISE]]) : !VPU.DistributedTensor<1x64x64x64x!qElemType3, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>
// CHECK-SAME{LITERAL}:                                    -> tensor<1x64x64x64x!qElemType3, {order = #NHWC}>
// CHECK:               [[GROUP_ST:%.+]] = VPU.GroupSparseTensor([[OUT_CP]]
// CHECK:               [[IN_CONV:%.+]] = VPU.Copy([[GROUP_ST]]
// CHECK:               [[CONV:%.+]] = VPU.NCE.Convolution([[IN_CONV]]
// CHECK:               [[OUT_CONV:%.+]] = VPU.Copy([[CONV]]
// CHECK:               return [[OUT_CONV]] : tensor<1x64x130x130x!qElemType2, {order = #NHWC}>

}

}

// -----

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @GatherDMA
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x128256x2048xf16>,
// CHECK-SAME:    [[INDICES:%.+]]: tensor<1x1x1024x1xi64>
func.func @GatherDMA(%input: tensor<1x1x128256x2048xf16>, %indices: tensor<1x1x1024x1xi64>) -> tensor<1x1x1024x2048xf16> {

    %gatherDMA = VPU.GatherDMA(%input, %indices) {axis_value = 2 : i64, batch_dims = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>} :
                tensor<1x1x128256x2048xf16>, tensor<1x1x1024x1xi64> -> tensor<1x1x1024x2048xf16>
    return %gatherDMA : tensor<1x1x1024x2048xf16>

    // CHECK:        [[INDICES_COPY:%.+]] = VPU.Copy([[INDICES]]) {out_mem_space = @CMX_NN} : tensor<1x1x1024x1xi64>
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x1x1024x1xi64, #NCHW, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1024, 1]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1024, 1]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[GATHER_DMA:%.+]] = VPU.GatherDMA([[INPUT]], [[INDICES_COPY]]) {axis_value = 2 : i64, batch_dims = 1 : i64} :
    // CHECK-SAME:      tensor<1x1x128256x2048xf16>,
    // CHECK-SAME:      !VPU.DistributedTensor<1x1x1024x1xi64, #NCHW, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1024, 1]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1024, 1]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x1x1024x2048xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 1, 1024, 683], [1, 1, 1024, 683], [1, 1, 1024, 682]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 683], [0, 0, 0, 1366]],
    // CHECK-SAME{LITERAL}:      memory_shapes =  [[1, 1, 1024, 683], [1, 1, 1024, 683], [1, 1, 1024, 682]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 683], [0, 0, 0, 1366]]}>

    // CHECK:        [[OUT:%.+]] = VPU.Copy([[GATHER_DMA]]) :
    // CHECK-SAME:      !VPU.DistributedTensor<1x1x1024x2048xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 1, 1024, 683], [1, 1, 1024, 683], [1, 1, 1024, 682]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 683], [0, 0, 0, 1366]],
    // CHECK-SAME{LITERAL}:      memory_shapes =  [[1, 1, 1024, 683], [1, 1, 1024, 683], [1, 1, 1024, 682]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 683], [0, 0, 0, 1366]]}>
    // CHECK-SAME:      -> tensor<1x1x1024x2048xf16>

    // CHECK:        return [[OUT]] : tensor<1x1x1024x2048xf16>
}
}
