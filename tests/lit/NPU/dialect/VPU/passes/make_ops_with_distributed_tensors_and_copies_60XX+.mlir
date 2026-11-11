//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --make-ops-with-distributed-tensor="enable-explicit-distributed-attr=true" --make-distributed-copies %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToDistributedOpSOHOverlapped
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>
func.func @ConvToDistributedOpSOHOverlapped(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %scale_table = const.Declare tensor<80x1x1x1xf32> = dense<1.0> : tensor<80x1x1x1xf32>
    %bias_table = const.Declare tensor<80x1x1x1xf32> = dense<2.0> : tensor<80x1x1x1xf32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %scale_table, %bias_table) rawFilterShape [80, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,

        strides = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>}
      : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}>, tensor<80x1x1x1xf32>, tensor<80x1x1x1xf32> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[SCALETABLE:%.+]] = const.Declare tensor<80x1x1x1xf32> = dense<1.000000e+00> : tensor<80x1x1x1xf32>
    //CHECK:        [[BIASTABLE:%.+]] = const.Declare tensor<80x1x1x1xf32> = dense<2.000000e+00> : tensor<80x1x1x1xf32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]])
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

    //CHECK:        [[SCALETABLE_CMX:%.+]] = VPU.Copy([[SCALETABLE]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<80x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[BIASTABLE_CMX:%.+]] = VPU.Copy([[BIASTABLE]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<80x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_CMX]],
    //CHECK-SAME:             [[SCALETABLE_CMX]],
    //CHECK-SAME:             [[BIASTABLE_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToDistributedOpSOK
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x28x28xf16, {order = #NHWC}>
func.func @ConvToDistributedOpSOK(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x96x28x28xf16, {order = #NHWC}> {
    %scale_table = const.Declare tensor<96x1x1x1xf32> = dense<1.0> : tensor<96x1x1x1xf32>
    %bias_table = const.Declare tensor<96x1x1x1xf32> = dense<2.0> : tensor<96x1x1x1xf32>
    %cst_0 = const.Declare tensor<96x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %scale_table, %bias_table) rawFilterShape [96, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

        strides = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>}
      : tensor<1x128x28x28xf16, {order = #NHWC}>, tensor<96x128x1x1xf16, {order = #NHWC}>, tensor<96x1x1x1xf32>, tensor<96x1x1x1xf32> -> tensor<1x96x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x96x28x28xf16, {order = #NHWC}>

    //CHECK:        [[SCALETABLE:%.+]] = const.Declare tensor<96x1x1x1xf32> = dense<1.000000e+00> : tensor<96x1x1x1xf32>
    //CHECK:        [[BIASTABLE:%.+]] = const.Declare tensor<96x1x1x1xf32> = dense<2.000000e+00> : tensor<96x1x1x1xf32>
    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<96x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]]

    //CHECK:        [[SCALETABLE_CMX:%.+]] = VPU.Copy([[SCALETABLE]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x1xf32, #NCHW, @CMX_NN
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]]

    //CHECK:        [[BIASTABLE_CMX:%.+]] = VPU.Copy([[BIASTABLE]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x1xf32, #NCHW, @CMX_NN
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]]

    //CHECK:        [[OUT_CMX:%.+]] = VPU.NCE.Convolution(
    //CHECK-SAME:             [[INPUT_CMX]],
    //CHECK-SAME:             [[WEIGHTS_CMX]],
    //CHECK-SAME:             [[SCALETABLE_CMX]],
    //CHECK-SAME:             [[BIASTABLE_CMX]])
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[OUT_CMX]]

    //CHECK:        return [[OUT]] : tensor<1x96x28x28xf16, {order = #NHWC}>
}

}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MatMulToDistributedOpSOG
// CHECK-SAME:    [[FUNC_INPUT1:%.+]]:  tensor<4x1x32x64x1xf16, {order = #GNHWC}>
// CHECK-SAME:    [[FUNC_INPUT2:%.+]]:  tensor<4x64x32x1x1xf16, {order = #GNHWC}>
func.func @MatMulToDistributedOpSOG4Clusters(%arg0:  tensor<4x1x32x64x1xf16, {order = #GNHWC}>, %arg1: tensor<4x64x32x1x1xf16, {order = #GNHWC}>)
    -> tensor<4x1x64x64x1xf16, {order = #GNHWC}> {
    %scale_table = const.Declare tensor<1x64x1x1x1xf32> = dense<1.0> : tensor<1x64x1x1x1xf32>
    // CHECK:   [[IN_SCALE_TABLE_CONST:%.+]] = const.Declare tensor<1x64x1x1x1xf32> = dense<1.000000e+00> : tensor<1x64x1x1x1xf32>
    %bias_table = const.Declare tensor<1x64x1x1x1xf32> = dense<2.0> : tensor<1x64x1x1x1xf32>
    // CHECK:   [[IN_BIAS_TABLE_CONST:%.+]] = const.Declare tensor<1x64x1x1x1xf32> = dense<2.000000e+00> : tensor<1x64x1x1x1xf32>

    %0 = VPU.NCE.MatMul(%arg0, %arg1, %scale_table, %bias_table) rawFilterShape [4, 1, 64, 32, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
         strides = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
    } -> tensor<4x1x64x64x1xf16, {order = #GNHWC}>

    return %0 : tensor<4x1x64x64x1xf16, {order = #GNHWC}>

    // CHECK:               [[IN1:%.+]] = VPU.Copy([[FUNC_INPUT1]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<4x1x32x64x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]]}>

    // CHECK:               [[IN2:%.+]] = VPU.Copy([[FUNC_INPUT2]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<4x64x32x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]]}>

    // CHECK:               [[IN_SCALE_TABLE:%.+]] = VPU.Copy([[IN_SCALE_TABLE_CONST]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x64x1x1x1xf32, #NCDHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]}>

    // CHECK:               [[IN_BIAS_TABLE:%.+]] = VPU.Copy([[IN_BIAS_TABLE_CONST]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x64x1x1x1xf32, #NCDHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]}>

    // CHECK:               [[MATMUL_OUT:%.+]] = VPU.NCE.MatMul
    // CHECK-SAME:                        [[IN1]]
    // CHECK-SAME:                        [[IN2]]
    // CHECK-SAME:                        [[IN_SCALE_TABLE]]
    // CHECK-SAME:                        [[IN_BIAS_TABLE]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<4x1x64x64x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]]}>

    // CHECK:               [[COPY_OUT:%.+]] = VPU.Copy([[MATMUL_OUT]]

    // CHECK: return [[COPY_OUT]] : tensor<4x1x64x64x1xf16, {order = #GNHWC}>
}

}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MatMulToDistributedOpSOG
// CHECK-SAME:  [[FUNC_INPUT1:%.+]]:  tensor<8x1x32x64x1xf16, {order = #GNHWC}>
// CHECK-SAME:  [[FUNC_INPUT2:%.+]]:  tensor<8x64x32x1x1xf16, {order = #GNHWC}>
func.func @MatMulToDistributedOpSOG6Clusters(%arg0:  tensor<8x1x32x64x1xf16, {order =#GNHWC}>, %arg1: tensor<8x64x32x1x1xf16, {order = #GNHWC}>)
    -> tensor<8x1x64x64x1xf16, {order = #GNHWC}> {
    %scale_table = const.Declare tensor<1x64x1x1x1xf32> = dense<1.0> : tensor<1x64x1x1x1xf32>
    // CHECK:   [[IN_SCALE_TABLE_CONST:%.+]] = const.Declare tensor<1x64x1x1x1xf32> = dense<1.000000e+00> : tensor<1x64x1x1x1xf32>
    %bias_table = const.Declare tensor<1x64x1x1x1xf32> = dense<2.0> : tensor<1x64x1x1x1xf32>
    // CHECK:   [[IN_BIAS_TABLE_CONST:%.+]] = const.Declare tensor<1x64x1x1x1xf32> = dense<2.000000e+00> : tensor<1x64x1x1x1xf32>

    %0 = VPU.NCE.MatMul(%arg0, %arg1, %scale_table, %bias_table) rawFilterShape [8, 1, 64, 32, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
         strides = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
    } -> tensor<8x1x64x64x1xf16, {order = #GNHWC}>

    return %0 : tensor<8x1x64x64x1xf16, {order = #GNHWC}>

    // CHECK:               [[IN1:%.+]] = VPU.Copy([[FUNC_INPUT1]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<8x1x32x64x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[2, 1, 32, 64, 1], [2, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[2, 1, 32, 64, 1], [2, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]]}>

    // CHECK:               [[IN2:%.+]] = VPU.Copy([[FUNC_INPUT2]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<8x64x32x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[2, 64, 32, 1, 1], [2, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[2, 64, 32, 1, 1], [2, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]]}>

    // CHECK:               [[IN_SCALE_TABLE:%.+]] = VPU.Copy([[IN_SCALE_TABLE_CONST]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x64x1x1x1xf32, #NCDHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]}>

    // CHECK:               [[IN_BIAS_TABLE:%.+]] = VPU.Copy([[IN_BIAS_TABLE_CONST]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x64x1x1x1xf32, #NCDHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1], [1, 64, 1, 1, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]}>

    // CHECK:               [[MATMUL_OUT:%.+]] = VPU.NCE.MatMul
    // CHECK-SAME:                        [[IN1]]
    // CHECK-SAME:                        [[IN2]]
    // CHECK-SAME:                        [[IN_SCALE_TABLE]]
    // CHECK-SAME:                        [[IN_BIAS_TABLE]]
    // CHECK-SAME:          -> !VPU.DistributedTensor<8x1x64x64x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[2, 1, 64, 64, 1], [2, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[2, 1, 64, 64, 1], [2, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]]}>

    // CHECK:               [[COPY_OUT:%.+]] = VPU.Copy([[MATMUL_OUT]]

    // CHECK: return [[COPY_OUT]] : tensor<8x1x64x64x1xf16, {order = #GNHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvWithReduceOutputHKSwitch
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x74x4x!qElemType, {order = #NHWC}>
func.func @ConvWithReduceOutputHKSwitch(%arg0: tensor<1x64x74x4x!qElemType, {order = #NHWC}>) -> (tensor<1x592x74x4xf16, {order = #NHWC}>, tensor<1x1x74x4xf16, {order = #NHWC}>) {
        %scale_table = const.Declare tensor<592x1x1x1xf32> = dense<1.000000e+00> : tensor<592x1x1x1xf32>
        %bias_table = const.Declare tensor<592x1x1x1xf32> = dense<0.000000e+00> : tensor<592x1x1x1xf32>
        %weights = const.Declare tensor<592x64x3x3x!qElemType, {order = #NHWC}> = dense<1> : tensor<592x64x3x3xui8>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
        %output, %reduce = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [592, 64, 3, 3] {
                axes_value = [1],
                loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>},
                mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>, weight_zp = 128 : i64, activation_zp = 128 : i64>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
                resultSegmentSizes = array<i32: 1, 1, 0, 0>,
                strides = [1, 1],
                vf_loop_index = 115 : i64,
                vf_loop_tile_index = 0 : i64,
                operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>}
            : tensor<1x64x74x4x!qElemType, {order = #NHWC}>, tensor<592x64x3x3x!qElemType, {order = #NHWC}>, tensor<592x1x1x1xf32>, tensor<592x1x1x1xf32> -> tensor<1x592x74x4xf16, {order = #NHWC}>, tensor<1x1x74x4xf16, {order = #NHWC}>
        return %output, %reduce : tensor<1x592x74x4xf16, {order = #NHWC}>, tensor<1x1x74x4xf16, {order = #NHWC}>

        // CHECK:        [[SCALETABLE:%.+]] = const.Declare tensor<592x1x1x1xf32> = dense<1.000000e+00> : tensor<592x1x1x1xf32>
        // CHECK:        [[BIASTABLE:%.+]] = const.Declare tensor<592x1x1x1xf32> = dense<0.000000e+00> : tensor<592x1x1x1xf32>
        // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<592x64x3x3x!qElemType, {order = #NHWC}> = dense<1> : tensor<592x64x3x3xui8>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

        // CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]])
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x64x74x4x!qElemType, #NHWC, @CMX_NN
        // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 13, 4], [1, 64, 13, 4], [1, 64, 12, 4], [1, 64, 12, 4], [1, 64, 12, 4], [1, 64, 12, 4]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 26, 0], [0, 0, 38, 0], [0, 0, 50, 0], [0, 0, 62, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 14, 4], [1, 64, 15, 4], [1, 64, 14, 4], [1, 64, 14, 4], [1, 64, 14, 4], [1, 64, 13, 4]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 12, 0], [0, 0, 25, 0], [0, 0, 37, 0], [0, 0, 49, 0], [0, 0, 61, 0]]}

        // CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]])
        // CHECK-SAME:       -> !VPU.DistributedTensor<592x64x3x3x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}

        // CHECK:        [[SCALETABLE_CMX:%.+]] = VPU.Copy([[SCALETABLE]])
        // CHECK-SAME:       -> !VPU.DistributedTensor<592x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}

        // CHECK:        [[BIASTABLE_CMX:%.+]] = VPU.Copy([[BIASTABLE]])
        // CHECK-SAME:       -> !VPU.DistributedTensor<592x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}

        // CHECK:        [[OUTPUT_CMX:%.+]], [[REDUCE_CMX:%.+]] = VPU.NCE.Convolution(
        // CHECK-SAME:             [[INPUT_CMX]], [[WEIGHTS_CMX]], [[SCALETABLE_CMX]], [[BIASTABLE_CMX]])
        // CHECK-SAME:       resultSegmentSizes = array<i32: 1, 1, 0, 0>
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x592x74x4xf16, #NHWC, @CMX_NN
        // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 592, 13, 4], [1, 592, 13, 4], [1, 592, 12, 4], [1, 592, 12, 4], [1, 592, 12, 4], [1, 592, 12, 4]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 26, 0], [0, 0, 38, 0], [0, 0, 50, 0], [0, 0, 62, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 592, 13, 4], [1, 592, 13, 4], [1, 592, 12, 4], [1, 592, 12, 4], [1, 592, 12, 4], [1, 592, 12, 4]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 26, 0], [0, 0, 38, 0], [0, 0, 50, 0], [0, 0, 62, 0]]}
        // CHECK-SAME:       !VPU.DistributedTensor<1x1x74x4xf16, #NHWC, @CMX_NN
        // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 13, 4], [1, 1, 13, 4], [1, 1, 12, 4], [1, 1, 12, 4], [1, 1, 12, 4], [1, 1, 12, 4]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 26, 0], [0, 0, 38, 0], [0, 0, 50, 0], [0, 0, 62, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 13, 4], [1, 1, 13, 4], [1, 1, 12, 4], [1, 1, 12, 4], [1, 1, 12, 4], [1, 1, 12, 4]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 26, 0], [0, 0, 38, 0], [0, 0, 50, 0], [0, 0, 62, 0]]}

        // CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[OUTPUT_CMX]])
        // CHECK-SAME:       -> tensor<1x592x74x4xf16, {order = #NHWC}>
        // CHECK:        [[REDUCE:%.+]] = VPU.Copy([[REDUCE_CMX]])
        // CHECK-SAME:       -> tensor<1x1x74x4xf16, {order = #NHWC}>
        // CHECK:        return [[OUTPUT]], [[REDUCE]] : tensor<1x592x74x4xf16, {order = #NHWC}>, tensor<1x1x74x4xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvWithReduceOutputSOH
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x74x4x!qElemType, {order = #NHWC}>
func.func @ConvWithReduceOutputSOH(%arg0: tensor<1x64x74x4x!qElemType, {order = #NHWC}>) -> (tensor<1x592x74x4xf16, {order = #NHWC}>, tensor<1x1x74x4xf16, {order = #NHWC}>) {
        %scale_table = const.Declare tensor<592x1x1x1xf32> = dense<1.000000e+00> : tensor<592x1x1x1xf32>
        %bias_table = const.Declare tensor<592x1x1x1xf32> = dense<0.000000e+00> : tensor<592x1x1x1xf32>
        %weights = const.Declare tensor<592x64x3x3x!qElemType, {order = #NHWC}> = dense<1> : tensor<592x64x3x3xui8>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
        %output, %reduce = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [592, 64, 3, 3] {
                axes_value = [1],
                loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>},
                mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>, weight_zp = 128 : i64, activation_zp = 128 : i64>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
                resultSegmentSizes = array<i32: 1, 1, 0, 0>,
                strides = [1, 1],
                vf_loop_index = 115 : i64,
                vf_loop_tile_index = 0 : i64,
                operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>}
            : tensor<1x64x74x4x!qElemType, {order = #NHWC}>, tensor<592x64x3x3x!qElemType, {order = #NHWC}>, tensor<592x1x1x1xf32>, tensor<592x1x1x1xf32> -> tensor<1x592x74x4xf16, {order = #NHWC}>, tensor<1x1x74x4xf16, {order = #NHWC}>
        return %output, %reduce : tensor<1x592x74x4xf16, {order = #NHWC}>, tensor<1x1x74x4xf16, {order = #NHWC}>

        // CHECK:        [[SCALETABLE:%.+]] = const.Declare tensor<592x1x1x1xf32> = dense<1.000000e+00> : tensor<592x1x1x1xf32>
        // CHECK:        [[BIASTABLE:%.+]] = const.Declare tensor<592x1x1x1xf32> = dense<0.000000e+00> : tensor<592x1x1x1xf32>
        // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<592x64x3x3x!qElemType, {order = #NHWC}> = dense<1> : tensor<592x64x3x3xui8>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

        // CHECK:        [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]])
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x64x74x4x!qElemType, #NHWC, @CMX_NN
        // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 13, 4], [1, 64, 13, 4], [1, 64, 12, 4], [1, 64, 12, 4], [1, 64, 12, 4], [1, 64, 12, 4]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 26, 0], [0, 0, 38, 0], [0, 0, 50, 0], [0, 0, 62, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 14, 4], [1, 64, 15, 4], [1, 64, 14, 4], [1, 64, 14, 4], [1, 64, 14, 4], [1, 64, 13, 4]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 12, 0], [0, 0, 25, 0], [0, 0, 37, 0], [0, 0, 49, 0], [0, 0, 61, 0]]}

        // CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]])
        // CHECK-SAME:       -> !VPU.DistributedTensor<592x64x3x3x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3], [592, 64, 3, 3]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}

        // CHECK:        [[SCALETABLE_CMX:%.+]] = VPU.Copy([[SCALETABLE]])
        // CHECK-SAME:       -> !VPU.DistributedTensor<592x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}

        // CHECK:        [[BIASTABLE_CMX:%.+]] = VPU.Copy([[BIASTABLE]])
        // CHECK-SAME:       -> !VPU.DistributedTensor<592x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1], [592, 1, 1, 1]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}

        // CHECK:        [[OUTPUT_CMX:%.+]], [[REDUCE_CMX:%.+]] = VPU.NCE.Convolution(
        // CHECK-SAME:             [[INPUT_CMX]], [[WEIGHTS_CMX]], [[SCALETABLE_CMX]], [[BIASTABLE_CMX]])
        // CHECK-SAME:       resultSegmentSizes = array<i32: 1, 1, 0, 0>
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x592x74x4xf16, #NHWC, @CMX_NN
        // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 592, 13, 4], [1, 592, 13, 4], [1, 592, 12, 4], [1, 592, 12, 4], [1, 592, 12, 4], [1, 592, 12, 4]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 26, 0], [0, 0, 38, 0], [0, 0, 50, 0], [0, 0, 62, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 592, 13, 4], [1, 592, 13, 4], [1, 592, 12, 4], [1, 592, 12, 4], [1, 592, 12, 4], [1, 592, 12, 4]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 26, 0], [0, 0, 38, 0], [0, 0, 50, 0], [0, 0, 62, 0]]}
        // CHECK-SAME:       !VPU.DistributedTensor<1x1x74x4xf16, #NHWC, @CMX_NN
        // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 13, 4], [1, 1, 13, 4], [1, 1, 12, 4], [1, 1, 12, 4], [1, 1, 12, 4], [1, 1, 12, 4]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 26, 0], [0, 0, 38, 0], [0, 0, 50, 0], [0, 0, 62, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 13, 4], [1, 1, 13, 4], [1, 1, 12, 4], [1, 1, 12, 4], [1, 1, 12, 4], [1, 1, 12, 4]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 26, 0], [0, 0, 38, 0], [0, 0, 50, 0], [0, 0, 62, 0]]}

        // CHECK:        [[OUTPUT:%.+]] = VPU.Copy([[OUTPUT_CMX]])
        // CHECK-SAME:       -> tensor<1x592x74x4xf16, {order = #NHWC}>
        // CHECK:        [[REDUCE:%.+]] = VPU.Copy([[REDUCE_CMX]])
        // CHECK-SAME:       -> tensor<1x1x74x4xf16, {order = #NHWC}>
        // CHECK:        return [[OUTPUT]], [[REDUCE]] : tensor<1x592x74x4xf16, {order = #NHWC}>, tensor<1x1x74x4xf16, {order = #NHWC}>
}

}
