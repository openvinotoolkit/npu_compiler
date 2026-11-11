//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%  allow-custom-values=true" --incremental-pipeline %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvIncrementalPipelineSplitWeightTableFormat
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @ConvIncrementalPipelineSplitWeightTableFormat(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %scale_table = const.Declare tensor<80x1x1x1xf32> = dense<1.0> : tensor<80x1x1x1xf32>
    %bias_table = const.Declare tensor<80x1x1x1xf32> = dense<2.0> : tensor<80x1x1x1xf32>
    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [80, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad =  #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>,
        ppe = #VPU.PPEStub<>, 
        strides = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
    } : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}>, tensor<80x1x1x1xf32>, tensor<80x1x1x1xf32> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>


    //CHECK: [[SCALE_TABLE:%.+]] = const.Declare tensor<80x1x1x1xf32> = dense<1.000000e+00> : tensor<80x1x1x1xf32>
    //CHECK: [[BIAS_TABLE:%.+]] = const.Declare tensor<80x1x1x1xf32> = dense<2.000000e+00> : tensor<80x1x1x1xf32>
    //CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK: [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 8, 28], [1, 64, 9, 28], [1, 64, 9, 28], [1, 64, 8, 28]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 13, 0], [0, 0, 20, 0]]}>

    //CHECK: [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:       -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK: [[SCALE_TABLE_COPY:%.+]] = VPU.Copy([[SCALE_TABLE]]) {out_mem_space = @CMX_NN} : tensor<80x1x1x1xf32>
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x1x1x1xf32, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK: [[BIAS_TABLE_COPY:%.+]] = VPU.Copy([[BIAS_TABLE]]) {out_mem_space = @CMX_NN} : tensor<80x1x1x1xf32>
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x1x1x1xf32, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT_COPY]], [[WEIGHTS_COPY]], [[SCALE_TABLE_COPY]], [[BIAS_TABLE_COPY]]) rawFilterShape [80, 64, 3, 3] {
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>, 
    // CHECK-SAME:         strides = [1, 1]}
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0]]}>

    //CHECK: [[CONV_COPY:%.+]] = VPU.Copy([[CONV]]) : !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0]]}>
    //CHECK-SAME:       -> tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK: return [[CONV_COPY]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvIncrementalPipelineWithoutAnyWeightTable
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @ConvIncrementalPipelineWithoutAnyWeightTable(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [80, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad =  #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>,
        ppe = #VPU.PPEStub<>, 
        strides = [1, 1]
    } : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>


    //CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK: [[INPUT_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 8, 28], [1, 64, 9, 28], [1, 64, 9, 28], [1, 64, 8, 28]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 13, 0], [0, 0, 20, 0]]}>

    //CHECK: [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:       -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT_COPY]], [[WEIGHTS_COPY]]) rawFilterShape [80, 64, 3, 3] {
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:           ppe = #VPU.PPEStub<>,
    // CHECK-SAME:         strides = [1, 1]}
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0]]}>

    //CHECK: [[CONV_COPY:%.+]] = VPU.Copy([[CONV]]) : !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28], [1, 80, 7, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0]]}>
    //CHECK-SAME:       -> tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK: return [[CONV_COPY]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}
