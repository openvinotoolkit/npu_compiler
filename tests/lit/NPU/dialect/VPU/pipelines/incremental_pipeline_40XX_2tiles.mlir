//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%  allow-custom-values=true num-of-dpu-groups=2" --incremental-pipeline %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SubgraphIncrementalPipelineCheck
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x16x227x227xf16, {order = #NHWC}>)
func.func @SubgraphIncrementalPipelineCheck(%arg0: tensor<1x16x227x227xf16, {order = #NHWC}>) -> tensor<1x96x111x111xf16, {order = #NHWC}> {
    %weights1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 3 : i64>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [13, 0, 0, 0]>, #const.Reorder<#NCHW>, #const.Reshape<[16, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %weightsTable1 = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.DepthConvolution(%arg0, %weights1, %weightsTable1) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
rawFilterShape = [16, 1, 1, 1],
        strides = [1, 1]
        } -> tensor<1x16x227x227xf16, {order = #NHWC}>

    %weights2 = const.Declare tensor<96x16x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x16x7x7xf16, {order = #NHWC}>
    %weightsTable2 = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>

    %1 = VPU.NCE.Convolution(%0, %weights2, %weightsTable2) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
rawFilterShape = [96, 16, 7, 7],
        strides = [2, 2]
        } : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<96x16x7x7xf16, {order = #NHWC}>, tensor<96x1x1x4xsi32> -> tensor<1x96x111x111xf16, {order = #NHWC}>

    return %1 : tensor<1x96x111x111xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS1:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 3 : i64>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [13, 0, 0, 0]>, #const.Reorder<#NCHW>, #const.Reshape<[16, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:   [[WEIGHTS2:%.+]] = const.Declare tensor<96x16x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x16x7x7xf16, {order = #NHWC}>
    // CHECK-DAG:   [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>

    // CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 16, 76, 227] : tensor<1x16x227x227xf16, {order = #NHWC}> to tensor<1x16x76x227xf16, {order = #NHWC}>
    // CHECK:       [[COPY_INPUT_0:%.+]] = VPU.Copy([[SLICE_0]]) {out_mem_space = @CMX_NN} : tensor<1x16x76x227xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x76x227xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 38, 227], [1, 16, 38, 227]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 38, 227], [1, 16, 38, 227]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]}>
    // CHECK:       [[COPY_WEIGHTS_0:%.+]] = VPU.Copy([[WEIGHTS1]]) {out_mem_space = @CMX_NN} : tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_TBL_0:%.+]] = VPU.Copy([[WEIGHTS_TABLE1]]) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32>
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[DWCONV_0:%.+]] = VPU.NCE.DepthConvolution([[COPY_INPUT_0]], [[COPY_WEIGHTS_0]], [[COPY_WEIGHTS_TBL_0]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:           rawFilterShape = [16, 1, 1, 1], strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x76x227xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 38, 227], [1, 16, 38, 227]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 38, 227], [1, 16, 38, 227]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]}>

    // CHECK:       [[COPY_OUT_0:%.+]] = VPU.Copy([[DWCONV_0]]) : !VPU.DistributedTensor<1x16x76x227xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 38, 227], [1, 16, 38, 227]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 38, 227], [1, 16, 38, 227]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]}>
    // CHECK-SAME:      -> tensor<1x16x76x227xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 76, 0] [1, 16, 76, 227] : tensor<1x16x227x227xf16, {order = #NHWC}> to tensor<1x16x76x227xf16, {order = #NHWC}>
    // CHECK:       [[COPY_INPUT_1:%.+]] = VPU.Copy([[SLICE_1]]) {out_mem_space = @CMX_NN} : tensor<1x16x76x227xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x76x227xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 38, 227], [1, 16, 38, 227]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 38, 227], [1, 16, 38, 227]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_1:%.+]] = VPU.Copy([[WEIGHTS1]]) {out_mem_space = @CMX_NN} : tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_TBL_1:%.+]] = VPU.Copy([[WEIGHTS_TABLE1]]) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32>
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[DWCONV_1:%.+]] = VPU.NCE.DepthConvolution([[COPY_INPUT_1]], [[COPY_WEIGHTS_1]], [[COPY_WEIGHTS_TBL_1]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:           rawFilterShape = [16, 1, 1, 1],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x16x76x227xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 16, 38, 227], [1, 16, 38, 227]]
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 16, 38, 227], [1, 16, 38, 227]]
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]}>

    // CHECK:       [[COPY_OUT_1:%.+]] = VPU.Copy([[DWCONV_1]]) : !VPU.DistributedTensor<1x16x76x227xf16, #NHWC, @CMX_NN,
     // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
     // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 38, 227], [1, 16, 38, 227]],
     // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]],
     // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 38, 227], [1, 16, 38, 227]],
     // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]}>
     // CHECK-SAME:     -> tensor<1x16x76x227xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_2:%.+]] = VPU.Slice [[ARG0]] [0, 0, 152, 0] [1, 16, 75, 227] : tensor<1x16x227x227xf16, {order = #NHWC}> to tensor<1x16x75x227xf16, {order = #NHWC}>
    // CHECK:       [[COPY_INPUT_2:%.+]] = VPU.Copy([[SLICE_2]]) {out_mem_space = @CMX_NN} : tensor<1x16x75x227xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x75x227xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 38, 227], [1, 16, 37, 227]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 38, 227], [1, 16, 37, 227]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_2:%.+]] = VPU.Copy([[WEIGHTS1]]) {out_mem_space = @CMX_NN} : tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_TBL_2:%.+]] = VPU.Copy([[WEIGHTS_TABLE1]]) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32>
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[DWCONV_2:%.+]] = VPU.NCE.DepthConvolution([[COPY_INPUT_2]], [[COPY_WEIGHTS_2]], [[COPY_WEIGHTS_TBL_2]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:           rawFilterShape = [16, 1, 1, 1],
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x16x75x227xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 16, 38, 227], [1, 16, 37, 227]]
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 16, 38, 227], [1, 16, 37, 227]]
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]}>
    // CHECK:       [[COPY_OUT_2:%.+]] = VPU.Copy([[DWCONV_2]]) : !VPU.DistributedTensor<1x16x75x227xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 38, 227], [1, 16, 37, 227]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 38, 227], [1, 16, 37, 227]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]}>
    // CHECK-SAME:      -> tensor<1x16x75x227xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT0:%.+]] = VPU.Concat([[COPY_OUT_0]], [[COPY_OUT_1]], [[COPY_OUT_2]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 0, 76, 0]
    // CHECK-SAME:          [0, 0, 152, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  } : tensor<1x16x76x227xf16, {order = #NHWC}>, tensor<1x16x76x227xf16, {order = #NHWC}>, tensor<1x16x75x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_CONV_0:%.+]] = VPU.Slice [[CONCAT0]] [0, 0, 0, 0] [1, 16, 79, 227] : tensor<1x16x227x227xf16, {order = #NHWC}> to tensor<1x16x79x227xf16, {order = #NHWC}>

    // CHECK:       [[COPY_CONV_INPUT_0:%.+]] = VPU.Copy([[SLICE_CONV_0]]) {out_mem_space = @CMX_NN} : tensor<1x16x79x227xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x79x227xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 16, 40, 227], [1, 16, 39, 227]]
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 40, 0]]
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 16, 43, 227], [1, 16, 41, 227]]
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]}>

    // CHECK:       [[COPY_CONV_WEIGHTS_0:%.+]] = VPU.Copy([[WEIGHTS2]]) {out_mem_space = @CMX_NN} : tensor<96x16x7x7xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<96x16x7x7xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[96, 16, 7, 7], [96, 16, 7, 7]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[96, 16, 7, 7], [96, 16, 7, 7]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_CONV_WEIGHTS_TBL_0:%.+]] = VPU.Copy([[WEIGHTS_TABLE2]]) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32>
    // CHECK-SAME:      -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[CONV_0:%.+]] = VPU.NCE.Convolution([[COPY_CONV_INPUT_0]], [[COPY_CONV_WEIGHTS_0]], [[COPY_CONV_WEIGHTS_TBL_0]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:           rawFilterShape = [96, 16, 7, 7], strides = [2, 2]}
    // CHECK-SAME:    -> !VPU.DistributedTensor<1x96x37x111xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 19, 111], [1, 96, 18, 111]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 19, 111], [1, 96, 18, 111]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]]}>

    // CHECK:       [[COPY_CONV_OUT_0:%.+]] = VPU.Copy([[CONV_0]]) : !VPU.DistributedTensor<1x96x37x111xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 19, 111], [1, 96, 18, 111]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 19, 111], [1, 96, 18, 111]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]]}>
    // CHECK-SAME:    -> tensor<1x96x37x111xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_CONV_1:%.+]] = VPU.Slice [[CONCAT0]] [0, 0, 74, 0] [1, 16, 79, 227] : tensor<1x16x227x227xf16, {order = #NHWC}> to tensor<1x16x79x227xf16, {order = #NHWC}>

    // CHECK:       [[COPY_CONV_INPUT_1:%.+]] = VPU.Copy([[SLICE_CONV_1]]) {out_mem_space = @CMX_NN} : tensor<1x16x79x227xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x79x227xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 40, 227], [1, 16, 39, 227]]
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 40, 0]]
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 43, 227], [1, 16, 41, 227]]
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]}>

    // CHECK:       [[COPY_CONV_WEIGHTS_1:%.+]] = VPU.Copy([[WEIGHTS2]]) {out_mem_space = @CMX_NN} : tensor<96x16x7x7xf16, {order = #NHWC}>
    // CHECK-SAME:    -> !VPU.DistributedTensor<96x16x7x7xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[96, 16, 7, 7], [96, 16, 7, 7]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[96, 16, 7, 7], [96, 16, 7, 7]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_CONV_WEIGHTS_TBL_1:%.+]] = VPU.Copy([[WEIGHTS_TABLE2]]) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32>
    // CHECK-SAME:     -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[COPY_CONV_INPUT_1]], [[COPY_CONV_WEIGHTS_1]], [[COPY_CONV_WEIGHTS_TBL_1]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:           rawFilterShape = [96, 16, 7, 7], strides = [2, 2]}
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x96x37x111xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 19, 111], [1, 96, 18, 111]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 19, 111], [1, 96, 18, 111]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]]}>

    // CHECK:       [[COPY_CONV_OUT_1:%.+]] = VPU.Copy([[CONV_1]])
   // CHECK-SAME:     : !VPU.DistributedTensor<1x96x37x111xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 19, 111], [1, 96, 18, 111]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 19, 111], [1, 96, 18, 111]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]]}>
    // CHECK-SAME:          -> tensor<1x96x37x111xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_CONV_2:%.+]] = VPU.Slice [[CONCAT0]] [0, 0, 148, 0] [1, 16, 79, 227] : tensor<1x16x227x227xf16, {order = #NHWC}> to tensor<1x16x79x227xf16, {order = #NHWC}>

    // CHECK:       [[COPY_CONV_INPUT_2:%.+]] = VPU.Copy([[SLICE_CONV_2]]) {out_mem_space = @CMX_NN} : tensor<1x16x79x227xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x79x227xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 40, 227], [1, 16, 39, 227]]
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 40, 0]]
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 43, 227], [1, 16, 41, 227]]
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0]]}>

    // CHECK:       [[COPY_CONV_WEIGHTS_2:%.+]] = VPU.Copy([[WEIGHTS2]]) {out_mem_space = @CMX_NN} : tensor<96x16x7x7xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<96x16x7x7xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[96, 16, 7, 7], [96, 16, 7, 7]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[96, 16, 7, 7], [96, 16, 7, 7]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_CONV_WEIGHTS_TBL_2:%.+]] = VPU.Copy([[WEIGHTS_TABLE2]]) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32>
    // CHECK-SAME:      -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[CONV_2:%.+]] = VPU.NCE.Convolution([[COPY_CONV_INPUT_2]], [[COPY_CONV_WEIGHTS_2]], [[COPY_CONV_WEIGHTS_TBL_2]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:          rawFilterShape = [96, 16, 7, 7]
    // CHECK-SAME:          strides = [2, 2]
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x96x37x111xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 19, 111], [1, 96, 18, 111]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 19, 111], [1, 96, 18, 111]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]]}>

    // CHECK:       [[COPY_CONV_OUT_2:%.+]] = VPU.Copy([[CONV_2]])
    // CHECK-SAME:     : !VPU.DistributedTensor<1x96x37x111xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 19, 111], [1, 96, 18, 111]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 19, 111], [1, 96, 18, 111]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]]}>
    // CHECK-SAME:          -> tensor<1x96x37x111xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT1:%.+]] = VPU.Concat([[COPY_CONV_OUT_0:%.+]], [[COPY_CONV_OUT_1:%.+]], [[COPY_CONV_OUT_2:%.+]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 0, 37, 0],
    // CHECK-SAME:          [0, 0, 74, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  } : tensor<1x96x37x111xf16, {order = #NHWC}>, tensor<1x96x37x111xf16, {order = #NHWC}>, tensor<1x96x37x111xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x96x111x111xf16, {order = #NHWC}>

    // CHECK:   return [[CONCAT1]] : tensor<1x96x111x111xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {
    0.014485318987977272:128,0.41047749238855696:128,0.47723535275926776:128,0.13488349166570926:128,0.0084370542975033026:128,0.14474457385493258:128,
    0.010526879628499349:128,0.31635287415747548:128,0.088105411155551094:128,0.007855018681170894:128,0.014248536147323309:128,0.086256950976801847:128,
    0.12526541691200405:128,0.53939514160156254:128,0.04168558494717467:128,0.16238974028942632:128,0.24466101702521828:128,0.69573235605277273:128,
    0.27111098345588236:128,0.35941073847751992:128,0.021734635970171761:128,0.78550184661266853:128,0.1631712745217716:128,0.045290417764701094:128,
    0.007855018681170894:128,0.12079764721440334:128,0.60792906518075984:128,0.10752303927552466:128,0.007855018681170894:128,0.13565839879653033:128,
    0.042179580763274549:128,0.23992203357172948:128}>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PerAxisQuantizedDWConvIncrementalPipelineCheck
func.func @PerAxisQuantizedDWConvIncrementalPipelineCheck(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_522 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_521 = const.Declare tensor<32x1x1x4xsi32> = dense<0> : tensor<32x1x1x4xsi32>

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_522, %cst_521) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
rawFilterShape = [32, 1, 1, 1], strides = [1, 1]
        } -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<0> : tensor<32x1x1x4xsi32>

    // CHECK:       [[SLICE_0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 52, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x32x52x256xf16, {order = #NHWC}>
    // CHECK:       [[COPY_INPUT_0:%.+]] =  VPU.Copy([[SLICE_0]]) {out_mem_space = @CMX_NN} : tensor<1x32x52x256xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x32x52x256xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 26, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 26, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_0:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_TBL_0:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[DW_CONV_0:%.+]] = VPU.NCE.DepthConvolution([[COPY_INPUT_0]], [[COPY_WEIGHTS_0]], [[COPY_WEIGHTS_TBL_0]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:            rawFilterShape = [32, 1, 1, 1],
    //CHECK-SAME:            strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x32x52x256x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 26, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 26, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>

    // CHECK:       [[COPY_OUTPUT_0:%.+]] = VPU.Copy([[DW_CONV_0]])
    // CHECK-SAME:      : !VPU.DistributedTensor<1x32x52x256x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 26, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 26, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>

    // CHECK:       [[SLICE_1:%.+]] = VPU.Slice %arg0 [0, 0, 52, 0] [1, 32, 51, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x32x51x256xf16, {order = #NHWC}>
    // CHECK:       [[COPY_INPUT_1:%.+]] = VPU.Copy([[SLICE_1]]) {out_mem_space = @CMX_NN} : tensor<1x32x51x256xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_1:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_TBL_1:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[DW_CONV_1:%.+]] = VPU.NCE.DepthConvolution([[COPY_INPUT_1]], [[COPY_WEIGHTS_1]], [[COPY_WEIGHTS_TBL_1]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           rawFilterShape = [32, 1, 1, 1],
    //CHECK-SAME:           strides = [1, 1]
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x32x51x256x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>

    // CHECK:       [[COPY_OUTPUT_1:%.+]] = VPU.Copy([[DW_CONV_1]])
    // CHECK-SAME:      : !VPU.DistributedTensor<1x32x51x256x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>
    // CHECK-SAME:      -> tensor<1x32x51x256x!qElemType, {order = #NHWC}>

    // CHECK:       [[SLICE_2:%.+]] = VPU.Slice %arg0 [0, 0, 103, 0] [1, 32, 51, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x32x51x256xf16, {order = #NHWC}>
    // CHECK:       [[COPY_INPUT_2:%.+]] = VPU.Copy([[SLICE_2]]) {out_mem_space = @CMX_NN} : tensor<1x32x51x256xf16, {order = #NHWC}>
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_2:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_TBL_2:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[DW_CONV_2:%.+]] = VPU.NCE.DepthConvolution([[COPY_INPUT_2]], [[COPY_WEIGHTS_2]], [[COPY_WEIGHTS_TBL_2]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:             rawFilterShape = [32, 1, 1, 1],
    // CHECK-SAME:             strides = [1, 1]
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x32x51x256x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>

    // CHECK:       [[COPY_OUTPUT_2:%.+]] = VPU.Copy([[DW_CONV_2]]
    // CHECK-SAME:      :  !VPU.DistributedTensor<1x32x51x256x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>
    // CHECK-SAME:      -> tensor<1x32x51x256x!qElemType, {order = #NHWC}>

    // CHECK:       [[SLICE_3:%.+]] = VPU.Slice %arg0 [0, 0, 154, 0] [1, 32, 51, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x32x51x256xf16, {order = #NHWC}>
    // CHECK:       [[COPY_INPUT_3:%.+]] = VPU.Copy([[SLICE_3]]) {out_mem_space = @CMX_NN} : tensor<1x32x51x256xf16, {order = #NHWC}>
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_3:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_TBL_3:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[DW_CONV_3:%.+]] = VPU.NCE.DepthConvolution([[COPY_INPUT_3]], [[COPY_WEIGHTS_3]], [[COPY_WEIGHTS_TBL_3]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:              rawFilterShape = [32, 1, 1, 1],
    // CHECK-SAME:              strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x32x51x256x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>

    // CHECK:       [[COPY_OUTPUT_3:%.+]] = VPU.Copy([[DW_CONV_3]])
    // CHECK-SAME:      : !VPU.DistributedTensor<1x32x51x256x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>
    // CHECK-SAME:      -> tensor<1x32x51x256x!qElemType, {order = #NHWC}>

    // CHECK:       [[SLICE_4:%.+]] = VPU.Slice %arg0 [0, 0, 205, 0] [1, 32, 51, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x32x51x256xf16, {order = #NHWC}>
    // CHECK:       [[COPY_INPUT_4:%.+]] = VPU.Copy([[SLICE_4]]) {out_mem_space = @CMX_NN} : tensor<1x32x51x256xf16, {order = #NHWC}>
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_4:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_WEIGHTS_TBL_4:%.+]] = VPU.Copy([[WEIGHTS_TABLE]]) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[DW_CONV_4:%.+]] = VPU.NCE.DepthConvolution([[COPY_INPUT_4]], [[COPY_WEIGHTS_4]], [[COPY_WEIGHTS_TBL_4]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:              rawFilterShape = [32, 1, 1, 1],
    // CHECK-SAME:              strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x32x51x256x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>

    // CHECK:       [[COPY_OUTPUT_4:%.+]] = VPU.Copy([[DW_CONV_4]])
    // CHECK-SAME:      : !VPU.DistributedTensor<1x32x51x256x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 26, 256], [1, 32, 25, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 26, 0]]}>
    // CHECK-SAME:      -> tensor<1x32x51x256x!qElemType, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[COPY_OUTPUT_0]], [[COPY_OUTPUT_1]], [[COPY_OUTPUT_2]], [[COPY_OUTPUT_3]], [[COPY_OUTPUT_4]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 0, 52, 0],
    // CHECK-SAME:          [0, 0, 103, 0],
    // CHECK-SAME:          [0, 0, 154, 0],
    // CHECK-SAME:          [0, 0, 205, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  } : tensor<1x32x52x256x!qElemType, {order = #NHWC}>, tensor<1x32x51x256x!qElemType, {order = #NHWC}>, tensor<1x32x51x256x!qElemType, {order = #NHWC}>, tensor<1x32x51x256x!qElemType, {order = #NHWC}>, tensor<1x32x51x256x!qElemType, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    // CHECK:   return [[CONCAT]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}
