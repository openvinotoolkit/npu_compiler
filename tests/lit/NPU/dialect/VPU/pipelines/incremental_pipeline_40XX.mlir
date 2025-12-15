//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%  allow-custom-values=true" --incremental-pipeline %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxpoolIncrementalPipelineCheck
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x16x1x4xf16, {order = #NHWC}>)
func.func @MaxpoolIncrementalPipelineCheck(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) { kernel_size = [1, 1], pad =  #VPU.Padding<bottom = 0, left = 0, right = 0, top = 0>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x1x4xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>

    //CHECK: [[OP0:%.+]] =  VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x16x1x4xf16, {order = #NHWC}>
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x1x4xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK: [[OP1:%.+]] = VPU.NCE.MaxPool([[OP0]]) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x1x4xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK: [[OP4:%.+]] = VPU.Copy([[OP1]]) : !VPU.DistributedTensor<1x16x1x4xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:       -> tensor<1x16x1x4xf16, {order = #NHWC}>

    //CHECK: return [[OP4]] : tensor<1x16x1x4xf16, {order = #NHWC}>
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvIncrementalPipeline
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @ConvIncrementalPipeline(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad =  #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}>, tensor<80x1x1x4xsi32> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>


    //CHECK: [[CONST:%.+]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK: [[CONST_0:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK: [[OP0:%.+]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]}>

    //CHECK: [[OP1:%.+]] = VPU.Copy([[CONST_0]]) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:       -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK: [[OP2:%.+]] = VPU.Copy([[CONST]]) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32>
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK: [[OP3:%.+]] = VPU.NCE.Convolution([[OP0]], [[OP1]], [[OP2]])
    //CHECK-SAME:          {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:           ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>

    //CHECK: [[OP4:%.+]] = VPU.Copy([[OP3]]) : !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>
    //CHECK-SAME:       -> tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK: return [[OP4]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEPermuteIncrementalPipeline
// CHECK-SAME:        [[INPUT:%.+]]: tensor<1x3x64x64xf16>
func.func @NCEPermuteIncrementalPipeline(%arg0: tensor<1x3x64x64xf16>) -> tensor<1x16x32x32x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x1x1x48xf16>,
                [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %nce_permute = VPU.NCE.Permute(%arg0)
            {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEStub<>}
    -> tensor<1x4x64x64x!qElemType, {order = #NHWC}>

    %compress_conv = VPU.NCE.CompressConvolution(%nce_permute, %weights, %wt)
            {cm_sp_pattern = 1 : i64,
            pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>, rawFilterShape = [16, 1, 3, 3], strides = [2, 2]}
    -> tensor<1x16x32x32x!qElemType, {order = #NHWC}>

    return %compress_conv : tensor<1x16x32x32x!qElemType, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}>
    //CHECK:        [[WT:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    //CHECK:        [[IN_CMX:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN} : tensor<1x3x64x64xf16>
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x3x64x64xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 10, 64], [1, 3, 10, 64]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 10, 64], [1, 3, 10, 64]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>

    //CHECK:        [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[IN_CMX]]) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEStub<>}
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x64x64x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 4, 11, 64], [1, 4, 11, 64], [1, 4, 11, 64], [1, 4, 11, 64], [1, 4, 10, 64], [1, 4, 10, 64]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 4, 13, 64], [1, 4, 14, 64], [1, 4, 13, 64], [1, 4, 12, 64], [1, 4, 11, 64], [1, 4, 10, 64]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>

    //CHECK:        [[COPY_TO_DDR:%.+]] = VPU.Copy([[NCE_PERMUTE]])
    //CHECK-SAME:           -> tensor<1x4x64x64x!qElemType, {order = #NHWC}>

    //CHECK:        [[COPY_TO_CMX:%.+]] = VPU.Copy([[COPY_TO_DDR]]) {out_mem_space = @CMX_NN} : tensor<1x4x64x64x!qElemType, {order = #NHWC}>
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x64x64x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 4, 11, 64], [1, 4, 11, 64], [1, 4, 11, 64], [1, 4, 11, 64], [1, 4, 10, 64], [1, 4, 10, 64]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 4, 13, 64], [1, 4, 14, 64], [1, 4, 13, 64], [1, 4, 12, 64], [1, 4, 11, 64], [1, 4, 10, 64]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>

    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN} : tensor<16x1x1x48x!qElemType, {order = #NHWC}>
    //CHECK-SAME:           -> !VPU.DistributedTensor<16x1x1x48x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[WT_CMX:%.+]] = VPU.Copy([[WT]]) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32>
    //CHECK-SAME:           -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:        [[COMPRESS_CONV:%.+]] = VPU.NCE.CompressConvolution([[COPY_TO_CMX]], [[WEIGHTS_CMX]], [[WT_CMX]]) {cm_sp_pattern = 1 : i64, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16, 1, 3, 3], strides = [2, 2]}
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>

    //CHECK:        [[OUT:%.+]] = VPU.Copy([[COMPRESS_CONV]]) : !VPU.DistributedTensor<1x16x32x32x!qElemType, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>
    //CHECK-SAME:           -> tensor<1x16x32x32x!qElemType, {order = #NHWC}>

    //CHECK:    return [[OUT]] : tensor<1x16x32x32x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TanHIncrementalPipeline
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x16x4xf16, {order = #NHWC}>)
func.func @TanHIncrementalPipeline(%arg0: tensor<1x1x16x4xf16, {order = #NHWC}>) -> tensor<1x1x16x4xf16, {order = #NHWC}> {
    %1 = VPU.Tanh(%arg0) : tensor<1x1x16x4xf16, {order = #NHWC}> -> tensor<1x1x16x4xf16, {order = #NHWC}>
    return %1 : tensor<1x1x16x4xf16, {order = #NHWC}>
}
    //CHECK:        [[OP0:%.+]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x1x16x4xf16, {order = #NHWC}>
    //CHECK-SAME:          -> !VPU.DistributedTensor<1x1x16x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 2, 4], [1, 1, 2, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 2, 4], [1, 1, 2, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>
    //CHECK:        [[TANH:%.+]] = VPU.Tanh([[OP0]]) : !VPU.DistributedTensor<1x1x16x4xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 2, 4], [1, 1, 2, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 2, 4], [1, 1, 2, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>
    //CHECK-SAME:          -> !VPU.DistributedTensor<1x1x16x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 2, 4], [1, 1, 2, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 2, 4], [1, 1, 2, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>
    //CHECK:        [[RES:%.+]] = VPU.Copy([[TANH]]) : !VPU.DistributedTensor<1x1x16x4xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 2, 4], [1, 1, 2, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 3, 4], [1, 1, 2, 4], [1, 1, 2, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>
    //CHECK:        -> tensor<1x1x16x4xf16, {order = #NHWC}>

    //CHECK: return [[RES]] : tensor<1x1x16x4xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvTanHIncrementalPipeline
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @ConvTanHIncrementalPipeline(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad =  #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}>, tensor<80x1x1x4xsi32> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    %1 = VPU.Tanh(%0) : tensor<1x80x28x28xf16, {order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %1 : tensor<1x80x28x28xf16, {order = #NHWC}>
}

	//CHECK: [[CONST:%.+]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
	//CHECK: [[CONST_0:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
	//CHECK: [[OP0:%.+]] =  VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]]
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]]
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]}>

	//CHECK: [[OP1:%.+]] = VPU.Copy([[CONST_0]]) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:     -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

	//CHECK: [[OP2:%.+]] = VPU.Copy([[CONST]]) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32>
    //CHECK-SAME: -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

	//CHECK: [[OP3:%.+]] =  VPU.NCE.Convolution([[OP0]], [[OP1]], [[OP2]])
    //CHECK-SAME:       {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:        ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>

	//CHECK: [[OP4:%.+]] = VPU.Copy([[OP3]]) :
    //CHECK-SAME:          !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>
    //CHECK-SAME:       -> tensor<1x80x28x28xf16, {order = #NHWC}>

    // CHECK:      [[OP5:%.+]] = VPU.Copy([[OP4]]) {out_mem_space = @CMX_NN} : tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:          -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>
    // CHECK:      [[OP6:%.+]] = VPU.Tanh([[OP5]]) : !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>
    //CHECK-SAME:          -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>
    // CHECK:      [[OP7:%.+]] = VPU.Copy([[OP6]]) : !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>
    // CHECK:      -> tensor<1x80x28x28xf16, {order = #NHWC}>

	//CHECK: return [[OP7]] : tensor<1x80x28x28xf16, {order = #NHWC}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @InterpolateIncrementalPipeline(
// CHECK-SAME:                    %[[VAL_0:.+]]: tensor<1x2x540x1920xf16>) -> tensor<1x2x256x512xf16> {
func.func @InterpolateIncrementalPipeline(%arg0: tensor<1x2x540x1920xf16>) -> tensor<1x2x256x512xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [256, 512]} : tensor<1x2x540x1920xf16> -> tensor<1x2x256x512xf16>
    return %0 : tensor<1x2x256x512xf16>

    // CHECK:   %[[VAL_3:.*]] = VPU.Copy(%[[VAL_0]]) {out_mem_space = @CMX_NN} : tensor<1x2x540x1920xf16> ->
    // CHECK-SAME:  !VPU.DistributedTensor<1x2x540x1920xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 2, 90, 1920], [1, 2, 91, 1920], [1, 2, 91, 1920], [1, 2, 91, 1920], [1, 2, 89, 1920], [1, 2, 88, 1920]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 90, 0], [0, 0, 181, 0], [0, 0, 272, 0], [0, 0, 363, 0], [0, 0, 452, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 2, 90, 1920], [1, 2, 91, 1920], [1, 2, 91, 1920], [1, 2, 91, 1920], [1, 2, 89, 1920], [1, 2, 88, 1920]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 90, 0], [0, 0, 181, 0], [0, 0, 272, 0], [0, 0, 363, 0], [0, 0, 452, 0]]}>
    // CHECK:   %[[VAL_1:.*]] = const.Declare tensor<1x1x1x512xsi32> = dense<
    // CHECK:   %[[VAL_4:.*]] = VPU.Copy(%[[VAL_1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xsi32> ->
    // CHECK-SAME:  !VPU.DistributedTensor<1x1x1x512xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   %[[VAL_2:.*]] = const.Declare tensor<1x1x1x1024xf16> = dense<
    // CHECK:   %[[VAL_5:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1024xf16> ->
    // CHECK-SAME:  !VPU.DistributedTensor<1x1x1x1024xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   %[[VAL_6:.*]] = VPU.Interpolate(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]]) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 2, 540, 1920], initial_output_dims_attr = [1, 2, 256, 512], operandSegmentSizes = array<i32: 1, 0, 0, 0, 1, 1>, scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [256, 512], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
    // CHECK-SAME:  !VPU.DistributedTensor<1x2x540x1920xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 2, 90, 1920], [1, 2, 91, 1920], [1, 2, 91, 1920], [1, 2, 91, 1920], [1, 2, 89, 1920], [1, 2, 88, 1920]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 90, 0], [0, 0, 181, 0], [0, 0, 272, 0], [0, 0, 363, 0], [0, 0, 452, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 2, 90, 1920], [1, 2, 91, 1920], [1, 2, 91, 1920], [1, 2, 91, 1920], [1, 2, 89, 1920], [1, 2, 88, 1920]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 90, 0], [0, 0, 181, 0], [0, 0, 272, 0], [0, 0, 363, 0], [0, 0, 452, 0]]}>,
    // CHECK-SAME:  !VPU.DistributedTensor<1x1x1x512xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
    // CHECK-SAME:  !VPU.DistributedTensor<1x1x1x1024xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> ->
    // CHECK-SAME:  !VPU.DistributedTensor<1x2x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 42, 512], [1, 2, 42, 512]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 42, 512], [1, 2, 42, 512]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]]}>
    // CHECK:   %[[VAL_7:.*]] = VPU.Copy(%[[VAL_6]]) :
    // CHECK-SAME:  !VPU.DistributedTensor<1x2x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 42, 512], [1, 2, 42, 512]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 42, 512], [1, 2, 42, 512]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]]}> -> tensor<1x2x256x512xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x2x256x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceSumSOK(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<1x1024x7x7xf32>) -> tensor<1x1024xf32> {
func.func @ReduceSumSOK(%arg0: tensor<1x1024x7x7xf32>) -> tensor<1x1024xf32> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf32> -> tensor<1x1024x1x1xf32>
  %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 1024]} : tensor<1x1024x1x1xf32> -> tensor<1x1024xf32>
  return %1 : tensor<1x1024xf32>

    // CHECK:           %[[VAL_1:.+]] =  VPU.Copy(%[[VAL_0]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf32>
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x1024x7x7xf32, #NCHW, @CMX_NN,
    // CHECK-SAME:   {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME:   compute_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]],
    // CHECK-SAME:   compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]],
    // CHECK-SAME:   memory_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]],
    // CHECK-SAME:   memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}>

    // CHECK:           %[[VAL_2:.+]] = VPU.ReduceSum(%[[VAL_1]]) {axes_value = [2, 3], keep_dims}
    // CHECK-SAME:   : !VPU.DistributedTensor<1x1024x7x7xf32, #NCHW, @CMX_NN,
    // CHECK-SAME:   {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME:   compute_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]],
    // CHECK-SAME:   compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]],
    // CHECK-SAME:   memory_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]],
    // CHECK-SAME:   memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}>
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x1024x1x1xf32, #NCHW, @CMX_NN,
    // CHECK-SAME:   {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME:   compute_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]],
    // CHECK-SAME:   compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]],
    // CHECK-SAME:   memory_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]],
    // CHECK-SAME:   memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}>

    // CHECK:           %[[VAL_3:.+]] = VPU.Copy(%[[VAL_2]]) : !VPU.DistributedTensor<1x1024x1x1xf32, #NCHW, @CMX_NN,
    // CHECK-SAME:   {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME:   compute_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]],
    // CHECK-SAME:   compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]],
    // CHECK-SAME:   memory_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]],
    // CHECK-SAME:   memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}>
    // CHECK-SAME:   -> tensor<1x1024x1x1xf32>
    // CHECK:           %[[VAL_4:.+]] = VPU.AffineReshape(%[[VAL_3]]) {dim_mapping = {{\[\[}}0], [1], [1], [1]], shape_value = [1, 1024]} : tensor<1x1024x1x1xf32> -> tensor<1x1024xf32>
    // CHECK:           return %[[VAL_4]] : tensor<1x1024xf32>
    // CHECK:      }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceSumSOH(
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<1x16x32x64xf32>) -> tensor<1x1x32x1xf32> {
func.func @ReduceSumSOH(%arg0: tensor<1x16x32x64xf32>) -> tensor<1x1x32x1xf32> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x16x32x64xf32> -> tensor<1x1x32x1xf32>
  return %0 : tensor<1x1x32x1xf32>

    // CHECK:           %[[VAL_1:.+]] = VPU.Copy(%[[VAL_0]]) {out_mem_space = @CMX_NN} : tensor<1x16x32x64xf32>
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x16x32x64xf32, #NCHW, @CMX_NN,
    // CHECK-SAME:   {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME:   compute_shapes = {{\[\[}}1, 16, 6, 64], [1, 16, 6, 64], [1, 16, 5, 64], [1, 16, 5, 64], [1, 16, 5, 64], [1, 16, 5, 64]],
    // CHECK-SAME:   compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    // CHECK-SAME:   memory_shapes = {{\[\[}}1, 16, 6, 64], [1, 16, 6, 64], [1, 16, 5, 64], [1, 16, 5, 64], [1, 16, 5, 64], [1, 16, 5, 64]],
    // CHECK-SAME:   memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>
    // CHECK:           %[[VAL_2:.+]] = VPU.ReduceSum(%[[VAL_1]]) {axes_value = [1, 3], keep_dims}
    // CHECK-SAME:   : !VPU.DistributedTensor<1x16x32x64xf32, #NCHW, @CMX_NN,
    // CHECK-SAME:   {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME:   compute_shapes = {{\[\[}}1, 16, 6, 64], [1, 16, 6, 64], [1, 16, 5, 64], [1, 16, 5, 64], [1, 16, 5, 64], [1, 16, 5, 64]],
    // CHECK-SAME:   compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    // CHECK-SAME:   memory_shapes = {{\[\[}}1, 16, 6, 64], [1, 16, 6, 64], [1, 16, 5, 64], [1, 16, 5, 64], [1, 16, 5, 64], [1, 16, 5, 64]],
    // CHECK-SAME:   memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x1x32x1xf32, #NCHW, @CMX_NN,
    // CHECK-SAME:   {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME:   compute_shapes = {{\[\[}}1, 1, 6, 1], [1, 1, 6, 1], [1, 1, 5, 1], [1, 1, 5, 1], [1, 1, 5, 1], [1, 1, 5, 1]],
    // CHECK-SAME:   compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    // CHECK-SAME:   memory_shapes = {{\[\[}}1, 1, 6, 1], [1, 1, 6, 1], [1, 1, 5, 1], [1, 1, 5, 1], [1, 1, 5, 1], [1, 1, 5, 1]],
    // CHECK-SAME:   memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>

    // CHECK:           %[[VAL_3:.+]] = VPU.Copy(%[[VAL_2]]) : !VPU.DistributedTensor<1x1x32x1xf32, #NCHW, @CMX_NN,
    // CHECK-SAME:   {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME:   compute_shapes = {{\[\[}}1, 1, 6, 1], [1, 1, 6, 1], [1, 1, 5, 1], [1, 1, 5, 1], [1, 1, 5, 1], [1, 1, 5, 1]],
    // CHECK-SAME:   compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    // CHECK-SAME:   memory_shapes = {{\[\[}}1, 1, 6, 1], [1, 1, 6, 1], [1, 1, 5, 1], [1, 1, 5, 1], [1, 1, 5, 1], [1, 1, 5, 1]],
    // CHECK-SAME:   memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>
    // CHECK-SAME:   -> tensor<1x1x32x1xf32>
    // CHECK:           return %[[VAL_3]] : tensor<1x1x32x1xf32>
    // CHECK:      }
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: func.func @LSTMSequenceBidirectionalSplitOverKernel(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<1x2x80x512xf16>) -> (tensor<1x2x80x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>) {
func.func @LSTMSequenceBidirectionalSplitOverKernel(%arg0: tensor<1x2x80x512xf16>) -> (tensor<1x2x80x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>) {
    %cst_0 = const.Declare tensor<1x2x1x128xf16> = dense<1.000000e+00> : tensor<1x2x1x128xf16>
    %cst_2 = const.Declare tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}> = dense<3.000000e+00> : tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>
    %cst_5 = const.Declare tensor<1x2x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x2x4x128xf16, {order = #NCWH}>
    %cst = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>
    %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_2, %cst_5, %cst) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 80 : i64} : tensor<1x2x80x512xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>, tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x2x80x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>
    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x2x80x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>

// CHECK-DAG:   [[VAL_1:%.+]] = const.Declare tensor<1x2x1x128xf16> = dense<1.000000e+00> : tensor<1x2x1x128xf16>
// CHECK-DAG:   [[VAL_2:%.+]] = const.Declare tensor<2x4x128x128xf16, {order = #NWHC}> = dense<3.000000e+00> : tensor<2x4x128x128xf16, {order = #NWHC}>
// CHECK-DAG:   [[VAL_33:%.+]] = const.Declare tensor<1x2x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x2x4x128xf16, {order = #NCWH}>
// CHECK-DAG:   [[VAL_3:%.+]] = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>

// CHECK:        [[VAL_4:%.+]] = VPU.Copy([[VAL_0]]) {out_mem_space = @CMX_NN} : tensor<1x2x80x512xf16>
// CHECK-SAME: -> !VPU.DistributedTensor<1x2x80x512xf16, #NCHW, @CMX_NN,
// CHECK-SAME: {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 80, 512], [1, 1, 80, 512]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 80, 512], [1, 1, 80, 512]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

// CHECK:         [[VAL_5:%.+]] = VPU.Copy([[VAL_1]]) {out_mem_space = @CMX_NN} : tensor<1x2x1x128xf16>
// CHECK-SAME:  -> !VPU.DistributedTensor<1x2x1x128xf16, #NCHW, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

// CHECK:         [[VAL_6:%.+]] = VPU.Copy([[VAL_1]]) {out_mem_space = @CMX_NN} : tensor<1x2x1x128xf16>
// CHECK-SAME:  -> !VPU.DistributedTensor<1x2x1x128xf16, #NCHW, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

// CHECK:         [[VAL_7:%.+]] = VPU.Copy([[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<2x4x128x128xf16, {order = #NWHC}>
// CHECK-SAME:  -> !VPU.DistributedTensor<2x4x128x128xf16, #NWHC, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}>

// CHECK:         [[VAL_88:%.+]] = VPU.Copy([[VAL_33]]) {out_mem_space = @CMX_NN} : tensor<1x2x4x128xf16, {order = #NCWH}>
// CHECK-SAME:  -> !VPU.DistributedTensor<1x2x4x128xf16, #NCWH, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 4, 128], [1, 1, 4, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 4, 128], [1, 1, 4, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

// CHECK:         [[VAL_8:%.+]] = VPU.Copy([[VAL_3]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x2xsi32>
// CHECK-SAME:  -> !VPU.DistributedTensor<1x1x1x2xsi32, #NCHW, @CMX_NN,
// CHECK-SAME:  {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 2], [1, 1, 1, 2]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 2], [1, 1, 1, 2]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:         [[VAL_9:%.+]], [[VAL_10:%.+]], [[VAL_11:%.+]] = VPU.LSTMSequence([[VAL_4]], [[VAL_5]], [[VAL_6]], [[VAL_7]], [[VAL_88]], [[VAL_8]]) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 80 : i64} :

// CHECK-SAME:    !VPU.DistributedTensor<1x2x80x512xf16, #NCHW, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 80, 512], [1, 1, 80, 512]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 80, 512], [1, 1, 80, 512]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>,

// CHECK-SAME:    !VPU.DistributedTensor<1x2x1x128xf16, #NCHW, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>,

// CHECK-SAME:    !VPU.DistributedTensor<1x2x1x128xf16, #NCHW, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>,

// CHECK-SAME:    !VPU.DistributedTensor<2x4x128x128xf16, #NWHC, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}>,

// CHECK-SAME:    !VPU.DistributedTensor<1x2x4x128xf16, #NCWH, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 4, 128], [1, 1, 4, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 4, 128], [1, 1, 4, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

// CHECK-SAME:    !VPU.DistributedTensor<1x1x1x2xsi32, #NCHW, @CMX_NN,
// CHECK-SAME:  {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 2], [1, 1, 1, 2]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 2], [1, 1, 1, 2]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}> ->

// CHECK-SAME:    !VPU.DistributedTensor<1x2x80x128xf16, #NCHW, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 80, 128], [1, 1, 80, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 80, 128], [1, 1, 80, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>,

// CHECK-SAME:    !VPU.DistributedTensor<1x2x1x128xf16, #NCHW, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>,

// CHECK-SAME:    !VPU.DistributedTensor<1x2x1x128xf16, #NCHW, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

// CHECK:         [[VAL_12:%.+]] = VPU.Copy([[VAL_9]]) : !VPU.DistributedTensor<1x2x80x128xf16, #NCHW, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 80, 128], [1, 1, 80, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 80, 128], [1, 1, 80, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}> -> tensor<1x2x80x128xf16>

// CHECK:         [[VAL_13:%.+]] = VPU.Copy([[VAL_10]]) : !VPU.DistributedTensor<1x2x1x128xf16, #NCHW, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}> -> tensor<1x2x1x128xf16>

// CHECK:         [[VAL_14:%.+]] = VPU.Copy([[VAL_11]]) : !VPU.DistributedTensor<1x2x1x128xf16, #NCHW, @CMX_NN,
// CHECK-SAME:  {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}> -> tensor<1x2x1x128xf16>

// CHECK:   return [[VAL_12]], [[VAL_13]], [[VAL_14]] : tensor<1x2x80x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @CumSumIncrementalPipelineCheck
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x9x16x13500xf16>) -> tensor<1x9x16x13500xf16> {
func.func @CumSumIncrementalPipelineCheck(%arg0: tensor<1x9x16x13500xf16>) -> tensor<1x9x16x13500xf16> {
    %0 = VPU.CumSum(%arg0) {axis_value = 1 : i64} : tensor<1x9x16x13500xf16> -> tensor<1x9x16x13500xf16>
    return %0 : tensor<1x9x16x13500xf16>

    //CHECK: [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 9, 16, 6750] : tensor<1x9x16x13500xf16> to tensor<1x9x16x6750xf16>
    //CHECK: [[COPY_IN_0:%.+]] = VPU.Copy([[SLICE_0]]) {out_mem_space = @CMX_NN} : tensor<1x9x16x6750xf16>
    //CHECK-SAME:                 -> !VPU.DistributedTensor<1x9x16x6750xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:        compute_shapes = [[1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 2, 6750], [1, 9, 2, 6750]],
    //CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:        memory_shapes = [[1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 2, 6750], [1, 9, 2, 6750]],
    //CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>
    //CHECK: [[CUMSUM_0:%.+]] = VPU.CumSum([[COPY_IN_0]]) {axis_value = 1 : i64} : !VPU.DistributedTensor<1x9x16x6750xf16, #NCHW, @CMX_NN,
    //CHECK-SAME{LITERAL}:        compute_shapes = [[1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 2, 6750], [1, 9, 2, 6750]],
    //CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:        memory_shapes = [[1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 2, 6750], [1, 9, 2, 6750]],
    //CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>
    //CHECK-SAME:                 {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK: [[COPY_OUT_0:%.+]] = VPU.Copy([[CUMSUM_0]]) : !VPU.DistributedTensor<1x9x16x6750xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK: [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 6750] [1, 9, 16, 6750] : tensor<1x9x16x13500xf16> to tensor<1x9x16x6750xf16>
    //CHECK: [[COPY_IN_1:%.+]] = VPU.Copy([[SLICE_1]]) {out_mem_space = @CMX_NN} : tensor<1x9x16x6750xf16>
    //CHECK-SAME:                 -> !VPU.DistributedTensor<1x9x16x6750xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:        compute_shapes = [[1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 2, 6750], [1, 9, 2, 6750]],
    //CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:        memory_shapes = [[1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 2, 6750], [1, 9, 2, 6750]],
    //CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>
    //CHECK: [[CUMSUM_1:%.+]] = VPU.CumSum([[COPY_IN_1]]) {axis_value = 1 : i64} : !VPU.DistributedTensor<1x9x16x6750xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                 {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:        compute_shapes = [[1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 2, 6750], [1, 9, 2, 6750]],
    //CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:        memory_shapes = [[1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 3, 6750], [1, 9, 2, 6750], [1, 9, 2, 6750]],
    //CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>
    //CHECK: [[COPY_OUT_1:%.+]] = VPU.Copy([[CUMSUM_1]]) : !VPU.DistributedTensor<1x9x16x6750xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[COPY_OUT_0]], [[COPY_OUT_1]])
    //CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 6750]]} : tensor<1x9x16x6750xf16>, tensor<1x9x16x6750xf16> -> tensor<1x9x16x13500xf16>
    //CHECK: return [[CONCAT]] : tensor<1x9x16x13500xf16>
}
