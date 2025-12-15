//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --make-distributed-copies --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConvInputTensor = !VPU.DistributedTensor<1x48x256x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 48, 43, 16], [1, 48, 43, 16], [1, 48, 43, 16], [1, 48, 43, 16], [1, 48, 42, 16], [1, 48, 42, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]],
    memory_shapes = [[1, 48, 45, 16], [1, 48, 46, 16], [1, 48, 45, 16], [1, 48, 44, 16], [1, 48, 43, 16], [1, 48, 42, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]]
}>

!ConvWeightsTensor = !VPU.SparseTensor<
    data=!VPU.DistributedTensor<256x48x3x2xf16, #NHWC, @CMX_NN, {
        mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
        compute_shapes = [[256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    }>,
    sparsity_map=!VPU.DistributedTensor<256x1x1x384xi1, #NHWC, @CMX_NN, {
        mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
        compute_shapes = [[256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    }>,
    is_weights,
    #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<256xi64>, alignment = 16 : i64>
>

!ConvOutputTensor = !VPU.DistributedTensor<1x256x128x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 256, 22, 16], [1, 256, 22, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]],
    memory_shapes = [[1, 256, 22, 16], [1, 256, 22, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]]
}>

!NextOverlappingInputTensor = !VPU.DistributedTensor<1x32x128x128xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 32, 22, 128], [1, 32, 22, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]],
    memory_shapes = [[1, 32, 23, 128], [1, 32, 24, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 22, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 43, 0], [0, 0, 64, 0], [0, 0, 85, 0], [0, 0, 106, 0]]
}>

// CHECK-LABEL: @OptimizeShapeCastCopies
// CHECK-SAME:      [[CONV_INPUT:%.+]]: !VPU.DistributedTensor<1x48x256x16xf16, #NHWC, @CMX_NN,
// CHECK-SAME:      [[CONV_WEIGHTS:%.+]]: !VPU.SparseTensor<data=!VPU.DistributedTensor<256x48x3x2xf16, #NHWC, @CMX_NN,
func.func @OptimizeShapeCastCopies(%conv_input: !ConvInputTensor, %conv_weights: !ConvWeightsTensor) -> !NextOverlappingInputTensor {
    %conv = VPU.NCE.Convolution(%conv_input, %conv_weights) {
        pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 48, 3, 2],
        strides = [2, 1]
    } : !ConvInputTensor, !ConvWeightsTensor -> !ConvOutputTensor
    %conv_cmx_to_ddr = VPU.UnrolledType(%conv : !ConvOutputTensor) -> tensor<1x256x128x16xf16, {order = #NHWC}>
    %shape_cast = VPU.ShapeCast {shape = [1, 32, 128, 128]} inputs(%conv_cmx_to_ddr : tensor<1x256x128x16xf16, {order = #NHWC}>) -> tensor<1x32x128x128xf16, {order = #NHWC}>
    %ddr_to_next_cmx = VPU.UnrolledType(%shape_cast : tensor<1x32x128x128xf16, {order = #NHWC}>) -> !NextOverlappingInputTensor
    return %ddr_to_next_cmx : !NextOverlappingInputTensor

    // CHECK:               [[CONV:%.+]] = VPU.NCE.Convolution([[CONV_INPUT]], [[CONV_WEIGHTS]]) {
    // CHECK:               [[SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 32, 128, 128]} inputs([[CONV]] : !VPU.DistributedTensor<1x256x128x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 256, 22, 16], [1, 256, 22, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 256, 23, 16], [1, 256, 24, 16], [1, 256, 23, 16], [1, 256, 23, 16], [1, 256, 23, 16], [1, 256, 22, 16]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 43, 0], [0, 0, 64, 0], [0, 0, 85, 0], [0, 0, 106, 0]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x32x128x128xf16, #NHWC, @CMX_NN,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 22, 128], [1, 32, 22, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 23, 128], [1, 32, 24, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 22, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 43, 0], [0, 0, 64, 0], [0, 0, 85, 0], [0, 0, 106, 0]]
    // CHECK:               return [[SHAPE_CAST]] : !VPU.DistributedTensor<1x32x128x128xf16, #NHWC, @CMX_NN,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 22, 128], [1, 32, 22, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 23, 128], [1, 32, 24, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 22, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 43, 0], [0, 0, 64, 0], [0, 0, 85, 0], [0, 0, 106, 0]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConvInputTensor = !VPU.DistributedTensor<1x48x256x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 48, 43, 16], [1, 48, 43, 16], [1, 48, 43, 16], [1, 48, 43, 16], [1, 48, 42, 16], [1, 48, 42, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]],
    memory_shapes = [[1, 48, 45, 16], [1, 48, 46, 16], [1, 48, 45, 16], [1, 48, 44, 16], [1, 48, 43, 16], [1, 48, 42, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]]
}>

!ConvWeightsTensor = !VPU.SparseTensor<
    data=!VPU.DistributedTensor<256x48x3x2xf16, #NHWC, @CMX_NN, {
        mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
        compute_shapes = [[256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2], [256, 48, 3, 2]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    }>,
    sparsity_map=!VPU.DistributedTensor<256x1x1x384xi1, #NHWC, @CMX_NN, {
        mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
        compute_shapes = [[256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384], [256, 1, 1, 384]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    }>,
    is_weights,
    #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<256xi64>, alignment = 16 : i64>
>

!ConvOutputTensor = !VPU.DistributedTensor<1x256x128x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 256, 22, 16], [1, 256, 22, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]],
    memory_shapes = [[1, 256, 22, 16], [1, 256, 22, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]]
}>

!NextOverlappingInputTensor = !VPU.DistributedTensor<1x32x128x128xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 32, 22, 128], [1, 32, 22, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]],
    memory_shapes = [[1, 32, 23, 128], [1, 32, 24, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 22, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 43, 0], [0, 0, 64, 0], [0, 0, 85, 0], [0, 0, 106, 0]]
}>

// CHECK-LABEL: @OptimizeShapeCastCopiesWithMultUsers
// CHECK-SAME:      [[CONV_INPUT:%.+]]: !VPU.DistributedTensor<1x48x256x16xf16, #NHWC, @CMX_NN,
// CHECK-SAME:      [[CONV_WEIGHTS:%.+]]: !VPU.SparseTensor<data=!VPU.DistributedTensor<256x48x3x2xf16, #NHWC, @CMX_NN,
func.func @OptimizeShapeCastCopiesWithMultUsers(%conv_input: !ConvInputTensor, %conv_weights: !ConvWeightsTensor) -> (!NextOverlappingInputTensor, !NextOverlappingInputTensor) {
    %conv = VPU.NCE.Convolution(%conv_input, %conv_weights) {
        pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 48, 3, 2],
        strides = [2, 1]
    } : !ConvInputTensor, !ConvWeightsTensor -> !ConvOutputTensor
    %conv_cmx_to_ddr = VPU.UnrolledType(%conv : !ConvOutputTensor) -> tensor<1x256x128x16xf16, {order = #NHWC}>
    %shape_cast = VPU.ShapeCast {shape = [1, 32, 128, 128]} inputs(%conv_cmx_to_ddr : tensor<1x256x128x16xf16, {order = #NHWC}>) -> tensor<1x32x128x128xf16, {order = #NHWC}>
    %ddr_to_next_cmx0 = VPU.UnrolledType(%shape_cast : tensor<1x32x128x128xf16, {order = #NHWC}>) -> !NextOverlappingInputTensor
    %ddr_to_next_cmx1 = VPU.UnrolledType(%shape_cast : tensor<1x32x128x128xf16, {order = #NHWC}>) -> !NextOverlappingInputTensor
    return %ddr_to_next_cmx0, %ddr_to_next_cmx1 : !NextOverlappingInputTensor, !NextOverlappingInputTensor

    // CHECK:               [[CONV:%.+]] = VPU.NCE.Convolution([[CONV_INPUT]], [[CONV_WEIGHTS]]) {
    // CHECK:               [[SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 32, 128, 128]} inputs([[CONV]] : !VPU.DistributedTensor<1x256x128x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 256, 22, 16], [1, 256, 22, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 256, 23, 16], [1, 256, 24, 16], [1, 256, 23, 16], [1, 256, 23, 16], [1, 256, 23, 16], [1, 256, 22, 16]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 43, 0], [0, 0, 64, 0], [0, 0, 85, 0], [0, 0, 106, 0]]
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x32x128x128xf16, #NHWC, @CMX_NN,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 22, 128], [1, 32, 22, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 23, 128], [1, 32, 24, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 22, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 43, 0], [0, 0, 64, 0], [0, 0, 85, 0], [0, 0, 106, 0]]
    // CHECK:               return [[SHAPE_CAST]], [[SHAPE_CAST]] : !VPU.DistributedTensor<1x32x128x128xf16, #NHWC, @CMX_NN,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 22, 128], [1, 32, 22, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 23, 128], [1, 32, 24, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 22, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 43, 0], [0, 0, 64, 0], [0, 0, 85, 0], [0, 0, 106, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!EltwiseTensorType = !VPU.DistributedTensor<1x256x128x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 256, 22, 16], [1, 256, 22, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]],
    memory_shapes = [[1, 256, 22, 16], [1, 256, 22, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16], [1, 256, 21, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]]
}>

!NextOverlappingInputTensorType = !VPU.DistributedTensor<1x32x128x128xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 32, 22, 128], [1, 32, 22, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128], [1, 32, 21, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]],
    memory_shapes = [[1, 32, 23, 128], [1, 32, 24, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 23, 128], [1, 32, 22, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 43, 0], [0, 0, 64, 0], [0, 0, 85, 0], [0, 0, 106, 0]]
}>

// CHECK-LABEL: @NotOptimizeShapeCastCopies
func.func @NotOptimizeShapeCastCopies(%eltwise_input1: !EltwiseTensorType, %eltwise_input2: !EltwiseTensorType) -> !NextOverlappingInputTensorType {
    %eltwise = VPU.NCE.Eltwise(%eltwise_input1, %eltwise_input2) {
        is_inplace = true,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } : !EltwiseTensorType, !EltwiseTensorType -> !EltwiseTensorType
    %conv_cmx_to_ddr = VPU.UnrolledType(%eltwise : !EltwiseTensorType) -> tensor<1x256x128x16xf16, {order = #NHWC}>
    %shape_cast = VPU.ShapeCast {shape = [1, 32, 128, 128]} inputs(%conv_cmx_to_ddr : tensor<1x256x128x16xf16, {order = #NHWC}>) -> tensor<1x32x128x128xf16, {order = #NHWC}>
    %ddr_to_next_cmx = VPU.UnrolledType(%shape_cast : tensor<1x32x128x128xf16, {order = #NHWC}>) -> !NextOverlappingInputTensorType
    return %ddr_to_next_cmx : !NextOverlappingInputTensorType

    // CHECK:               [[ELTWISE:%.+]] = VPU.NCE.Eltwise
    // CHECK-NOT:           VPU.ShapeCast

    // CHECK:               [[COPY1:%.+]] = VPU.Copy([[ELTWISE]]
    // CHECK:               [[SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 32, 128, 128]} inputs([[COPY1]]
    // CHECK:               [[COPY2:%.+]] = VPU.Copy([[SHAPE_CAST]]
}
