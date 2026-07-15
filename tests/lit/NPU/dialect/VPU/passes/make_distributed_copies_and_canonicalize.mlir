//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --make-distributed-copies --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

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
    %conv = VPU.NCE.Convolution(%conv_input, %conv_weights) rawFilterShape [256, 48, 3, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        
        strides = [2, 1]
    } : !ConvInputTensor, !ConvWeightsTensor -> !ConvOutputTensor
    %conv_cmx_to_ddr = VPU.UnrolledType(%conv : !ConvOutputTensor) -> tensor<1x256x128x16xf16, {order = #NHWC}>
    %shape_cast = VPU.ShapeCast {shape = [1, 32, 128, 128]} inputs(%conv_cmx_to_ddr : tensor<1x256x128x16xf16, {order = #NHWC}>) -> tensor<1x32x128x128xf16, {order = #NHWC}>
    %ddr_to_next_cmx = VPU.UnrolledType(%shape_cast : tensor<1x32x128x128xf16, {order = #NHWC}>) -> !NextOverlappingInputTensor
    return %ddr_to_next_cmx : !NextOverlappingInputTensor

    // CHECK:               [[CONV:%.+]] = VPU.NCE.Convolution([[CONV_INPUT]], [[CONV_WEIGHTS]]){{.*}} {
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
    %conv = VPU.NCE.Convolution(%conv_input, %conv_weights) rawFilterShape [256, 48, 3, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        
        strides = [2, 1]
    } : !ConvInputTensor, !ConvWeightsTensor -> !ConvOutputTensor
    %conv_cmx_to_ddr = VPU.UnrolledType(%conv : !ConvOutputTensor) -> tensor<1x256x128x16xf16, {order = #NHWC}>
    %shape_cast = VPU.ShapeCast {shape = [1, 32, 128, 128]} inputs(%conv_cmx_to_ddr : tensor<1x256x128x16xf16, {order = #NHWC}>) -> tensor<1x32x128x128xf16, {order = #NHWC}>
    %ddr_to_next_cmx0 = VPU.UnrolledType(%shape_cast : tensor<1x32x128x128xf16, {order = #NHWC}>) -> !NextOverlappingInputTensor
    %ddr_to_next_cmx1 = VPU.UnrolledType(%shape_cast : tensor<1x32x128x128xf16, {order = #NHWC}>) -> !NextOverlappingInputTensor
    return %ddr_to_next_cmx0, %ddr_to_next_cmx1 : !NextOverlappingInputTensor, !NextOverlappingInputTensor

    // CHECK:               [[CONV:%.+]] = VPU.NCE.Convolution([[CONV_INPUT]], [[CONV_WEIGHTS]]){{.*}} {
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

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedInput = !VPU.DistributedTensor<1x16x90x160xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 16, 23, 160], [1, 16, 23, 160], [1, 16, 22, 160], [1, 16, 22, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0], [0, 0, 68, 0]],
    memory_shapes = [[1, 16, 24, 160], [1, 16, 25, 160], [1, 16, 24, 160], [1, 16, 23, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 45, 0], [0, 0, 67, 0]]}>
!DistributedWeights = !VPU.DistributedTensor<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!DistributedOutput = !VPU.DistributedTensor<1x32x90x160xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 32, 23, 160], [1, 32, 23, 160], [1, 32, 22, 160], [1, 32, 22, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0], [0, 0, 68, 0]],
    memory_shapes = [[1, 32, 23, 160], [1, 32, 23, 160], [1, 32, 22, 160], [1, 32, 22, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0], [0, 0, 68, 0]]}>
!DistributedShapeCast = !VPU.DistributedTensor<1x128x90x40xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 128, 23, 40], [1, 128, 23, 40], [1, 128, 22, 40], [1, 128, 22, 40]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0], [0, 0, 68, 0]],
    memory_shapes = [[1, 128, 24, 40], [1, 128, 25, 40], [1, 128, 24, 40], [1, 128, 23, 40]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 45, 0], [0, 0, 67, 0]]}>

// CHECK-LABEL: @DoNotOptimizeShapeCastCopiesForSiblingEltwise
func.func @DoNotOptimizeShapeCastCopiesForSiblingEltwise(%input: !DistributedInput, %weights: !DistributedWeights, %eltwise_input : tensor<1x32x90x160xf16, {order = #NHWC}>) -> (!DistributedShapeCast, tensor<1x32x90x160xf16, {order = #NHWC}>) {
    %conv = VPU.NCE.Convolution(%input, %weights) rawFilterShape [32, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : !DistributedInput, !DistributedWeights -> !DistributedOutput
    %conv_to_ddr = VPU.UnrolledType(%conv : !DistributedOutput) -> tensor<1x32x90x160xf16, {order = #NHWC}>
    %shape_cast = VPU.ShapeCast {shape = [1, 128, 90, 40]} inputs(%conv_to_ddr : tensor<1x32x90x160xf16, {order = #NHWC}>) -> tensor<1x128x90x40xf16, {order = #NHWC}>
    %shape_cast_to_cmx = VPU.UnrolledType(%shape_cast : tensor<1x128x90x40xf16, {order = #NHWC}>) -> !DistributedShapeCast

    %conv_to_cmx = VPU.UnrolledType(%conv_to_ddr : tensor<1x32x90x160xf16, {order = #NHWC}>) -> !DistributedOutput
    %eltwise_input_to_cmx = VPU.UnrolledType(%eltwise_input : tensor<1x32x90x160xf16, {order = #NHWC}>) -> !DistributedOutput
    %eltwise = VPU.NCE.Eltwise(%conv_to_cmx, %eltwise_input_to_cmx) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> !DistributedOutput
    %eltwise_to_ddr = VPU.UnrolledType(%eltwise : !DistributedOutput) -> tensor<1x32x90x160xf16, {order = #NHWC}>

    return %shape_cast_to_cmx, %eltwise_to_ddr : !DistributedShapeCast, tensor<1x32x90x160xf16, {order = #NHWC}>

    // The eltwise should not have distributed types
    // CHECK:       VPU.ShapeCast
    // CHECK-SAME:    inputs({{.+}} : tensor<1x32x90x160xf16, {order = #NHWC}>)
    // CHECK-SAME:    -> tensor<1x128x90x40xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!ShapeCastInputTensor = !VPU.DistributedTensor<1x32x4x640xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 32, 2, 640], [1, 32, 2, 640]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]],
    memory_shapes = [[1, 32, 2, 640], [1, 32, 2, 640]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]
}>

!ShapeCastOutputTensor = !VPU.DistributedTensor<1x1280x4x16xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1280, 2, 16], [1, 1280, 2, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]],
    memory_shapes = [[1, 1280, 2, 16], [1, 1280, 2, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]
}>

// CHECK-LABEL: @DoNotOptimizeShapeCastCopiesWhenTiledDimIsNotMemoryCompatible
// CHECK-SAME:      [[INPUT:%.+]]: !VPU.DistributedTensor<1x32x4x640xf16, #NCHW, @CMX_NN,
func.func @DoNotOptimizeShapeCastCopiesWhenTiledDimIsNotMemoryCompatible(%input: !ShapeCastInputTensor) -> !ShapeCastOutputTensor {
    %cmx_to_ddr = VPU.UnrolledType(%input : !ShapeCastInputTensor) -> tensor<1x32x4x640xf16, {order = #NCHW}>
    %shape_cast = VPU.ShapeCast {shape = [1, 1280, 4, 16]} inputs(%cmx_to_ddr : tensor<1x32x4x640xf16, {order = #NCHW}>) -> tensor<1x1280x4x16xf16, {order = #NCHW}>
    %ddr_to_cmx = VPU.UnrolledType(%shape_cast : tensor<1x1280x4x16xf16, {order = #NCHW}>) -> !ShapeCastOutputTensor
    return %ddr_to_cmx : !ShapeCastOutputTensor

    // CHECK:               [[COPY_IN:%.+]] = VPU.Copy([[INPUT]]
    // CHECK:               [[SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 1280, 4, 16]} inputs([[COPY_IN]]
    // CHECK-SAME:              tensor<1x32x4x640xf16, {order = #NCHW}>
    // CHECK-SAME:           -> tensor<1x1280x4x16xf16, {order = #NCHW}>
    // CHECK:               [[COPY_OUT:%.+]] = VPU.Copy([[SHAPE_CAST]]
    // CHECK:               return [[COPY_OUT]] : !VPU.DistributedTensor<1x1280x4x16xf16, #NCHW, @CMX_NN,
}
