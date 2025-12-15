//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --make-distributed-copies --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConvInputTensor0 = !VPU.DistributedTensor<1x64x22x672xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 64, 8, 672], [1, 64, 7, 672], [1, 64, 7, 672]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 15, 0]],
    memory_shapes = [[1, 64, 9, 672], [1, 64, 9, 672], [1, 64, 8, 672]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0]]
}>

!ConvWeightsTensor0 = !VPU.DistributedTensor<64x64x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvWeightsTableTensor0 = !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvOutputTensor0 = !VPU.DistributedTensor<1x64x20x672xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 64, 7, 672], [1, 64, 7, 672], [1, 64, 6, 672]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0]],
    memory_shapes = [[1, 64, 7, 672], [1, 64, 7, 672], [1, 64, 6, 672]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0]]
}>

!ConvInputTensor1 = !VPU.DistributedTensor<1x256x20x168xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 256, 7, 168], [1, 256, 7, 168], [1, 256, 6, 168]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0]],
    memory_shapes = [[1, 256, 8, 168], [1, 256, 8, 168], [1, 256, 8, 168]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0]]
}>

!ConvWeightsTensor1 = !VPU.SparseTensor<
    data=!VPU.DistributedTensor<48x256x3x3xf16, #NHWC, @CMX_NN, {
        mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
        compute_shapes = [[48, 256, 3, 3], [48, 256, 3, 3], [48, 256, 3, 3]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[48, 256, 3, 3], [48, 256, 3, 3], [48, 256, 3, 3]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    }>,
    sparsity_map=!VPU.DistributedTensor<48x1x1x2304xi1, #NHWC, @CMX_NN, {
        mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
        compute_shapes = [[48, 1, 1, 2304], [48, 1, 1, 2304], [48, 1, 1, 2304]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[48, 1, 1, 2304], [48, 1, 1, 2304], [48, 1, 1, 2304]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    }>,
    is_weights,
    #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<576> : tensor<48xi64>, alignment = 16 : i64>
>

!ConvWeightsTableTensor1 = !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ConvOutputTensor1 = !VPU.DistributedTensor<1x48x18x168xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 48, 6, 168], [1, 48, 6, 168], [1, 48, 6, 168]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0]],
    memory_shapes = [[1, 48, 6, 168], [1, 48, 6, 168], [1, 48, 6, 168]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0]]
}>

// CHECK-LABEL: @NotOptimizeShapeCastCopiesAsNotFitIntoCMX
func.func @NotOptimizeShapeCastCopiesAsNotFitIntoCMX(%conv_input: !ConvInputTensor0,
                                                     %conv_weights: !ConvWeightsTensor0,
                                                     %conv_weights_table: !ConvWeightsTableTensor0,
                                                     %conv_weights1: !ConvWeightsTensor1,
                                                     %conv_weights_table1: !ConvWeightsTableTensor1)
                                                     -> !ConvOutputTensor1 {
    %conv0 = VPU.NCE.Convolution(%conv_input, %conv_weights, %conv_weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <LPRELU>,
        clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
        prelu_alpha = [0.199951171875], adder = 0.000000e+00 : f64>,
        rawFilterShape = [64, 64, 3, 3],
        strides = [1, 1]
    } : !ConvInputTensor0, !ConvWeightsTensor0, !ConvWeightsTableTensor0 -> !ConvOutputTensor0

    %conv_cmx_to_ddr0 = VPU.UnrolledType(%conv0 : !ConvOutputTensor0) -> tensor<1x64x20x672xf16, {order = #NHWC}>

    %shape_cast = VPU.ShapeCast {shape = [1, 256, 20, 168]} inputs(%conv_cmx_to_ddr0 : tensor<1x64x20x672xf16, {order = #NHWC}>) -> tensor<1x256x20x168xf16, {order = #NHWC}>

    %ddr_to_next_cmx0 = VPU.UnrolledType(%shape_cast : tensor<1x256x20x168xf16, {order = #NHWC}>) -> !ConvInputTensor1

    %conv1 = VPU.NCE.Convolution(%ddr_to_next_cmx0, %conv_weights1, %conv_weights_table1) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>,
        clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
        prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
        rawFilterShape = [48, 256, 3, 3],
        strides = [1, 1]
    } : !ConvInputTensor1, !ConvWeightsTensor1, !ConvWeightsTableTensor1 -> !ConvOutputTensor1

    return %conv1 : !ConvOutputTensor1

    // CHECK:               [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution
    // CHECK:               [[COPY0:%.+]] = VPU.Copy([[CONVOLUTION0]]
    // CHECK:               [[SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 256, 20, 168]} inputs([[COPY0]]
    // CHECK:               [[COPY1:%.+]] = VPU.Copy([[SHAPE_CAST]]
    // CHECK:               [[CONVOLUTION1:%.+]] = VPU.NCE.Convolution

    // CHECK:               return [[CONVOLUTION1]]
}
