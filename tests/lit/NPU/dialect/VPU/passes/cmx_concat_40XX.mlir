//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --cmx-concat --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Distributed = !VPU.DistributedTensor<
    1x64x36x36xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]
}>

!Distributed2 = !VPU.DistributedTensor<
    64x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!Distributed3 = !VPU.DistributedTensor<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!Distributed4 = !VPU.DistributedTensor<
    64x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments, compute_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!Distributed5 = !VPU.DistributedTensor<
    1x128x36x36xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    memory_shapes = [[1, 128, 7, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 7, 36]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]
}>

// CHECK-LABEL: @InsertAvgPoolingWhenNCEOpHasExtraUser
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x64x36x36xf16, {order = #NHWC}>
func.func @InsertAvgPoolingWhenNCEOpHasExtraUser(%arg0: tensor<1x64x36x36xf16, {order = #NHWC}>)
           -> tensor<1x128x36x36xf16, {order = #NHWC}> {
    %convWeights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %convWeightsTable = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %dwConvWeights = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %maxPoolWeightsTable = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> =
        dense<1> : tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // Input 1 of Concat
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x36x36xf16, {order = #NHWC}> -> !Distributed
    %1 = VPU.Copy(%convWeights) {out_mem_space = @CMX_NN} : tensor<64x64x1x1xf16, {order = #NHWC}> -> !Distributed2
    %2 = VPU.Copy(%convWeightsTable) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32> -> !Distributed3

    %3 = VPU.NCE.Convolution(%0, %1, %2) rawFilterShape [64, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                
                strides = [1, 1]
            } : !Distributed, !Distributed2, !Distributed3 -> !Distributed

    %4 = VPU.Copy(%3) : !Distributed -> tensor<1x64x36x36xf16, {order = #NHWC}>
    // Input 2 of Concat
    %5 = VPU.Copy(%dwConvWeights) {out_mem_space = @CMX_NN} : tensor<64x16x1x1xf16, {order = #NHWC}> -> !Distributed4

    %6 = VPU.NCE.DepthConvolution(%3, %5) rawFilterShape [64, 1, 3, 3] {
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
                lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                
                strides = [1, 1]
            } -> !Distributed

    %7 = VPU.Copy(%6) : !Distributed -> tensor<1x64x36x36xf16, {order = #NHWC}>

    %8 = VPU.Concat(%4, %7) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x36x36xf16, {order = #NHWC}>, tensor<1x64x36x36xf16, {order = #NHWC}> -> tensor<1x128x36x36xf16, {order = #NHWC}>

    // Concat output
    %9 = VPU.Copy(%8) {out_mem_space = @CMX_NN} : tensor<1x128x36x36xf16, {order = #NHWC}> -> !Distributed5

    %10 = VPU.NCE.MaxPool(%9, %maxPoolWeightsTable) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> !Distributed5

    %11 = VPU.Copy(%10) : !Distributed5 -> tensor<1x128x36x36xf16, {order = #NHWC}>

    return %11 : tensor<1x128x36x36xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:       [[CST_1:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[CST_2:%.+]] = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<1> : tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:       [[COPY_IN_0:%.+]] = VPU.Copy([[INPUT]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[COPY_IN_1:%.+]] = VPU.Copy([[CST]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          64x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[COPY_IN_2:%.+]] = VPU.Copy([[CST_0]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          64x1x1x4xsi32, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[COPY_IN_0]], [[COPY_IN_1]], [[COPY_IN_2]]) rawFilterShape [64, 64, 1, 1] {
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[CONV]]) {
    // CHECK-SAME:                  kernel_size = [1, 1],
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[COPY_IN_3:%.+]] = VPU.Copy([[CST_1]])
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          64x16x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[DW_CONV:%.+]] = VPU.NCE.DepthConvolution([[CONV]], [[COPY_IN_3]]) rawFilterShape [64, 1, 3, 3] {
    // CHECK-SAME:                          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                          ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -128 : i64, clamp_high = 127 : i64,
    // CHECK-SAME:                              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:                          strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[CMX_CONCAT:%.+]] = VPU.Concat([[AVGPOOL]], [[DW_CONV]]) {
    // CHECK-SAME:          static_offsets = [
    // CHECK-SAME:              [0, 0, 0, 0],
    // CHECK-SAME:              [0, 64, 0, 0]
    // CHECK-SAME:          ]} :
    // CHECK-SAME:              !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>,
    // CHECK-SAME:              !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x128x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                          compute_shapes = [[1, 128, 7, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 7, 36]],
    // CHECK-SAME{LITERAL}:                          compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]],
    // CHECK-SAME{LITERAL}:                          memory_shapes = [[1, 128, 7, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 7, 36]],
    // CHECK-SAME{LITERAL}:                          memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[CMX_CONCAT]], [[CST_2]] ) {
    // CHECK-SAME:                          kernel_size = [1, 1],
    // CHECK-SAME:                          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                          strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x128x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 128, 7, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[COPY_OUT:%.+]] = VPU.Copy([[MAXPOOL]])
    // CHECK-SAME:                       -> tensor<1x128x36x36xf16, {order = #NHWC}>

    // CHECK:       return [[COPY_OUT]] : tensor<1x128x36x36xf16, {order = #NHWC}>
}
