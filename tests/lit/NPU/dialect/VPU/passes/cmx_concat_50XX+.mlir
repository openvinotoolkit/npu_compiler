//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --cmx-concat --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

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
    64x1x1x4xsi32, #NHWC, @CMX_NN, {
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

// CHECK-LABEL: @InsertAvgPoolingWhenNCEOpHasExtraUserFpPPE
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x64x36x36xf16, {order = #NHWC}>
func.func @InsertAvgPoolingWhenNCEOpHasExtraUserFpPPE(%arg0: tensor<1x64x36x36xf16, {order = #NHWC}>)
           -> tensor<1x128x36x36xf16, {order = #NHWC}> {
    %convWeights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %dwConvWeights = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %maxPoolWeightsTable = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> =
        dense<1> : tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // Input 1 of Concat
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x36x36xf16, {order = #NHWC}> -> !Distributed
    %1 = VPU.Copy(%convWeights) {out_mem_space = @CMX_NN} : tensor<64x64x1x1xf16, {order = #NHWC}> -> !Distributed2
    %3 = VPU.NCE.Convolution(%0, %1) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                    scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
                rawFilterShape = [64, 64, 1, 1],
                strides = [1, 1],
                output_padding = [0, 8, 0, 0]
            } : !Distributed, !Distributed2 -> !Distributed
    %4 = VPU.Copy(%3) : !Distributed -> tensor<1x64x36x36xf16, {order = #NHWC}>

    // Input 2 of Concat
    %5 = VPU.Copy(%dwConvWeights) {out_mem_space = @CMX_NN} : tensor<64x16x1x1xf16, {order = #NHWC}> -> !Distributed4
    %6 = VPU.NCE.DepthConvolution(%3, %5) {
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                    scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
                rawFilterShape = [64, 1, 3, 3],
                strides = [1, 1]
            } -> !Distributed
    %7 = VPU.Copy(%6) : !Distributed -> tensor<1x64x36x36xf16, {order = #NHWC}>

    %8 = VPU.Concat(%4, %7) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x36x36xf16, {order = #NHWC}>, tensor<1x64x36x36xf16, {order = #NHWC}> -> tensor<1x128x36x36xf16, {order = #NHWC}>

    // Concat output
    %9 = VPU.Copy(%8) {out_mem_space = @CMX_NN} : tensor<1x128x36x36xf16, {order = #NHWC}> -> !Distributed5
    %10 = VPU.NCE.MaxPool(%9, %maxPoolWeightsTable) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> !Distributed5
    %11 = VPU.Copy(%10) : !Distributed5 -> tensor<1x128x36x36xf16, {order = #NHWC}>

    return %11 : tensor<1x128x36x36xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_0:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_2:%.+]] = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<1> : tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:       [[COPY_IN_0:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN} : tensor<1x64x36x36xf16, {order = #NHWC}>
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[COPY_IN_1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN} : tensor<64x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          64x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1], [64, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[COPY_IN_0]], [[COPY_IN_1]]) {
    // CHECK-SAME:              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:              ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:                  clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:                  clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:                  scale = 1.000000e+00 : f64,
    // CHECK-SAME:                  prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:              rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:              strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[CONV]]) {
    // CHECK-SAME:                  input_padding = [0, 8, 0, 0],
    // CHECK-SAME:                  kernel_size = [1, 1],
    // CHECK-SAME:                  output_padding = [0, 8, 0, 0],
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:                     clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:                     clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:                     scale = 1.000000e+00 : f64,
    // CHECK-SAME:                     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          1x64x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36], [1, 64, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 64, 7, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 8, 36], [1, 64, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

    // CHECK:       [[COPY_IN_2:%.+]] = VPU.Copy([[CST_0]]) {out_mem_space = @CMX_NN} : tensor<64x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:                      -> !VPU.DistributedTensor<
    // CHECK-SAME:                          64x16x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:       [[DW_CONV:%.+]] = VPU.NCE.DepthConvolution([[CONV]], [[COPY_IN_2]]) {
    // CHECK-SAME:              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:              ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:                  clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:                  clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:                  scale = 1.000000e+00 : f64,
    // CHECK-SAME:                  prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:              rawFilterShape = [64, 1, 3, 3],
    // CHECK-SAME:              strides = [1, 1]}
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

    // CHECK:       [[COPY_OUT:%.+]] = VPU.Copy([[MAXPOOL]]) :
    // CHECK-SAME:                         !VPU.DistributedTensor<
    // CHECK-SAME:                          1x128x36x36xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:                          mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36], [1, 128, 6, 36]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 128, 7, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 8, 36], [1, 128, 7, 36]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>
    // CHECK:       -> tensor<1x128x36x36xf16, {order = #NHWC}>

    // CHECK:       return [[COPY_OUT]] : tensor<1x128x36x36xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistTypeEltwiseInput = !VPU.DistributedTensor<1x16x64x64xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 16, 22, 64], [1, 16, 21, 64], [1, 16, 21, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]],
    memory_shapes = [[1, 16, 22, 64], [1, 16, 21, 64], [1, 16, 21, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]]}>
!DistTypeEltwise = !VPU.DistributedTensor<1x8x64x64xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 8, 22, 64], [1, 8, 21, 64], [1, 8, 21, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]],
    memory_shapes = [[1, 8, 22, 64], [1, 8, 21, 64], [1, 8, 21, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]]}>
!DistTypeConcat = !VPU.DistributedTensor<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 16, 22, 64], [1, 16, 21, 64], [1, 16, 21, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]],
    memory_shapes = [[1, 16, 22, 64], [1, 16, 21, 64], [1, 16, 21, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]]}>
!DistTypeConvInput = !VPU.DistributedTensor<1x16x67x67xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 16, 23, 67], [1, 16, 22, 67], [1, 16, 22, 67]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 45, 0]],
    memory_shapes = [[1, 16, 25, 67], [1, 16, 24, 67], [1, 16, 24, 67]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]]}>
!DistTypeConvWeights = !VPU.DistributedTensor<8x16x4x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[8, 16, 4, 4], [8, 16, 4, 4], [8, 16, 4, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[8, 16, 4, 4], [8, 16, 4, 4], [8, 16, 4, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!DistTypeConv = !VPU.DistributedTensor<1x8x64x64xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 8, 22, 64], [1, 8, 21, 64], [1, 8, 21, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]],
    memory_shapes = [[1, 8, 22, 64], [1, 8, 21, 64], [1, 8, 21, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]]}>
!DistTypeConv2Weights = !VPU.DistributedTensor<96x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[96, 16, 1, 1], [96, 16, 1, 1], [96, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[96, 16, 1, 1], [96, 16, 1, 1], [96, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!DistTypeConv2 = !VPU.DistributedTensor<1x96x64x64xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 96, 22, 64], [1, 96, 21, 64], [1, 96, 21, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]],
    memory_shapes = [[1, 96, 23, 64], [1, 96, 23, 64], [1, 96, 22, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 42, 0]]}>

// CHECK-LABEL: @CannotInsertDummyNCEOpDueToAlignedChannels
module @CannotInsertDummyNCEOpDueToAlignedChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:      @main
    // CHECK-SAME:  [[ELTWISE_INPUT1:%[^:]+]]: !VPU.DistributedTensor
    // CHECK-SAME:  [[ELTWISE_INPUT2:%[^:]+]]: !VPU.DistributedTensor
    // CHECK-SAME:  [[CONV_INPUT:%.+]]: tensor<1x16x67x67xf16, {order = #NHWC}>
    // CHECK-SAME:  [[CONV_WEIGHTS:%.+]]: tensor<8x16x4x4xf16, {order = #NHWC}>
    // CHECK-SAME:  [[CONV2_WEIGHTS:%.+]]: tensor<96x16x1x1xf16, {order = #NHWC}
    func.func @main(%eltwise_input1: !DistTypeEltwiseInput, %eltwise_input2: !DistTypeEltwiseInput,
                    %conv_input: tensor<1x16x67x67xf16, {order = #NHWC}>, %conv_weights: tensor<8x16x4x4xf16, {order = #NHWC}>,
                    %conv2_weights: tensor<96x16x1x1xf16, {order = #NHWC}>)
            -> (!DistTypeEltwise, !DistTypeConv2) {
        %nce_eltwise = VPU.NCE.Eltwise(%eltwise_input1, %eltwise_input2) {
            input_padding = [0, 8, 0, 0], op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
        } -> !DistTypeEltwise
        %nce_eltwise_copy = VPU.Copy(%nce_eltwise) : !DistTypeEltwise -> tensor<1x8x64x64xf16, {order = #NHWC}>

        %conv_input_copy = VPU.Copy(%conv_input) {out_mem_space = @CMX_NN} : tensor<1x16x67x67xf16, {order = #NHWC}> -> !DistTypeConvInput
        %conv_weights_copy = VPU.Copy(%conv_weights) {out_mem_space = @CMX_NN} : tensor<8x16x4x4xf16, {order = #NHWC}> -> !DistTypeConvWeights
        %nce_conv = VPU.NCE.Convolution(%conv_input_copy, %conv_weights_copy) {
            input_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [8, 16, 4, 4], strides = [1, 1]
        } : !DistTypeConvInput, !DistTypeConvWeights -> !DistTypeConv
        %nce_conv_copy = VPU.Copy(%nce_conv) : !DistTypeConv -> tensor<1x8x64x64xf16, {order = #NHWC}>

        %concat = VPU.Concat(%nce_eltwise_copy, %nce_conv_copy) {static_offsets = [[0, 0, 0, 0], [0, 8, 0, 0]]} : tensor<1x8x64x64xf16, {order = #NHWC}>, tensor<1x8x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
        %concat_copy = VPU.Copy(%concat) {out_mem_space = @CMX_NN} : tensor<1x16x64x64xf16, {order = #NHWC}> -> !DistTypeConcat

        %conv2_weights_copy = VPU.Copy(%conv2_weights) {out_mem_space = @CMX_NN} : tensor<96x16x1x1xf16, {order = #NHWC}> -> !DistTypeConv2Weights
        %nce_conv2 = VPU.NCE.Convolution(%concat_copy, %conv2_weights_copy) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <LRELUX>, clamp_low = 0.000000e+00 : f64, clamp_high = 6.000000e+00 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [96, 16, 1, 1], strides = [1, 1]
        } : !DistTypeConcat, !DistTypeConv2Weights -> !DistTypeConv2

        return %nce_eltwise, %nce_conv2: !DistTypeEltwise, !DistTypeConv2
    }

    // CHECK:  [[NCE_ELTWISE:%.+]] = VPU.NCE.Eltwise([[ELTWISE_INPUT1]], [[ELTWISE_INPUT2]])
    // CHECK:  [[NCE_ELTWISE_COPY:%.+]] = VPU.Copy([[NCE_ELTWISE]])

    // CHECK:  [[CONV_INPUT_COPY:%.+]] = VPU.Copy([[CONV_INPUT]])
    // CHECK:  [[CONV_WEIGHTS_COPY:%.+]] = VPU.Copy([[CONV_WEIGHTS]])
    // CHECK:  [[NCE_CONV:%.+]] = VPU.NCE.Convolution([[CONV_INPUT_COPY]], [[CONV_WEIGHTS_COPY]])
    // CHECK:  [[NCE_CONV_COPY:%.+]] = VPU.Copy([[NCE_CONV]])

    // CHECK:  [[CONCAT:%.+]] = VPU.Concat([[NCE_ELTWISE_COPY]], [[NCE_CONV_COPY]])
    // CHECK:  [[CONCAT_COPY:%.+]] = VPU.Copy([[CONCAT]])

    // CHECK:  [[CONV2_WEIGHTS_COPY:%.+]] = VPU.Copy([[CONV2_WEIGHTS]])
    // CHECK:  [[NCE_CONV2:%.+]] = VPU.NCE.Convolution([[CONCAT_COPY]], [[CONV2_WEIGHTS_COPY]])

    // CHECK:  return [[NCE_ELTWISE]], [[NCE_CONV2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConvInputDDRType = !VPU.SparseTensor<
    data=tensor<1x32x16x16xf16, {order = #NHWC}>,
    sparsity_map=tensor<1x32x35x35xi1, {order = #NHWC}>,
    storage_element_table=tensor<1x1x35x35xi32, {order = #NHWC}>,
    #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>
>

!ConvInputType = !VPU.SparseTensor<
    data=tensor<1x32x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
    sparsity_map=tensor<1x32x35x35xi1, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
    storage_element_table=tensor<1x1x35x35xi32, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
    #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>
>

!ConvWeightsDDRType = !VPU.SparseTensor<
    data=tensor<8x32x4x4xf16, {order = #NHWC}>,
    sparsity_map=tensor<8x1x1x512xi1>, is_weights,
    #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<384> : tensor<8xi64>, alignment = 16 : i64>
>

!ConvWeightsType = !VPU.SparseTensor<
    data=tensor<8x32x4x4xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
    sparsity_map=tensor<8x1x1x512xi1, {mem_space = [@CMX_NN, 0], order = #NCHW}>, is_weights,
    #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<384> : tensor<8xi64>, alignment = 16 : i64>
>

// CHECK-LABEL: @CannotInsertDummyNCEOpDueToAlignedChannelsWithNonDistType
module @CannotInsertDummyNCEOpDueToAlignedChannelsWithNonDistType {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:      @main
    // CHECK-SAME:  [[ELTWISE_INPUT1:%[^:]+]]: tensor<1x16x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK-SAME:  [[ELTWISE_INPUT2:%[^:]+]]: tensor<1x16x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK-SAME:  [[CONV1_INPUT:%[^:]+]]: !VPU.SparseTensor
    // CHECK-SAME:  [[CONV1_WEIGHTS:%[^:]+]]: !VPU.SparseTensor
    // CHECK-SAME:  [[CONV2_WEIGHTS:%.+]]: tensor<96x16x1x1xf16, {order = #NHWC}>
    func.func @main(%eltwise_input1: tensor<1x16x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, %eltwise_input2: tensor<1x16x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, %conv1_input: !ConvInputDDRType, %conv1_weights: !ConvWeightsDDRType, %conv2_weights: tensor<96x16x1x1xf16, {order = #NHWC}>)
                    -> (tensor<1x8x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x96x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>) {
        %elt1 = VPU.NCE.Eltwise(%eltwise_input1, %eltwise_input2) {input_padding = [0, 8, 0, 0], op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
		       -> tensor<1x8x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        %elt1_copy = VPU.Copy(%elt1) : tensor<1x8x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x8x32x32xf16, {order = #NHWC}>

        %conv1_input_copy = VPU.Copy(%conv1_input) {out_mem_space = [@CMX_NN, 0]} : !ConvInputDDRType  -> !ConvInputType
        %conv1_weights_copy = VPU.Copy(%conv1_weights) {out_mem_space = [@CMX_NN, 0]} : !ConvWeightsDDRType -> !ConvWeightsType

        %conv1 = VPU.NCE.Convolution(%conv1_input_copy, %conv1_weights_copy) {
            input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [8, 32, 4, 4], strides = [1, 1]}
            : !ConvInputType, !ConvWeightsType  -> tensor<1x8x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        %conv1_copy = VPU.Copy(%conv1) : tensor<1x8x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x8x32x32xf16, {order = #NHWC}>

        %concat = VPU.Concat(%elt1_copy, %conv1_copy) {static_offsets = [[0, 0, 0, 0], [0, 8, 0, 0]]} : tensor<1x8x32x32xf16, {order = #NHWC}>, tensor<1x8x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

        %conv2_input_copy = VPU.Copy(%concat) {out_mem_space = [@CMX_NN, 0]} : tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        %conv2_weights_copy = VPU.Copy(%conv2_weights) {out_mem_space = [@CMX_NN, 0]} : tensor<96x16x1x1xf16, {order = #NHWC}> -> tensor<96x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        %conv2 = VPU.NCE.Convolution(%conv2_input_copy, %conv2_weights_copy) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <LRELUX>, clamp_low = 0.000000e+00 : f64, clamp_high = 6.000000e+00 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [96, 16, 1, 1], strides = [1, 1]}
                : tensor<1x16x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<96x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
                -> tensor<1x96x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        return %elt1 ,%conv2 : tensor<1x8x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x96x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    }

    // CHECK:  [[NCE_ELTWISE:%.+]] = VPU.NCE.Eltwise([[ELTWISE_INPUT1]], [[ELTWISE_INPUT2]])
    // CHECK:  [[NCE_ELTWISE_COPY:%.+]] = VPU.Copy([[NCE_ELTWISE]])

    // CHECK:  [[CONV1_INPUT_COPY:%.+]] = VPU.Copy([[CONV1_INPUT]])
    // CHECK:  [[CONV1_WEIGHTS_COPY:%.+]] = VPU.Copy([[CONV1_WEIGHTS]])
    // CHECK:  [[NCE_CONV1:%.+]] = VPU.NCE.Convolution([[CONV1_INPUT_COPY]], [[CONV1_WEIGHTS_COPY]])
    // CHECK:  [[NCE_CONV1_COPY:%.+]] = VPU.Copy([[NCE_CONV1]])

    // CHECK:  [[CONCAT:%.+]] = VPU.Concat([[NCE_ELTWISE_COPY]], [[NCE_CONV1_COPY]])
    // CHECK:  [[CONCAT_COPY:%.+]] = VPU.Copy([[CONCAT]])

    // CHECK:  [[CONV2_WEIGHTS_COPY:%.+]] = VPU.Copy([[CONV2_WEIGHTS]])
    // CHECK:  [[NCE_CONV2:%.+]] = VPU.NCE.Convolution([[CONCAT_COPY]], [[CONV2_WEIGHTS_COPY]])

    // CHECK:  return [[NCE_ELTWISE]], [[NCE_CONV2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.1>

!DistTypeConcat = !VPU.DistributedTensor<1x16x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!DistTypeConvInput = !VPU.DistributedTensor<1x16x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!DistTypeConvWeights = !VPU.DistributedTensor<8x16x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[8, 16, 1, 1], [8, 16, 1, 1], [8, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[8, 16, 1, 1], [8, 16, 1, 1], [8, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!DistTypeConv = !VPU.DistributedTensor<1x8x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 8, 1, 1], [1, 8, 1, 1], [1, 8, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 8, 1, 1], [1, 8, 1, 1], [1, 8, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!DistTypeConvOutWeights = !VPU.DistributedTensor<16x16x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!DistTypeConvOut = !VPU.DistributedTensor<1x16x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK-LABEL: @NCEOutputsAreNotAligned
module @NCEOutputsAreNotAligned {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:      @main
    // CHECK-SAME:  [[CONV_INPUT:%.+]]: tensor<1x16x1x1x!qElemType, {order = #NHWC}>
    // CHECK-SAME:  [[CONV_WEIGHTS:%.+]]: tensor<8x16x1x1x!qElemType, {order = #NHWC}>
    // CHECK-SAME:  [[CONV_OUT_WEIGHTS:%.+]]: tensor<16x16x1x1x!qElemType, {order = #NHWC}
    func.func @main(%conv_input: tensor<1x16x1x1x!qElemType, {order = #NHWC}>, %conv_weights: tensor<8x16x1x1x!qElemType, {order = #NHWC}>,
                    %conv_out_weights: tensor<16x16x1x1x!qElemType, {order = #NHWC}>)
            -> !DistTypeConvOut {
        %conv1_input_copy = VPU.Copy(%conv_input) {out_mem_space = @CMX_NN} : tensor<1x16x1x1x!qElemType, {order = #NHWC}> -> !DistTypeConvInput
        %conv1_weights_copy = VPU.Copy(%conv_weights) {out_mem_space = @CMX_NN} : tensor<8x16x1x1x!qElemType, {order = #NHWC}> -> !DistTypeConvWeights
        %conv1 = VPU.NCE.Convolution(%conv1_input_copy, %conv1_weights_copy) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [8, 16, 1, 1], strides = [1, 1]
        } : !DistTypeConvInput, !DistTypeConvWeights -> !DistTypeConv
        %conv1_copy = VPU.Copy(%conv1) : !DistTypeConv -> tensor<1x8x1x1x!qElemType, {order = #NHWC}>

        %conv2_input_copy = VPU.Copy(%conv_input) {out_mem_space = @CMX_NN} : tensor<1x16x1x1x!qElemType, {order = #NHWC}> -> !DistTypeConvInput
        %conv2_weights_copy = VPU.Copy(%conv_weights) {out_mem_space = @CMX_NN} : tensor<8x16x1x1x!qElemType, {order = #NHWC}> -> !DistTypeConvWeights
        %conv2 = VPU.NCE.Convolution(%conv2_input_copy, %conv2_weights_copy) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [8, 16, 1, 1], strides = [1, 1]
        } : !DistTypeConvInput, !DistTypeConvWeights -> !DistTypeConv
        %conv2_copy = VPU.Copy(%conv2) : !DistTypeConv -> tensor<1x8x1x1x!qElemType, {order = #NHWC}>

        %concat = VPU.Concat(%conv1_copy, %conv2_copy) {static_offsets = [[0, 0, 0, 0], [0, 8, 0, 0]]} : tensor<1x8x1x1x!qElemType, {order = #NHWC}>, tensor<1x8x1x1x!qElemType, {order = #NHWC}> -> tensor<1x16x1x1x!qElemType, {order = #NHWC}>
        %concat_copy = VPU.Copy(%concat) {out_mem_space = @CMX_NN} : tensor<1x16x1x1x!qElemType, {order = #NHWC}> -> !DistTypeConcat

        %conv_out_weights_copy = VPU.Copy(%conv_out_weights) {out_mem_space = @CMX_NN} : tensor<16x16x1x1x!qElemType, {order = #NHWC}> -> !DistTypeConvOutWeights
        %conv_out = VPU.NCE.Convolution(%concat_copy, %conv_out_weights_copy) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
        } : !DistTypeConcat, !DistTypeConvOutWeights -> !DistTypeConvOut

        return %conv_out: !DistTypeConvOut
    }

    // CHECK:  [[CONV1_INPUT_COPY:%.+]] = VPU.Copy([[CONV_INPUT]])
    // CHECK:  [[CONV1_WEIGHTS_COPY:%.+]] = VPU.Copy([[CONV_WEIGHTS]])
    // CHECK:  [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV1_INPUT_COPY]], [[CONV1_WEIGHTS_COPY]])
    // CHECK:  [[CONV1_COPY:%.+]] = VPU.Copy([[CONV1]])

    // CHECK:  [[CONV2_INPUT_COPY:%.+]] = VPU.Copy([[CONV_INPUT]])
    // CHECK:  [[CONV2_WEIGHTS_COPY:%.+]] = VPU.Copy([[CONV_WEIGHTS]])
    // CHECK:  [[CONV2:%.+]] = VPU.NCE.Convolution([[CONV2_INPUT_COPY]], [[CONV2_WEIGHTS_COPY]])
    // CHECK:  [[CONV2_COPY:%.+]] = VPU.Copy([[CONV2]])

    // CHECK:  [[CONCAT:%.+]] = VPU.Concat([[CONV1_COPY]], [[CONV2_COPY]])
    // CHECK:  [[CONCAT_COPY:%.+]] = VPU.Copy([[CONCAT]])

    // CHECK:  [[CONV_OUT_WEIGHTS_COPY:%.+]] = VPU.Copy([[CONV_OUT_WEIGHTS]])
    // CHECK:  [[CONV_OUT:%.+]] = VPU.NCE.Convolution([[CONCAT_COPY]], [[CONV_OUT_WEIGHTS_COPY]])

    // CHECK:  return [[CONV_OUT]]
}
