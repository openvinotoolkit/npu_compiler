//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --relocate-weight-table-for-reuse --mlir-print-elementsattrs-with-hex-if-larger=-1 %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RelocateWtTableReuseOneCluster
func.func @RelocateWtTableReuseOneCluster(%input: tensor<1x512x28x28xf16, {order = #NHWC}>) -> tensor<1x16x28x28xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x512x1x1x!quant.uniform<i8:f16, 0.0025660674945980895>, {order = #NHWC}> = dense<1> : tensor<16x512x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<i8:f16, 0.0025660674945980895>>, #const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32> = dense<[[[[512, 0, 992488312, 1114456573]]], [[[1024, 0, 992488312, -1056687949]]], [[[1536, 0, 992488312, 1098861863]]], [[[2048, 0, 992488312, 1097001740]]], [[[2560, 0, 992488312, -1030979791]]], [[[3072, 0, 992488312, 1109445115]]], [[[3584, 0, 992488312, 1093641443]]], [[[4096, 0, 992488312, 1118011884]]], [[[4608, 0, 992488312, 1093071328]]], [[[5120, 0, 992488312, -1035068378]]], [[[5632, 0, 992488312, -1038370697]]], [[[6144, 0, 992488312, -1034779696]]], [[[6656, 0, 992488312, -1039789946]]], [[[7168, 0, 992488312, -1055977720]]], [[[7680, 0, 992488312, 1102850196]]], [[[8192, 0, 992488312, 1103977139]]]]> : tensor<16x1x1x4xsi32>

    %conv = VPU.NCE.Convolution(%input, %weights, %wt)
                {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                fp_prelu_alpha = 0.0025660675019025803 : f64>, rawFilterShape = [16, 512, 1, 1], strides = [1, 1]}
                -> tensor<1x16x28x28xf16, {order = #NHWC}>

    return %conv : tensor<1x16x28x28xf16, {order = #NHWC}>

    // CHECK:                [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> =
    // CHECK-SAME{LITERAL}:    dense<[[[[0, 16777215, 992488312, 1114456573]]], [[[512, 16777215, 992488312, -1056687949]]],
    // CHECK-SAME{LITERAL}:    [[[1024, 16777215, 992488312, 1098861863]]], [[[1536, 16777215, 992488312, 1097001740]]],
    // CHECK-SAME{LITERAL}:    [[[7168, 16777215, 992488312, 1102850196]]], [[[7680, 16777215, 992488312, 1103977139]]]]> : tensor<16x1x1x4xsi32>

    // CHECK:                VPU.NCE.Convolution({{%.+}}, {{%.+}}, [[WEIGHTS_TABLE]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InDistributed = !VPU.DistributedTensor<1x128x1x1xf16, #NHWC, @CMX_NN,
    {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1], [1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 128, 1, 1], [1, 128, 1, 1], [1, 128, 1, 1], [1, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!WeightsDistributed = !VPU.DistributedTensor<64x128x1x1xf16, #NHWC, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    memory_shapes = [[16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]}>

!WtDistributed = !VPU.DistributedTensor<64x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN,
  {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
  compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
  memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]}>

!OutDistributed = !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN,
  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
  compute_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]],
  compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
  memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1]],
  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK-LABEL: @RelocateWtTableReuseMultiCluster
func.func @RelocateWtTableReuseMultiCluster(%input: tensor<1x128x1x1xf16, {order = #NHWC}>, %weights: tensor<64x128x1x1xf16, {order = #NHWC}>) -> !OutDistributed {
    %wt = const.Declare tensor<64x1x1x4xsi32> = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]], [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]], [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]], [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]], [[[4096, 0, 1065353216, 0]]], [[[4352, 0, 1065353216, 0]]], [[[4608, 0, 1065353216, 0]]], [[[4864, 0, 1065353216, 0]]], [[[5120, 0, 1065353216, 0]]], [[[5376, 0, 1065353216, 0]]], [[[5632, 0, 1065353216, 0]]], [[[5888, 0, 1065353216, 0]]], [[[6144, 0, 1065353216, 0]]], [[[6400, 0, 1065353216, 0]]], [[[6656, 0, 1065353216, 0]]], [[[6912, 0, 1065353216, 0]]], [[[7168, 0, 1065353216, 0]]], [[[7424, 0, 1065353216, 0]]], [[[7680, 0, 1065353216, 0]]], [[[7936, 0, 1065353216, 0]]], [[[8192, 0, 1065353216, 0]]], [[[8448, 0, 1065353216, 0]]], [[[8704, 0, 1065353216, 0]]], [[[8960, 0, 1065353216, 0]]], [[[9216, 0, 1065353216, 0]]], [[[9472, 0, 1065353216, 0]]], [[[9728, 0, 1065353216, 0]]], [[[9984, 0, 1065353216, 0]]], [[[10240, 0, 1065353216, 0]]], [[[10496, 0, 1065353216, 0]]], [[[10752, 0, 1065353216, 0]]], [[[11008, 0, 1065353216, 0]]], [[[11264, 0, 1065353216, 0]]], [[[11520, 0, 1065353216, 0]]], [[[11776, 0, 1065353216, 0]]], [[[12032, 0, 1065353216, 0]]], [[[12288, 0, 1065353216, 0]]], [[[12544, 0, 1065353216, 0]]], [[[12800, 0, 1065353216, 0]]], [[[13056, 0, 1065353216, 0]]], [[[13312, 0, 1065353216, 0]]], [[[13568, 0, 1065353216, 0]]], [[[13824, 0, 1065353216, 0]]], [[[14080, 0, 1065353216, 0]]], [[[14336, 0, 1065353216, 0]]], [[[14592, 0, 1065353216, 0]]], [[[14848, 0, 1065353216, 0]]], [[[15104, 0, 1065353216, 0]]], [[[15360, 0, 1065353216, 0]]], [[[15616, 0, 1065353216, 0]]], [[[15872, 0, 1065353216, 0]]], [[[16128, 0, 1065353216, 0]]]]> : tensor<64x1x1x4xsi32>
    %distr_in = VPU.UnrolledType(%input : tensor<1x128x1x1xf16, {order = #NHWC}>) -> !InDistributed

    %distr_weights = VPU.UnrolledType(%weights : tensor<64x128x1x1xf16, {order = #NHWC}>) -> !WeightsDistributed
    %distr_wt = VPU.UnrolledType(%wt : tensor<64x1x1x4xsi32>) -> !WtDistributed
    %conv = VPU.NCE.Convolution(%distr_in, %distr_weights, %distr_wt) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>,
            clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]}
            -> !OutDistributed


    return %conv : !OutDistributed

    // CHECK:                [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> =
    // CHECK-SAME{LITERAL}:    dense<[[[[0, 16777215, 1065353216, 0]]], [[[256, 16777215, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:    [[[3840, 16777215, 1065353216, 0]]], [[[0, 16777215, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:    [[[3840, 16777215, 1065353216, 0]]], [[[0, 16777215, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:    [[[3840, 16777215, 1065353216, 0]]], [[[0, 16777215, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:    [[[3584, 16777215, 1065353216, 0]]], [[[3840, 16777215, 1065353216, 0]]]]> : tensor<64x1x1x4xsi32>

    // CHECK:                [[DISTR_WEIGHTS_TABLE:%.+]] = VPU.UnrolledType(%cst : tensor<64x1x1x4xsi32>)

    // CHECK:                VPU.NCE.Convolution({{%.+}}, {{%.+}}, [[DISTR_WEIGHTS_TABLE]])
}
