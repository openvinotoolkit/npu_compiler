//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% weights-table-reuse-mode=ENABLED" --relocate-weight-table-for-reuse --mlir-print-elementsattrs-with-hex-if-larger=-1 %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:   @NotRelocateForNewWeightTableFormatConv
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x64x64xf16, {order = #NHWC}>
func.func @NotRelocateForNewWeightTableFormatConv(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x512x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x5x5xf16>, [#const.Reorder<#NHWC>]
    %scale_table = const.Declare tensor<256x1x1x1xf32> = dense<1.0> : tensor<256x1x1x1xf32>
    %bias_table = const.Declare tensor<256x1x1x1xf32> = dense<2.0> : tensor<256x1x1x1xf32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [256, 512, 5, 5] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
        ppe = #VPU.PPEStub<>,
        
        strides = [1, 1],
        tilingStrategy = [1, 4, 4, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
    } : tensor<1x512x64x64xf16, {order = #NHWC}>, tensor<256x512x5x5xf16, {order = #NHWC}>, tensor<256x1x1x1xf32>, tensor<256x1x1x1xf32> -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>


    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<256x512x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x5x5xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG: [[SCALE_TABLE:%.+]] = const.Declare tensor<256x1x1x1xf32> = dense<1.000000e+00> : tensor<256x1x1x1xf32>
    // CHECK-DAG: [[BIAS_TABLE:%.+]] = const.Declare tensor<256x1x1x1xf32> = dense<2.000000e+00> : tensor<256x1x1x1xf32>
    // CHECK:     [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK      return [[CONV]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

// Negative test: NPU60XX uses new weight table format, pass is a no-op: weight table remains unchanged

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotRelocateWeightTableForNewFormatSCF
func.func @NotRelocateWeightTableForNewFormatSCF(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>, %arg1: tensor<64x128x1x1xf16, {order = #NHWC}>) -> tensor<1x64x1x1xf16, {order = #NHWC}> {
    %wt = const.Declare tensor<64x1x1x4xsi32> = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]], [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]], [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]], [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]], [[[4096, 0, 1065353216, 0]]], [[[4352, 0, 1065353216, 0]]], [[[4608, 0, 1065353216, 0]]], [[[4864, 0, 1065353216, 0]]], [[[5120, 0, 1065353216, 0]]], [[[5376, 0, 1065353216, 0]]], [[[5632, 0, 1065353216, 0]]], [[[5888, 0, 1065353216, 0]]], [[[6144, 0, 1065353216, 0]]], [[[6400, 0, 1065353216, 0]]], [[[6656, 0, 1065353216, 0]]], [[[6912, 0, 1065353216, 0]]], [[[7168, 0, 1065353216, 0]]], [[[7424, 0, 1065353216, 0]]], [[[7680, 0, 1065353216, 0]]], [[[7936, 0, 1065353216, 0]]], [[[8192, 0, 1065353216, 0]]], [[[8448, 0, 1065353216, 0]]], [[[8704, 0, 1065353216, 0]]], [[[8960, 0, 1065353216, 0]]], [[[9216, 0, 1065353216, 0]]], [[[9472, 0, 1065353216, 0]]], [[[9728, 0, 1065353216, 0]]], [[[9984, 0, 1065353216, 0]]], [[[10240, 0, 1065353216, 0]]], [[[10496, 0, 1065353216, 0]]], [[[10752, 0, 1065353216, 0]]], [[[11008, 0, 1065353216, 0]]], [[[11264, 0, 1065353216, 0]]], [[[11520, 0, 1065353216, 0]]], [[[11776, 0, 1065353216, 0]]], [[[12032, 0, 1065353216, 0]]], [[[12288, 0, 1065353216, 0]]], [[[12544, 0, 1065353216, 0]]], [[[12800, 0, 1065353216, 0]]], [[[13056, 0, 1065353216, 0]]], [[[13312, 0, 1065353216, 0]]], [[[13568, 0, 1065353216, 0]]], [[[13824, 0, 1065353216, 0]]], [[[14080, 0, 1065353216, 0]]], [[[14336, 0, 1065353216, 0]]], [[[14592, 0, 1065353216, 0]]], [[[14848, 0, 1065353216, 0]]], [[[15104, 0, 1065353216, 0]]], [[[15360, 0, 1065353216, 0]]], [[[15616, 0, 1065353216, 0]]], [[[15872, 0, 1065353216, 0]]], [[[16128, 0, 1065353216, 0]]]]> : tensor<64x1x1x4xsi32>
    %0 = tensor.empty() : tensor<1x64x1x1xf16, {order = #NHWC}>
    %1 = scf.forall (%arg2) = (0) to (64) step (32) shared_outs(%arg3 = %0) -> (tensor<1x64x1x1xf16, {order = #NHWC}>) {
      %extracted_slice = tensor.extract_slice %arg1[%arg2, 0, 0, 0] [32, 128, 1, 1] [1, 1, 1, 1] : tensor<64x128x1x1xf16, {order = #NHWC}> to tensor<32x128x1x1xf16, {order = #NHWC}>
      %extracted_slice_0 = tensor.extract_slice %wt[%arg2, 0, 0, 0] [32, 1, 1, 4] [1, 1, 1, 1] : tensor<64x1x1x4xsi32> to tensor<32x1x1x4xsi32>
      %2 = VPU.NCE.Convolution(%arg0, %extracted_slice, %extracted_slice_0) rawFilterShape [32, 128, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x128x1x1xf16, {order = #NHWC}>, tensor<32x128x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x1x1xf16, {order = #NHWC}>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %2 into %arg3[0, %arg2, 0, 0] [1, 32, 1, 1] [1, 1, 1, 1] : tensor<1x32x1x1xf16, {order = #NHWC}> into tensor<1x64x1x1xf16, {order = #NHWC}>
      }
    }

    // Weight table is NOT relocated on NPU60XX (new weight table format)
    // CHECK:                [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> =
    // CHECK-SAME{LITERAL}:    dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:    [[[15616, 0, 1065353216, 0]]], [[[15872, 0, 1065353216, 0]]], [[[16128, 0, 1065353216, 0]]]]> : tensor<64x1x1x4xsi32>

    // CHECK:                scf.forall
    // CHECK:                  tensor.extract_slice [[WEIGHTS_TABLE]]
    // CHECK:                  VPU.NCE.Convolution
    return %1 : tensor<1x64x1x1xf16, {order = #NHWC}>
}
