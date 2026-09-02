//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --merge-vertical-fusion-subgraphs %s | FileCheck %s
// REQUIRES: platform-NPU5010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK-LABEL: @DoNotMergeNotFitInCMXTiles
//CHECK-SAME:  [[INPUT:%.+]]: tensor<1x160x48x48xf16>
func.func @DoNotMergeNotFitInCMXTiles(%arg0: tensor<1x160x48x48xf16>) -> tensor<1x160x48x48xf16> {
  %cst = const.Declare tensor<160x1x1x4xsi32> = dense<1> : tensor<160x1x1x4xsi32>
  %cst_0 = const.Declare tensor<960x1x1x4xsi32> = dense<1> : tensor<960x1x1x4xsi32>
  %cst_1 = const.Declare tensor<960x96x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<960x1x1x3x3xf32>, [#const.Reshape<[960, 1, 3, 3]>, #const.CastElemType<f16>, #const.ExpandDilated<[4, 4]>, #const.Reorder<#NCHW>, #const.Reshape<[960, 81, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
  %cst_2 = const.Declare tensor<960x1x1x4xsi32> = dense<0> : tensor<960x1x1x4xsi32>
  %cst_3 = const.Declare tensor<160x960x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<160x960x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_4 = const.Declare tensor<960x160x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<960x160x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 160 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x160x48x48xf16, {order = #NHWC}>
  %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x160x48x48xf16, {order = #NHWC}>, %cst_4 as %arg2: tensor<960x160x1x1xf16, {order = #NHWC}>, %cst_2 as %arg3: tensor<960x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 6, 1]} -> tensor<1x960x48x48xf16, {order = #NHWC}> {
    %4 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) rawFilterShape [960, 160, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <LRELUX>, clamp_low = 0.000000e+00 : f64, clamp_high = 6.000000e+00 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x160x48x48xf16, {order = #NHWC}>, tensor<960x160x1x1xf16, {order = #NHWC}>, tensor<960x1x1x4xsi32> -> tensor<1x960x48x48xf16, {order = #NHWC}>
    VPU.Yield %4
  }
  %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x960x48x48xf16, {order = #NHWC}>, %cst_1 as %arg2: tensor<960x96x1x1xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<960x1x1x4xsi32>) attributes {tilingStrategy = [1, 10, 1, 1]} -> tensor<1x960x48x48xf16, {order = #NHWC}> {
    %4 = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3) rawFilterShape [960, 1, 9, 9] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 4 : i64, right = 4 : i64, top = 4 : i64, bottom = 4 : i64>, ppe = #VPU.PPEFp<mode = <LRELUX>, clamp_low = 0.000000e+00 : f64, clamp_high = 6.000000e+00 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} -> tensor<1x960x48x48xf16, {order = #NHWC}>
    VPU.Yield %4
  }
  %3 = VPU.VerticalFusion (%2 as %arg1: tensor<1x960x48x48xf16, {order = #NHWC}>, %cst_3 as %arg2: tensor<160x960x1x1xf16, {order = #NHWC}>, %cst as %arg3: tensor<160x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x160x48x48xf16> {
    %4 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) rawFilterShape [160, 960, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x960x48x48xf16, {order = #NHWC}>, tensor<160x960x1x1xf16, {order = #NHWC}>, tensor<160x1x1x4xsi32> -> tensor<1x160x48x48xf16>
    VPU.Yield %4
  }
  return %3 : tensor<1x160x48x48xf16>

  //CHECK:  [[PERMUTE:%.+]] = VPU.NCE.Permute([[INPUT]]
  //CHECK:  [[VF_0:%.+]] = VPU.VerticalFusion ([[PERMUTE]]
  //CHECK:  VPU.NCE.Convolution
  //CHECK:  [[VF_1:%.+]] = VPU.VerticalFusion ([[VF_0]]
  //CHECK:  VPU.NCE.DepthConvolution
  //CHECK:  [[VF_2:%.+]] = VPU.VerticalFusion ([[VF_1]]
  //CHECK:  VPU.NCE.Convolution

  //CHECK: return [[VF_2]]  : tensor<1x160x48x48xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotMergeSegSokDpuEltwise
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x4096x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<7168x4096x1x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:  [[INPUT2:%.+]]: tensor<1x7168x1x1xf16, {order = #NHWC}>
func.func @NotMergeSegSokDpuEltwise(%arg0: tensor<1x4096x1x1xf16, {order = #NHWC}>,
          %arg1: tensor<7168x4096x1x1x!qElemType, {order = #NHWC}>,
          %arg2: tensor<1x7168x1x1xf16, {order = #NHWC}>) -> tensor<1x7168x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<7168x1x1x4xsi32> = dense<1> : tensor<14336x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [7168, 1, 1, 4]>]

    %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x4096x1x1xf16, {order = #NHWC}>, %arg1 as %arg4: tensor<7168x4096x1x1x!qElemType, {order = #NHWC}>, %cst as %arg5: tensor<7168x1x1x4xsi32>) attributes {tilingStrategy = [1, 4, 1, 1]} -> tensor<1x7168x1x1xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg3, %arg4, %arg5) rawFilterShape [7168, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x4096x1x1xf16, {order = #NHWC}>, tensor<7168x4096x1x1x!qElemType, {order = #NHWC}>, tensor<7168x1x1x4xsi32> -> tensor<1x7168x1x1xf16, {order = #NHWC}>
      VPU.Yield %2
    }

    %1 = VPU.VerticalFusion (%0 as %arg3: tensor<1x7168x1x1xf16, {order = #NHWC}>, %arg2 as %arg4: tensor<1x7168x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x7168x1x1xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Eltwise(%arg3, %arg4) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, op_type = #VPU.eltwise_type<MULTIPLY>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x7168x1x1xf16, {order = #NHWC}>
      VPU.Yield %2
    }

    return %1 : tensor<1x7168x1x1xf16, {order = #NHWC}>

    // CHECK: [[CST:%.+]] = const.Declare
    // CHECK: [[VF_0:%.+]] = VPU.VerticalFusion
    // CHECK:     VPU.NCE.Convolution
    // CHECK:     VPU.Yield

    // CHECK: [[VF_1:%.+]] = VPU.VerticalFusion
    // CHECK:     VPU.NCE.Eltwise
    // CHECK:     VPU.Yield

    // CHECK: return [[VF_1]]
}

// -----

!qElemType = !quant.uniform<!QuantileType.quantile<ui4:f16, {-3.500000e+00,-2.500000e+00,-1.875000e+00,-1.375000e+00,-1.000000e+00,-6.250000e-01,-3.125000e-01,0.000000e+00,2.812500e-01,5.625000e-01,8.750000e-01,1.125000e+00,1.500000e+00,2.000000e+00,2.500000e+00,3.500000e+00}>:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.0053171685763767785>
!qElemType2 = !quant.uniform<f8E4M3FN:f16, 0.0088348165154457092>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
}

config.Resources 1 of @NCE {
    config.MemoryResource 1473536 bytes of @CMX_NN {
      config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64
    }
}

// CHECK-LABEL: @MergeEltwiseWithInputArg
func.func @MergeEltwiseWithInputArg(%arg0: tensor<1x2048x1x1xf16, {order = #NHWC}>,
    %arg1: tensor<1024x2048x1x1x!qElemType, {order = #NHWC}>,
    %arg2: tensor<1x1024x1x1xf16, {order = #NHWC}>,
    %arg3: tensor<1x1024x1x1xf16, {order = #NHWC}>,
    %arg4: tensor<1x1x1x128xf32>,
    %arg5: tensor<1x1x1x128xf16>) -> (tensor<1x8x1x128xf16>, tensor<1x8x1x128xf16>, tensor<1x16x8x8x!qElemType1, {order = #NHWC}>) {
  %cst = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>

  %0 = VPU.VerticalFusion (%arg0 as %arg6: tensor<1x2048x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x2048x1x1x!qElemType2, {order = #NHWC}> {
    %inner = VPU.NCE.AveragePool(%arg6) {kernel_size = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -4.480000e+02 : f64, clamp_high = 4.480000e+02 : f64, scale = 113.18854197500568 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x2048x1x1x!qElemType2, {order = #NHWC}>
    VPU.Yield %inner
  }

  %1 = VPU.VerticalFusion (%0 as %arg6: tensor<1x2048x1x1x!qElemType2, {order = #NHWC}>, %arg1 as %arg7: tensor<1024x2048x1x1x!qElemType, {order = #NHWC}>, %cst as %arg8: tensor<1024x1x1x4xsi32>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x1024x1x1xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Convolution(%arg6, %arg7, %arg8) rawFilterShape [1024, 2048, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 0.0088348165154457092 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x2048x1x1x!qElemType2, {order = #NHWC}>, tensor<1024x2048x1x1x!qElemType, {order = #NHWC}>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }

  %2 = VPU.VerticalFusion (%1 as %arg6: tensor<1x1024x1x1xf16, {order = #NHWC}>, %arg2 as %arg7: tensor<1x1024x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1024x1x1xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Eltwise(%arg6, %arg7) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<MULTIPLY>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }

  %3 = VPU.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16>
  %4 = VPU.ShapeCast {shape = [1, 8, 1, 128]} inputs(%3 : tensor<1x1024x1x1xf16>) -> tensor<1x8x1x128xf16>

  %5 = VPU.VerticalFusion (%0 as %arg6: tensor<1x2048x1x1x!qElemType2, {order = #NHWC}>, %arg1 as %arg7: tensor<1024x2048x1x1x!qElemType, {order = #NHWC}>, %cst as %arg8: tensor<1024x1x1x4xsi32>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x1024x1x1xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Convolution(%arg6, %arg7, %arg8) rawFilterShape [1024, 2048, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 0.0088348165154457092 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x2048x1x1x!qElemType2, {order = #NHWC}>, tensor<1024x2048x1x1x!qElemType, {order = #NHWC}>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }

  %6 = VPU.VerticalFusion (%5 as %arg6: tensor<1x1024x1x1xf16, {order = #NHWC}>, %arg3 as %arg7: tensor<1x1024x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1024x1x1xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Eltwise(%arg6, %arg7) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<MULTIPLY>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }

  %7 = VPU.PermuteCast(%6) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16>
  %8 = VPU.ShapeCast {shape = [1, 1, 8, 128]} inputs(%7 : tensor<1x1024x1x1xf16>) -> tensor<1x1x8x128xf16>

  %9 = VPU.VerticalFusion (%arg4 as %arg6: tensor<1x1x1x128xf32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1x1x128xf16> {
    %inner = VPU.Convert(%arg6) {dstElemType = f16} : tensor<1x1x1x128xf32> -> tensor<1x1x1x128xf16>
    VPU.Yield %inner
  }

  %10 = VPU.VerticalFusion (%8 as %arg6: tensor<1x1x8x128xf16>, %9 as %arg7: tensor<1x1x1x128xf16>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1x8x128xf16> {
    %inner = VPU.RMS(%arg6, %arg7) {eps = 9.9999999747524271E-7 : f64} : tensor<1x1x8x128xf16>, tensor<1x1x1x128xf16> -> tensor<1x1x8x128xf16>
    VPU.Yield %inner
  }

  %11 = VPU.AffineReshape(%10) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 8, 1, 128]} : tensor<1x1x8x128xf16> -> tensor<1x8x1x128xf16>

  %12 = VPU.VerticalFusion (%11 as %arg6: tensor<1x8x1x128xf16>, %arg5 as %arg7: tensor<1x1x1x128xf16>, %arg5 as %arg8: tensor<1x1x1x128xf16>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x8x1x128xf16> {
    %inner = VPU.RoPE(%arg6, %arg7, %arg8) {mode = #IE.rope_mode<SPLIT_HALF>} : tensor<1x8x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16> -> tensor<1x8x1x128xf16>
    VPU.Yield %inner
  }

  %13 = VPU.PermuteCast(%4) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x8x1x128xf16> -> tensor<1x8x1x128xf16, {order = #NCWH}>
  %14 = VPU.LayoutCast(%13) {dst_order = #NHWC} : tensor<1x8x1x128xf16, {order = #NCWH}> -> tensor<1x8x1x128xf16, {order = #NHWC}>
  %15 = VPU.ShapeCast {shape = [1, 16, 8, 8]} inputs(%14 : tensor<1x8x1x128xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}>

  %16 = VPU.VerticalFusion (%15 as %arg6: tensor<1x16x8x8xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x16x8x8x!qElemType1, {order = #NHWC}> {
    %inner = VPU.NCE.AveragePool(%arg6) {kernel_size = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -4.480000e+02 : f64, clamp_high = 4.480000e+02 : f64, scale = 188.070019905485 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x16x8x8x!qElemType1, {order = #NHWC}>
    VPU.Yield %inner
  }

  return %4, %12, %16 : tensor<1x8x1x128xf16>, tensor<1x8x1x128xf16>, tensor<1x16x8x8x!qElemType1, {order = #NHWC}>

  // CHECK:       [[CST:%.+]] = const.Declare tensor<1024x1x1x4xsi32>

  // CHECK:       [[VF_AVGPOOL:%.+]] = VPU.VerticalFusion (%arg0
  // CHECK:           VPU.NCE.AveragePool
  // CHECK:           VPU.Yield

  // CHECK:       [[VF_CONV_ELT0:%.+]] = VPU.VerticalFusion ([[VF_AVGPOOL]]
  // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
  // CHECK:           VPU.NCE.Convolution
  // CHECK:           VPU.NCE.Eltwise
  // CHECK:           VPU.Yield

  // CHECK:       [[PERM0:%.+]] = VPU.PermuteCast([[VF_CONV_ELT0]])
  // CHECK:       [[SHAPE0:%.+]] = VPU.ShapeCast {shape = [1, 8, 1, 128]}

  // CHECK:       [[VF_CONV_ELT1:%.+]] = VPU.VerticalFusion ([[VF_AVGPOOL]]
  // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
  // CHECK:           VPU.NCE.Convolution
  // CHECK:           VPU.NCE.Eltwise
  // CHECK:           VPU.Yield

  // CHECK:       [[PERM1:%.+]] = VPU.PermuteCast([[VF_CONV_ELT1]])
  // CHECK:       [[SHAPE1:%.+]] = VPU.ShapeCast {shape = [1, 1, 8, 128]}

  // CHECK:       [[VF_CONVERT:%.+]] = VPU.VerticalFusion (%arg4
  // CHECK:           VPU.Convert
  // CHECK:           VPU.Yield

  // CHECK:       [[VF_RMS:%.+]] = VPU.VerticalFusion ([[SHAPE1]]
  // CHECK-SAME:      [[VF_CONVERT]]
  // CHECK:           VPU.RMS
  // CHECK:           VPU.Yield

  // CHECK:       [[RESHAPE:%.+]] = VPU.AffineReshape([[VF_RMS]])

  // CHECK:       [[VF_ROPE:%.+]] = VPU.VerticalFusion ([[RESHAPE]]
  // CHECK:           VPU.RoPE
  // CHECK:           VPU.Yield

  // CHECK:       [[PERM2:%.+]] = VPU.PermuteCast([[SHAPE0]])
  // CHECK:       [[LAYOUT:%.+]] = VPU.LayoutCast([[PERM2]])
  // CHECK:       [[SHAPE2:%.+]] = VPU.ShapeCast {shape = [1, 16, 8, 8]} inputs([[LAYOUT]]

  // CHECK:       [[VF_QUANT:%.+]] = VPU.VerticalFusion ([[SHAPE2]]
  // CHECK:           VPU.NCE.AveragePool
  // CHECK:           VPU.Yield

  // CHECK:       return [[SHAPE0]], [[VF_ROPE]], [[VF_QUANT]]
}
