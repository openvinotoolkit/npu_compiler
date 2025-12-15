//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --merge-vertical-fusion-subgraphs %s | FileCheck %s
// REQUIRES: arch-NPU50XX


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
    %4 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <LRELUX>, clamp_low = 0.000000e+00 : f64, clamp_high = 6.000000e+00 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, rawFilterShape = [960, 160, 1, 1], strides = [1, 1]} : tensor<1x160x48x48xf16, {order = #NHWC}>, tensor<960x160x1x1xf16, {order = #NHWC}>, tensor<960x1x1x4xsi32> -> tensor<1x960x48x48xf16, {order = #NHWC}>
    VPU.Yield %4
  }
  %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x960x48x48xf16, {order = #NHWC}>, %cst_1 as %arg2: tensor<960x96x1x1xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<960x1x1x4xsi32>) attributes {tilingStrategy = [1, 10, 1, 1]} -> tensor<1x960x48x48xf16, {order = #NHWC}> {
    %4 = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 4 : i64, right = 4 : i64, top = 4 : i64, bottom = 4 : i64>, ppe = #VPU.PPEFp<mode = <LRELUX>, clamp_low = 0.000000e+00 : f64, clamp_high = 6.000000e+00 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, rawFilterShape = [960, 1, 9, 9], strides = [1, 1]} -> tensor<1x960x48x48xf16, {order = #NHWC}>
    VPU.Yield %4
  }
  %3 = VPU.VerticalFusion (%2 as %arg1: tensor<1x960x48x48xf16, {order = #NHWC}>, %cst_3 as %arg2: tensor<160x960x1x1xf16, {order = #NHWC}>, %cst as %arg3: tensor<160x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x160x48x48xf16> {
    %4 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, rawFilterShape = [160, 960, 1, 1], strides = [1, 1]} : tensor<1x960x48x48xf16, {order = #NHWC}>, tensor<160x960x1x1xf16, {order = #NHWC}>, tensor<160x1x1x4xsi32> -> tensor<1x160x48x48xf16>
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
      %2 = VPU.NCE.Convolution(%arg3, %arg4, %arg5) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, rawFilterShape = [7168, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1x1xf16, {order = #NHWC}>, tensor<7168x4096x1x1x!qElemType, {order = #NHWC}>, tensor<7168x1x1x4xsi32> -> tensor<1x7168x1x1xf16, {order = #NHWC}>
      VPU.Yield %2
    }

    %1 = VPU.VerticalFusion (%0 as %arg3: tensor<1x7168x1x1xf16, {order = #NHWC}>, %arg2 as %arg4: tensor<1x7168x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x7168x1x1xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Eltwise(%arg3, %arg4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, op_type = #VPU.eltwise_type<MULTIPLY>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x7168x1x1xf16, {order = #NHWC}>
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
