//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --merge-vertical-fusion-subgraphs="enable-vertical-fusion-pipelining=true workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @BuildSubgraphWithTwoDimTilingIncludingC
//CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x2304x4x144xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x2304x4x144xf16, {order = #NHWC}>)
func.func @BuildSubgraphWithTwoDimTilingIncludingC(%arg0: tensor<1x2304x4x144xf16, {order = #NHWC}>, %arg1: tensor<1x2304x4x144xf16, {order = #NHWC}>) -> tensor<1x2304x4x144xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<2304x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<2304x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<2304x1x1x4xsi32> = dense<1> : tensor<2304x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x2304x4x144xf16, {order = #NHWC}>,
                             %arg1 as %arg3: tensor<1x2304x4x144xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 2, 1, 1]}
                              -> tensor<1x2304x4x144xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Eltwise(%arg2, %arg3) {is_inplace = true,
                                          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                          op_type = #VPU.eltwise_type<ADD>,
                                          ppe = #VPU.PPEFp<mode = <NOOP>,
                                          clamp_low = -3.4028234663852886E+38 : f64,
                                          clamp_high = 3.4028234663852886E+38 : f64,
                                          scale = 1.000000e+00 : f64,
                                          prelu_alpha = [1.000000e+00],
                                          bias = 0.000000e+00 : f64,
                                          adder = 0.000000e+00 : f64>} -> tensor<1x2304x4x144xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x2304x4x144xf16, {order = #NHWC}>,
                             %cst as %arg3: tensor<2304x16x1x1xf16, {order = #NHWC}>,
                             %cst_0 as %arg4: tensor<2304x1x1x4xsi32>) attributes {tilingStrategy = [1, 4, 1, 1]} -> tensor<1x2304x4x144xf16, {order = #NHWC}> {
      %2 = VPU.NCE.DepthConvolution(%arg2, %arg3, %arg4) rawFilterShape [2304, 1, 1, 1] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                                          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                                          ppe = #VPU.PPEFp<mode = <NOOP>,
                                                          clamp_low = -3.4028234663852886E+38 : f64,
                                                          clamp_high = 3.4028234663852886E+38 : f64,
                                                          scale = 1.000000e+00 : f64,
                                                          prelu_alpha = [1.000000e+00],
                                                          bias = 0.000000e+00 : f64,
                                                          adder = 0.000000e+00 : f64>,
                                                          
                                                          strides = [1, 1]} -> tensor<1x2304x4x144xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    return %1 : tensor<1x2304x4x144xf16, {order = #NHWC}>


    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
    //CHECK-SAME:           attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 4, 1, 4]} -> tensor<1x2304x4x144xf16, {order = #NHWC}> {
    //CHECK: [[ELTWISE:%.+]] = VPU.NCE.Eltwise
    //CHECK: [[CONV:%.+]] = VPU.NCE.DepthConvolution([[ELTWISE]]
    //CHECK: VPU.Yield [[CONV]]
    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x2304x4x144xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotMergeICSplitConvolutionsWithAccumulator
func.func @NotMergeICSplitConvolutionsWithAccumulator(
    %input0: tensor<1x1376x256x4xf16, {order = #NHWC}>,
    %weights0: tensor<4096x1376x1x1xf16, {order = #NHWC}>,
    %input1: tensor<1x1376x256x4xf16, {order = #NHWC}>,
    %weights1: tensor<4096x1376x1x1xf16, {order = #NHWC}>)
    -> tensor<1x4096x256x4xf16, {order = #NHWC}> {

  %ic_tile_1 = VPU.VerticalFusion (
      %input1 as %arg0: tensor<1x1376x256x4xf16, {order = #NHWC}>,
      %weights1 as %arg1: tensor<4096x1376x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 29, 1, 1]} -> tensor<1x4096x256x4xf16, {order = #NHWC}> {
    %output = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [4096, 1376, 1, 1] {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]} : tensor<1x1376x256x4xf16, {order = #NHWC}>, tensor<4096x1376x1x1xf16, {order = #NHWC}> -> tensor<1x4096x256x4xf16, {order = #NHWC}>
    VPU.Yield %output
  }

  %ic_tile_0 = VPU.VerticalFusion (
      %input0 as %arg0: tensor<1x1376x256x4xf16, {order = #NHWC}>,
      %weights0 as %arg1: tensor<4096x1376x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 29, 1, 1]} -> tensor<1x4096x256x4xf16, {order = #NHWC}> {
    %output = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [4096, 1376, 1, 1] {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]} : tensor<1x1376x256x4xf16, {order = #NHWC}>, tensor<4096x1376x1x1xf16, {order = #NHWC}> -> tensor<1x4096x256x4xf16, {order = #NHWC}>
    VPU.Yield %output
  }

  %accumulator = VPU.VerticalFusion (
      %ic_tile_0 as %arg0: tensor<1x4096x256x4xf16, {order = #NHWC}>,
      %ic_tile_1 as %arg1: tensor<1x4096x256x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 6, 1, 1]} -> tensor<1x4096x256x4xf16, {order = #NHWC}> {
    %output = VPU.NCE.Eltwise(%arg0, %arg1) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>} -> tensor<1x4096x256x4xf16, {order = #NHWC}>
    VPU.Yield %output
  }

  return %accumulator : tensor<1x4096x256x4xf16, {order = #NHWC}>

  // CHECK: [[VF0:%.+]] = VPU.VerticalFusion
  // CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution
  // CHECK: [[VF1:%.+]] = VPU.VerticalFusion
  // CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution
  // CHECK: [[VF2:%.+]] = VPU.VerticalFusion
  // CHECK: [[ELTWISE:%.+]] = VPU.NCE.Eltwise
  // CHECK: return [[VF2]] : tensor<1x4096x256x4xf16, {order = #NHWC}>
}
