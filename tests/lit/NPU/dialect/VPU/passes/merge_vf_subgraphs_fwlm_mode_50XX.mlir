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
      %2 = VPU.NCE.Eltwise(%arg2, %arg3) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true,
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
      %2 = VPU.NCE.DepthConvolution(%arg2, %arg3, %arg4) rawFilterShape [2304, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
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
!up_proj_qtype = !quant.uniform<u8:f16, 0.0012835108182009528:127>
!down_proj_qtype = !quant.uniform<i8:f16, 0.0015691562026154762>

config.Resources 1 of @NCE at 2.100000e+03 MHz {
  config.MemoryResource 2096128 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}
// CHECK-LABEL: @MergeDequantWithInvalidCost
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x3072x16x4xf16, {order = #NHWC}>, [[MULTIPLY_INPUT:%.+]]: tensor<1x8192x16x4xf16, {order = #NHWC}>)
func.func @MergeDequantWithInvalidCost(
    %input: tensor<1x3072x16x4xf16, {order = #NHWC}>,
    %multiply_input: tensor<1x8192x16x4xf16, {order = #NHWC}>)
  -> tensor<1x3072x16x4xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<8192x3072x1x1x!up_proj_qtype, {order = #NHWC}> = dense<1> : tensor<8192x3072x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!up_proj_qtype>, #const.Reorder<#NHWC>]
  %weight_table = const.Declare tensor<8192x1x1x4xsi32> = dense<1> : tensor<8192x1x1x4xsi32>
  %down_weights = const.Declare tensor<3072x8192x1x1x!down_proj_qtype, {order = #NHWC}> = dense<1> : tensor<3072x8192x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 0.0015691562026154762:128>>, #const.ConvertElemType<!down_proj_qtype>, #const.Reorder<#NHWC>]
  %down_weight_table = const.Declare tensor<3072x1x1x4xsi32> = dense<1> : tensor<3072x1x1x4xsi32>

  %dequant = VPU.VerticalFusion (%weights as %arg0: tensor<8192x3072x1x1x!up_proj_qtype, {order = #NHWC}>) attributes {tilingStrategy = [40, 1, 1, 1]} -> tensor<8192x3072x1x1xf16, {order = #NHWC}> {
    %output = VPU.Dequantize(%arg0) {dstElemType = f16} : tensor<8192x3072x1x1x!up_proj_qtype, {order = #NHWC}> -> tensor<8192x3072x1x1xf16, {order = #NHWC}>
    VPU.Yield %output
  }

  %conv = VPU.VerticalFusion (
      %input as %arg0: tensor<1x3072x16x4xf16, {order = #NHWC}>,
      %dequant as %arg1: tensor<8192x3072x1x1xf16, {order = #NHWC}>,
      %weight_table as %arg2: tensor<8192x1x1x4xsi32>) attributes {tilingStrategy = [1, 128, 1, 1]} -> tensor<1x8192x16x4xf16, {order = #NHWC}> {
    %output = VPU.NCE.Convolution(%arg0, %arg1, %arg2) rawFilterShape [8192, 3072, 1, 1] {
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
        clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
        prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
        adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
        : tensor<1x3072x16x4xf16, {order = #NHWC}>, tensor<8192x3072x1x1xf16, {order = #NHWC}>, tensor<8192x1x1x4xsi32>
        -> tensor<1x8192x16x4xf16, {order = #NHWC}>
    VPU.Yield %output
  }

  %multiply = VPU.VerticalFusion (
      %conv as %arg0: tensor<1x8192x16x4xf16, {order = #NHWC}>,
      %multiply_input as %arg1: tensor<1x8192x16x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x8192x16x4xf16, {order = #NHWC}> {
    %output = VPU.NCE.Eltwise(%arg0, %arg1) {is_inplace = true,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<MULTIPLY>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
        clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
        prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
        adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>}
        -> tensor<1x8192x16x4xf16, {order = #NHWC}>
    VPU.Yield %output
  }

  %down_conv = VPU.VerticalFusion (
      %multiply as %arg0: tensor<1x8192x16x4xf16, {order = #NHWC}>,
      %down_weights as %arg1: tensor<3072x8192x1x1x!down_proj_qtype, {order = #NHWC}>,
      %down_weight_table as %arg2: tensor<3072x1x1x4xsi32>) attributes {tilingStrategy = [1, 28, 1, 1]} -> tensor<1x3072x16x4xf16, {order = #NHWC}> {
    %output = VPU.NCE.Convolution(%arg0, %arg1, %arg2) rawFilterShape [3072, 8192, 1, 1] {
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
        clamp_high = 3.4028234663852886E+38 : f64, scale = 0.0015691562026154762 : f64,
        prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
        adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
        : tensor<1x8192x16x4xf16, {order = #NHWC}>, tensor<3072x8192x1x1x!down_proj_qtype, {order = #NHWC}>, tensor<3072x1x1x4xsi32>
        -> tensor<1x3072x16x4xf16, {order = #NHWC}>
    VPU.Yield %output
  }

  return %down_conv : tensor<1x3072x16x4xf16, {order = #NHWC}>

  // Dequantize has invalid VPUNN cost, while it still can be merged with the convolution user
  // CHECK: [[VF:%.+]] = VPU.VerticalFusion
  // CHECK-SAME: attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 128, 1, 1]}
  // CHECK: VPU.Dequantize
  // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution
  // CHECK: VPU.Yield
  // CHECK: [[ELTWISE_VF:%.+]] = VPU.VerticalFusion ([[VF]]
  // CHECK-SAME: tilingStrategy = [1, 1, 2, 1]
  // CHECK: VPU.NCE.Eltwise
  // CHECK: VPU.Yield
  // The following Conv can not be fused because there is no benefit from the cost perspective
  // CHECK: [[DOWN_VF:%.+]] = VPU.VerticalFusion ([[ELTWISE_VF]]
  // CHECK-SAME: tilingStrategy = [1, 28, 1, 1]
  // CHECK: [[DOWN_CONV:%.+]] = VPU.NCE.Convolution
  // CHECK: VPU.Yield [[DOWN_CONV]]
  // CHECK: return [[DOWN_VF]] : tensor<1x3072x16x4xf16, {order = #NHWC}>
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
    %output = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
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
