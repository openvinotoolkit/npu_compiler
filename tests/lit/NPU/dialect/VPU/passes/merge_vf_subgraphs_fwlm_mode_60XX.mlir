//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --merge-vertical-fusion-subgraphs="workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvEltwiseAddMerge
module @ConvEltwiseAddMerge {
config.Resources 4 of @NCE at 1.700000e+03 MHz
config.MemoryResource 2096128 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
func.func @main(%arg0: tensor<1x960x64x64xf16, {order = #NHWC}>,
                %arg1: tensor<320x960x1x1xf16, {order = #NHWC}>,
                %arg2: tensor<1x320x64x64xf16, {order = #NHWC}>)
    -> tensor<1x320x64x64xf16, {order = #NHWC}> {

  // First VF: Conv producing main output
  %vf_conv = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x960x64x64xf16, {order = #NHWC}>,
                                 %arg1 as %arg4: tensor<320x960x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 11]}
        -> tensor<1x320x64x64xf16, {order = #NHWC}> {
    %conv_out = VPU.NCE.Convolution(%arg3, %arg4) rawFilterShape [320, 960, 1, 1] {
      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
      resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      strides = [1, 1]
    } : tensor<1x960x64x64xf16, {order = #NHWC}>, tensor<320x960x1x1xf16, {order = #NHWC}>
      -> tensor<1x320x64x64xf16, {order = #NHWC}>
    VPU.Yield %conv_out
  }

  // Second VF: Eltwise consuming Conv output and another input
  %vf_eltwise = VPU.VerticalFusion (%vf_conv as %arg3: tensor<1x320x64x64xf16, {order = #NHWC}>,
                                    %arg2 as %arg4: tensor<1x320x64x64xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 11, 1]}
        -> tensor<1x320x64x64xf16, {order = #NHWC}> {
    %add_out = VPU.NCE.Eltwise(%arg3, %arg4) {
      is_inplace = true,
      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      op_type = #VPU.eltwise_type<ADD>,
      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
      resultSegmentSizes = array<i32: 1, 0, 0, 0>
    } : tensor<1x320x64x64xf16, {order = #NHWC}>, tensor<1x320x64x64xf16, {order = #NHWC}>
      -> tensor<1x320x64x64xf16, {order = #NHWC}>
    VPU.Yield %add_out
  }

  //  CHECK: [[VF:%.+]] = VPU.VerticalFusion
  //  CHECK-SAME: {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 1, 8]}
  //  CHECK-NEXT: [[CONV_OUT:%.+]] = VPU.NCE.Convolution
  //  CHECK-NEXT: [[ELTWISE_OUT:%.+]] = VPU.NCE.Eltwise([[CONV_OUT]]
  //  CHECK-NEXT: VPU.Yield [[ELTWISE_OUT]]
  //  CHECK: return [[VF]] : tensor<1x320x64x64xf16, {order = #NHWC}>
  return %vf_eltwise : tensor<1x320x64x64xf16, {order = #NHWC}>
}
}
