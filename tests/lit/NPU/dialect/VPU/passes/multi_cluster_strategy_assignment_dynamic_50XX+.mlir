//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=HostCompile allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputConvDynamicType = tensor<1x32x?x?x!quant.uniform<u8:f16, 0.0013198380376778396:130>, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
!inputD2SDynamicType = tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
!inputEltwiseDynamicType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
!outputDynamicType = tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

module @ConvD2SEltwise {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }
// CHECK-LABEL: @ConvD2SEltwise
func.func @ConvD2SEltwise(%arg0: !inputConvDynamicType, %arg1: !inputEltwiseDynamicType) -> !outputDynamicType {
  %weights = const.Declare tensor<12x32x3x3x!quant.uniform<u8:f16, 0.0060863536946913774:128>, {order = #NHWC}> = dense<1> : tensor<16x32x3x3xui8, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [12, 32, 3, 3]>, #const.CastElemType<!quant.uniform<u8:f16, 0.0060863536946913774:128>>]

  %0 = VPU.NCE.Convolution(%arg0, %weights) {
    input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
    rawFilterShape = [12, 32, 3, 3], strides = [1, 1]
  } : !inputConvDynamicType, tensor<12x32x3x3x!quant.uniform<u8:f16, 0.0060863536946913774:128>, {order = #NHWC}> -> !inputD2SDynamicType

  // CHECK: VPU.NCE.Convolution
  // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

  %1 = VPU.DepthToSpace(%0) {
    block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, padded_channels = #IE.ChannelPadding<input = 0 : i64, output = 13 : i64>
  } : !inputD2SDynamicType -> !inputEltwiseDynamicType

  // CHECK: VPU.DepthToSpace
  // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

  %2 = VPU.NCE.Eltwise(%1, %arg1) {
    input_padding = [0, 13, 0, 0], op_type = #VPU.eltwise_type<ADD>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  } -> !outputDynamicType

  // CHECK: VPU.NCE.Eltwise
  // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

  return %2 : !outputDynamicType
}
}
