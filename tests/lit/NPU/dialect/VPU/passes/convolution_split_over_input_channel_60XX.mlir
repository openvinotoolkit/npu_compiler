//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --convolution-split-over-input-channel %s | FileCheck %s
// REQUIRES: platform-NPU6010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE
// CHECK-LABEL: func.func @SplitOverInputChannelOn3T
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x3072x128x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<768x3072x1x1xf16, {order = #NHWC}>
func.func @SplitOverInputChannelOn3T(%arg0: tensor<1x3072x128x4xf16, {order = #NHWC}>, %arg1: tensor<768x3072x1x1xf16, {order = #NHWC}>) -> tensor<1x768x128x4xf16, {order = #NHWC}> {
  %scale = const.Declare tensor<768x1x1x1xf32> = dense<1.0> : tensor<768x1x1x1xf32>
  %bias = const.Declare tensor<768x1x1x1xf32> = dense<0.0> : tensor<768x1x1x1xf32>
  %0 = VPU.NCE.Convolution(%arg0, %arg1, %scale, %bias) rawFilterShape [768, 3072, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
     strides = [1, 1], tilingStrategy = [1, 2, 4, 1],
    operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
  } : tensor<1x3072x128x4xf16, {order = #NHWC}>, tensor<768x3072x1x1xf16, {order = #NHWC}>, tensor<768x1x1x1xf32>, tensor<768x1x1x1xf32> -> tensor<1x768x128x4xf16, {order = #NHWC}>

  return %0 : tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<768x1x1x1xf32> = dense<1.000000e+00> : tensor<768x1x1x1xf32>
  // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<768x1x1x1xf32> = dense<0.000000e+00> : tensor<768x1x1x1xf32>

  // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT_0]]
  // CHECK-SAME:  [0, 0, 0, 0] [1, 1536, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1536x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT_1]]
  // CHECK-SAME:  [0, 0, 0, 0] [768, 1536, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1536x1x1xf16, {order = #NHWC}>
  // CHECK:       [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[SLICE1]], [[SCALE_TABLE]], [[BIAS_TABLE]])
  // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
  // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 1, 5, 1]
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT_0]]
  // CHECK-SAME:  [0, 1536, 0, 0] [1, 1536, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1536x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT_1]]
  // CHECK-SAME:  [0, 1536, 0, 0] [768, 1536, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1536x1x1xf16, {order = #NHWC}>
  // CHECK:       [[CONVOLUTION1:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[SLICE3]], [[SCALE_TABLE]], [[BIAS_TABLE]])
  // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
  // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 1, 5, 1]
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>
  // CHECK:       [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[CONVOLUTION0]], [[CONVOLUTION1]])
  // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
  // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:            scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 1, 1, 1]} -> tensor<1x768x128x4xf16, {order = #NHWC}>
  // CHECK:       return [[ELTWISE0]] : tensor<1x768x128x4xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 1 of @NCE
// CHECK-LABEL: func.func @SplitOverInputChannelOn1T
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x3072x128x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<768x3072x1x1xf16, {order = #NHWC}>
func.func @SplitOverInputChannelOn1T(%arg0: tensor<1x3072x128x4xf16, {order = #NHWC}>, %arg1: tensor<768x3072x1x1xf16, {order = #NHWC}>) -> tensor<1x768x128x4xf16, {order = #NHWC}> {
  %scale = const.Declare tensor<768x1x1x1xf32> = dense<1.0> : tensor<768x1x1x1xf32>
  %bias = const.Declare tensor<768x1x1x1xf32> = dense<0.0> : tensor<768x1x1x1xf32>
  %0 = VPU.NCE.Convolution(%arg0, %arg1, %scale, %bias) rawFilterShape [768, 3072, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                      scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
     strides = [1, 1], tilingStrategy = [1, 3, 16, 1],
    operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
  } : tensor<1x3072x128x4xf16, {order = #NHWC}>, tensor<768x3072x1x1xf16, {order = #NHWC}>, tensor<768x1x1x1xf32>, tensor<768x1x1x1xf32> -> tensor<1x768x128x4xf16, {order = #NHWC}>

  return %0 : tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<768x1x1x1xf32> = dense<1.000000e+00> : tensor<768x1x1x1xf32>
  // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<768x1x1x1xf32> = dense<0.000000e+00> : tensor<768x1x1x1xf32>

  // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 1024, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1024x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [768, 1024, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1024x1x1xf16, {order = #NHWC}>
  // CHECK:       [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[SLICE1]], [[SCALE_TABLE]], [[BIAS_TABLE]])
  // CHECK-SAME:  ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 8, 1, 1]
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 1024, 0, 0] [1, 1024, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1024x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 1024, 0, 0] [768, 1024, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1024x1x1xf16, {order = #NHWC}>
  // CHECK:       [[CONVOLUTION1:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[SLICE3]], [[SCALE_TABLE]], [[BIAS_TABLE]])
  // CHECK-SAME:  ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 8, 1, 1]
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE4:%.+]] = VPU.Slice [[INPUT_0]] [0, 2048, 0, 0] [1, 1024, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1024x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE5:%.+]] = VPU.Slice [[INPUT_1]] [0, 2048, 0, 0] [768, 1024, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1024x1x1xf16, {order = #NHWC}>
  // CHECK:       [[CONVOLUTION2:%.+]] = VPU.NCE.Convolution([[SLICE4]], [[SLICE5]], [[SCALE_TABLE]], [[BIAS_TABLE]])
  // CHECK-SAME:  ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 8, 1, 1]
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>
  // CHECK:       [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[CONVOLUTION0]], [[CONVOLUTION1]])
  // CHECK-SAME:  {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:  tilingStrategy = [1, 2, 1, 1]
  // CHECK:       [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[ELTWISE0]], [[CONVOLUTION2]])
  // CHECK-SAME:  {op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:  tilingStrategy = [1, 2, 1, 1]} -> tensor<1x768x128x4xf16, {order = #NHWC}>
  // CHECK:       return [[ELTWISE1]] : tensor<1x768x128x4xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 1 of @NCE
// CHECK-LABEL: func.func @SplitOverInputChannelWithFP32Result
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x8192x256x4xf16, {order = #NHWC}>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<3072x8192x1x1xf16, {order = #NHWC}>
func.func @SplitOverInputChannelWithFP32Result(%arg0: tensor<1x8192x256x4xf16, {order = #NHWC}>, %arg1: tensor<3072x8192x1x1xf16, {order = #NHWC}>) -> tensor<1x3072x256x4xf32, {order = #NHWC}>  {
  %cst = const.Declare tensor<3072x1x1x4xsi32> = dense<1> : tensor<3072x1x1x4xsi32>

  %1 = VPU.NCE.Convolution(%arg0, %arg1, %cst) rawFilterShape [3072, 8192, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
     strides = [1, 1], tilingStrategy = [1, 32, 43, 1]
  } : tensor<1x8192x256x4xf16, {order = #NHWC}>, tensor<3072x8192x1x1xf16, {order = #NHWC}>, tensor<3072x1x1x4xsi32> -> tensor<1x3072x256x4xf32, {order = #NHWC}>

  return %1 : tensor<1x3072x256x4xf32, {order = #NHWC}>

  // CHECK:       [[CONST:%.+]] = const.Declare
  // CHECK-COUNT-32:       {{%.+ = VPU\.NCE\.Convolution}}
  // CHECK-COUNT-30:       {{%.+ = VPU\.NCE\.Eltwise.* tilingStrategy = \[1, 10, 1, 1\].*tensor<1x3072x256x4xf16, \{order = #NHWC\}>}}

  // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise
  // CHECK-SAME:         tilingStrategy = [1, 13, 1, 1]
  // CHECK-SAME:       -> tensor<1x3072x256x4xf32, {order = #NHWC}>
  // CHECK-NEXT:       return [[ELTWISE]] : tensor<1x3072x256x4xf32, {order = #NHWC}>
}
}
