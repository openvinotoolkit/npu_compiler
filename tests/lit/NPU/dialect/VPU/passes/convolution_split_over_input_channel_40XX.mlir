//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --mlir-print-elementsattrs-with-hex-if-larger=-1 --convolution-split-over-input-channel %s | FileCheck %s
// REQUIRES: arch-NPU40XX


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE
// CHECK-LABEL: func.func @SplitOverInputChannelOn3T
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x3072x128x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<768x3072x1x1xf16, {order = #NHWC}>
func.func @SplitOverInputChannelOn3T(%arg0: tensor<1x3072x128x4xf16, {order = #NHWC}>, %arg1: tensor<768x3072x1x1xf16, {order = #NHWC}>) -> tensor<1x768x128x4xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %arg1, %cst) {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 3121953301 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [768, 3072, 1, 1], strides = [1, 1], tilingStrategy = [1, 2, 4, 1]
  } : tensor<1x3072x128x4xf16, {order = #NHWC}>, tensor<768x3072x1x1xf16, {order = #NHWC}>, tensor<768x1x1x4xsi32>
  -> tensor<1x768x128x4xf16, {order = #NHWC}>

  return %0 : tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[CONST0:%.+]] = const.Declare tensor<768x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:       dense<[[[[0, 0, 1065353216, 0]]], [[[3072, 192, 1065353216, 0]]], [[[6144, 384, 1065353216, 0]]]
  // CHECK:       [[CONST1:%.+]] = const.Declare tensor<768x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:       dense<[[[[0, 0, 1065353216, 1]]], [[[3072, 192, 1065353216, 1]]], [[[6144, 384, 1065353216, 1]]]

  // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT_0]]
  // CHECK-SAME:  [0, 0, 0, 0] [1, 1536, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1536x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT_1]]
  // CHECK-SAME:  [0, 0, 0, 0] [768, 1536, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1536x1x1xf16, {order = #NHWC}>

  // CHECK:       [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[SLICE1]], [[CONST1]])
  // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                        fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 8, 1, 1]
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT_0]]
  // CHECK-SAME:  [0, 1536, 0, 0] [1, 1536, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1536x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT_1]]
  // CHECK-SAME:  [0, 1536, 0, 0] [768, 1536, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1536x1x1xf16, {order = #NHWC}>

  // CHECK:       [[CONVOLUTION1:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[SLICE3]], [[CONST0]])
  // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                        fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 8, 1, 1]
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[CONVOLUTION0]], [[CONVOLUTION1]])
  // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 3121953301 : i64,
  // CHECK-SAME:                        quant_scale = [1.4012984643248171E-45], fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 1, 1, 1]} -> tensor<1x768x128x4xf16, {order = #NHWC}>
  // CHECK:       return [[ELTWISE0]] : tensor<1x768x128x4xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 6 of @NCE
// CHECK-LABEL: func.func @SplitOverInputChannelOn6T
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x3072x128x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<768x3072x1x1xf16, {order = #NHWC}>
func.func @SplitOverInputChannelOn6T(%arg0: tensor<1x3072x128x4xf16, {order = #NHWC}>, %arg1: tensor<768x3072x1x1xf16, {order = #NHWC}>) -> tensor<1x768x128x4xf16, {order = #NHWC}> {
  // scale = 1082549862 = 0x40866666 = 4.2f
  // bias will have the same value
  %cst = const.Declare tensor<768x1x1x4xsi32> = dense<1082549862> : tensor<768x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %arg1, %cst) {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e-01 : f64>,
    rawFilterShape = [768, 3072, 1, 1],
    strides = [1, 1], tilingStrategy = [1, 2, 4, 1]
  } : tensor<1x3072x128x4xf16, {order = #NHWC}>, tensor<768x3072x1x1xf16, {order = #NHWC}>, tensor<768x1x1x4xsi32>
  -> tensor<1x768x128x4xf16, {order = #NHWC}>

  return %0 : tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[CONST0:%.+]] = const.Declare tensor<768x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1065353216, 0]]], [[[3072, 192, 1065353216, 0]]], [[[6144, 384, 1065353216, 0]]]
  // CHECK:       [[CONST1:%.+]] = const.Declare tensor<768x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1065353216, 1082549862]]], [[[3072, 192, 1065353216, 1082549862]]], [[[6144, 384, 1065353216, 1082549862]]]

  // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT_0]]
  // CHECK-SAME:  [0, 0, 0, 0] [1, 1536, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1536x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT_1]]
  // CHECK-SAME:  [0, 0, 0, 0] [768, 1536, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1536x1x1xf16, {order = #NHWC}>

  // CHECK:       [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[SLICE1]], [[CONST1]])
  // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                        fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 1, 5, 1]
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT_0]]
  // CHECK-SAME:  [0, 1536, 0, 0] [1, 1536, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1536x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT_1]]
  // CHECK-SAME:  [0, 1536, 0, 0] [768, 1536, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1536x1x1xf16, {order = #NHWC}>

  // CHECK:       [[CONVOLUTION1:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[SLICE3]], [[CONST0]])
  // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                        fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 1, 5, 1]
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[CONVOLUTION0]], [[CONVOLUTION1]])
  // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                        quant_scale = [4.1999998092651367], fp_prelu_alpha = 1.000000e-01 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]} -> tensor<1x768x128x4xf16, {order = #NHWC}>
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
  %cst = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %arg1, %cst) {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [768, 3072, 1, 1], strides = [1, 1], tilingStrategy = [1, 3, 16, 1]
  } : tensor<1x3072x128x4xf16, {order = #NHWC}>, tensor<768x3072x1x1xf16, {order = #NHWC}>, tensor<768x1x1x4xsi32>
  -> tensor<1x768x128x4xf16, {order = #NHWC}>

  return %0 : tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[CONST0:%.+]] = const.Declare tensor<768x1x1x4xsi32>
  // CHECK-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[2048, 128, 1065353216, 0]]], [[[4096, 256, 1065353216, 0]]]
  // CHECK:       [[CONST1:%.+]] = const.Declare tensor<768x1x1x4xsi32>
  // CHECK-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 1]]], [[[2048, 128, 1065353216, 1]]], [[[4096, 256, 1065353216, 1]]]

  // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 1024, 128, 4]
  // CHECK-SAME:      : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1024x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [768, 1024, 1, 1]
  // CHECK-SAME:      : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1024x1x1xf16, {order = #NHWC}>

  // CHECK:       [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[SLICE1]], [[CONST1]])
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                        fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 7, 1, 1]
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 1024, 0, 0] [1, 1024, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1024x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 1024, 0, 0] [768, 1024, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1024x1x1xf16, {order = #NHWC}>

  // CHECK:       [[CONVOLUTION1:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[SLICE3]], [[CONST0]])
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64
  // CHECK-SAME:                        fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 7, 1, 1]
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[SLICE4:%.+]] = VPU.Slice [[INPUT_0]] [0, 2048, 0, 0] [1, 1024, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1024x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE5:%.+]] = VPU.Slice [[INPUT_1]] [0, 2048, 0, 0] [768, 1024, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1024x1x1xf16, {order = #NHWC}>

  // CHECK:       [[CONVOLUTION2:%.+]] = VPU.NCE.Convolution([[SLICE4]], [[SLICE5]], [[CONST0]])
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64
  // CHECK-SAME:                        fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 7, 1, 1]
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[CONVOLUTION0]], [[CONVOLUTION1]])
  // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64
  // CHECK-SAME:                        fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]} -> tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[ELTWISE0]], [[CONVOLUTION2]])
  // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                        quant_scale = [1.4012984643248171E-45], fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]} -> tensor<1x768x128x4xf16, {order = #NHWC}>
  // CHECK:       return [[ELTWISE1]] : tensor<1x768x128x4xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0084632095168618599:87>

// CHECK-LABEL: func.func @SplitOverInputChannelWithDequantizeOpAsWeights
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x3072x128x4xf16, {order = #NHWC}>
func.func @SplitOverInputChannelWithDequantizeOpAsWeights(%arg0: tensor<1x3072x128x4xf16, {order = #NHWC}>) -> tensor<1x768x128x4xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
  %cst_0 = const.Declare tensor<768x3072x1x1x!quant.uniform<u8:f16, 0.0084632095168618599:87>, {order = #NHWC}> = dense<1> : tensor<3072x768xui8>, [#const.Reshape<[1, 1, 3072, 768]>, #const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 0.0084632095168618599:87>>, #const.AffineReshape<[[0], [0], [1, 2], [3]], [1, 3072, 1, 768]>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>>, #const.AffineReshape<[[0], [0], [0], [1, 2, 3]], [768, 3072, 1, 1]>, #const.MemPermute<#NHWC, #NHWC>]
  %0 = VPU.Dequantize(%cst_0) {
    dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, tilingStrategy = [2, 1, 1, 1]
  } : tensor<768x3072x1x1x!quant.uniform<u8:f16, 0.0084632095168618599:87>, {order = #NHWC}> -> tensor<768x3072x1x1xf16, {order = #NHWC}>

  %1 = VPU.NCE.Convolution(%arg0, %0, %cst) {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [768, 3072, 1, 1], strides = [1, 1], tilingStrategy = [1, 2, 6, 1]
  } : tensor<1x3072x128x4xf16, {order = #NHWC}>, tensor<768x3072x1x1xf16, {order = #NHWC}>, tensor<768x1x1x4xsi32>
    -> tensor<1x768x128x4xf16, {order = #NHWC}>

  return %1 : tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK-DAG:       [[CONST0:%.+]] = const.Declare tensor<768x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}:    [[[0, 0, 1065353216, 1]]], [[[3072, 192, 1065353216, 1]]], [[[6144, 384, 1065353216, 1]]], [[[9216, 576, 1065353216, 1]]]
  // CHECK-DAG:       [[CONST1:%.+]] = const.Declare tensor<768x1536x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<3072x768xui8>, [
  // CHECK-DAG-SAME:                   #const.SubView<[1536, 0], [1536, 768]>

  // CHECK-DAG:       [[CONST2:%.+]] = const.Declare tensor<768x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}:    [[[0, 0, 1065353216, 1]]], [[[3072, 192, 1065353216, 1]]], [[[6144, 384, 1065353216, 1]]], [[[9216, 576, 1065353216, 1]]]
  // CHECK-DAG:       [[CONST3:%.+]] = const.Declare tensor<768x1536x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<3072x768xui8>, [
  // CHECK-DAG-SAME:                   #const.SubView<[0, 0], [1536, 768]>

  // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 1536, 128, 4]
  // CHECK-SAME:      : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1536x128x4xf16, {order = #NHWC}>
  // CHECK:       [[DEQUANTIZE0:%.+]] = VPU.Dequantize([[CONST3]]) {
  // CHECK-SAME:      dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, tilingStrategy = [1, 1, 1, 1]}
  // CHECK-SAME:      : tensor<768x1536x1x1x!qElemType, {order = #NHWC}> -> tensor<768x1536x1x1xf16, {order = #NHWC}>

  // CHECK:       [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[DEQUANTIZE0]], [[CONST2]])
  // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 1, 5, 1]

  // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT_0]] [0, 1536, 0, 0] [1, 1536, 128, 4]
  // CHECK-SAME:      : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1536x128x4xf16, {order = #NHWC}>
  // CHECK:       [[DEQUANTIZE1:%.+]] = VPU.Dequantize([[CONST1]]) {
  // CHECK-SAME:      dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, tilingStrategy = [1, 1, 1, 1]}
  // CHECK-SAME:      : tensor<768x1536x1x1x!qElemType, {order = #NHWC}> -> tensor<768x1536x1x1xf16, {order = #NHWC}>

  // CHECK:       [[CONVOLUTION1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[DEQUANTIZE1]], [[CONST0]])
  // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 1, 5, 1]

  // CHECK:       [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[CONVOLUTION0]], [[CONVOLUTION1]])
  // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
  // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.4012984643248171E-45], fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]

  // CHECK:       return [[ELTWISE0]] : tensor<1x768x128x4xf16, {order = #NHWC}>
}
