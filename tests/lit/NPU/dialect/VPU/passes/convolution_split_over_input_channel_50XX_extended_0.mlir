//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convolution-split-over-input-channel %s | FileCheck %s
// REQUIRES: arch-NPU50XX


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE
// CHECK-LABEL: func.func @SplitOverInputChannelWithDequantizeOpAsWeightsAndTilingStrategy
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x4608x128x4xf16, {order = #NHWC}>
func.func @SplitOverInputChannelWithDequantizeOpAsWeightsAndTilingStrategy(%arg0: tensor<1x4608x128x4xf16, {order = #NHWC}>) -> tensor<1x768x128x4xf16, {order = #NHWC}>  {
  %cst = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
  %cst_0 = const.Declare tensor<768x4608x1x1x!quant.uniform<u8:f16, 0.0084632095168618599:87>, {order = #NHWC}> = dense<1> : tensor<4608x768xui8>, [#const.Reshape<[1, 1, 4608, 768]>, #const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 0.0084632095168618599:87>>, #const.AffineReshape<[[0], [0], [1, 2], [3]], [1, 4608, 1, 768]>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>>, #const.AffineReshape<[[0], [0], [0], [1, 2, 3]], [768, 4608, 1, 1]>, #const.MemPermute<#NHWC, #NHWC>]

  %0 = VPU.Dequantize(%cst_0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, tilingStrategy = [3, 1, 1, 1]} : tensor<768x4608x1x1x!quant.uniform<u8:f16, 0.0084632095168618599:87>, {order = #NHWC}> -> tensor<768x4608x1x1xf16, {order = #NHWC}>
  %1 = VPU.NCE.Convolution(%arg0, %0, %cst) {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    rawFilterShape = [768, 4608, 1, 1], strides = [1, 1], tilingStrategy = [1, 2, 19, 1]
  } : tensor<1x4608x128x4xf16, {order = #NHWC}>, tensor<768x4608x1x1xf16, {order = #NHWC}>, tensor<768x1x1x4xsi32> -> tensor<1x768x128x4xf16, {order = #NHWC}>

  return %1 : tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[CONST0:%.+]] = const.Declare tensor<768x1x1x4xsi32>
  // CHECK:       [[CONST1:%.+]] = const.Declare tensor<768x2304x1x1x!qElemType, {order = #NHWC}> = dense<1>
  // CHECK-SAME:      : tensor<4608x768xui8>, [#const.SubView<[2304, 0], [2304, 768]>, #const.Reshape<[1, 1, 2304, 768]>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>

  // CHECK:       [[CONST2:%.+]] = const.Declare tensor<768x1x1x4xsi32>
  // CHECK:       [[CONST3:%.+]] = const.Declare tensor<768x2304x1x1x!qElemType, {order = #NHWC}> = dense<1>
  // CHECK-SAME:      : tensor<4608x768xui8>, [#const.SubView<[0, 0], [2304, 768]>, #const.Reshape<[1, 1, 2304, 768]>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>

  // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 2304, 128, 4] : tensor<1x4608x128x4xf16, {order = #NHWC}> to tensor<1x2304x128x4xf16, {order = #NHWC}>
  // CHECK:       [[DEQUANTIZE0:%.+]] = VPU.Dequantize([[CONST3]]) {
  // CHECK-SAME:      dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
  // CHECK-SAME:    : tensor<768x2304x1x1x!qElemType, {order = #NHWC}> -> tensor<768x2304x1x1xf16, {order = #NHWC}>
  // CHECK:       [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[DEQUANTIZE0]], [[CONST2]]) {
  // CHECK-SAME:        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                         scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:    : tensor<1x2304x128x4xf16, {order = #NHWC}>, tensor<768x2304x1x1xf16, {order = #NHWC}>, tensor<768x1x1x4xsi32> -> tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT_0]] [0, 2304, 0, 0] [1, 2304, 128, 4] : tensor<1x4608x128x4xf16, {order = #NHWC}> to tensor<1x2304x128x4xf16, {order = #NHWC}>
  // CHECK:       [[DEQUANTIZE1:%.+]] = VPU.Dequantize([[CONST1]]) {
  // CHECK-SAME:      dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
  // CHECK-SAME:    : tensor<768x2304x1x1x!qElemType, {order = #NHWC}> -> tensor<768x2304x1x1xf16, {order = #NHWC}>
  // CHECK:       [[CONVOLUTION1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[DEQUANTIZE1]], [[CONST0]]) {
  // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                       scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:      : tensor<1x2304x128x4xf16, {order = #NHWC}>, tensor<768x2304x1x1xf16, {order = #NHWC}>, tensor<768x1x1x4xsi32>
  // CHECK-SAME:    -> tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[CONVOLUTION0]], [[CONVOLUTION1]]) {
  // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                       scale = 1.4012984643248171E-45 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
  // CHECK-SAME:      -> tensor<1x768x128x4xf16, {order = #NHWC}>
  // CHECK:       return [[ELTWISE0]] : tensor<1x768x128x4xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SplitOverInputChannelWithF32Output
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x3072x128x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<768x3072x1x1xf16, {order = #NHWC}>
func.func @SplitOverInputChannelWithF32Output(%arg0: tensor<1x3072x128x4xf16, {order = #NHWC}>, %arg1: tensor<768x3072x1x1xf16, {order = #NHWC}>) -> tensor<1x768x128x4xf32, {order = #NHWC}> {
  %cst = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %arg1, %cst) {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -2.4028234663852886E+20 : f64, clamp_high = 4.4028234663852886E+20 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 2.000000e+00 : f64>,
    rawFilterShape = [768, 3072, 1, 1], strides = [1, 1], tilingStrategy = [1, 3, 16, 1]}
  : tensor<1x3072x128x4xf16, {order = #NHWC}>, tensor<768x3072x1x1xf16, {order = #NHWC}>, tensor<768x1x1x4xsi32> -> tensor<1x768x128x4xf32, {order = #NHWC}>

  return %0 : tensor<1x768x128x4xf32, {order = #NHWC}>

  // CHECK:       [[CONST0:%.+]] = const.Declare tensor<768x1x1x4xsi32>
  // CHECK:       [[CONST1:%.+]] = const.Declare tensor<768x1x1x4xsi32>
  // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 1024, 128, 4]
  // CHECK-SAME:      : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1024x128x4xf16, {order = #NHWC}>

  // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [768, 1024, 1, 1]
  // CHECK-SAME:      : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1024x1x1xf16, {order = #NHWC}>

  // CHECK:       [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[SLICE1]], [[CONST1]])
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 1024, 0, 0] [1, 1024, 128, 4]
  // CHECK-SAME:      : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1024x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 1024, 0, 0] [768, 1024, 1, 1]
  // CHECK-SAME:      : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1024x1x1xf16, {order = #NHWC}>

  // CHECK:       [[CONVOLUTION1:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[SLICE3]], [[CONST0]])
  // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:      scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[SLICE4:%.+]] = VPU.Slice [[INPUT_0]] [0, 2048, 0, 0] [1, 1024, 128, 4]
  // CHECK-SAME:      : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1024x128x4xf16, {order = #NHWC}>
  // CHECK:       [[SLICE5:%.+]] = VPU.Slice [[INPUT_1]] [0, 2048, 0, 0] [768, 1024, 1, 1]
  // CHECK-SAME:      : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1024x1x1xf16, {order = #NHWC}>

  // CHECK:       [[CONVOLUTION2:%.+]] = VPU.NCE.Convolution([[SLICE4]], [[SLICE5]], [[CONST0]])
  // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:      scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>

  // CHECK:       [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[CONVOLUTION0]], [[CONVOLUTION1]])
  // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:      scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>
  // CHECK:       [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[ELTWISE0]], [[CONVOLUTION2]])
  // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -2.4028234663852887E+20 : f64, clamp_high = 4.4028234663852887E+20 : f64,
  // CHECK-SAME:      scale = 1.4012984643248171E-45 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 2.000000e+00 : f64>
  // CHECK-SAME:  -> tensor<1x768x128x4xf32, {order = #NHWC}>
  // CHECK:       return [[ELTWISE1]] : tensor<1x768x128x4xf32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SplitOverInputChannelWithSparsifiedWeights_excluded
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x4304x256x4xf16, {order = #NHWC}>
func.func @SplitOverInputChannelWithSparsifiedWeights_excluded(%arg0: tensor<1x4304x256x4xf16, {order = #NHWC}>) -> tensor<1x1152x256x4xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<1152x1x1x4xsi32> = dense<1> : tensor<1152x1x1x4xsi32>
  %cst_0 = const.Declare tensor<1152x1x1x4352xi1> = dense<1.0> : tensor<1152x4304x1x1xf16, {order = #NHWC}>, [#const.GetSparsityMap]
  %cst_1 = const.Declare tensor<1152x4304x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1152x4304x1x1xf16, {order = #NHWC}>, [#const.Sparsify<false>]
  %0 = VPU.GroupSparseTensor(%cst_1, %cst_0) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<1152xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<1152x4304x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<1152x1x1x4352xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<1152xi64>, alignment = 16 : i64>>
  %1 = VPU.NCE.Convolution(%arg0, %0, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [1152, 4304, 1, 1], strides = [1, 1], tilingStrategy = [1, 24, 3, 1]} : tensor<1x4304x256x4xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<1152x4304x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<1152x1x1x4352xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<1152xi64>, alignment = 16 : i64>>, tensor<1152x1x1x4xsi32> -> tensor<1x1152x256x4xf16, {order = #NHWC}>
  return %1 : tensor<1x1152x256x4xf16, {order = #NHWC}>

  // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1152x1x1x4xsi32> = dense<1> : tensor<1152x1x1x4xsi32>
  // CHECK:       [[CONST1:%.+]] = const.Declare tensor<1152x1x1x4352xi1> = dense<1.000000e+00> : tensor<1152x4304x1x1xf16, {order = #NHWC}>, [#const.GetSparsityMap]
  // CHECK:       [[CONST2:%.+]] = const.Declare tensor<1152x4304x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1152x4304x1x1xf16, {order = #NHWC}>, [#const.Sparsify<false>]
  // CHECK:       [[GROUPSPARSETENSOR0:%.+]] = VPU.GroupSparseTensor([[CONST2]], [[CONST1]])
  // CHECK:       [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[INPUT_0]], [[GROUPSPARSETENSOR0]], [[CONST0]])
  // CHECK:       return [[CONVOLUTION0]] : tensor<1x1152x256x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @SplitOverInputChannelWithDequantizedAsWeights_excluded
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x3072x128x4xf16, {order = #NHWC}>
func.func @SplitOverInputChannelWithDequantizedAsWeights_excluded(%arg0: tensor<1x3072x128x4xf16, {order = #NHWC}>) -> tensor<1x768x128x4xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
  %cst_0 = const.Declare tensor<768x3072x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<3072x768xui8>, [#const.Reshape<[1, 1, 3072, 768]>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.AffineReshape<[[0], [0], [1, 2], [3]], [1, 3072, 1, 768]>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>>, #const.AffineReshape<[[0], [0], [0], [1, 2, 3]], [768, 3072, 1, 1]>, #const.MemPermute<#NHWC, #NHWC>]
  %1 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [768, 3072, 1, 1], strides = [1, 1], tilingStrategy = [1, 2, 6, 1]} : tensor<1x3072x128x4xf16, {order = #NHWC}>, tensor<768x3072x1x1x!qElemType, {order = #NHWC}>, tensor<768x1x1x4xsi32> -> tensor<1x768x128x4xf16, {order = #NHWC}>

  return %1 : tensor<1x768x128x4xf16, {order = #NHWC}>

        // CHECK:       [[CONST0:%.+]] = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
        // CHECK:       [[CONST1:%.+]] = const.Declare tensor<768x3072x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<3072x768xui8>, [#const.Reshape<[1, 1, 3072, 768]>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.AffineReshape<
        // CHECK-SAME{LITERAL}:  [[0], [0], [1, 2], [3]], [1, 3072, 1, 768]>, #const.Transpose<#NWHC>, #const.AffineReshape<[[0], [0], [0], [1, 2, 3]], [768, 3072, 1, 1]>, #const.MemPermute<#NHWC, #NHWC>]
        // CHECK:       [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[INPUT_0]], [[CONST1]], [[CONST0]])
        // CHECK:       return [[CONVOLUTION0]] : tensor<1x768x128x4xf16, {order = #NHWC}>
}
