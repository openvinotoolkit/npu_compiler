//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --decompose-softmax-in-sdpa %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @DecomposedOpsInSdpaSoftmax
module @DecomposedOpsInSdpaSoftmax {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:   [[ARG2:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %scale1 = const.Declare tensor<4096x1x1x1xf32> = dense<1.200000e-04> : tensor<4096x1x1x1xf32>
    %bias1 = const.Declare tensor<4096x1x1x1xf32> = dense<0.000000e+00> : tensor<4096x1x1x1xf32>
    %scale2 = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>
    %bias2 = const.Declare tensor<48x1x1x1xf32> = dense<0.000000e+00> : tensor<48x1x1x1xf32>

    %209 = VPU.NCE.Convolution(%arg0, %arg1, %scale1, %bias1) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, tensor<4096x1x1x1xf32>, tensor<4096x1x1x1xf32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %207 = VPU.SoftMax(%209) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %208 = VPU.NCE.Convolution(%207, %arg2, %scale2, %bias2) rawFilterShape [48, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<48x1x1x1xf32>, tensor<48x1x1x1xf32> -> tensor<1x48x1024x4xf16, {order = #NHWC}>


    //  CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]])
    //  CHECK: [[REDUCESUM:%.+]] = VPU.NCE.Reduce([[SOFTMAX]])
    //  CHECK-SAME: mode = <RCP_SFM>
    //  CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], [[ARG2]],
    //  CHECK: [[EXPAND:%.+]] = VPU.Expand([[REDUCESUM]])
    //  CHECK: [[CST_1:%.+]] = const.Declare
    //  CHECK: [[CST_2:%.+]] = const.Declare
    //  CHECK: [[CST_3:%.+]] = const.Declare
    //  CHECK: [[CONV_2:%.+]] = VPU.NCE.Convolution([[EXPAND]], [[CST_1]], [[CST_2]], [[CST_3]])
    //  CHECK: [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[CONV_2]])
    //  CHECK: return [[ELTWISE]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

    return %208 : tensor<1x48x1024x4xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @DecomposedOpsInSdpaSoftmaxNonNeutralBias
module @DecomposedOpsInSdpaSoftmaxNonNeutralBias {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:   [[ARG2:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %scale1 = const.Declare tensor<4096x1x1x1xf32> = dense<1.200000e-04> : tensor<4096x1x1x1xf32>
    %bias1 = const.Declare tensor<4096x1x1x1xf32> = dense<0.000000e+00> : tensor<4096x1x1x1xf32>
    %scale2 = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>
    %bias2 = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>

    %209 = VPU.NCE.Convolution(%arg0, %arg1, %scale1, %bias1) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, tensor<4096x1x1x1xf32>, tensor<4096x1x1x1xf32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %207 = VPU.SoftMax(%209) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %208 = VPU.NCE.Convolution(%207, %arg2, %scale2, %bias2) rawFilterShape [48, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 1.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<48x1x1x1xf32>, tensor<48x1x1x1xf32> -> tensor<1x48x1024x4xf16, {order = #NHWC}>

    //  CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]],
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]])
    //  CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], [[ARG2]],

    //  CHECK: return [[CONV_1]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

    return %208 : tensor<1x48x1024x4xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @DecomposedOpsInSdpaSoftmaxNonNoopMode
module @DecomposedOpsInSdpaSoftmaxNonNoopMode {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:   [[ARG2:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %scale1 = const.Declare tensor<4096x1x1x1xf32> = dense<1.200000e-04> : tensor<4096x1x1x1xf32>
    %bias1 = const.Declare tensor<4096x1x1x1xf32> = dense<0.000000e+00> : tensor<4096x1x1x1xf32>
    %scale2 = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>
    %bias2 = const.Declare tensor<48x1x1x1xf32> = dense<0.000000e+00> : tensor<48x1x1x1xf32>

    %209 = VPU.NCE.Convolution(%arg0, %arg1, %scale1, %bias1) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, tensor<4096x1x1x1xf32>, tensor<4096x1x1x1xf32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %207 = VPU.SoftMax(%209) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %208 = VPU.NCE.Convolution(%207, %arg2, %scale2, %bias2) rawFilterShape [48, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <SWISH>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<48x1x1x1xf32>, tensor<48x1x1x1xf32> -> tensor<1x48x1024x4xf16, {order = #NHWC}>

    //  CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]],
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]])
    //  CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], [[ARG2]],

    //  CHECK: return [[CONV_1]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

    return %208 : tensor<1x48x1024x4xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @DecomposedOpsInSdpaSoftmaxLowRatio
module @DecomposedOpsInSdpaSoftmaxLowRatio {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<80x48x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:   [[ARG2:%.+]]: tensor<48x80x1x1xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<80x48x1x1x!qElemType1, {order = #NHWC}>, %arg2: tensor<48x80x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %scale1 = const.Declare tensor<80x1x1x1xf32> = dense<1.200000e-04> : tensor<80x1x1x1xf32>
    %bias1 = const.Declare tensor<80x1x1x1xf32> = dense<0.000000e+00> : tensor<80x1x1x1xf32>
    %scale2 = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>
    %bias2 = const.Declare tensor<48x1x1x1xf32> = dense<0.000000e+00> : tensor<48x1x1x1xf32>

    %209 = VPU.NCE.Convolution(%arg0, %arg1, %scale1, %bias1) rawFilterShape [80, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<80x48x1x1x!qElemType1, {order = #NHWC}>, tensor<80x1x1x1xf32>, tensor<80x1x1x1xf32> -> tensor<1x80x1024x4xf16, {order = #NHWC}>

    %207 = VPU.SoftMax(%209) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x80x1024x4xf16, {order = #NHWC}> -> tensor<1x80x1024x4xf16, {order = #NHWC}>

    %208 = VPU.NCE.Convolution(%207, %arg2, %scale2, %bias2) rawFilterShape [48, 80, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x80x1024x4xf16, {order = #NHWC}>, tensor<48x80x1x1xf16, {order = #NHWC}>, tensor<48x1x1x1xf32>, tensor<48x1x1x1xf32> -> tensor<1x48x1024x4xf16, {order = #NHWC}>

    //  CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]],
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]])
    //  CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], [[ARG2]],

    //  CHECK: return [[CONV_1]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

    return %208 : tensor<1x48x1024x4xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DecomposedOpsInSdpaSoftmaxTooSmall
module @DecomposedOpsInSdpaSoftmaxTooSmall {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<16x512x1x1xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x512x4x1xf16, {order = #NHWC}>, %arg1: tensor<16x512x1x1xf16, {order = #NHWC}>) -> tensor<1x16x4x1xf16, {order = #NHWC}> {
    %scale = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    %bias = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
    %207 = VPU.SoftMax(%arg0) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x512x4x1xf16, {order = #NHWC}> -> tensor<1x512x4x1xf16, {order = #NHWC}>

    %208 = VPU.NCE.Convolution(%207, %arg1, %scale, %bias) rawFilterShape [16, 512, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x512x4x1xf16, {order = #NHWC}>, tensor<16x512x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x4x1xf16, {order = #NHWC}>

    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
    //  CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], [[ARG1]],

    //  CHECK: return [[CONV]] : tensor<1x16x4x1xf16, {order = #NHWC}>

    return %208 : tensor<1x16x4x1xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @DecomposedOpsInSdpaSoftmaxICExceedLimit
module @DecomposedOpsInSdpaSoftmaxICExceedLimit {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x16384x16x4xf16, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<48x16384x1x1xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x16384x16x4xf16, {order = #NHWC}>, %arg1: tensor<48x16384x1x1xf16, {order = #NHWC}>) -> tensor<1x48x16x4xf16, {order = #NHWC}> {
    %scale = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>
    %bias = const.Declare tensor<48x1x1x1xf32> = dense<0.000000e+00> : tensor<48x1x1x1xf32>
    %207 = VPU.SoftMax(%arg0) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16384x16x4xf16, {order = #NHWC}> -> tensor<1x16384x16x4xf16, {order = #NHWC}>

    %208 = VPU.NCE.Convolution(%207, %arg1, %scale, %bias) rawFilterShape [48, 16384, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x16384x16x4xf16, {order = #NHWC}>, tensor<48x16384x1x1xf16, {order = #NHWC}>, tensor<48x1x1x1xf32>, tensor<48x1x1x1xf32> -> tensor<1x48x16x4xf16, {order = #NHWC}>

    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
    //  CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], [[ARG1]],

    //  CHECK: return [[CONV]] : tensor<1x48x16x4xf16, {order = #NHWC}>

    return %208 : tensor<1x48x16x4xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @DecomposedOpsInSdpaSoftmaxWithAttnMask
module @DecomposedOpsInSdpaSoftmaxWithAttnMask {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:   [[ARG2:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>
  // CHECK:   [[ARG3:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, %arg2: tensor<1x4096x1024x4xf16, {order = #NHWC}>, %arg3: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %scale1 = const.Declare tensor<4096x1x1x1xf32> = dense<1.200000e-04> : tensor<4096x1x1x1xf32>
    %bias1 = const.Declare tensor<4096x1x1x1xf32> = dense<0.000000e+00> : tensor<4096x1x1x1xf32>
    %scale2 = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>
    %bias2 = const.Declare tensor<48x1x1x1xf32> = dense<0.000000e+00> : tensor<48x1x1x1xf32>

    %210 = VPU.NCE.Convolution(%arg0, %arg1, %scale1, %bias1) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, tensor<4096x1x1x1xf32>, tensor<4096x1x1x1xf32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
    %209 = VPU.NCE.Eltwise(%210, %arg2) {mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
    %207 = VPU.SoftMax(%209) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
    %208 = VPU.NCE.Convolution(%207, %arg3, %scale2, %bias2) rawFilterShape [48, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<48x1x1x1xf32>, tensor<48x1x1x1xf32> -> tensor<1x48x1024x4xf16, {order = #NHWC}>

    return %208 : tensor<1x48x1024x4xf16, {order = #NHWC}>

    //  CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution
    //  CHECK: [[ADD:%.+]] = VPU.NCE.Eltwise
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[ADD]])
    //  CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]],

    //  CHECK: return [[CONV_1]]
  }
}
