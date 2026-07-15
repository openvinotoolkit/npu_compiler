//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --decompose-softmax-in-sdpa %s | FileCheck %s
// REQUIRES: platform-NPU5010

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
    %209 = VPU.NCE.Convolution(%arg0, %arg1)  rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %207 = VPU.SoftMax(%209) {SkipNormalization, axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %208 = VPU.NCE.Convolution(%207, %arg2) rawFilterShape [48, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>


    //  CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]])
    //  CHECK: [[REDUCESUM:%.+]] = VPU.NCE.Reduce([[SOFTMAX]])
    //  CHECK-SAME: mode = <RCP_SFM>
    //  CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], [[ARG2]])
    //  CHECK: [[EXPAND:%.+]] = VPU.Expand([[REDUCESUM]])
    //  CHECK: [[CST_1:%.+]] = const.Declare
    //  CHECK: [[CST_2:%.+]] = const.Declare
    //  CHECK: [[CONV_2:%.+]] = VPU.NCE.Convolution([[EXPAND]], [[CST_1]], [[CST_2]])
    //  CHECK: [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[CONV_2]])
    //  CHECK: return [[ELTWISE]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

    return %208 : tensor<1x48x1024x4xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @DecomposedOpsInSdpaSoftmaxWithFp32OduPermute
module @DecomposedOpsInSdpaSoftmaxWithFp32OduPermute {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:   [[ARG2:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf32> {
    %209 = VPU.NCE.Convolution(%arg0, %arg1)  rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %207 = VPU.SoftMax(%209) {SkipNormalization, axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %208 = VPU.NCE.Convolution(%207, %arg2) rawFilterShape [48, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf32>


    //  CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]])
    //  CHECK: [[REDUCESUM:%.+]] = VPU.NCE.Reduce([[SOFTMAX]])
    //  CHECK-SAME: mode = <RCP_SFM>
    //  CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], [[ARG2]])
    //  CHECK-SAME: -> tensor<1x48x1024x4xf16, {order = #NHWC}>
    //  CHECK: [[EXPAND:%.+]] = VPU.Expand([[REDUCESUM]])
    //  CHECK: [[CST_1:%.+]] = const.Declare
    //  CHECK: [[CST_2:%.+]] = const.Declare
    //  CHECK: [[CONV_2:%.+]] = VPU.NCE.Convolution([[EXPAND]], [[CST_1]], [[CST_2]])
    //  CHECK: [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[CONV_2]])
    //  CHECK-SAME: -> tensor<1x48x1024x4xf32>
    //  CHECK: return [[ELTWISE]] : tensor<1x48x1024x4xf32>

    return %208 : tensor<1x48x1024x4xf32>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>
!qElemType2 = !quant.uniform<i8:f16, 0.2>

// CHECK-LABEL: @DecomposedOpsInSdpaSoftmaxWithOutputQuantization
module @DecomposedOpsInSdpaSoftmaxWithOutputQuantization {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:   [[ARG2:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4x!qElemType2> {
    %209 = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %207 = VPU.SoftMax(%209) {SkipNormalization, axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %208 = VPU.NCE.Convolution(%207, %arg2) rawFilterShape [48, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.280000e+02 : f64, clamp_high = 1.270000e+02 : f64, scale = 5.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4x!qElemType2>


    //  CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]])
    //  CHECK: [[REDUCESUM:%.+]] = VPU.NCE.Reduce([[SOFTMAX]])
    //  CHECK-SAME: mode = <RCP_SFM>
    //  CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], [[ARG2]])
    //  CHECK-SAME: clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.562500e-02 : f64
    //  CHECK-SAME: -> tensor<1x48x1024x4xf16, {order = #NHWC}>
    //  CHECK: [[EXPAND:%.+]] = VPU.Expand([[REDUCESUM]])
    //  CHECK: [[CST_1:%.+]] = const.Declare
    //  CHECK: [[CST_2:%.+]] = const.Declare
    //  CHECK: [[CONV_2:%.+]] = VPU.NCE.Convolution([[EXPAND]], [[CST_1]], [[CST_2]])
    //  CHECK: [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[CONV_2]])
    //  CHECK-SAME: clamp_low = -1.280000e+02 : f64, clamp_high = 1.270000e+02 : f64, scale = 3.200000e+02 : f64
    //  CHECK-SAME: -> tensor<1x48x1024x4x!qElemType2>
    //  CHECK: return [[ELTWISE]] : tensor<1x48x1024x4x!qElemType2>

    return %208 : tensor<1x48x1024x4x!qElemType2>
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
    %209 = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %207 = VPU.SoftMax(%209) {SkipNormalization, axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %208 = VPU.NCE.Convolution(%207, %arg2) rawFilterShape [48, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 1.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>

    //  CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]])
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]])
    //  CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], [[ARG2]])

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
    %209 = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %207 = VPU.SoftMax(%209) {SkipNormalization, axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %208 = VPU.NCE.Convolution(%207, %arg2) rawFilterShape [48, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <SWISH>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>

    //  CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]])
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]])
    //  CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], [[ARG2]])

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
  // CHECK:   [[ARG1:%.+]]: tensor<1024x48x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:   [[ARG2:%.+]]: tensor<48x1024x1x1xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<1024x48x1x1x!qElemType1, {order = #NHWC}>, %arg2: tensor<48x1024x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %209 = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [1024, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<1024x48x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x1024x1024x4xf16, {order = #NHWC}>

    %207 = VPU.SoftMax(%209) {SkipNormalization, axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x1024x4xf16, {order = #NHWC}> -> tensor<1x1024x1024x4xf16, {order = #NHWC}>

    %208 = VPU.NCE.Convolution(%207, %arg2) rawFilterShape [48, 1024, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} : tensor<1x1024x1024x4xf16, {order = #NHWC}>, tensor<48x1024x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>

    //  CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]])
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]])
    //  CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], [[ARG2]])

    //  CHECK: return [[CONV_1]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

    return %208 : tensor<1x48x1024x4xf16, {order = #NHWC}>
  }
}
