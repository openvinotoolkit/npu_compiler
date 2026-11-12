//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --extract-reduce-max-from-exp-stabilized %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @ExtractReduceMaxFromExpStabilized
module @ExtractReduceMaxFromExpStabilized {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
    %scale1 = const.Declare tensor<4096x1x1x1xf32> = dense<1.200000e-04> : tensor<4096x1x1x1xf32>
    %bias1 = const.Declare tensor<4096x1x1x1xf32> = dense<0.000000e+00> : tensor<4096x1x1x1xf32>

    %conv1 = VPU.NCE.Convolution(%arg0, %arg1, %scale1, %bias1) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, tensor<4096x1x1x1xf32>, tensor<4096x1x1x1xf32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %softmax = VPU.SoftMax(%conv1) {SkipNormalization, axisInd = 1 : i64} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    return %softmax : tensor<1x4096x1024x4xf16, {order = #NHWC}>

    //  CHECK: [[CONV_0:%.+]], [[RMAX:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]],
    //  CHECK-SAME: axes_value = [1]
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]], [[RMAX]])
    //  CHECK-SAME: SkipNormalization
    //  CHECK: return [[SOFTMAX]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @ExtractReduceMaxFromExpStabilizedWithPadSize
module @ExtractReduceMaxFromExpStabilizedWithPadSize {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
    %scale1 = const.Declare tensor<4096x1x1x1xf32> = dense<1.200000e-04> : tensor<4096x1x1x1xf32>
    %bias1 = const.Declare tensor<4096x1x1x1xf32> = dense<0.000000e+00> : tensor<4096x1x1x1xf32>

    %conv1 = VPU.NCE.Convolution(%arg0, %arg1, %scale1, %bias1) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, tensor<4096x1x1x1xf32>, tensor<4096x1x1x1xf32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %softmax = VPU.SoftMax(%conv1) {SkipNormalization, axisInd = 1 : i64, padSize = 8 : i64} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    return %softmax : tensor<1x4096x1024x4xf16, {order = #NHWC}>

    //  CHECK: [[NEW_SCALE:%.+]] = const.Declare tensor<4096x1x1x1xf32> = dense<"0x{{.*}}0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F"
    //  CHECK: [[NEW_BIAS:%.+]] = const.Declare tensor<4096x1x1x1xf32> = dense<"0x{{.*}}FFFF7FFFFFFF7FFFFFFF7FFFFFFF7FFFFFFF7FFFFFFF7FFFFFFF7FFFFFFF7FFF"
    //  CHECK: [[CONV_0:%.+]], [[RMAX:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]], [[NEW_SCALE]], [[NEW_BIAS]]
    //  CHECK-SAME: axes_value = [1]
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]], [[RMAX]])
    //  CHECK-SAME: SkipNormalization
    //  CHECK-SAME: padSize = 8 : i64
    //  CHECK: return [[SOFTMAX]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @DoNotExtractReduceMaxWithNonConstWeightsTable
module @DoNotExtractReduceMaxWithNonConstWeightsTable {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, %scale1: tensor<4096x1x1x1xf32>, %bias1: tensor<4096x1x1x1xf32>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {

    %conv1 = VPU.NCE.Convolution(%arg0, %arg1, %scale1, %bias1) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, tensor<4096x1x1x1xf32>, tensor<4096x1x1x1xf32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %softmax = VPU.SoftMax(%conv1) {SkipNormalization, axisInd = 1 : i64, padSize = 8 : i64} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    return %softmax : tensor<1x4096x1024x4xf16, {order = #NHWC}>

    //  CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]],
    //  CHECK-SAME: scale = 1.200000e-04 : f64
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]])
    //  CHECK-SAME: SkipNormalization
    //  CHECK-SAME: padSize = 8 : i64
    //  CHECK: return [[SOFTMAX]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @DoNotExtractReduceMaxFromNonExpStabilizedSoftMax
module @DoNotExtractReduceMaxFromNonExpStabilizedSoftMax {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
    %scale1 = const.Declare tensor<4096x1x1x1xf32> = dense<1.200000e-04> : tensor<4096x1x1x1xf32>
    %bias1 = const.Declare tensor<4096x1x1x1xf32> = dense<0.000000e+00> : tensor<4096x1x1x1xf32>

    %conv1 = VPU.NCE.Convolution(%arg0, %arg1, %scale1, %bias1) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, tensor<4096x1x1x1xf32>, tensor<4096x1x1x1xf32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %softmax = VPU.SoftMax(%conv1) {axisInd = 1 : i64} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    return %softmax : tensor<1x4096x1024x4xf16, {order = #NHWC}>

    //  CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]],
    //  CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV_0]])
    //  CHECK: return [[SOFTMAX]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotExtractReduceMaxWhenConvRequiresCTiling
module @DoNotExtractReduceMaxWhenConvRequiresCTiling {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x2304x1024x4xf16, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<4096x2304x1x1xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x2304x1024x4xf16, {order = #NHWC}>, %arg1: tensor<4096x2304x1x1xf16, {order = #NHWC}>, %arg2: tensor<4096x1x1x1xf32>, %arg3: tensor<4096x1x1x1xf32>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
    %conv = VPU.NCE.Convolution(%arg0, %arg1, %arg2, %arg3) rawFilterShape [4096, 2304, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+02 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x2304x1024x4xf16, {order = #NHWC}>, tensor<4096x2304x1x1xf16, {order = #NHWC}>, tensor<4096x1x1x1xf32>, tensor<4096x1x1x1xf32> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %softmax = VPU.SoftMax(%conv) {SkipNormalization, axisInd = 1 : i64} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    return %softmax : tensor<1x4096x1024x4xf16, {order = #NHWC}>

    // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]],
    // CHECK-NOT: axes_value = [1]
    // CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV]])
    // CHECK-SAME: SkipNormalization
    // CHECK: return [[SOFTMAX]]
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotExtractReduceMaxWhenConvOutputCExceedsDimensionLimit
module @DoNotExtractReduceMaxWhenConvOutputCExceedsDimensionLimit {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<8208x16x1x1xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>, %arg1: tensor<8208x16x1x1xf16, {order = #NHWC}>, %arg2: tensor<8208x1x1x1xf32>, %arg3: tensor<8208x1x1x1xf32>) -> tensor<1x8208x1x1xf16, {order = #NHWC}> {
    %conv = VPU.NCE.Convolution(%arg0, %arg1, %arg2, %arg3) rawFilterShape [8208, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<8208x16x1x1xf16, {order = #NHWC}>, tensor<8208x1x1x1xf32>, tensor<8208x1x1x1xf32> -> tensor<1x8208x1x1xf16, {order = #NHWC}>

    %softmax = VPU.SoftMax(%conv) {SkipNormalization, axisInd = 1 : i64} : tensor<1x8208x1x1xf16, {order = #NHWC}> -> tensor<1x8208x1x1xf16, {order = #NHWC}>

    return %softmax : tensor<1x8208x1x1xf16, {order = #NHWC}>

    // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]],
    // CHECK-NOT: axes_value = [1]
    // CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV]])
    // CHECK-SAME: SkipNormalization
    // CHECK: return [[SOFTMAX]]
  }
}
