//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --concat-repeating-blocks-outlining="min-seq-length=2 sdpa-head-threshold=2" %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --concat-repeating-blocks-outlining="min-seq-length=5 sdpa-min-seq-length=4 sdpa-head-threshold=2" %s | FileCheck %s --check-prefix=CHECK-SDPA
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 1.000000e+00:0>

// CHECK-LABEL: module @OutlineConcatWithZeroPointTable
module @OutlineConcatWithZeroPointTable {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x8x8xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x32x8x8xf16>
    }

    func.func @main(%input: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x32x8x8xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<16x16x1x1xi8>, [#const.Reorder<#NHWC>]
        %scale = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<16x1x1x1xf32>
        %bias = const.Declare tensor<16x1x1x1xf32> = dense<0.0> : tensor<16x1x1x1xf32>
        %zero_points = const.Declare tensor<16x1x1x1xsi8> = dense<0> : tensor<16x1x1x1xsi8>
        %zero_point_table = VPU.ZeroPointTable(%zero_points) {workloadSizes = [16], zeroPointTableData = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]} : tensor<16x1x1x1xsi8> -> tensor<16x1x1x1xi8>

        %conv1 = VPU.NCE.Convolution(%input, %weights, %scale, %bias, %zero_point_table) rawFilterShape [16, 16, 1, 1] {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xi8> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %softmax1 = VPU.SoftMax(%conv1) {axisInd = 1 : i64} : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %conv2 = VPU.NCE.Convolution(%input, %weights, %scale, %bias, %zero_point_table) rawFilterShape [16, 16, 1, 1] {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xi8> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %softmax2 = VPU.SoftMax(%conv2) {axisInd = 1 : i64} : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %concat = VPU.Concat(%softmax1, %softmax2) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]}
            : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x32x8x8xf16, {order = #NHWC}>
        %relu = VPU.ReLU(%concat) : tensor<1x32x8x8xf16, {order = #NHWC}> -> tensor<1x32x8x8xf16, {order = #NHWC}>

        return %relu : tensor<1x32x8x8xf16, {order = #NHWC}>

        // CHECK:  func.func nested @main_concat1([[ARG0:%.+]]: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x32x8x8xf16, {order = #NHWC}> {
        // CHECK:      [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}>
        // CHECK:      [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32>
        // CHECK:      [[BIAS:%.+]] = const.Declare tensor<16x1x1x1xf32>
        // CHECK:      [[ZERO_POINTS:%.+]] = const.Declare tensor<16x1x1x1xsi8>
        // CHECK:      [[ZERO_POINT_TABLE:%.+]] = VPU.ZeroPointTable([[ZERO_POINTS]])
        // CHECK:      [[CONV1:%.+]] = VPU.NCE.Convolution([[ARG0]], [[WEIGHTS]], [[SCALE]], [[BIAS]], [[ZERO_POINT_TABLE]])
        // CHECK:      [[SOFTMAX1:%.+]] = VPU.SoftMax([[CONV1]])
        // CHECK:      [[CONV2:%.+]] = VPU.NCE.Convolution([[ARG0]], [[WEIGHTS]], [[SCALE]], [[BIAS]], [[ZERO_POINT_TABLE]])
        // CHECK:      [[SOFTMAX2:%.+]] = VPU.SoftMax([[CONV2]])
        // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[SOFTMAX1]], [[SOFTMAX2]])
        // CHECK:      return [[CONCAT]]
        // CHECK:  }
        // CHECK:  func.func nested @main_concat2([[ARG0:%.+]]: tensor<1x32x8x8xf16, {order = #NHWC}>) -> tensor<1x32x8x8xf16, {order = #NHWC}> {
        // CHECK:      [[RELU:%.+]] = VPU.ReLU([[ARG0]])
        // CHECK:      return [[RELU]]
        // CHECK:  }
        // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x32x8x8xf16, {order = #NHWC}> {
        // CHECK:      [[CALL:%.+]] = call @main_concat1([[INPUT]]) : (tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x32x8x8xf16, {order = #NHWC}>
        // CHECK:      [[CALL2:%.+]] = call @main_concat2([[CALL]]) : (tensor<1x32x8x8xf16, {order = #NHWC}>) -> tensor<1x32x8x8xf16, {order = #NHWC}>
        // CHECK:      return [[CALL2]]
        // CHECK:  }
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-SDPA-LABEL: module @DoNotOutlineSDPABelowMinSeqLength
module @DoNotOutlineSDPABelowMinSeqLength {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x8x8xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x8x8xf16>
    }

    func.func @main(%input: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x48x8x8xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

        %qk0 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %sm0 = VPU.SoftMax(%qk0) {axisInd = 1 : i64} : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %v0 = VPU.NCE.Convolution(%sm0, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>

        %qk1 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %sm1 = VPU.SoftMax(%qk1) {axisInd = 1 : i64} : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %v1 = VPU.NCE.Convolution(%sm1, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>

        %qk2 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %sm2 = VPU.SoftMax(%qk2) {axisInd = 1 : i64} : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %v2 = VPU.NCE.Convolution(%sm2, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>

        %concat = VPU.Concat(%v0, %v1, %v2) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]}
            : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x48x8x8xf16, {order = #NHWC}>

        return %concat : tensor<1x48x8x8xf16, {order = #NHWC}>

        // CHECK-SDPA:      func.func @main
        // CHECK-SDPA-NOT:  call
        // CHECK-SDPA:      VPU.SoftMax
        // CHECK-SDPA:      VPU.Concat
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-SDPA-LABEL: module @OutlineSDPAWithReducedSeqLength
module @OutlineSDPAWithReducedSeqLength {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x8x8xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x8x8xf16>
    }

    func.func @main(%input: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x48x8x8xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

        %qk0 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %sm0 = VPU.SoftMax(%qk0) {axisInd = 1 : i64} : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %v0 = VPU.NCE.Convolution(%sm0, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %act0 = VPU.ReLU(%v0) : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>

        %qk1 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %sm1 = VPU.SoftMax(%qk1) {axisInd = 1 : i64} : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %v1 = VPU.NCE.Convolution(%sm1, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %act1 = VPU.ReLU(%v1) : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>

        %qk2 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %sm2 = VPU.SoftMax(%qk2) {axisInd = 1 : i64} : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %v2 = VPU.NCE.Convolution(%sm2, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %act2 = VPU.ReLU(%v2) : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>

        %concat = VPU.Concat(%act0, %act1, %act2) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]}
            : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x48x8x8xf16, {order = #NHWC}>
        %relu = VPU.ReLU(%concat) : tensor<1x48x8x8xf16, {order = #NHWC}> -> tensor<1x48x8x8xf16, {order = #NHWC}>

        return %relu : tensor<1x48x8x8xf16, {order = #NHWC}>

        // CHECK-SDPA:  func.func nested @main_concat1([[ARG0:%.+]]: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x48x8x8xf16, {order = #NHWC}> {
        // CHECK-SDPA:      [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK-SDPA:      [[QK0:%.+]] = VPU.NCE.Convolution([[ARG0]], [[WEIGHTS]])
        // CHECK-SDPA:      [[SM0:%.+]] = VPU.SoftMax([[QK0]])
        // CHECK-SDPA:      [[V0:%.+]] = VPU.NCE.Convolution([[SM0]], [[WEIGHTS]])
        // CHECK-SDPA:      [[ACT0:%.+]] = VPU.ReLU([[V0]])
        // CHECK-SDPA:      [[QK1:%.+]] = VPU.NCE.Convolution([[ARG0]], [[WEIGHTS]])
        // CHECK-SDPA:      [[SM1:%.+]] = VPU.SoftMax([[QK1]])
        // CHECK-SDPA:      [[V1:%.+]] = VPU.NCE.Convolution([[SM1]], [[WEIGHTS]])
        // CHECK-SDPA:      [[ACT1:%.+]] = VPU.ReLU([[V1]])
        // CHECK-SDPA:      [[QK2:%.+]] = VPU.NCE.Convolution([[ARG0]], [[WEIGHTS]])
        // CHECK-SDPA:      [[SM2:%.+]] = VPU.SoftMax([[QK2]])
        // CHECK-SDPA:      [[V2:%.+]] = VPU.NCE.Convolution([[SM2]], [[WEIGHTS]])
        // CHECK-SDPA:      [[ACT2:%.+]] = VPU.ReLU([[V2]])
        // CHECK-SDPA:      [[CONCAT:%.+]] = VPU.Concat([[ACT0]], [[ACT1]], [[ACT2]])
        // CHECK-SDPA:      return [[CONCAT]]
        // CHECK-SDPA:  }
        // CHECK-SDPA:  func.func nested @main_concat2([[ARG0:%.+]]: tensor<1x48x8x8xf16, {order = #NHWC}>) -> tensor<1x48x8x8xf16, {order = #NHWC}> {
        // CHECK-SDPA:      [[RELU:%.+]] = VPU.ReLU([[ARG0]])
        // CHECK-SDPA:      return [[RELU]]
        // CHECK-SDPA:  }
        // CHECK-SDPA:  func.func @main([[INPUT:%.+]]: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x48x8x8xf16, {order = #NHWC}> {
        // CHECK-SDPA:      [[CALL1:%.+]] = call @main_concat1([[INPUT]])
        // CHECK-SDPA:      [[CALL2:%.+]] = call @main_concat2([[CALL1]])
        // CHECK-SDPA:      return [[CALL2]]
        // CHECK-SDPA:  }
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-SDPA-LABEL: module @DoNotOutlineMaskedSDPA
module @DoNotOutlineMaskedSDPA {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x8x8xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x32x8x8xf16>
    }

    func.func @main(%input: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x32x8x8xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %mask = const.Declare tensor<1x16x8x8xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x16x8x8xf16>, [#const.Reorder<#NHWC>]

        %qk0 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %add0 = VPU.Add(%qk0, %mask) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %sm0 = VPU.SoftMax(%add0) {axisInd = 1 : i64} : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %v0 = VPU.NCE.Convolution(%sm0, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>

        %qk1 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %add1 = VPU.Add(%qk1, %mask) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %sm1 = VPU.SoftMax(%add1) {axisInd = 1 : i64} : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %v1 = VPU.NCE.Convolution(%sm1, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>

        %concat = VPU.Concat(%v0, %v1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]}
            : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x32x8x8xf16, {order = #NHWC}>

        return %concat : tensor<1x32x8x8xf16, {order = #NHWC}>

        // CHECK-SDPA:      func.func @main
        // CHECK-SDPA-NOT:  call
        // CHECK-SDPA:      VPU.Add
        // CHECK-SDPA:      VPU.SoftMax
        // CHECK-SDPA:      VPU.Concat
    }
}
