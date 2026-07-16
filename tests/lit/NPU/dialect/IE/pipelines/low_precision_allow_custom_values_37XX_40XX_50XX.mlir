//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --low-precision %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

!qElemType = !quant.uniform<u8<0:254>:f32:0, {0.0078740157480314959:127,0.0086614175105658095:127,0.0094488192731001247:127,0.010236220096978615:127}>
!qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @QuantizedConv
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x3x62x62xui8>) -> tensor<1x4x60x60xf32>
func.func @QuantizedConv(%input: tensor<1x3x62x62xui8>) -> tensor<1x4x60x60xf32> {
    %0 = IE.Convert(%input) {dstElemType = f32} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf32>

    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>

    %input_fq = IE.FakeQuantize(%0, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x62x62xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x62x62xf32>

    %weights = const.Declare tensor<4x3x3x3xf32> = dense<128> : tensor<4x3x3x3xui8>, [#const.CastElemType<f32>]

    %weights_in_low = const.Declare tensor<1xf32> = dense<0.0> : tensor<1xf32>
    %weights_in_high = const.Declare tensor<1xf32> = dense<255.0> : tensor<1xf32>

    %weights_out_low = const.Declare tensor<4x1x1x1xf32> = dense<[[[[-1.0]]], [[[-1.1]]], [[[-1.2]]], [[[-1.3]]]]> : tensor<4x1x1x1xf32>
    %weights_out_high = const.Declare tensor<4x1x1x1xf32> = dense<[[[[1.0]]], [[[1.1]]], [[[1.2]]], [[[1.3]]]]> : tensor<4x1x1x1xf32>

    %weights_fq = IE.FakeQuantize(%weights, %weights_in_low, %weights_in_high, %weights_out_low, %weights_out_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<4x3x3x3xf32>, tensor<1xf32>, tensor<1xf32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32> -> tensor<4x3x3x3xf32>

    %conv = IE.Convolution(%input_fq, %weights_fq)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>

    %last_fq = IE.FakeQuantize(%conv, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    return %last_fq : tensor<1x4x60x60xf32>

    // CHECK-DAG:     [[WEIGHTS:%.+]] = const.Declare
    // CHECK-SAME:    dense<128> : tensor<4x3x3x3xui8>,
    // CHECK-SAME:    [#const.CastElemType<f32>, #const.CastElemType<!qElemType>]

    // CHECK:     [[INPUT_QUANT:%.+]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !qElemType1} :
    // CHECK-SAME:     tensor<1x3x62x62xui8> -> tensor<1x3x62x62x!qElemType1>

    // CHECK:     [[CONV:%.+]] = IE.Convolution([[INPUT_QUANT]], [[WEIGHTS]])

    // CHECK:     [[RELU:%.+]] = IE.ReLU([[CONV]])

    // CHECK:     return [[RELU]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 2.000000e+00:128>
// CHECK: !qElemType = !quant.uniform<i8:f16, 2.000000e+00>
func.func @MixedPrecisionI8Convolution(%arg0: tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-2.560000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.540000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<2x2x1x1xf16> = dense<[[[[0.000000e+00]], [[2.550000e+02]]], [[[1.310000e+02]], [[1.290000e+02]]]]> : tensor<2x2x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<2x2x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<2x2x1x1xf16>
    %1 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x2x1x1xf32> -> tensor<1x2x1x1xf16>
    %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2x1x1xf16>, tensor<2x2x1x1xf16> -> tensor<1x2x1x1xf16>
    %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x2x1x1xf16> -> tensor<1x2x1x1xf32>
    return %3 : tensor<1x2x1x1xf32>
    // CHECK:     [[CST:%.+]] = const.Declare tensor<2x2x1x1x!qElemType>
    // CHECK-SAME:  : tensor<2x2x1x1xf32>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>]
    // CHECK:     [[VAL0:%.+]] = IE.Convert([[ARG0:%.+]])  {dstElemType = f16} : tensor<1x2x1x1xf32> -> tensor<1x2x1x1xf16>
    // CHECK:     [[CONV:%.+]] = IE.Convolution([[VAL0]], [[CST]])
    // CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2x1x1xf16>, tensor<2x2x1x1x!qElemType> -> tensor<1x2x1x1xf16>
    // CHECK:     [[CAST:%.+]] = IE.Convert([[CONV]])  {dstElemType = f32} : tensor<1x2x1x1xf16> -> tensor<1x2x1x1xf32>
    // CHECK:     return [[CAST]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0058323649799122534:59>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>
// CHECK-LABEL: @ConvertQuantizeCastAgnosticOp
func.func @ConvertQuantizeCastAgnosticOp(%arg0: tensor<1x96x800x1279xf16>, %arg1: tensor<16x96x1x1xf16>) -> tensor<1x4x1600x2558xui8> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-0.34410953521> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<1.1431435> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>, [#const.CastElemType<f16>]
    %3 = IE.Convolution(%arg0, %arg1,  %cst_3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x800x1279xf16>, tensor<16x96x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x800x1279xf16>
    %0 = IE.FakeQuantize(%3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x800x1279xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x800x1279xf16>
    %1 = IE.Convert(%0) {dstElemType = ui8} : tensor<1x16x800x1279xf16> -> tensor<1x16x800x1279xui8>
    %2 = IE.DepthToSpace(%1) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x800x1279xui8> -> tensor<1x4x1600x2558xui8>
    return %2 : tensor<1x4x1600x2558xui8>

    // CHECK:     [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<f16>]
    // CHECK:     [[CONV:%.+]] = IE.Convolution([[ARG0:%.+]], [[ARG1:%.+]], [[CST]])
    // CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x800x1279xf16>, tensor<16x96x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x800x1279x!qElemType>
    // CHECK:     [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[CONV]]) {dstElemType = !qElemType1} :
    // CHECK-SAME: tensor<1x16x800x1279x!qElemType> -> tensor<1x16x800x1279x!qElemType1>
    // CHECK:     [[DEPTH_TO_SPACE:%.+]] = IE.DepthToSpace([[QUANTIZE_CAST]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x800x1279x!qElemType1> -> tensor<1x4x1600x2558x!qElemType1>
    // CHECK:     [[QUANTIZE_CAST_1:%.+]] = IE.QuantizeCast([[DEPTH_TO_SPACE]]) {dstElemType = ui8} :
    // CHECK-SAME: tensor<1x4x1600x2558x!qElemType1> -> tensor<1x4x1600x2558xui8>
    // CHECK:     return [[QUANTIZE_CAST_1]] : tensor<1x4x1600x2558xui8>
}

// -----

func.func @FuseQuantizeAndActivationOps(%arg0: tensor<1x2x1x1xf16>) -> tensor<1x2x1x1xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<2x2x1x1xf16> = dense<[[[[0.000000e+00]], [[2.550000e+02]]], [[[1.310000e+02]], [[1.290000e+02]]]]> : tensor<2x2x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<1.02377498> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%cst_1, %cst_2, %cst_3, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<2x2x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<2x2x1x1xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_2, %cst_4, %cst_2, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x1xf16>
    %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2x1x1xf16>, tensor<2x2x1x1xf16> -> tensor<1x2x1x1xf16>
    %3 = IE.FakeQuantize(%2, %cst_2, %cst_3, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x1xf16>
    %4 = IE.LeakyRelu(%3) {negative_slope = 0.20000000298023224 : f64} : tensor<1x2x1x1xf16> -> tensor<1x2x1x1xf16>
    %5 = IE.FakeQuantize(%4, %cst_2, %cst_4, %cst_2, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x1xf16>
    return %5 : tensor<1x2x1x1xf16>

    // CHECK:     [[CST:%.+]] = const.Declare tensor<2x2x1x1x!qElemType>
    // CHECK:     [[ADD0:%.+]] = IE.Add([[ARG0:%.+]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x1x1x!qElemType1>
    // CHECK:     [[QCAST1:%.+]] = IE.QuantizeCast([[ADD0]]) {dstElemType = !qElemType2} : tensor<1x2x1x1x!qElemType1> -> tensor<1x2x1x1x!qElemType2>
    // CHECK:     [[CONV:%.+]] = IE.Convolution([[QCAST1]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.20000000298023224 : f64>, strides = [1, 1]} : tensor<1x2x1x1x!qElemType2>, tensor<2x2x1x1x!qElemType> -> tensor<1x2x1x1x!qElemType2>
    // CHECK:     [[QCAST2:%.+]] = IE.QuantizeCast([[CONV]]) {dstElemType = !qElemType3} : tensor<1x2x1x1x!qElemType2> -> tensor<1x2x1x1x!qElemType3>
    // CHECK:     [[ADD1:%.+]] = IE.Add([[QCAST2]], [[QCAST2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2x1x1x!qElemType3>, tensor<1x2x1x1x!qElemType3> -> tensor<1x2x1x1xf16>
    // CHECK:     return [[ADD1]] : tensor<1x2x1x1xf16>
}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f16, 0.011764705882352941:-43>

module {
    config.PipelineOptions @Options {
        config.Option @config.AsymmetricPerTensorZP : true
    }

    // CHECK: func.func @AsymmetricWeightsI8([[INPUT:%.+]]: tensor<1x3x16x16xf16>)
    func.func @AsymmetricWeightsI8(%input: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
        %weights = const.Declare tensor<3x3x1x1xf16> = dense<1> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>]
        %low = const.Declare tensor<1x1x1x1xf16> = dense<[[[[-1.0]]]]> : tensor<1x1x1x1xf16>
        %high = const.Declare tensor<1x1x1x1xf16> = dense<[[[[2.0]]]]> : tensor<1x1x1x1xf16>
        %fq_weights = IE.FakeQuantize(%weights, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<3x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<3x3x1x1xf16>
        %conv = IE.Convolution(%input, %fq_weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>
        return %conv : tensor<1x3x16x16xf16>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<3x3x1x1x!qElemType> = dense<1>
        // CHECK-SAME:    : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>, #const.Quantize<!qElemType>, #const.CastElemType<!qElemType>]
        // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]])
        // CHECK-SAME:    : tensor<1x3x16x16xf16>, tensor<3x3x1x1x!qElemType> -> tensor<1x3x16x16xf16>
        // CHECK:       return [[CONV]]
    }
}

// -----

// CHECK: !qElemType = !quant.uniform<i4:f16, 2.000000e-01:-3>

module {
    config.PipelineOptions @Options {
        config.Option @config.AsymmetricPerTensorZP : true
    }

    // CHECK: func.func @AsymmetricWeightsI4([[INPUT:%.+]]: tensor<1x3x16x16xf16>)
    func.func @AsymmetricWeightsI4(%input: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
        %weights = const.Declare tensor<3x3x1x1xf16> = dense<1> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>]
        %low = const.Declare tensor<1x1x1x1xf16> = dense<[[[[-1.0]]]]> : tensor<1x1x1x1xf16>
        %high = const.Declare tensor<1x1x1x1xf16> = dense<[[[[2.0]]]]> : tensor<1x1x1x1xf16>
        %fq_weights = IE.FakeQuantize(%weights, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<3x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<3x3x1x1xf16>
        %conv = IE.Convolution(%input, %fq_weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>
        return %conv : tensor<1x3x16x16xf16>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<3x3x1x1x!qElemType> = dense<1>
        // CHECK-SAME:    : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>, #const.Quantize<!qElemType>, #const.CastElemType<!qElemType>]
        // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]])
        // CHECK-SAME:    : tensor<1x3x16x16xf16>, tensor<3x3x1x1x!qElemType> -> tensor<1x3x16x16xf16>
        // CHECK:       return [[CONV]]
    }
}

// -----

// CHECK: !qElemType = !quant.uniform<i2:f16, 1.000000e+00:-1>

module {
    config.PipelineOptions @Options {
        config.Option @config.AsymmetricPerTensorZP : true
    }

    // CHECK: func.func @AsymmetricWeightsI2([[INPUT:%.+]]: tensor<1x3x16x16xf16>)
    func.func @AsymmetricWeightsI2(%input: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
        %weights = const.Declare tensor<3x3x1x1xf16> = dense<1> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>]
        %low = const.Declare tensor<1x1x1x1xf16> = dense<[[[[-1.0]]]]> : tensor<1x1x1x1xf16>
        %high = const.Declare tensor<1x1x1x1xf16> = dense<[[[[2.0]]]]> : tensor<1x1x1x1xf16>
        %fq_weights = IE.FakeQuantize(%weights, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64} : tensor<3x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<3x3x1x1xf16>
        %conv = IE.Convolution(%input, %fq_weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>
        return %conv : tensor<1x3x16x16xf16>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<3x3x1x1x!qElemType> = dense<1>
        // CHECK-SAME:    : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>, #const.Quantize<!qElemType>, #const.CastElemType<!qElemType>]
        // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]])
        // CHECK-SAME:    : tensor<1x3x16x16xf16>, tensor<3x3x1x1x!qElemType> -> tensor<1x3x16x16xf16>
        // CHECK:       return [[CONV]]
    }
}
