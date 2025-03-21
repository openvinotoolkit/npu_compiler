//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --low-precision %s | FileCheck %s
// REQUIRES: arch-NPU37XX

!qElemType = !quant.uniform<u8<0:254>:f32:0, {0.0078740157480314959:127,0.0086614175105658095:127,0.0094488192731001247:127,0.010236220096978615:127}>
!qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @QuantizedConv
// CHECK-SAME:      ([[INPUT:%.*]]: tensor<1x3x62x62xui8>) -> tensor<1x4x60x60xf32>
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

    // CHECK-DAG:     [[WEIGHTS:%.*]] = const.Declare
    // CHECK-SAME:    dense<128> : tensor<4x3x3x3xui8>,
    // CHECK-SAME:    [#const.CastElemType<f32>, #const.CastElemType<!qElemType>]

    // CHECK:     [[INPUT_QUANT:%.*]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !qElemType1} :
    // CHECK-SAME:     tensor<1x3x62x62xui8> -> tensor<1x3x62x62x!qElemType1>

    // CHECK:     [[CONV:%.*]] = IE.Convolution([[INPUT_QUANT]], [[WEIGHTS]])

    // CHECK:     [[OUTPUT_QUANT:%.*]] = IE.QuantizeCast([[CONV]]) {dstElemType = !qElemType2} :
    // CHECK-SAME:     tensor<1x4x60x60x!qElemType1> -> tensor<1x4x60x60x!qElemType2>

    // CHECK:     [[OUT_DEQ:%.*]] = IE.Add([[OUTPUT_QUANT]], [[OUTPUT_QUANT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK-SAME:     -> tensor<1x4x60x60xf32>

    // CHECK:     return [[OUT_DEQ]]
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
    // CHECK:     [[CST:%.*]] = const.Declare tensor<2x2x1x1x!qElemType>
    // CHECK-SAME:  : tensor<2x2x1x1xf32>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>]
    // CHECK:     [[VAL0:%.*]] = IE.Convert([[ARG0:%.*]])  {dstElemType = f16} : tensor<1x2x1x1xf32> -> tensor<1x2x1x1xf16>
    // CHECK:     [[CONV:%.*]] = IE.Convolution([[VAL0]], [[CST]])
    // CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2x1x1xf16>, tensor<2x2x1x1x!qElemType> -> tensor<1x2x1x1xf16>
    // CHECK:     [[CAST:%.*]] = IE.Convert([[CONV]])  {dstElemType = f32} : tensor<1x2x1x1xf16> -> tensor<1x2x1x1xf32>
    // CHECK:     return [[CAST]]
}
