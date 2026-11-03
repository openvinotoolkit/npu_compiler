//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% enable-asymmetric-per-tensor-zp=true" --low-precision %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK: !qElemType = !quant.uniform<i8:f16, 0.011764706112444401:-43>
// CHECK: !qElemType1 = !quant.uniform<i8:f16, 0.011764705882352941:-43>
// CHECK: !qElemType2 = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: func.func @AsymmetricWeightsI8([[INPUT:%.+]]: tensor<1x3x16x16xf16>)
func.func @AsymmetricWeightsI8(%input: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %weights = const.Declare tensor<3x3x1x1xf16> = dense<1> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>]
    %low = const.Declare tensor<1x1x1x1xf16> = dense<[[[[-1.0]]]]> : tensor<1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1xf16> = dense<[[[[2.0]]]]> : tensor<1x1x1x1xf16>
    %fq_weights = IE.FakeQuantize(%weights, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<3x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<3x3x1x1xf16>
    %conv = IE.Convolution(%input, %fq_weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>
    return %conv : tensor<1x3x16x16xf16>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<3x3x1x1x!qElemType> = dense<1>
    // CHECK-SAME:    [#const.CastElemType<f16>, #const.Quantize<!qElemType1>, #const.CastElemType<si8>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType>]
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]])
    // CHECK-SAME:    : tensor<1x3x16x16xf16>, tensor<3x3x1x1x!qElemType> -> tensor<1x3x16x16xf16>
    // CHECK:       return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<i4:f16, 0.20000000298023224:-3>
// CHECK: !qElemType1 = !quant.uniform<i4:f16, 2.000000e-01:-3>
// CHECK: !qElemType2 = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK: func.func @AsymmetricWeightsI4([[INPUT:%.+]]: tensor<1x3x16x16xf16>)
func.func @AsymmetricWeightsI4(%input: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %weights = const.Declare tensor<3x3x1x1xf16> = dense<1> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>]
    %low = const.Declare tensor<1x1x1x1xf16> = dense<[[[[-1.0]]]]> : tensor<1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1xf16> = dense<[[[[2.0]]]]> : tensor<1x1x1x1xf16>
    %fq_weights = IE.FakeQuantize(%weights, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 16 : i64} : tensor<3x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<3x3x1x1xf16>
    %conv = IE.Convolution(%input, %fq_weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>
    return %conv : tensor<1x3x16x16xf16>

    // CHECK:   [[WEIGHTS:%.+]] = const.Declare tensor<3x3x1x1x!qElemType> = dense<1>
    // CHECK-SAME:    [#const.CastElemType<f16>, #const.Quantize<!qElemType1>, #const.CastElemType<si4>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType>]
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]])
    // CHECK-SAME:    : tensor<1x3x16x16xf16>, tensor<3x3x1x1x!qElemType> -> tensor<1x3x16x16xf16>
    // CHECK:       return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<i2:f16, 1.000000e+00:-1>
// CHECK: !qElemType1 = !quant.uniform<i2:f16, 1.000000e+00>

// CHECK: func.func @AsymmetricWeightsI2([[INPUT:%.+]]: tensor<1x3x16x16xf16>)
func.func @AsymmetricWeightsI2(%input: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %weights = const.Declare tensor<3x3x1x1xf16> = dense<1> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>]
    %low = const.Declare tensor<1x1x1x1xf16> = dense<[[[[-1.0]]]]> : tensor<1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1xf16> = dense<[[[[2.0]]]]> : tensor<1x1x1x1xf16>
    %fq_weights = IE.FakeQuantize(%weights, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 4 : i64} : tensor<3x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<3x3x1x1xf16>
    %conv = IE.Convolution(%input, %fq_weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>
    return %conv : tensor<1x3x16x16xf16>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<3x3x1x1x!qElemType> = dense<1>
    // CHECK-SAME:    [#const.CastElemType<f16>, #const.Quantize<!qElemType>, #const.CastElemType<si2>, #const.CastElemType<!qElemType1>, #const.CastElemType<!qElemType>]
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]])
    // CHECK-SAME:    : tensor<1x3x16x16xf16>, tensor<3x3x1x1x!qElemType> -> tensor<1x3x16x16xf16>
    // CHECK:       return [[CONV]]
}
