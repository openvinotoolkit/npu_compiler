//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-fake-convert-to-dynamic-dequantize --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

// CHECK-LABEL: @ConvertFakeConvertWeightsToDynamicDequantize
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x4x4xf16>
func.func @ConvertFakeConvertWeightsToDynamicDequantize(%input: tensor<1x1x4x4xf16>) -> tensor<1x3x4x4xf16> {
    %weights = const.Declare tensor<3x1x1x1xf16> = dense<[[[[-448.0]]], [[[0.0]]], [[[448.0]]]]> : tensor<3x1x1x1xf16>
    %scale = const.Declare tensor<1xf16> = dense<0.500000e+00> : tensor<1xf16>
    %shift = const.Declare tensor<3x1x1x1xf16> = dense<5.000000e+00> : tensor<3x1x1x1xf16>
    %0 = IE.FakeConvert(%weights, %scale, %shift) {dst_type = f8E4M3FN} : tensor<3x1x1x1xf16>, tensor<1xf16>, tensor<3x1x1x1xf16> -> tensor<3x1x1x1xf16>
    %1 = IE.Convolution(%input, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x4x4xf16>, tensor<3x1x1x1xf16> -> tensor<1x3x4x4xf16>

    return %1 : tensor<1x3x4x4xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<3x1x1x1xf8E4M3FN>
    // CHECK-SAME{LITERAL}: dense<[[[[-2.240000e+02]]], [[[-5.000000e+00]]], [[[2.240000e+02]]]]>
    // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00>
    // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<3x1x1x1xf8E4M3FN> = dense<-5.000000e+00>
    // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f16, vpux.weights_import_dyn_dequant}
    // CHECK-SAME: : tensor<3x1x1x1xf8E4M3FN>, tensor<1xf32>, tensor<3x1x1x1xf8E4M3FN> -> tensor<3x1x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]])
    // CHECK: return [[CONV]] : tensor<1x3x4x4xf16>
}

// -----

// A weights FakeConvert without a shift becomes a DynamicDequantize without a zero-point.

// CHECK-LABEL: @ConvertFakeConvertWeightsNoShiftToDynamicDequantize
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x4x4xf16>
func.func @ConvertFakeConvertWeightsNoShiftToDynamicDequantize(%input: tensor<1x1x4x4xf16>) -> tensor<1x3x4x4xf16> {
    %weights = const.Declare tensor<3x1x1x1xf16> = dense<[[[[-448.0]]], [[[0.0]]], [[[448.0]]]]> : tensor<3x1x1x1xf16>
    %scale = const.Declare tensor<1xf16> = dense<0.500000e+00> : tensor<1xf16>
    %0 = IE.FakeConvert(%weights, %scale) {dst_type = f8E4M3FN} : tensor<3x1x1x1xf16>, tensor<1xf16> -> tensor<3x1x1x1xf16>
    %1 = IE.Convolution(%input, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x4x4xf16>, tensor<3x1x1x1xf16> -> tensor<1x3x4x4xf16>

    return %1 : tensor<1x3x4x4xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<3x1x1x1xf8E4M3FN>
    // CHECK-SAME{LITERAL}: dense<[[[[-2.240000e+02]]], [[[0.000000e+00]]], [[[2.240000e+02]]]]>
    // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00>
    // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]]) {dstElemType = f16, vpux.weights_import_dyn_dequant}
    // CHECK-SAME: : tensor<3x1x1x1xf8E4M3FN>, tensor<1xf32> -> tensor<3x1x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]])
    // CHECK: return [[CONV]] : tensor<1x3x4x4xf16>
}

// -----

// A shift that is not exactly representable in the destination f8 type is left as a FakeConvert so that the
// ConvertFakeConvertToFakeQuantize fallback keeps the shift at the weights' float precision.

// CHECK-LABEL: @DoNotConvertFakeConvertWeightsWithInexactShift
func.func @DoNotConvertFakeConvertWeightsWithInexactShift(%input: tensor<1x1x4x4xf16>) -> tensor<1x3x4x4xf16> {
    %weights = const.Declare tensor<3x1x1x1xf16> = dense<[[[[-448.0]]], [[[0.0]]], [[[448.0]]]]> : tensor<3x1x1x1xf16>
    %scale = const.Declare tensor<1xf16> = dense<0.500000e+00> : tensor<1xf16>
    %shift = const.Declare tensor<3x1x1x1xf16> = dense<5.300000e+00> : tensor<3x1x1x1xf16>
    %0 = IE.FakeConvert(%weights, %scale, %shift) {dst_type = f8E4M3FN} : tensor<3x1x1x1xf16>, tensor<1xf16>, tensor<3x1x1x1xf16> -> tensor<3x1x1x1xf16>
    %1 = IE.Convolution(%input, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x4x4xf16>, tensor<3x1x1x1xf16> -> tensor<1x3x4x4xf16>

    return %1 : tensor<1x3x4x4xf16>

    // CHECK: [[FC:%.+]] = IE.FakeConvert
    // CHECK-NOT: IE.DynamicDequantize
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT:%.+]], [[FC]])
    // CHECK: return [[CONV]] : tensor<1x3x4x4xf16>
}
