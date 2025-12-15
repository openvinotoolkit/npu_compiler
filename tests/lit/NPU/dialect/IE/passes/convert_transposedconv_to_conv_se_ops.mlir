//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-transposed-conv-to-conv="enable-sep-transposed-conv=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: func.func @DoNotConvertTransposedConvToConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x32x23x30xf16>)
func.func @DoNotConvertTransposedConvToConv(%input: tensor<1x32x23x30xf16>) -> tensor<1x16x46x60xf16> {
    %weights = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<16x32x2x2xf16>
    %out = IE.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>
    return %out : tensor<1x16x46x60xf16>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare
    // CHECK-NOT:  IE.Upsampling
    // CHECK-NOT:  IE.Convolution
    // CHECK:      [[OUT:%.+]] = IE.TransposedConvolution([[INPUT]], [[WEIGHTS]])
    // CHECK:      return [[OUT]]
}

// CHECK-LABEL: func.func @DoNotConvertTransposedConvToConvNonConstFilter
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x30x30xf16>, [[WEIGHTS:%.+]]: tensor<16x1x16x16xf16>)
func.func @DoNotConvertTransposedConvToConvNonConstFilter(%input: tensor<1x16x30x30xf16>, %weights: tensor<16x1x16x16xf16>) -> tensor<1x16x74x74xf16> {
    %out = IE.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x16x30x30xf16>, tensor<16x1x16x16xf16> -> tensor<1x16x74x74xf16>

    return %out : tensor<1x16x74x74xf16>

    // CHECK-NOT:  IE.Upsampling
    // CHECK-NOT:  IE.Convolution
    // CHECK:      [[OUT:%.+]] = IE.TransposedConvolution([[INPUT]], [[WEIGHTS]])
    // CHECK:      return [[OUT]]
}

// -----

// CHECK-LABEL: func.func @DoNotConvertTransposedConvToConvWithOutputPadding
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x32x23x30xf16>)
func.func @DoNotConvertTransposedConvToConvWithOutputPadding(%input: tensor<1x32x23x30xf16>) -> tensor<1x16x47x61xf16> {
    %weights = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<16x32x2x2xf16>
    %out = IE.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x47x61xf16>
    return %out : tensor<1x16x47x61xf16>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare
    // CHECK-NOT:  IE.Upsampling
    // CHECK-NOT:  IE.Convolution
    // CHECK:      [[OUT:%.+]] = IE.TransposedConvolution([[INPUT]], [[WEIGHTS]])
    // CHECK:      return [[OUT]]
}

// -----

// CHECK-LABEL: func.func @ConvertTransposedConvToConvLargeKernelSize
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x32x23x30xf16>)
func.func @ConvertTransposedConvToConvLargeKernelSize(%input: tensor<1x32x23x30xf16>) -> tensor<1x16x60x74xf16> {
    %weights = const.Declare tensor<16x32x16x16xf16> = dense<1.000000e+00> : tensor<16x32x16x16xf16>
    %out = IE.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30xf16>, tensor<16x32x16x16xf16> -> tensor<1x16x60x74xf16>
    return %out : tensor<1x16x60x74xf16>

    // CHECK:  [[WEIGHTS:%.+]] = const.Declare tensor<16x32x16x16xf16>
    // CHECK:  [[UPSAMPLING:%.+]] = IE.Upsampling([[INPUT]])
    // CHECK:  [[OUT:%.+]] = IE.Convolution([[UPSAMPLING]], [[WEIGHTS]])
    // CHECK:  return [[OUT]]
}
