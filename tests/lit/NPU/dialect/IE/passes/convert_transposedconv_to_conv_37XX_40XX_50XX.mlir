//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-transposed-conv-to-conv %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @DoNotConvertTransposedConvToConv
module @DoNotConvertTransposedConvToConv {

config.PipelineOptions @Options {
    config.Option @config.EnableSEPtrsOperations : true 
}

// CHECK: func.func @main
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x32x23x30xf16>)
func.func @main(%input: tensor<1x32x23x30xf16>) -> tensor<1x16x46x60xf16> {
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
}

// -----

// CHECK-LABEL: @DoNotConvertTransposedConvToConvNonConstFilter
module @DoNotConvertTransposedConvToConvNonConstFilter {

config.PipelineOptions @Options {
    config.Option @config.EnableSEPtrsOperations : true 
}

// CHECK: func.func @main
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x30x30xf16>, [[WEIGHTS:%.+]]: tensor<16x1x16x16xf16>)
func.func @main(%input: tensor<1x16x30x30xf16>, %weights: tensor<16x1x16x16xf16>) -> tensor<1x16x74x74xf16> {
    %out = IE.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x16x30x30xf16>, tensor<16x1x16x16xf16> -> tensor<1x16x74x74xf16>

    return %out : tensor<1x16x74x74xf16>

    // CHECK-NOT:  IE.Upsampling
    // CHECK-NOT:  IE.Convolution
    // CHECK:      [[OUT:%.+]] = IE.TransposedConvolution([[INPUT]], [[WEIGHTS]])
    // CHECK:      return [[OUT]]
}
}

// -----

// CHECK-LABEL: @DoNotConvertTransposedConvToConvWithOutputPadding
module @DoNotConvertTransposedConvToConvWithOutputPadding {

config.PipelineOptions @Options {
    config.Option @config.EnableSEPtrsOperations : true 
}

// CHECK: func.func @main
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x32x23x30xf16>)
func.func @main(%input: tensor<1x32x23x30xf16>) -> tensor<1x16x47x61xf16> {
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
}
