//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-group-transposed-conv-to-groupconv %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @DoNotConvertGroupTransposedConvToGroupConv
module @DoNotConvertGroupTransposedConvToGroupConv {

config.PipelineOptions @Options {
    config.Option @config.EnableSEPtrsOperations : true
}

// CHECK: func.func @main
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x64x64xf16>)
func.func @main(%input: tensor<1x64x64x64xf16>) -> tensor<1x64x130x130xf16> {
    %weights = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>
    %out = IE.GroupTransposedConvolution(%input, %weights) {
            dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x130x130xf16>
    return %out : tensor<1x64x130x130xf16>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare
    // CHECK-NOT:  IE.Upsampling
    // CHECK-NOT:  IE.GroupConvolution
    // CHECK:      [[OUT:%.+]] = IE.GroupTransposedConvolution([[INPUT]], [[WEIGHTS]])
    // CHECK:      return [[OUT]]
}
}

// -----

// CHECK-LABEL: @DoNotConvertGroupTransposedConvToGroupConv
module @DoNotConvertGroupTransposedConvToGroupConv {

config.PipelineOptions @Options {
    config.Option @config.EnableSEPtrsOperations : true
}

// CHECK: func.func @main
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x64x64xf16>)
func.func @main(%input: tensor<1x64x64x64xf16>) -> tensor<1x64x128x128xf16> {
    %weights = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>
    %out = IE.GroupTransposedConvolution(%input, %weights) {
            dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]
        } : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x128x128xf16>
    return %out : tensor<1x64x128x128xf16>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare
    // CHECK-NOT:  IE.Upsampling
    // CHECK-NOT:  IE.GroupConvolution
    // CHECK:      [[OUT:%.+]] = IE.GroupTransposedConvolution([[INPUT]], [[WEIGHTS]])
    // CHECK:      return [[OUT]]
}
}

// -----

// CHECK-LABEL: @DoNotConvertGroupTransposedConvToGroupConvWithOutputPadding
module @DoNotConvertGroupTransposedConvToGroupConvWithOutputPadding {

config.PipelineOptions @Options {
    config.Option @config.EnableSEPtrsOperations : true
}

// CHECK: func.func @main
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x64x64xf16>)
func.func @main(%input: tensor<1x64x64x64xf16>) -> tensor<1x64x131x131xf16> {
    %weights = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>
    %out = IE.GroupTransposedConvolution(%input, %weights) {
            dilations = [1, 1], spatial_output_padding = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x131x131xf16>
    return %out : tensor<1x64x131x131xf16>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare
    // CHECK-NOT:  IE.Upsampling
    // CHECK-NOT:  IE.GroupConvolution
    // CHECK:      [[OUT:%.+]] = IE.GroupTransposedConvolution([[INPUT]], [[WEIGHTS]])
    // CHECK:      return [[OUT]]
}
}
