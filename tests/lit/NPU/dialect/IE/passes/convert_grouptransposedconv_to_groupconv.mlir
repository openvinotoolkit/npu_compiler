//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-group-transposed-conv-to-groupconv %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ConvertGroupTransposedConvToGroupConv
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x64x64x64xf16>)
func.func @ConvertGroupTransposedConvToGroupConv(%arg0: tensor<1x64x64x64xf16>) -> tensor<1x64x130x130xf16> {
    %FILTERS = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>

    %RESULT = IE.GroupTransposedConvolution(%arg0, %FILTERS) {dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x130x130xf16>
    return %RESULT : tensor<1x64x130x130xf16>

    // CHECK:       [[UPS:%.+]] = IE.Upsampling([[ARG_0]]) {pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [3, 3], pads_width = [3, 3]>, upsampling_factor = [2, 2, 1]} : tensor<1x64x64x64xf16> -> tensor<1x64x133x133xf16>
    // CHECK:       [[CST_4D:%.+]] = const.Declare tensor<64x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>, [#const.Reshape<[64, 1, 4, 4]>]
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[UPS]], [[CST_4D]]) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x133x133xf16>, tensor<64x1x4x4xf16> -> tensor<1x64x130x130xf16>
    // CHECK:       return [[GROUPCONV]]
}

// -----

// CHECK-LABEL: @ConvertGroupTransposedConvToGroupConvWithPadding
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x64x64x64xf16>)
func.func @ConvertGroupTransposedConvToGroupConvWithPadding(%arg0: tensor<1x64x64x64xf16>) -> tensor<1x64x128x128xf16> {
    %FILTERS = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>

    %RESULT = IE.GroupTransposedConvolution(%arg0, %FILTERS) {dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x128x128xf16>
    return %RESULT : tensor<1x64x128x128xf16>

    // CHECK:       [[UPS:%.+]] = IE.Upsampling([[ARG_0]]) {pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [2, 2], pads_width = [2, 2]>, upsampling_factor = [2, 2, 1]} : tensor<1x64x64x64xf16> -> tensor<1x64x131x131xf16>
    // CHECK:       [[CST_4D:%.+]] = const.Declare tensor<64x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>, [#const.Reshape<[64, 1, 4, 4]>]
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[UPS]], [[CST_4D]]) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x131x131xf16>, tensor<64x1x4x4xf16> -> tensor<1x64x128x128xf16>
    // CHECK:       return [[GROUPCONV]]
}

// -----

// CHECK-LABEL: @ConvertGroupTransposedConvToGroupConvWithOutputPadding
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x64x64x64xf16>)
func.func @ConvertGroupTransposedConvToGroupConvWithOutputPadding(%arg0: tensor<1x64x64x64xf16>) -> tensor<1x64x131x131xf16> {
    %FILTERS = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>

    %RESULT = IE.GroupTransposedConvolution(%arg0, %FILTERS) {dilations = [1, 1], spatial_output_padding = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x131x131xf16>
    return %RESULT : tensor<1x64x131x131xf16>

    // CHECK:       [[UPS:%.+]] = IE.Upsampling([[ARG_0]]) {pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [3, 4], pads_width = [3, 4]>, upsampling_factor = [2, 2, 1]} : tensor<1x64x64x64xf16> -> tensor<1x64x134x134xf16>
    // CHECK:       [[CST_4D:%.+]] = const.Declare tensor<64x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>, [#const.Reshape<[64, 1, 4, 4]>]
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[UPS]], [[CST_4D]]) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x134x134xf16>, tensor<64x1x4x4xf16> -> tensor<1x64x131x131xf16>
    // CHECK:       return [[GROUPCONV]]
}

// -----

// CHECK-LABEL: @ConvertGroupTransposedConvToGroupConvWithMultiplyAsFilter
func.func @ConvertGroupTransposedConvToGroupConvWithMultiplyAsFilter(%INPUT: tensor<1x64x64x64xf16>) -> tensor<1x64x130x130xf16> {
    %FILTERS_INITIAL = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>
    %FACTOR = const.Declare tensor<64x1x1x4x4xf16> = dense<2.000000e+00> : tensor<64x1x1x4x4xf16>
    %FILTERS = IE.Multiply(%FILTERS_INITIAL, %FACTOR) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<64x1x1x4x4xf16>, tensor<64x1x1x4x4xf16> -> tensor<64x1x1x4x4xf16>

    %RESULT = IE.GroupTransposedConvolution(%INPUT, %FILTERS) {dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x130x130xf16>
    return %RESULT : tensor<1x64x130x130xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<64x1x1x4x4xf16> = dense<2.000000e+00> : tensor<64x1x1x4x4xf16>
    // CHECK:       [[FILTER:%.+]] = IE.Multiply([[CST]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<64x1x1x4x4xf16>, tensor<64x1x1x4x4xf16> -> tensor<64x1x1x4x4xf16>

    // CHECK:       [[UPS:%.+]] = IE.Upsampling
    // CHECK-SAME:      #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [3, 3], pads_width = [3, 3]>
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x64x64x64xf16> -> tensor<1x64x133x133xf16>
    // CHECK:       [[CST_4D:%.+]] = IE.Reshape([[FILTER]]) {shape_value = [64, 1, 4, 4]} : tensor<64x1x1x4x4xf16> -> tensor<64x1x4x4xf16>
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[UPS]], [[CST_4D]]) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x133x133xf16>, tensor<64x1x4x4xf16> -> tensor<1x64x130x130xf16>
    // CHECK:       return [[GROUPCONV]]
}

// -----

// CHECK-LABEL: @ConvertDilatedGroupTransposedConvToGroupConv
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x64x3x3xf16>)
func.func @ConvertDilatedGroupTransposedConvToGroupConv(%arg0: tensor<1x64x3x3xf16>) -> tensor<1x64x5x5xf16> {
    %FILTERS = const.Declare tensor<64x1x1x2x2xf16> = dense<1.000000e+00> : tensor<64x1x1x2x2xf16>

    %RESULT = IE.GroupTransposedConvolution(%arg0, %FILTERS) {dilations = [2, 2], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x3x3xf16>, tensor<64x1x1x2x2xf16> -> tensor<1x64x5x5xf16>
    return %RESULT : tensor<1x64x5x5xf16>

    // Stride-1 dilated depthwise: emit ExpandDilated + direct GroupConv(dilations=[1,1]) with
    // equivalent padding. LegalizeDilatedConvolution runs before this pass, so dilation must be
    // expanded here explicitly. E#222712
    // CHECK:       [[CST_4D:%.+]] = const.Declare tensor<64x1x2x2xf16>
    // CHECK:       [[EXPAND:%.+]] = IE.ExpandDilated([[CST_4D]]) {dilations = [2, 2]}
    // CHECK-SAME:      tensor<64x1x2x2xf16> -> tensor<64x1x3x3xf16>
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[ARG_0]], [[EXPAND]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [2, 2], pads_end = [2, 2]
    // CHECK:       return [[GROUPCONV]]
}

// -----

// CHECK-LABEL: @ConvertGroupTransposedConvToGroupConvLargeKernelSize
module @ConvertGroupTransposedConvToGroupConvLargeKernelSize {

config.PipelineOptions @Options {
    config.Option @config.EnableSEPtrsOperations : true
}

// CHECK: func.func @main
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x64x64xf16>)
func.func @main(%input: tensor<1x64x64x64xf16>) -> tensor<1x64x142x142xf16> {
    %weights = const.Declare tensor<64x1x1x16x16xf16> = dense<1.000000e+00> : tensor<64x1x1x16x16xf16>
    %out = IE.GroupTransposedConvolution(%input, %weights) {
            dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x64x64x64xf16>, tensor<64x1x1x16x16xf16> -> tensor<1x64x142x142xf16>
    return %out : tensor<1x64x142x142xf16>

    // CHECK:  [[UPSAMPLING:%.+]] = IE.Upsampling([[INPUT]])
    // CHECK:  [[WEIGHTS:%.+]] = const.Declare tensor<64x1x16x16xf16>
    // CHECK:  [[OUT:%.+]] = IE.GroupConvolution([[UPSAMPLING]], [[WEIGHTS]])
    // CHECK:  return [[OUT]]
}
}
