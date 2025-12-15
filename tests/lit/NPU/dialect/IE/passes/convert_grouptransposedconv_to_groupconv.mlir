//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-group-transposed-conv-to-groupconv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertGroupTransposedConvToGroupConv
func.func @ConvertGroupTransposedConvToGroupConv(%arg0: tensor<1x64x64x64xf16>) -> tensor<1x64x130x130xf16> {
    %FILTERS = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>

    %RESULT = IE.GroupTransposedConvolution(%arg0, %FILTERS) {dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x130x130xf16>
    return %RESULT : tensor<1x64x130x130xf16>

    // CHECK:       [[UPS:%.*]] = IE.Upsampling(%arg0) {pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [3, 3], pads_width = [3, 3]>, upsampling_factor = [2, 2, 1]} : tensor<1x64x64x64xf16> -> tensor<1x64x133x133xf16>
    // CHECK:       [[CST_4D:%.*]] = const.Declare tensor<64x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>, [#const.Reshape<[64, 1, 4, 4]>]
    // CHECK:       [[GROUPCONV:%.*]] = IE.GroupConvolution([[UPS]], [[CST_4D]]) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x133x133xf16>, tensor<64x1x4x4xf16> -> tensor<1x64x130x130xf16>
    // CHECK:       return [[GROUPCONV]]
}

// -----

// CHECK-LABEL: @ConvertGroupTransposedConvToGroupConvWithPadding
func.func @ConvertGroupTransposedConvToGroupConvWithPadding(%arg0: tensor<1x64x64x64xf16>) -> tensor<1x64x128x128xf16> {
    %FILTERS = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>

    %RESULT = IE.GroupTransposedConvolution(%arg0, %FILTERS) {dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x128x128xf16>
    return %RESULT : tensor<1x64x128x128xf16>

    // CHECK:       [[UPS:%.*]] = IE.Upsampling(%arg0) {pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [2, 2], pads_width = [2, 2]>, upsampling_factor = [2, 2, 1]} : tensor<1x64x64x64xf16> -> tensor<1x64x131x131xf16>
    // CHECK:       [[CST_4D:%.*]] = const.Declare tensor<64x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>, [#const.Reshape<[64, 1, 4, 4]>]
    // CHECK:       [[GROUPCONV:%.*]] = IE.GroupConvolution([[UPS]], [[CST_4D]]) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x131x131xf16>, tensor<64x1x4x4xf16> -> tensor<1x64x128x128xf16>
    // CHECK:       return [[GROUPCONV]]
}

// -----

// CHECK-LABEL: @ConvertGroupTransposedConvToGroupConvWithOutputPadding
func.func @ConvertGroupTransposedConvToGroupConvWithOutputPadding(%arg0: tensor<1x64x64x64xf16>) -> tensor<1x64x131x131xf16> {
    %FILTERS = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>

    %RESULT = IE.GroupTransposedConvolution(%arg0, %FILTERS) {dilations = [1, 1], spatial_output_padding = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x131x131xf16>
    return %RESULT : tensor<1x64x131x131xf16>

    // CHECK:       [[UPS:%.*]] = IE.Upsampling(%arg0) {pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [3, 4], pads_width = [3, 4]>, upsampling_factor = [2, 2, 1]} : tensor<1x64x64x64xf16> -> tensor<1x64x134x134xf16>
    // CHECK:       [[CST_4D:%.*]] = const.Declare tensor<64x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>, [#const.Reshape<[64, 1, 4, 4]>]
    // CHECK:       [[GROUPCONV:%.*]] = IE.GroupConvolution([[UPS]], [[CST_4D]]) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x134x134xf16>, tensor<64x1x4x4xf16> -> tensor<1x64x131x131xf16>
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
