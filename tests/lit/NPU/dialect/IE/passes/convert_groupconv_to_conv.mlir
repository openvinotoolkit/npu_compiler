//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-groupconv-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertGroupConvToSingleConv
func.func @ConvertGroupConvToSingleConv(%arg0: tensor<1x64x80x80xf16>) -> tensor<1x64x80x80xf16> {
    %weights = const.Declare tensor<64x16x3x3xf16> = dense<1.0> : tensor<64x16x3x3xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %weights, %bias) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    return %result : tensor<1x64x80x80xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[ORG_WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>

    // CHECK-DAG:   [[WEIGHTS0:%.+]] = const.Declare tensor<16x64x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[0, 0, 0, 0], [16, 16, 3, 3]>, #const.PadWithZero<[0, 0, 0, 0], [0, 48, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS1:%.+]] = const.Declare tensor<16x64x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[16, 0, 0, 0], [16, 16, 3, 3]>, #const.PadWithZero<[0, 16, 0, 0], [0, 32, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS2:%.+]] = const.Declare tensor<16x64x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[32, 0, 0, 0], [16, 16, 3, 3]>, #const.PadWithZero<[0, 32, 0, 0], [0, 16, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS3:%.+]] = const.Declare tensor<16x64x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[48, 0, 0, 0], [16, 16, 3, 3]>, #const.PadWithZero<[0, 48, 0, 0], [0, 0, 0, 0]>]

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[WEIGHTS0]], [[WEIGHTS1]], [[WEIGHTS2]], [[WEIGHTS3]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:                      tensor<16x64x3x3xf16>, tensor<16x64x3x3xf16>, tensor<16x64x3x3xf16>, tensor<16x64x3x3xf16> -> tensor<64x64x3x3xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[CONCAT]], [[BIAS]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x64x80x80xf16>, tensor<64x64x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @ConvertGroupConvToMultiConvDueToNonConstWeights
func.func @ConvertGroupConvToMultiConvDueToNonConstWeights(%arg0: tensor<1x64x80x80xf16>, %arg1: tensor<64x16x3x3xf16>) -> tensor<1x64x80x80xf16> {
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %arg1, %bias) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    return %result : tensor<1x64x80x80xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>
    // CHECK:       [[INPUT0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS0:%.*]] = IE.Slice %arg1 [0, 0, 0, 0] [16, 16, 3, 3]
    // CHECK-SAME:      : tensor<64x16x3x3xf16> to tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[BIAS0:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV_0:%.*]] = IE.Convolution([[INPUT0]], [[WEIGHTS0]], [[BIAS0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    // CHECK-DAG:   [[INPUT1:%.*]] = IE.Slice %arg0 [0, 16, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS1:%.*]] = IE.Slice %arg1 [16, 0, 0, 0] [16, 16, 3, 3]
    // CHECK-SAME:      : tensor<64x16x3x3xf16> to tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[BIAS1:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 16, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV_1:%.*]] = IE.Convolution([[INPUT1]], [[WEIGHTS1]], [[BIAS1]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    // CHECK-DAG:   [[INPUT2:%.*]] = IE.Slice %arg0 [0, 32, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS2:%.*]] = IE.Slice %arg1 [32, 0, 0, 0] [16, 16, 3, 3]
    // CHECK-SAME:      : tensor<64x16x3x3xf16> to tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[BIAS2:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 32, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV_2:%.*]] = IE.Convolution([[INPUT2]], [[WEIGHTS2]], [[BIAS2]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    // CHECK-DAG:   [[INPUT3:%.*]] = IE.Slice %arg0 [0, 48, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS3:%.*]] = IE.Slice %arg1 [48, 0, 0, 0] [16, 16, 3, 3]
    // CHECK-SAME:      : tensor<64x16x3x3xf16> to tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[BIAS3:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 48, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV_3:%.*]] = IE.Convolution([[INPUT3]], [[WEIGHTS3]], [[BIAS3]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>
    // CHECK:       [[RESULT:%.*]] = IE.Concat([[CONV_0]], [[CONV_1]], [[CONV_2]], [[CONV_3]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<1x16x80x80xf16>, tensor<1x16x80x80xf16>, tensor<1x16x80x80xf16>
    // CHECK-SAME:      -> tensor<1x64x80x80xf16>
    // CHECK:       return [[RESULT]]
}

// -----

// CHECK-LABEL: @ConvertGroupConvToSingleConvWithAsymmetricalWeights
func.func @ConvertGroupConvToSingleConvWithAsymmetricalWeights(%arg0: tensor<1x16x1x30xf16>) -> tensor<1x8x1x28xf16> {
    %weights = const.Declare tensor<8x8x1x3xf16> = dense<1.0> : tensor<2x4x8x3xf16>, [#const.Reshape<[8, 8, 3]>, #const.Reshape<[8, 8, 1, 3]>]
    %result = IE.GroupConvolution(%arg0, %weights) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x30xf16>, tensor<8x8x1x3xf16> -> tensor<1x8x1x28xf16>

    return %result : tensor<1x8x1x28xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[ORG_WEIGHTS:%.+]] = const.Declare tensor<8x8x1x3xf16> = dense<1.000000e+00> : tensor<2x4x8x3xf16>, [#const.Reshape<[8, 8, 3]>, #const.Reshape<[8, 8, 1, 3]>]

    // CHECK-DAG:   [[WEIGHTS0:%.+]] = const.Declare tensor<4x16x1x3xf16> = dense<1.000000e+00> : tensor<2x4x8x3xf16>
    // CHECK-SAME:                     [#const.Reshape<[8, 8, 3]>, #const.SubView<[0, 0, 0], [4, 8, 3]>, #const.Reshape<[4, 8, 1, 3]>, #const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS1:%.+]] = const.Declare tensor<4x16x1x3xf16> = dense<1.000000e+00> : tensor<2x4x8x3xf16>
    // CHECK-SAME:                     [#const.Reshape<[8, 8, 3]>, #const.SubView<[4, 0, 0], [4, 8, 3]>, #const.Reshape<[4, 8, 1, 3]>, #const.PadWithZero<[0, 8, 0, 0], [0, 0, 0, 0]>]

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[WEIGHTS0]], [[WEIGHTS1]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:                      tensor<4x16x1x3xf16>, tensor<4x16x1x3xf16> -> tensor<8x16x1x3xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[CONCAT]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x1x30xf16>, tensor<8x16x1x3xf16> -> tensor<1x8x1x28xf16>

    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @ConvertGroupConvToSingleConvWhenChannelNotAligned
func.func @ConvertGroupConvToSingleConvWhenChannelNotAligned(%arg0: tensor<1x16x80x80xf16>) -> tensor<1x16x80x80xf16> {
    %weights = const.Declare tensor<16x4x3x3xf16> = dense<1.0> : tensor<16x4x3x3xf16>
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %weights, %bias) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x80x80xf16>, tensor<16x4x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    return %result : tensor<1x16x80x80xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[ORG_WEIGHTS:%.+]] = const.Declare tensor<16x4x3x3xf16> = dense<1.000000e+00> : tensor<16x4x3x3xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    // CHECK-DAG:   [[WEIGHTS0:%.+]] = const.Declare tensor<4x16x3x3xf16> = dense<1.000000e+00> : tensor<16x4x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[0, 0, 0, 0], [4, 4, 3, 3]>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS1:%.+]] = const.Declare tensor<4x16x3x3xf16> = dense<1.000000e+00> : tensor<16x4x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[4, 0, 0, 0], [4, 4, 3, 3]>, #const.PadWithZero<[0, 4, 0, 0], [0, 8, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS2:%.+]] = const.Declare tensor<4x16x3x3xf16> = dense<1.000000e+00> : tensor<16x4x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[8, 0, 0, 0], [4, 4, 3, 3]>, #const.PadWithZero<[0, 8, 0, 0], [0, 4, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS3:%.+]] = const.Declare tensor<4x16x3x3xf16> = dense<1.000000e+00> : tensor<16x4x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[12, 0, 0, 0], [4, 4, 3, 3]>, #const.PadWithZero<[0, 12, 0, 0], [0, 0, 0, 0]>]

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[WEIGHTS0]], [[WEIGHTS1]], [[WEIGHTS2]], [[WEIGHTS3]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:                      tensor<4x16x3x3xf16>, tensor<4x16x3x3xf16>, tensor<4x16x3x3xf16>, tensor<4x16x3x3xf16> -> tensor<16x16x3x3xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[CONCAT]], [[BIAS]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @NotConvertForDWConv
func.func @NotConvertForDWConv(%arg0: tensor<1x16x80x80xf16>) -> tensor<1x16x80x80xf16> {
    %weights = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %weights, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x80x80xf16>, tensor<16x1x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    return %result : tensor<1x16x80x80xf16>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x1x3x3xf16> = dense<1.000000e+00> : tensor<16x1x3x3xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.GroupConvolution(%arg0, [[WEIGHTS]], [[BIAS]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x1x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>
}

// -----

// CHECK-LABEL: @ConvertForDWConvIfCannotLoweringToNCEDepthConv
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x6x1x31xf16>
func.func @ConvertForDWConvIfCannotLoweringToNCEDepthConv(%arg0: tensor<1x6x1x31xf16>) -> tensor<1x6x1x16xf16> {
    %weights = const.Declare tensor<6x1x1x16xf16> = dense<1.0> : tensor<6x1x1x16xf16>
    %result = IE.GroupConvolution(%arg0, %weights) {dilations = [1, 1], groups = 6 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x6x1x31xf16>, tensor<6x1x1x16xf16> -> tensor<1x6x1x16xf16>

    return %result : tensor<1x6x1x16xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<6x1x1x16xf16> = dense<1.000000e+00> : tensor<6x1x1x16xf16>
    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x6x1x16xf16> = dense<1.000000e+00> : tensor<6x1x1x16xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 16]>, #const.PadWithZero<[0, 0, 0, 0], [0, 5, 0, 0]>]
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x6x1x16xf16> = dense<1.000000e+00> : tensor<6x1x1x16xf16>, [#const.SubView<[1, 0, 0, 0], [1, 1, 1, 16]>, #const.PadWithZero<[0, 1, 0, 0], [0, 4, 0, 0]>]
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x6x1x16xf16> = dense<1.000000e+00> : tensor<6x1x1x16xf16>, [#const.SubView<[2, 0, 0, 0], [1, 1, 1, 16]>, #const.PadWithZero<[0, 2, 0, 0], [0, 3, 0, 0]>]
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x6x1x16xf16> = dense<1.000000e+00> : tensor<6x1x1x16xf16>, [#const.SubView<[3, 0, 0, 0], [1, 1, 1, 16]>, #const.PadWithZero<[0, 3, 0, 0], [0, 2, 0, 0]>]
    // CHECK-DAG:   [[CST4:%.+]] = const.Declare tensor<1x6x1x16xf16> = dense<1.000000e+00> : tensor<6x1x1x16xf16>, [#const.SubView<[4, 0, 0, 0], [1, 1, 1, 16]>, #const.PadWithZero<[0, 4, 0, 0], [0, 1, 0, 0]>]
    // CHECK-DAG:   [[CST5:%.+]] = const.Declare tensor<1x6x1x16xf16> = dense<1.000000e+00> : tensor<6x1x1x16xf16>, [#const.SubView<[5, 0, 0, 0], [1, 1, 1, 16]>, #const.PadWithZero<[0, 5, 0, 0], [0, 0, 0, 0]>]

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[CST0]], [[CST1]], [[CST2]], [[CST3]], [[CST4]], [[CST5]]) {per_axis = #IE.Concat<axis = 0 : i64>} :
    // CHECK-SAME:                      tensor<1x6x1x16xf16>, tensor<1x6x1x16xf16>, tensor<1x6x1x16xf16>, tensor<1x6x1x16xf16>, tensor<1x6x1x16xf16>, tensor<1x6x1x16xf16> -> tensor<6x6x1x16xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CONCAT]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x6x1x31xf16>, tensor<6x6x1x16xf16> -> tensor<1x6x1x16xf16>

    // CHECK:       return [[CONV]] : tensor<1x6x1x16xf16>
}

// -----

// CHECK-LABEL: @ConvertQuantizedGroupConvToSingleConv
func.func @ConvertQuantizedGroupConvToSingleConv(%arg0: tensor<1x64x80x80xf16>) -> tensor<1x64x80x80xf16> {
    %weights = const.Declare tensor<64x16x3x3xf16> = dense<1.0> : tensor<64x16x3x3xf16>
    %weights_low = const.Declare tensor<64x1x1x1xf16> = dense<-1.270000e+02> : tensor<64x1x1x1xf16>
    %weights_high = const.Declare tensor<64x1x1x1xf16> = dense<1.270000e+02> : tensor<64x1x1x1xf16>
    %fq_weights = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                    levels = 255 : i64
                } : tensor<64x16x3x3xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16> -> tensor<64x16x3x3xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %fq_weights, %bias) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    return %result : tensor<1x64x80x80xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[ORG_WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-DAG:   [[FQ_LOW:%.+]] = const.Declare tensor<64x1x1x1xf16> = dense<-1.270000e+02> : tensor<64x1x1x1xf16>
    // CHECK-DAG:   [[FQ_HIGH:%.+]] = const.Declare tensor<64x1x1x1xf16> = dense<1.270000e+02> : tensor<64x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>

    // CHECK-DAG:   [[WEIGHTS0_SLICE:%.+]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[0, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[WEIGHTS0_PAD_AFTER:%.+]] = const.Declare tensor<16x48x3x3xf16> = dense<0.000000e+00> : tensor<16x48x3x3xf16>
    // CHECK-DAG:   [[WEIGHTS0:%.+]] = IE.Concat([[WEIGHTS0_SLICE]], [[WEIGHTS0_PAD_AFTER]]) {
    // CHECK-SAME:                     per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x16x3x3xf16>, tensor<16x48x3x3xf16> -> tensor<16x64x3x3xf16>

    // CHECK-DAG:   [[WEIGHTS1_PAD_BEFORE:%.+]] = const.Declare tensor<16x16x3x3xf16> = dense<0.000000e+00> : tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[WEIGHTS1_SLICE:%.+]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[16, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[WEIGHTS1_PAD_AFTER:%.+]] = const.Declare tensor<16x32x3x3xf16> = dense<0.000000e+00> : tensor<16x32x3x3xf16>
    // CHECK-DAG:   [[WEIGHTS1:%.+]] = IE.Concat([[WEIGHTS1_PAD_BEFORE]], [[WEIGHTS1_SLICE]], [[WEIGHTS1_PAD_AFTER]]) {
    // CHECK-SAME:                     per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x16x3x3xf16>, tensor<16x16x3x3xf16>, tensor<16x32x3x3xf16> -> tensor<16x64x3x3xf16>

    // CHECK-DAG:   [[WEIGHTS2_PAD_BEFORE:%.+]] = const.Declare tensor<16x32x3x3xf16> = dense<0.000000e+00> : tensor<16x32x3x3xf16>
    // CHECK-DAG:   [[WEIGHTS2_SLICE:%.+]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[32, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[WEIGHTS2_PAD_AFTER:%.+]] = const.Declare tensor<16x16x3x3xf16> = dense<0.000000e+00> : tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[WEIGHTS2:%.+]] = IE.Concat([[WEIGHTS2_PAD_BEFORE]], [[WEIGHTS2_SLICE]], [[WEIGHTS2_PAD_AFTER]]) {
    // CHECK-SAME:                     per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x32x3x3xf16>, tensor<16x16x3x3xf16>, tensor<16x16x3x3xf16> -> tensor<16x64x3x3xf16>

    // CHECK-DAG:   [[WEIGHTS3_PAD_BEFORE:%.+]] = const.Declare tensor<16x48x3x3xf16> = dense<0.000000e+00> : tensor<16x48x3x3xf16>
    // CHECK-DAG:   [[WEIGHTS3_SLICE:%.+]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[48, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[WEIGHTS3:%.+]] = IE.Concat([[WEIGHTS3_PAD_BEFORE]], [[WEIGHTS3_SLICE]]) {
    // CHECK-SAME:                     per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x48x3x3xf16>, tensor<16x16x3x3xf16> -> tensor<16x64x3x3xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[WEIGHTS0]], [[WEIGHTS1]], [[WEIGHTS2]], [[WEIGHTS3]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:                      tensor<16x64x3x3xf16>, tensor<16x64x3x3xf16>, tensor<16x64x3x3xf16>, tensor<16x64x3x3xf16> -> tensor<64x64x3x3xf16>

    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[CONCAT]], [[FQ_LOW]], [[FQ_HIGH]], [[FQ_LOW]], [[FQ_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<64x64x3x3xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16> -> tensor<64x64x3x3xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[FQ]], [[BIAS]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x64x80x80xf16>, tensor<64x64x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    // CHECK:       return [[CONV]]
}


// -----

// CHECK-LABEL: @ConvertGroupConvToSingleConvOutChannelEqualGroup
func.func @ConvertGroupConvToSingleConvOutChannelEqualGroup(%arg0: tensor<1x64x80x80xf16>) -> tensor<1x2x80x80xf16> {
    %weights = const.Declare tensor<2x32x3x3xf16> = dense<1.0> : tensor<2x32x3x3xf16>
    %bias = const.Declare tensor<1x2x1x1xf16> = dense<1.0> : tensor<1x2x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %weights, %bias) {dilations = [1, 1], groups = 2 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<2x32x3x3xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x80x80xf16>

    return %result : tensor<1x2x80x80xf16>


    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[ORG_WEIGHTS:%.+]] = const.Declare tensor<2x32x3x3xf16> = dense<1.000000e+00> : tensor<2x32x3x3xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<1.000000e+00> : tensor<1x2x1x1xf16>
    // CHECK-DAG:   [[WEIGHTS0:%.+]] = const.Declare tensor<1x64x3x3xf16> = dense<1.000000e+00> : tensor<2x32x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [1, 32, 3, 3]>, #const.PadWithZero<[0, 0, 0, 0], [0, 32, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS1:%.+]] = const.Declare tensor<1x64x3x3xf16> = dense<1.000000e+00> : tensor<2x32x3x3xf16>, [#const.SubView<[1, 0, 0, 0], [1, 32, 3, 3]>, #const.PadWithZero<[0, 32, 0, 0], [0, 0, 0, 0]>]
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[WEIGHTS0]], [[WEIGHTS1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x64x3x3xf16>, tensor<1x64x3x3xf16> -> tensor<2x64x3x3xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[CONCAT]], [[BIAS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<2x64x3x3xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x80x80xf16>
    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @ConvertGroupConvWithBigFilterToMultiConv
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2048x16x16xf16>
func.func @ConvertGroupConvWithBigFilterToMultiConv(%arg0: tensor<1x2048x16x16xf16>) -> tensor<1x2048x16x16xf16> {
    %weights = const.Declare tensor<2048x512x3x3xf16> = dense<1.0> : tensor<2048x512x3x3xf16>
    %bias = const.Declare tensor<1x2048x1x1xf16> = dense<1.0> : tensor<1x2048x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %weights, %bias) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x2048x16x16xf16>, tensor<2048x512x3x3xf16>, tensor<1x2048x1x1xf16> -> tensor<1x2048x16x16xf16>

    return %result : tensor<1x2048x16x16xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<2048x512x3x3xf16> = dense<1.000000e+00> : tensor<2048x512x3x3xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x2048x1x1xf16> = dense<1.000000e+00> : tensor<1x2048x1x1xf16>
    // CHECK:       [[INPUT0:%.*]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 16, 16]
    // CHECK-SAME:      : tensor<1x2048x16x16xf16> to tensor<1x512x16x16xf16>
    // CHECK-DAG:   [[WEIGHTS0:%.*]] = const.Declare tensor<512x512x3x3xf16> = dense<1.000000e+00> : tensor<2048x512x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [512, 512, 3, 3]>]
    // CHECK-DAG:   [[BIAS0:%.*]] = const.Declare tensor<1x512x1x1xf16> = dense<1.000000e+00> : tensor<1x2048x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 512, 1, 1]>]
    // CHECK:       [[CONV_0:%.*]] = IE.Convolution([[INPUT0]], [[WEIGHTS0]], [[BIAS0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x16x16xf16>

    // CHECK-DAG:   [[INPUT1:%.*]] = IE.Slice [[INPUT]] [0, 512, 0, 0] [1, 512, 16, 16]
    // CHECK-SAME:      : tensor<1x2048x16x16xf16> to tensor<1x512x16x16xf16>
    // CHECK-DAG:   [[WEIGHTS1:%.*]] = const.Declare tensor<512x512x3x3xf16> = dense<1.000000e+00> : tensor<2048x512x3x3xf16>, [#const.SubView<[512, 0, 0, 0], [512, 512, 3, 3]>]
    // CHECK-DAG:   [[BIAS1:%.*]] = const.Declare tensor<1x512x1x1xf16> = dense<1.000000e+00> : tensor<1x2048x1x1xf16>, [#const.SubView<[0, 512, 0, 0], [1, 512, 1, 1]>]
    // CHECK:       [[CONV_1:%.*]] = IE.Convolution([[INPUT1]], [[WEIGHTS1]], [[BIAS1]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x16x16xf16>

    // CHECK-DAG:   [[INPUT2:%.*]] = IE.Slice [[INPUT]] [0, 1024, 0, 0] [1, 512, 16, 16]
    // CHECK-SAME:      : tensor<1x2048x16x16xf16> to tensor<1x512x16x16xf16>
    // CHECK-DAG:   [[WEIGHTS2:%.*]] = const.Declare tensor<512x512x3x3xf16> = dense<1.000000e+00> : tensor<2048x512x3x3xf16>, [#const.SubView<[1024, 0, 0, 0], [512, 512, 3, 3]>]
    // CHECK-DAG:   [[BIAS2:%.*]] = const.Declare tensor<1x512x1x1xf16> = dense<1.000000e+00> : tensor<1x2048x1x1xf16>, [#const.SubView<[0, 1024, 0, 0], [1, 512, 1, 1]>]
    // CHECK:       [[CONV_2:%.*]] = IE.Convolution([[INPUT2]], [[WEIGHTS2]], [[BIAS2]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x16x16xf16>

    // CHECK-DAG:   [[INPUT3:%.*]] = IE.Slice [[INPUT]] [0, 1536, 0, 0] [1, 512, 16, 16]
    // CHECK-SAME:      : tensor<1x2048x16x16xf16> to tensor<1x512x16x16xf16>
    // CHECK-DAG:   [[WEIGHTS3:%.*]] = const.Declare tensor<512x512x3x3xf16> = dense<1.000000e+00> : tensor<2048x512x3x3xf16>, [#const.SubView<[1536, 0, 0, 0], [512, 512, 3, 3]>]
    // CHECK-DAG:   [[BIAS3:%.*]] = const.Declare tensor<1x512x1x1xf16> = dense<1.000000e+00> : tensor<1x2048x1x1xf16>, [#const.SubView<[0, 1536, 0, 0], [1, 512, 1, 1]>]
    // CHECK:       [[CONV_3:%.*]] = IE.Convolution([[INPUT3]], [[WEIGHTS3]], [[BIAS3]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x512x16x16xf16>, tensor<512x512x3x3xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x16x16xf16>
    // CHECK:       [[RESULT:%.*]] = IE.Concat([[CONV_0]], [[CONV_1]], [[CONV_2]], [[CONV_3]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:      : tensor<1x512x16x16xf16>, tensor<1x512x16x16xf16>, tensor<1x512x16x16xf16>, tensor<1x512x16x16xf16>
    // CHECK-SAME:      -> tensor<1x2048x16x16xf16>
    // CHECK:       return [[RESULT]]
}

// -----

// CHECK-LABEL: @ConvertPerChannelGroupConvToMultiConv
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x16x16xf16>
func.func @ConvertPerChannelGroupConvToMultiConv(%arg0: tensor<1x2x16x16xf16>) -> tensor<1x32x16x16xf16> {
    %weights = const.Declare tensor<32x1x3x3xf16> = dense<1.0> : tensor<32x1x3x3xf16>
    %weights_low = const.Declare tensor<32x1x1x1xf16> = dense<0.0> : tensor<32x1x1x1xf16>
    %weights_high = const.Declare tensor<32x1x1x1xf16> = dense<1.0> : tensor<32x1x1x1xf16>
    %fq_weights = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                    levels = 255 : i64
                } : tensor<32x1x3x3xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16> -> tensor<32x1x3x3xf16>
    %input_low = const.Declare tensor<1x2x1x1xf16> = dense<[[[[0.0, 0.1]]]]> : tensor<1x1x1x2xf16>, [#const.Reshape<[1, 2, 1, 1]>]
    %input_high = const.Declare tensor<1x2x1x1xf16> = dense<[[[[1.0, 1.1]]]]> : tensor<1x1x1x2xf16>, [#const.Reshape<[1, 2, 1, 1]>]
    %fq_input = IE.FakeQuantize(%arg0, %input_low, %input_high, %input_low, %input_high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                    levels = 255 : i64
                } : tensor<1x2x16x16xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x16x16xf16>
    %result = IE.GroupConvolution(%fq_input, %fq_weights) {dilations = [1, 1], groups = 2 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x2x16x16xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x16x16xf16>

    return %result : tensor<1x32x16x16xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<32x1x3x3xf16> = dense<1.000000e+00> : tensor<32x1x3x3xf16>
    // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<32x1x1x1xf16> = dense<0.000000e+00> : tensor<32x1x1x1xf16>
    // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<32x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x1x1xf16>
    // CHECK:       [[FAKE_QUANTIZE_0:%.*]] = IE.FakeQuantize([[CST]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x1x3x3xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16> -> tensor<32x1x3x3xf16>

    // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<1x2x1x1xf16> = dense<
    // CHECK-SAME{LITERAL}:    [[[[0.000000e+00, 9.997550e-02]]]]> : tensor<1x1x1x2xf16>, [#const.Reshape<[1, 2, 1, 1]>]
    // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<1x2x1x1xf16> = dense<
    // CHECK-SAME{LITERAL}:    [[[[1.000000e+00, 1.099610e+00]]]]> : tensor<1x1x1x2xf16>, [#const.Reshape<[1, 2, 1, 1]>]
    // CHECK:       [[FAKE_QUANTIZE_1:%.*]] = IE.FakeQuantize([[INPUT]], [[CST_2]], [[CST_3]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x2x16x16xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x16x16xf16>

    // CHECK:       [[SLICE_0:%.*]] = IE.Slice [[FAKE_QUANTIZE_1]] [0, 0, 0, 0] [1, 1, 16, 16]
    // CHECK-SAME:      : tensor<1x2x16x16xf16> to tensor<1x1x16x16xf16>
    // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<16x1x3x3xf16> = dense<1.000000e+00> : tensor<32x1x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [16, 1, 3, 3]>]
    // CHECK-DAG:   [[CST_5:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<0.000000e+00> : tensor<32x1x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[CST_6:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<0.000000e+00> : tensor<32x1x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[CST_7:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[CST_8:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 1]>]

    // CHECK:       [[FAKE_QUANTIZE_2:%.*]] = IE.FakeQuantize([[CST_4]], [[CST_5]], [[CST_7]], [[CST_6]], [[CST_8]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<16x1x3x3xf16>, tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16> -> tensor<16x1x3x3xf16>
    // CHECK:       [[CONV_0:%.*]] = IE.Convolution([[SLICE_0]],  [[FAKE_QUANTIZE_2]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x1x16x16xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x16x16xf16>

    // CHECK:       [[SLICE_1:%.*]] = IE.Slice [[FAKE_QUANTIZE_1]] [0, 1, 0, 0] [1, 1, 16, 16]
    // CHECK-SAME:      : tensor<1x2x16x16xf16> to tensor<1x1x16x16xf16>
    // CHECK-DAG:   [[CST_9:%.*]] = const.Declare tensor<16x1x3x3xf16> = dense<1.000000e+00> : tensor<32x1x3x3xf16>, [#const.SubView<[16, 0, 0, 0], [16, 1, 3, 3]>]
    // CHECK-DAG:   [[CST_10:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<0.000000e+00> : tensor<32x1x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[CST_11:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<0.000000e+00> : tensor<32x1x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[CST_12:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[CST_13:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 1]>]

    // CHECK:       [[FAKE_QUANTIZE_3:%.*]] = IE.FakeQuantize([[CST_9]], [[CST_10]], [[CST_12]], [[CST_11]], [[CST_13]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<16x1x3x3xf16>, tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16> -> tensor<16x1x3x3xf16>
    // CHECK:       [[CONV_1:%.*]] = IE.Convolution([[SLICE_1]],  [[FAKE_QUANTIZE_3]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x1x16x16xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x16x16xf16>

    // CHECK:       [[RESULT:%.*]] = IE.Concat([[CONV_0]], [[CONV_1]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:      : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16> -> tensor<1x32x16x16xf16>
    // CHECK:       return [[RESULT]]
}

// -----

// CHECK: func.func @ConvertGroupConvToMultiConvWithLargeKernel([[INPUT:%.+]]: tensor<1x128x1x112xf16>, [[WEIGHTS:%.+]]: tensor<128x64x1x22xf16>)
func.func @ConvertGroupConvToMultiConvWithLargeKernel(%arg0: tensor<1x128x1x112xf16>, %arg1: tensor<128x64x1x22xf16>) -> tensor<1x128x1x91xf16> {
    %group_conv = IE.GroupConvolution(%arg0, %arg1) {
                    dilations = [1, 1],
                    groups = 2,
                    pads_begin = [0, 0],
                    pads_end = [0, 0],
                    strides = [1, 1]
                } : tensor<1x128x1x112xf16>, tensor<128x64x1x22xf16> -> tensor<1x128x1x91xf16>

    return %group_conv : tensor<1x128x1x91xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK:       [[INPUT_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 64, 1, 112] : tensor<1x128x1x112xf16> to tensor<1x64x1x112xf16>
    // CHECK:       [[WEIGHTS_0:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 0, 0] [64, 64, 1, 22] : tensor<128x64x1x22xf16> to tensor<64x64x1x22xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[INPUT_0]], [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x112xf16>, tensor<64x64x1x22xf16> -> tensor<1x64x1x91xf16>
    // CHECK:       [[INPUT_1:%.+]] = IE.Slice [[INPUT]] [0, 64, 0, 0] [1, 64, 1, 112] : tensor<1x128x1x112xf16> to tensor<1x64x1x112xf16>
    // CHECK:       [[WEIGHTS_1:%.+]] = IE.Slice [[WEIGHTS]] [64, 0, 0, 0] [64, 64, 1, 22] : tensor<128x64x1x22xf16> to tensor<64x64x1x22xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[INPUT_1]], [[WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x112xf16>, tensor<64x64x1x22xf16> -> tensor<1x64x1x91xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[CONV_0]], [[CONV_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x64x1x91xf16>, tensor<1x64x1x91xf16> -> tensor<1x128x1x91xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x128x1x91xf16>
}

// -----

// CHECK: func.func @ConvertGroupConvToMultiConvWithLargeKernelAndPad([[INPUT:%.+]]: tensor<1x128x1x100xf16>, [[WEIGHTS:%.+]]: tensor<128x64x1x64xf16>)
func.func @ConvertGroupConvToMultiConvWithLargeKernelAndPad(%arg0: tensor<1x128x1x100xf16>, %arg1: tensor<128x64x1x64xf16>) -> tensor<1x128x1x101xf16> {
    %group_conv = IE.GroupConvolution(%arg0, %arg1) {
                    dilations = [1, 1],
                    groups = 2,
                    pads_begin = [0, 32],
                    pads_end = [0, 32],
                    strides = [1, 1]
                } : tensor<1x128x1x100xf16>, tensor<128x64x1x64xf16> -> tensor<1x128x1x101xf16>

    return %group_conv : tensor<1x128x1x101xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK:       [[INPUT_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 64, 1, 100] : tensor<1x128x1x100xf16> to tensor<1x64x1x100xf16>
    // CHECK:       [[WEIGHTS_0:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 0, 0] [64, 64, 1, 64] : tensor<128x64x1x64xf16> to tensor<64x64x1x64xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[INPUT_0]], [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 32], pads_end = [0, 32], strides = [1, 1]} : tensor<1x64x1x100xf16>, tensor<64x64x1x64xf16> -> tensor<1x64x1x101xf16>
    // CHECK:       [[INPUT_1:%.+]] = IE.Slice [[INPUT]] [0, 64, 0, 0] [1, 64, 1, 100] : tensor<1x128x1x100xf16> to tensor<1x64x1x100xf16>
    // CHECK:       [[WEIGHTS_1:%.+]] = IE.Slice [[WEIGHTS]] [64, 0, 0, 0] [64, 64, 1, 64] : tensor<128x64x1x64xf16> to tensor<64x64x1x64xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[INPUT_1]], [[WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 32], pads_end = [0, 32], strides = [1, 1]} : tensor<1x64x1x100xf16>, tensor<64x64x1x64xf16> -> tensor<1x64x1x101xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[CONV_0]], [[CONV_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x64x1x101xf16>, tensor<1x64x1x101xf16> -> tensor<1x128x1x101xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x128x1x101xf16>
}

// -----

// CHECK-LABEL: @ConvertQuantizedGroupConvToSingleConvWithAffineReshape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x80x80xf16>
func.func @ConvertQuantizedGroupConvToSingleConvWithAffineReshape(%arg0: tensor<1x64x80x80xf16>) -> tensor<1x64x80x80xf16> {
    %weights = const.Declare tensor<64x16x1x3xf16> = dense<1.0> : tensor<64x16x1x3xf16>
    %weights_low = const.Declare tensor<64x1x1x1xf16> = dense<-1.270000e+02> : tensor<64x1x1x1xf16>
    %weights_high = const.Declare tensor<64x1x1x1xf16> = dense<1.270000e+02> : tensor<64x1x1x1xf16>
    %fq_weights = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                    levels = 255 : i64
                } : tensor<64x16x1x3xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16> -> tensor<64x16x1x3xf16>
    %affine_reshape = IE.AffineReshape(%fq_weights) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [64, 16, 3, 1]} : tensor<64x16x1x3xf16> -> tensor<64x16x3x1xf16>
    %result = IE.GroupConvolution(%arg0, %affine_reshape) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<64x16x3x1xf16> -> tensor<1x64x80x80xf16>

    return %result : tensor<1x64x80x80xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[FQ_LOW:%.+]] = const.Declare tensor<64x1x1x1xf16> = dense<-1.270000e+02> : tensor<64x1x1x1xf16>
    // CHECK-DAG:   [[FQ_HIGH:%.+]] = const.Declare tensor<64x1x1x1xf16> = dense<1.270000e+02> : tensor<64x1x1x1xf16>

    // CHECK-DAG:   [[WEIGHTS0_SLICE:%.+]] = const.Declare tensor<16x16x3x1xf16> = dense<1.000000e+00> : tensor<64x16x1x3xf16>
    // CHECK-SAME:      [#const.SubView<[0, 0, 0, 0], [16, 16, 1, 3]>, #const.Reshape<[16, 16, 3, 1]>]
    // CHECK-DAG:   [[WEIGHTS0_PAD_AFTER:%.+]] = const.Declare tensor<16x48x3x1xf16> = dense<0.000000e+00> : tensor<16x48x3x1xf16>
    // CHECK-DAG:   [[WEIGHTS0:%.+]] = IE.Concat([[WEIGHTS0_SLICE]], [[WEIGHTS0_PAD_AFTER]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x16x3x1xf16>, tensor<16x48x3x1xf16> -> tensor<16x64x3x1xf16>

    // CHECK-DAG:   [[WEIGHTS1_PAD_BEFORE:%.+]] = const.Declare tensor<16x16x3x1xf16> = dense<0.000000e+00> : tensor<16x16x3x1xf16>
    // CHECK-DAG:   [[WEIGHTS1_SLICE:%.+]] = const.Declare tensor<16x16x3x1xf16> = dense<1.000000e+00> : tensor<64x16x1x3xf16>
    // CHECK-SAME:      [#const.SubView<[16, 0, 0, 0], [16, 16, 1, 3]>, #const.Reshape<[16, 16, 3, 1]>]
    // CHECK-DAG:   [[WEIGHTS1_PAD_AFTER:%.+]] = const.Declare tensor<16x32x3x1xf16> = dense<0.000000e+00> : tensor<16x32x3x1xf16>
    // CHECK-DAG:   [[WEIGHTS1:%.+]] = IE.Concat([[WEIGHTS1_PAD_BEFORE]], [[WEIGHTS1_SLICE]], [[WEIGHTS1_PAD_AFTER]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x16x3x1xf16>, tensor<16x16x3x1xf16>, tensor<16x32x3x1xf16> -> tensor<16x64x3x1xf16>

    // CHECK-DAG:   [[WEIGHTS2_PAD_BEFORE:%.+]] = const.Declare tensor<16x32x3x1xf16> = dense<0.000000e+00> : tensor<16x32x3x1xf16>
    // CHECK-DAG:   [[WEIGHTS2_SLICE:%.+]] = const.Declare tensor<16x16x3x1xf16> = dense<1.000000e+00> : tensor<64x16x1x3xf16>
    // CHECK-SAME:      [#const.SubView<[32, 0, 0, 0], [16, 16, 1, 3]>, #const.Reshape<[16, 16, 3, 1]>]
    // CHECK-DAG:   [[WEIGHTS2_PAD_AFTER:%.+]] = const.Declare tensor<16x16x3x1xf16> = dense<0.000000e+00> : tensor<16x16x3x1xf16>
    // CHECK-DAG:   [[WEIGHTS2:%.+]] = IE.Concat([[WEIGHTS2_PAD_BEFORE]], [[WEIGHTS2_SLICE]], [[WEIGHTS2_PAD_AFTER]]) {
    // CHECK-SAME:      tensor<16x32x3x1xf16>, tensor<16x16x3x1xf16>, tensor<16x16x3x1xf16> -> tensor<16x64x3x1xf16>

    // CHECK-DAG:   [[WEIGHTS3_PAD_BEFORE:%.+]] = const.Declare tensor<16x48x3x1xf16> = dense<0.000000e+00> : tensor<16x48x3x1xf16>
    // CHECK-DAG:   [[WEIGHTS3_SLICE:%.+]] = const.Declare tensor<16x16x3x1xf16> = dense<1.000000e+00> : tensor<64x16x1x3xf16>
    // CHECK-SAME:      [#const.SubView<[48, 0, 0, 0], [16, 16, 1, 3]>, #const.Reshape<[16, 16, 3, 1]>]
    // CHECK-DAG:   [[WEIGHTS3:%.+]] = IE.Concat([[WEIGHTS3_PAD_BEFORE]], [[WEIGHTS3_SLICE]]) {
    // CHECK-SAME:      tensor<16x48x3x1xf16>, tensor<16x16x3x1xf16> -> tensor<16x64x3x1xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[WEIGHTS0]], [[WEIGHTS1]], [[WEIGHTS2]], [[WEIGHTS3]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:      tensor<16x64x3x1xf16>, tensor<16x64x3x1xf16>, tensor<16x64x3x1xf16>, tensor<16x64x3x1xf16> -> tensor<64x64x3x1xf16>

    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[CONCAT]], [[FQ_LOW]], [[FQ_HIGH]], [[FQ_LOW]], [[FQ_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<64x64x3x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16> -> tensor<64x64x3x1xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FQ]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x64x80x80xf16>, tensor<64x64x3x1xf16> -> tensor<1x64x80x80xf16>

    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @ConvertQuantizedGroupConvToMultiConvWithAffineReshapeWhenAxisMerged
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x96x80x80xf16>
func.func @ConvertQuantizedGroupConvToMultiConvWithAffineReshapeWhenAxisMerged(%arg0: tensor<1x96x80x80xf16>) -> tensor<1x96x80x80xf16> {
    %weights = const.Declare tensor<96x16x3x3xf16> = dense<1.0> : tensor<96x16x3x3xf16>
    %weights_low = const.Declare tensor<96x1x3x1xf16> = dense<-1.270000e+02> : tensor<96x1x3x1xf16>
    %weights_high = const.Declare tensor<96x1x3x1xf16> = dense<1.270000e+02> : tensor<96x1x3x1xf16>
    %fq_weights = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                    levels = 255 : i64
                } : tensor<96x16x3x3xf16>, tensor<96x1x3x1xf16>, tensor<96x1x3x1xf16>, tensor<96x1x3x1xf16>, tensor<96x1x3x1xf16> -> tensor<96x16x3x3xf16>
    %affine_reshape = IE.AffineReshape(%fq_weights) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [96, 48, 3, 1]} : tensor<96x16x3x3xf16> -> tensor<96x48x3x1xf16>
    %result = IE.GroupConvolution(%arg0, %affine_reshape) {dilations = [1, 1], groups = 2 : i64, pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]} : tensor<1x96x80x80xf16>, tensor<96x48x3x1xf16> -> tensor<1x96x80x80xf16>

    return %result : tensor<1x96x80x80xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<96x16x3x3xf16> = dense<1.000000e+00> : tensor<96x16x3x3xf16>
    // CHECK-DAG:   [[FQ_LOW:%.+]] = const.Declare tensor<96x1x3x1xf16> = dense<-1.270000e+02> : tensor<96x1x3x1xf16>
    // CHECK-DAG:   [[FQ_HIGH:%.+]] = const.Declare tensor<96x1x3x1xf16> = dense<1.270000e+02> : tensor<96x1x3x1xf16>

    // CHECK:       [[FQ_0:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[FQ_LOW]], [[FQ_HIGH]], [[FQ_LOW]], [[FQ_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<96x16x3x3xf16>, tensor<96x1x3x1xf16>, tensor<96x1x3x1xf16>, tensor<96x1x3x1xf16>, tensor<96x1x3x1xf16> -> tensor<96x16x3x3xf16>
    // CHECK:       [[AFFINE_RESHAPE_0:%.+]] = IE.AffineReshape([[FQ_0]])
    // CHECK{LITERAL}:      {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [96, 48, 3, 1]} : tensor<96x16x3x3xf16> -> tensor<96x48x3x1xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 48, 80, 80]
    // CHECK-SAME:      tensor<1x96x80x80xf16> to tensor<1x48x80x80xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[AFFINE_RESHAPE_0]] [0, 0, 0, 0] [48, 48, 3, 1]
    // CHECK-SAME:      tensor<96x48x3x1xf16> to tensor<48x48x3x1xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[SLICE_0]], [[SLICE_1]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x48x80x80xf16>, tensor<48x48x3x1xf16> -> tensor<1x48x80x80xf16>

    // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[INPUT]] [0, 48, 0, 0] [1, 48, 80, 80]
    // CHECK-SAME:      tensor<1x96x80x80xf16> to tensor<1x48x80x80xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice [[AFFINE_RESHAPE_0]] [48, 0, 0, 0] [48, 48, 3, 1]
    // CHECK-SAME:      tensor<96x48x3x1xf16> to tensor<48x48x3x1xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[SLICE_2]], [[SLICE_3]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x48x80x80xf16>, tensor<48x48x3x1xf16> -> tensor<1x48x80x80xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[CONV_0]], [[CONV_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x48x80x80xf16>, tensor<1x48x80x80xf16> -> tensor<1x96x80x80xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

// CHECK-LABEL: @ConvertQuantizedGroupConvToMultiConvWithAffineReshapeWhenAxisSplit
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x96x80x80xf16>
func.func @ConvertQuantizedGroupConvToMultiConvWithAffineReshapeWhenAxisSplit(%arg0: tensor<1x96x80x80xf16>) -> tensor<1x96x80x80xf16> {
    %weights = const.Declare tensor<96x96x1x1xf16> = dense<1.0> : tensor<96x96x1x1xf16>
    %weights_low = const.Declare tensor<1x96x1x1xf16> = dense<-1.270000e+02> : tensor<1x96x1x1xf16>
    %weights_high = const.Declare tensor<1x96x1x1xf16> = dense<1.270000e+02> : tensor<1x96x1x1xf16>
    %fq_weights = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                    levels = 255 : i64
                } : tensor<96x96x1x1xf16>, tensor<1x96x1x1xf16>, tensor<1x96x1x1xf16>, tensor<1x96x1x1xf16>, tensor<1x96x1x1xf16> -> tensor<96x96x1x1xf16>
    %affine_reshape = IE.AffineReshape(%fq_weights) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [96, 48, 2, 1]} : tensor<96x96x1x1xf16> -> tensor<96x48x2x1xf16>
    %result = IE.GroupConvolution(%arg0, %affine_reshape) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [1, 0], strides = [1, 1]} : tensor<1x96x80x80xf16>, tensor<96x48x2x1xf16> -> tensor<1x96x80x80xf16>

    return %result : tensor<1x96x80x80xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<96x96x1x1xf16> = dense<1.000000e+00> : tensor<96x96x1x1xf16>
    // CHECK-DAG:   [[FQ_LOW:%.+]] = const.Declare tensor<1x96x1x1xf16> = dense<-1.270000e+02> : tensor<1x96x1x1xf16>
    // CHECK-DAG:   [[FQ_HIGH:%.+]] = const.Declare tensor<1x96x1x1xf16> = dense<1.270000e+02> : tensor<1x96x1x1xf16>

    // CHECK:       [[FQ_0:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[FQ_LOW]], [[FQ_HIGH]], [[FQ_LOW]], [[FQ_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<96x96x1x1xf16>, tensor<1x96x1x1xf16>, tensor<1x96x1x1xf16>, tensor<1x96x1x1xf16>, tensor<1x96x1x1xf16> -> tensor<96x96x1x1xf16>
    // CHECK:       [[AFFINE_RESHAPE_0:%.+]] = IE.AffineReshape([[FQ_0]])
    // CHECK{LITERAL}:      {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [96, 48, 2, 1]} : tensor<96x96x1x1xf16> -> tensor<96x48x2x1xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 48, 80, 80]
    // CHECK-SAME:      tensor<1x96x80x80xf16> to tensor<1x48x80x80xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[AFFINE_RESHAPE_0]] [0, 0, 0, 0] [48, 48, 2, 1]
    // CHECK-SAME:      tensor<96x48x2x1xf16> to tensor<48x48x2x1xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[SLICE_0]], [[SLICE_1]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x48x80x80xf16>, tensor<48x48x2x1xf16> -> tensor<1x48x80x80xf16>

    // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[INPUT]] [0, 48, 0, 0] [1, 48, 80, 80]
    // CHECK-SAME:      tensor<1x96x80x80xf16> to tensor<1x48x80x80xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice [[AFFINE_RESHAPE_0]] [48, 0, 0, 0] [48, 48, 2, 1]
    // CHECK-SAME:      tensor<96x48x2x1xf16> to tensor<48x48x2x1xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[SLICE_2]], [[SLICE_3]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x48x80x80xf16>, tensor<48x48x2x1xf16> -> tensor<1x48x80x80xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[CONV_0]], [[CONV_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x48x80x80xf16>, tensor<1x48x80x80xf16> -> tensor<1x96x80x80xf16>
    // CHECK:       return [[CONCAT]]
}
