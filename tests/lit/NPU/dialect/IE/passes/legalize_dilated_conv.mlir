//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --legalize-dilated-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @LegalizeDilatedConvolution
func.func @LegalizeDilatedConvolution(%arg0: tensor<1x2x16x16xf32>) -> tensor<1x8x16x16xf32> {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x2x16x16xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x2x16x16xf32>

    %weights_low = const.Declare tensor<f32> = dense<1.0> : tensor<f32>
    %weights_high = const.Declare tensor<f32> = dense<10.0> : tensor<f32>
    %weights = const.Declare tensor<8x2x3x3xf32> = dense<5.0> : tensor<8x2x3x3xf32>
    %1 = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 } :
        tensor<8x2x3x3xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<8x2x3x3xf32>

    %2 = IE.Convolution(%0, %1)
        {
            strides = [1, 1],
            pads_begin = [2, 2],
            pads_end = [2, 2],
            dilations = [2, 2]
        } :
        tensor<1x2x16x16xf32>, tensor<8x2x3x3xf32> -> tensor<1x8x16x16xf32>

    return %2 : tensor<1x8x16x16xf32>

    // CHECK-DAG: [[MIN_IN:%.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MAX_IN:%.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK: [[FQ0:%.*]] = IE.FakeQuantize(%arg0, [[MIN_IN]], [[MAX_IN]], [[MIN_IN]], [[MAX_IN]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x16x16xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x2x16x16xf32>

    // CHECK-DAG: [[MIN_WEIGHTS:%.*]] = const.Declare tensor<f32> = dense<1.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MAX_WEIGHTS:%.*]] = const.Declare tensor<f32> = dense<1.000000e+01> : tensor<f32>
    // CHECK-DAG: [[FILTERS:%.*]] = const.Declare tensor<8x2x3x3xf32> = dense<5.000000e+00> : tensor<8x2x3x3xf32>
    // CHECK: [[FQ1:%.*]] = IE.FakeQuantize([[FILTERS]], [[MIN_WEIGHTS]], [[MAX_WEIGHTS]], [[MIN_WEIGHTS]], [[MAX_WEIGHTS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<8x2x3x3xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<8x2x3x3xf32>

    // CHECK: [[EXPAND_DILATED:%.*]] = IE.ExpandDilated([[FQ1]]) {dilations = [2, 2]} : tensor<8x2x3x3xf32> -> tensor<8x2x5x5xf32>

    // CHECK: [[CONV:%.*]] = IE.Convolution([[FQ0]], [[EXPAND_DILATED]]) {dilations = [1, 1], pads_begin = [2, 2], pads_end = [2, 2], strides = [1, 1]} : tensor<1x2x16x16xf32>, tensor<8x2x5x5xf32> -> tensor<1x8x16x16xf32>
}

// CHECK-LABEL: @LegalizeDilatedGroupConvolution
func.func @LegalizeDilatedGroupConvolution(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %filter = const.Declare tensor<3x1x3x3xf16> = dense<1.0> : tensor<3x1x3x3xf16>
    %0 = IE.GroupConvolution(%arg0, %filter)
        {
            dilations = [2, 2],
            groups = 3,
            pads_begin = [2, 2],
            pads_end = [2, 2],
            strides = [1, 1]
        } :
        tensor<1x3x30x30xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x30x30xf16>
    return %0 : tensor<1x3x30x30xf16>

    // CHECK-DAG: [[FILTERS:%.*]] = const.Declare tensor<3x1x3x3xf16>
    // CHECK: [[EXPAND_DILATED:%.*]] = IE.ExpandDilated([[FILTERS]]) {dilations = [2, 2]} : tensor<3x1x3x3xf16> -> tensor<3x1x5x5xf16>
    // CHECK: [[CONV:%.*]] = IE.GroupConvolution(%arg0, [[EXPAND_DILATED]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [2, 2], pads_end = [2, 2], strides = [1, 1]} : tensor<1x3x30x30xf16>, tensor<3x1x5x5xf16> -> tensor<1x3x30x30xf16>

    // CHECK: return [[CONV]]
}

// CHECK-LABEL: @LegalizeDilatedGroupConvolutionWithConstantWeights
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x30x30xf16>)
func.func @LegalizeDilatedGroupConvolutionWithConstantWeights(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %filter = const.Declare tensor<1x3x3x3xf16> = dense<1.0> : tensor<1x3x3x3xf16>
    %0 = IE.AffineReshape(%filter) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [3, 1, 3, 3]} : tensor<1x3x3x3xf16> -> tensor<3x1x3x3xf16>
    %1 = IE.GroupConvolution(%arg0, %0)
        {
            dilations = [2, 2],
            groups = 3,
            pads_begin = [2, 2],
            pads_end = [2, 2],
            strides = [1, 1]
        } :
        tensor<1x3x30x30xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x30x30xf16>
    return %1 : tensor<1x3x30x30xf16>

    // CHECK-DAG: [[FILTERS:%.*]] = const.Declare tensor<3x1x3x3xf16> = dense<1.000000e+00> : tensor<1x3x3x3xf16>, [#const.AffineReshape<{{\[\[}}0], [0], [1, 2], [3]], [3, 1, 3, 3]>]
    // CHECK: [[EXPAND_DILATED:%.*]] = IE.ExpandDilated([[FILTERS]]) {dilations = [2, 2]} : tensor<3x1x3x3xf16> -> tensor<3x1x5x5xf16>
    // CHECK: [[CONV:%.*]] = IE.GroupConvolution([[INPUT]], [[EXPAND_DILATED]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [2, 2], pads_end = [2, 2], strides = [1, 1]} : tensor<1x3x30x30xf16>, tensor<3x1x5x5xf16> -> tensor<1x3x30x30xf16>

    // CHECK: return [[CONV]]
}

// CHECK-LABEL: @ConvertDilatedConvolutionToConvolution1
func.func @ConvertDilatedConvolutionToConvolution1(%arg0: tensor<1x64x20x20xf16>) -> tensor<1x64x18x2xf16> {
    %FILTERS = const.Declare tensor<64x64x3x3xf16> = dense<1.000000e+00> : tensor<64x64x3x3xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 9]} : tensor<1x64x20x20xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x18x2xf16>
    return %RESULT : tensor<1x64x18x2xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<64x64x3x3xf16> = dense<1.000000e+00> : tensor<64x64x3x3xf16>
    // CHECK:       [[SLICE_0:%.*]] = IE.Slice [[CST]] [0, 0, 0, 0] [64, 64, 3, 1] : tensor<64x64x3x3xf16> to tensor<64x64x3x1xf16>
    // CHECK:       [[SLICE_1:%.*]] = IE.Slice [[CST]] [0, 0, 0, 1] [64, 64, 3, 1] : tensor<64x64x3x3xf16> to tensor<64x64x3x1xf16>
    // CHECK:       [[SLICE_2:%.*]] = IE.Slice [[CST]]  [0, 0, 0, 2] [64, 64, 3, 1] : tensor<64x64x3x3xf16> to tensor<64x64x3x1xf16>
    // CHECK:       [[SLICE_3:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 20, 2] : tensor<1x64x20x20xf16> to tensor<1x64x20x2xf16>
    // CHECK:       [[CONV_0:%.*]] = IE.Convolution([[SLICE_3]], [[SLICE_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x20x2xf16>, tensor<64x64x3x1xf16> -> tensor<1x64x18x2xf16>
    // CHECK:       [[SLICE_4:%.*]] = IE.Slice %arg0 [0, 0, 0, 9] [1, 64, 20, 2] : tensor<1x64x20x20xf16> to tensor<1x64x20x2xf16>
    // CHECK:       [[CONV_1:%.*]] = IE.Convolution([[SLICE_4]], [[SLICE_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x20x2xf16>, tensor<64x64x3x1xf16> -> tensor<1x64x18x2xf16>
    // CHECK:       [[SLICE_5:%.*]] = IE.Slice %arg0 [0, 0, 0, 18] [1, 64, 20, 2] : tensor<1x64x20x20xf16> to tensor<1x64x20x2xf16>
    // CHECK:       [[CONV_2:%.*]] = IE.Convolution([[SLICE_5]], [[SLICE_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x20x2xf16>, tensor<64x64x3x1xf16> -> tensor<1x64x18x2xf16>
    // CHECK:       [[ADD_0:%.*]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x18x2xf16>, tensor<1x64x18x2xf16> -> tensor<1x64x18x2xf16>
    // CHECK:       [[ADD_1:%.*]] = IE.Add([[ADD_0]], [[CONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x18x2xf16>, tensor<1x64x18x2xf16> -> tensor<1x64x18x2xf16>
    // CHECK:       return [[ADD_1]] : tensor<1x64x18x2xf16>
}

// CHECK-LABEL: @ConvertDilatedConvolutionToConvolution2
func.func @ConvertDilatedConvolutionToConvolution2(%arg0: tensor<1x64x20x20xf16>) -> tensor<1x64x2x18xf16> {
    %FILTERS = const.Declare tensor<64x64x3x3xf16> = dense<1.000000e+00> : tensor<64x64x3x3xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [9, 1]} : tensor<1x64x20x20xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x2x18xf16>
    return %RESULT : tensor<1x64x2x18xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<64x64x3x3xf16> = dense<1.000000e+00> : tensor<64x64x3x3xf16>
    // CHECK:       [[SLICE_0:%.*]] = IE.Slice [[CST]] [0, 0, 0, 0] [64, 64, 1, 3] : tensor<64x64x3x3xf16> to tensor<64x64x1x3xf16>
    // CHECK:       [[SLICE_1:%.*]] = IE.Slice [[CST]] [0, 0, 1, 0] [64, 64, 1, 3] : tensor<64x64x3x3xf16> to tensor<64x64x1x3xf16>
    // CHECK:       [[SLICE_2:%.*]] = IE.Slice [[CST]] [0, 0, 2, 0] [64, 64, 1, 3] : tensor<64x64x3x3xf16> to tensor<64x64x1x3xf16>
    // CHECK:       [[SLICE_3:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 2, 20] : tensor<1x64x20x20xf16> to tensor<1x64x2x20xf16>
    // CHECK:       [[CONV_0:%.*]] = IE.Convolution([[SLICE_3]], [[SLICE_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x2x20xf16>, tensor<64x64x1x3xf16> -> tensor<1x64x2x18xf16>
    // CHECK:       [[SLICE_4:%.*]] = IE.Slice %arg0 [0, 0, 9, 0] [1, 64, 2, 20] : tensor<1x64x20x20xf16> to tensor<1x64x2x20xf16>
    // CHECK:       [[CONV_1:%.*]] = IE.Convolution([[SLICE_4]], [[SLICE_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x2x20xf16>, tensor<64x64x1x3xf16> -> tensor<1x64x2x18xf16>
    // CHECK:       [[SLICE_5:%.*]] = IE.Slice %arg0 [0, 0, 18, 0] [1, 64, 2, 20] : tensor<1x64x20x20xf16> to tensor<1x64x2x20xf16>
    // CHECK:       [[CONV_2:%.*]] = IE.Convolution([[SLICE_5]], [[SLICE_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x2x20xf16>, tensor<64x64x1x3xf16> -> tensor<1x64x2x18xf16>
    // CHECK:       [[ADD_0:%.*]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2x18xf16>, tensor<1x64x2x18xf16> -> tensor<1x64x2x18xf16>
    // CHECK:       [[ADD_1:%.*]] = IE.Add([[ADD_0]], [[CONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2x18xf16>, tensor<1x64x2x18xf16> -> tensor<1x64x2x18xf16>
    // CHECK:       return [[ADD_1]] : tensor<1x64x2x18xf16>
}

// CHECK-LABEL: @ConvertXDilatedGroupConvolutionToGroupConvolution
func.func @ConvertXDilatedGroupConvolutionToGroupConvolution(%arg0: tensor<1x512x1x32xf16>) -> tensor<1x512x1x48xf16> {
    %FILTERS = const.Declare tensor<512x1x1x3xf16> = dense<1.000000e+00> : tensor<512x1x1x3xf16>
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [1, 8], groups = 512 : i64, pads_begin = [0, 16], pads_end = [0, 16], strides = [1, 1]} : tensor<1x512x1x32xf16>, tensor<512x1x1x3xf16> -> tensor<1x512x1x48xf16>
    return %RESULT : tensor<1x512x1x48xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<512x1x1x3xf16> = dense<1.000000e+00> : tensor<512x1x1x3xf16>
    // CHECK:       [[SLICE_0:%.*]] = IE.Slice [[CST]] [0, 0, 0, 0] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>
    // CHECK:       [[SLICE_1:%.*]] = IE.Slice [[CST]] [0, 0, 0, 1] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>
    // CHECK:       [[SLICE_2:%.*]] = IE.Slice [[CST]] [0, 0, 0, 2] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>
    // CHECK:       [[SLICE_3:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 1, 32] : tensor<1x512x1x32xf16> to tensor<1x512x1x32xf16>
    // CHECK:       [[CONV_0:%.*]] = IE.GroupConvolution([[SLICE_3]], [[SLICE_0]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 16], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x32xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       [[SLICE_4:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 1, 32] : tensor<1x512x1x32xf16> to tensor<1x512x1x32xf16>
    // CHECK:       [[CONV_1:%.*]] = IE.GroupConvolution([[SLICE_4]], [[SLICE_1]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 8], pads_end = [0, 8], strides = [1, 1]} : tensor<1x512x1x32xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       [[SLICE_5:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 1, 32] : tensor<1x512x1x32xf16> to tensor<1x512x1x32xf16>
    // CHECK:       [[CONV_2:%.*]] = IE.GroupConvolution([[SLICE_5]], [[SLICE_2]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 16], strides = [1, 1]} : tensor<1x512x1x32xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       [[ADD_0:%.*]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x1x48xf16>, tensor<1x512x1x48xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       [[ADD_1:%.*]] = IE.Add([[ADD_0]], [[CONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x1x48xf16>, tensor<1x512x1x48xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       return [[ADD_1]] : tensor<1x512x1x48xf16>
}

// CHECK-LABEL: @ConvertYDilatedGroupConvolutionToGroupConvolution
func.func @ConvertYDilatedGroupConvolutionToGroupConvolution(%arg0: tensor<1x512x32x1xf16>) -> tensor<1x512x48x1xf16> {
    %FILTERS = const.Declare tensor<512x1x3x1xf16> = dense<1.000000e+00> : tensor<512x1x3x1xf16>
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [8, 1], groups = 512 : i64, pads_begin = [16, 0], pads_end = [16, 0], strides = [1, 1]} : tensor<1x512x32x1xf16>, tensor<512x1x3x1xf16> -> tensor<1x512x48x1xf16>
    return %RESULT : tensor<1x512x48x1xf16>

    // CHECK-DAG:      [[CST:%.*]] = const.Declare tensor<512x1x3x1xf16> = dense<1.000000e+00> : tensor<512x1x3x1xf16>
    // CHECK:      [[SLICE_0:%.*]] = IE.Slice [[CST]] [0, 0, 0, 0] [512, 1, 1, 1] : tensor<512x1x3x1xf16> to tensor<512x1x1x1xf16>
    // CHECK:       [[SLICE_1:%.*]] = IE.Slice [[CST]] [0, 0, 1, 0] [512, 1, 1, 1] : tensor<512x1x3x1xf16> to tensor<512x1x1x1xf16>
    // CHECK:       [[SLICE_2:%.*]] = IE.Slice [[CST]] [0, 0, 2, 0] [512, 1, 1, 1] : tensor<512x1x3x1xf16> to tensor<512x1x1x1xf16>
    // CHECK:       [[SLICE_3:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 32, 1] : tensor<1x512x32x1xf16> to tensor<1x512x32x1xf16>
    // CHECK:       [[CONV_0:%.*]] = IE.GroupConvolution([[SLICE_3]], [[SLICE_0]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [16, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x32x1xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x48x1xf16>
    // CHECK:       [[SLICE_4:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 32, 1] : tensor<1x512x32x1xf16> to tensor<1x512x32x1xf16>
    // CHECK:       [[CONV_1:%.*]] = IE.GroupConvolution([[SLICE_4]], [[SLICE_1]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [8, 0], pads_end = [8, 0], strides = [1, 1]} : tensor<1x512x32x1xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x48x1xf16>
    // CHECK:       [[SLICE_5:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 32, 1] : tensor<1x512x32x1xf16> to tensor<1x512x32x1xf16>
    // CHECK:       [[CONV_2:%.*]] = IE.GroupConvolution([[SLICE_5]], [[SLICE_2]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [16, 0], strides = [1, 1]} : tensor<1x512x32x1xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x48x1xf16>
    // CHECK:       [[ADD_0:%.*]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x48x1xf16>, tensor<1x512x48x1xf16> -> tensor<1x512x48x1xf16>
    // CHECK:       [[ADD_1:%.*]] = IE.Add([[ADD_0]], [[CONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x48x1xf16>, tensor<1x512x48x1xf16> -> tensor<1x512x48x1xf16>
    // CHECK:       return [[ADD_1]] : tensor<1x512x48x1xf16>
}

// CHECK-LABEL: @ConvertXDilatedStridedGroupConvolutionToGroupConvolution
func.func @ConvertXDilatedStridedGroupConvolutionToGroupConvolution(%arg0: tensor<1x512x1x32xf16>) -> tensor<1x512x1x24xf16> {
    %FILTERS = const.Declare tensor<512x1x1x3xf16> = dense<1.000000e+00> : tensor<512x1x1x3xf16>
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [1, 8], groups = 512 : i64, pads_begin = [0, 16], pads_end = [0, 16], strides = [1, 2]} : tensor<1x512x1x32xf16>, tensor<512x1x1x3xf16> -> tensor<1x512x1x24xf16>
    return %RESULT : tensor<1x512x1x24xf16>

    // CHECK-DAG:      [[CST:%.*]] = const.Declare tensor<512x1x1x3xf16> = dense<1.000000e+00> : tensor<512x1x1x3xf16>
    // CHECK:      [[SLICE_0:%.*]] = IE.Slice [[CST]] [0, 0, 0, 0] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>
    // CHECK:      [[SLICE_1:%.*]] = IE.Slice [[CST]] [0, 0, 0, 1] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>
    // CHECK:      [[SLICE_2:%.*]] = IE.Slice [[CST]] [0, 0, 0, 2] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>
    // CHECK:      [[SLICE_3:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 1, 32] : tensor<1x512x1x32xf16> to tensor<1x512x1x32xf16>
    // CHECK:      [[CONV_0:%.*]] = IE.GroupConvolution([[SLICE_3]], [[SLICE_0]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 16], pads_end = [0, 0], strides = [1, 2]} : tensor<1x512x1x32xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x24xf16>
    // CHECK:      [[SLICE_4:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 1, 32] : tensor<1x512x1x32xf16> to tensor<1x512x1x32xf16>
    // CHECK:      [[CONV_1:%.*]] = IE.GroupConvolution([[SLICE_4]], [[SLICE_1]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 8], pads_end = [0, 8], strides = [1, 2]} : tensor<1x512x1x32xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x24xf16>
    // CHECK:      [[SLICE_5:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 1, 32] : tensor<1x512x1x32xf16> to tensor<1x512x1x32xf16>
    // CHECK:      [[CONV_2:%.*]] = IE.GroupConvolution([[SLICE_5]], [[SLICE_2]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 16], strides = [1, 2]} : tensor<1x512x1x32xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x24xf16>
    // CHECK:      [[ADD_0:%.*]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x1x24xf16>, tensor<1x512x1x24xf16> -> tensor<1x512x1x24xf16>
    // CHECK:      [[ADD_1:%.*]] = IE.Add([[ADD_0]], [[CONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x1x24xf16>, tensor<1x512x1x24xf16> -> tensor<1x512x1x24xf16>
    // CHECK:      return [[ADD_1]] : tensor<1x512x1x24xf16>
}

// CHECK-LABEL: @ConvertYDilatedStridedGroupConvolutionToGroupConvolution
func.func @ConvertYDilatedStridedGroupConvolutionToGroupConvolution(%arg0: tensor<1x512x32x1xf16>) -> tensor<1x512x24x1xf16> {
    %FILTERS = const.Declare tensor<512x1x3x1xf16> = dense<1.000000e+00> : tensor<512x1x3x1xf16>
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [8, 1], groups = 512 : i64, pads_begin = [16, 0], pads_end = [16, 0], strides = [2, 1]} : tensor<1x512x32x1xf16>, tensor<512x1x3x1xf16> -> tensor<1x512x24x1xf16>
    return %RESULT : tensor<1x512x24x1xf16>

    // CHECK-DAG:      [[CST:%.*]] = const.Declare tensor<512x1x3x1xf16> = dense<1.000000e+00> : tensor<512x1x3x1xf16>
    // CHECK:      [[SLICE_0:%.*]] = IE.Slice [[CST]] [0, 0, 0, 0] [512, 1, 1, 1] : tensor<512x1x3x1xf16> to tensor<512x1x1x1xf16>
    // CHECK:      [[SLICE_1:%.*]] = IE.Slice [[CST]] [0, 0, 1, 0] [512, 1, 1, 1] : tensor<512x1x3x1xf16> to tensor<512x1x1x1xf16>
    // CHECK:      [[SLICE_2:%.*]] = IE.Slice [[CST]] [0, 0, 2, 0] [512, 1, 1, 1] : tensor<512x1x3x1xf16> to tensor<512x1x1x1xf16>
    // CHECK:      [[SLICE_3:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 32, 1] : tensor<1x512x32x1xf16> to tensor<1x512x32x1xf16>
    // CHECK:      [[CONV_0:%.*]] = IE.GroupConvolution([[SLICE_3]], [[SLICE_0]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [16, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x512x32x1xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x24x1xf16>
    // CHECK:      [[SLICE_4:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 32, 1] : tensor<1x512x32x1xf16> to tensor<1x512x32x1xf16>
    // CHECK:      [[CONV_1:%.*]] = IE.GroupConvolution([[SLICE_4]], [[SLICE_1]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [8, 0], pads_end = [8, 0], strides = [2, 1]} : tensor<1x512x32x1xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x24x1xf16>
    // CHECK:      [[SLICE_5:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 32, 1] : tensor<1x512x32x1xf16> to tensor<1x512x32x1xf16>
    // CHECK:      [[CONV_2:%.*]] = IE.GroupConvolution([[SLICE_5]], [[SLICE_2]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [16, 0], strides = [2, 1]} : tensor<1x512x32x1xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x24x1xf16>
    // CHECK:      [[ADD_0:%.*]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x24x1xf16>, tensor<1x512x24x1xf16> -> tensor<1x512x24x1xf16>
    // CHECK:      [[ADD_1:%.*]] = IE.Add([[ADD_0]], [[CONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x24x1xf16>, tensor<1x512x24x1xf16> -> tensor<1x512x24x1xf16>
    // CHECK:      return [[ADD_1]] : tensor<1x512x24x1xf16>
}

// CHECK-LABEL: @LegalizeDilatedConvolutionTwoDimension
func.func @LegalizeDilatedConvolutionTwoDimension(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x32x18x24xf16> {
    %filter = const.Declare tensor<32x3x3x3xf16> = dense<1.0> : tensor<32x3x3x3xf16>
    %bias = const.Declare tensor<1x32x1x1xf16> = dense<1.0> : tensor<1x32x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias)
         {dilations = [8, 6], pads_begin = [1, 2], pads_end = [3, 4], strides = [1, 1], post_op = #IE.Relu<>} :
         tensor<1x3x30x30xf16>, tensor<32x3x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x18x24xf16>
    return %0 : tensor<1x32x18x24xf16>

    // CHECK-DAG: [[FILTERS:%.*]] = const.Declare tensor<32x3x3x3xf16> = dense<1.000000e+00> : tensor<32x3x3x3xf16>
    // CHECK-DAG: [[BIAS:%.*]] = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>

    // CHECK: [[FILTERS_SLICE0:%.*]]  = IE.Slice [[FILTERS]] [0, 0, 0, 0] [32, 3, 1, 1] : tensor<32x3x3x3xf16> to tensor<32x3x1x1xf16>
    // CHECK: [[FILTERS_SLICE1:%.*]] = IE.Slice [[FILTERS]] [0, 0, 1, 0] [32, 3, 1, 1] : tensor<32x3x3x3xf16> to tensor<32x3x1x1xf16>
    // CHECK: [[FILTERS_SLICE2:%.*]] = IE.Slice [[FILTERS]] [0, 0, 2, 0] [32, 3, 1, 1] : tensor<32x3x3x3xf16> to tensor<32x3x1x1xf16>
    // CHECK: [[FILTERS_SLICE3:%.*]] = IE.Slice [[FILTERS]] [0, 0, 0, 1] [32, 3, 1, 1] : tensor<32x3x3x3xf16> to tensor<32x3x1x1xf16>
    // CHECK: [[FILTERS_SLICE4:%.*]] = IE.Slice [[FILTERS]] [0, 0, 1, 1] [32, 3, 1, 1] : tensor<32x3x3x3xf16> to tensor<32x3x1x1xf16>
    // CHECK: [[FILTERS_SLICE5:%.*]] = IE.Slice [[FILTERS]] [0, 0, 2, 1] [32, 3, 1, 1] : tensor<32x3x3x3xf16> to tensor<32x3x1x1xf16>
    // CHECK: [[FILTERS_SLICE6:%.*]] = IE.Slice [[FILTERS]] [0, 0, 0, 2] [32, 3, 1, 1] : tensor<32x3x3x3xf16> to tensor<32x3x1x1xf16>
    // CHECK: [[FILTERS_SLICE7:%.*]] = IE.Slice [[FILTERS]] [0, 0, 1, 2] [32, 3, 1, 1] : tensor<32x3x3x3xf16> to tensor<32x3x1x1xf16>
    // CHECK: [[FILTERS_SLICE8:%.*]] = IE.Slice [[FILTERS]] [0, 0, 2, 2] [32, 3, 1, 1] : tensor<32x3x3x3xf16> to tensor<32x3x1x1xf16>

    // CHECK-DAG: [[BIAS_1:%.*]] = const.Declare tensor<1x32x1x1xf16> = dense<0.000000e+00> : tensor<1x32x1x1xf16>
    // CHECK: [[ACT_SLICE0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 17, 22] : tensor<1x3x30x30xf16> to tensor<1x3x17x22xf16>
    // CHECK: [[CONV0:%.*]] = IE.Convolution([[ACT_SLICE0]], [[FILTERS_SLICE0]], [[BIAS]]) {dilations = [1, 1], pads_begin = [1, 2], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x17x22xf16>, tensor<32x3x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ACT_SLICE1:%.*]] = IE.Slice %arg0 [0, 0, 7, 0] [1, 3, 18, 22] : tensor<1x3x30x30xf16> to tensor<1x3x18x22xf16>
    // CHECK: [[CONV1:%.*]] = IE.Convolution([[ACT_SLICE1]], [[FILTERS_SLICE1]], [[BIAS_1]]) {dilations = [1, 1], pads_begin = [0, 2], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x18x22xf16>, tensor<32x3x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ACT_SLICE2:%.*]]  = IE.Slice %arg0 [0, 0, 15, 0] [1, 3, 15, 22] : tensor<1x3x30x30xf16> to tensor<1x3x15x22xf16>
    // CHECK: [[CONV2:%.*]] = IE.Convolution([[ACT_SLICE2]], [[FILTERS_SLICE2]], [[BIAS_1]]) {dilations = [1, 1], pads_begin = [0, 2], pads_end = [3, 0], strides = [1, 1]} : tensor<1x3x15x22xf16>, tensor<32x3x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ACT_SLICE3:%.*]]  = IE.Slice %arg0 [0, 0, 0, 4] [1, 3, 17, 24] : tensor<1x3x30x30xf16> to tensor<1x3x17x24xf16>
    // CHECK: [[CONV3:%.*]] = IE.Convolution([[ACT_SLICE3]], [[FILTERS_SLICE3]], [[BIAS_1]]) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x17x24xf16>, tensor<32x3x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ACT_SLICE4:%.*]]  = IE.Slice %arg0 [0, 0, 7, 4] [1, 3, 18, 24] : tensor<1x3x30x30xf16> to tensor<1x3x18x24xf16>
    // CHECK: [[CONV4:%.*]] = IE.Convolution([[ACT_SLICE4]], [[FILTERS_SLICE4]], [[BIAS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x18x24xf16>, tensor<32x3x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ACT_SLICE5:%.*]]  = IE.Slice %arg0 [0, 0, 15, 4] [1, 3, 15, 24] : tensor<1x3x30x30xf16> to tensor<1x3x15x24xf16>
    // CHECK: [[CONV5:%.*]] = IE.Convolution([[ACT_SLICE5]], [[FILTERS_SLICE5]], [[BIAS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [3, 0], strides = [1, 1]} : tensor<1x3x15x24xf16>, tensor<32x3x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ACT_SLICE6:%.*]]  = IE.Slice %arg0 [0, 0, 0, 10] [1, 3, 17, 20] : tensor<1x3x30x30xf16> to tensor<1x3x17x20xf16>
    // CHECK: [[CONV6:%.*]] = IE.Convolution([[ACT_SLICE6]], [[FILTERS_SLICE6]], [[BIAS_1]]) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 4], strides = [1, 1]} : tensor<1x3x17x20xf16>, tensor<32x3x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ACT_SLICE7:%.*]]  = IE.Slice %arg0 [0, 0, 7, 10] [1, 3, 18, 20] : tensor<1x3x30x30xf16> to tensor<1x3x18x20xf16>
    // CHECK: [[CONV7:%.*]] = IE.Convolution([[ACT_SLICE7]], [[FILTERS_SLICE7]], [[BIAS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 4], strides = [1, 1]} : tensor<1x3x18x20xf16>, tensor<32x3x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ACT_SLICE8:%.*]]  = IE.Slice %arg0 [0, 0, 15, 10] [1, 3, 15, 20] : tensor<1x3x30x30xf16> to tensor<1x3x15x20xf16>
    // CHECK: [[CONV8:%.*]] = IE.Convolution([[ACT_SLICE8]], [[FILTERS_SLICE8]], [[BIAS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [3, 4], strides = [1, 1]} : tensor<1x3x15x20xf16>, tensor<32x3x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x18x24xf16>

    // CHECK: [[ADD0:%.*]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x18x24xf16>, tensor<1x32x18x24xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ADD1:%.*]] = IE.Add([[ADD0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x18x24xf16>, tensor<1x32x18x24xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ADD2:%.*]] = IE.Add([[ADD1]], [[CONV3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x18x24xf16>, tensor<1x32x18x24xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ADD3:%.*]] = IE.Add([[ADD2]], [[CONV4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x18x24xf16>, tensor<1x32x18x24xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ADD4:%.*]] = IE.Add([[ADD3]], [[CONV5]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x18x24xf16>, tensor<1x32x18x24xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ADD5:%.*]] = IE.Add([[ADD4]], [[CONV6]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x18x24xf16>, tensor<1x32x18x24xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ADD6:%.*]] = IE.Add([[ADD5]], [[CONV7]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x18x24xf16>, tensor<1x32x18x24xf16> -> tensor<1x32x18x24xf16>
    // CHECK: [[ADD7:%.*]] = IE.Add([[ADD6]], [[CONV8]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, post_op = #IE.Relu<>} : tensor<1x32x18x24xf16>, tensor<1x32x18x24xf16> -> tensor<1x32x18x24xf16>

    // CHECK: return [[ADD7]]
}

// CHECK-LABEL: @LegalizeDilatedConvolution1
func.func @LegalizeDilatedConvolution1(%arg0: tensor<1x3x64x64xf16>) -> tensor<1x8x48x48xf16> {
    %filter = const.Declare tensor<8x3x3x3xf16>  = dense<1.0> : tensor<8x3x3x3xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } :
    tensor<1x3x64x64xf16>, tensor<8x3x3x3xf16> -> tensor<1x8x48x48xf16>
    return %0 : tensor<1x8x48x48xf16>

    // CHECK-DAG: [[FILTERS:%.*]] = const.Declare tensor<8x3x3x3xf16> = dense<1.000000e+00> : tensor<8x3x3x3xf16>
    // CHECK: [[SLICE_0:%.*]] = IE.Slice [[FILTERS]] [0, 0, 0, 0] [8, 3, 1, 1] : tensor<8x3x3x3xf16> to tensor<8x3x1x1xf16>
    // CHECK: [[SLICE_1:%.*]] = IE.Slice [[FILTERS]] [0, 0, 1, 0] [8, 3, 1, 1] : tensor<8x3x3x3xf16> to tensor<8x3x1x1xf16>
    // CHECK: [[SLICE_2:%.*]] = IE.Slice [[FILTERS]] [0, 0, 2, 0] [8, 3, 1, 1] : tensor<8x3x3x3xf16> to tensor<8x3x1x1xf16>
    // CHECK: [[SLICE_3:%.*]] = IE.Slice [[FILTERS]] [0, 0, 0, 1] [8, 3, 1, 1] : tensor<8x3x3x3xf16> to tensor<8x3x1x1xf16>
    // CHECK: [[SLICE_4:%.*]] = IE.Slice [[FILTERS]] [0, 0, 1, 1] [8, 3, 1, 1] : tensor<8x3x3x3xf16> to tensor<8x3x1x1xf16>
    // CHECK: [[SLICE_5:%.*]] = IE.Slice [[FILTERS]] [0, 0, 2, 1] [8, 3, 1, 1] : tensor<8x3x3x3xf16> to tensor<8x3x1x1xf16>
    // CHECK: [[SLICE_6:%.*]] = IE.Slice [[FILTERS]] [0, 0, 0, 2] [8, 3, 1, 1] : tensor<8x3x3x3xf16> to tensor<8x3x1x1xf16>
    // CHECK: [[SLICE_7:%.*]] = IE.Slice [[FILTERS]] [0, 0, 1, 2] [8, 3, 1, 1] : tensor<8x3x3x3xf16> to tensor<8x3x1x1xf16>
    // CHECK: [[SLICE_8:%.*]] = IE.Slice [[FILTERS]] [0, 0, 2, 2] [8, 3, 1, 1] : tensor<8x3x3x3xf16> to tensor<8x3x1x1xf16>
    // CHECK: [[SLICE_9:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 48, 48] : tensor<1x3x64x64xf16> to tensor<1x3x48x48xf16>
    // CHECK: [[CONV0:%.*]] = IE.Convolution([[SLICE_9]], [[SLICE_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x48x48xf16>, tensor<8x3x1x1xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[SLICE_10:%.*]] = IE.Slice %arg0 [0, 0, 8, 0] [1, 3, 48, 48] : tensor<1x3x64x64xf16> to tensor<1x3x48x48xf16>
    // CHECK: [[CONV1:%.*]] = IE.Convolution([[SLICE_10]], [[SLICE_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x48x48xf16>, tensor<8x3x1x1xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[SLICE_11:%.*]] = IE.Slice %arg0 [0, 0, 16, 0] [1, 3, 48, 48] : tensor<1x3x64x64xf16> to tensor<1x3x48x48xf16>
    // CHECK: [[CONV2:%.*]] = IE.Convolution([[SLICE_11]], [[SLICE_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x48x48xf16>, tensor<8x3x1x1xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[SLICE_12:%.*]] = IE.Slice %arg0 [0, 0, 0, 8] [1, 3, 48, 48] : tensor<1x3x64x64xf16> to tensor<1x3x48x48xf16>
    // CHECK: [[CONV3:%.*]] = IE.Convolution([[SLICE_12]], [[SLICE_3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x48x48xf16>, tensor<8x3x1x1xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[SLICE_13:%.*]] = IE.Slice %arg0 [0, 0, 8, 8] [1, 3, 48, 48] : tensor<1x3x64x64xf16> to tensor<1x3x48x48xf16>
    // CHECK: [[CONV4:%.*]] = IE.Convolution([[SLICE_13]], [[SLICE_4]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x48x48xf16>, tensor<8x3x1x1xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[SLICE_14:%.*]] = IE.Slice %arg0 [0, 0, 16, 8] [1, 3, 48, 48] : tensor<1x3x64x64xf16> to tensor<1x3x48x48xf16>
    // CHECK: [[CONV5:%.*]] = IE.Convolution([[SLICE_14]], [[SLICE_5]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x48x48xf16>, tensor<8x3x1x1xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[SLICE_15:%.*]] = IE.Slice %arg0 [0, 0, 0, 16] [1, 3, 48, 48] : tensor<1x3x64x64xf16> to tensor<1x3x48x48xf16>
    // CHECK: [[CONV6:%.*]] = IE.Convolution([[SLICE_15]], [[SLICE_6]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x48x48xf16>, tensor<8x3x1x1xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[SLICE_16:%.*]] = IE.Slice %arg0 [0, 0, 8, 16] [1, 3, 48, 48] : tensor<1x3x64x64xf16> to tensor<1x3x48x48xf16>
    // CHECK: [[CONV7:%.*]] = IE.Convolution([[SLICE_16]], [[SLICE_7]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x48x48xf16>, tensor<8x3x1x1xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[SLICE_17:%.*]] = IE.Slice %arg0 [0, 0, 16, 16] [1, 3, 48, 48] : tensor<1x3x64x64xf16> to tensor<1x3x48x48xf16>
    // CHECK: [[CONV8:%.*]] = IE.Convolution([[SLICE_17]], [[SLICE_8]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x48x48xf16>, tensor<8x3x1x1xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[ADD0:%.*]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x48x48xf16>, tensor<1x8x48x48xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[ADD1:%.*]] = IE.Add([[ADD0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x48x48xf16>, tensor<1x8x48x48xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[ADD2:%.*]] = IE.Add([[ADD1]], [[CONV3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x48x48xf16>, tensor<1x8x48x48xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[ADD3:%.*]] = IE.Add([[ADD2]], [[CONV4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x48x48xf16>, tensor<1x8x48x48xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[ADD4:%.*]] = IE.Add([[ADD3]], [[CONV5]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x48x48xf16>, tensor<1x8x48x48xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[ADD5:%.*]] = IE.Add([[ADD4]], [[CONV6]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x48x48xf16>, tensor<1x8x48x48xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[ADD6:%.*]] = IE.Add([[ADD5]], [[CONV7]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x48x48xf16>, tensor<1x8x48x48xf16> -> tensor<1x8x48x48xf16>
    // CHECK: [[ADD7:%.*]] = IE.Add([[ADD6]], [[CONV8]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x48x48xf16>, tensor<1x8x48x48xf16> -> tensor<1x8x48x48xf16>
    // CHECK: return [[ADD7]]
  }

// -----

// CHECK-LABEL: @LegalizeDilatedConvolutionWeightsAsInputs
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x3x30x30xf16>, [[WEIGHTS:%.+]]: tensor<3x3x3x3xf16>)
func.func @LegalizeDilatedConvolutionWeightsAsInputs(%input: tensor<1x3x30x30xf16>, %weights: tensor<3x3x3x3xf16>) -> tensor<1x3x30x30xf16> {
    %conv = IE.Convolution(%input, %weights) {
            dilations = [2, 2],
            pads_begin = [2, 2],
            pads_end = [2, 2],
            strides = [1, 1]
        } : tensor<1x3x30x30xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x30x30xf16>
    return %conv : tensor<1x3x30x30xf16>

    // CHECK:  [[WEIGHTS_SLICE0:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 3, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE1:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 1, 0] [3, 3, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE2:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 2, 0] [3, 3, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE3:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 0, 1] [3, 3, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE4:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 1, 1] [3, 3, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE5:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 2, 1] [3, 3, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE6:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 0, 2] [3, 3, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE7:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 1, 2] [3, 3, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE8:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 2, 2] [3, 3, 1, 1]

    // CHECK:  [[INPUT_SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 28, 28]
    // CHECK:  [[CONV0:%.+]] = IE.Convolution([[INPUT_SLICE0]], [[WEIGHTS_SLICE0]]) {dilations = [1, 1], pads_begin = [2, 2], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 30, 28]
    // CHECK:  [[CONV1:%.+]] = IE.Convolution([[INPUT_SLICE1]], [[WEIGHTS_SLICE1]]) {dilations = [1, 1], pads_begin = [0, 2], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE2:%.+]] = IE.Slice [[INPUT]] [0, 0, 2, 0] [1, 3, 28, 28]
    // CHECK:  [[CONV2:%.+]] = IE.Convolution([[INPUT_SLICE2]], [[WEIGHTS_SLICE2]]) {dilations = [1, 1], pads_begin = [0, 2], pads_end = [2, 0], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE3:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 28, 30]
    // CHECK:  [[CONV3:%.+]] = IE.Convolution([[INPUT_SLICE3]], [[WEIGHTS_SLICE3]]) {dilations = [1, 1], pads_begin = [2, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE4:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 30, 30]
    // CHECK:  [[CONV4:%.+]] = IE.Convolution([[INPUT_SLICE4]], [[WEIGHTS_SLICE4]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE5:%.+]] = IE.Slice [[INPUT]] [0, 0, 2, 0] [1, 3, 28, 30]
    // CHECK:  [[CONV5:%.+]] = IE.Convolution([[INPUT_SLICE5]], [[WEIGHTS_SLICE5]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [2, 0], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE6:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 2] [1, 3, 28, 28]
    // CHECK:  [[CONV6:%.+]] = IE.Convolution([[INPUT_SLICE6]], [[WEIGHTS_SLICE6]]) {dilations = [1, 1], pads_begin = [2, 0], pads_end = [0, 2], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE7:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 2] [1, 3, 30, 28]
    // CHECK:  [[CONV7:%.+]] = IE.Convolution([[INPUT_SLICE7]], [[WEIGHTS_SLICE7]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 2], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE8:%.+]] = IE.Slice [[INPUT]] [0, 0, 2, 2] [1, 3, 28, 28]
    // CHECK:  [[CONV8:%.+]] = IE.Convolution([[INPUT_SLICE8]], [[WEIGHTS_SLICE8]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [2, 2], strides = [1, 1]}

    // CHECK:  [[ADD0:%.+]] = IE.Add([[CONV0]], [[CONV1]])
    // CHECK:  [[ADD1:%.+]] = IE.Add([[ADD0]], [[CONV2]])
    // CHECK:  [[ADD2:%.+]] = IE.Add([[ADD1]], [[CONV3]])
    // CHECK:  [[ADD3:%.+]] = IE.Add([[ADD2]], [[CONV4]])
    // CHECK:  [[ADD4:%.+]] = IE.Add([[ADD3]], [[CONV5]])
    // CHECK:  [[ADD5:%.+]] = IE.Add([[ADD4]], [[CONV6]])
    // CHECK:  [[ADD6:%.+]] = IE.Add([[ADD5]], [[CONV7]])
    // CHECK:  [[ADD7:%.+]] = IE.Add([[ADD6]], [[CONV8]])

    // CHECK:  return [[ADD7]]
}

// -----

// CHECK-LABEL: @LegalizeDilatedGroupConvolutionWeightsAsInputs
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x3x30x30xf16>, [[WEIGHTS:%.+]]: tensor<3x1x3x3xf16>)
func.func @LegalizeDilatedGroupConvolutionWeightsAsInputs(%input: tensor<1x3x30x30xf16>, %weights: tensor<3x1x3x3xf16>) -> tensor<1x3x30x30xf16> {
    %conv = IE.GroupConvolution(%input, %weights) {
            dilations = [2, 2],
            groups = 3,
            pads_begin = [2, 2],
            pads_end = [2, 2],
            strides = [1, 1]
        } : tensor<1x3x30x30xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x30x30xf16>
    return %conv : tensor<1x3x30x30xf16>


    // CHECK:  [[WEIGHTS_SLICE0:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 1, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE1:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 1, 0] [3, 1, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE2:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 2, 0] [3, 1, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE3:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 0, 1] [3, 1, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE4:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 1, 1] [3, 1, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE5:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 2, 1] [3, 1, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE6:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 0, 2] [3, 1, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE7:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 1, 2] [3, 1, 1, 1]
    // CHECK:  [[WEIGHTS_SLICE8:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 2, 2] [3, 1, 1, 1]

    // CHECK:  [[INPUT_SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 28, 28]
    // CHECK:  [[CONV0:%.+]] = IE.GroupConvolution([[INPUT_SLICE0]], [[WEIGHTS_SLICE0]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [2, 2], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 30, 28]
    // CHECK:  [[CONV1:%.+]] = IE.GroupConvolution([[INPUT_SLICE1]], [[WEIGHTS_SLICE1]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 2], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE2:%.+]] = IE.Slice [[INPUT]] [0, 0, 2, 0] [1, 3, 28, 28]
    // CHECK:  [[CONV2:%.+]] = IE.GroupConvolution([[INPUT_SLICE2]], [[WEIGHTS_SLICE2]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 2], pads_end = [2, 0], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE3:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 28, 30]
    // CHECK:  [[CONV3:%.+]] = IE.GroupConvolution([[INPUT_SLICE3]], [[WEIGHTS_SLICE3]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [2, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE4:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 30, 30]
    // CHECK:  [[CONV4:%.+]] = IE.GroupConvolution([[INPUT_SLICE4]], [[WEIGHTS_SLICE4]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE5:%.+]] = IE.Slice [[INPUT]] [0, 0, 2, 0] [1, 3, 28, 30]
    // CHECK:  [[CONV5:%.+]] = IE.GroupConvolution([[INPUT_SLICE5]], [[WEIGHTS_SLICE5]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [2, 0], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE6:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 2] [1, 3, 28, 28]
    // CHECK:  [[CONV6:%.+]] = IE.GroupConvolution([[INPUT_SLICE6]], [[WEIGHTS_SLICE6]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [2, 0], pads_end = [0, 2], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE7:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 2] [1, 3, 30, 28]
    // CHECK:  [[CONV7:%.+]] = IE.GroupConvolution([[INPUT_SLICE7]], [[WEIGHTS_SLICE7]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 2], strides = [1, 1]}
    // CHECK:  [[INPUT_SLICE8:%.+]] = IE.Slice [[INPUT]] [0, 0, 2, 2] [1, 3, 28, 28]
    // CHECK:  [[CONV8:%.+]] = IE.GroupConvolution([[INPUT_SLICE8]], [[WEIGHTS_SLICE8]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [2, 2], strides = [1, 1]}

    // CHECK:  [[ADD0:%.+]] = IE.Add([[CONV0]], [[CONV1]])
    // CHECK:  [[ADD1:%.+]] = IE.Add([[ADD0]], [[CONV2]])
    // CHECK:  [[ADD2:%.+]] = IE.Add([[ADD1]], [[CONV3]])
    // CHECK:  [[ADD3:%.+]] = IE.Add([[ADD2]], [[CONV4]])
    // CHECK:  [[ADD4:%.+]] = IE.Add([[ADD3]], [[CONV5]])
    // CHECK:  [[ADD5:%.+]] = IE.Add([[ADD4]], [[CONV6]])
    // CHECK:  [[ADD6:%.+]] = IE.Add([[ADD5]], [[CONV7]])
    // CHECK:  [[ADD7:%.+]] = IE.Add([[ADD6]], [[CONV8]])

    // CHECK:  return [[ADD7]]
}
