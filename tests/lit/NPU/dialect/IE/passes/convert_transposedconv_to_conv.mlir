//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-transposed-conv-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @ConvertTransposedConv2DToConv2D
func.func @ConvertTransposedConv2DToConv2D(%input: tensor<1x32x23x30xf16>) -> tensor<1x16x46x60xf16> {
    %weights = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<16x32x2x2xf16>
    %output = IE.TransposedConvolution(%input, %weights) {strides = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0]} : tensor<1x32x23x30xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>
    return %output : tensor<1x16x46x60xf16>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<16x32x2x2xf16>
    // CHECK-NOT:   IE.TransposedConvolution
    // CHECK:       [[UPS:%.*]] = IE.Upsampling
    // CHECK-SAME:      #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [1, 1], pads_width = [1, 1]>
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x32x23x30xf16> -> tensor<1x32x47x61xf16>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[UPS]], [[WEIGHTS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x32x47x61xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>
    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @ConvertTransposedConv2DToConv2DFQFilterFQParamsReused
func.func @ConvertTransposedConv2DToConv2DFQFilterFQParamsReused(%input: tensor<1x32x23x30xf16>) -> tensor<1x16x46x60xf16> {
    %weights = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<16x32x2x2xf16>
    %weights_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %weights_high = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    %weights_fq = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<16x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x32x2x2xf16>

    %output = IE.TransposedConvolution(%input, %weights_fq) {strides = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0]} : tensor<1x32x23x30xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>
    return %output : tensor<1x16x46x60xf16>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<16x32x2x2xf16>
    // CHECK-DAG:   [[WEIGHTS_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00>
    // CHECK-DAG:   [[WEIGHTS_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00>
    // CHECK:       [[WEIGHTS_FQ:%.*]] = IE.FakeQuantize([[WEIGHTS]]
    // CHECK-SAME:      tensor<16x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x32x2x2xf16>

    // CHECK-NOT:   IE.TransposedConvolution

    // CHECK:       [[UPSAMPLING:%.*]] = IE.Upsampling
    // CHECK-SAME:      #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [1, 1], pads_width = [1, 1]>
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x32x23x30xf16> -> tensor<1x32x47x61xf16>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[UPSAMPLING]], [[WEIGHTS_FQ]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x32x47x61xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>
    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @ConvertTransposedConv2DToConv2DFQFilterFQParamsUnique
func.func @ConvertTransposedConv2DToConv2DFQFilterFQParamsUnique(%input: tensor<1x32x23x30xf16>) -> tensor<1x16x46x60xf16> {
    %weights = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<16x32x2x2xf16>
    %weights_input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %weights_input_high = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %weights_output_low = const.Declare tensor<1x1x1x1xf16> = dense<10.000000e+00> : tensor<1x1x1x1xf16>
    %weights_output_high = const.Declare tensor<1x1x1x1xf16> = dense<255.000000e+00> : tensor<1x1x1x1xf16>

    %weights_fq = IE.FakeQuantize(%weights, %weights_input_low, %weights_input_high, %weights_output_low, %weights_output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<16x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x32x2x2xf16>

    %output = IE.TransposedConvolution(%input, %weights_fq) {strides = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0]} : tensor<1x32x23x30xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>
    return %output : tensor<1x16x46x60xf16>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<16x32x2x2xf16>
    // CHECK-DAG:   [[WEIGHTS_INPUT_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00>
    // CHECK-DAG:   [[WEIGHTS_INPUT_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-DAG:   [[WEIGHTS_OUTPUT_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01>
    // CHECK-DAG:   [[WEIGHTS_OUTPUT_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02>
    // CHECK:       [[WEIGHTS_FQ:%.*]] = IE.FakeQuantize([[WEIGHTS]]
    // CHECK-SAME:      tensor<16x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x32x2x2xf16>

    // CHECK-NOT:   IE.TransposedConvolution

    // CHECK:       [[UPSAMPLING:%.*]] = IE.Upsampling
    // CHECK-SAME:      #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [1, 1], pads_width = [1, 1]>
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x32x23x30xf16> -> tensor<1x32x47x61xf16>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[UPSAMPLING]], [[WEIGHTS_FQ]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x32x47x61xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>
    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @ConvertTransposedConv2DToConv2DNonConstFilter
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x16x30x30xf16>, [[INPUT1:%.+]]: tensor<16x1x16x16xf16>)
func.func @ConvertTransposedConv2DToConv2DNonConstFilter(%input0: tensor<1x16x30x30xf16>, %input1: tensor<16x1x16x16xf16>) -> tensor<1x16x74x74xf16> {
     // transposed conv input
    %conv1_filter = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>

    %conv1 = IE.Convolution(%input0, %conv1_filter) {
                dilations = [1, 1],
                pads_begin = [0, 0],
                pads_end = [0, 0],
                strides = [1, 1]
            } : tensor<1x16x30x30xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x30x30xf16>

    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<100.000000e+00> : tensor<1x1x1x1xf16>

    %fq_input = IE.FakeQuantize(%conv1, %cst_0, %cst_1, %cst_0, %cst_1) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                levels = 256 : i64
            } : tensor<1x16x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x30x30xf16>

    // transposed conv weights
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %fq_weights = IE.FakeQuantize(%input1, %cst_2, %cst_3, %cst_2, %cst_3) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                    levels = 256 : i64
                } : tensor<16x1x16x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x1x16x16xf16>

    // transposed conv
    %2 = IE.TransposedConvolution(%fq_input, %fq_weights) {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0>,
            output_padding = [0, 0],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [2, 2]
        } : tensor<1x16x30x30xf16>, tensor<16x1x16x16xf16> -> tensor<1x16x74x74xf16>

    return %2 : tensor<1x16x74x74xf16>

    // CHECK-NOT:   IE.Upsampling
    // CHECK:       [[CONV1:%.*]] = IE.Convolution([[INPUT0]]
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x30x30xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x30x30xf16>
    // CHECK:       [[FQ1:%.*]] = IE.FakeQuantize([[CONV1]],
    // CHECK-SAME:      tensor<1x16x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x30x30xf16>
    // CHECK:       [[FQ2:%.*]] = IE.FakeQuantize([[INPUT1]],
    // CHECK-SAME:      tensor<16x1x16x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x1x16x16xf16>
    // CHECK:       [[TRANSPOSED_CONV:%.*]] = IE.TransposedConvolution([[FQ1]], [[FQ2]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      output_padding = [0, 0],
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [2, 2]
    // CHECK-SAME:      tensor<1x16x30x30xf16>, tensor<16x1x16x16xf16> -> tensor<1x16x74x74xf16>
    // CHECK:       return [[TRANSPOSED_CONV]]
}

// -----

// CHECK-LABEL: @ConvertTransposedConv2DToConv2DWhitOutputPadding
func.func @ConvertTransposedConv2DToConv2DWhitOutputPadding(%input: tensor<1x32x23x30xf16>) -> tensor<1x16x47x61xf16> {
    %weights = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<16x32x2x2xf16>
    %output = IE.TransposedConvolution(%input, %weights) {strides = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [1, 1]} : tensor<1x32x23x30xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x47x61xf16>
    return %output : tensor<1x16x47x61xf16>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<16x32x2x2xf16>
    // CHECK-NOT:   IE.TransposedConvolution
    // CHECK:       [[UPSAMPLING:%.*]] = IE.Upsampling
    // CHECK-SAME:      #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [1, 2], pads_width = [1, 2]>
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x32x23x30xf16> -> tensor<1x32x48x62xf16>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[UPSAMPLING]], [[WEIGHTS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x32x48x62xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x47x61xf16>
    // CHECK:       return [[CONV]] : tensor<1x16x47x61xf16>
}

// -----

// CHECK-LABEL: @ConvertTransposedConv2DToConv2DWhitOutputPaddingNotFuse
func.func  @ConvertTransposedConv2DToConv2DWhitOutputPaddingNotFuse(%input: tensor<1x32x7x7xf16>) -> tensor<1x16x6x6xf16> {
    %weights = const.Declare tensor<16x32x3x3xf16> = dense<1.000000e+00> : tensor<16x32x3x3xf16>
    %output = IE.TransposedConvolution(%input, %weights) {strides = [1, 1], pads_begin = [2, 2], pads_end = [2, 2], dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [1, 1]} : tensor<1x32x7x7xf16>, tensor<16x32x3x3xf16> -> tensor<1x16x6x6xf16>
    return %output : tensor<1x16x6x6xf16>

    // CHECK:       [[WEIGHTS:%.*]] = const.Declare tensor<16x32x3x3xf16> = dense<1.000000e+00> : tensor<16x32x3x3xf16>
    // CHECK-NOT:   IE.TransposedConvolution
    // CHECK:       [[UPSAMPLING:%.*]] = IE.Upsampling
    // CHECK-SAME:      #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 0], pads_width = [0, 0]>
    // CHECK-SAME:      upsampling_factor = [1, 1, 1]
    // CHECK-SAME:      tensor<1x32x7x7xf16> -> tensor<1x32x7x7xf16>
    // CHECK:       [[SLICE0:%.*]]  = IE.Slice [[UPSAMPLING]] [0, 0, 6, 0] [1, 32, 1, 7] : tensor<1x32x7x7xf16> to tensor<1x32x1x7xf16>
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[UPSAMPLING]], [[SLICE0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x7x7xf16>, tensor<1x32x1x7xf16> -> tensor<1x32x8x7xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[CONCAT0]] [0, 0, 0, 6] [1, 32, 8, 1] : tensor<1x32x8x7xf16> to tensor<1x32x8x1xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[CONCAT0]], [[SLICE1]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x32x8x7xf16>, tensor<1x32x8x1xf16> -> tensor<1x32x8x8xf16>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[CONCAT1]], [[WEIGHTS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x32x8x8xf16>, tensor<16x32x3x3xf16> -> tensor<1x16x6x6xf16>
    // CHECK:       return [[CONV]] : tensor<1x16x6x6xf16>
}

// -----

// CHECK-LABEL: @ConvertTransposedConv2DToConv2DWhitAsymmetricPadding
func.func @ConvertTransposedConv2DToConv2DWhitAsymmetricPadding(%input: tensor<1x128x24x42xf16>) -> tensor<1x128x48x84xf16> {
    %weights = const.Declare tensor<128x128x5x5xf16> = dense<1.000000e+00> : tensor<128x128x5x5xf16>
    %output = IE.TransposedConvolution(%input, %weights) {strides = [2, 2], pads_begin = [1, 1], pads_end = [2, 2], dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0]} : tensor<1x128x24x42xf16>, tensor<128x128x5x5xf16> -> tensor<1x128x48x84xf16>
    return %output : tensor<1x128x48x84xf16>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<128x128x5x5xf16> = dense<1.000000e+00> : tensor<128x128x5x5xf16
    // CHECK-NOT:   IE.TransposedConvolution
    // CHECK:       [[UPSAMPLING:%.*]] = IE.Upsampling
    // CHECK-SAME:      #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [3, 2], pads_width = [3, 2]>
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x128x24x42xf16> -> tensor<1x128x52x88xf16>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[UPSAMPLING]], [[WEIGHTS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x128x52x88xf16>, tensor<128x128x5x5xf16> -> tensor<1x128x48x84xf16>
    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @ConvertTransposedConv2DToConv2DWhitPadding
func.func @ConvertTransposedConv2DToConv2DWhitPadding(%input: tensor<1x128x24x42xf16>) -> tensor<1x128x48x83xf16> {
    %weights = const.Declare tensor<128x128x2x1xf16> = dense<1.000000e+00> : tensor<128x128x2x1xf16>
    %output = IE.TransposedConvolution(%input, %weights) {strides = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0]} : tensor<1x128x24x42xf16>, tensor<128x128x2x1xf16> -> tensor<1x128x48x83xf16>
    return %output : tensor<1x128x48x83xf16>

    // CHECK:      [[WEIGHTS:%.*]] = const.Declare tensor<128x128x2x1xf16> = dense<1.000000e+00> : tensor<128x128x2x1xf16>
    // CHECK-NOT:  IE.TransposedConvolution
    // CHECK:      [[UPSAMPLING:%.*]] = IE.Upsampling(%arg0)
    // CHECK-SAME:      #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [1, 1], pads_width = [0, 0]>,
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x128x24x42xf16> -> tensor<1x128x49x83xf16>
    // CHECK:      [[CONV:%.*]] = IE.Convolution([[UPSAMPLING]], [[WEIGHTS]])
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x128x49x83xf16>, tensor<128x128x2x1xf16> -> tensor<1x128x48x83xf16>
    // CHECK:      return [[CONV]]
}

// -----

// CHECK-LABEL: @ConvertTransposedConv2DWithBiasToConv2D
func.func @ConvertTransposedConv2DWithBiasToConv2D(%input: tensor<1x32x23x30xf16>) -> tensor<1x16x46x60xf16> {
    %weights = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<16x32x2x2xf16>
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    %output = IE.TransposedConvolution(%input, %weights, %bias) {strides = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1>, output_padding = [0, 0]} : tensor<1x32x23x30xf16>, tensor<16x32x2x2xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x46x60xf16>
    return %output : tensor<1x16x46x60xf16>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<16x32x2x2xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK-NOT:   IE.TransposedConvolution
    // CHECK:       [[UPS:%.*]] = IE.Upsampling
    // CHECK-SAME:      #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [1, 1], pads_width = [1, 1]>
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x32x23x30xf16> -> tensor<1x32x47x61xf16>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[UPS]], [[WEIGHTS]], [[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x32x47x61xf16>, tensor<16x32x2x2xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x46x60xf16>
    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @ConvertTransposedConv2DWithOutputShapeToConv2D
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x128x128xf16>
func.func @ConvertTransposedConv2DWithOutputShapeToConv2D(%input: tensor<1x32x128x128xf16>) -> tensor<1x64x128x128xf16> {
    %weights = const.Declare tensor<64x32x2x2xf16> = dense<1.000000e+00> : tensor<64x32x2x2xf16>
    %output_shape = const.Declare tensor<2xsi32> = dense<128> : tensor<2xsi32>
    %output = IE.TransposedConvolution(%input, %weights, %output_shape) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 1, 0>, output_padding = [0, 0], pads_begin = [64, 64], pads_end = [64, 64], strides = [2, 2]} : tensor<1x32x128x128xf16>, tensor<64x32x2x2xf16>, tensor<2xsi32> -> tensor<1x64x128x128xf16>
    return %output : tensor<1x64x128x128xf16>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<64x32x2x2xf16> = dense<1.000000e+00> : tensor<64x32x2x2xf16>
    // CHECK-DAG:   [[OUTPUT_SHAPE:%.+]] = const.Declare tensor<2xsi32> = dense<128> : tensor<2xsi32>
    // CHECK-NOT:   IE.TransposedConvolution
    // CHECK:       [[UPS:%.+]] = IE.Upsampling
    // CHECK-SAME:      #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 0], pads_width = [0, 0]>
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x32x128x128xf16> -> tensor<1x32x255x255xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[UPS]], [[WEIGHTS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x32x255x255xf16>, tensor<64x32x2x2xf16> -> tensor<1x64x254x254xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 63, 63] [1, 64, 128, 128] : tensor<1x64x254x254xf16> to tensor<1x64x128x128xf16>
    // CHECK:       return [[SLICE]]
}

// -----

// CHECK-LABEL: @ConvertTransposedConv2DWithNonConstFilterToConv2D
// CHECK-SAME:      ([[INPUT0:%.+]]: tensor<1x512x4x4xf16>, [[INPUT1:%.+]]: tensor<256x512x3x3xf16>)
func.func @ConvertTransposedConv2DWithNonConstFilterToConv2D(%input0: tensor<1x512x4x4xf16>, %input1: tensor<256x512x3x3xf16>) -> tensor<1x256x9x9xf16> {
    %output = IE.TransposedConvolution(%input0, %input1) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x512x4x4xf16>, tensor<256x512x3x3xf16> -> tensor<1x256x9x9xf16>

    return %output : tensor<1x256x9x9xf16>

    // CHECK-NOT:   IE.TransposedConvolution
    // CHECK:       [[UPS:%.+]] = IE.Upsampling([[INPUT0]])
    // CHECK-SAME:      #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [2, 2], pads_width = [2, 2]>
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x512x4x4xf16> -> tensor<1x512x11x11xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[UPS]], [[INPUT1]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x512x11x11xf16>, tensor<256x512x3x3xf16> -> tensor<1x256x9x9xf16>

    // CHECK:       return [[CONV]]
}
