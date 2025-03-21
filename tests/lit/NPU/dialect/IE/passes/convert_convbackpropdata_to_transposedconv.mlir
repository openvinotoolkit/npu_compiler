//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-convbackpropdata-to-transposedconv --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @ConvertConvBackpropDataToTransposedConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x23x30xf16>)
func.func @ConvertConvBackpropDataToTransposedConv(%input: tensor<1x16x23x30xf16>) -> tensor<1x32x46x59xf16> {
    %filter = const.Declare tensor<16x32x2x1xf16> = dense<1.000000e+00> : tensor<16x32x2x1xf16>
    %output = IE.ConvolutionBackpropData(%input, %filter) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x16x23x30xf16>, tensor<16x32x2x1xf16> -> tensor<1x32x46x59xf16>
    return %output : tensor<1x32x46x59xf16>

    // CHECK:       [[FILTER:%.+]] = const.Declare tensor<32x16x2x1xf16> = dense<1.000000e+00> : tensor<16x32x2x1xf16>, [#const.Reverse<1 : i64>, #const.Transpose<#map>]
    // CHECK-NOT:   IE.ConvolutionBackpropData
    // CHECK:       [[OUTPUT:%.+]] = IE.TransposedConvolution([[INPUT]], [[FILTER]]) {
    // CHECK-SAME:      dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x16x23x30xf16>, tensor<32x16x2x1xf16> -> tensor<1x32x46x59xf16>
    // CHECK:       return [[OUTPUT]]
}

// -----

// CHECK-LABEL: @ConvertConvBackpropDataToTransposedConv1D
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x23xf16>)
func.func @ConvertConvBackpropDataToTransposedConv1D(%input: tensor<1x16x23xf16>) -> tensor<1x32x46xf16> {
    %filter = const.Declare tensor<16x32x2xf16> = dense<1.000000e+00> : tensor<16x32x2xf16>
    %output = IE.ConvolutionBackpropData(%input, %filter) {
            dilations = [1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0], pads_begin = [0], pads_end = [0], strides = [2]
        } : tensor<1x16x23xf16>, tensor<16x32x2xf16> -> tensor<1x32x46xf16>
    return %output : tensor<1x32x46xf16>

    // CHECK:       [[FILTER:%.+]] = const.Declare tensor<32x16x2xf16> = dense<1.000000e+00> : tensor<16x32x2xf16>, [#const.Reverse<1 : i64>, #const.Transpose<#HCW>]
    // CHECK-NOT:   IE.ConvolutionBackpropData
    // CHECK:       [[OUTPUT:%.+]] = IE.TransposedConvolution([[INPUT]], [[FILTER]]) {
    // CHECK-SAME:      dilations = [1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0], pads_begin = [0], pads_end = [0], strides = [2]
    // CHECK-SAME:  } : tensor<1x16x23xf16>, tensor<32x16x2xf16> -> tensor<1x32x46xf16>
    // CHECK:       return [[OUTPUT]]
}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @ConvertQuantizedConvBackpropDataToTransposedConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x23x30xf16>)
func.func @ConvertQuantizedConvBackpropDataToTransposedConv(%input: tensor<1x16x23x30xf16>) -> tensor<1x32x46x59xf16> {
    %input_low = const.Declare tensor<1xf16> = dense<0.0> : tensor<1xf16>
    %input_high = const.Declare tensor<1xf16> = dense<255.0> : tensor<1xf16>
    %input_fq = IE.FakeQuantize(%input, %input_low, %input_high, %input_low, %input_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x16x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x16x23x30xf16>

    %filter = const.Declare tensor<16x32x2x1xf16> = dense<1.000000e+00> : tensor<16x32x2x1xf16>
    %filter_low = const.Declare tensor<1x32x1x1xf16> = dense<0.000000e+00> : tensor<1x32x1x1xf16>
    %filter_high = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>
    %filter_fq = IE.FakeQuantize(%filter, %filter_low, %filter_high, %filter_low, %filter_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<16x32x2x1xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<16x32x2x1xf16>

    %output = IE.ConvolutionBackpropData(%input_fq, %filter_fq) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x16x23x30xf16>, tensor<16x32x2x1xf16> -> tensor<1x32x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<254.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %output_low, %output_high, %output_low, %output_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    return %output_fq : tensor<1x32x46x59xf16>

    // CHECK-DAG:   [[INPUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<1xf16>
    // CHECK-DAG:   [[INPUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.550000e+02> : tensor<1xf16>
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<32x16x2x1xf16> = dense<1.000000e+00> : tensor<16x32x2x1xf16>, [#const.Reverse<1 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:   [[FILTER_LOW:%.+]] = const.Declare tensor<32x1x1x1xf16> = dense<0.000000e+00> : tensor<1x32x1x1xf16>, [#const.Transpose<#map>]
    // CHECK-DAG:   [[FILTER_HIGH:%.+]] = const.Declare tensor<32x1x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>, [#const.Transpose<#map>]
    // CHECK-DAG:   [[OUTPUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:   [[OUTPUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.540000e+02> : tensor<1xf16>

    // CHECK:       [[INPUT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[INPUT_LOW]], [[INPUT_HIGH]], [[INPUT_LOW]], [[INPUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x16x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x16x23x30xf16>

    // CHECK:       [[FILTER_FQ:%.+]] = IE.FakeQuantize([[FILTER]], [[FILTER_LOW]], [[FILTER_HIGH]], [[FILTER_LOW]], [[FILTER_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<32x16x2x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16> -> tensor<32x16x2x1xf16>

    // CHECK-NOT:   IE.ConvolutionBackpropData
    // CHECK:       [[OUTPUT:%.+]] = IE.TransposedConvolution([[INPUT_FQ]], [[FILTER_FQ]]) {
    // CHECK-SAME:      dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x16x23x30xf16>, tensor<32x16x2x1xf16> -> tensor<1x32x46x59xf16>

    // CHECK:       [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[OUTPUT]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    // CHECK:       return [[OUTPUT_FQ]]
}

// -----

// CHECK-LABEL: @ConvertGroupConvBackpropDataToGroupTransposedConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x32x23x30xf16>)
func.func @ConvertGroupConvBackpropDataToGroupTransposedConv(%input: tensor<1x32x23x30xf16>) -> tensor<1x64x46x59xf16> {
    %filter = const.Declare tensor<2x16x32x2x1xf16> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>
    %output = IE.GroupConvolutionBackpropData(%input, %filter) {
            dilations = [1, 1], output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30xf16>, tensor<2x16x32x2x1xf16> -> tensor<1x64x46x59xf16>
    return %output : tensor<1x64x46x59xf16>

    // CHECK:       [[FILTER:%.+]] = const.Declare tensor<2x32x16x2x1xf16> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>, [#const.Reverse<2 : i64>, #const.Transpose<#map>]
    // CHECK-NOT:   IE.GroupConvolutionBackpropData
    // CHECK:       [[OUTPUT:%.+]] = IE.GroupTransposedConvolution([[INPUT]], [[FILTER]]) {
    // CHECK-SAME:      dilations = [1, 1], output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x32x23x30xf16>, tensor<2x32x16x2x1xf16> -> tensor<1x64x46x59xf16>
    // CHECK:       return [[OUTPUT]]
}

// -----

// CHECK-LABEL: @ConvertGroupConvBackpropDataToGroupTransposedConv1D
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x32x23xf16>)
func.func @ConvertGroupConvBackpropDataToGroupTransposedConv1D(%input: tensor<1x32x23xf16>) -> tensor<1x64x46xf16> {
    %filter = const.Declare tensor<2x16x32x2xf16> = dense<1.000000e+00> : tensor<2x16x32x2xf16>
    %output = IE.GroupConvolutionBackpropData(%input, %filter) {
            dilations = [1], output_padding = [0], pads_begin = [0], pads_end = [0], strides = [2]
        } : tensor<1x32x23xf16>, tensor<2x16x32x2xf16> -> tensor<1x64x46xf16>
    return %output : tensor<1x64x46xf16>

    // CHECK:       [[FILTER:%.+]] = const.Declare tensor<2x32x16x2xf16> = dense<1.000000e+00> : tensor<2x16x32x2xf16>, [#const.Reverse<2 : i64>, #const.Transpose<#NHCW>]
    // CHECK-NOT:   IE.GroupConvolutionBackpropData
    // CHECK:       [[OUTPUT:%.+]] = IE.GroupTransposedConvolution([[INPUT]], [[FILTER]]) {
    // CHECK-SAME:      dilations = [1], output_padding = [0], pads_begin = [0], pads_end = [0], strides = [2]
    // CHECK-SAME:  } : tensor<1x32x23xf16>, tensor<2x32x16x2xf16> -> tensor<1x64x46xf16>
    // CHECK:       return [[OUTPUT]]
}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3, d4)>

// CHECK-LABEL: @ConvertQuantizedGroupConvBackpropDataToGroupTransposedConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x32x23x30xf16>)
func.func @ConvertQuantizedGroupConvBackpropDataToGroupTransposedConv(%input: tensor<1x32x23x30xf16>) -> tensor<1x64x46x59xf16> {
    %input_low = const.Declare tensor<1xf16> = dense<0.0> : tensor<1xf16>
    %input_high = const.Declare tensor<1xf16> = dense<255.0> : tensor<1xf16>
    %input_fq = IE.FakeQuantize(%input, %input_low, %input_high, %input_low, %input_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x32x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x23x30xf16>

    %filter = const.Declare tensor<2x16x32x2x1xf16> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>
    %filter_low = const.Declare tensor<1x1x32x1x1xf16> = dense<0.000000e+00> : tensor<1x1x32x1x1xf16>
    %filter_high = const.Declare tensor<1x1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x1x32x1x1xf16>
    %filter_fq = IE.FakeQuantize(%filter, %filter_low, %filter_high, %filter_low, %filter_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<2x16x32x2x1xf16>, tensor<1x1x32x1x1xf16>, tensor<1x1x32x1x1xf16>, tensor<1x1x32x1x1xf16>, tensor<1x1x32x1x1xf16> -> tensor<2x16x32x2x1xf16>

    %output = IE.GroupConvolutionBackpropData(%input_fq, %filter_fq) {
            dilations = [1, 1], output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30xf16>, tensor<2x16x32x2x1xf16> -> tensor<1x64x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<254.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %output_low, %output_high, %output_low, %output_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    return %output_fq : tensor<1x64x46x59xf16>

    // CHECK-DAG:   [[INPUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<1xf16>
    // CHECK-DAG:   [[INPUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.550000e+02> : tensor<1xf16>
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<2x32x16x2x1xf16> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>, [#const.Reverse<2 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:   [[FILTER_LOW:%.+]] = const.Declare tensor<1x32x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x32x1x1xf16>, [#const.Transpose<#map>]
    // CHECK-DAG:   [[FILTER_HIGH:%.+]] = const.Declare tensor<1x32x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x32x1x1xf16>, [#const.Transpose<#map>]
    // CHECK-DAG:   [[OUTPUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:   [[OUTPUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.540000e+02> : tensor<1xf16>

    // CHECK:       [[INPUT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[INPUT_LOW]], [[INPUT_HIGH]], [[INPUT_LOW]], [[INPUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x32x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x23x30xf16>

    // CHECK:       [[FILTER_FQ:%.+]] = IE.FakeQuantize([[FILTER]], [[FILTER_LOW]], [[FILTER_HIGH]], [[FILTER_LOW]], [[FILTER_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<2x32x16x2x1xf16>, tensor<1x32x1x1x1xf16>, tensor<1x32x1x1x1xf16>, tensor<1x32x1x1x1xf16>, tensor<1x32x1x1x1xf16> -> tensor<2x32x16x2x1xf16>

    // CHECK-NOT:   IE.GroupConvolutionBackpropData
    // CHECK:       [[OUTPUT:%.+]] = IE.GroupTransposedConvolution([[INPUT_FQ]], [[FILTER_FQ]]) {
    // CHECK-SAME:      dilations = [1, 1], output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x32x23x30xf16>, tensor<2x32x16x2x1xf16> -> tensor<1x64x46x59xf16>

    // CHECK:       [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[OUTPUT]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    // CHECK:       return [[OUTPUT_FQ]]
}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @ConvertQuantizedConvertedConvBackpropDataToTransposedConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x23x30xf16>)
func.func @ConvertQuantizedConvertedConvBackpropDataToTransposedConv(%input: tensor<1x16x23x30xf16>) -> tensor<1x32x46x59xf16> {
    %input_low = const.Declare tensor<1xf16> = dense<0.0> : tensor<1xf16>
    %input_high = const.Declare tensor<1xf16> = dense<255.0> : tensor<1xf16>
    %input_fq = IE.FakeQuantize(%input, %input_low, %input_high, %input_low, %input_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x16x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x16x23x30xf16>

    %filter = const.Declare tensor<16x32x2x1xf16> = dense<1.000000e+00> : tensor<16x32x2x1xf16>
    %filter_low = const.Declare tensor<1x32x1x1xf16> = dense<0.000000e+00> : tensor<1x32x1x1xf16>
    %filter_high = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>
    %filter_fq = IE.FakeQuantize(%filter, %filter_low, %filter_high, %filter_low, %filter_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<16x32x2x1xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<16x32x2x1xf16>

    %filter_convert = IE.Convert(%filter_fq) {dstElemType = f32} : tensor<16x32x2x1xf16> -> tensor<16x32x2x1xf32>

    %output = IE.ConvolutionBackpropData(%input_fq, %filter_convert) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x16x23x30xf16>, tensor<16x32x2x1xf32> -> tensor<1x32x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<254.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %output_low, %output_high, %output_low, %output_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    return %output_fq : tensor<1x32x46x59xf16>

    // CHECK-DAG:   [[INPUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<1xf16>
    // CHECK-DAG:   [[INPUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.550000e+02> : tensor<1xf16>
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<32x16x2x1xf16> = dense<1.000000e+00> : tensor<16x32x2x1xf16>, [#const.Reverse<1 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:   [[FILTER_LOW:%.+]] = const.Declare tensor<32x1x1x1xf16> = dense<0.000000e+00> : tensor<1x32x1x1xf16>, [#const.Transpose<#map>]
    // CHECK-DAG:   [[FILTER_HIGH:%.+]] = const.Declare tensor<32x1x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>, [#const.Transpose<#map>]
    // CHECK-DAG:   [[OUTPUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:   [[OUTPUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.540000e+02> : tensor<1xf16>

    // CHECK:       [[INPUT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[INPUT_LOW]], [[INPUT_HIGH]], [[INPUT_LOW]], [[INPUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x16x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x16x23x30xf16>

    // CHECK:       [[FILTER_FQ:%.+]] = IE.FakeQuantize([[FILTER]], [[FILTER_LOW]], [[FILTER_HIGH]], [[FILTER_LOW]], [[FILTER_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<32x16x2x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16> -> tensor<32x16x2x1xf16>

    // CHECK:       [[FILTER_CONVERT:%.+]] = IE.Convert([[FILTER_FQ]]) {dstElemType = f32} : tensor<32x16x2x1xf16> -> tensor<32x16x2x1xf32>

    // CHECK-NOT:   IE.ConvolutionBackpropData
    // CHECK:       [[OUTPUT:%.+]] = IE.TransposedConvolution([[INPUT_FQ]], [[FILTER_CONVERT]]) {
    // CHECK-SAME:      dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x16x23x30xf16>, tensor<32x16x2x1xf32> -> tensor<1x32x46x59xf16>

    // CHECK:       [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[OUTPUT]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    // CHECK:       return [[OUTPUT_FQ]]
}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3, d4)>

// CHECK-LABEL: @ConvertQuantizedConvertedGroupConvBackpropDataToGroupTransposedConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x32x23x30xf16>)
func.func @ConvertQuantizedConvertedGroupConvBackpropDataToGroupTransposedConv(%input: tensor<1x32x23x30xf16>) -> tensor<1x64x46x59xf16> {
    %input_low = const.Declare tensor<1xf16> = dense<0.0> : tensor<1xf16>
    %input_high = const.Declare tensor<1xf16> = dense<255.0> : tensor<1xf16>
    %input_fq = IE.FakeQuantize(%input, %input_low, %input_high, %input_low, %input_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x32x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x23x30xf16>

    %filter = const.Declare tensor<2x16x32x2x1xf16> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>
    %filter_low = const.Declare tensor<1x1x32x1x1xf16> = dense<0.000000e+00> : tensor<1x1x32x1x1xf16>
    %filter_high = const.Declare tensor<1x1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x1x32x1x1xf16>
    %filter_fq = IE.FakeQuantize(%filter, %filter_low, %filter_high, %filter_low, %filter_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<2x16x32x2x1xf16>, tensor<1x1x32x1x1xf16>, tensor<1x1x32x1x1xf16>, tensor<1x1x32x1x1xf16>, tensor<1x1x32x1x1xf16> -> tensor<2x16x32x2x1xf16>

    %filter_convert = IE.Convert(%filter_fq) {dstElemType = f32} : tensor<2x16x32x2x1xf16> -> tensor<2x16x32x2x1xf32>

    %output = IE.GroupConvolutionBackpropData(%input_fq, %filter_convert) {
            dilations = [1, 1], output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30xf16>, tensor<2x16x32x2x1xf32> -> tensor<1x64x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<254.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %output_low, %output_high, %output_low, %output_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    return %output_fq : tensor<1x64x46x59xf16>

    // CHECK-DAG:   [[INPUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<1xf16>
    // CHECK-DAG:   [[INPUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.550000e+02> : tensor<1xf16>
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<2x32x16x2x1xf16> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>, [#const.Reverse<2 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:   [[FILTER_LOW:%.+]] = const.Declare tensor<1x32x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x32x1x1xf16>, [#const.Transpose<#map>]
    // CHECK-DAG:   [[FILTER_HIGH:%.+]] = const.Declare tensor<1x32x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x32x1x1xf16>, [#const.Transpose<#map>]
    // CHECK-DAG:   [[OUTPUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:   [[OUTPUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.540000e+02> : tensor<1xf16>

    // CHECK:       [[INPUT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[INPUT_LOW]], [[INPUT_HIGH]], [[INPUT_LOW]], [[INPUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x32x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x23x30xf16>

    // CHECK:       [[FILTER_FQ:%.+]] = IE.FakeQuantize([[FILTER]], [[FILTER_LOW]], [[FILTER_HIGH]], [[FILTER_LOW]], [[FILTER_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<2x32x16x2x1xf16>, tensor<1x32x1x1x1xf16>, tensor<1x32x1x1x1xf16>, tensor<1x32x1x1x1xf16>, tensor<1x32x1x1x1xf16> -> tensor<2x32x16x2x1xf16>

    // CHECK:       [[FILTER_CONVERT:%.+]] = IE.Convert([[FILTER_FQ]]) {dstElemType = f32} : tensor<2x32x16x2x1xf16> -> tensor<2x32x16x2x1xf32>

    // CHECK-NOT:   IE.GroupConvolutionBackpropData
    // CHECK:       [[OUTPUT:%.+]] = IE.GroupTransposedConvolution([[INPUT_FQ]], [[FILTER_CONVERT]]) {
    // CHECK-SAME:      dilations = [1, 1], output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x32x23x30xf16>, tensor<2x32x16x2x1xf32> -> tensor<1x64x46x59xf16>

    // CHECK:       [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[OUTPUT]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    // CHECK:       return [[OUTPUT_FQ]]
}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @ConvertConvBackpropDataWithNonConstFilterToTransposedConv
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x512x4x4xf16>, [[INPUT1:%.+]]: tensor<512x256x3x3xf16>)
func.func @ConvertConvBackpropDataWithNonConstFilterToTransposedConv(%input0: tensor<1x512x4x4xf16>, %input1: tensor<512x256x3x3xf16>) -> tensor<1x256x9x9xf16> {
    %output = IE.ConvolutionBackpropData(%input0, %input1) {
            dilations = [1, 1], output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x512x4x4xf16>, tensor<512x256x3x3xf16> -> tensor<1x256x9x9xf16>

    return %output : tensor<1x256x9x9xf16>

    // CHECK-NOT:   IE.ConvolutionBackpropData
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT1]]) {order_value = #map} : tensor<512x256x3x3xf16> -> tensor<256x512x3x3xf16>
    // CHECK:       [[REVERSE:%.+]] = IE.Reverse([[TRANSPOSE]]) {axis_value = [2, 3], mode = #IE.reverse_mode<INDEX>} : tensor<256x512x3x3xf16> -> tensor<256x512x3x3xf16>
    // CHECK:       [[OUTPUT:%.+]] = IE.TransposedConvolution([[INPUT0]], [[REVERSE]]) {
    // CHECK-SAME:      dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x512x4x4xf16>, tensor<256x512x3x3xf16> -> tensor<1x256x9x9xf16>

    // CHECK:       return [[OUTPUT]]
}
