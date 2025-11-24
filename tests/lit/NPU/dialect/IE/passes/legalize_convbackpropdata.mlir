//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --legalize-convbackpropdata --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @LegalizeConvBackpropData
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x23x30xf16>)
func.func @LegalizeConvBackpropData(%input: tensor<1x16x23x30xf16>) -> tensor<1x32x46x59xf16> {
    %filter = const.Declare tensor<16x32x2x1xf16> = dense<1.000000e+00> : tensor<16x32x2x1xf16>
    %output = IE.ConvolutionBackpropData(%input, %filter) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x16x23x30xf16>, tensor<16x32x2x1xf16> -> tensor<1x32x46x59xf16>
    return %output : tensor<1x32x46x59xf16>

    // CHECK:       [[FILTER:%.+]] = const.Declare tensor<32x16x2x1xf16> = dense<1.000000e+00> : tensor<16x32x2x1xf16>, [#const.Reverse<1 : i64>, #const.Transpose<#map>]
    // CHECK-NOT:   IE.ConvolutionBackpropData
    // CHECK:       [[OUTPUT:%.+]] = IE.TransposedConvolution([[INPUT]], [[FILTER]]) {
    // CHECK-SAME:      dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x16x23x30xf16>, tensor<32x16x2x1xf16> -> tensor<1x32x46x59xf16>
    // CHECK:       return [[OUTPUT]]
}

// -----

// CHECK-LABEL: @LegalizeConvBackpropData1D
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x23xf16>)
func.func @LegalizeConvBackpropData1D(%input: tensor<1x16x23xf16>) -> tensor<1x32x46xf16> {
    %filter = const.Declare tensor<16x32x2xf16> = dense<1.000000e+00> : tensor<16x32x2xf16>
    %output = IE.ConvolutionBackpropData(%input, %filter) {
            dilations = [1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0], pads_begin = [0], pads_end = [0], strides = [2]
        } : tensor<1x16x23xf16>, tensor<16x32x2xf16> -> tensor<1x32x46xf16>
    return %output : tensor<1x32x46xf16>

    // CHECK:       [[FILTER:%.+]] = const.Declare tensor<32x16x2xf16> = dense<1.000000e+00> : tensor<16x32x2xf16>, [#const.Reverse<1 : i64>, #const.Transpose<#HCW>]
    // CHECK-NOT:   IE.ConvolutionBackpropData
    // CHECK:       [[OUTPUT:%.+]] = IE.TransposedConvolution([[INPUT]], [[FILTER]]) {
    // CHECK-SAME:      dilations = [1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0], pads_end = [0], spatial_output_padding = [0], strides = [2]
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
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
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
    // CHECK-SAME:      dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
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
            dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30xf16>, tensor<2x16x32x2x1xf16> -> tensor<1x64x46x59xf16>
    return %output : tensor<1x64x46x59xf16>

    // CHECK:       [[FILTER:%.+]] = const.Declare tensor<2x32x16x2x1xf16> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>, [#const.Reverse<2 : i64>, #const.Transpose<#map>]
    // CHECK-NOT:   IE.GroupConvolutionBackpropData
    // CHECK:       [[OUTPUT:%.+]] = IE.GroupTransposedConvolution([[INPUT]], [[FILTER]]) {
    // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x32x23x30xf16>, tensor<2x32x16x2x1xf16> -> tensor<1x64x46x59xf16>
    // CHECK:       return [[OUTPUT]]
}

// -----

// CHECK-LABEL: @ConvertGroupConvBackpropDataToGroupTransposedConv1D
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x32x23xf16>)
func.func @ConvertGroupConvBackpropDataToGroupTransposedConv1D(%input: tensor<1x32x23xf16>) -> tensor<1x64x46xf16> {
    %filter = const.Declare tensor<2x16x32x2xf16> = dense<1.000000e+00> : tensor<2x16x32x2xf16>
    %output = IE.GroupConvolutionBackpropData(%input, %filter) {
            dilations = [1], spatial_output_padding = [0], pads_begin = [0], pads_end = [0], strides = [2]
        } : tensor<1x32x23xf16>, tensor<2x16x32x2xf16> -> tensor<1x64x46xf16>
    return %output : tensor<1x64x46xf16>

    // CHECK:       [[FILTER:%.+]] = const.Declare tensor<2x32x16x2xf16> = dense<1.000000e+00> : tensor<2x16x32x2xf16>, [#const.Reverse<2 : i64>, #const.Transpose<#NHCW>]
    // CHECK-NOT:   IE.GroupConvolutionBackpropData
    // CHECK:       [[OUTPUT:%.+]] = IE.GroupTransposedConvolution([[INPUT]], [[FILTER]]) {
    // CHECK-SAME:      dilations = [1], pads_begin = [0], pads_end = [0], spatial_output_padding = [0], strides = [2]
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
            dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
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
    // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
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
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
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
    // CHECK-SAME:      dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
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
            dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
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
    // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
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
            dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x512x4x4xf16>, tensor<512x256x3x3xf16> -> tensor<1x256x9x9xf16>

    return %output : tensor<1x256x9x9xf16>

    // CHECK-NOT:   IE.ConvolutionBackpropData
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT1]]) {order_value = #map} : tensor<512x256x3x3xf16> -> tensor<256x512x3x3xf16>
    // CHECK:       [[REVERSE:%.+]] = IE.Reverse([[TRANSPOSE]]) {axis_value = [2, 3], mode = #IE.reverse_mode<INDEX>} : tensor<256x512x3x3xf16> -> tensor<256x512x3x3xf16>
    // CHECK:       [[OUTPUT:%.+]] = IE.TransposedConvolution([[INPUT0]], [[REVERSE]]) {
    // CHECK-SAME:      dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x512x4x4xf16>, tensor<256x512x3x3xf16> -> tensor<1x256x9x9xf16>

    // CHECK:       return [[OUTPUT]]
}

// -----

// CHECK:  #HCW = affine_map<(d0, d1, d2) -> (d1, d0, d2)>

// CHECK-LABEL: @Convert1DConvBackpropDataWithNonConstFilterToTransposedConv
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x768x5120xf32>, [[INPUT1:%.+]]: tensor<768x384x7xf32>)
func.func @Convert1DConvBackpropDataWithNonConstFilterToTransposedConv(%input0: tensor<1x768x5120xf32>, %input1: tensor<768x384x7xf32>) -> tensor<1x384x15360xf32> {
    %output = IE.ConvolutionBackpropData(%input0, %input1) {
            dilations = [1], pads_begin = [2], pads_end = [2], spatial_output_padding = [0], strides = [3]
        } : tensor<1x768x5120xf32>, tensor<768x384x7xf32> -> tensor<1x384x15360xf32>

    return %output : tensor<1x384x15360xf32>

    // CHECK-NOT:   IE.ConvolutionBackpropData
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT1]]) {order_value = #HCW} : tensor<768x384x7xf32> -> tensor<384x768x7xf32>
    // CHECK:       [[REVERSE:%.+]] = IE.Reverse([[TRANSPOSE]]) {axis_value = [2], mode = #IE.reverse_mode<INDEX>} : tensor<384x768x7xf32> -> tensor<384x768x7xf32>
    // CHECK:       [[OUTPUT:%.+]] = IE.TransposedConvolution([[INPUT0]], [[REVERSE]]) {
    // CHECK-SAME:          dilations = [1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [2], pads_end = [2], spatial_output_padding = [0], strides = [3]
    // CHECK-SAME:      } : tensor<1x768x5120xf32>, tensor<384x768x7xf32> -> tensor<1x384x15360xf32>

    // CHECK:       return [[OUTPUT]]
}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3, d4) -> (d1, d0, d2, d3, d4)>

// CHECK-LABEL: @Convert3DConvBackpropDataWithNonConstFilterToTransposedConv
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x768x512x512x512xf32>, [[INPUT1:%.+]]: tensor<768x384x7x7x7xf32>)
func.func @Convert3DConvBackpropDataWithNonConstFilterToTransposedConv(%input0: tensor<1x768x512x512x512xf32>, %input1: tensor<768x384x7x7x7xf32>) -> tensor<1x384x1536x1536x1536xf32> {
    %output = IE.ConvolutionBackpropData(%input0, %input1) {
            dilations = [1, 1, 1], pads_begin = [2, 2, 2], pads_end = [2, 2, 2], spatial_output_padding = [0, 0, 0], strides = [3, 3, 3]
        } : tensor<1x768x512x512x512xf32>, tensor<768x384x7x7x7xf32> -> tensor<1x384x1536x1536x1536xf32>

    return %output : tensor<1x384x1536x1536x1536xf32>

    // CHECK-NOT:   IE.ConvolutionBackpropData
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT1]]) {order_value = #map} : tensor<768x384x7x7x7xf32> -> tensor<384x768x7x7x7xf32>
    // CHECK:       [[REVERSE:%.+]] = IE.Reverse([[TRANSPOSE]]) {axis_value = [2, 3, 4], mode = #IE.reverse_mode<INDEX>} : tensor<384x768x7x7x7xf32> -> tensor<384x768x7x7x7xf32>
    // CHECK:       [[OUTPUT:%.+]] = IE.TransposedConvolution([[INPUT0]], [[REVERSE]]) {
    // CHECK-SAME:          dilations = [1, 1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [2, 2, 2], pads_end = [2, 2, 2], spatial_output_padding = [0, 0, 0], strides = [3, 3, 3]
    // CHECK-SAME:      } : tensor<1x768x512x512x512xf32>, tensor<384x768x7x7x7xf32> -> tensor<1x384x1536x1536x1536xf32>

    // CHECK:       return [[OUTPUT]]
}

// -----

// CHECK-LABEL: @ConvertGroupConvBackpropDataWithNonConstFilterToGroupTransposedConv
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x128x32x32xf16>, [[ARG1:%.+]]: tensor<128x1x1x4x4xf16>)
func.func @ConvertGroupConvBackpropDataWithNonConstFilterToGroupTransposedConv(%arg0: tensor<1x128x32x32xf16>, %arg1: tensor<128x1x1x4x4xf16>) -> tensor<1x128x64x64xf16> {
    %0 = IE.GroupConvolutionBackpropData(%arg0, %arg1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x128x32x32xf16>, tensor<128x1x1x4x4xf16> -> tensor<1x128x64x64xf16>
    return %0 : tensor<1x128x64x64xf16>

    // CHECK-NOT:   IE.GroupConvolutionBackpropData
    // CHECK:       [[REVERSE:%.+]] = IE.Reverse([[ARG1]]) {axis_value = [3, 4], mode = #IE.reverse_mode<INDEX>} : tensor<128x1x1x4x4xf16> -> tensor<128x1x1x4x4xf16>
    // CHECK:       [[OUTPUT:%.+]] = IE.GroupTransposedConvolution([[ARG0]], [[REVERSE]]) {
    // CHECK-SAME:      dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], spatial_output_padding = [0, 0], strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x128x32x32xf16>, tensor<128x1x1x4x4xf16> -> tensor<1x128x64x64xf16>

    // CHECK:       return [[OUTPUT]]

}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @LegalizeConvBackpropDataTo2x2SplitConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x2x2xf32>)
func.func @LegalizeConvBackpropDataTo2x2SplitConv(%input: tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32> {
    %filter = const.Declare tensor<1x1x4x4xf32> = dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00], [9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01], [1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]]]]> : tensor<1x1x4x4xf32>
    %output = IE.ConvolutionBackpropData(%input, %filter){
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], spatial_output_padding = [0, 0], strides = [2, 2]
        } : tensor<1x1x2x2xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
  
    return %output : tensor<1x1x4x4xf32>

    // CHECK-DAG: [[SPLIT_FILTER_1:%.+]] = const.Declare tensor<1x1x2x2xf32> 
    // CHECK-SAME{LITERAL}: dense<[[[[1.600000e+01, 1.400000e+01], [8.000000e+00, 6.000000e+00]]]]> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_2:%.+]] = const.Declare tensor<1x1x2x2xf32>
    // CHECK-SAME{LITERAL}: dense<[[[[1.500000e+01, 1.300000e+01], [7.000000e+00, 5.000000e+00]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_3:%.+]] = const.Declare tensor<1x1x2x2xf32>
    // CHECK-SAME{LITERAL}: dense<[[[[1.200000e+01, 1.000000e+01], [4.000000e+00, 2.000000e+00]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_4:%.+]] = const.Declare tensor<1x1x2x2xf32>
    // CHECK-SAME{LITERAL}: dense<[[[[1.100000e+01, 9.000000e+00], [3.000000e+00, 1.000000e+00]]]]> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_2]]) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_3]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [1, 0], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_4]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]]) 
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]}
    // CHECK:               tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x4x2x2xf32>
    // CHECK: [[D2S:%.+]] = IE.DepthToSpace([[CONCAT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x2x2xf32> -> tensor<1x1x4x4xf32>
    
    // CHECK: return [[D2S]] : tensor<1x1x4x4xf32>
}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @LegalizeQuantizedConvBackpropDataTo2x2SplitConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x2x2xf16>)
func.func @LegalizeQuantizedConvBackpropDataTo2x2SplitConv(%input: tensor<1x1x2x2xf16>) -> tensor<1x1x4x4xf16> {
    %input_low = const.Declare tensor<1xf16> = dense<0.0> : tensor<1xf16>
    %input_high = const.Declare tensor<1xf16> = dense<255.0> : tensor<1xf16>
    %input_fq = IE.FakeQuantize(%input, %input_low, %input_high, %input_low, %input_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x1x2x2xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x1x2x2xf16>

    %filter = const.Declare tensor<1x1x4x4xf16> = dense<[[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]> : tensor<1x1x4x4xsi8>, [#const.CastElemType<f16>]
    %filter_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %filter_high = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %filter_fq = IE.FakeQuantize(%filter, %filter_low, %filter_high, %filter_low, %filter_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x1x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x4x4xf16>

    %output = IE.ConvolutionBackpropData(%input_fq, %filter_fq) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]
        } : tensor<1x1x2x2xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>

    %output_low = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<254.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %output_low, %output_high, %output_low, %output_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x1x4x4xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x1x4x4xf16>

    return %output_fq : tensor<1x1x4x4xf16>

    // CHECK-DAG: [[SPLIT_FILTER_1:%.+]] = const.Declare tensor<1x1x2x2xf16> 
    // CHECK-SAME{LITERAL}: dense<[[[[11, 9], [3, 1]]]]>  : tensor<1x1x2x2xsi8>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f16>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_2:%.+]] = const.Declare tensor<1x1x2x2xf16>
    // CHECK-SAME{LITERAL}: dense<[[[[12, 10], [4, 2]]]]> : tensor<1x1x2x2xsi8>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f16>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_3:%.+]] = const.Declare tensor<1x1x2x2xf16>
    // CHECK-SAME{LITERAL}: dense<[[[[15, 13], [7, 5]]]]> : tensor<1x1x2x2xsi8>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f16>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_4:%.+]] = const.Declare tensor<1x1x2x2xf16>
    // CHECK-SAME{LITERAL}: dense<[[[[16, 14], [8, 6]]]]> : tensor<1x1x2x2xsi8>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f16>, #const.Transpose<#map>

    // CHECK-DAG:   [[OUTPUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:   [[OUTPUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.540000e+02> : tensor<1xf16>

    // CHECK-DAG:   [[FILTER_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[FILTER_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[INPUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<1xf16>
    // CHECK-DAG:   [[INPUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.550000e+02> : tensor<1xf16>

    // CHECK:       [[INPUT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[INPUT_LOW]], [[INPUT_HIGH]], [[INPUT_LOW]], [[INPUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x1x2x2xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x1x2x2xf16>

    // CHECK:       [[FILTER_FQ_1:%.+]] = IE.FakeQuantize([[SPLIT_FILTER_4]], [[FILTER_LOW]], [[FILTER_HIGH]], [[FILTER_LOW]], [[FILTER_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>

    // CHECK:       [[FILTER_FQ_2:%.+]] = IE.FakeQuantize([[SPLIT_FILTER_3]], [[FILTER_LOW]], [[FILTER_HIGH]], [[FILTER_LOW]], [[FILTER_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>

    // CHECK:       [[FILTER_FQ_3:%.+]] = IE.FakeQuantize([[SPLIT_FILTER_2]], [[FILTER_LOW]], [[FILTER_HIGH]], [[FILTER_LOW]], [[FILTER_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>

    // CHECK:       [[FILTER_FQ_4:%.+]] = IE.FakeQuantize([[SPLIT_FILTER_1]], [[FILTER_LOW]], [[FILTER_HIGH]], [[FILTER_LOW]], [[FILTER_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT_FQ]], [[FILTER_FQ_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x2x2xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT_FQ]], [[FILTER_FQ_2]]) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x2x2xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[INPUT_FQ]], [[FILTER_FQ_3]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [1, 0], strides = [1, 1]} : tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x2x2xf16>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[INPUT_FQ]], [[FILTER_FQ_4]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x2x2xf16>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]]) 
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]}
    // CHECK:               tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<1x4x2x2xf16>
    // CHECK: [[D2S:%.+]] = IE.DepthToSpace([[CONCAT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x2x2xf16> -> tensor<1x1x4x4xf16>
    
    // CHECK:       [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[D2S]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x1x4x4xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x1x4x4xf16>

    // CHECK:       return [[OUTPUT_FQ]]
}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @LegalizeConvBackpropDataTo2x2SplitConvWithBiasAdd
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x2x2xf32>)
func.func @LegalizeConvBackpropDataTo2x2SplitConvWithBiasAdd(%input: tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32> {
    %filter = const.Declare tensor<1x1x4x4xf32> = dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00], [9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01], [1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]]]]> : tensor<1x1x4x4xf32> 
    %bias = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %convTranspose = IE.ConvolutionBackpropData(%input, %filter){dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x1x2x2xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
    %add = IE.Add(%convTranspose, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x4x4xf32>
 
    return %add : tensor<1x1x4x4xf32>

    // CHECK-DAG: [[BIAS:%.+]] = const.Declare tensor<1x1x1x1xf32> 
    // CHECK-DAG: [[SPLIT_FILTER_1:%.+]] = const.Declare tensor<1x1x2x2xf32> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_2:%.+]] = const.Declare tensor<1x1x2x2xf32> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_3:%.+]] = const.Declare tensor<1x1x2x2xf32> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_4:%.+]] = const.Declare tensor<1x1x2x2xf32> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_2]]) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_3]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [1, 0], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_4]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>

    // CHECK:   [[ADD0:%.*]] = IE.Add([[CONV0]], [[BIAS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x2x2xf32>
    // CHECK:   [[ADD1:%.*]] = IE.Add([[CONV1]], [[BIAS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x2x2xf32>
    // CHECK:   [[ADD2:%.*]] = IE.Add([[CONV2]], [[BIAS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x2x2xf32>
    // CHECK:   [[ADD3:%.*]] = IE.Add([[CONV3]], [[BIAS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x2x2xf32>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[ADD0]], [[ADD1]], [[ADD2]], [[ADD3]]) 
    // CHECK{LITERAL}:  {static_offsets =  [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]}
    // CHECK:               tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x4x2x2xf32>
    // CHECK: [[D2S:%.+]] = IE.DepthToSpace([[CONCAT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x2x2xf32> -> tensor<1x1x4x4xf32>

    // CHECK:       return [[D2S]]
}    

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @LegalizeQuantizedConvBackpropDataTo2x2SplitConvWithFilterConvert
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x2x2xf16>)
func.func @LegalizeQuantizedConvBackpropDataTo2x2SplitConvWithFilterConvert(%input: tensor<1x1x2x2xf16>) -> tensor<1x1x4x4xf16> {
    %input_low = const.Declare tensor<1xf16> = dense<0.0> : tensor<1xf16>
    %input_high = const.Declare tensor<1xf16> = dense<255.0> : tensor<1xf16>
    %input_fq = IE.FakeQuantize(%input, %input_low, %input_high, %input_low, %input_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x1x2x2xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x1x2x2xf16>

    %filter = const.Declare tensor<1x1x4x4xf16> = dense<[[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]> : tensor<1x1x4x4xsi8>, [#const.CastElemType<f16>]
    %filter_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %filter_high = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %filter_fq = IE.FakeQuantize(%filter, %filter_low, %filter_high, %filter_low, %filter_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x1x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x4x4xf16>

    %filter_convert = IE.Convert(%filter_fq) {dstElemType = f32} : tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf32>

    %output = IE.ConvolutionBackpropData(%input_fq, %filter_convert) {
            dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]
        } : tensor<1x1x2x2xf16>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf16>

    %output_low = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<254.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %output_low, %output_high, %output_low, %output_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
        } : tensor<1x1x4x4xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x1x4x4xf16>

    return %output_fq : tensor<1x1x4x4xf16>


    // CHECK-DAG: [[SPLIT_FILTER_1:%.+]] = const.Declare tensor<1x1x2x2xf16> 
    // CHECK-SAME{LITERAL}: dense<[[[[11, 9], [3, 1]]]]>  : tensor<1x1x2x2xsi8>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f16>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_2:%.+]] = const.Declare tensor<1x1x2x2xf16>
    // CHECK-SAME{LITERAL}: dense<[[[[12, 10], [4, 2]]]]> : tensor<1x1x2x2xsi8>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f16>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_3:%.+]] = const.Declare tensor<1x1x2x2xf16>
    // CHECK-SAME{LITERAL}: dense<[[[[15, 13], [7, 5]]]]> : tensor<1x1x2x2xsi8>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f16>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_4:%.+]] = const.Declare tensor<1x1x2x2xf16>
    // CHECK-SAME{LITERAL}: dense<[[[[16, 14], [8, 6]]]]> : tensor<1x1x2x2xsi8>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f16>, #const.Transpose<#map>

    // CHECK-DAG:   [[OUTPUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:   [[OUTPUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.540000e+02> : tensor<1xf16>

    // CHECK-DAG:   [[FILTER_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[FILTER_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[INPUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<1xf16>
    // CHECK-DAG:   [[INPUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.550000e+02> : tensor<1xf16>

    // CHECK:       [[INPUT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[INPUT_LOW]], [[INPUT_HIGH]], [[INPUT_LOW]], [[INPUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x1x2x2xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x1x2x2xf16>

    // CHECK:       [[FILTER_FQ_1:%.+]] = IE.FakeQuantize([[SPLIT_FILTER_4]], [[FILTER_LOW]], [[FILTER_HIGH]], [[FILTER_LOW]], [[FILTER_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>

    // CHECK:       [[FILTER_FQ_2:%.+]] = IE.FakeQuantize([[SPLIT_FILTER_3]], [[FILTER_LOW]], [[FILTER_HIGH]], [[FILTER_LOW]], [[FILTER_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>

    // CHECK:       [[FILTER_FQ_3:%.+]] = IE.FakeQuantize([[SPLIT_FILTER_2]], [[FILTER_LOW]], [[FILTER_HIGH]], [[FILTER_LOW]], [[FILTER_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>

    // CHECK:       [[FILTER_FQ_4:%.+]] = IE.FakeQuantize([[SPLIT_FILTER_1]], [[FILTER_LOW]], [[FILTER_HIGH]], [[FILTER_LOW]], [[FILTER_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>

    // CHECK:       [[FILTER_CONVERT_1:%.+]] = IE.Convert([[FILTER_FQ_1]]) {dstElemType = f32} : tensor<1x1x2x2xf16> -> tensor<1x1x2x2xf32>
    // CHECK:       [[FILTER_CONVERT_2:%.+]] = IE.Convert([[FILTER_FQ_2]]) {dstElemType = f32} : tensor<1x1x2x2xf16> -> tensor<1x1x2x2xf32>
    // CHECK:       [[FILTER_CONVERT_3:%.+]] = IE.Convert([[FILTER_FQ_3]]) {dstElemType = f32} : tensor<1x1x2x2xf16> -> tensor<1x1x2x2xf32>
    // CHECK:       [[FILTER_CONVERT_4:%.+]] = IE.Convert([[FILTER_FQ_4]]) {dstElemType = f32} : tensor<1x1x2x2xf16> -> tensor<1x1x2x2xf32>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT_FQ]], [[FILTER_CONVERT_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x2x2xf16>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT_FQ]], [[FILTER_CONVERT_2]]) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x1x2x2xf16>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[INPUT_FQ]], [[FILTER_CONVERT_3]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [1, 0], strides = [1, 1]} : tensor<1x1x2x2xf16>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf16>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[INPUT_FQ]], [[FILTER_CONVERT_4]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x2x2xf16>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf16>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]]) 
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]}
    // CHECK:               tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<1x4x2x2xf16>
    // CHECK: [[D2S:%.+]] = IE.DepthToSpace([[CONCAT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x2x2xf16> -> tensor<1x1x4x4xf16>
    
    // CHECK:       [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[D2S]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x1x4x4xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x1x4x4xf16>

    // CHECK:       return [[OUTPUT_FQ]]

}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @LegalizeConvBackpropDataTo2x2SplitConvWithNonBiasAdd
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x2x2xf32>)
func.func @LegalizeConvBackpropDataTo2x2SplitConvWithNonBiasAdd(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32> {
    %cst = const.Declare tensor<1x1x4x4xf32> = dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00], [9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01], [1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]]]]> : tensor<1x1x4x4xf32> 
    %0 = IE.ConvolutionBackpropData(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x1x2x2xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
    %1 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 1], pads_end_attr = [0, 0, 1, 1]} : tensor<1x1x2x2xf32> -> tensor<1x1x4x4xf32>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
    return %2 : tensor<1x1x4x4xf32>

    // CHECK-DAG: [[SPLIT_FILTER_1:%.+]] = const.Declare tensor<1x1x2x2xf32> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_2:%.+]] = const.Declare tensor<1x1x2x2xf32> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_3:%.+]] = const.Declare tensor<1x1x2x2xf32> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_4:%.+]] = const.Declare tensor<1x1x2x2xf32> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_2]]) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_3]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [1, 0], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_4]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]]) 
    // CHECK{LITERAL}:  {static_offsets =  [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]}
    // CHECK:               tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x4x2x2xf32>
    // CHECK: [[D2S:%.+]] = IE.DepthToSpace([[CONCAT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x2x2xf32> -> tensor<1x1x4x4xf32>

    // CHECK:       [[PAD:%.+]] = IE.Pad([[INPUT]]) {
    // CHECK-SAME:      mode = #IE.pad_mode<CONSTANT>
    // CHECK-SAME:      pad_value_attr = 0.000000e+00
    // CHECK-SAME:      pads_begin_attr = [0, 0, 1, 1]
    // CHECK-SAME:      pads_end_attr = [0, 0, 1, 1]
    // CHECK-SAME:      : tensor<1x1x2x2xf32> -> tensor<1x1x4x4xf32>

    // CHECK:   [[ADD:%.*]] = IE.Add([[D2S]], [[PAD]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>

    // CHECK:       return [[ADD]]
}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @LegalizeConvBackpropDataTo2x2SplitConvWith2AddOps
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x2x2xf32>)

func.func @LegalizeConvBackpropDataTo2x2SplitConvWith2AddOps(%arg0: tensor<1x1x2x2xf32>) -> (tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>) {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x4x4xf32> = dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00], [9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01], [1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]]]]> : tensor<1x1x4x4xf32> 
    %0 = IE.ConvolutionBackpropData(%arg0, %cst_1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x1x2x2xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>
    %1 = IE.Add(%0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x4x4xf32>
    %2 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x4x4xf32>
    return %2, %1 : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK-DAG: [[SPLIT_FILTER_1:%.+]] = const.Declare tensor<1x1x2x2xf32> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_2:%.+]] = const.Declare tensor<1x1x2x2xf32> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_3:%.+]] = const.Declare tensor<1x1x2x2xf32> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_4:%.+]] = const.Declare tensor<1x1x2x2xf32> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_2]]) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_3]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [1, 0], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_4]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]]) 
    // CHECK{LITERAL}:  {static_offsets =  [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]}
    // CHECK:               tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x4x2x2xf32>
    // CHECK: [[D2S:%.+]] = IE.DepthToSpace([[CONCAT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x2x2xf32> -> tensor<1x1x4x4xf32>

    // CHECK:   [[ADD0:%.*]] = IE.Add([[D2S]], [[CST1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x4x4xf32>

    // CHECK:   [[ADD1:%.*]] = IE.Add([[D2S]], [[CST0]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x4x4xf32>

    // CHECK:       return [[ADD1]], [[ADD0]] : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>
}
