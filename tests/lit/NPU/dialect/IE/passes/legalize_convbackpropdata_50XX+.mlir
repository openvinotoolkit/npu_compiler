//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --legalize-convbackpropdata --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @ConvertQuantizedGroupConvBackpropDataToGroupTransposedConvF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x23x30xf16>
func.func @ConvertQuantizedGroupConvBackpropDataToGroupTransposedConvF8E4M3FN(%input: tensor<1x32x23x30xf16>) -> tensor<1x64x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-4.480000e+02> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<4.480000e+02> : tensor<1xf16>
    %fq = IE.FakeQuantize(%input, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x32x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x23x30xf16>

    %filter = const.Declare tensor<2x16x32x2x1xf16> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>
    %filter_fq = IE.FakeQuantize(%filter, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<2x16x32x2x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<2x16x32x2x1xf16>

    %output = IE.GroupConvolutionBackpropData(%fq, %filter_fq) {
        dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x32x23x30xf16>, tensor<2x16x32x2x1xf16> -> tensor<1x64x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    return %output_fq : tensor<1x64x46x59xf16>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<2x32x16x2x1xf16> = dense<1.000000e+00> : tensor<2x16x32x2x1xf16>, [#const.Reverse<2 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-4.480000e+02> : tensor<1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<4.480000e+02> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>

    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x23x30xf16>
    // CHECK:    [[FILTER_FQ:%.+]] = IE.FakeQuantize([[FILTER]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<2x32x16x2x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<2x32x16x2x1xf16>

    // CHECK:    [[CONV:%.+]] = IE.GroupTransposedConvolution([[FQ]], [[FILTER_FQ]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x32x23x30xf16>, tensor<2x32x16x2x1xf16> -> tensor<1x64x46x59xf16>
    // CHECK:    [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW]], [[HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    // CHECK:    return [[OUTPUT_FQ]] : tensor<1x64x46x59xf16>
}

// -----

// CHECK-LABEL: @ConvertQuantizedConvertedConvBackpropDataToTransposedConvF8E5M2
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x23x30xf16>)
func.func @ConvertQuantizedConvertedConvBackpropDataToTransposedConvF8E5M2(%input: tensor<1x16x23x30xf16>) -> tensor<1x32x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>
    %fq = IE.FakeQuantize(%input, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x16x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x16x23x30xf16>

    %filter = const.Declare tensor<16x32x2x1xf16> = dense<1.000000e+00> : tensor<16x32x2x1xf16>
    %filter_fq = IE.FakeQuantize(%filter, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<16x32x2x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<16x32x2x1xf16>

    %filter_convert = IE.Convert(%filter_fq) {dstElemType = f32} : tensor<16x32x2x1xf16> -> tensor<16x32x2x1xf32>

    %output = IE.ConvolutionBackpropData(%fq, %filter_convert) {
        dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x16x23x30xf16>, tensor<16x32x2x1xf32> -> tensor<1x32x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    return %output_fq : tensor<1x32x46x59xf16>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<32x16x2x1xf16> = dense<1.000000e+00> : tensor<16x32x2x1xf16>, [#const.Reverse<1 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>

    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x16x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x16x23x30xf16>
    // CHECK:    [[FILTER_FQ:%.+]] = IE.FakeQuantize([[FILTER]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<32x16x2x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<32x16x2x1xf16>
    // CHECK:    [[CONVERT:%.+]] = IE.Convert([[FILTER_FQ]]) {dstElemType = f32} : tensor<32x16x2x1xf16> -> tensor<32x16x2x1xf32>

    // CHECK:    [[CONV:%.+]] = IE.TransposedConvolution([[FQ]], [[CONVERT]])
    // CHECK-SAME:  {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x16x23x30xf16>, tensor<32x16x2x1xf32> -> tensor<1x32x46x59xf16>
    // CHECK:    [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW]], [[HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    // CHECK:    return [[OUTPUT_FQ]] : tensor<1x32x46x59xf16>
}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @LegalizeConvBackpropDataTo3x3SplitConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x64x65xf32>)
func.func @LegalizeConvBackpropDataTo3x3SplitConv(%arg0: tensor<1x1x64x65xf32>) -> tensor<1x1x128x130xf32> {
    %cst = const.Declare tensor<1x1x4x4xf32> = dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00], [9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01], [1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]]]]> : tensor<1x1x4x4xf32> 
    %0 = IE.ConvolutionBackpropData(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x1x64x65xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x128x130xf32>
    return %0 : tensor<1x1x128x130xf32>

    
    // CHECK-DAG: [[SPLIT_FILTER_1:%.+]] = const.Declare tensor<1x1x3x3xf32> 
    // CHECK-SAME{LITERAL}: dense<[[[[1.600000e+01, 1.400000e+01, 0.000000e+00], [8.000000e+00, 6.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]]]]> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_2:%.+]] = const.Declare tensor<1x1x3x3xf32>
    // CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 1.500000e+01, 1.300000e+01], [0.000000e+00, 7.000000e+00, 5.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_3:%.+]] = const.Declare tensor<1x1x3x3xf32>
    // CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 0.000000e+00, 0.000000e+00], [1.200000e+01, 1.000000e+01, 0.000000e+00], [4.000000e+00, 2.000000e+00, 0.000000e+00]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_4:%.+]] = const.Declare tensor<1x1x3x3xf32>
    // CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.100000e+01, 9.000000e+00], [0.000000e+00, 3.000000e+00, 1.000000e+00]]]]> 
    // CHECK-SAME{LITERAL}: #const.CastElemType<f32>, #const.Transpose<#map>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<1x1x3x3xf32> -> tensor<1x1x64x65xf32>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<1x1x3x3xf32> -> tensor<1x1x64x65xf32>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_3]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<1x1x3x3xf32> -> tensor<1x1x64x65xf32>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[INPUT]], [[SPLIT_FILTER_4]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<1x1x3x3xf32> -> tensor<1x1x64x65xf32>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]]) 
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]}
    // CHECK:               tensor<1x1x64x65xf32>, tensor<1x1x64x65xf32>, tensor<1x1x64x65xf32>, tensor<1x1x64x65xf32> -> tensor<1x4x64x65xf32>
    // CHECK: [[D2S:%.+]] = IE.DepthToSpace([[CONCAT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x64x65xf32> -> tensor<1x1x128x130xf32>
    
    // CHECK: return [[D2S]] : tensor<1x1x128x130xf32>
  }
