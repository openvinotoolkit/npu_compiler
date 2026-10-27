//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --legalize-convbackpropdata --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010
// COM: F8 is only supported on NPU50+, no need to run these tests on all platforms.

// CHECK-LABEL: @ConvertDynDeqGroupConvBackpropDataToGroupTransposedConvF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x23x30xf16>
func.func @ConvertDynDeqGroupConvBackpropDataToGroupTransposedConvF8E4M3FN(%input: tensor<1x32x23x30xf16>) -> tensor<1x64x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-4.480000e+02> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<4.480000e+02> : tensor<1xf16>
    %fq = IE.FakeQuantize(%input, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x32x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x23x30xf16>

    %filter = const.Declare tensor<2x16x32x2x1xf8E4M3FN> = dense<1.000000e+00> : tensor<2x16x32x2x1xf8E4M3FN>
    %filter_scale = const.Declare tensor<1x1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1x1xf16>
    %filter_dq = IE.DynamicDequantize(%filter, %filter_scale) {dstElemType = f16} :
            tensor<2x16x32x2x1xf8E4M3FN>, tensor<1x1x1x1x1xf16> -> tensor<2x16x32x2x1xf16>

    %output = IE.GroupConvolutionBackpropData(%fq, %filter_dq) {
        dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x32x23x30xf16>, tensor<2x16x32x2x1xf16> -> tensor<1x64x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    return %output_fq : tensor<1x64x46x59xf16>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<2x32x16x2x1xf8E4M3FN> = dense<1.000000e+00> : tensor<2x16x32x2x1xf8E4M3FN>, [#const.Reverse<2 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:    [[FILTER_SCALE:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1x1xf16>
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-4.480000e+02> : tensor<1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<4.480000e+02> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>

    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x23x30xf16>
    // CHECK:    [[FILTER_DQ:%.+]] = IE.DynamicDequantize([[FILTER]], [[FILTER_SCALE]]) {dstElemType = f16} :
    // CHECK-SAME:      tensor<2x32x16x2x1xf8E4M3FN>, tensor<1x1x1x1x1xf16> -> tensor<2x32x16x2x1xf16>

    // CHECK:    [[CONV:%.+]] = IE.GroupTransposedConvolution([[FQ]], [[FILTER_DQ]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x32x23x30xf16>, tensor<2x32x16x2x1xf16> -> tensor<1x64x46x59xf16>
    // CHECK:    [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW]], [[HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    // CHECK:    return [[OUTPUT_FQ]] : tensor<1x64x46x59xf16>
}

// -----
!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @ConvertDynDeqGroupConvBackpropDataToGroupTransposedConvF8E4M3FNPerTensor
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x23x30x!qElemType>
func.func @ConvertDynDeqGroupConvBackpropDataToGroupTransposedConvF8E4M3FNPerTensor(%input: tensor<1x32x23x30x!qElemType>) -> tensor<1x64x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-4.480000e+02> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<4.480000e+02> : tensor<1xf16>
    %dq = IE.Dequantize(%input) {dstElemType = f16, low_fp_type = f8E4M3FN} : tensor<1x32x23x30x!qElemType> -> tensor<1x32x23x30xf16>

    %filter = const.Declare tensor<2x16x32x2x1xf8E4M3FN> = dense<1.000000e+00> : tensor<2x16x32x2x1xf8E4M3FN>
    %filter_scale = const.Declare tensor<1x1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1x1xf16>
    %filter_dq = IE.DynamicDequantize(%filter, %filter_scale) {dstElemType = f16} :
            tensor<2x16x32x2x1xf8E4M3FN>, tensor<1x1x1x1x1xf16> -> tensor<2x16x32x2x1xf16>

    %output = IE.GroupConvolutionBackpropData(%dq, %filter_dq) {
        dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x32x23x30xf16>, tensor<2x16x32x2x1xf16> -> tensor<1x64x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    return %output_fq : tensor<1x64x46x59xf16>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<2x32x16x2x1xf8E4M3FN> = dense<1.000000e+00> : tensor<2x16x32x2x1xf8E4M3FN>, [#const.Reverse<2 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:    [[FILTER_SCALE:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1x1xf16>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>

    // CHECK:    [[DQ:%.+]] = IE.Dequantize([[INPUT]])
    // CHECK-SAME:  {dstElemType = f16, low_fp_type = f8E4M3FN} : tensor<1x32x23x30x!qElemType> -> tensor<1x32x23x30xf16>
    // CHECK:    [[FILTER_DQ:%.+]] = IE.DynamicDequantize([[FILTER]], [[FILTER_SCALE]]) {dstElemType = f16} :
    // CHECK-SAME:      tensor<2x32x16x2x1xf8E4M3FN>, tensor<1x1x1x1x1xf16> -> tensor<2x32x16x2x1xf16>

    // CHECK:    [[CONV:%.+]] = IE.GroupTransposedConvolution([[DQ]], [[FILTER_DQ]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x32x23x30xf16>, tensor<2x32x16x2x1xf16> -> tensor<1x64x46x59xf16>
    // CHECK:    [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW:%.+]], [[HIGH:%.+]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    // CHECK:    return [[OUTPUT_FQ]] : tensor<1x64x46x59xf16>
}

// -----
!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK:  #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3, d4)>

// CHECK-LABEL: @ConvertDynDeqGroupConvBackpropDataToGroupTransposedConvF8E4M3FNPerAxis
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x23x30x!qElemType>
func.func @ConvertDynDeqGroupConvBackpropDataToGroupTransposedConvF8E4M3FNPerAxis(%input: tensor<1x32x23x30x!qElemType>) -> tensor<1x64x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-4.480000e+02> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<4.480000e+02> : tensor<1xf16>
    %dq = IE.Dequantize(%input) {dstElemType = f16, low_fp_type = f8E4M3FN} : tensor<1x32x23x30x!qElemType> -> tensor<1x32x23x30xf16>

    %filter = const.Declare tensor<2x16x32x2x1xf8E4M3FN> = dense<1.000000e+00> : tensor<2x16x32x2x1xf8E4M3FN>
    %filter_scale = const.Declare tensor<1x1x32x1x1xf16> = dense<[[[[[1.0]],[[2.0]],[[3.0]],[[4.0]],[[5.0]],[[6.0]],[[7.0]],[[8.0]],[[9.0]],[[10.0]],[[11.0]],[[12.0]],[[13.0]],[[14.0]],[[15.0]],[[16.0]],[[17.0]],[[18.0]],[[19.0]],[[20.0]],[[21.0]],[[22.0]],[[23.0]],[[24.0]],[[25.0]],[[26.0]],[[27.0]],[[28.0]],[[29.0]],[[30.0]],[[31.0]],[[32.0]]]]]> : tensor<1x1x32x1x1xf16>
    %filter_dq = IE.DynamicDequantize(%filter, %filter_scale) {dstElemType = f16} :
            tensor<2x16x32x2x1xf8E4M3FN>, tensor<1x1x32x1x1xf16> -> tensor<2x16x32x2x1xf16>

    %output = IE.GroupConvolutionBackpropData(%dq, %filter_dq) {
        dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x32x23x30xf16>, tensor<2x16x32x2x1xf16> -> tensor<1x64x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    return %output_fq : tensor<1x64x46x59xf16>

    // CHECK: [[FILTER_SCALE:%.+]] = const.Declare tensor<1x32x1x1x1xf16>
    // CHECK-SAME{LITERAL}: dense<[[[[[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]], [[7.000000e+00]], [[8.000000e+00]], [[9.000000e+00]], [[1.000000e+01]], [[1.100000e+01]], [[1.200000e+01]], [[1.300000e+01]], [[1.400000e+01]], [[1.500000e+01]], [[1.600000e+01]], [[1.700000e+01]], [[1.800000e+01]], [[1.900000e+01]], [[2.000000e+01]], [[2.100000e+01]], [[2.200000e+01]], [[2.300000e+01]], [[2.400000e+01]], [[2.500000e+01]], [[2.600000e+01]], [[2.700000e+01]], [[2.800000e+01]], [[2.900000e+01]], [[3.000000e+01]], [[3.100000e+01]], [[3.200000e+01]]]]]> : tensor<1x1x32x1x1xf16>
    // CHECK-SAME: #const.Transpose<#map>

    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
    // CHECK: [[FILTER:%.+]] = const.Declare tensor<2x32x16x2x1xf8E4M3FN> = dense<1.000000e+00> : tensor<2x16x32x2x1xf8E4M3FN>, [#const.Reverse<2 : i64>, #const.Transpose<#map>]

    // CHECK:    [[DQ:%.+]] = IE.Dequantize([[INPUT]])
    // CHECK-SAME:  {dstElemType = f16, low_fp_type = f8E4M3FN} : tensor<1x32x23x30x!qElemType> -> tensor<1x32x23x30xf16>
    // CHECK:    [[FILTER_DQ:%.+]] = IE.DynamicDequantize([[FILTER]], [[FILTER_SCALE]]) {dstElemType = f16} :
    // CHECK-SAME:      tensor<2x32x16x2x1xf8E4M3FN>, tensor<1x32x1x1x1xf16> -> tensor<2x32x16x2x1xf16>

    // CHECK:    [[CONV:%.+]] = IE.GroupTransposedConvolution([[DQ]], [[FILTER_DQ]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x32x23x30xf16>, tensor<2x32x16x2x1xf16> -> tensor<1x64x46x59xf16>
    // CHECK:    [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW:%.+]], [[HIGH:%.+]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x64x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x64x46x59xf16>

    // CHECK:    return [[OUTPUT_FQ]] : tensor<1x64x46x59xf16>
}

// -----

// CHECK-LABEL: @ConvertDynDeqConvertedConvBackpropDataToTransposedConvF8E5M2
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x23x30xf16>)
func.func @ConvertDynDeqConvertedConvBackpropDataToTransposedConvF8E5M2(%input: tensor<1x16x23x30xf16>) -> tensor<1x32x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>
    %fq = IE.FakeQuantize(%input, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x16x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x16x23x30xf16>

    %filter = const.Declare tensor<16x32x2x1xf8E5M2> = dense<1.000000e+00> : tensor<16x32x2x1xf8E5M2>
    %filter_scale = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %filter_dq = IE.DynamicDequantize(%filter, %filter_scale) {dstElemType = f16} :
            tensor<16x32x2x1xf8E5M2>, tensor<1x1x1x1xf16> -> tensor<16x32x2x1xf16>

    %filter_convert = IE.Convert(%filter_dq) {dstElemType = f32} : tensor<16x32x2x1xf16> -> tensor<16x32x2x1xf32>

    %output = IE.ConvolutionBackpropData(%fq, %filter_convert) {
        dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x16x23x30xf16>, tensor<16x32x2x1xf32> -> tensor<1x32x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    return %output_fq : tensor<1x32x46x59xf16>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<32x16x2x1xf8E5M2> = dense<1.000000e+00> : tensor<16x32x2x1xf8E5M2>, [#const.Reverse<1 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:    [[FILTER_SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>

    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x16x23x30xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x16x23x30xf16>
    // CHECK:    [[FILTER_DQ:%.+]] = IE.DynamicDequantize([[FILTER]], [[FILTER_SCALE]]) {dstElemType = f16} :
    // CHECK-SAME:      tensor<32x16x2x1xf8E5M2>, tensor<1x1x1x1xf16> -> tensor<32x16x2x1xf16>
    // CHECK:    [[CONVERT:%.+]] = IE.Convert([[FILTER_DQ]]) {dstElemType = f32} : tensor<32x16x2x1xf16> -> tensor<32x16x2x1xf32>

    // CHECK:    [[CONV:%.+]] = IE.TransposedConvolution([[FQ]], [[CONVERT]])
    // CHECK-SAME:  {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x16x23x30xf16>, tensor<32x16x2x1xf32> -> tensor<1x32x46x59xf16>
    // CHECK:    [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW]], [[HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    // CHECK:    return [[OUTPUT_FQ]] : tensor<1x32x46x59xf16>
}

// -----
!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @ConvertDynDeqConvertedConvBackpropDataToTransposedConvF8E5M2PerTensor
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x23x30x!qElemType>)
func.func @ConvertDynDeqConvertedConvBackpropDataToTransposedConvF8E5M2PerTensor(%input: tensor<1x16x23x30x!qElemType>) -> tensor<1x32x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>
    %dq = IE.Dequantize(%input) {dstElemType = f16, low_fp_type = f8E5M2} : tensor<1x16x23x30x!qElemType> -> tensor<1x16x23x30xf16>

    %filter = const.Declare tensor<16x32x2x1xf8E5M2> = dense<1.000000e+00> : tensor<16x32x2x1xf8E5M2>
    %filter_scale = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %filter_dq = IE.DynamicDequantize(%filter, %filter_scale) {dstElemType = f16} :
            tensor<16x32x2x1xf8E5M2>, tensor<1x1x1x1xf16> -> tensor<16x32x2x1xf16>

    %filter_convert = IE.Convert(%filter_dq) {dstElemType = f32} : tensor<16x32x2x1xf16> -> tensor<16x32x2x1xf32>

    %output = IE.ConvolutionBackpropData(%dq, %filter_convert) {
        dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x16x23x30xf16>, tensor<16x32x2x1xf32> -> tensor<1x32x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    return %output_fq : tensor<1x32x46x59xf16>

    // CHECK-DAG:    [[FILTER:%.+]] = const.Declare tensor<32x16x2x1xf8E5M2> = dense<1.000000e+00> : tensor<16x32x2x1xf8E5M2>, [#const.Reverse<1 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:    [[FILTER_SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>

    // CHECK:    [[DQ:%.+]] = IE.Dequantize([[INPUT]])
    // CHECK-SAME:  {dstElemType = f16, low_fp_type = f8E5M2} : tensor<1x16x23x30x!qElemType> -> tensor<1x16x23x30xf16>
    // CHECK:    [[FILTER_DQ:%.+]] = IE.DynamicDequantize([[FILTER]], [[FILTER_SCALE]]) {dstElemType = f16} :
    // CHECK-SAME:      tensor<32x16x2x1xf8E5M2>, tensor<1x1x1x1xf16> -> tensor<32x16x2x1xf16>
    // CHECK:    [[CONVERT:%.+]] = IE.Convert([[FILTER_DQ]]) {dstElemType = f32} : tensor<32x16x2x1xf16> -> tensor<32x16x2x1xf32>

    // CHECK:    [[CONV:%.+]] = IE.TransposedConvolution([[DQ]], [[CONVERT]])
    // CHECK-SAME:  {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x16x23x30xf16>, tensor<32x16x2x1xf32> -> tensor<1x32x46x59xf16>
    // CHECK:    [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW]], [[HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    // CHECK:    return [[OUTPUT_FQ]] : tensor<1x32x46x59xf16>
}

// -----
!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @ConvertDynDeqConvertedConvBackpropDataToTransposedConvF8E5M2PerAxis
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x23x30x!qElemType>)
func.func @ConvertDynDeqConvertedConvBackpropDataToTransposedConvF8E5M2PerAxis(%input: tensor<1x16x23x30x!qElemType>) -> tensor<1x32x46x59xf16> {
    %low = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    %high = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>
    %dq = IE.Dequantize(%input) {dstElemType = f16, low_fp_type = f8E5M2} : tensor<1x16x23x30x!qElemType> -> tensor<1x16x23x30xf16>

    %filter = const.Declare tensor<16x32x2x1xf8E5M2> = dense<1.000000e+00> : tensor<16x32x2x1xf8E5M2>
    %filter_scale = const.Declare tensor<1x32x1x1xf16> = dense<[[[[1.0]],[[2.0]],[[3.0]],[[4.0]],[[5.0]],[[6.0]],[[7.0]],[[8.0]],[[9.0]],[[10.0]],[[11.0]],[[12.0]],[[13.0]],[[14.0]],[[15.0]],[[16.0]],[[17.0]],[[18.0]],[[19.0]],[[20.0]],[[21.0]],[[22.0]],[[23.0]],[[24.0]],[[25.0]],[[26.0]],[[27.0]],[[28.0]],[[29.0]],[[30.0]],[[31.0]],[[32.0]]]]> : tensor<1x32x1x1xf16>
    %filter_dq = IE.DynamicDequantize(%filter, %filter_scale) {dstElemType = f16} :
            tensor<16x32x2x1xf8E5M2>, tensor<1x32x1x1xf16> -> tensor<16x32x2x1xf16>

    %filter_convert = IE.Convert(%filter_dq) {dstElemType = f32} : tensor<16x32x2x1xf16> -> tensor<16x32x2x1xf32>

    %output = IE.ConvolutionBackpropData(%dq, %filter_convert) {
        dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x16x23x30xf16>, tensor<16x32x2x1xf32> -> tensor<1x32x46x59xf16>

    %output_low = const.Declare tensor<1xf16> = dense<-1.0> : tensor<1xf16>
    %output_high = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
    %output_fq = IE.FakeQuantize(%output, %low, %high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x32x46x59xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x32x46x59xf16>

    return %output_fq : tensor<1x32x46x59xf16>

    // CHECK: [[FILTER_SCALE:%.+]] = const.Declare tensor<32x1x1x1xf16>
    // CHECK-SAME{LITERAL}: dense<[[[[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]], [[7.000000e+00]], [[8.000000e+00]], [[9.000000e+00]], [[1.000000e+01]], [[1.100000e+01]], [[1.200000e+01]], [[1.300000e+01]], [[1.400000e+01]], [[1.500000e+01]], [[1.600000e+01]], [[1.700000e+01]], [[1.800000e+01]], [[1.900000e+01]], [[2.000000e+01]], [[2.100000e+01]], [[2.200000e+01]], [[2.300000e+01]], [[2.400000e+01]], [[2.500000e+01]], [[2.600000e+01]], [[2.700000e+01]], [[2.800000e+01]], [[2.900000e+01]], [[3.000000e+01]], [[3.100000e+01]], [[3.200000e+01]]]]> : tensor<1x32x1x1xf16>
    // CHECK-SAME: #const.Transpose<#map>

    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.000000e+00> : tensor<1xf16>
    // CHECK: [[FILTER:%.+]] = const.Declare tensor<32x16x2x1xf8E5M2> = dense<1.000000e+00> : tensor<16x32x2x1xf8E5M2>, [#const.Reverse<1 : i64>, #const.Transpose<#map>]
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-5.734400e+04> : tensor<1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<5.734400e+04> : tensor<1xf16>

    // CHECK:    [[DQ:%.+]] = IE.Dequantize([[INPUT]])
    // CHECK-SAME:  {dstElemType = f16, low_fp_type = f8E5M2} : tensor<1x16x23x30x!qElemType> -> tensor<1x16x23x30xf16>
    // CHECK:    [[FILTER_DQ:%.+]] = IE.DynamicDequantize([[FILTER]], [[FILTER_SCALE]]) {dstElemType = f16} :
    // CHECK-SAME:      tensor<32x16x2x1xf8E5M2>, tensor<32x1x1x1xf16> -> tensor<32x16x2x1xf16>
    // CHECK:    [[CONVERT:%.+]] = IE.Convert([[FILTER_DQ]]) {dstElemType = f32} : tensor<32x16x2x1xf16> -> tensor<32x16x2x1xf32>

    // CHECK:    [[CONV:%.+]] = IE.TransposedConvolution([[DQ]], [[CONVERT]])
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

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// Note the padding positions hold the raw zero-point value (5), not 0, in each split filter below.

// CHECK-LABEL: @LegalizeConvBackpropDataTo3x3SplitConvPerTensorZeroPoint
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x64x65xf32>)
func.func @LegalizeConvBackpropDataTo3x3SplitConvPerTensorZeroPoint(%arg0: tensor<1x1x64x65xf32>) -> tensor<1x1x128x130xf32> {
    %cst = const.Declare tensor<1x1x4x4xui8> = dense<[[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]> : tensor<1x1x4x4xui8>
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %zp = const.Declare tensor<1x1x1x1xui8> = dense<5> : tensor<1x1x1x1xui8>
    %filter = IE.DynamicDequantize(%cst, %scale, %zp) {dstElemType = f32} : tensor<1x1x4x4xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8> -> tensor<1x1x4x4xf32>
    %0 = IE.ConvolutionBackpropData(%arg0, %filter) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x1x64x65xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x128x130xf32>
    return %0 : tensor<1x1x128x130xf32>

    // CHECK-DAG: [[SPLIT_FILTER_4:%.+]] = const.Declare tensor<1x1x3x3xui8>
    // CHECK-SAME{LITERAL}: dense<[[[[5, 5, 5], [5, 11, 9], [5, 3, 1]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<ui8>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_3:%.+]] = const.Declare tensor<1x1x3x3xui8>
    // CHECK-SAME{LITERAL}: dense<[[[[5, 5, 5], [12, 10, 5], [4, 2, 5]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<ui8>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_2:%.+]] = const.Declare tensor<1x1x3x3xui8>
    // CHECK-SAME{LITERAL}: dense<[[[[5, 15, 13], [5, 7, 5], [5, 5, 5]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<ui8>, #const.Transpose<#map>

    // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1x1x1x1xui8> = dense<5> : tensor<1x1x1x1xui8>

    // CHECK-DAG: [[SPLIT_FILTER_1:%.+]] = const.Declare tensor<1x1x3x3xui8>
    // CHECK-SAME{LITERAL}: dense<[[[[16, 14, 5], [8, 6, 5], [5, 5, 5]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<ui8>, #const.Transpose<#map>

    // CHECK: [[DYNDEQ0:%.+]] = IE.DynamicDequantize([[SPLIT_FILTER_1]], [[SCALE]], [[ZP]]) {dstElemType = f32} : tensor<1x1x3x3xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8> -> tensor<1x1x3x3xf32>
    // CHECK: [[DYNDEQ1:%.+]] = IE.DynamicDequantize([[SPLIT_FILTER_2]], [[SCALE]], [[ZP]]) {dstElemType = f32} : tensor<1x1x3x3xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8> -> tensor<1x1x3x3xf32>
    // CHECK: [[DYNDEQ2:%.+]] = IE.DynamicDequantize([[SPLIT_FILTER_3]], [[SCALE]], [[ZP]]) {dstElemType = f32} : tensor<1x1x3x3xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8> -> tensor<1x1x3x3xf32>
    // CHECK: [[DYNDEQ3:%.+]] = IE.DynamicDequantize([[SPLIT_FILTER_4]], [[SCALE]], [[ZP]]) {dstElemType = f32} : tensor<1x1x3x3xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8> -> tensor<1x1x3x3xf32>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT]], [[DYNDEQ0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<1x1x3x3xf32> -> tensor<1x1x64x65xf32>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT]], [[DYNDEQ1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<1x1x3x3xf32> -> tensor<1x1x64x65xf32>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[INPUT]], [[DYNDEQ2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<1x1x3x3xf32> -> tensor<1x1x64x65xf32>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[INPUT]], [[DYNDEQ3]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<1x1x3x3xf32> -> tensor<1x1x64x65xf32>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]}
    // CHECK:               tensor<1x1x64x65xf32>, tensor<1x1x64x65xf32>, tensor<1x1x64x65xf32>, tensor<1x1x64x65xf32> -> tensor<1x4x64x65xf32>
    // CHECK: [[D2S:%.+]] = IE.DepthToSpace([[CONCAT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x4x64x65xf32> -> tensor<1x1x128x130xf32>

    // CHECK: return [[D2S]] : tensor<1x1x128x130xf32>
  }

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @LegalizeConvBackpropDataTo3x3SplitConvPerChannelScaleNoZp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x64x65xf32>)
func.func @LegalizeConvBackpropDataTo3x3SplitConvPerChannelScaleNoZp(%arg0: tensor<1x1x64x65xf32>) -> tensor<1x2x128x130xf32> {
    %cst = const.Declare tensor<1x2x4x4xui8> = dense<[[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                                                        [[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]]]]> : tensor<1x2x4x4xui8>
    %scale = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.000000e+00]], [[3.000000e+00]]]]> : tensor<1x2x1x1xf32>
    %filter = IE.DynamicDequantize(%cst, %scale) {dstElemType = f32} : tensor<1x2x4x4xui8>, tensor<1x2x1x1xf32> -> tensor<1x2x4x4xf32>
    %0 = IE.ConvolutionBackpropData(%arg0, %filter) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x1x64x65xf32>, tensor<1x2x4x4xf32> -> tensor<1x2x128x130xf32>
    return %0 : tensor<1x2x128x130xf32>

    // CHECK-DAG: [[SPLIT_FILTER_4:%.+]] = const.Declare tensor<2x1x3x3xui8>
    // CHECK-SAME{LITERAL}: dense<[[[[0, 0, 0], [0, 11, 9], [0, 3, 1]], [[0, 0, 0], [0, 27, 25], [0, 19, 17]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<ui8>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_3:%.+]] = const.Declare tensor<2x1x3x3xui8>
    // CHECK-SAME{LITERAL}: dense<[[[[0, 0, 0], [12, 10, 0], [4, 2, 0]], [[0, 0, 0], [28, 26, 0], [20, 18, 0]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<ui8>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_2:%.+]] = const.Declare tensor<2x1x3x3xui8>
    // CHECK-SAME{LITERAL}: dense<[[[[0, 15, 13], [0, 7, 5], [0, 0, 0]], [[0, 31, 29], [0, 23, 21], [0, 0, 0]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<ui8>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_1:%.+]] = const.Declare tensor<2x1x3x3xui8>
    // CHECK-SAME{LITERAL}: dense<[[[[16, 14, 0], [8, 6, 0], [0, 0, 0]], [[32, 30, 0], [24, 22, 0], [0, 0, 0]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<ui8>, #const.Transpose<#map>

    // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<2x1x1x1xf32> =
    // CHECK-SAME{LITERAL}: dense<[[[[2.000000e+00]], [[3.000000e+00]]]]> : tensor<1x2x1x1xf32>

    // CHECK: [[DYNDEQ0:%.+]] = IE.DynamicDequantize([[SPLIT_FILTER_1]], [[SCALE]]) {dstElemType = f32} : tensor<2x1x3x3xui8>, tensor<2x1x1x1xf32> -> tensor<2x1x3x3xf32>
    // CHECK: [[DYNDEQ1:%.+]] = IE.DynamicDequantize([[SPLIT_FILTER_2]], [[SCALE]]) {dstElemType = f32} : tensor<2x1x3x3xui8>, tensor<2x1x1x1xf32> -> tensor<2x1x3x3xf32>
    // CHECK: [[DYNDEQ2:%.+]] = IE.DynamicDequantize([[SPLIT_FILTER_3]], [[SCALE]]) {dstElemType = f32} : tensor<2x1x3x3xui8>, tensor<2x1x1x1xf32> -> tensor<2x1x3x3xf32>
    // CHECK: [[DYNDEQ3:%.+]] = IE.DynamicDequantize([[SPLIT_FILTER_4]], [[SCALE]]) {dstElemType = f32} : tensor<2x1x3x3xui8>, tensor<2x1x1x1xf32> -> tensor<2x1x3x3xf32>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT]], [[DYNDEQ0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<2x1x3x3xf32> -> tensor<1x2x64x65xf32>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT]], [[DYNDEQ1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<2x1x3x3xf32> -> tensor<1x2x64x65xf32>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[INPUT]], [[DYNDEQ2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<2x1x3x3xf32> -> tensor<1x2x64x65xf32>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[INPUT]], [[DYNDEQ3]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<2x1x3x3xf32> -> tensor<1x2x64x65xf32>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}
    // CHECK:               tensor<1x2x64x65xf32>, tensor<1x2x64x65xf32>, tensor<1x2x64x65xf32>, tensor<1x2x64x65xf32> -> tensor<1x8x64x65xf32>
    // CHECK: [[D2S:%.+]] = IE.DepthToSpace([[CONCAT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x8x64x65xf32> -> tensor<1x2x128x130xf32>

    // CHECK: return [[D2S]] : tensor<1x2x128x130xf32>
  }

// -----

// CHECK-LABEL: @LegalizeConvBackpropDataTo2x2SplitConvPerChannelScaleAndZp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x64x65xf32>)
func.func @LegalizeConvBackpropDataTo2x2SplitConvPerChannelScaleAndZp(%arg0: tensor<1x1x64x65xf32>) -> tensor<1x2x128x130xf32> {
    %cst = const.Declare tensor<1x2x4x4xui8> = dense<[[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                                                        [[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]]]]> : tensor<1x2x4x4xui8>
    %scale = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.000000e+00]], [[3.000000e+00]]]]> : tensor<1x2x1x1xf32>
    %zp = const.Declare tensor<1x2x1x1xui8> = dense<[[[[4]], [[6]]]]> : tensor<1x2x1x1xui8>
    %filter = IE.DynamicDequantize(%cst, %scale, %zp) {dstElemType = f32} : tensor<1x2x4x4xui8>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xui8> -> tensor<1x2x4x4xf32>
    %0 = IE.ConvolutionBackpropData(%arg0, %filter) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x1x64x65xf32>, tensor<1x2x4x4xf32> -> tensor<1x2x128x130xf32>
    return %0 : tensor<1x2x128x130xf32>

    // CHECK-DAG: [[SPLIT_FILTER_4:%.+]] = const.Declare tensor<2x1x2x2xui8>
    // CHECK-SAME{LITERAL}: dense<[[[[11, 9], [3, 1]], [[27, 25], [19, 17]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<ui8>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_3:%.+]] = const.Declare tensor<2x1x2x2xui8>
    // CHECK-SAME{LITERAL}: dense<[[[[12, 10], [4, 2]], [[28, 26], [20, 18]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<ui8>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_2:%.+]] = const.Declare tensor<2x1x2x2xui8>
    // CHECK-SAME{LITERAL}: dense<[[[[15, 13], [7, 5]], [[31, 29], [23, 21]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<ui8>, #const.Transpose<#map>
    // CHECK-DAG: [[SPLIT_FILTER_1:%.+]] = const.Declare tensor<2x1x2x2xui8>
    // CHECK-SAME{LITERAL}: dense<[[[[16, 14], [8, 6]], [[32, 30], [24, 22]]]]>
    // CHECK-SAME{LITERAL}: #const.CastElemType<ui8>, #const.Transpose<#map>

    // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<2x1x1x1xf32> =
    // CHECK-SAME{LITERAL}: dense<[[[[2.000000e+00]], [[3.000000e+00]]]]> : tensor<1x2x1x1xf32>
    // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<2x1x1x1xui8> =
    // CHECK-SAME{LITERAL}: dense<[[[[4]], [[6]]]]> : tensor<1x2x1x1xui8>

    // CHECK: [[DYNDEQ0:%.+]] = IE.DynamicDequantize([[SPLIT_FILTER_1]], [[SCALE]], [[ZP]]) {dstElemType = f32} : tensor<2x1x2x2xui8>, tensor<2x1x1x1xf32>, tensor<2x1x1x1xui8> -> tensor<2x1x2x2xf32>
    // CHECK: [[DYNDEQ1:%.+]] = IE.DynamicDequantize([[SPLIT_FILTER_2]], [[SCALE]], [[ZP]]) {dstElemType = f32} : tensor<2x1x2x2xui8>, tensor<2x1x1x1xf32>, tensor<2x1x1x1xui8> -> tensor<2x1x2x2xf32>
    // CHECK: [[DYNDEQ2:%.+]] = IE.DynamicDequantize([[SPLIT_FILTER_3]], [[SCALE]], [[ZP]]) {dstElemType = f32} : tensor<2x1x2x2xui8>, tensor<2x1x1x1xf32>, tensor<2x1x1x1xui8> -> tensor<2x1x2x2xf32>
    // CHECK: [[DYNDEQ3:%.+]] = IE.DynamicDequantize([[SPLIT_FILTER_4]], [[SCALE]], [[ZP]]) {dstElemType = f32} : tensor<2x1x2x2xui8>, tensor<2x1x1x1xf32>, tensor<2x1x1x1xui8> -> tensor<2x1x2x2xf32>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT]], [[DYNDEQ0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<2x1x2x2xf32> -> tensor<1x2x64x65xf32>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT]], [[DYNDEQ1]]) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<2x1x2x2xf32> -> tensor<1x2x64x65xf32>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[INPUT]], [[DYNDEQ2]]) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [1, 0], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<2x1x2x2xf32> -> tensor<1x2x64x65xf32>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[INPUT]], [[DYNDEQ3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x64x65xf32>, tensor<2x1x2x2xf32> -> tensor<1x2x64x65xf32>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}
    // CHECK:               tensor<1x2x64x65xf32>, tensor<1x2x64x65xf32>, tensor<1x2x64x65xf32>, tensor<1x2x64x65xf32> -> tensor<1x8x64x65xf32>
    // CHECK: [[D2S:%.+]] = IE.DepthToSpace([[CONCAT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x8x64x65xf32> -> tensor<1x2x128x130xf32>

    // CHECK: return [[D2S]] : tensor<1x2x128x130xf32>
  }
