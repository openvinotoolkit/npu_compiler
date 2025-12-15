//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-groupconv-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @ConvertQuantizedGroupConvToSingleConvF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]:  tensor<1x64x80x80xf16>
func.func @ConvertQuantizedGroupConvToSingleConvF8E4M3FN(%input: tensor<1x64x80x80xf16>) -> tensor<1x64x80x80xf16> {
    %weights = const.Declare tensor<64x16x3x3xf16> = dense<1.0> : tensor<64x16x3x3xf16>
    %weights_low = const.Declare tensor<64x1x1x1xf16> = dense<-4.480000e+02> : tensor<64x1x1x1xf16>
    %weights_high = const.Declare tensor<64x1x1x1xf16> = dense<4.480000e+02> : tensor<64x1x1x1xf16>
    %fq_weights = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                    low_fp_type = f8E4M3FN
                } : tensor<64x16x3x3xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16> -> tensor<64x16x3x3xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %result = IE.GroupConvolution(%input, %fq_weights, %bias) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    return %result : tensor<1x64x80x80xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[ORG_WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-DAG:   [[FQ_LOW:%.+]] = const.Declare tensor<64x1x1x1xf16> = dense<-4.480000e+02> : tensor<64x1x1x1xf16>
    // CHECK-DAG:   [[FQ_HIGH:%.+]] = const.Declare tensor<64x1x1x1xf16> = dense<4.480000e+02> : tensor<64x1x1x1xf16>
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
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<64x64x3x3xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16> -> tensor<64x64x3x3xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FQ]], [[BIAS]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x64x80x80xf16>, tensor<64x64x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @ConvertPerChannelGroupConvToMultiConvF8E5M2
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x16x16xf16>
func.func @ConvertPerChannelGroupConvToMultiConvF8E5M2(%input: tensor<1x2x16x16xf16>) -> tensor<1x32x16x16xf16> {
    %weights = const.Declare tensor<32x1x3x3xf16> = dense<1.0> : tensor<32x1x3x3xf16>
    %low = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf16>
    %fq_weights = IE.FakeQuantize(%weights, %low, %high, %low, %high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                    low_fp_type = f8E5M2
                } : tensor<32x1x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<32x1x3x3xf16>
    %input_out_low = const.Declare tensor<1x2x1x1xf16> = dense<[[[[-1.0, -1.1]]]]> : tensor<1x1x1x2xf16>, [#const.Reshape<[1, 2, 1, 1]>]
    %input_out_high = const.Declare tensor<1x2x1x1xf16> = dense<[[[[1.0, 1.1]]]]> : tensor<1x1x1x2xf16>, [#const.Reshape<[1, 2, 1, 1]>]
    %fq_input = IE.FakeQuantize(%input, %low, %high, %input_out_low, %input_out_high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                    low_fp_type = f8E5M2
                } : tensor<1x2x16x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x16x16xf16>
    %result = IE.GroupConvolution(%fq_input, %fq_weights) {dilations = [1, 1], groups = 2 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x2x16x16xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x16x16xf16>

    return %result : tensor<1x32x16x16xf16>

    // CHECK-NOT:   IE.GroupConvolution

    // CHECK-DAG:   [[ORG_WEIGHTS:%.+]] = const.Declare tensor<32x1x3x3xf16> = dense<1.000000e+00> : tensor<32x1x3x3xf16>
    // CHECK-DAG:   [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf16>
    // CHECK:       [[FQ_0:%.+]] = IE.FakeQuantize([[ORG_WEIGHTS]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:    {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<32x1x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<32x1x3x3xf16>

    // CHECK-DAG:   [[INPUT_OUT_LOW:%.+]] = const.Declare tensor<1x2x1x1xf16>
    // CHECK-DAG-SAME{LITERAL}:  = dense<[[[[-1.000000e+00, -1.099610e+00]]]]> : tensor<1x1x1x2xf16>, [#const.Reshape<[1, 2, 1, 1]>]
    // CHECK-DAG:   [[INPUT_OUT_HIGH:%.+]] = const.Declare tensor<1x2x1x1xf16>
    // CHECK-DAG-SAME{LITERAL}:  = dense<[[[[1.000000e+00, 1.099610e+00]]]]> : tensor<1x1x1x2xf16>, [#const.Reshape<[1, 2, 1, 1]>]
    // CHECK:       [[FQ_1:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[INPUT_OUT_LOW]], [[INPUT_OUT_HIGH]])
    //CHECK-SAME:     {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x2x16x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x16x16xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[FQ_1]] [0, 0, 0, 0] [1, 1, 16, 16] : tensor<1x2x16x16xf16> to tensor<1x1x16x16xf16>
    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<16x1x3x3xf16> = dense<1.000000e+00> : tensor<32x1x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [16, 1, 3, 3]>]
    // CHECK:       [[FQ_2:%.+]] = IE.FakeQuantize([[WEIGHTS_0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:    {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<16x1x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x1x3x3xf16>

    // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[SLICE_0]], [[FQ_2]])
    // CHECK-SAME:    {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x16x16xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x16x16xf16>

    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[FQ_1]] [0, 1, 0, 0] [1, 1, 16, 16] : tensor<1x2x16x16xf16> to tensor<1x1x16x16xf16>
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<16x1x3x3xf16> = dense<1.000000e+00> : tensor<32x1x3x3xf16>, [#const.SubView<[16, 0, 0, 0], [16, 1, 3, 3]>]
    // CHECK:       [[FQ_3:%.+]] = IE.FakeQuantize([[WEIGHTS_1]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:    {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<16x1x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x1x3x3xf16>

    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[SLICE_1]], [[FQ_3]])
    // CHECK-SAME:    {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x16x16xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x16x16xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[CONV_0]], [[CONV_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16> -> tensor<1x32x16x16xf16>

    // CHECK:       return [[CONCAT]] : tensor<1x32x16x16xf16>
}
