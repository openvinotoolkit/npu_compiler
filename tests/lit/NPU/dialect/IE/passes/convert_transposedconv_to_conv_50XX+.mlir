//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-transposed-conv-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @ConvertTransposedConv2DToConv2DFQFilterF8E4M3FN
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x32x23x30xf16>
func.func @ConvertTransposedConv2DToConv2DFQFilterF8E4M3FN(%input: tensor<1x32x23x30xf16>) -> tensor<1x16x46x60xf16> {
    %weights = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<16x32x2x2xf16>
    %low = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%weights, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<16x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x32x2x2xf16>

    %1 = IE.TransposedConvolution(%input, %0) {
        strides = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0]
    } : tensor<1x32x23x30xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>

    return %1 : tensor<1x16x46x60xf16>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<16x32x2x2xf16>
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>

    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<16x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x32x2x2xf16>

    // CHECK:    [[UPSAMPLING:%.+]] = IE.Upsampling([[INPUT]])
    // CHECK-SAME:  {pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [1, 1], pads_width = [1, 1]>, upsampling_factor = [2, 2, 1]} : tensor<1x32x23x30xf16> -> tensor<1x32x47x61xf16>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[UPSAMPLING]], [[FQ]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x47x61xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>

    // CHECK:    return [[CONV]] : tensor<1x16x46x60xf16>
}
