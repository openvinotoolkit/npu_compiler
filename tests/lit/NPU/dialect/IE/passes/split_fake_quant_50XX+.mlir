//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-fake-quant --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK:  !qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

// CHECK-LABEL: @SingleQuantParamsF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x3x30x30xf32>
func.func @SingleQuantParamsF8E4M3FN(%input: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%input, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK-NOT:    IE.FakeQuantize

    // CHECK:    [[QUAN:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x3x30x30xf32> -> tensor<1x3x30x30x!qElemType>
    // CHECK:    [[DEQUAN:%.+]] = IE.Dequantize([[QUAN]]) {dstElemType = f32} : tensor<1x3x30x30x!qElemType> -> tensor<1x3x30x30xf32>

    // CHECK:    return [[DEQUAN]] : tensor<1x3x30x30xf32>
}

// -----

// CHECK-LABEL: @DontSingleQuantParamsF8E4M3FNNonZeroZeroPoint
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x3x30x30xf32>
func.func @DontSingleQuantParamsF8E4M3FNNonZeroZeroPoint(%input: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%input, %low, %input_high, %low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[IN_HIGH]], [[LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    // CHECK:    return [[FQ]] : tensor<1x3x30x30xf32>
}

// -----

// CHECK:  !qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
// CHECK:  !qElemType1 = !quant.uniform<f8E5M2:f16:1, {7.812500e-03,1.562500e-02,0.006173270089285714,3.906250e-03}>

// CHECK-LABEL: @PerChannelQuantOutputF8E5M2
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x15x15xf16>
func.func @PerChannelQuantOutputF8E5M2(%input: tensor<1x4x15x15xf16>) -> tensor<1x4x15x15xf16> {
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x4x1x1xf16> = dense<[[[[-448.0]], [[-896.0]], [[-354.0]], [[-224.0]]]]> : tensor<1x4x1x1xf16>
    %output_high =  const.Declare tensor<1x4x1x1xf16> = dense<[[[[448.0]], [[896.0]], [[354.0]], [[224.0]]]]> : tensor<1x4x1x1xf16>

    %1 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x4x15x15xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x4x1x1xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x15x15xf16>

    return %1 : tensor<1x4x15x15xf16>

    // CHECK-NOT:    IE.FakeQuantize

    // CHECK:    [[QUAN:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x4x15x15xf16> -> tensor<1x4x15x15x!qElemType>
    // CHECK:    [[Q_CAST:%.+]] = IE.QuantizeCast([[QUAN]]) {dstElemType = !qElemType1} : tensor<1x4x15x15x!qElemType> -> tensor<1x4x15x15x!qElemType1>
    // CHECK:    [[DEQUAN:%.+]] = IE.Dequantize([[Q_CAST]]) {dstElemType = f16} : tensor<1x4x15x15x!qElemType1> -> tensor<1x4x15x15xf16>

    // CHECK:    return [[DEQUAN]] : tensor<1x4x15x15xf16>
}

// -----

// CHECK:  !qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>

// CHECK-LABEL: @ConstantsSplitFakeQuantF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x16x16xf16>
func.func @ConstantsSplitFakeQuantF8E4M3FN(%input: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %weights = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    %low = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%weights, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<16x16x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x16x1x1xf16>

    %1 = IE.Convolution(%input, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // CHECK-NOT:    IE.FakeQuantize

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<!qElemType>]

    // CHECK:    [[DEQUAN:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUAN]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    // CHECK:    return [[CONV]] : tensor<1x16x16x16xf16>
}
