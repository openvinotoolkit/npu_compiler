//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-scale-shift-depthwise %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @ConvertScaleWithFakeQuantizeInputWithoutWeightsToDepthwiseF8E4M3FN
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x3x224x224xf16>
func.func @ConvertScaleWithFakeQuantizeInputWithoutWeightsToDepthwiseF8E4M3FN(%input: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
    %low = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>
    %bias = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>

    %0 = IE.FakeQuantize(%input, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x3x224x224xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x224x224xf16>

    %1 = IE.ScaleShift(%0, %bias) {
        operandSegmentSizes = array<i32: 1, 0, 1>
    } : tensor<1x3x224x224xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>

    return %1 : tensor<1x3x224x224xf16>

    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>

    // CHECK:    [[FQ_0:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x3x224x224xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x224x224xf16>
        // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<3x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 3 : i64>]
    // CHECK:    [[FQ_1:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[WEIGHTS]], [[WEIGHTS]], [[WEIGHTS]], [[WEIGHTS]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16> -> tensor<3x1x1x1xf16>
    // CHECK:    [[CONV:%.+]] = IE.GroupConvolution([[FQ_0]], [[FQ_1]], [[BIAS]])
    // CHECK-SAME:  {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x224x224xf16>, tensor<3x1x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>

    // CHECK:    return [[CONV]] : tensor<1x3x224x224xf16>
}
