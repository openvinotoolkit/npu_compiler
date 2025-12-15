//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-subtract-to-add --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @SubtractWithFQConstInputsSameShapeF8E4M3FN
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x1x1x64xf16>
func.func @SubtractWithFQConstInputsSameShapeF8E4M3FN(%input: tensor<1x1x1x64xf16>) -> tensor<1x1x1x64xf16> {
    %weights = const.Declare tensor<1x1x1x64xf16> = dense<5.000000e+00> : tensor<1x1x1x64xf16>
    %in_low = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    %in_high = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>
    %out_low_1 = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1x1x1xf16>
    %out_high_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %out_low_2 = const.Declare tensor<1x1x1x1xf16> = dense<-5.0> : tensor<1x1x1x1xf16>
    %out_high_2 = const.Declare tensor<1x1x1x1xf16> = dense<5.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%input, %in_low, %in_high, %out_low_1, %out_high_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    %1 = IE.FakeQuantize(%weights, %in_low, %in_high, %out_low_2, %out_high_2) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    %2 = IE.Subtract(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>

    return %2 : tensor<1x1x1x64xf16>

    // CHECK-NOT:    IE.Subtract

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<1x1x1x64xf16> = dense<5.000000e+00> : tensor<1x1x1x64xf16>, [#const.Rescale<-1.000000e+00 : f64>]
    // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[OUT_LOW_0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[OUT_HIGH_0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[OUT_LOW_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-5.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[OUT_HIGH_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e+00> : tensor<1x1x1x1xf16>

    // CHECK:    [[FQ_0:%.+]] = IE.FakeQuantize([[INPUT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_0]], [[OUT_HIGH_0]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    [[FQ_1:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_1]], [[OUT_HIGH_1]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>

    // CHECK:    [[ADD:%.+]] = IE.Add([[FQ_0]], [[FQ_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>

    // CHECK:    return [[ADD]] : tensor<1x1x1x64xf16>
}

// -----

// CHECK-LABEL: @SubtractWithFQActInputsSameShapeF8E5M2
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x1x1x64xf16>
func.func @SubtractWithFQActInputsSameShapeF8E5M2(%input: tensor<1x1x1x64xf16>) -> tensor<1x1x1x64xf16> {
    %weights = const.Declare tensor<1x1x1x64xf16> = dense<5.000000e+00> : tensor<1x1x1x64xf16>
    %low = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf16>

    %0 = IE.ReLU(%input) : tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    %1 = IE.ReLU(%weights) : tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>

    %2 = IE.FakeQuantize(%0, %low , %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    %3 = IE.FakeQuantize(%1, %low , %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    %4 = IE.Subtract(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>

    return %4 : tensor<1x1x1x64xf16>

    // CHECK-NOT:    IE.Subtract

    // CHECK-DAG:    [[FILTWE_AND_FILTER_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[FILTER_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<1x1x1x64xf16> = dense<5.000000e+00> : tensor<1x1x1x64xf16>
    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf16>

    // CHECK:    [[ACT_INPUT:%.+]] = IE.ReLU([[INPUT]]) : tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    [[ACT_WEIGHTS:%.+]] = IE.ReLU([[WEIGHTS]]) : tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>

    // CHECK:    [[FQ_0:%.+]] = IE.FakeQuantize([[ACT_INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    [[FQ_1:%.+]] = IE.FakeQuantize([[ACT_WEIGHTS]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    [[FQ_2:%.+]] = IE.FakeQuantize([[FILTWE_AND_FILTER_LOW]], [[FILTWE_AND_FILTER_LOW]], [[FILTER_HIGH]], [[FILTWE_AND_FILTER_LOW]], [[FILTER_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

    // CHECK:    [[CONV:%.+]] = IE.GroupConvolution([[FQ_1]], [[FQ_2]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>

    // CHECK:    [[FQ_3:%.+]] = IE.FakeQuantize([[CONV]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    [[ADD:%.+]] = IE.Add([[FQ_0]], [[FQ_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>

    // CHECK:    return [[ADD]] : tensor<1x1x1x64xf16>
}
