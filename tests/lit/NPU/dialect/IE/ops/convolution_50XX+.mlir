//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @GroupsToAttrWithFQInputF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x96x96x96xf32>
func.func @GroupsToAttrWithFQInputF8E4M3FN(%input: tensor<1x96x96x96xf32>) -> tensor<1x96x96x96xf32> {
    %weights = const.Declare tensor<3x32x32x3x3xf32> = dense<1.0> : tensor<3x32x32x3x3xf32>
    %in_low = const.Declare tensor<1x1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1x1xf32>
    %in_high = const.Declare tensor<1x1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1x1xf32>
    %out_low_0 = const.Declare tensor<1x1x1x1xf32> = dense<-2.0> : tensor<1x1x1x1xf32>
    %out_high_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %out_low_1 = const.Declare tensor<3x32x1x1x1xf32> = dense<-0.40> : tensor<3x32x1x1x1xf32>
    %out_high_1 = const.Declare tensor<3x32x1x1x1xf32> = dense<0.40> : tensor<3x32x1x1x1xf32>

    %0 = IE.FakeQuantize(%input, %in_low, %in_high, %out_low_0, %out_high_0) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x96x96x96xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x96x96x96xf32>

    %1 = IE.FakeQuantize(%weights, %in_low, %in_high, %out_low_1, %out_high_1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<3x32x32x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<3x32x1x1x1xf32>, tensor<3x32x1x1x1xf32> -> tensor<3x32x32x3x3xf32>

    %2 = IE.GroupConvolution(%0, %1) {
        strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]
    } : tensor<1x96x96x96xf32>, tensor<3x32x32x3x3xf32> -> tensor<1x96x96x96xf32>

    return %2 : tensor<1x96x96x96xf32>

    // CHECK-DAG:    [[IN_LOW_RS:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1]>]
    // CHECK-DAG:    [[IN_HIGH_RS:%.+]] = const.Declare tensor<1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1]>]
    // CHECK-DAG:    [[OUT_LOW_RS:%.+]] = const.Declare tensor<96x1x1x1xf32> = dense<-4.000000e-01> : tensor<3x32x1x1x1xf32>, [#const.Reshape<[96, 1, 1, 1]>]
    // CHECK-DAG:    [[OUT_HIGH_RS:%.+]] = const.Declare tensor<96x1x1x1xf32> = dense<4.000000e-01> : tensor<3x32x1x1x1xf32>, [#const.Reshape<[96, 1, 1, 1]>]
    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<96x32x3x3xf32> = dense<1.000000e+00> : tensor<3x32x32x3x3xf32>, [#const.Reshape<[96, 32, 3, 3]>]
    // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK:    [[FQ_0:%.+]] = IE.FakeQuantize([[INPUT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x96x96x96xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x96x96x96xf32>

    // CHECK:    [[FQ_1:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW_RS]], [[IN_HIGH_RS]], [[OUT_LOW_RS]], [[OUT_HIGH_RS:%.+]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<96x32x3x3xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<96x1x1x1xf32>, tensor<96x1x1x1xf32> -> tensor<96x32x3x3xf32>

    // CHECK:    [[CONV:%.+]] = IE.GroupConvolution([[FQ_0]], [[FQ_1]])
    // CHECK-SAME:  {dilations = [1, 1], groups = 3 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x96x96x96xf32>, tensor<96x32x3x3xf32> -> tensor<1x96x96x96xf32>

    // CHECK:    return [[CONV]] : tensor<1x96x96x96xf32>
}
