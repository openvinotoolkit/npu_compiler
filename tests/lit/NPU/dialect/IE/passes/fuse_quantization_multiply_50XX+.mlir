//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-fq-and-mul %s | FileCheck %s
// REQUIRES: platform-NPU5010
// COM: F8 is only supported on NPU50+, no need to run these tests on all platforms.

// CHECK-LABEL: @FuseFakeQuantizeAndMultiplyF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x288x20x20xf32>
func.func @FuseFakeQuantizeAndMultiplyF8E4M3FN(%input: tensor<1x288x20x20xf32>) -> tensor<1x288x20x20xf32> {
    %weights = const.Declare tensor<288x16x3x3xf32> = dense<1.0> : tensor<288x16x3x3xf32>
    %in_low = const.Declare tensor<1x1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1x1xf32>
    %in_high = const.Declare tensor<1x1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1x1xf32>
    %out_low = const.Declare tensor<288x1x1x1xf32> = dense<-1.270000e+02> : tensor<288x1x1x1xf32>
    %out_high = const.Declare tensor<288x1x1x1xf32> = dense<1.270000e+02> : tensor<288x1x1x1xf32>

    %0 = IE.FakeQuantize(%weights, %in_low, %in_high, %out_low, %out_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<288x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32> -> tensor<288x16x3x3xf32>

    %scale = const.Declare tensor<288x1x1x1xf32> = dense<2.0> : tensor<288x1x1x1xf32>
    %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<288x16x3x3xf32>, tensor<288x1x1x1xf32> -> tensor<288x16x3x3xf32>

    %2 = IE.Reshape(%1) {shape_value = [18, 16, 16, 3, 3]} : tensor<288x16x3x3xf32> -> tensor<18x16x16x3x3xf32>

    %3 = IE.GroupConvolution(%input, %2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    return %3 : tensor<1x288x20x20xf32>

    //CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<288x16x3x3xf32> = dense<1.000000e+00> : tensor<288x16x3x3xf32>
    //CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-4.480000e+02> : tensor<1x1x1x1xf32>
    //CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<4.480000e+02> : tensor<1x1x1x1xf32>
    //CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<288x1x1x1xf32> = dense<-1.270000e+02> : tensor<288x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    //CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<288x1x1x1xf32> = dense<1.270000e+02> : tensor<288x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]

    //CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<288x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32> -> tensor<288x16x3x3xf32>

    //CHECK:    [[RESHAPE:%.+]] = IE.Reshape([[FQ]]) {shape_value = [18, 16, 16, 3, 3]}
    //CHECK:    [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[RESHAPE]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    //CHECK:    return [[CONV]] : tensor<1x288x20x20xf32>
}
