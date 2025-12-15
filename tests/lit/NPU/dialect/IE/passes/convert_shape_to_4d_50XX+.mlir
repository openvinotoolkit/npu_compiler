//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-shape-to-4d --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: func.func @FakeQuantizePerChannelF8E4M3FN(
// CHECK-SAME:   [[INPUT:%.+]]: tensor<2x3x4x512x64xf32>
func.func @FakeQuantizePerChannelF8E4M3FN(%input: tensor<2x3x4x512x64xf32>) -> (tensor<2x3x4x512x64xf32>) {
    %input_low = const.Declare tensor<f32> = dense<-4.480000e+02> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<4.480000e+02> : tensor<f32>
    %output_low = const.Declare tensor<1x1x1x512x1xf32> = dense<-1.0> : tensor<1x1x1x512x1xf32>
    %output_high = const.Declare tensor<1x1x1x512x1xf32> = dense<1.0> : tensor<1x1x1x512x1xf32>

    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<2x3x4x512x64xf32>, tensor<f32>, tensor<f32>, tensor<1x1x1x512x1xf32>, tensor<1x1x1x512x1xf32> -> tensor<2x3x4x512x64xf32>

    return %0 : tensor<2x3x4x512x64xf32>

    // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-4.480000e+02>
    // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<4.480000e+02>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1x1x512x1xf32> = dense<-1.000000e+00>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x512x1xf32> = dense<1.000000e+00>

    // CHECK:    [[RESHAPE_0:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 24, 512, 64]} : tensor<2x3x4x512x64xf32> -> tensor<1x24x512x64xf32>
    // CHECK:    [[FQ_0:%.+]] = IE.FakeQuantize([[RESHAPE_0]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN}
    // CHECK-SAME: -> tensor<1x24x512x64xf32>
    // CHECK:    [[RESHAPE_1:%.+]] = IE.Reshape([[FQ_0]]) {shape_value = [2, 3, 4, 512, 64]} : tensor<1x24x512x64xf32> -> tensor<2x3x4x512x64xf32>

    //CHECK:    return [[RESHAPE_1]] : tensor<2x3x4x512x64xf32>
}

// -----

// CHECK-LABEL: func.func @FakeQuantizePerTensorF8E5M2(
// CHECK-SAME:   [[INPUT:%.+]]: tensor<512x64xf32>
func.func @FakeQuantizePerTensorF8E5M2(%input: tensor<512x64xf32>) -> (tensor<512x64xf32>) {
    %input_low = const.Declare tensor<f32> = dense<-5.734400e+04> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<5.734400e+04> : tensor<f32>
    %output_low = const.Declare tensor<f32> = dense<-1.0> : tensor<f32>
    %output_high = const.Declare tensor<f32> = dense<1.0> : tensor<f32>

    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<512x64xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<512x64xf32>

    return %0 : tensor<512x64xf32>

    // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-5.734400e+04>
    // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.734400e+04>
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00>
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00>

    // CHECK:    [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 512, 64]} : tensor<512x64xf32> -> tensor<1x1x512x64xf32>

    // CHECK:    [[FQ_0:%.+]] = IE.FakeQuantize([[RESHAPE_0]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2}
    // CHECK-SAME: -> tensor<1x1x512x64xf32>

    // CHECK:    [[RESHAPE_1:%.+]] = IE.AffineReshape([[FQ_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 64]} : tensor<1x1x512x64xf32> -> tensor<512x64xf32>

    // CHECK:    return [[RESHAPE_1]] : tensor<512x64xf32>
}

// -----

// CHECK-LABEL: @FakeConvertf8E4M3FN
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x80x300xf16>,
// CHECK-SAME:  [[SCALE:%.+]]: tensor<1xf16>
func.func @FakeConvertf8E4M3FN(%input: tensor<1x80x300xf16>, %scale: tensor<1xf16>) -> tensor<1x80x300xf16> {
    %0 = IE.FakeConvert(%input, %scale) {dst_type = f8E4M3FN} : tensor<1x80x300xf16>, tensor<1xf16> -> tensor<1x80x300xf16>
    return %0 : tensor<1x80x300xf16>

    // CHECK:    [[RESHAPE_SCL:%.+]] = IE.AffineReshape([[SCALE]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1xf16> -> tensor<1x1x1x1xf16>

    // CHECK:    [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 80, 300]} : tensor<1x80x300xf16> -> tensor<1x1x80x300xf16>

    // CHECK:    [[FC:%.+]] = IE.FakeConvert([[RESHAPE_IN]], [[RESHAPE_SCL]]) {dst_type = f8E4M3FN} : tensor<1x1x80x300xf16>, tensor<1x1x1x1xf16>
    // CHECK-SAME: -> tensor<1x1x80x300xf16>

    // CHECK:    [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[FC]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 80, 300]} : tensor<1x1x80x300xf16> -> tensor<1x80x300xf16>

    // CHECK:    return [[RESHAPE_OUT]] : tensor<1x80x300xf16>
}

// -----

// CHECK-LABEL: module @ReduceSumCompatibleWithNCE
module @ReduceSumCompatibleWithNCE {
    config.PipelineOptions @Options {
        config.Option @config.ReduceSupported : true
    }

    // CHECK:  ([[INPUT:%.+]]: tensor<1x512x768xf16>)
    func.func @main(%input: tensor<1x512x768xf16>) -> (tensor<1x768xf16>) {
        %0 = IE.ReduceSum(%input) {axes_value = [1]} : tensor<1x512x768xf16> -> tensor<1x768xf16>
        return %0 : tensor<1x768xf16>
    }

    // CHECK:                [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 512, 768, 1]} : tensor<1x512x768xf16> -> tensor<1x512x768x1xf16>
    // CHECK:                [[REDUCE:%.+]] = IE.ReduceSum([[RESHAPE_IN]]) {axes_value = [1], keep_dims} : tensor<1x512x768x1xf16> -> tensor<1x1x768x1xf16>
    // CHECK:                [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[REDUCE]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [1]], shape_value = [1, 768]} : tensor<1x1x768x1xf16> -> tensor<1x768xf16>
    // CHECK:                return [[RESHAPE_OUT]]
}
