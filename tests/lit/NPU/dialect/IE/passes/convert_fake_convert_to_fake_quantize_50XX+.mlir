//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-fake-convert-to-fake-quantize --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK-LABEL: @ConvertFakeConvertToFakeQuantizeF8E3M4
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x80x3000xf16>
func.func @ConvertFakeConvertToFakeQuantizeF8E3M4(%input: tensor<1x80x3000xf16>) -> tensor<1x80x3000xf16> {
    %scale = const.Declare tensor<1xf16> = dense<2.000000e+00> : tensor<f32>, [#const.Reshape<[1]>, #const.CastElemType<f16>]
    %shift = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<f32>, [#const.Reshape<[1]>, #const.CastElemType<f16>]
    %0 = IE.FakeConvert(%input, %scale, %shift) {dst_type = f8E4M3FN} : tensor<1x80x3000xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x80x3000xf16>

    return %0 : tensor<1x80x3000xf16>

    // CHECK-DAG:   [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-2.230000e+02> : tensor<1xf16>
    // CHECK-DAG:   [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.250000e+02> : tensor<1xf16>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:                   low_fp_type = f8E4M3FN
    // CHECK-SAME:               } : tensor<1x80x3000xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x80x3000xf16>

    // CHECK:   return [[FQ]] : tensor<1x80x3000xf16>
}

// -----

// CHECK-LABEL: @ConvertFakeConvertToFakeQuantizeF8E2M5
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x80x3000xf16>
func.func @ConvertFakeConvertToFakeQuantizeF8E2M5(%input: tensor<1x80x3000xf16>) -> tensor<1x80x3000xf16> {
    %scale = const.Declare tensor<1xf16> = dense<2.000000e+00> : tensor<f32>, [#const.Reshape<[1]>, #const.CastElemType<f16>]
    %shift = const.Declare tensor<1xf16> = dense<10.000000e+00> : tensor<f32>, [#const.Reshape<[1]>, #const.CastElemType<f16>]
    %0 = IE.FakeConvert(%input, %scale, %shift) {dst_type = f8E5M2} : tensor<1x80x3000xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x80x3000xf16>

    return %0 : tensor<1x80x3000xf16>

    // CHECK-DAG:   [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-2.865600e+04> : tensor<1xf16>
    // CHECK-DAG:   [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.868800e+04> : tensor<1xf16>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:                   low_fp_type = f8E5M2
    // CHECK-SAME:               } : tensor<1x80x3000xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x80x3000xf16>

    // CHECK:   return [[FQ]] : tensor<1x80x3000xf16>
}

// -----

// CHECK-LABEL: @ConvertFakeConvertToFakeQuantizeNoShift
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x8x1500x64xf16>
func.func @ConvertFakeConvertToFakeQuantizeNoShift(%input: tensor<1x8x1500x64xf16>) -> tensor<1x8x1500x64xf16> {
    %scale = const.Declare tensor<1xf16> = dense<2.000000e+00> : tensor<1xf16>
    %0 = IE.FakeConvert(%input, %scale) {dst_type = f8E4M3FN} : tensor<1x8x1500x64xf16>, tensor<1xf16> ->  tensor<1x8x1500x64xf16>

    return %0 : tensor<1x8x1500x64xf16>

    // CHECK-DAG:   [[LOW:%.+]] = const.Declare tensor<1xf16> = dense<-2.240000e+02> : tensor<1xf16>
    // CHECK-DAG:   [[HIGH:%.+]] = const.Declare tensor<1xf16> = dense<2.240000e+02> : tensor<1xf16>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:                   low_fp_type = f8E4M3FN
    // CHECK-SAME:               } : tensor<1x8x1500x64xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1x8x1500x64xf16>

    // CHECK:   return [[FQ]] : tensor<1x8x1500x64xf16>
}

// -----

// CHECK-LABEL: @ConvertFakeConvertToFakeQuantizeWeightsAndShift
func.func @ConvertFakeConvertToFakeQuantizeWeightsAndShift() -> tensor<3x1x1x1xf16> {
    %input = const.Declare tensor<3x1x1x1xf16> = dense<[[[[-448.0]]], [[[0.0]]], [[[448.0]]]]> : tensor<3x1x1x1xf16>
    %scale = const.Declare tensor<1xf16> = dense<0.500000e+00> : tensor<1xf16>
    %shift = const.Declare tensor<3x1x1x1xf16> = dense<5.000000e+00> : tensor<3x1x1x1xf16>
    %0 = IE.FakeConvert(%input, %scale, %shift) {dst_type = f8E4M3FN} : tensor<3x1x1x1xf16>, tensor<1xf16>, tensor<3x1x1x1xf16> -> tensor<3x1x1x1xf16>

    return %0 : tensor<3x1x1x1xf16>

    // CHECK-DAG:   [[INPUT:%.+]] = const.Declare tensor<3x1x1x1xf16>
    // CHECK-DAG:   [[LOW:%.+]] = const.Declare tensor<3x1x1x1xf16> = dense<-8.910000e+02> : tensor<3x1x1x1xf16>
    // CHECK-DAG:   [[HIGH:%.+]] = const.Declare tensor<3x1x1x1xf16> = dense<9.010000e+02> : tensor<3x1x1x1xf16>

    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:                   low_fp_type = f8E4M3FN
    // CHECK-SAME:               } : tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16> -> tensor<3x1x1x1xf16>

    // CHECK:   return [[FQ]] : tensor<3x1x1x1xf16>
}

// -----

// CHECK-LABEL: @FakeConvertNoConvertToFakeQuantize
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x80x300xf16>,
// CHECK-SAME:  [[SCALE:%.+]]: tensor<1xf16>
func.func @FakeConvertNoConvertToFakeQuantize(%input: tensor<1x80x300xf16>, %scale: tensor<1xf16>) -> tensor<1x80x300xf16> {
    %0 = IE.FakeConvert(%input, %scale) {dst_type = f8E4M3FN} : tensor<1x80x300xf16>, tensor<1xf16> -> tensor<1x80x300xf16>
    return %0 : tensor<1x80x300xf16>

    // CHECK:       [[FC:%.+]] = IE.FakeConvert([[INPUT]], [[SCALE]]) {dst_type = f8E4M3FN}
    // CHECK-SAME:               : tensor<1x80x300xf16>, tensor<1xf16> -> tensor<1x80x300xf16>

    // CHECK:   return [[FC]] : tensor<1x80x300xf16>
}
