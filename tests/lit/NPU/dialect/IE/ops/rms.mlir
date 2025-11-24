//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @FoldMulIntoRMS
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x32x64xf16>
func.func @FoldMulIntoRMS(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x32x64xf16> {
    %gamma = const.Declare tensor<64xf16> = dense<2.0> : tensor<64xf16>
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<3.0> : tensor<1x1x1x1xf16>

    %rms = IE.RMS(%arg0, %gamma) {epsilon = 9.9999997473787516E-6 : f64} : tensor<1x16x32x64xf16>, tensor<64xf16> -> tensor<1x16x32x64xf16>
    %mul = IE.Multiply(%rms, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x64xf16>

    return %mul : tensor<1x16x32x64xf16>

    // CHECK-DAG:   [[GAMMA:%.+]] = const.Declare tensor<64xf16> = dense<2.000000e+00> : tensor<64xf16>, [#const.Rescale<3.000000e+00 : f64>]
    // CHECK:       [[RMS:%.+]] = IE.RMS([[INPUT]], [[GAMMA]])
    // CHECK:       return [[RMS]]
}

// -----

// CHECK-LABEL: @RMSHasMultipleUses
func.func @RMSHasMultipleUses(%arg0: tensor<1x16x32x64xf16>) -> (tensor<1x16x32x64xf16>, tensor<1x16x32x64xf16>) {
    %gamma = const.Declare tensor<64xf16> = dense<2.0> : tensor<64xf16>
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<3.0> : tensor<1x1x1x1xf16>

    %rms = IE.RMS(%arg0, %gamma) {epsilon = 9.9999997473787516E-6 : f64} : tensor<1x16x32x64xf16>, tensor<64xf16> -> tensor<1x16x32x64xf16>
    %mul = IE.Multiply(%rms, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x64xf16>

    return %rms, %mul : tensor<1x16x32x64xf16>, tensor<1x16x32x64xf16>

    // CHECK:       [[RMS:%.+]] = IE.RMS
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[RMS]],
    // CHECK:       return [[RMS]], [[MUL]]
}

// -----

// CHECK-LABEL: @MulWithNonConstScale
func.func @MulWithNonConstScale(%arg0: tensor<1x16x32x64xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x16x32x64xf16> {
    %gamma = const.Declare tensor<64xf16> = dense<2.0> : tensor<64xf16>

    %rms = IE.RMS(%arg0, %gamma) {epsilon = 9.9999997473787516E-6 : f64} : tensor<1x16x32x64xf16>, tensor<64xf16> -> tensor<1x16x32x64xf16>
    %mul = IE.Multiply(%rms, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x64xf16>

    return %mul : tensor<1x16x32x64xf16>

    // CHECK:       [[RMS:%.+]] = IE.RMS
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[RMS]],
    // CHECK:       return [[MUL]]
}
