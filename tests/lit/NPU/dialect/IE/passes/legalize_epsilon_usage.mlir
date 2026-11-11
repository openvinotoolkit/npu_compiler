//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --legalize-epsilon-usage %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @NormalizeL2FixSmallEpsilon
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x40x64xf16>)
func.func @NormalizeL2FixSmallEpsilon(%arg0: tensor<1x1x40x64xf16>) -> tensor<1x1x40x64xf16> {
    %0 = IE.NormalizeL2(%arg0) {axes_value = [3], eps = 9.999999960041972E-13 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x1x40x64xf16> -> tensor<1x1x40x64xf16>
    return %0 : tensor<1x1x40x64xf16>
    // CHECK:  [[NORMALIZE:%.+]] = IE.NormalizeL2([[ARG0]]) {axes_value = [3], eps = 9.9999997171806853E-10 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x1x40x64xf16> -> tensor<1x1x40x64xf16>
    // CHECK: return [[NORMALIZE]] : tensor<1x1x40x64xf16>
}

// -----

// CHECK-LABEL: @RMSFixSmallEpsilon
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1024x1792xf16>)
func.func @RMSFixSmallEpsilon(%arg0: tensor<1x1024x1792xf16>) -> tensor<1x1024x1792xf16> {
    %cst = const.Declare tensor<1792xf16> = dense<1.000000e+00> : tensor<1792xf32>, [#const.CastElemType<f16>]
    %0 = IE.RMS(%arg0, %cst) {eps = 9.999999960041972E-13 : f64} : tensor<1x1024x1792xf16>, tensor<1792xf16> -> tensor<1x1024x1792xf16>
    return %0 : tensor<1x1024x1792xf16>
    // CHECK:  [[GAMMA:%.+]] = const.Declare tensor<1792xf16> = dense<1.000000e+00>
    // CHECK:  [[RMS:%.+]] = IE.RMS([[ARG0]], [[GAMMA]]) {eps = 9.9999997171806853E-10 : f64}
    // CHECK: return [[RMS]] : tensor<1x1024x1792xf16>
}
