//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ConvertConstToAttr
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x128x50x85xf16>
func.func @ConvertConstToAttr(%arg0: tensor<1x128x50x85xf16>) -> tensor<1x128x50x85xf16> {
    %0 = const.Declare tensor<1xsi64> = dense<[0]> : tensor<1xsi64>
    // CHECK-NOT:   const.Declare
        %1 = IE.NormalizeL2(%arg0, %0) {eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x128x50x85xf16>, tensor<1xsi64> -> tensor<1x128x50x85xf16>
    // CHECK:       [[VAL0:%.+]] = IE.NormalizeL2([[ARG_0]]) {axes_value = [0], eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x128x50x85xf16> -> tensor<1x128x50x85xf16>

    return %1 : tensor<1x128x50x85xf16>
    // CHECK:       return [[VAL0]]
}

// -----
// CHECK-LABEL: @Convert3ConstToAttr
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x128x50x85xf16>
func.func @Convert3ConstToAttr(%arg0: tensor<1x128x50x85xf16>) -> tensor<1x128x50x85xf16> {
    %0 = const.Declare tensor<3xsi64> = dense<[0, 1, -1]> : tensor<3xsi64>
    // CHECK-NOT:   const.Declare
        %1 = IE.NormalizeL2(%arg0, %0) {eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x128x50x85xf16>, tensor<3xsi64> -> tensor<1x128x50x85xf16>
    // CHECK:       [[VAL0:%.+]] = IE.NormalizeL2([[ARG_0]]) {axes_value = [0, 1, 3], eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x128x50x85xf16> -> tensor<1x128x50x85xf16>

    return %1 : tensor<1x128x50x85xf16>
    // CHECK:       return [[VAL0]]
}

// -----
