//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @ConvertConstToAttr
func.func @ConvertConstToAttr(%arg0: tensor<1x128x50x85xf16>) -> tensor<1x128x50x85xf16> {
    %0 = const.Declare tensor<1xsi64> = dense<[0]> : tensor<1xsi64>
    // CHECK-NOT:   const.Declare
        %1 = IE.NormalizeL2(%arg0, %0) {eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x128x50x85xf16>, tensor<1xsi64> -> tensor<1x128x50x85xf16>
    // CHECK:       %[[VAL0:.*]] = IE.NormalizeL2(%arg0) {axes_value = [0], eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x128x50x85xf16> -> tensor<1x128x50x85xf16>

    return %1 : tensor<1x128x50x85xf16>
    // CHECK:       return %[[VAL0]]
}

// -----
// CHECK-LABEL: @Convert3ConstToAttr
func.func @Convert3ConstToAttr(%arg0: tensor<1x128x50x85xf16>) -> tensor<1x128x50x85xf16> {
    %0 = const.Declare tensor<3xsi64> = dense<[0, 1, -1]> : tensor<3xsi64>
    // CHECK-NOT:   const.Declare
        %1 = IE.NormalizeL2(%arg0, %0) {eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x128x50x85xf16>, tensor<3xsi64> -> tensor<1x128x50x85xf16>
    // CHECK:       %[[VAL0:.*]] = IE.NormalizeL2(%arg0) {axes_value = [0, 1, -1], eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x128x50x85xf16> -> tensor<1x128x50x85xf16>

    return %1 : tensor<1x128x50x85xf16>
    // CHECK:       return %[[VAL0]]
}

// -----
