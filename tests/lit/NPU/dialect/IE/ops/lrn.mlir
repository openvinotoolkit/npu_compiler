//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ConvertConstToAttr
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x128x50x85xf16>
func.func @ConvertConstToAttr(%arg0: tensor<1x128x50x85xf16>) -> tensor<1x128x50x85xf16> {
    %0 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-NOT:   const.Declare
    %1 = IE.LRN(%arg0, %0) {alpha = 1.000000e-04 : f64, beta = 7.500000e-01 : f64, bias = 1.000000e+00 : f64, size = 5 : i64} : tensor<1x128x50x85xf16>, tensor<1xsi64> -> tensor<1x128x50x85xf16>
    // CHECK:       [[VAL0:%.+]] = IE.LRN([[ARG_0]]) {alpha = 1.000000e-04 : f64, axes_value = [1], beta = 7.500000e-01 : f64, bias = 1.000000e+00 : f64, size = 5 : i64} : tensor<1x128x50x85xf16> -> tensor<1x128x50x85xf16>

    return %1 : tensor<1x128x50x85xf16>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @Convert3ConstToAttr
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x128x50x85xf16>
func.func @Convert3ConstToAttr(%arg0: tensor<1x128x50x85xf16>) -> tensor<1x128x50x85xf16> {
    %0 = const.Declare tensor<3xsi64> = dense<[0, 1, -1]> : tensor<3xsi64>
    // CHECK-NOT:   const.Declare
    %1 = IE.LRN(%arg0, %0) {alpha = 1.000000e-04 : f64, beta = 7.500000e-01 : f64, bias = 1.000000e+00 : f64, size = 5 : i64} : tensor<1x128x50x85xf16>, tensor<3xsi64> -> tensor<1x128x50x85xf16>
    // CHECK:       [[VAL0:%.+]] = IE.LRN([[ARG_0]]) {alpha = 1.000000e-04 : f64, axes_value = [0, 1, 3], beta = 7.500000e-01 : f64, bias = 1.000000e+00 : f64, size = 5 : i64} : tensor<1x128x50x85xf16> -> tensor<1x128x50x85xf16>

    return %1 : tensor<1x128x50x85xf16>
    // CHECK:       return [[VAL0]]
}