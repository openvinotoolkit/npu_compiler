//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --canonicalize --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @FuseClamps1
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x30x30x30xf16>
func.func @FuseClamps1(%arg0: tensor<1x30x30x30xf16>) -> tensor<1x30x30x30xf16> {
    %0 = IE.Clamp(%arg0) {max = 20.000000e+00 : f64, min = 1.000000e+00 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    %1 = IE.Clamp(%0) {max = 10.000000e+00 : f64, min = 5.000000e+00 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    return %1 : tensor<1x30x30x30xf16>

    // CHECK:        [[CLAMP:%.+]] = IE.Clamp([[ARG_0]]) {max = 1.000000e+01 : f64, min = 5.000000e+00 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    // CHECK-NEXT:       return [[CLAMP]]
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @FuseClamps2
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x30x30x30xf16>
func.func @FuseClamps2(%arg0: tensor<1x30x30x30xf16>) -> tensor<1x30x30x30xf16> {
    %0 = IE.Clamp(%arg0) {max = 20.000000e+00 : f64, min = 1.000000e+00 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    %1 = IE.Clamp(%0) {max = 10.000000e+00 : f64, min = 5.000000e+00 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    %3 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    %4 = IE.Add(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x30x30x30xf16>, tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    return %4 : tensor<1x30x30x30xf16>

    // CHECK:        [[CLAMP:%.+]] = IE.Clamp([[ARG_0]]) {max = 1.000000e+01 : f64, min = 5.000000e+00 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    // CHECK-NEXT:       [[TRANSPOSE_1:%.+]] = IE.Transpose([[CLAMP]]) {order_value = #NHCW} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    // CHECK:        [[TRANSPOSE_2:%.+]] = IE.Transpose([[CLAMP]]) {order_value = #NHCW} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    // CHECK:        [[ADD:%.+]] = IE.Add([[TRANSPOSE_1]], [[TRANSPOSE_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x30x30x30xf16>, tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    // CHECK:       return [[ADD]]
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @ClampsNotFused
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x30x30x30xf16>
func.func @ClampsNotFused(%arg0: tensor<1x30x30x30xf16>) -> tensor<1x30x30x30xf16> {
    %0 = IE.Clamp(%arg0) {max = 20.000000e+00 : f64, min = 1.000000e+00 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    %1 = IE.Clamp(%0) {max = 10.000000e+00 : f64, min = 5.000000e+00 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    %2 = IE.Transpose(%0) {order_value = #NHCW} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x30x30x30xf16>, tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    return %3 : tensor<1x30x30x30xf16>

    // CHECK:        [[CLAMP_1:%.+]] = IE.Clamp([[ARG_0]]) {max = 2.000000e+01 : f64, min = 1.000000e+00 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    // CHECK:        [[CLAMP_2:%.+]] = IE.Clamp([[CLAMP_1]]) {max = 1.000000e+01 : f64, min = 5.000000e+00 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    // CHECK-NEXT:       [[TRANSPOSE:%.+]] = IE.Transpose([[CLAMP_1]]) {order_value = #NHCW} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    // CHECK:        [[ADD:%.+]] = IE.Add([[CLAMP_2]], [[TRANSPOSE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x30x30x30xf16>, tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @ConvertClampMaxAttrToFP16
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x30x30x30xf16>
func.func @ConvertClampMaxAttrToFP16(%arg0: tensor<1x30x30x30xf16>) -> tensor<1x30x30x30xf16> {
    %0 = IE.Clamp(%arg0) {max = 3.4028234663852886E+38 : f64, min = 9.9999997473787516E-6 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    return %0 : tensor<1x30x30x30xf16>

    // CHECK:        [[CLAMP:%.+]] = IE.Clamp([[ARG_0]]) {max = 6.550400e+04 : f64, min = 9.9999997473787516E-6 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    // CHECK-NEXT:       return [[CLAMP]]
}

// -----

// CHECK-LABEL: @ConvertClampMinAttrToFP16
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x30x30x30xf16>
func.func @ConvertClampMinAttrToFP16(%arg0: tensor<1x30x30x30xf16>) -> tensor<1x30x30x30xf16> {
    %0 = IE.Clamp(%arg0) {max = 9.9999997473787516E-6 : f64, min = -3.4028234663852886E+38 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    return %0 : tensor<1x30x30x30xf16>

    // CHECK:        [[CLAMP:%.+]] = IE.Clamp([[ARG_0]]) {max = 9.9999997473787516E-6 : f64, min = -6.550400e+04 : f64} : tensor<1x30x30x30xf16> -> tensor<1x30x30x30xf16>
    // CHECK-NEXT:       return [[CLAMP]]
}

// -----

// CHECK-LABEL: @NotConvertClampMinMaxAttrToFP16Max
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x1xf32>
func.func @NotConvertClampMinMaxAttrToFP16Max(%arg0: tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32> {
    %0 = IE.Clamp(%arg0) {max = 8500000.00 : f64, min = 8454140.00 : f64} : tensor<1x1024x1xf32> -> tensor<1x1024x1xf32>
    return %0 : tensor<1x1024x1xf32>

    // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[INPUT]]) {max = 8.500000e+06 : f64, min = 8.454140e+06 : f64} : tensor<1x1024x1xf32> -> tensor<1x1024x1xf32>
    // CHECK:       return [[CLAMP]]
}

// -----

// CHECK-LABEL: @NotConvertClampMinMaxAttrToFP16Lowest
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x1xf32>
func.func @NotConvertClampMinMaxAttrToFP16Lowest(%arg0: tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32> {
    %0 = IE.Clamp(%arg0) {max = -8454140.00 : f64, min = -8500000.00 : f64} : tensor<1x1024x1xf32> -> tensor<1x1024x1xf32>
    return %0 : tensor<1x1024x1xf32>

    // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[INPUT]]) {max = -8.454140e+06 : f64, min = -8.500000e+06 : f64} : tensor<1x1024x1xf32> -> tensor<1x1024x1xf32>
    // CHECK:       return [[CLAMP]]
}

// -----

// CHECK-LABEL: @NotConvertClampMinMaxAttrSI32
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1xsi32>
func.func @NotConvertClampMinMaxAttrSI32(%arg0: tensor<1x1xsi32>) -> tensor<1x1xsi32> {
    %0 = IE.Clamp(%arg0) {max = 1.001590e+05 : f64, min = 0.000000e+00 : f64} : tensor<1x1xsi32> -> tensor<1x1xsi32>
    return %0 : tensor<1x1xsi32>

    // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[INPUT]]) {max = 1.001590e+05 : f64, min = 0.000000e+00 : f64} : tensor<1x1xsi32> -> tensor<1x1xsi32>
    // CHECK:       return [[CLAMP]]
}
