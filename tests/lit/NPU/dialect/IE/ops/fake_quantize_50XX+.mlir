//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @FuseFQuantF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x3x16x16xf16>
func.func @FuseFQuantF8E4M3FN(%input: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %low = const.Declare tensor<f32> = dense<-4.480000e+02> : tensor<f32>
    %high = const.Declare tensor<f32> = dense<4.480000e+02> : tensor<f32>

    %0 = IE.FakeQuantize(%input, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x3x16x16xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x16x16xf16>

    %1 = IE.FakeQuantize(%0, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x3x16x16xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>

    // CHECK:    [[LOW:%.+]] = const.Declare tensor<f32> = dense<-4.480000e+02> : tensor<f32>
    // CHECK:    [[HIGH:%.+]] = const.Declare tensor<f32> = dense<4.480000e+02> : tensor<f32>

    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x3x16x16xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x16x16xf16>

    // CHECK:    return [[FQ]] : tensor<1x3x16x16xf16>
}

// -----

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @TransposeGroupsF8E5M2
func.func @TransposeGroupsF8E5M2() -> tensor<1280x20x128xf32> {
    %weights = const.Declare tensor<1280x20x128xf32> = dense<4.500000e+01>  : tensor<1280x20x128xf32>
    %in_low = const.Declare tensor<1x1x1xf32> = dense<-5.734400e+04> : tensor<1x1x1xf32>
    %in_high = const.Declare tensor<1x1x1xf32> = dense<5.734400e+04> : tensor<1x1x1xf32>
    %out_low = const.Declare tensor<1280x20x1xf32> = dense<-2.0>  : tensor<1280x20x1xf32>
    %out_high = const.Declare tensor<1280x20x1xf32> = dense<2.0>  : tensor<1280x20x1xf32>

    %0 = IE.FakeQuantize(%weights, %in_low, %in_high, %out_low, %out_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1280x20x128xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1280x20x1xf32>, tensor<1280x20x1xf32> -> tensor<1280x20x128xf32>

    return %0 : tensor<1280x20x128xf32>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<128x1280x20xf32> = dense<4.500000e+01> : tensor<1280x20x128xf32>, [#const.Transpose<#map>]
    // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-5.734400e+04> : tensor<1x1x1xf32>, [#const.Transpose<#map>]
    // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<5.734400e+04> : tensor<1x1x1xf32>, [#const.Transpose<#map>]
    // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1x1280x20xf32> = dense<-2.000000e+00> : tensor<1280x20x1xf32>, [#const.Transpose<#map>]
    // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1280x20xf32> = dense<2.000000e+00> : tensor<1280x20x1xf32>, [#const.Transpose<#map>]

    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-DAG:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<128x1280x20xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1280x20xf32>, tensor<1x1280x20xf32> -> tensor<128x1280x20xf32>

    // CHECK:    [[TRANS:%.+]] = IE.Transpose([[FQ]]) {order_value = #HWC} : tensor<128x1280x20xf32> -> tensor<1280x20x128xf32>

    // CHECK:    return [[TRANS]] : tensor<1280x20x128xf32>
}
