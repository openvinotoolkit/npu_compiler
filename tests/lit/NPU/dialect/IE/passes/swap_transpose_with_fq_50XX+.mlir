//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-transpose-with-fq --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapTransposeWithPerTensorFQuantF8E4M3FN
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x70x1x28xf16>
func.func @SwapTransposeWithPerTensorFQuantF8E4M3FN(%input: tensor<1x70x1x28xf16>) -> tensor<1x1x28x70xf16> {
    %low = const.Declare tensor<f32> = dense<-4.480000e+02> : tensor<f32>
    %high = const.Declare tensor<f32> = dense<4.480000e+02> : tensor<f32>

    %0 = IE.FakeQuantize(%input, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x70x1x28xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x70x1x28xf16>

    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>

    return %1 : tensor<1x1x28x70xf16>

    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<f32> = dense<-4.480000e+02> : tensor<f32>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<f32> = dense<4.480000e+02> : tensor<f32>

    // CHECK:    [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>

    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[TRANSPOSE]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>

    // CHECK:    return [[FQ]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapConvertTransposeWithFQuantF8E5M2
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x70x1x28xui8>
func.func @SwapConvertTransposeWithFQuantF8E5M2(%input: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16> {
    %low = const.Declare tensor<f32> = dense<-5.734400e+04> : tensor<f32>
    %high = const.Declare tensor<f32> = dense<5.734400e+04> : tensor<f32>

    %0 = IE.Convert(%input) {dstElemType = f16} : tensor<1x70x1x28xui8> -> tensor<1x70x1x28xf16>
    %1 = IE.FakeQuantize(%0, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x70x1x28xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x70x1x28xf16>

    %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %3 = IE.FakeQuantize(%2, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
    } : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>

    return %3 : tensor<1x1x28x70xf16>

    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<f32> = dense<-5.734400e+04> : tensor<f32>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<f32> = dense<5.734400e+04> : tensor<f32>

    // CHECK:    [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x70x1x28xui8> -> tensor<1x70x1x28xf16>
    // CHECK:    [[TRANSPOSE:%.+]] = IE.Transpose([[CONVERT]]) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>

    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[TRANSPOSE]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>

    // CHECK:    return [[FQ]] : tensor<1x1x28x70xf16>
}
