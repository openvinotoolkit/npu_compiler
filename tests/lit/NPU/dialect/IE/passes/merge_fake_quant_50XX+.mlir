//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --merge-fake-quant %s | FileCheck %s
// REQUIRES: platform-NPU5010 || platform-NPU6010
// COM: F8 is only supported on NPU50+, no need to run these tests on all platforms.

!qElemType = !quant.uniform<f8E4M3FN:f32, 1.0:0>

// CHECK-LABEL: @MergeQuantizeDequantizeF8E4M3FN
// CHECK-SAME:      [[ARG_0:%.+]]: tensor<1x4xf32>
func.func @MergeQuantizeDequantizeF8E4M3FN(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf32> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x4x!qElemType> -> tensor<1x4xf32>
    return %1 : tensor<1x4xf32>

    // CHECK-DAG:       [[MIN:%.+]] = const.Declare tensor<1x1xf32> = dense<-4.480000e+02> : tensor<1x1xf32>
    // CHECK-DAG:       [[MAX:%.+]] = const.Declare tensor<1x1xf32> = dense<4.480000e+02> : tensor<1x1xf32>

    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[ARG_0]], [[MIN]], [[MAX]], [[MIN]], [[MAX]])
    // CHECK-SAME:      levels_or_low_fp_type = f8E4M3FN

    // CHECK:       return [[FQ]]
}

// -----

!qElemType = !quant.uniform<f8E5M2:f32, 1.0:0>

// CHECK-LABEL: @MergeQuantizeCastF8E5M2
// CHECK-SAME:      [[ARG_0:%.+]]: tensor<1x4xf32>
func.func @MergeQuantizeCastF8E5M2(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf32> -> tensor<1x4x!qElemType>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType} : tensor<1x4x!qElemType> -> tensor<1x4x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f32} : tensor<1x4x!qElemType> -> tensor<1x4xf32>
    return %2 : tensor<1x4xf32>

    // CHECK-DAG:       [[MIN1:%.+]] = const.Declare tensor<1x1xf32> = dense<-5.734400e+04> : tensor<1x1xf32>
    // CHECK-DAG:       [[MIN2:%.+]] = const.Declare tensor<1x1xf32> = dense<-5.734400e+04> : tensor<1x1xf32>
    // CHECK-DAG:       [[MAX1:%.+]] = const.Declare tensor<1x1xf32> = dense<5.734400e+04> : tensor<1x1xf32>
    // CHECK-DAG:       [[MAX2:%.+]] = const.Declare tensor<1x1xf32> = dense<5.734400e+04> : tensor<1x1xf32>

    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[ARG_0]], [[MIN1]], [[MAX1]], [[MIN2]], [[MAX2]])
    // CHECK-SAME:      levels_or_low_fp_type = f8E5M2

    // CHECK:       return [[FQ]]
}
