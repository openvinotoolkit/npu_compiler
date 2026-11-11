//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --merge-fake-quant %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

!qElemType = !quant.uniform<u8:f32, 1.0:0>

// CHECK-LABEL: @PerTensor
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x4xf32>
func.func @PerTensor(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf32> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x4x!qElemType> -> tensor<1x4xf32>
    return %1 : tensor<1x4xf32>

    // CHECK-DAG:       [[MIN:%.+]] = const.Declare tensor<1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
    // CHECK-DAG:       [[MAX:%.+]] = const.Declare tensor<1x1xf32> = dense<2.550000e+02> : tensor<1x1xf32>

    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[ARG_0]], [[MIN]], [[MAX]], [[MIN]], [[MAX]])
    // CHECK-SAME:      levels_or_low_fp_type = 256

    // CHECK:       return [[FQ]]
}

// CHECK-LABEL: @QuantizeCast
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x4xf32>
func.func @QuantizeCast(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !quant.uniform<u8:f32, 1.0:0>} : tensor<1x4xf32> -> tensor<1x4x!quant.uniform<u8:f32, 1.0:0>>
    %1 = IE.QuantizeCast(%0) {dstElemType = !quant.uniform<u8:f32, 0.05:0>} : tensor<1x4x!quant.uniform<u8:f32, 1.0:0>> -> tensor<1x4x!quant.uniform<u8:f32, 0.05:0>>
    %2 = IE.Dequantize(%1) {dstElemType = f32} : tensor<1x4x!quant.uniform<u8:f32, 0.05:0>> -> tensor<1x4xf32>
    return %2 : tensor<1x4xf32>

    // CHECK-DAG:       [[MIN1:%.+]] = const.Declare tensor<1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
    // CHECK-DAG:       [[MAX1:%.+]] = const.Declare tensor<1x1xf32> = dense<2.550000e+02> : tensor<1x1xf32>
    // CHECK-DAG:       [[MIN2:%.+]] = const.Declare tensor<1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
    // CHECK-DAG:       [[MAX2:%.+]] = const.Declare tensor<1x1xf32> = dense<1.275000e+01> : tensor<1x1xf32>

    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[ARG_0]], [[MIN1]], [[MAX1]], [[MIN2]], [[MAX2]])
    // CHECK-SAME:      levels_or_low_fp_type = 256

    // CHECK:       return [[FQ]]
}
