//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --merge-fake-quant %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

!qElemType0 = type !quant.uniform<u8:f32, 1.0:0>

// CHECK-LABEL: @PerTensor
func @PerTensor(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType0} : tensor<1x4xf32> -> tensor<1x4x!qElemType0>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x4x!qElemType0> -> tensor<1x4xf32>
    return %1 : tensor<1x4xf32>

    // CHECK:       [[MIN:%.*]] = const.Declare tensor<1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
    // CHECK:       [[MAX:%.*]] = const.Declare tensor<1x1xf32> = dense<2.550000e+02> : tensor<1x1xf32>

    // CHECK:       [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[MIN]], [[MAX]], [[MIN]], [[MAX]])
    // CHECK-SAME:      levels = 256

    // CHECK:       return [[FQ]]
}

// CHECK-LABEL: @QuantizeCast
func @QuantizeCast(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !quant.uniform<u8:f32, 1.0:0>} : tensor<1x4xf32> -> tensor<1x4x!quant.uniform<u8:f32, 1.0:0>>
    %1 = IE.QuantizeCast(%0) {dstElemType = !quant.uniform<u8:f32, 0.05:0>} : tensor<1x4x!quant.uniform<u8:f32, 1.0:0>> -> tensor<1x4x!quant.uniform<u8:f32, 0.05:0>>
    %2 = IE.Dequantize(%1) {dstElemType = f32} : tensor<1x4x!quant.uniform<u8:f32, 0.05:0>> -> tensor<1x4xf32>
    return %2 : tensor<1x4xf32>

    // CHECK:       [[MIN:%.*]] = const.Declare tensor<1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
    // CHECK:       [[MAX1:%.*]] = const.Declare tensor<1x1xf32> = dense<2.550000e+02> : tensor<1x1xf32>
    // CHECK:       [[MAX2:%.*]] = const.Declare tensor<1x1xf32> = dense<1.275000e+01> : tensor<1x1xf32>

    // CHECK:       [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[MIN]], [[MAX1]], [[MIN]], [[MAX2]])
    // CHECK-SAME:      levels = 256

    // CHECK:       return [[FQ]]
}
