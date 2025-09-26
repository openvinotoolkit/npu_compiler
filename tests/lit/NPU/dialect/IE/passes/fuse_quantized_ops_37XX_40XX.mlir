//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-quantized-ops %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DoNotFusePerChannelMaxPool
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x4x16x16x!qElemType>)
func.func @DoNotFusePerChannelMaxPool(%arg0: tensor<1x4x16x16x!qElemType>) -> tensor<1x4x16x16x!qElemType> {
    %dequantize = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
    %maxPool = IE.MaxPool(%dequantize) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    %quantize = IE.Quantize(%maxPool) {dstElemType = !qElemType}: tensor<1x4x16x16xf16> -> tensor<1x4x16x16x!qElemType>

    return %quantize : tensor<1x4x16x16x!qElemType>

    //CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[ARG0]])
    //CHECK:  [[MAXPOOL:%.+]] = IE.MaxPool([[DEQUANT]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    //CHECK:  [[QUANT:%.+]] = IE.Quantize([[MAXPOOL]])
    //CHECK:  return [[QUANT]]
}
