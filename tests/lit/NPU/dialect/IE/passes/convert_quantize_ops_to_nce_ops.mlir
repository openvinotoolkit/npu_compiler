//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-quantize-ops-to-nce-ops  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<u8:f32, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>
!qElemType2 = !quant.uniform<u8:f32, 5.000000e-01>

module @ConvertQuantizeToEltwise {

func.func @PerTensor(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x4xf32> -> tensor<1x4x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x4x!qElemType1> -> tensor<1x4xf32>

    // CHECK:  %[[VAL0:.*]] = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4xf32>, tensor<1x4xf32> -> tensor<1x4x!qElemType>
    // CHECK:  %[[VAL1:.*]] = IE.QuantizeCast(%[[VAL0]]) {dstElemType = !qElemType1} : tensor<1x4x!qElemType> -> tensor<1x4x!qElemType1>
    // CHECK:  %[[VAL2:.*]] = IE.QuantizeCast(%[[VAL1]]) {dstElemType = !qElemType2} : tensor<1x4x!qElemType1> -> tensor<1x4x!qElemType2>
    // CHECK:  %[[VAL3:.*]] = IE.Add(%[[VAL2]], %[[VAL2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x!qElemType2>, tensor<1x4x!qElemType2> -> tensor<1x4xf32>

    return %1 : tensor<1x4xf32>

    // CHECK:  return %[[VAL3]] : tensor<1x4xf32>
}

}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>
!qElemType1 = !quant.uniform<u8<0:254>:f16, 1.000000e+00>
module @ConvertQuantizeToDwConv {

func.func @PerChannel(%arg0 : tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    // CHECK-DAG:   %[[VAL0:.*]] = const.Declare tensor<3x1x1x1xf16> = dense<1.000000e+00> : tensor<3x1x1x1xf16>
    // CHECK:       %[[VAL1:.*]] = IE.GroupConvolution(%arg0, %[[VAL0]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 3 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x3x16x16xf16>, tensor<3x1x1x1xf16> -> tensor<1x3x16x16x!qElemType>

    // CHECK-DAG:   %[[VAL2:.*]] = const.Declare tensor<3x1x1x1x!qElemType1> = dense<1.000000e+00> : tensor<3x1x1x1xf16>, [#const.CastElemType<!qElemType1>]
    // CHECK:       %[[VAL3:.*]] = IE.GroupConvolution(%[[VAL1]], %[[VAL2]])  {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 3 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x3x16x16x!qElemType>, tensor<3x1x1x1x!qElemType1> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>

    // CHECK:  return %[[VAL3]] : tensor<1x3x16x16xf16>
}

}

// -----

!qElemType = !quant.uniform<u8:f32, 2.000000e+00>

module @ConvertQuantizeToAvgPool {

func.func @PerTensor(%arg0 : tensor<1x400x800x400xf32>) -> tensor<1x400x800x400xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x400x800x400xf32> -> tensor<1x400x800x400x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x400x800x400x!qElemType> -> tensor<1x400x800x400xf32>
    return %1 : tensor<1x400x800x400xf32>

    // CHECK:  %[[AVGPOOL_0:.*]] = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x400x800x400xf32> -> tensor<1x400x800x400x!qElemType>
    // CHECK:  %[[AVGPOOL_1:.*]] = IE.AvgPool(%[[AVGPOOL_0]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x400x800x400x!qElemType> -> tensor<1x400x800x400xf32>
    // CHECK:  return %[[AVGPOOL_1]] : tensor<1x400x800x400xf32>
}

}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.004>

module @Float8 {
// CHECK-LABEL:  func.func @Float8
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x8x4x64xf16>)
func.func @Float8(%arg0 : tensor<1x8x4x64xf16>) -> tensor<1x8x4x64xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x8x4x64xf16> -> tensor<1x8x4x64x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x8x4x64x!qElemType> -> tensor<1x8x4x64xf16>
    return %1 : tensor<1x8x4x64xf16>
}
    // CHECK:  [[AVGPOOL_0:%.+]] = IE.AvgPool([[INPUT]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x8x4x64xf16> -> tensor<1x8x4x64x!qElemType>
    // CHECK:  [[AVGPOOL_1:%.+]] = IE.AvgPool([[AVGPOOL_0]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x8x4x64x!qElemType> -> tensor<1x8x4x64xf16>
    // CHECK:  return [[AVGPOOL_1]] : tensor<1x8x4x64xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:3, {
    0.956:128, 0.785:128, 0.567:128, 0.487:128,
    0.957:128, 0.786:128, 0.568:128, 0.488:128,
    0.958:128, 0.787:128, 0.569:128, 0.489:128,
    0.959:128, 0.788:128, 0.560:128, 0.480:128}>
!qElemType1 = !quant.uniform<u8<0:254>:f16, 1.000000e+00>
module @NotConvertQuantizeToDwConv {

// CHECK-LABEL:  func.func @NotPerChannel
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x3x16x16xf16>)
func.func @NotPerChannel(%arg0 : tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>

    // CHECK:  [[QUANTIZE:%.*]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.*]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    // CHECK:  return [[DEQUANTIZE]] : tensor<1x3x16x16xf16>
}

}
