//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-quantized-ops %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

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

// -----

// The bias=100 with these scales gives bias_int32 = 100 / (3.9215E-5 x 7.874E-4) ~= 3.24E9 > INT32_MAX.
// On NPU37XX/40XX the bias is stored as int32 in the weight table, so this overflow must block fusion.
!qElemType = !quant.uniform<u8<1:255>:f16:0, {7.8740158653634745E-4:128,7.8740158653634745E-4:128,7.8740158653634745E-4:128}>
!qElemType1 = !quant.uniform<u8:f16, 3.9215E-5:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>
// CHECK-DAG: [[$QTYPE:!.+]] = !quant.uniform<u8<1:255>:f16:0, {7.8740158653634745E-4:128,7.8740158653634745E-4:128,7.8740158653634745E-4:128}>
// CHECK-DAG: [[$QTYPE1:!.+]] = !quant.uniform<u8:f16, 3.921500e-05:128>
// CHECK-DAG: [[$QTYPE2:!.+]] = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @DoNotFuseToConvWithBiasOverflowingInt32
// CHECK-SAME:  ([[ARG:%.+]]: tensor<1x3x16x16xf16>)
func.func @DoNotFuseToConvWithBiasOverflowingInt32(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %bias = const.Declare tensor<1x3x1x1xf16> = dense<100.0> : tensor<1x3x1x1xf16>
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2} : tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  return %6 : tensor<1x3x14x14xf16>

  // CHECK:     [[QUANT:%.+]]   = IE.Quantize([[ARG]]) {dstElemType = [[$QTYPE1]]} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x[[$QTYPE1]]>
  // CHECK:     [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} : tensor<1x3x16x16x[[$QTYPE1]]> -> tensor<1x3x16x16xf16>
  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<3x3x3x3x[[$QTYPE]]>
  // CHECK-DAG: [[BIAS:%.+]]    = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+02>
  // CHECK:     [[WDEQUANT:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f16}
  // CHECK:     [[CONV:%.+]]    = IE.Convolution([[DEQUANT]], [[WDEQUANT]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x14x14xf16>
  // CHECK:     [[REQUANT:%.+]] = IE.Quantize([[CONV]]) {dstElemType = [[$QTYPE2]]}
  // CHECK:     [[OUT:%.+]]     = IE.Dequantize([[REQUANT]]) {dstElemType = f16}
  // CHECK:     return [[OUT]]
}

// -----

// Same bias overflow scenario as DoNotFuseToConvWithBiasOverflowingInt32 but for GroupConvolution.
!qElemType = !quant.uniform<u8<1:255>:f16:0, {7.8740158653634745E-4:128,7.8740158653634745E-4:128,7.8740158653634745E-4:128}>
!qElemType1 = !quant.uniform<u8:f16, 3.9215E-5:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>
// CHECK-DAG: [[$QTYPE:!.+]] = !quant.uniform<u8<1:255>:f16:0, {7.8740158653634745E-4:128,7.8740158653634745E-4:128,7.8740158653634745E-4:128}>
// CHECK-DAG: [[$QTYPE1:!.+]] = !quant.uniform<u8:f16, 3.921500e-05:128>
// CHECK-DAG: [[$QTYPE2:!.+]] = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @DoNotFuseToGroupConvWithBiasOverflowingInt32
// CHECK-SAME:  ([[ARG:%.+]]: tensor<1x3x16x16xf16>)
func.func @DoNotFuseToGroupConvWithBiasOverflowingInt32(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x1x3x3x!qElemType> = dense<1.0> : tensor<3x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %bias = const.Declare tensor<1x3x1x1xf16> = dense<100.0> : tensor<1x3x1x1xf16>
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x1x3x3x!qElemType> -> tensor<3x1x3x3xf16>
  %4 = IE.GroupConvolution(%2, %3, %bias) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x1x3x3xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2} : tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  return %6 : tensor<1x3x14x14xf16>

  // CHECK:     [[QUANT:%.+]]   = IE.Quantize([[ARG]]) {dstElemType = [[$QTYPE1]]} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x[[$QTYPE1]]>
  // CHECK:     [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} : tensor<1x3x16x16x[[$QTYPE1]]> -> tensor<1x3x16x16xf16>
  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<3x1x3x3x[[$QTYPE]]>
  // CHECK-DAG: [[BIAS:%.+]]    = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+02>
  // CHECK:     [[WDEQUANT:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f16}
  // CHECK:     [[GCONV:%.+]]   = IE.GroupConvolution([[DEQUANT]], [[WDEQUANT]], [[BIAS]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x1x3x3xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x14x14xf16>
  // CHECK:     [[REQUANT:%.+]] = IE.Quantize([[GCONV]]) {dstElemType = [[$QTYPE2]]}
  // CHECK:     [[OUT:%.+]]     = IE.Dequantize([[REQUANT]]) {dstElemType = f16}
  // CHECK:     return [[OUT]]
}
