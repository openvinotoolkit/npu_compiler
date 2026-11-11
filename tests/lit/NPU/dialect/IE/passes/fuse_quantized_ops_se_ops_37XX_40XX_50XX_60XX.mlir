//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% enable-se-ptrs-operations=true" --fuse-quantized-ops %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

!qElemType = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK: @FuseQuantParamsIntoTransposedConv([[ARG:%.+]]: tensor<1x3x16x16xf16>) -> tensor<1x3x32x32xf16>
func.func @FuseQuantParamsIntoTransposedConv(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x32x32xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x4x4x!qElemType> = dense<1.0> : tensor<3x3x4x4xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x4x4x!qElemType> -> tensor<3x3x4x4xf16>
  %4 = IE.TransposedConvolution(%2, %3) {
      dilations = [1, 1],
      operandSegmentSizes = array<i32: 1, 1, 0, 0>,
      spatial_output_padding = [0, 0],
      pads_begin = [1, 1],
      pads_end = [1, 1],
      strides = [2, 2]
    } : tensor<1x3x16x16xf16>, tensor<3x3x4x4xf16> -> tensor<1x3x32x32xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x32x32xf16> -> tensor<1x3x32x32x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x32x32x!qElemType2> -> tensor<1x3x32x32xf16>

  return %6 : tensor<1x3x32x32xf16>

  //CHECK: [[VAL1:%.+]] = IE.Quantize([[ARG]]) {dstElemType = [[QTYPE1]]}
  //CHECK: [[CST:%.+]] = const.Declare tensor<3x3x4x4x[[QTYPE]]> = dense<1.000000e+00> : tensor<3x3x4x4xf16>, [#const.CastElemType<ui8>, #const.CastElemType<[[QTYPE]]>]
  //CHECK: [[VAL2:%.+]] = IE.TransposedConvolution([[VAL1]], [[CST]]) {
  //CHECK-SAME:   dilations = [1, 1],
  //CHECK-SAME:   operandSegmentSizes = array<i32: 1, 1, 0, 0>,
  //CHECK-SAME:   pads_begin = [1, 1],
  //CHECK-SAME:   pads_end = [1, 1],
  //CHECK-SAME:   spatial_output_padding = [0, 0],
  //CHECK-SAME:   strides = [2, 2]
  //CHECK-SAME:   } : tensor<1x3x16x16x[[QTYPE1]]>, tensor<3x3x4x4x[[QTYPE]]>
  //CHECK-SAME:    -> tensor<1x3x32x32x[[QTYPE2]]>
  //CHECK: [[VAL3:%.+]] = IE.Dequantize([[VAL2]]) {dstElemType = f16}
  //CHECK: return [[VAL3]]
}

// -----
// CHECK: !qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
// CHECK: !qElemType1 = !quant.uniform<i8:f16:0, {0.010680671751968504:1,0.0081200787401574797:1,0.010596087598425197:1}>
// CHECK: !qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>
!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<i8:f16:0, {0.010680671751968504:1,0.0081200787401574797:1,0.010596087598425197:1}>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @DoNotFuseQuantParamsIntoTransposedConvWithDifferentIntegerSignedness
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @DoNotFuseQuantParamsIntoTransposedConvWithDifferentIntegerSignedness(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x32x32xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x4x4x!qElemType1> = dense<1.0> : tensor<3x3x4x4xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType1>]
  %2 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x4x4x!qElemType1> -> tensor<3x3x4x4xf16>
  %3 = IE.TransposedConvolution(%1, %2) {
      dilations = [1, 1],
      operandSegmentSizes = array<i32: 1, 1, 0, 0>,
      spatial_output_padding = [0, 0],
      pads_begin = [1, 1],
      pads_end = [1, 1],
      strides = [2, 2]
    } : tensor<1x3x16x16xf16>, tensor<3x3x4x4xf16> -> tensor<1x3x32x32xf16>
  %4 = IE.Quantize(%3) {dstElemType = !qElemType2}: tensor<1x3x32x32xf16> -> tensor<1x3x32x32x!qElemType2>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x32x32x!qElemType2> -> tensor<1x3x32x32xf16>

  return %5 : tensor<1x3x32x32xf16>

  // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<3x3x4x4x!qElemType1> =
  // CHECK-SAME:   dense<1.000000e+00> : tensor<3x3x4x4xf16>,
  // CHECK-SAME:   [#const.CastElemType<si8>, #const.CastElemType<!qElemType1>]
  // CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f16} : tensor<3x3x4x4x!qElemType1> -> tensor<3x3x4x4xf16>
  // CHECK: [[TCONV:%.+]] = IE.TransposedConvolution([[DEQUANTIZE]], [[DEQUANTIZE1]]) {
  // CHECK-SAME:   dilations = [1, 1],
  // CHECK-SAME:   operandSegmentSizes = array<i32: 1, 1, 0, 0>,
  // CHECK-SAME:   pads_begin = [1, 1],
  // CHECK-SAME:   pads_end = [1, 1],
  // CHECK-SAME:   spatial_output_padding = [0, 0],
  // CHECK-SAME:   strides = [2, 2]
  // CHECK-SAME:   } : tensor<1x3x16x16xf16>, tensor<3x3x4x4xf16>
  // CHECK-SAME:    -> tensor<1x3x32x32xf16>
  // CHECK: [[QUANTIZE1:%.+]] = IE.Quantize([[TCONV]]) {dstElemType = !qElemType2} : tensor<1x3x32x32xf16> -> tensor<1x3x32x32x!qElemType2>
  // CHECK: [[DEQUANTIZE2:%.+]] = IE.Dequantize([[QUANTIZE1]]) {dstElemType = f16} : tensor<1x3x32x32x!qElemType2> -> tensor<1x3x32x32xf16>
  // CHECK: return [[DEQUANTIZE2]]
}

// -----

!qElemType = !quant.uniform<i8:f16:0, {0.010680671751968504:1,0.0081200787401574797:1,0.010596087598425197:1}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @DoNotFuseQuantParamsIntoTransposedConvWithDifferentIntegerSignedness
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @DoNotFuseQuantParamsIntoTransposedConvWithDifferentIntegerSignedness(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x32x32xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x4x4x!qElemType> = dense<1.0> : tensor<3x3x4x4xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x4x4x!qElemType> -> tensor<3x3x4x4xf16>
  %4 = IE.TransposedConvolution(%2, %3) {
      dilations = [1, 1],
      operandSegmentSizes = array<i32: 1, 1, 0, 0>,
      spatial_output_padding = [0, 0],
      pads_begin = [1, 1],
      pads_end = [1, 1],
      strides = [2, 2]
    } : tensor<1x3x16x16xf16>, tensor<3x3x4x4xf16> -> tensor<1x3x32x32xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x32x32xf16> -> tensor<1x3x32x32x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x32x32x!qElemType2> -> tensor<1x3x32x32xf16>

  return %6 : tensor<1x3x32x32xf16>

  //CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  //CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  //CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<3x3x4x4x!qElemType1> =
  //CHECK-SAME:   dense<1.000000e+00> : tensor<3x3x4x4xf16>,
  //CHECK-SAME:   [#const.CastElemType<si8>, #const.CastElemType<!qElemType1>]
  //CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f16} : tensor<3x3x4x4x!qElemType1> -> tensor<3x3x4x4xf16>
  //CHECK: [[TCONV:%.*]] = IE.TransposedConvolution([[DEQUANTIZE]], [[DEQUANTIZE1]]) {
  //CHECK-SAME:   dilations = [1, 1],
  //CHECK-SAME:   operandSegmentSizes = array<i32: 1, 1, 0, 0>,
  //CHECK-SAME:   pads_begin = [1, 1],
  //CHECK-SAME:   pads_end = [1, 1],
  //CHECK-SAME:   spatial_output_padding = [0, 0],
  //CHECK-SAME:   strides = [2, 2]
  //CHECK-SAME:   } : tensor<1x3x16x16xf16>, tensor<3x3x4x4xf16>
  //CHECK-SAME:    -> tensor<1x3x32x32xf16>
  //CHECK: [[QUANTIZE1:%.+]] = IE.Quantize([[TCONV]]) {dstElemType = !qElemType2} : tensor<1x3x32x32xf16> -> tensor<1x3x32x32x!qElemType2>
  //CHECK: [[DEQUANTIZE2:%.+]] = IE.Dequantize([[QUANTIZE1]]) {dstElemType = f16} : tensor<1x3x32x32x!qElemType2> -> tensor<1x3x32x32xf16>
  //CHECK: return [[DEQUANTIZE2]]
}
