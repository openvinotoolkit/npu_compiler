//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-quantized-ops="se-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<u8:f16, 2.4627450980392158>

//CHECK:  !qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
//CHECK:  !qElemType1 = !quant.uniform<u8:f16, 2.4627450980392158>

//CHECK-LABEL: @FuseQuantParamsIntoInterp
func.func @FuseQuantParamsIntoInterp(%arg0: tensor<1x16x10x10xf16>) -> tensor<1x16x20x20xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x10x10xf16> -> tensor<1x16x10x10x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x10x10x!qElemType> -> tensor<1x16x10x10xf16>
  %4 = IE.Interpolate(%2) {
            attr = #IE.Interpolate<mode = <NEAREST>,
                                   shape_calc_mode = <SCALES>,
                                   coord_mode = <ASYMMETRIC>,
                                   nearest_mode = <FLOOR>,
                                   antialias = false,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
                                   axes_attr = [2, 3],
                                   operandSegmentSizes = array<i32: 1, 0, 0, 0>,
                                   scales_attr = [2.000000e+00, 2.000000e+00],
                                   sizes_attr = [20, 20]} :
        tensor<1x16x10x10xf16> -> tensor<1x16x20x20xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType1}: tensor<1x16x20x20xf16> -> tensor<1x16x20x20x!qElemType1>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x16x20x20x!qElemType1> -> tensor<1x16x20x20xf16>

  return %6 : tensor<1x16x20x20xf16>

  //CHECK:      [[QUANT:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType}
  //CHECK-SAME:   tensor<1x16x10x10xf16> -> tensor<1x16x10x10x!qElemType>

  //CHECK:      [[INTERP:%.*]] = IE.Interpolate([[QUANT]])
  //CHECK-SAME:   tensor<1x16x10x10x!qElemType> -> tensor<1x16x20x20x!qElemType1>

  //CHECK:      [[DEQUANT:%.*]] = IE.Dequantize([[INTERP]]) {dstElemType = f16}
  //CHECK-SAME:   tensor<1x16x20x20x!qElemType1> -> tensor<1x16x20x20xf16>

  //CHECK:      return [[DEQUANT]] : tensor<1x16x20x20xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<u8:f16, 2.4627450980392158>

// Do not quantize interoplate that will not be run on NCE due to non integer scales

//CHECK:  !qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
//CHECK:  !qElemType1 = !quant.uniform<u8:f16, 2.4627450980392158>
//CHECK-LABEL: @DonotFuseQuantParamsIntoInterp
func.func @DonotFuseQuantParamsIntoInterp(%arg0: tensor<1x16x10x10xf16>) -> tensor<1x16x25x25xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x10x10xf16> -> tensor<1x16x10x10x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x10x10x!qElemType> -> tensor<1x16x10x10xf16>
  %4 = IE.Interpolate(%2) {
            attr = #IE.Interpolate<mode = <NEAREST>,
                                   shape_calc_mode = <SCALES>,
                                   coord_mode = <ASYMMETRIC>,
                                   nearest_mode = <FLOOR>,
                                   antialias = false,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
                                   axes_attr = [2, 3],
                                   operandSegmentSizes = array<i32: 1, 0, 0, 0>,
                                   scales_attr = [2.500000e+00, 2.500000e+00],
                                   sizes_attr = [25, 25]} :
        tensor<1x16x10x10xf16> -> tensor<1x16x25x25xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType1}: tensor<1x16x25x25xf16> -> tensor<1x16x25x25x!qElemType1>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x16x25x25x!qElemType1> -> tensor<1x16x25x25xf16>

  return %6 : tensor<1x16x25x25xf16>

  //CHECK:      [[QUANT:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType}
  //CHECK-SAME:   tensor<1x16x10x10xf16> -> tensor<1x16x10x10x!qElemType>

  //CHECK:      [[DEQUANT:%.*]] = IE.Dequantize([[QUANT]]) {dstElemType = f16}
  //CHECK-SAME:   tensor<1x16x10x10x!qElemType> -> tensor<1x16x10x10xf16>

  //CHECK:      [[INTERP:%.*]] = IE.Interpolate([[DEQUANT]])
  //CHECK-SAME:   tensor<1x16x10x10xf16> -> tensor<1x16x25x25xf16>

  //CHECK:      [[QUANT_OUT:%.*]] = IE.Quantize([[INTERP]]) {dstElemType = !qElemType1}
  //CHECK-SAME:   tensor<1x16x25x25xf16> -> tensor<1x16x25x25x!qElemType1>

  //CHECK:      [[DEQUANT_OUT:%.*]] = IE.Dequantize([[QUANT_OUT]]) {dstElemType = f16}
  //CHECK-SAME:   tensor<1x16x25x25x!qElemType1> -> tensor<1x16x25x25xf16>

  //CHECK:      return [[DEQUANT_OUT]] : tensor<1x16x25x25xf16>
}

// -----

!qElemType = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @FuseQuantParamsIntoTransposedConv
func.func @FuseQuantParamsIntoTransposedConv(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x32x32xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x4x4x!qElemType> = dense<1.0> : tensor<3x3x4x4xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x4x4x!qElemType> -> tensor<3x3x4x4xf16>
  %4 = IE.TransposedConvolution(%2, %3) {
      dilations = [1, 1],
      operandSegmentSizes = array<i32: 1, 1, 0, 0>,
      output_padding = [0, 0],
      pads_begin = [1, 1],
      pads_end = [1, 1],
      strides = [2, 2]
    } : tensor<1x3x16x16xf16>, tensor<3x3x4x4xf16> -> tensor<1x3x32x32xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x32x32xf16> -> tensor<1x3x32x32x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x32x32x!qElemType2> -> tensor<1x3x32x32xf16>

  return %6 : tensor<1x3x32x32xf16>

  //CHECK: [[CST:%.*]] = const.Declare tensor<3x3x4x4x!qElemType> = dense<1.000000e+00> : tensor<3x3x4x4xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  //CHECK: [[VAL2:%.*]] = IE.TransposedConvolution([[VAL1]], [[CST]]) {
  //CHECK-SAME:   dilations = [1, 1],
  //CHECK-SAME:   operandSegmentSizes = array<i32: 1, 1, 0, 0>,
  //CHECK-SAME:   output_padding = [0, 0],
  //CHECK-SAME:   pads_begin = [1, 1],
  //CHECK-SAME:   pads_end = [1, 1],
  //CHECK-SAME:   strides = [2, 2]
  //CHECK-SAME:   } : tensor<1x3x16x16x!qElemType1>, tensor<3x3x4x4x!qElemType>
  //CHECK-SAME:    -> tensor<1x3x32x32x!qElemType2>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x32x32x!qElemType2> -> tensor<1x3x32x32xf16>
  //CHECK: return [[VAL3]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK:  !qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DonotFuseQuantPerAxisParamsIntoInterp
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x4x10x10xf16>)
func.func @DonotFuseQuantPerAxisParamsIntoInterp(%arg0: tensor<1x4x10x10xf16>) -> tensor<1x4x25x25xf16> {
  %quantize0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4x10x10xf16> -> tensor<1x4x10x10x!qElemType>
  %dequantize0 = IE.Dequantize(%quantize0) {dstElemType = f16} : tensor<1x4x10x10x!qElemType> -> tensor<1x4x10x10xf16>
  %interpolate = IE.Interpolate(%dequantize0) {
            attr = #IE.Interpolate<mode = <NEAREST>,
                                   shape_calc_mode = <SCALES>,
                                   coord_mode = <ASYMMETRIC>,
                                   nearest_mode = <FLOOR>,
                                   antialias = false,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
                                   axes_attr = [2, 3],
                                   operandSegmentSizes = array<i32: 1, 0, 0, 0>,
                                   scales_attr = [2.500000e+00, 2.500000e+00],
                                   sizes_attr = [25, 25]} :
        tensor<1x4x10x10xf16> -> tensor<1x4x25x25xf16>
  %quantize1 = IE.Quantize(%interpolate) {dstElemType = !qElemType}: tensor<1x4x25x25xf16> -> tensor<1x4x25x25x!qElemType>
  %dequantize1 = IE.Dequantize(%quantize1) {dstElemType = f16} : tensor<1x4x25x25x!qElemType> -> tensor<1x4x25x25xf16>

  return %dequantize1 : tensor<1x4x25x25xf16>

  //CHECK:      [[QUANT0:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType}
  //CHECK-SAME:   tensor<1x4x10x10xf16> -> tensor<1x4x10x10x!qElemType>

  //CHECK:      [[DEQUANT0:%.+]] = IE.Dequantize([[QUANT0]]) {dstElemType = f16}
  //CHECK-SAME:   tensor<1x4x10x10x!qElemType> -> tensor<1x4x10x10xf16>

  //CHECK:      [[INTERP:%.+]] = IE.Interpolate([[DEQUANT0]])
  //CHECK-SAME:   tensor<1x4x10x10xf16> -> tensor<1x4x25x25xf16>

  //CHECK:      [[QUANT1:%.+]] = IE.Quantize([[INTERP]]) {dstElemType = !qElemType}
  //CHECK-SAME:   tensor<1x4x25x25xf16> -> tensor<1x4x25x25x!qElemType>

  //CHECK:      [[DEQUANT1:%.*]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16}
  //CHECK-SAME:   tensor<1x4x25x25x!qElemType> -> tensor<1x4x25x25xf16>

  //CHECK:      return [[DEQUANT1]] : tensor<1x4x25x25xf16>
}
