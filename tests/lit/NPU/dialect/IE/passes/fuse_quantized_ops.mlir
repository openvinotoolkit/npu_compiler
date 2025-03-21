//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-quantized-ops %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @FuseQuantParamsIntoConv
func.func @FuseQuantParamsIntoConv(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  return %6 : tensor<1x3x14x14xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<3x3x3x3xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  //CHECK: [[VAL2:%.*]] = IE.Convolution([[VAL1]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16x!qElemType1>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType2>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>
  //CHECK: return [[VAL3]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @FuseQuantParamsIntoEltwiseAdd
func.func @FuseQuantParamsIntoEltwiseAdd(%arg0: tensor<1x3x16x16xf16>, %arg1: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %3 = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %5 = IE.Add(%2, %4) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  %6 = IE.Quantize(%5) {dstElemType = !qElemType1}: tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %7 = IE.Dequantize(%6) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>

  return %7 : tensor<1x3x16x16xf16>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  //CHECK: [[VAL2:%.*]] = IE.Add([[VAL0]], [[VAL1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16x!qElemType>, tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16x!qElemType1>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  //CHECK: return [[VAL3]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @FuseQuantParamsIntoSlice
func.func @FuseQuantParamsIntoSlice(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x8xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %2 = IE.Slice %1 [0, 0, 0, 8] [1, 3, 16, 8] : tensor<1x3x16x16xf16> to tensor<1x3x16x8xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType}: tensor<1x3x16x8xf16> -> tensor<1x3x16x8x!qElemType>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x3x16x8x!qElemType> -> tensor<1x3x16x8xf16>

    return %4 : tensor<1x3x16x8xf16>

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.Slice [[VAL0]] [0, 0, 0, 8] [1, 3, 16, 8] : tensor<1x3x16x16x!qElemType> to tensor<1x3x16x8x!qElemType>
    //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x16x8x!qElemType> -> tensor<1x3x16x8xf16>
    //CHECK: return [[VAL2]] : tensor<1x3x16x8xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.2501400218290441>
!qElemType1 = !quant.uniform<u8:f16, 0.03027489793066885>

// CHECK-LABEL: @FuseQuantParamsIntoSliceUnlessInputFQEqualsOutputFQ
func.func @FuseQuantParamsIntoSliceUnlessInputFQEqualsOutputFQ(%arg0: tensor<1x128x46x80xf16>) -> tensor<1x128x45x80xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x128x46x80xf16> -> tensor<1x128x46x80x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x128x46x80x!qElemType> -> tensor<1x128x46x80xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 128, 45, 80] : tensor<1x128x46x80xf16> to tensor<1x128x45x80xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType1} : tensor<1x128x45x80xf16> -> tensor<1x128x45x80x!qElemType1>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x128x45x80x!qElemType1> -> tensor<1x128x45x80xf16>

    return %4 : tensor<1x128x45x80xf16>


    //CHECK: %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x128x46x80xf16> -> tensor<1x128x46x80x!qElemType>
    //CHECK: %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x128x46x80x!qElemType> -> tensor<1x128x46x80xf16>
    //CHECK: %2 = IE.Slice %1 [0, 0, 0, 0] [1, 128, 45, 80] : tensor<1x128x46x80xf16> to tensor<1x128x45x80xf16>
    //CHECK: %3 = IE.Quantize(%2) {dstElemType = !qElemType1} : tensor<1x128x45x80xf16> -> tensor<1x128x45x80x!qElemType1>
    //CHECK: %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x128x45x80x!qElemType1> -> tensor<1x128x45x80xf16>
    //CHECK: return %4 : tensor<1x128x45x80xf16>

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>

// CHECK-LABEL: @FuseQuantParamsIntoPool
func.func @FuseQuantParamsIntoPool(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %3 = IE.MaxPool(%2) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  %4 = IE.Quantize(%3) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  return %5 : tensor<1x3x16x16xf16>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.MaxPool([[VAL0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16x!qElemType>
  //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>

// CHECK-LABEL: @AvoidFuseQuantParamsIntoPool
func.func @AvoidFuseQuantParamsIntoPool(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %3 = IE.MaxPool(%2) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.Sigmoid", attrs = {}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  %4 = IE.Quantize(%3) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  return %5 : tensor<1x3x16x16xf16>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  //CHECK: [[VAL2:%.*]] = IE.MaxPool([[VAL1]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.Sigmoid", attrs = {}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  //CHECK: [[VAL3:%.*]] = IE.Quantize([[VAL2]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  //CHECK: [[VAL4:%.*]] = IE.Dequantize([[VAL3]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  //CHECK: return [[VAL4]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @FuseQuantParamsIntoConcat
func.func @FuseQuantParamsIntoConcat(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<1x2x3x4xf16>) -> tensor<1x4x3x4xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>

    %2 = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>

    %4 = IE.Concat (%1, %3) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>

    %5 = IE.Quantize(%4) {dstElemType = !qElemType}: tensor<1x4x3x4xf16> -> tensor<1x4x3x4x!qElemType>
    %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x4x3x4x!qElemType> -> tensor<1x4x3x4xf16>

    return %6 : tensor<1x4x3x4xf16>

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    //CHECK: [[VAL2:%.*]] = IE.Concat([[VAL0]], [[VAL1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x2x3x4x!qElemType>, tensor<1x2x3x4x!qElemType> -> tensor<1x4x3x4x!qElemType>
    //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x4x3x4x!qElemType> -> tensor<1x4x3x4xf16>
    //CHECK: return [[VAL3]] : tensor<1x4x3x4xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39320635328105852:128>
!qElemType1 = !quant.uniform<u8:f16, 0.39320638320025275:128>

// CHECK-LABEL: @FuseQParamsIntoConcatWithDiffInTypes
func.func @FuseQParamsIntoConcatWithDiffInTypes(%arg0: tensor<1x16x180x320xf16>, %arg1: tensor<1x16x180x320xf16>) -> tensor<1x32x180x320xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType>
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x180x320x!qElemType> -> tensor<1x16x180x320xf16>

  %2 = IE.Quantize(%arg1) {dstElemType = !qElemType1} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType1>
  %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x16x180x320x!qElemType1> -> tensor<1x16x180x320xf16>

  %4 = IE.Concat(%1, %3) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x16x180x320xf16>, tensor<1x16x180x320xf16> -> tensor<1x32x180x320xf16>

  %5 = IE.Quantize(%4) {dstElemType = !qElemType1} : tensor<1x32x180x320xf16> -> tensor<1x32x180x320x!qElemType1>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x32x180x320x!qElemType1> -> tensor<1x32x180x320xf16>
  return %6 : tensor<1x32x180x320xf16>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x16x180x320x!qElemType> -> tensor<1x16x180x320xf16>
  //CHECK: [[VAL2:%.*]] = IE.Quantize(%arg1) {dstElemType = !qElemType1} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType1>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x16x180x320x!qElemType1> -> tensor<1x16x180x320xf16>
  //CHECK: [[VAL4:%.*]] = IE.Concat([[VAL1]], [[VAL3]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x16x180x320xf16>, tensor<1x16x180x320xf16> -> tensor<1x32x180x320xf16>
  //CHECK: [[VAL5:%.*]] = IE.Quantize([[VAL4]]) {dstElemType = !qElemType1} : tensor<1x32x180x320xf16> -> tensor<1x32x180x320x!qElemType1>
  //CHECK: [[VAL6:%.*]] = IE.Dequantize([[VAL5]]) {dstElemType = f16} : tensor<1x32x180x320x!qElemType1> -> tensor<1x32x180x320xf16>
  //CHECK: return [[VAL6]] : tensor<1x32x180x320xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @FuseQuantParamsIntoGroupConv
func.func @FuseQuantParamsIntoGroupConv(%arg0: tensor<1x3x10x10xf16>) -> tensor<1x3x10x10xf16> {
    %cst = const.Declare tensor<3x1x3x3x!qElemType> = dense<2.000000e+00> : tensor<3x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x10x10x!qElemType> -> tensor<1x3x10x10xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x1x3x3x!qElemType> -> tensor<3x1x3x3xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 3 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x10x10xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x10x10xf16>

    %4 = IE.Quantize(%3) {dstElemType = !qElemType} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType>
    %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x10x10x!qElemType> -> tensor<1x3x10x10xf16>

    return %5 : tensor<1x3x10x10xf16>

    //CHECK: [[CST:%.*]] = const.Declare tensor<3x1x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<3x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.GroupConvolution([[VAL0]], [[CST]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x10x10x!qElemType>, tensor<3x1x3x3x!qElemType> -> tensor<1x3x10x10x!qElemType>
    //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x10x10x!qElemType> -> tensor<1x3x10x10xf16>

    //CHECK: return [[VAL2]] : tensor<1x3x10x10xf16>
}

// -----

!qElemType = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @NoFuseWhenAnyOutputNotQuantized
func.func @NoFuseWhenAnyOutputNotQuantized(%arg0: tensor<1x3x16x16xf16>) -> (tensor<1x3x14x14xf16>, tensor<1x3x14x14xf16>) {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>
  return %6, %4 : tensor<1x3x14x14xf16>, tensor<1x3x14x14xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<3x3x3x3xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  //CHECK: [[VAL4:%.*]] = IE.Convolution([[VAL2]], [[VAL3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  //CHECK: [[VAL5:%.*]] = IE.Quantize([[VAL4]])  {dstElemType = !qElemType2} : tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  //CHECK: [[VAL6:%.*]] = IE.Dequantize([[VAL5]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>
  //CHECK: return [[VAL6]], [[VAL4]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<u8:f16, 2.4627450980392158:128>

// CHECK-LABEL: @NoFuseQuantParamsIntoTileWhenInputFQNotEqualsOutputFQ
func.func @NoFuseQuantParamsIntoTileWhenInputFQNotEqualsOutputFQ(%arg0: tensor<1x64x1x1xf16>) -> tensor<1x64x11x11xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x64x1x1x!qElemType> -> tensor<1x64x1x1xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16> -> tensor<1x64x11x11xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType1} : tensor<1x64x11x11xf16> -> tensor<1x64x11x11x!qElemType1>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x64x11x11x!qElemType1> -> tensor<1x64x11x11xf16>

    return %4 : tensor<1x64x11x11xf16>

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x64x1x1x!qElemType> -> tensor<1x64x1x1xf16>
    //CHECK: [[VAL2:%.*]] = IE.Tile([[VAL1]]) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16> -> tensor<1x64x11x11xf16>
    //CHECK: [[VAL3:%.*]] = IE.Quantize([[VAL2]])  {dstElemType = !qElemType1} : tensor<1x64x11x11xf16> -> tensor<1x64x11x11x!qElemType1>
    //CHECK: [[VAL4:%.*]] = IE.Dequantize([[VAL3]]) {dstElemType = f16} : tensor<1x64x11x11x!qElemType1> -> tensor<1x64x11x11xf16>
    //CHECK: return [[VAL4]] : tensor<1x64x11x11xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @FuseQuantParamsIntoTile
func.func @FuseQuantParamsIntoTile(%arg0: tensor<1x64x1x1xf16>) -> tensor<1x64x11x11xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x64x1x1x!qElemType> -> tensor<1x64x1x1xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16> -> tensor<1x64x11x11xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType} : tensor<1x64x11x11xf16> -> tensor<1x64x11x11x!qElemType>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x64x11x11x!qElemType> -> tensor<1x64x11x11xf16>

    return %4 : tensor<1x64x11x11xf16>

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0)  {dstElemType = !qElemType} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.Tile([[VAL0]]) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1x!qElemType> -> tensor<1x64x11x11x!qElemType>
    //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x64x11x11x!qElemType> -> tensor<1x64x11x11xf16>
    //CHECK: return [[VAL2]] : tensor<1x64x11x11xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128}>

// CHECK-LABEL: @DoNotFusePerAxisQuantParamsIntoSlice
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @DoNotFusePerAxisQuantParamsIntoSlice(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x8xf16> {
    %quantize0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %dequantize0 = IE.Dequantize(%quantize0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %slice = IE.Slice %dequantize0 [0, 0, 0, 8] [1, 3, 16, 8] : tensor<1x3x16x16xf16> to tensor<1x3x16x8xf16>
    %quantize1 = IE.Quantize(%slice) {dstElemType = !qElemType}: tensor<1x3x16x8xf16> -> tensor<1x3x16x8x!qElemType>
    %dequantize1 = IE.Dequantize(%quantize1) {dstElemType = f16} : tensor<1x3x16x8x!qElemType> -> tensor<1x3x16x8xf16>

    return %dequantize1 : tensor<1x3x16x8xf16>

    //CHECK: [[QUANT0:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    //CHECK: [[DEQUANT0:%.+]] = IE.Dequantize([[QUANT0]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    //CHECK: [[SLICE:%.+]] = IE.Slice [[DEQUANT0]] [0, 0, 0, 8] [1, 3, 16, 8] : tensor<1x3x16x16xf16> to tensor<1x3x16x8xf16>
    //CHECK: [[QUANT1:%.+]] = IE.Quantize([[SLICE]]) {dstElemType = !qElemType} : tensor<1x3x16x8xf16> -> tensor<1x3x16x8x!qElemType>
    //CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16} : tensor<1x3x16x8x!qElemType> -> tensor<1x3x16x8xf16>
    //CHECK: return [[DEQUANT1]] : tensor<1x3x16x8xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DoNotFusePerAxisQuantParamsIntoTile
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x4x1x1xf16>)
func.func @DoNotFusePerAxisQuantParamsIntoTile(%arg0: tensor<1x4x1x1xf16>) -> tensor<1x4x11x11xf16> {
    %quantize0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4x1x1xf16> -> tensor<1x4x1x1x!qElemType>
    %dequantize0 = IE.Dequantize(%quantize0) {dstElemType = f16} : tensor<1x4x1x1x!qElemType> -> tensor<1x4x1x1xf16>
    %tile = IE.Tile(%dequantize0) {repeats_values = [1, 1, 11, 11]} : tensor<1x4x1x1xf16> -> tensor<1x4x11x11xf16>
    %quantize1 = IE.Quantize(%tile) {dstElemType = !qElemType} : tensor<1x4x11x11xf16> -> tensor<1x4x11x11x!qElemType>
    %dequantize1 = IE.Dequantize(%quantize1) {dstElemType = f16} : tensor<1x4x11x11x!qElemType> -> tensor<1x4x11x11xf16>

    return %dequantize1 : tensor<1x4x11x11xf16>

    //CHECK: [[QUANT0:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x4x1x1xf16> -> tensor<1x4x1x1x!qElemType>
    //CHECK: [[DEQUANT0:%.+]] = IE.Dequantize([[QUANT0]]) {dstElemType = f16} : tensor<1x4x1x1x!qElemType> -> tensor<1x4x1x1xf16>
    //CHECK: [[TILE:%.+]] = IE.Tile([[DEQUANT0]]) {repeats_values = [1, 1, 11, 11]} : tensor<1x4x1x1xf16> -> tensor<1x4x11x11xf16>
    //CHECK: [[QUANT1:%.+]] = IE.Quantize([[TILE]]) {dstElemType = !qElemType} : tensor<1x4x11x11xf16> -> tensor<1x4x11x11x!qElemType>
    //CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16} : tensor<1x4x11x11x!qElemType> -> tensor<1x4x11x11xf16>
    //CHECK: return [[DEQUANT1]] : tensor<1x4x11x11xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128}>
!qElemType1 = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DoNotFusePerAxisQuantParamsIntoConcat
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x2x3x4xf16>, [[ARG1:%.+]]: tensor<1x2x3x4xf16>)
func.func @DoNotFusePerAxisQuantParamsIntoConcat(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<1x2x3x4xf16>) -> tensor<1x4x3x4xf16> {
    %quantize0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    %dequantize0 = IE.Dequantize(%quantize0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>

    %quantize1 = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    %dequantize1 = IE.Dequantize(%quantize1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>

    %concat = IE.Concat (%dequantize0, %dequantize1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>

    %quantize2 = IE.Quantize(%concat) {dstElemType = !qElemType1}: tensor<1x4x3x4xf16> -> tensor<1x4x3x4x!qElemType1>
    %dequantize2 = IE.Dequantize(%quantize2) {dstElemType = f16} : tensor<1x4x3x4x!qElemType1> -> tensor<1x4x3x4xf16>

    return %dequantize2 : tensor<1x4x3x4xf16>

    //CHECK: [[QUANT0:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    //CHECK: [[DEQUANT0:%.+]] = IE.Dequantize([[QUANT0]]) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    //CHECK: [[QUANT1:%.+]] = IE.Quantize([[ARG1]]) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    //CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    //CHECK: [[CONCAT:%.+]] = IE.Concat([[DEQUANT0]], [[DEQUANT1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    //CHECK: [[QUANT2:%.+]] = IE.Quantize([[CONCAT]]) {dstElemType = !qElemType1} : tensor<1x4x3x4xf16> -> tensor<1x4x3x4x!qElemType1>
    //CHECK: [[DEQUANT2:%.+]] = IE.Dequantize([[QUANT2]]) {dstElemType = f16} : tensor<1x4x3x4x!qElemType1> -> tensor<1x4x3x4xf16>
    //CHECK: return [[DEQUANT2]] : tensor<1x4x3x4xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DoNotFusePerChannelEltwiseWithPostOp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x4x16x16x!qElemType>, [[ARG1:%.+]]: tensor<1x4x16x16x!qElemType>)
func.func @DoNotFusePerChannelEltwiseWithPostOp(%arg0: tensor<1x4x16x16x!qElemType>, %arg1: tensor<1x4x16x16x!qElemType>) -> tensor<1x4x16x16x!qElemType> {
    %dequantize0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
    %dequantize1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
    %add = IE.Add(%dequantize0, %dequantize1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 2.500000e-01 : f64}>} : tensor<1x4x16x16xf16>, tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    %quantize = IE.Quantize(%add) {dstElemType = !qElemType}: tensor<1x4x16x16xf16> -> tensor<1x4x16x16x!qElemType>

    return %quantize : tensor<1x4x16x16x!qElemType>

    //CHECK:  [[DEQUANT0:%.+]] = IE.Dequantize([[ARG0]])
    //CHECK:  [[DEQUANT1:%.+]] = IE.Dequantize([[ARG1]])
    //CHECK:  [[ADD:%.+]] = IE.Add([[DEQUANT0]], [[DEQUANT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 2.500000e-01 : f64}>} : tensor<1x4x16x16xf16>, tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    //CHECK:  [[QUANT:%.+]] = IE.Quantize([[ADD]])
    //CHECK:  return [[QUANT]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128}>

// CHECK-LABEL: @DoNotFuseConvWithPostOp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @DoNotFuseConvWithPostOp(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %quantize0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %dequantize0 = IE.Dequantize(%quantize0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %dequantize1 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  %convolution = IE.Convolution(%dequantize0, %dequantize1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 2.500000e-01 : f64}>, strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %quantize2 = IE.Quantize(%convolution) {dstElemType = !qElemType}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType>
  %dequantize2 = IE.Dequantize(%quantize2) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>

  return %dequantize2 : tensor<1x3x14x14xf16>

  //CHECK:  [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.000000e+00> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  //CHECK:  [[QUANT0:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  //CHECK:  [[DEQUANT0:%.+]] = IE.Dequantize([[QUANT0]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  //CHECK:  [[DEQUANT1:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  //CHECK:  [[CONVOLUTION:%.+]] = IE.Convolution([[DEQUANT0]], [[DEQUANT1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 2.500000e-01 : f64}>, strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  //CHECK:  [[QUANT2:%.+]] = IE.Quantize([[CONVOLUTION]]) {dstElemType = !qElemType} : tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType>
  //CHECK:  [[DEQUANT2:%.+]] = IE.Dequantize([[QUANT2]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>

  //CHECK:  return [[DEQUANT2]] : tensor<1x3x14x14xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.051102941176470587:121>
!qElemType1 = !quant.uniform<u8:f16, 0.034064797794117647:55>

// CHECK-LABEL: @LReluQuantMultipleConsumers
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x1x128x512x!qElemType>)
func.func @LReluQuantMultipleConsumers(%arg0: tensor<1x1x128x512x!qElemType>) -> (tensor<1x1x128x512x!qElemType1>, tensor<1x1x128x512x!qElemType1>) {
    %cst = const.Declare tensor<1x1x128x512x!qElemType> = dense<2.000000e+00> : tensor<1x1x128x512xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x512x!qElemType> -> tensor<1x1x128x512xf16>
    %1 = IE.LeakyRelu(%0) {negative_slope = 0.300048828125 : f64} : tensor<1x1x128x512xf16> -> tensor<1x1x128x512xf16>
    %2 = IE.Quantize(%1) {dstElemType = !qElemType1 } : tensor<1x1x128x512xf16> -> tensor<1x1x128x512x!qElemType1>
    %3 = IE.Add(%1, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x128x512xf16>, tensor<1x1x128x512x!qElemType> -> tensor<1x1x128x512x!qElemType1>
    return %2, %3 : tensor<1x1x128x512x!qElemType1>, tensor<1x1x128x512x!qElemType1>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x1x128x512x!qElemType> = dense<2.000000e+00> : tensor<1x1x128x512xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    //CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[INPUT]]) {dstElemType = f16} : tensor<1x1x128x512x!qElemType> -> tensor<1x1x128x512xf16>
    //CHECK: [[LEAKYRELU:%.+]] = IE.LeakyRelu([[DEQUANT]]) {negative_slope = 0.300048828125 : f64} : tensor<1x1x128x512xf16> -> tensor<1x1x128x512xf16>
    //CHECK: [[QUANT:%.+]] = IE.Quantize([[LEAKYRELU]]) {dstElemType = !qElemType1} : tensor<1x1x128x512xf16> -> tensor<1x1x128x512x!qElemType1>
    //CHECK: [[ADD:%.+]] = IE.Add([[LEAKYRELU]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x128x512xf16>, tensor<1x1x128x512x!qElemType> -> tensor<1x1x128x512x!qElemType1>
    //CHECK: return [[QUANT]], [[ADD]] : tensor<1x1x128x512x!qElemType1>, tensor<1x1x128x512x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>

func.func @FuseQuantParamsIntoAvgPoolAsymmetricKernel(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x15x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %3 = IE.AvgPool(%2) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x15x14xf16>
  %4 = IE.Quantize(%3) {dstElemType = !qElemType} : tensor<1x3x15x14xf16> -> tensor<1x3x15x14x!qElemType>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x15x14x!qElemType> -> tensor<1x3x15x14xf16>
  return %5 : tensor<1x3x15x14xf16>

  // CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  // CHECK: [[VAL1:%.*]] = IE.AvgPool([[VAL0]]) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x15x14x!qElemType>
  // CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x15x14x!qElemType> -> tensor<1x3x15x14xf16>
  // CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>

func.func @FuseQuantParamsIntoAvgPoolSymmetricKernel(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %3 = IE.AvgPool(%2) {kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x14x14xf16>
  %4 = IE.Quantize(%3) {dstElemType = !qElemType} : tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
  return %5 : tensor<1x3x14x14xf16>

  // CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  // CHECK: [[VAL1:%.*]] = IE.AvgPool([[VAL0]]) {kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x14x14x!qElemType>
  // CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
  // CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039117518593283261>
!qElemType1 = !quant.uniform<u8:f16, 0.0039005478223164878>

func.func @DoNotFuseQuantParamsIntoAvgPoolWithExcludePadsAttr(%arg0: tensor<1x3x135x240x!qElemType>) -> tensor<1x3x68x120x!qElemType1> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x135x240x!qElemType> -> tensor<1x3x135x240xf16>
  %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x3x135x240xf16> -> tensor<1x3x68x120xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType1} : tensor<1x3x68x120xf16> -> tensor<1x3x68x120x!qElemType1>
  return %2 : tensor<1x3x68x120x!qElemType1>

  // CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize(%arg0) {
  // CHECK-SAME:      dstElemType = f16
  // CHECK-SAME:      } : tensor<1x3x135x240x!qElemType> -> tensor<1x3x135x240xf16>
  // CHECK: [[AVGPOOL:%.*]] = IE.AvgPool([[DEQUANTIZE]]) {exclude_pads, kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x3x135x240xf16> -> tensor<1x3x68x120xf16>
  // CHECK: [[RESULT:%.*]] = IE.Quantize([[AVGPOOL]]) {
  // CHECK-SAME:      dstElemType = !qElemType1
  // CHECK-SAME:      } : tensor<1x3x68x120xf16> -> tensor<1x3x68x120x!qElemType1>
  // CHECK: return [[RESULT]] : tensor<1x3x68x120x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8<1:255>:f16:0, {1.010680671751968504:128,1.0081200787401574797:128,1.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.0534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 3.1368405211205576E-7>

// CHECK-LABEL: @NotFuseQuantParamsIntoConvForInvalidApproximation
func.func @NotFuseQuantParamsIntoConvForInvalidApproximation(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  return %6 : tensor<1x3x14x14xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<3x3x3x3xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  //CHECK: [[VAL4:%.*]] = IE.Convolution([[VAL2]], [[VAL3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  //CHECK: [[VAL5:%.*]] = IE.Quantize([[VAL4]]) {dstElemType = !qElemType2} : tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  //CHECK: [[VAL6:%.*]] = IE.Dequantize([[VAL5]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  //CHECK: return [[VAL6]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 3.1368405211205576E-7>

// CHECK-LABEL: @NotFuseQuantParamsIntoGroupConvForInvalidApproximation
func.func @NotFuseQuantParamsIntoGroupConvForInvalidApproximation(%arg0: tensor<1x3x10x10xf16>) -> tensor<1x3x10x10xf16> {
    %cst = const.Declare tensor<3x1x3x3x!qElemType> = dense<2.000000e+00> : tensor<3x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x10x10x!qElemType> -> tensor<1x3x10x10xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x1x3x3x!qElemType> -> tensor<3x1x3x3xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 3 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x10x10xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x10x10xf16>

    %4 = IE.Quantize(%3) {dstElemType = !qElemType1} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType1>
    %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x10x10x!qElemType1> -> tensor<1x3x10x10xf16>

    return %5 : tensor<1x3x10x10xf16>

    //CHECK: [[CST:%.*]] = const.Declare tensor<3x1x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<3x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x3x10x10x!qElemType> -> tensor<1x3x10x10xf16>
    //CHECK: [[VAL2:%.*]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x1x3x3x!qElemType> -> tensor<3x1x3x3xf16>
    //CHECK: [[VAL3:%.*]] = IE.GroupConvolution([[VAL1]], [[VAL2]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x10x10xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x10x10xf16>
    //CHECK: [[VAL4:%.*]] = IE.Quantize([[VAL3]]) {dstElemType = !qElemType1} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType1>
    //CHECK: [[VAL5:%.*]] = IE.Dequantize([[VAL4]]) {dstElemType = f16} : tensor<1x3x10x10x!qElemType1> -> tensor<1x3x10x10xf16>

    //CHECK: return [[VAL5]] : tensor<1x3x10x10xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseGroupConvolutionWithChannelsAndDilation
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x72x56x56xf16, {order = #NHWC}>
func.func @FuseGroupConvolutionWithChannelsAndDilation(%arg0: tensor<1x72x56x56xf16, {order = #NHWC}>) -> tensor<1x72x28x28xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<72x1x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<72x1x3x3xf16>, [#const.Reorder<#NHWC>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %bias = const.Declare tensor<1x72x1x1xf16> = dense<1.0> : tensor<1x72x1x1xf16>

    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x72x56x56xf16, {order = #NHWC}> -> tensor<1x72x56x56x!qElemType, {order = #NHWC}>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x72x56x56x!qElemType, {order = #NHWC}> -> tensor<1x72x56x56xf16, {order = #NHWC}>

    %3 = IE.Dequantize(%filter) {dstElemType = f16} : tensor<72x1x3x3x!qElemType, {order = #NHWC}> -> tensor<72x1x3x3xf16, {order = #NHWC}>

    %4 = IE.GroupConvolution(%2, %3, %bias) {
        dilations = [3, 3], groups = 72, pads_begin = [3, 3], pads_end = [3, 3], strides = [2, 2]
    } : tensor<1x72x56x56xf16, {order = #NHWC}>, tensor<72x1x3x3xf16, {order = #NHWC}>, tensor<1x72x1x1xf16> -> tensor<1x72x28x28xf16, {order = #NHWC}>
    %5 = IE.Quantize(%4) {dstElemType = !qElemType} : tensor<1x72x28x28xf16, {order = #NHWC}> -> tensor<1x72x28x28x!qElemType, {order = #NHWC}>
    %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x72x28x28x!qElemType, {order = #NHWC}> -> tensor<1x72x28x28xf16, {order = #NHWC}>

    return %6 : tensor<1x72x28x28xf16, {order = #NHWC}>
    // CHECK:       [[FILTER:%.+]] = const.Declare tensor<72x1x3x3x!qElemType, {order = #NHWC}>
    // CHECK:       [[BIAS:%.+]]  = const.Declare tensor<1x72x1x1xf16> = dense<1.000000e+00> : tensor<1x72x1x1xf16>
    // CHECK:       [[QUANTINPUT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x72x56x56xf16,
    // CHECK-SAME:  {order = #NHWC}> -> tensor<1x72x56x56x!qElemType, {order = #NHWC}>
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[QUANTINPUT]], [[FILTER]], [[BIAS]]) {dilations = [3, 3], groups = 72 : i64,
    // CHECK-SAME:  pads_begin = [3, 3], pads_end = [3, 3], strides = [2, 2]} : tensor<1x72x56x56x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:  tensor<72x1x3x3x!qElemType, {order = #NHWC}>, tensor<1x72x1x1xf16> -> tensor<1x72x28x28x!qElemType, {order = #NHWC}>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[GROUPCONV]]) {dstElemType = f16} : tensor<1x72x28x28x!qElemType, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x72x28x28xf16, {order = #NHWC}>

    // CHECK        return [[DEQUANT]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
!qElemType1 = !quant.uniform<u8:f16, 1.0534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 3.1368405211205576E-7>

// CHECK-LABEL: @FuseQuantParamsIntoMatMul
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x8x16x16xf16>
func.func @FuseQuantParamsIntoMatMul(%arg0: tensor<1x8x16x16xf16>) -> tensor<1x8x16x32xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x8x16x16xf16> -> tensor<1x8x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x8x16x16x!qElemType1> -> tensor<1x8x16x16xf16>
  %weights = const.Declare tensor<1x8x16x32x!qElemType> = dense<1.0> : tensor<1x8x16x32xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<1x8x16x32x!qElemType> -> tensor<1x8x16x32xf16>
  %4 = IE.MatMul(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x8x16x16xf16>, tensor<1x8x16x32xf16> -> tensor<1x8x16x32xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x8x16x32xf16> -> tensor<1x8x16x32x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x8x16x32x!qElemType2> -> tensor<1x8x16x32xf16>

  return %6 : tensor<1x8x16x32xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<1x8x16x32x!qElemType> = dense<1.000000e+00> :
  // CHECK-SAME: tensor<1x8x16x32xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  // CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1}
  // CHECK-SAME: : tensor<1x8x16x16xf16> -> tensor<1x8x16x16x!qElemType1>
  // CHECK: [[VAL1:%.+]] = IE.MatMul([[VAL0]], [[CST]])
  // CHECK-SAME: : tensor<1x8x16x16x!qElemType1>, tensor<1x8x16x32x!qElemType> -> tensor<1x8x16x32x!qElemType2>
  // CHECK: [[VAL2:%.+]] = IE.Dequantize([[VAL1]]) {dstElemType = f16}
  // CHECK-SAME: : tensor<1x8x16x32x!qElemType2> -> tensor<1x8x16x32xf16>
  // CHECK: return [[VAL2]] : tensor<1x8x16x32xf16>
}
