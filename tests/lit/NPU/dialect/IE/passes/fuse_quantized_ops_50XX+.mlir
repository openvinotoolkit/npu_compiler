//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --fuse-quantized-ops %s | FileCheck %s
// REQUIRES: arch-NPU50XX

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e-01>
// CHECK:  !qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16, 1.000000e-01>

// CHECK-LABEL: TestFuseQuantParamsIntoReduceOp
module @TestFuseQuantParamsIntoReduceOp {

  config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
        config.Option @config.ReduceSupported : true
  }

  // CHECK-LABEL:    func.func @FuseQuantParamsIntoReduceMean
  // CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x30x30xf16>)
  func.func @FuseQuantParamsIntoReduceMean(%arg0: tensor<1x16x30x30xf16>) -> tensor<1x1x30x30xf16> {
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x30x30xf16> -> tensor<1x16x30x30x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x30x30x!qElemType> -> tensor<1x16x30x30xf16>
    %3 = IE.ReduceMean(%2) {axes_value = [1], keep_dims} : tensor<1x16x30x30xf16> -> tensor<1x1x30x30xf16>
    %4 = IE.Quantize(%3) {dstElemType = !qElemType1} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType1>
    %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x1x30x30x!qElemType1> -> tensor<1x1x30x30xf16>
    return %5 : tensor<1x1x30x30xf16>

  // CHECK: [[VAL0:%.*]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x16x30x30xf16> -> tensor<1x16x30x30x!qElemType>
  // CHECK: [[VAL1:%.*]] = IE.ReduceMean([[VAL0]]) {axes_value = [1], keep_dims} : tensor<1x16x30x30x!qElemType> -> tensor<1x1x30x30x!qElemType1>
  // CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x1x30x30x!qElemType1> -> tensor<1x1x30x30xf16>
  // CHECK: return [[VAL2]]
  }

  // CHECK-LABEL:    func.func @FuseQuantParamsIntoReduceSum
  // CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x30x30xf16>)
  func.func @FuseQuantParamsIntoReduceSum(%arg0: tensor<1x16x30x30xf16>) -> tensor<1x1x30x30xf16> {
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x30x30xf16> -> tensor<1x16x30x30x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x30x30x!qElemType> -> tensor<1x16x30x30xf16>
    %3 = IE.ReduceSum(%2) {axes_value = [1], keep_dims} : tensor<1x16x30x30xf16> -> tensor<1x1x30x30xf16>
    %4 = IE.Quantize(%3) {dstElemType = !qElemType1} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType1>
    %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x1x30x30x!qElemType1> -> tensor<1x1x30x30xf16>
    return %5 : tensor<1x1x30x30xf16>

  // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x16x30x30xf16> -> tensor<1x16x30x30x!qElemType>
  // CHECK: [[SUM:%.+]] = IE.ReduceSum([[QUANTIZE]]) {axes_value = [1], keep_dims} : tensor<1x16x30x30x!qElemType> -> tensor<1x1x30x30x!qElemType1>
  // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[SUM]]) {dstElemType = f16} : tensor<1x1x30x30x!qElemType1> -> tensor<1x1x30x30xf16>
  // CHECK: return [[DEQUANTIZE]]
  }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0034409466911764705>
!qElemType1 = !quant.uniform<u8:f16, 0.12503063725490196:128>

// CHECK-LABEL:    func.func @FuseQuantParamsIntoEltwiseSubtract
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x3x16x16xf16>, [[INPUT1:%.+]]: tensor<1x3x16x16xf16>)
func.func @FuseQuantParamsIntoEltwiseSubtract(%arg0: tensor<1x3x16x16xf16>, %arg1: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %3 = IE.Quantize(%arg1) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %5 = IE.Subtract(%2, %4) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  %6 = IE.Quantize(%5) {dstElemType = !qElemType}: tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %7 = IE.Dequantize(%6) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

  return %7 : tensor<1x3x16x16xf16>

  //CHECK: [[VAL0:%.*]] = IE.Quantize([[INPUT0]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Quantize([[INPUT1]]) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  //CHECK: [[VAL2:%.*]] = IE.Subtract([[VAL0]], [[VAL1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16x!qElemType>, tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16x!qElemType>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  //CHECK: return [[VAL3]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0034409466911764705>
!qElemType1 = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @FusePerChannelMaxPool
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x4x16x16x!qElemType>)
func.func @FusePerChannelMaxPool(%arg0: tensor<1x4x16x16x!qElemType>) -> tensor<1x4x16x16x!qElemType1> {
  %dequantize = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
  %maxPool = IE.MaxPool(%dequantize) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
  %quantize = IE.Quantize(%maxPool) {dstElemType = !qElemType1}: tensor<1x4x16x16xf16> -> tensor<1x4x16x16x!qElemType1>

  return %quantize : tensor<1x4x16x16x!qElemType1>

  //CHECK: [[MAXPOOL:%.+]] = IE.MaxPool([[ARG0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16x!qElemType1>
  //CHECK: return [[MAXPOOL]] : tensor<1x4x16x16x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<u8:f16, -0.01013327205882353>
!qElemType2 = !quant.uniform<u8:f16, 0.39320638320025275:128>

// CHECK-LABEL: @DoNotFuseQParamsIntoSubWithNegativeScale
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x16x180x320xf16>, [[INPUT_1:%.+]]: tensor<1x16x180x320xf16>
func.func @DoNotFuseQParamsIntoSubWithNegativeScale(%arg0: tensor<1x16x180x320xf16>, %arg1: tensor<1x16x180x320xf16>) -> tensor<1x16x180x320xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType>
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x180x320x!qElemType> -> tensor<1x16x180x320xf16>

  %2 = IE.Quantize(%arg1) {dstElemType = !qElemType1} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType1>
  %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x16x180x320x!qElemType1> -> tensor<1x16x180x320xf16>

  %4 = IE.Subtract(%1, %3) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x16x180x320xf16>, tensor<1x16x180x320xf16> -> tensor<1x16x180x320xf16>

  %5 = IE.Quantize(%4) {dstElemType = !qElemType2} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x16x180x320x!qElemType2> -> tensor<1x16x180x320xf16>
  return %6 : tensor<1x16x180x320xf16>

  // CHECK: [[QUANT0:%.+]]  = IE.Quantize([[INPUT_0]]) {dstElemType = !qElemType} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType>
  // CHECK: [[DEQUANT0:%.+]] = IE.Dequantize([[QUANT0]]) {dstElemType = f16} : tensor<1x16x180x320x!qElemType> -> tensor<1x16x180x320xf16>
  // CHECK: [[QUANT1:%.+]] = IE.Quantize([[INPUT_1]]) {dstElemType = !qElemType1} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType1>
  // CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16} : tensor<1x16x180x320x!qElemType1> -> tensor<1x16x180x320xf16>
  // CHECK: [[SUB:%.+]] = IE.Subtract([[DEQUANT0]], [[DEQUANT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x180x320xf16>, tensor<1x16x180x320xf16> -> tensor<1x16x180x320xf16>
  // CHECK: [[QUANT2:%.+]] = IE.Quantize([[SUB]]) {dstElemType = !qElemType2} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType2>
  // CHECK: [[DEQUANT2:%.+]] = IE.Dequantize([[QUANT2]]) {dstElemType = f16} : tensor<1x16x180x320x!qElemType2> -> tensor<1x16x180x320xf16>
  // CHECK: return [[DEQUANT2]] : tensor<1x16x180x320xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DoNotFusePerChannelSubtract
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x4x16x16x!qElemType>, [[ARG1:%.+]]: tensor<1x4x16x16x!qElemType>)
func.func @DoNotFusePerChannelSubtract(%arg0: tensor<1x4x16x16x!qElemType>, %arg1: tensor<1x4x16x16x!qElemType>) -> tensor<1x4x16x16x!qElemType> {
    %dequantize0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
    %dequantize1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
    %sub = IE.Subtract(%dequantize0, %dequantize1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x16x16xf16>, tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    %quantize = IE.Quantize(%sub) {dstElemType = !qElemType}: tensor<1x4x16x16xf16> -> tensor<1x4x16x16x!qElemType>

    return %quantize : tensor<1x4x16x16x!qElemType>

    //CHECK:  [[DEQUANT0:%.+]] = IE.Dequantize([[ARG0]])
    //CHECK:  [[DEQUANT1:%.+]] = IE.Dequantize([[ARG1]])
    //CHECK:  [[SUB:%.+]] = IE.Subtract([[DEQUANT0]], [[DEQUANT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x16x16xf16>, tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    //CHECK:  [[QUANT:%.+]] = IE.Quantize([[SUB]])
    //CHECK:  return [[QUANT]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DoNotFusePerChannelMultiply
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x4x16x16x!qElemType>, [[ARG1:%.+]]: tensor<1x4x16x16x!qElemType>)
func.func @DoNotFusePerChannelMultiply(%arg0: tensor<1x4x16x16x!qElemType>, %arg1: tensor<1x4x16x16x!qElemType>) -> tensor<1x4x16x16x!qElemType> {
    %dequantize0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
    %dequantize1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
    %mult = IE.Multiply(%dequantize0, %dequantize1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x16x16xf16>, tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    %quantize = IE.Quantize(%mult) {dstElemType = !qElemType}: tensor<1x4x16x16xf16> -> tensor<1x4x16x16x!qElemType>

    return %quantize : tensor<1x4x16x16x!qElemType>

    //CHECK:  [[DEQUANT0:%.+]] = IE.Dequantize([[ARG0]])
    //CHECK:  [[DEQUANT1:%.+]] = IE.Dequantize([[ARG1]])
    //CHECK:  [[MUL:%.+]] = IE.Multiply([[DEQUANT0]], [[DEQUANT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x16x16xf16>, tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    //CHECK:  [[QUANT:%.+]] = IE.Quantize([[MUL]])
    //CHECK:  return [[QUANT]]
}
