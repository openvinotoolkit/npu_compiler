//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% revision-id=3" --fuse-outstanding-quant %s | FileCheck %s
// REQUIRES: platform-NPU6010
// Baseline: REVISION_B via revision-id=3.

// CHECK: !qElemType = !quant.uniform<u8:f16, 1.0000152590218967>
!qElemType = !quant.uniform<u16:f16, 5.000000e-01:2222>
!qElemType1 = !quant.uniform<u8:f16, 1.0000152590218967>

// CHECK-LABEL: func.func @SoftMaxConvolutionWithOutstandingQuantHasOneUser
// CHECK:    ([[ARG0:%.+]]: tensor<1x3x16x16xf16>) -> tensor<1x16x18x18xf16>
func.func @SoftMaxConvolutionWithOutstandingQuantHasOneUser(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x16x18x18xf16> {
  %0 = IE.SoftMax(%arg0) {
    axisInd = 1
  } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>

  %1 = IE.Quantize(%0) {
    dstElemType = !qElemType
  } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %weights = const.Declare tensor<16x3x1x1x!qElemType1> = dense<1.000000e+00> : tensor<16x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]

  %2 = IE.Convolution(%1, %weights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x16x!qElemType>, tensor<16x3x1x1x!qElemType1> -> tensor<1x16x18x18xf16>

  return %2 : tensor<1x16x18x18xf16>
  // CHECK:       [[VAL0:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<16x3x1x1x!qElemType> =
  // CHECK-SAME:  dense<1.000000e+00> : tensor<16x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]


  // CHECK-NOT:   IE.Quantize

  // CHECK:       [[CONV:%.+]] = IE.Convolution([[VAL0]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<16x3x1x1x!qElemType> -> tensor<1x16x18x18xf16>
  // CHECK:   return [[CONV]] : tensor<1x16x18x18xf16>

}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f16, 1.0000152590218967>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 5.000000e-01:2>
!qElemType = !quant.uniform<u8:f16, 1.0000152590218967>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01:2>


// CHECK-LABEL: func.func @DoNotFuseConvolutionWithOutstandingU8QuantHasOneUser
// CHECK:    ([[ARG0:%.+]]: tensor<1x3x16x16xf16>) -> tensor<1x16x18x18xf16>
func.func @DoNotFuseConvolutionWithOutstandingU8QuantHasOneUser(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x16x18x18xf16> {
  %0 = IE.SoftMax(%arg0) {
    axisInd = 1
  } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<16x3x1x1x!qElemType> = dense<1.000000e+00> : tensor<16x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %1 = IE.Quantize(%0) {
    dstElemType = !qElemType1
  } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>

  %2 = IE.Convolution(%1, %weights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x16x!qElemType1>, tensor<16x3x1x1x!qElemType> -> tensor<1x16x18x18xf16>

  return %2 : tensor<1x16x18x18xf16>
  // CHECK:  [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  // CHECK:  [[CST:%.+]] = const.Declare tensor<16x3x1x1x!qElemType> =
  // CHECK-SAME:  dense<1.000000e+00> : tensor<16x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[SOFTMAX]]) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[QUANTIZE]], [[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x16x!qElemType1>, tensor<16x3x1x1x!qElemType> -> tensor<1x16x18x18xf16>
  // CHECK:   return [[CONV]] : tensor<1x16x18x18xf16>
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f16, 5.000000e-01:2>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 1.0000152590218967>
!qElemType = !quant.uniform<u8:f16, 5.000000e-01:2>
!qElemType1 = !quant.uniform<u8:f16, 1.0000152590218967>

// CHECK-LABEL: func.func @DoNotFuseConvolutionWithOutstandingNoConstWeights
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x16x16xf16>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<16x3x1x1x!qElemType>)
func.func @DoNotFuseConvolutionWithOutstandingNoConstWeights(%arg0: tensor<1x3x16x16xf16>, %arg1: tensor<16x3x1x1x!qElemType>) -> tensor<1x16x18x18xf16> {
  %0 = IE.SoftMax(%arg0) {
    axisInd = 1
  } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>

  %1 = IE.Quantize(%0) {
    dstElemType = !qElemType1
  } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Convolution(%1, %arg1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x16x!qElemType1>, tensor<16x3x1x1x!qElemType> -> tensor<1x16x18x18xf16>

  return %2 : tensor<1x16x18x18xf16>
  // CHECK:  [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[SOFTMAX]]) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[QUANTIZE]], [[ARG1]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x16x!qElemType1>, tensor<16x3x1x1x!qElemType> -> tensor<1x16x18x18xf16>
  // CHECK-NEXT:   return [[CONV]] : tensor<1x16x18x18xf16>

}

// -----

// CHECK: !qElemType = !quant.uniform<u16:f16, 8.2007573709031714E-6:23225>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.0013098501691631243:122>
!qElemType = !quant.uniform<u16:f16, 8.2007573709031714E-6:23225>
!qElemType1 = !quant.uniform<u8:f16, 0.0013098501691631243:122>
!qElemType2 = !quant.uniform<u16:f16, 4.5571154766645928E-6:25096>

// CHECK-LABEL: func.func @DoNotFuseConvolutionWithOutstandingU16QuantAndAsymetricU16Output
// CHECK:    ([[ARG0:%.+]]: tensor<1x96x64x64xf16>) -> tensor<1x96x64x64x!qElemType
func.func @DoNotFuseConvolutionWithOutstandingU16QuantAndAsymetricU16Output(%arg0: tensor<1x96x64x64xf16>) -> tensor<1x96x64x64x!qElemType> {
  %weights = const.Declare tensor<96x96x3x3x!qElemType1> = dense<1> : tensor<96x96x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>]
  %quant = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x96x64x64xf16> -> tensor<1x96x64x64x!qElemType2>
  %conv = IE.Convolution(%quant, %weights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.Swish<beta = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x96x64x64x!qElemType2>, tensor<96x96x3x3x!qElemType1> -> tensor<1x96x64x64x!qElemType>
  return %conv : tensor<1x96x64x64x!qElemType>
  // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<96x96x3x3x!qElemType1> =
  // CHECK-SAME:  dense<1> : tensor<96x96x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>]
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WEIGHTS]])
  // CHECK-SAME: {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.Swish<beta = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x96x64x64xf16>, tensor<96x96x3x3x!qElemType1> -> tensor<1x96x64x64x!qElemType>
  // CHECK:  return [[CONV]] : tensor<1x96x64x64x!qElemType>
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f16, 1.0000152590218967>
!qElemType  = !quant.uniform<u16:f16, 5.000000e-01:2222>
!qElemType1 = !quant.uniform<u8:f16, 1.0000152590218967>

// CHECK-LABEL: func.func @SoftMaxMatMulWithOutstandingQuantHasOneUser
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x64xf16>) -> tensor<1x32xf16>
func.func @SoftMaxMatMulWithOutstandingQuantHasOneUser(%arg0: tensor<1x64xf16>) -> tensor<1x32xf16> {
  %sm = IE.SoftMax(%arg0) {
    axisInd = 1
  } : tensor<1x64xf16> -> tensor<1x64xf16>

  %quant = IE.Quantize(%sm) {
    dstElemType = !qElemType
  } : tensor<1x64xf16> -> tensor<1x64x!qElemType>

  %weights = const.Declare tensor<64x32x!qElemType1> = dense<1.000000e+00> : tensor<64x32xf16>,
      [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]

  %mm = IE.MatMul(%quant, %weights) : tensor<1x64x!qElemType>, tensor<64x32x!qElemType1> -> tensor<1x32xf16>

  return %mm: tensor<1x32xf16>

  // CHECK:      [[VAL0:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x64xf16> -> tensor<1x64xf16>
  // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<64x32x!qElemType> =
  // CHECK-SAME:  dense<1.000000e+00> : tensor<64x32xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  // CHECK-NOT:  IE.Quantize

  // CHECK:      [[MM:%.+]] = IE.MatMul([[VAL0]], [[CST]]) : tensor<1x64xf16>, tensor<64x32x!qElemType> -> tensor<1x32xf16>
  // CHECK:      return [[MM]] : tensor<1x32xf16>
}

// -----

// CHECK: !qElemType  = !quant.uniform<u8:f16, 1.0000152590218967>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 5.000000e-01:2>
!qElemType  = !quant.uniform<u8:f16, 1.0000152590218967>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01:2>

// CHECK-LABEL: func.func @DoNotFuseMatMulWithOutstandingU8QuantHasOneUser
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x64xf16>) -> tensor<1x32xf16>
func.func @DoNotFuseMatMulWithOutstandingU8QuantHasOneUser(%arg0: tensor<1x64xf16>) -> tensor<1x32xf16> {
  %sm = IE.SoftMax(%arg0) {
    axisInd = 1
  } : tensor<1x64xf16> -> tensor<1x64xf16>

  %weights = const.Declare tensor<64x32x!qElemType> = dense<1.000000e+00> : tensor<64x32xf16>,
      [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  %quant = IE.Quantize(%sm) {
    dstElemType = !qElemType1
  } : tensor<1x64xf16> -> tensor<1x64x!qElemType1>

  %mm = IE.MatMul(%quant, %weights) : tensor<1x64x!qElemType1>, tensor<64x32x!qElemType> -> tensor<1x32xf16>

  return %mm : tensor<1x32xf16>

  // CHECK:  [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x64xf16> -> tensor<1x64xf16>
  // CHECK:  [[CST:%.+]] = const.Declare tensor<64x32x!qElemType> =
  // CHECK-SAME:  dense<1.000000e+00> : tensor<64x32xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[SOFTMAX]]) {dstElemType = !qElemType1} : tensor<1x64xf16> -> tensor<1x64x!qElemType1>
  // CHECK:  [[MM:%.+]] = IE.MatMul([[QUANTIZE]], [[CST]]) : tensor<1x64x!qElemType1>, tensor<64x32x!qElemType> -> tensor<1x32xf16>
  // CHECK:  return [[MM]] : tensor<1x32xf16>
}

// -----

// CHECK: !qElemType  = !quant.uniform<u8:f16, 5.000000e-01:2>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 1.0000152590218967>
!qElemType  = !quant.uniform<u8:f16, 5.000000e-01:2>
!qElemType1 = !quant.uniform<u8:f16, 1.0000152590218967>

// CHECK-LABEL: func.func @DoNotFuseMatMulWithOutstandingNoConstWeights
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x64xf16>,
// CHECK-SAME:     [[ARG1:%.+]]: tensor<64x32x!qElemType>)
func.func @DoNotFuseMatMulWithOutstandingNoConstWeights(%arg0: tensor<1x64xf16>, %arg1: tensor<64x32x!qElemType>) -> tensor<1x32xf16> {
  %sm = IE.SoftMax(%arg0) {
    axisInd = 1
  } : tensor<1x64xf16> -> tensor<1x64xf16>

  %quant = IE.Quantize(%sm) {
    dstElemType = !qElemType1
  } : tensor<1x64xf16> -> tensor<1x64x!qElemType1>

  %mm = IE.MatMul(%quant, %arg1) : tensor<1x64x!qElemType1>, tensor<64x32x!qElemType> -> tensor<1x32xf16>

  return %mm : tensor<1x32xf16>

  // CHECK:  [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x64xf16> -> tensor<1x64xf16>
  // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[SOFTMAX]]) {dstElemType = !qElemType1} : tensor<1x64xf16> -> tensor<1x64x!qElemType1>
  // CHECK:  [[MM:%.+]] = IE.MatMul([[QUANTIZE]], [[ARG1]]) : tensor<1x64x!qElemType1>, tensor<64x32x!qElemType> -> tensor<1x32xf16>
  // CHECK-NEXT:  return [[MM]] : tensor<1x32xf16>
}

// -----

// CHECK: !qElemType  = !quant.uniform<u16:f16, 8.2007573709031714E-6:23225>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.0013098501691631243:122>
!qElemType  = !quant.uniform<u16:f16, 8.2007573709031714E-6:23225>
!qElemType1 = !quant.uniform<u8:f16, 0.0013098501691631243:122>
!qElemType2 = !quant.uniform<u16:f16, 4.5571154766645928E-6:25096>

// CHECK-LABEL: func.func @DoNotFuseMatMulWithOutstandingU16QuantAndAsymmetricU16Output
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x96xf16>) -> tensor<1x96x!qElemType>
func.func @DoNotFuseMatMulWithOutstandingU16QuantAndAsymmetricU16Output(%arg0: tensor<1x96xf16>) -> tensor<1x96x!qElemType> {
  %weights = const.Declare tensor<96x96x!qElemType1> = dense<1> : tensor<96x96xui8>,
      [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>]

  %quant = IE.Quantize(%arg0) {
    dstElemType = !qElemType2
  } : tensor<1x96xf16> -> tensor<1x96x!qElemType2>

  %mm = IE.MatMul(%quant, %weights) : tensor<1x96x!qElemType2>, tensor<96x96x!qElemType1> -> tensor<1x96x!qElemType>

  return %mm : tensor<1x96x!qElemType>

  // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<96x96x!qElemType1> =
  // CHECK-SAME:  dense<1> : tensor<96x96xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>]
  // CHECK:  [[MM:%.+]] = IE.MatMul([[ARG0]], [[WEIGHTS]]) : tensor<1x96xf16>, tensor<96x96x!qElemType1> -> tensor<1x96x!qElemType>
  // CHECK:  return [[MM]] : tensor<1x96x!qElemType>
}
