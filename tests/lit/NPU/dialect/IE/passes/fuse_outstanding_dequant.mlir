//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-outstanding-dequant %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: func.func @Conv2dSoftMaxWithOutstandingDequant
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @Conv2dSoftMaxWithOutstandingDequant(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %cst = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>

  %0 = IE.Convolution(%arg0, %cst) {
    dilations = [1, 1],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    strides = [1, 1]
  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.Dequantize(%0) {
    dstElemType = f16
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  return %2 : tensor<1x16x3x3xf16>

  // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> :
  // CHECK-SAME:   tensor<16x16x1x1xf16>

  // CHECK:      [[VAL0:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
  // CHECK-SAME:   dilations = [1, 1],
  // CHECK-SAME:   pads_begin = [0, 0],
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:  IE.Dequantize

  // CHECK:      [[VAL1:%.+]] = IE.SoftMax([[VAL0]]) {axisInd = 1 : i64} :
  // CHECK-SAME:   tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: func.func @Conv2dSoftMaxWithOutstandingDequantPerAxes
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @Conv2dSoftMaxWithOutstandingDequantPerAxes(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %cst = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>

  %0 = IE.Convolution(%arg0, %cst) {
    dilations = [1, 1],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    strides = [1, 1]
  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.Dequantize(%0) {
    dstElemType = f16
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  return %2 : tensor<1x16x3x3xf16>

  // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> :
  // CHECK-SAME:   tensor<16x16x1x1xf16>

  // CHECK:      [[VAL0:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
  // CHECK-SAME:   dilations = [1, 1],
  // CHECK-SAME:   pads_begin = [0, 0],
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:  IE.Dequantize

  // CHECK:      [[VAL1:%.+]] = IE.SoftMax([[VAL0]]) {axisInd = 1 : i64} :
  // CHECK-SAME:   tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: func.func @GroupConvSoftMaxWithOutstandingDequant
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @GroupConvSoftMaxWithOutstandingDequant(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %cst = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1x1xf16>

  %0 = IE.GroupConvolution(%arg0, %cst) {
    dilations = [1, 1],
    groups = 16 : i64,
    pads_begin = [0, 0],
    pads_end = [0, 0],
    strides = [1, 1]
  } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.Dequantize(%0) {
    dstElemType = f16
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  return %2 : tensor<1x16x3x3xf16>

  // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> :
  // CHECK-SAME:   tensor<16x1x1x1xf16>

  // CHECK:      [[VAL0:%.+]] = IE.GroupConvolution([[INPUT]], [[CST]]) {
  // CHECK-SAME:   dilations = [1, 1]
  // CHECK-SAME:   groups = 16 : i64,
  // CHECK-SAME:   pads_begin = [0, 0
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:  IE.Dequantize

  // CHECK:      [[VAL1:%.+]] = IE.SoftMax([[VAL0]]) {axisInd = 1 : i64} :
  // CHECK-SAME:   tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: func.func @GroupConvSoftMaxWithOutstandingDequantPerAxes
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @GroupConvSoftMaxWithOutstandingDequantPerAxes(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %cst = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1x1xf16>

  %0 = IE.GroupConvolution(%arg0, %cst) {
    dilations = [1, 1],
    groups = 16 : i64,
    pads_begin = [0, 0],
    pads_end = [0, 0],
    strides = [1, 1]
  } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.Dequantize(%0) {
    dstElemType = f16
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  return %2 : tensor<1x16x3x3xf16>

  // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> :
  // CHECK-SAME:   tensor<16x1x1x1xf16>

  // CHECK:      [[VAL0:%.+]] = IE.GroupConvolution([[INPUT]], [[CST]]) {
  // CHECK-SAME:   dilations = [1, 1]
  // CHECK-SAME:   groups = 16 : i64,
  // CHECK-SAME:   pads_begin = [0, 0
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:  IE.Dequantize

  // CHECK:      [[VAL1:%.+]] = IE.SoftMax([[VAL0]]) {axisInd = 1 : i64} :
  // CHECK-SAME:   tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: func.func @AvgPoolSoftMaxWithOutstandingDequant
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @AvgPoolSoftMaxWithOutstandingDequant(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %0 = IE.AvgPool(%arg0) {
    kernel_size = [1, 1],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    rounding_type = #IE.rounding_type<FLOOR>,
    strides = [1, 1]
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.Dequantize(%0) {
    dstElemType = f16
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  return %2 : tensor<1x16x3x3xf16>

  // CHECK:       [[VAL0:%.+]] = IE.AvgPool([[INPUT]]) {
  // CHECK-SAME:    kernel_size = [1, 1],
  // CHECK-SAME:    pads_begin = [0, 0],
  // CHECK-SAME:    pads_end = [0, 0],
  // CHECK-SAME:    rounding_type = #IE.rounding_type<FLOOR>,
  // CHECK-SAME:    strides = [1, 1]
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[VAL1:%.+]] = IE.SoftMax([[VAL0]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @AddSoftMaxWithOutstandingDequant
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @AddSoftMaxWithOutstandingDequant(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.Dequantize(%0) {
    dstElemType = f16
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  return %2 : tensor<1x16x3x3xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[VAL1:%.+]] = IE.SoftMax([[VAL0]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: func.func @AddSoftMaxWithOutstandingDequantPerAxes
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @AddSoftMaxWithOutstandingDequantPerAxes(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.Dequantize(%0) {
    dstElemType = f16
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  return %2 : tensor<1x16x3x3xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[VAL1:%.+]] = IE.SoftMax([[VAL0]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @AddAffineReshapeWithOutstandingDequant
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x2x48x25xf16>) -> tensor<2x48x5x5xf16>
func.func @AddAffineReshapeWithOutstandingDequant(%arg0: tensor<1x2x48x25xf16>) -> tensor<2x48x5x5xf16> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25x!qElemType>

  %1 = IE.AffineReshape(%0) {
    dim_mapping = [[0], [0], [1], [2, 3]],
    shape_value = [2, 48, 5, 5]
  } : tensor<1x2x48x25x!qElemType> -> tensor<2x48x5x5x!qElemType>

  %2 = IE.Dequantize(%1) {
    dstElemType = f16
  } : tensor<2x48x5x5x!qElemType> -> tensor<2x48x5x5xf16>

  %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<2x48x5x5xf16> -> tensor<2x48x5x5xf16>

  return %3 : tensor<2x48x5x5xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25xf16>

  // CHECK:       [[VAL1:%.+]] = IE.AffineReshape([[VAL0]]) {
  // CHECK-SAME{LITERAL}:dim_mapping = [[0], [0], [1], [2, 3]],
  // CHECK-SAME:    shape_value = [2, 48, 5, 5]
  // CHECK-SAME:  } : tensor<1x2x48x25xf16> -> tensor<2x48x5x5xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[VAL2:%.+]] = IE.SoftMax([[VAL1]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<2x48x5x5xf16> -> tensor<2x48x5x5xf16>

  // CHECK:       return [[VAL2]] : tensor<2x48x5x5xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @AddAffineReshapeReshapeWithOutstandingDequant
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x2x48x25xf16>) -> tensor<2x48x25xf16>
func.func @AddAffineReshapeReshapeWithOutstandingDequant(%arg0: tensor<1x2x48x25xf16>) -> tensor<2x48x25xf16> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25x!qElemType>

  %1 = IE.AffineReshape(%0) {
    dim_mapping = [[0], [0], [1], [2, 3]],
    shape_value = [2, 48, 5, 5]
  } : tensor<1x2x48x25x!qElemType> -> tensor<2x48x5x5x!qElemType>

  %2 = IE.Reshape(%1) {
    shape_value = [2, 48, 25]
  } : tensor<2x48x5x5x!qElemType> -> tensor<2x48x25x!qElemType>

  %3 = IE.Dequantize(%2) {
    dstElemType = f16
  } : tensor<2x48x25x!qElemType> -> tensor<2x48x25xf16>

  %4 = IE.SoftMax(%3) {axisInd = 1} : tensor<2x48x25xf16> -> tensor<2x48x25xf16>

  return %4 : tensor<2x48x25xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25xf16>

  // CHECK:       [[VAL1:%.+]] = IE.AffineReshape([[VAL0]]) {
  // CHECK-SAME{LITERAL}:dim_mapping = [[0], [0], [1], [2, 3]],
  // CHECK-SAME:    shape_value = [2, 48, 5, 5]
  // CHECK-SAME:  } : tensor<1x2x48x25xf16> -> tensor<2x48x5x5xf16>

  // CHECK:       [[VAL2:%.+]] = IE.Reshape([[VAL1]]) {
  // CHECK-SAME:    shape_value = [2, 48, 25]
  // CHECK-SAME:  } : tensor<2x48x5x5xf16> -> tensor<2x48x25xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[VAL3:%.+]] = IE.SoftMax([[VAL2]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<2x48x25xf16> -> tensor<2x48x25xf16>

  // CHECK:       return [[VAL3]] : tensor<2x48x25xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.5>
!qElemType1 = !quant.uniform<u8:f16, 0.25>

// CHECK-LABEL: func.func @AddQuantizeCastReshapeDequantNotRemove
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x2x48x25xf16>) -> tensor<2x48x25xf16>
func.func @AddQuantizeCastReshapeDequantNotRemove(%arg0: tensor<1x2x48x25xf16>) -> tensor<2x48x25xf16> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25x!qElemType>

  %1 = IE.QuantizeCast(%0) {
    dstElemType = !qElemType1
  } : tensor<1x2x48x25x!qElemType> -> tensor<1x2x48x25x!qElemType1>

  %2 = IE.Reshape(%1) {
    shape_value = [2, 48, 25]
  } : tensor<1x2x48x25x!qElemType1> -> tensor<2x48x25x!qElemType1>

  %3 = IE.Dequantize(%2) {
    dstElemType = f16
  } : tensor<2x48x25x!qElemType1> -> tensor<2x48x25xf16>

  %4 = IE.SoftMax(%3) {axisInd = 1} : tensor<2x48x25xf16> -> tensor<2x48x25xf16>

  return %4 : tensor<2x48x25xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25x!qElemType>

  // CHECK:       [[VAL1:%.+]] = IE.QuantizeCast([[VAL0]]) {
  // CHECK-SAME:    dstElemType = !qElemType1
  // CHECK-SAME:  } : tensor<1x2x48x25x!qElemType> -> tensor<1x2x48x25x!qElemType1>

  // CHECK:       [[VAL2:%.+]] = IE.Reshape([[VAL1]]) {
  // CHECK-SAME:    shape_value = [2, 48, 25]
  // CHECK-SAME:  } : tensor<1x2x48x25x!qElemType1> -> tensor<2x48x25x!qElemType1>

  // CHECK:       [[VAL3:%.+]] = IE.Dequantize([[VAL2]]) {
  // CHECK-SAME:    dstElemType = f16
  // CHECK-SAME:  } : tensor<2x48x25x!qElemType1> -> tensor<2x48x25xf16>

  // CHECK:       [[VAL4:%.+]] = IE.SoftMax([[VAL3]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<2x48x25xf16> -> tensor<2x48x25xf16>

  // CHECK:       return [[VAL4]] : tensor<2x48x25xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @AddConcatDequantNotRemove
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x2x48x25xf16>) -> tensor<1x2x48x50xf16>
func.func @AddConcatDequantNotRemove(%arg0: tensor<1x2x48x25xf16>) -> tensor<1x2x48x50xf16> {
  %cst = const.Declare tensor<1x2x48x25x!qElemType> = dense<1.0> :
    tensor<1x2x48x25xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25x!qElemType>

  %1 = IE.Concat(%cst, %0) {
    per_axis = #IE.Concat<axis = 3 : i64>
  } : tensor<1x2x48x25x!qElemType>, tensor<1x2x48x25x!qElemType> -> tensor<1x2x48x50x!qElemType>

  %2 = IE.Dequantize(%1) {
    dstElemType = f16
  } : tensor<1x2x48x50x!qElemType> -> tensor<1x2x48x50xf16>

  %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<1x2x48x50xf16> -> tensor<1x2x48x50xf16>

  return %3 : tensor<1x2x48x50xf16>

  // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<1x2x48x25x!qElemType> = dense<1.000000e+00> :
  // CHECK-SAME:   tensor<1x2x48x25xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  // CHECK:       [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25x!qElemType>

  // CHECK:       [[VAL1:%.+]] = IE.Concat([[CST]], [[VAL0]]) {
  // CHECK-SAME:    per_axis = #IE.Concat<axis = 3 : i64>
  // CHECK-SAME:  } : tensor<1x2x48x25x!qElemType>, tensor<1x2x48x25x!qElemType> -> tensor<1x2x48x50x!qElemType>

  // CHECK:       [[VAL2:%.+]] = IE.Dequantize([[VAL1]]) {
  // CHECK-SAME:    dstElemType = f16
  // CHECK-SAME:  } : tensor<1x2x48x50x!qElemType> -> tensor<1x2x48x50xf16>

  // CHECK:       [[VAL3:%.+]] = IE.SoftMax([[VAL2]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x2x48x50xf16> -> tensor<1x2x48x50xf16>

  // CHECK:       return [[VAL3]] : tensor<1x2x48x50xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @AddSplitDequantHasOneUse
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x2x48x25xf16>) -> tensor<1x1x48x25xf16>
func.func @AddSplitDequantHasOneUse(%arg0: tensor<1x2x48x25xf16>) -> tensor<1x1x48x25xf16> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25x!qElemType>

  %1:2 = IE.Split(%0) {
    axis_value = 1 : i64, num_splits = 2 : i64
  } : tensor<1x2x48x25x!qElemType> -> tensor<1x1x48x25x!qElemType>, tensor<1x1x48x25x!qElemType>

  %2 = IE.Dequantize(%1#0) {
    dstElemType = f16
  } : tensor<1x1x48x25x!qElemType> -> tensor<1x1x48x25xf16>

  %3 = IE.SoftMax(%2) {axisInd = 2} : tensor<1x1x48x25xf16> -> tensor<1x1x48x25xf16>

  return %3 : tensor<1x1x48x25xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25xf16>

  // CHECK:       [[VAL1:%.+]]:2 = IE.Split([[VAL0]]) {
  // CHECK-SAME:    axis_value = 1 : i64, num_splits = 2 : i64
  // CHECK-SAME:  } : tensor<1x2x48x25xf16> -> tensor<1x1x48x25xf16>, tensor<1x1x48x25xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[VAL2:%.+]] = IE.SoftMax([[VAL1]]#0) {axisInd = 2 : i64} :
  // CHECK-SAME:    tensor<1x1x48x25xf16> -> tensor<1x1x48x25xf16>

  // CHECK:       return [[VAL2]] : tensor<1x1x48x25xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @AddSplitDequantNotHasOneUseNotRemove
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x2x48x25xf16>) -> (tensor<1x1x48x25xf16>, tensor<1x1x48x25x!qElemType>)
func.func @AddSplitDequantNotHasOneUseNotRemove(%arg0: tensor<1x2x48x25xf16>) -> (tensor<1x1x48x25xf16>, tensor<1x1x48x25x!qElemType>) {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25x!qElemType>

  %1:2 = IE.Split(%0) {
    axis_value = 1 : i64, num_splits = 2 : i64
  } : tensor<1x2x48x25x!qElemType> -> tensor<1x1x48x25x!qElemType>, tensor<1x1x48x25x!qElemType>

  %2 = IE.Dequantize(%1#0) {
    dstElemType = f16
  } : tensor<1x1x48x25x!qElemType> -> tensor<1x1x48x25xf16>

  %3 = IE.SoftMax(%2) {axisInd = 2} : tensor<1x1x48x25xf16> -> tensor<1x1x48x25xf16>

  return %3, %1#1 : tensor<1x1x48x25xf16>, tensor<1x1x48x25x!qElemType>

  // CHECK:       [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25x!qElemType>

  // CHECK:       [[VAL1:%.+]]:2 = IE.Split([[VAL0]]) {
  // CHECK-SAME:    axis_value = 1 : i64, num_splits = 2 : i64
  // CHECK-SAME:  } : tensor<1x2x48x25x!qElemType> -> tensor<1x1x48x25x!qElemType>, tensor<1x1x48x25x!qElemType>

  // CHECK:       [[VAL2:%.+]] = IE.Dequantize([[VAL1]]#0) {
  // CHECK-SAME:    dstElemType = f16
  // CHECK-SAME:  } : tensor<1x1x48x25x!qElemType> -> tensor<1x1x48x25xf16>

  // CHECK:       [[VAL3:%.+]] = IE.SoftMax([[VAL2]]) {axisInd = 2 : i64} :
  // CHECK-SAME:    tensor<1x1x48x25xf16> -> tensor<1x1x48x25xf16>

  // CHECK:       return [[VAL3]], [[VAL1]]#1 : tensor<1x1x48x25xf16>, tensor<1x1x48x25x!qElemType>
}
