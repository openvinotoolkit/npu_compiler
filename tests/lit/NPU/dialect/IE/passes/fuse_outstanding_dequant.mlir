//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-outstanding-dequant %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: func.func @Conv2dSoftMaxWithOutstandingDequantWithImplicitRelu
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @Conv2dSoftMaxWithOutstandingDequantWithImplicitRelu(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
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
  // CHECK:      [[VAL1:%.+]] = IE.ReLU([[VAL0]]) :

  // CHECK:      [[VAL2:%.+]] = IE.SoftMax([[VAL1]]) {axisInd = 1 : i64} :
  // CHECK-SAME:   tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[VAL2]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803:120>

// CHECK-LABEL: func.func @Conv2dSoftMaxWithOutstandingDequantWithoutImplicitRelu
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @Conv2dSoftMaxWithOutstandingDequantWithoutImplicitRelu(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
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

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803:127>

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

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803:45>

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

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

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

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

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

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

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

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

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

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

// CHECK-LABEL: func.func @AddSliceDequantNotHasOneUseNotRemove
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x48x48xf16>) -> (tensor<1x16x32x32xf16>, tensor<1x16x32x32x!qElemType>)
func.func @AddSliceDequantNotHasOneUseNotRemove(%arg0: tensor<1x16x48x48xf16>) -> (tensor<1x16x32x32xf16>, tensor<1x16x32x32x!qElemType>) {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x48x48xf16>, tensor<1x16x48x48xf16> -> tensor<1x16x48x48x!qElemType>

  %1 = IE.Slice %0 [0, 0, 8, 8] [1, 16, 32, 32] : tensor<1x16x48x48x!qElemType> to tensor<1x16x32x32x!qElemType>

  %2 = IE.Dequantize(%1) {
    dstElemType = f16
  } : tensor<1x16x32x32x!qElemType> -> tensor<1x16x32x32xf16>

  %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

  return %3, %1 : tensor<1x16x32x32xf16>, tensor<1x16x32x32x!qElemType>

  // CHECK:       [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x48x48xf16>, tensor<1x16x48x48xf16> -> tensor<1x16x48x48x!qElemType>

  // CHECK:       [[VAL1:%.+]] = IE.Slice [[VAL0]] [0, 0, 8, 8] [1, 16, 32, 32] :
  // CHECK-SAME:    tensor<1x16x48x48x!qElemType> to tensor<1x16x32x32x!qElemType>

  // CHECK:       [[VAL2:%.+]] = IE.Dequantize([[VAL1]]) {
  // CHECK-SAME:    dstElemType = f16
  // CHECK-SAME:  } : tensor<1x16x32x32x!qElemType> -> tensor<1x16x32x32xf16>

  // CHECK:       [[VAL3:%.+]] = IE.SoftMax([[VAL2]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

  // CHECK:       return [[VAL3]], [[VAL1]] : tensor<1x16x32x32xf16>, tensor<1x16x32x32x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {0.0048406556540844491:128,0.0028785893730088777:128,0.0037009462422015619:128,0.0028234841776829142:128,0.0036912409698261935:128,0.0031632916600096458:128,0.00418546340044807:128,0.0027643728489969289:128,0.0060231732387168732:128,0.0028949776116539449:128,0.0044727944860271382:128,0.0053863244898178994:128,0.0038609912582472259:128,0.0028460528336319269:128,0.0047786067513858567:128,0.0029905208185607313:128}>
!qElemType1 = !quant.uniform<i8:f16:0, {0.0048406556540844491,0.0028785893730088777,0.0037009462422015619,0.0028234841776829142,0.0036912409698261935,0.0031632916600096458,0.00418546340044807,0.0027643728489969289,0.0060231732387168732,0.0028949776116539449,0.0044727944860271382,0.0053863244898178994,0.0038609912582472259,0.0028460528336319269,0.0047786067513858567,0.0029905208185607313}>
!qElemType2 = !quant.uniform<u8:f16, 0.011894785189161114>
!qElemType3 = !quant.uniform<u8:f16, 0.0039216639948826213>
!qElemTypeDequantizeInput = !quant.uniform<u8:f16, 0.017833509632185395>        // float range: [0 to 4.547544956207275]
!qElemTypeQuantizeOutput = !quant.uniform<u8:f16, 0.019608233021754844:127>     // float range: [-2.490246 to 2.509854]

// CHECK-LABEL:  @QuantizedConvWithImplicitRelu
func.func @QuantizedConvWithImplicitRelu(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %cst = const.Declare tensor<16x16x1x1x!qElemType> = dense<0> : tensor<16x16x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>]
  %cst_0 = const.Declare tensor<1x16x1x1xf16> = dense<0.000000e+00> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType2>
  %1 = IE.Convolution(%0, %cst, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType2>, tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemTypeDequantizeInput>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemTypeDequantizeInput> -> tensor<1x16x1x1xf16>
  %3 = IE.Quantize(%2) {dstElemType = !qElemTypeQuantizeOutput} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemTypeQuantizeOutput>
  %4 = IE.QuantizeCast(%3) {dstElemType = !qElemType3} : tensor<1x16x1x1x!qElemTypeQuantizeOutput> -> tensor<1x16x1x1x!qElemType3>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x16x1x1x!qElemType3> -> tensor<1x16x1x1xf16>
  return %5 : tensor<1x16x1x1xf16>

  // CHECK:  IE.Quantize
  // CHECK-NEXT:  IE.Convolution
  // CHECK-NEXT:  IE.ReLU
  // CHECK-NEXT:  IE.Quantize
  // CHECK-NEXT:  IE.QuantizeCast
  // CHECK-NEXT:  IE.Dequantize
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {0.0048406556540844491:128,0.0028785893730088777:128,0.0037009462422015619:128,0.0028234841776829142:128,0.0036912409698261935:128,0.0031632916600096458:128,0.00418546340044807:128,0.0027643728489969289:128,0.0060231732387168732:128,0.0028949776116539449:128,0.0044727944860271382:128,0.0053863244898178994:128,0.0038609912582472259:128,0.0028460528336319269:128,0.0047786067513858567:128,0.0029905208185607313:128}>
!qElemType1 = !quant.uniform<i8:f16:0, {0.0048406556540844491,0.0028785893730088777,0.0037009462422015619,0.0028234841776829142,0.0036912409698261935,0.0031632916600096458,0.00418546340044807,0.0027643728489969289,0.0060231732387168732,0.0028949776116539449,0.0044727944860271382,0.0053863244898178994,0.0038609912582472259,0.0028460528336319269,0.0047786067513858567,0.0029905208185607313}>
!qElemType2 = !quant.uniform<u8:f16, 0.011894785189161114>
!qElemType3 = !quant.uniform<u8:f16, 0.0039216639948826213>
!qElemTypeDequantizeInput = !quant.uniform<u8:f16, 0.017833509632185395:90>
!qElemTypeQuantizeOutput = !quant.uniform<u8:f16, 0.019608233021754844:127>

// CHECK-LABEL:  @QuantizedConvWithoutImplicitReluUnsigned
func.func @QuantizedConvWithoutImplicitReluUnsigned(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %cst = const.Declare tensor<16x16x1x1x!qElemType> = dense<0> : tensor<16x16x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>]
  %cst_0 = const.Declare tensor<1x16x1x1xf16> = dense<0.000000e+00> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType2>
  %1 = IE.Convolution(%0, %cst, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType2>, tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemTypeDequantizeInput>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemTypeDequantizeInput> -> tensor<1x16x1x1xf16>
  %3 = IE.Quantize(%2) {dstElemType = !qElemTypeQuantizeOutput} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemTypeQuantizeOutput>
  %4 = IE.QuantizeCast(%3) {dstElemType = !qElemType3} : tensor<1x16x1x1x!qElemTypeQuantizeOutput> -> tensor<1x16x1x1x!qElemType3>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x16x1x1x!qElemType3> -> tensor<1x16x1x1xf16>
  return %5 : tensor<1x16x1x1xf16>

  // Dequantize Op after Convolution should be removed
  // CHECK:  IE.Quantize
  // CHECK-NEXT:  IE.Convolution
  // CHECK-NOT:  IE.Dequantize
  // CHECK-NOT:  IE.ReLU
  // CHECK-NEXT:  IE.Quantize
  // CHECK-NEXT:  IE.QuantizeCast
  // CHECK-NEXT:  IE.Dequantize
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {0.0048406556540844491:128,0.0028785893730088777:128,0.0037009462422015619:128,0.0028234841776829142:128,0.0036912409698261935:128,0.0031632916600096458:128,0.00418546340044807:128,0.0027643728489969289:128,0.0060231732387168732:128,0.0028949776116539449:128,0.0044727944860271382:128,0.0053863244898178994:128,0.0038609912582472259:128,0.0028460528336319269:128,0.0047786067513858567:128,0.0029905208185607313:128}>
!qElemType1 = !quant.uniform<i8:f16:0, {0.0048406556540844491,0.0028785893730088777,0.0037009462422015619,0.0028234841776829142,0.0036912409698261935,0.0031632916600096458,0.00418546340044807,0.0027643728489969289,0.0060231732387168732,0.0028949776116539449,0.0044727944860271382,0.0053863244898178994,0.0038609912582472259,0.0028460528336319269,0.0047786067513858567,0.0029905208185607313}>
!qElemType2 = !quant.uniform<u8:f16, 0.011894785189161114>
!qElemType3 = !quant.uniform<i8:f16, 0.0039216639948826213>
!qElemTypeDequantizeInput = !quant.uniform<i8:f16, 0.017833509632185395>
!qElemTypeQuantizeOutput = !quant.uniform<i8:f16, 0.019608233021754844>

// CHECK-LABEL:  @QuantizedConvWithoutImplicitReluSigned
func.func @QuantizedConvWithoutImplicitReluSigned(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %cst = const.Declare tensor<16x16x1x1x!qElemType> = dense<0> : tensor<16x16x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>]
  %cst_0 = const.Declare tensor<1x16x1x1xf16> = dense<0.000000e+00> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType2>
  %1 = IE.Convolution(%0, %cst, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType2>, tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemTypeDequantizeInput>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemTypeDequantizeInput> -> tensor<1x16x1x1xf16>
  %3 = IE.Quantize(%2) {dstElemType = !qElemTypeQuantizeOutput} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemTypeQuantizeOutput>
  %4 = IE.QuantizeCast(%3) {dstElemType = !qElemType3} : tensor<1x16x1x1x!qElemTypeQuantizeOutput> -> tensor<1x16x1x1x!qElemType3>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x16x1x1x!qElemType3> -> tensor<1x16x1x1xf16>
  return %5 : tensor<1x16x1x1xf16>

  // Dequantize Op after Convolution should be removed
  // CHECK:  IE.Quantize
  // CHECK-NEXT:  IE.Convolution
  // CHECK-NOT:  IE.Dequantize
  // CHECK-NOT:  IE.ReLU
  // CHECK-NEXT:  IE.Quantize
  // CHECK-NEXT:  IE.QuantizeCast
  // CHECK-NEXT:  IE.Dequantize
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

// CHECK-LABEL: func.func @SliceWithOutstandingDequant
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x48x48xf16>) -> tensor<1x16x32x32xf16>
func.func @SliceWithOutstandingDequant(%arg0: tensor<1x16x48x48xf16>) -> tensor<1x16x32x32xf16> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x48x48xf16>, tensor<1x16x48x48xf16> -> tensor<1x16x48x48x!qElemType>

  %1 = IE.Slice %0 [0, 0, 8, 8] [1, 16, 32, 32] : tensor<1x16x48x48x!qElemType> to tensor<1x16x32x32x!qElemType>

  %2 = IE.Dequantize(%1) {
    dstElemType = f16
  } : tensor<1x16x32x32x!qElemType> -> tensor<1x16x32x32xf16>

  %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

  return %3 : tensor<1x16x32x32xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x48x48xf16>, tensor<1x16x48x48xf16> -> tensor<1x16x48x48xf16>

  // CHECK:       [[VAL1:%.+]] = IE.Slice [[VAL0]] [0, 0, 8, 8] [1, 16, 32, 32] :
  // CHECK-SAME:    tensor<1x16x48x48xf16> to tensor<1x16x32x32xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[VAL2:%.+]] = IE.SoftMax([[VAL1]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

  // CHECK:       return [[VAL2]] : tensor<1x16x32x32xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

// CHECK-LABEL: func.func @DepthToSpaceWithOutstandingDequantNoRemoval
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x32x32xf16>) -> tensor<1x4x64x64xf16>
func.func @DepthToSpaceWithOutstandingDequantNoRemoval(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x4x64x64xf16> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32x!qElemType>

  %1 = IE.DepthToSpace(%0) {
    block_size = 2 : i64,
    mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
  } : tensor<1x16x32x32x!qElemType> -> tensor<1x4x64x64x!qElemType>

  %2 = IE.Dequantize(%1) {
    dstElemType = f16
  } : tensor<1x4x64x64x!qElemType> -> tensor<1x4x64x64xf16>

  %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<1x4x64x64xf16> -> tensor<1x4x64x64xf16>

  return %3 : tensor<1x4x64x64xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32x!qElemType>

  // CHECK:       [[VAL1:%.+]] = IE.DepthToSpace([[VAL0]]) {
  // CHECK-SAME:    block_size = 2 : i64,
  // CHECK-SAME:    mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
  // CHECK-SAME:  } : tensor<1x16x32x32x!qElemType> -> tensor<1x4x64x64x!qElemType>

  // CHECK:       [[VAL2:%.+]] = IE.Dequantize([[VAL1]]) {
  // CHECK-SAME:    dstElemType = f16
  // CHECK-SAME:  } : tensor<1x4x64x64x!qElemType> -> tensor<1x4x64x64xf16>

  // CHECK:       [[VAL3:%.+]] = IE.SoftMax([[VAL2]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x4x64x64xf16> -> tensor<1x4x64x64xf16>

  // CHECK:       return [[VAL3]] : tensor<1x4x64x64xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

// CHECK-LABEL: func.func @SliceFromBlockArgWithDequantNoRemoval
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x48x48x!qElemType>) -> tensor<1x16x32x32xf16>
func.func @SliceFromBlockArgWithDequantNoRemoval(%arg0: tensor<1x16x48x48x!qElemType>) -> tensor<1x16x32x32xf16> {
  %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 16, 32, 32] : tensor<1x16x48x48x!qElemType> to tensor<1x16x32x32x!qElemType>

  %1 = IE.Dequantize(%0) {
    dstElemType = f16
  } : tensor<1x16x32x32x!qElemType> -> tensor<1x16x32x32xf16>

  %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

  return %2 : tensor<1x16x32x32xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 32, 32] :
  // CHECK-SAME:    tensor<1x16x48x48x!qElemType> to tensor<1x16x32x32x!qElemType>

  // CHECK:       [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {
  // CHECK-SAME:    dstElemType = f16
  // CHECK-SAME:  } : tensor<1x16x32x32x!qElemType> -> tensor<1x16x32x32xf16>

  // CHECK:       [[VAL2:%.+]] = IE.SoftMax([[VAL1]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

  // CHECK:       return [[VAL2]] : tensor<1x16x32x32xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>

// CHECK-LABEL: func.func @AvgPoolWithPostOpOutstandingDequantNoRemoval
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x16x16xf16>) -> tensor<1x16x8x8xf16>
func.func @AvgPoolWithPostOpOutstandingDequantNoRemoval(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x8x8xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16x!qElemType>
  %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Gelu<>, rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x8x8x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x8x8x!qElemType> -> tensor<1x16x8x8xf16>
  return %2 : tensor<1x16x8x8xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Quantize([[INPUT]]) {
  // CHECK-SAME:    dstElemType = !qElemType
  // CHECK-SAME:  } : tensor<1x16x16x16xf16> -> tensor<1x16x16x16x!qElemType>

  // CHECK:       [[VAL1:%.+]] = IE.AvgPool([[VAL0]]) {
  // CHECK-SAME:    exclude_pads, kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Gelu<>, rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
  // CHECK-SAME:  } : tensor<1x16x16x16x!qElemType> -> tensor<1x16x8x8x!qElemType>

  // CHECK:       [[VAL2:%.+]] = IE.Dequantize([[VAL1]]) {
  // CHECK-SAME:    dstElemType = f16
  // CHECK-SAME:  } : tensor<1x16x8x8x!qElemType> -> tensor<1x16x8x8xf16>

  // CHECK:       return [[VAL2]] : tensor<1x16x8x8xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>

// CHECK-LABEL: func.func @GroupConvWithNonReluPostOpOutstandingDequantNoRemoval
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @GroupConvWithNonReluPostOpOutstandingDequantNoRemoval(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %cst = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00> : tensor<16x1x1x1xf16>
  %0 = IE.GroupConvolution(%arg0, %cst) {
    dilations = [1, 1],
    groups = 16 : i64,
    pads_begin = [0, 0],
    pads_end = [0, 0],
    post_op = #IE.Gelu<>,
    strides = [1, 1]
  } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType>
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
  return %1 : tensor<1x16x3x3xf16>

  // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16>

  // CHECK:      [[VAL0:%.+]] = IE.GroupConvolution([[INPUT]], [[CST]]) {
  // CHECK-SAME:   dilations = [1, 1]
  // CHECK-SAME:   groups = 16 : i64,
  // CHECK-SAME:   pads_begin = [0, 0],
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   post_op = #IE.Gelu<>,
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  // CHECK:      [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {
  // CHECK-SAME:   dstElemType = f16
  // CHECK-SAME: } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>

// CHECK-LABEL: func.func @AddWithNonReluPostOpOutstandingDequantNoRemoval
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @AddWithNonReluPostOpOutstandingDequantNoRemoval(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    post_op = #IE.Gelu<>
  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
  return %1 : tensor<1x16x3x3xf16>

  // CHECK:      [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:   auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:   post_op = #IE.Gelu<>
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  // CHECK:      [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {
  // CHECK-SAME:   dstElemType = f16
  // CHECK-SAME: } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>

// CHECK-LABEL: func.func @GroupConvWithNonReluPostOpOutstandingDequantNoRemoval
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @GroupConvWithNonReluPostOpOutstandingDequantNoRemoval(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %cst = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00> : tensor<16x1x1x1xf16>
  %0 = IE.GroupConvolution(%arg0, %cst) {
    dilations = [1, 1],
    groups = 16 : i64,
    pads_begin = [0, 0],
    pads_end = [0, 0],
    post_op = #IE.Gelu<>,
    strides = [1, 1]
  } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType>
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
  return %1 : tensor<1x16x3x3xf16>

  // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16>

  // CHECK:      [[VAL0:%.+]] = IE.GroupConvolution([[INPUT]], [[CST]]) {
  // CHECK-SAME:   dilations = [1, 1]
  // CHECK-SAME:   groups = 16 : i64,
  // CHECK-SAME:   pads_begin = [0, 0],
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   post_op = #IE.Gelu<>,
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  // CHECK:      [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {
  // CHECK-SAME:   dstElemType = f16
  // CHECK-SAME: } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>

// CHECK-LABEL: func.func @AddWithNonReluPostOpOutstandingDequantNoRemoval
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @AddWithNonReluPostOpOutstandingDequantNoRemoval(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    post_op = #IE.Gelu<>
  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
  return %1 : tensor<1x16x3x3xf16>

  // CHECK:      [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:   auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:   post_op = #IE.Gelu<>
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  // CHECK:      [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {
  // CHECK-SAME:   dstElemType = f16
  // CHECK-SAME: } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.045763225181429994>

// CHECK-LABEL: func.func @ConvWithReluPostOpAndDequantize
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<151x768x1x1xf16>) -> tensor<151x1536x1x1xf16>
func.func @ConvWithReluPostOpAndDequantize(%arg0: tensor<151x768x1x1xf16>) -> tensor<151x1536x1x1xf16> {
  %weights = const.Declare tensor<1536x768x1x1xf16> = dense<1.0> : tensor<1536x768x1x1xf16>
  %bias = const.Declare tensor<1x1536x1x1xf16> = dense<0.5> : tensor<1x1536x1x1xf16>

  %0 = IE.Convolution(%arg0, %weights, %bias) {
    dilations = [1, 1],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    post_op = #IE.Relu<>,
    strides = [1, 1]
  } : tensor<151x768x1x1xf16>, tensor<1536x768x1x1xf16>, tensor<1x1536x1x1xf16> -> tensor<151x1536x1x1x!qElemType>

  %1 = IE.Dequantize(%0) {
    dstElemType = f16
  } : tensor<151x1536x1x1x!qElemType> -> tensor<151x1536x1x1xf16>

  return %1 : tensor<151x1536x1x1xf16>

  // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<1536x768x1x1xf16> = dense<1.000000e+00> :
  // CHECK-SAME:   tensor<1536x768x1x1xf16>
  // CHECK-DAG:  [[BIAS:%.+]] = const.Declare tensor<1x1536x1x1xf16> = dense<5.000000e-01> :
  // CHECK-SAME:   tensor<1x1536x1x1xf16>

  // CHECK:      [[VAL0:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) {
  // CHECK-SAME:   dilations = [1, 1],
  // CHECK-SAME:   pads_begin = [0, 0],
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   post_op = #IE.Relu<>,
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<151x768x1x1xf16>, tensor<1536x768x1x1xf16>, tensor<1x1536x1x1xf16> -> tensor<151x1536x1x1xf16>

  // CHECK-NOT:  IE.Dequantize
  // CHECK:      [[VAL1:%.+]] = IE.ReLU([[VAL0]]) :
  // CHECK-SAME:   tensor<151x1536x1x1xf16> -> tensor<151x1536x1x1xf16>

  // CHECK:      return [[VAL1]] : tensor<151x1536x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.12009554844276578:255>

// CHECK-LABEL: func.func @MaxPoolDequantize
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<4x64x200x63x!qElemType>) -> tensor<4x64x1x1xf16>
func.func @MaxPoolDequantize(%arg0: tensor<4x64x200x63x!qElemType>) -> tensor<4x64x1x1xf16> {
  %0 = IE.MaxPool(%arg0) {kernel_size = [8, 7], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 7]} : tensor<4x64x200x63x!qElemType> -> tensor<4x64x25x9x!qElemType>
  %1 = IE.MaxPool(%0) {kernel_size = [5, 9], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [5, 9]} : tensor<4x64x25x9x!qElemType> -> tensor<4x64x5x1x!qElemType>
  %2 = IE.MaxPool(%1) {kernel_size = [5, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [5, 1]} : tensor<4x64x5x1x!qElemType> -> tensor<4x64x1x1x!qElemType>
  %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<4x64x1x1x!qElemType> -> tensor<4x64x1x1xf16>

  return %3 : tensor<4x64x1x1xf16>

  // CHECK:       [[VAL0:%.+]] = IE.MaxPool([[INPUT]]) {
  // CHECK-SAME:    kernel_size = [8, 7]
  // CHECK-SAME:  } : tensor<4x64x200x63x!qElemType> -> tensor<4x64x25x9x!qElemType>

  // CHECK:       [[VAL1:%.+]] = IE.MaxPool([[VAL0]]) {
  // CHECK-SAME:    kernel_size = [5, 9]
  // CHECK-SAME:  } : tensor<4x64x25x9x!qElemType> -> tensor<4x64x5x1x!qElemType>

  // CHECK:       [[VAL2:%.+]] = IE.MaxPool([[VAL1]]) {
  // CHECK-SAME:    kernel_size = [5, 1]
  // CHECK-SAME:  } : tensor<4x64x5x1x!qElemType> -> tensor<4x64x1x1x!qElemType>

  // CHECK:       [[VAL3:%.+]] = IE.Dequantize([[VAL2]]) {
  // CHECK-SAME:    dstElemType = f16
  // CHECK-SAME:  } : tensor<4x64x1x1x!qElemType> -> tensor<4x64x1x1xf16>

  // CHECK:       return [[VAL3]] : tensor<4x64x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.12009554844276578:255>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @MaxPoolNoFuseDequantizeThroughOp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<4x64x200x63x!qElemType>) -> tensor<4x25x9x64xf16>
func.func @MaxPoolNoFuseDequantizeThroughOp(%arg0: tensor<4x64x200x63x!qElemType>) -> tensor<4x25x9x64xf16> {
  %0 = IE.MaxPool(%arg0) {kernel_size = [8, 7], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 7]} : tensor<4x64x200x63x!qElemType> -> tensor<4x64x25x9x!qElemType>
  %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<4x64x25x9x!qElemType> -> tensor<4x25x9x64x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<4x25x9x64x!qElemType> -> tensor<4x25x9x64xf16>

  return %2 : tensor<4x25x9x64xf16>

  // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[INPUT]]) {
  // CHECK-SAME:    kernel_size = [8, 7]
  // CHECK-SAME:  } : tensor<4x64x200x63x!qElemType> -> tensor<4x64x25x9x!qElemType>

  // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[MAXPOOL]]) {
  // CHECK-SAME:    order_value = #NHWC
  // CHECK-SAME:  } : tensor<4x64x25x9x!qElemType> -> tensor<4x25x9x64x!qElemType>

  // CHECK:       [[DQ:%.+]] = IE.Dequantize([[TRANSPOSE]]) {
  // CHECK-SAME:    dstElemType = f16
  // CHECK-SAME:  } : tensor<4x25x9x64x!qElemType> -> tensor<4x25x9x64xf16>

  // CHECK:       return [[DQ]] : tensor<4x25x9x64xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.012968034837760177:121>

// CHECK-LABEL: func.func @FuseWithClampLowZeroAndOutputNotQuantized
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x384x20x20xf16>) -> tensor<1x384x20x20xf16>
func.func @FuseWithClampLowZeroAndOutputNotQuantized(%arg0: tensor<1x384x20x20xf16>) -> tensor<1x384x20x20xf16> {
    %cst = const.Declare tensor<384x384x1x1x!qElemType> = dense<1> : tensor<384x384x1x1xsi8>

    %0 = IE.Convolution(%arg0, %cst) {clamp = {max = 1.671716570854187 : f64, min = 0.0 : f64}, dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x384x20x20xf16>, tensor<384x384x1x1x!qElemType> -> tensor<1x384x20x20x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x384x20x20x!qElemType1> -> tensor<1x384x20x20xf16>
    return %1 : tensor<1x384x20x20xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<384x384x1x1x!qElemType> = dense<1> : tensor<384x384x1x1xsi8>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {clamp = {max = 1.671716570854187 : f64, min = 0.000000e+00 : f64}, dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x384x20x20xf16>, tensor<384x384x1x1x!qElemType> -> tensor<1x384x20x20xf16>

    // CHECK: return [[CONV]] : tensor<1x384x20x20xf16>
}
