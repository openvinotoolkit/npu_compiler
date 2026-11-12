//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-outstanding-quant %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @SoftMaxAddWithOutstandingQuantHasOneUser
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @SoftMaxAddWithOutstandingQuantHasOneUser(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %0 = IE.SoftMax(%arg0) {
    axisInd = 1
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  %1 = IE.Quantize(%0) {
    dstElemType = !qElemType
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  %2 = IE.Add(%1, %1) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x3x3x!qElemType>, tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  return %2 : tensor<1x16x3x3xf16>

  // CHECK:       [[VAL0:%.+]] = IE.SoftMax([[INPUT]]) {
  // CHECK-SAME:    axisInd = 1 : i64
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:   IE.Quantize

  // CHECK:       [[VAL1:%.+]] = IE.Add([[VAL0]], [[VAL0]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @Conv2DSoftMaxAddWithOutstandingQuant
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @Conv2DSoftMaxAddWithOutstandingQuant(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %cst = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>

  %0 = IE.Convolution(%arg0, %cst) {
    dilations = [1, 1],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    strides = [1, 1]
  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.SoftMax(%arg0) {
    axisInd = 1
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  %2 = IE.Quantize(%1) {
    dstElemType = !qElemType
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  %3 = IE.Add(%0, %2) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x3x3x!qElemType>, tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  return %3 : tensor<1x16x3x3xf16>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> :
  // CHECK-SAME:    tensor<16x16x1x1xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
  // CHECK-SAME:    dilations = [1, 1],
  // CHECK-SAME:    pads_begin = [0, 0],
  // CHECK-SAME:    pads_end = [0, 0],
  // CHECK-SAME:    strides = [1, 1]
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       [[VAL1:%.+]] = IE.SoftMax([[INPUT]]) {
  // CHECK-SAME:    axisInd = 1 : i64
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:   IE.Quantize

  // CHECK:       [[VAL2:%.+]] = IE.Add([[VAL0]], [[VAL1]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[VAL2]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @SoftMaxSoftMaxAddWithOutstandingQuant
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x3x3xf16>, [[INPUT1:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @SoftMaxSoftMaxAddWithOutstandingQuant(%arg0: tensor<1x16x3x3xf16>, %arg1: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %0 = IE.SoftMax(%arg0) {
    axisInd = 1
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  %1 = IE.Quantize(%0) {
    dstElemType = !qElemType
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  %2 = IE.SoftMax(%arg1) {
    axisInd = 1
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  %3 = IE.Quantize(%2) {
    dstElemType = !qElemType
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  %4 = IE.Add(%1, %3) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x3x3x!qElemType>, tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  return %4 : tensor<1x16x3x3xf16>

  // CHECK:       [[VAL0:%.+]] = IE.SoftMax([[INPUT0]]) {
  // CHECK-SAME:    axisInd = 1 : i64
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:   IE.Quantize

  // CHECK:       [[VAL1:%.+]] = IE.SoftMax([[INPUT1]]) {
  // CHECK-SAME:    axisInd = 1 : i64
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:   IE.Quantize

  // CHECK:       [[VAL2:%.+]] = IE.Add([[VAL0]], [[VAL1]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[VAL2]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: func.func @SoftMaxTransposeAddNotHasOneUserNotFuse
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @SoftMaxTransposeAddNotHasOneUserNotFuse(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %0 = IE.SoftMax(%arg0) {
    axisInd = 1
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  %1 = IE.Quantize(%0) {
    dstElemType = !qElemType
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  %2 = IE.Transpose(%1) {
    order_value = #NCWH
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType>

  %3 = IE.Add(%1, %2) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x3x3x!qElemType>, tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  return %3 : tensor<1x16x3x3xf16>

  // CHECK:       [[VAL0:%.+]] = IE.SoftMax([[INPUT]]) {
  // CHECK-SAME:    axisInd = 1 : i64
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       [[VAL1:%.+]] = IE.Quantize([[VAL0]]) {
  // CHECK-SAME:    dstElemType = !qElemType
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  // CHECK:       [[VAL2:%.+]] = IE.Transpose([[VAL1]]) {
  // CHECK-SAME:    order_value = #NCWH
  // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType>

  // CHECK:       [[VAL3:%.+]] = IE.Add([[VAL1]], [[VAL2]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType>, tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[VAL3]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: func.func @Conv2DReshapeSoftMaxTransposeAddWithOutstandingQuant
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x3x3xf16>, [[INPUT1:%.+]]: tensor<1x12x3x4xf16>) -> tensor<1x12x4x3xf16>
func.func @Conv2DReshapeSoftMaxTransposeAddWithOutstandingQuant(%arg0: tensor<1x16x3x3xf16>, %arg1: tensor<1x12x3x4xf16>) -> tensor<1x12x4x3xf16> {
  %cst = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>

  %0 = IE.Convolution(%arg0, %cst) {
    dilations = [1, 1],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    strides = [1, 1]
  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.Reshape(%0) {
    shape_value = [1, 12, 4, 3]
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x12x4x3x!qElemType>

  %2 = IE.SoftMax(%arg1) {
    axisInd = 1
  } : tensor<1x12x3x4xf16> -> tensor<1x12x3x4xf16>

  %3 = IE.Quantize(%2) {
    dstElemType = !qElemType
  } : tensor<1x12x3x4xf16> -> tensor<1x12x3x4x!qElemType>

  %4 = IE.Transpose(%3) {
    order_value = #NCWH
  } : tensor<1x12x3x4x!qElemType> -> tensor<1x12x4x3x!qElemType>

  %5 = IE.Add(%1, %4) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x12x4x3x!qElemType>, tensor<1x12x4x3x!qElemType> -> tensor<1x12x4x3xf16>

  return %5 : tensor<1x12x4x3xf16>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> :
  // CHECK-SAME:    tensor<16x16x1x1xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Convolution([[INPUT0]], [[CST]]) {
  // CHECK-SAME:    dilations = [1, 1],
  // CHECK-SAME:    pads_begin = [0, 0],
  // CHECK-SAME:    pads_end = [0, 0],
  // CHECK-SAME:    strides = [1, 1]
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {
  // CHECK-SAME:    shape_value = [1, 12, 4, 3]
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x12x4x3xf16>

  // CHECK:       [[VAL2:%.+]] = IE.SoftMax([[INPUT1]]) {
  // CHECK-SAME:    axisInd = 1 : i64
  // CHECK-SAME:  } : tensor<1x12x3x4xf16> -> tensor<1x12x3x4xf16>

  // CHECK-NOT:   IE.Quantize

  // CHECK:       [[VAL3:%.+]] = IE.Transpose([[VAL2]]) {
  // CHECK-SAME:    order_value = #NCWH
  // CHECK-SAME:  } : tensor<1x12x3x4xf16> -> tensor<1x12x4x3xf16>

  // CHECK:       [[VAL4:%.+]] = IE.Add([[VAL1]], [[VAL3]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x12x4x3xf16>, tensor<1x12x4x3xf16> -> tensor<1x12x4x3xf16>

  // CHECK:       return [[VAL4]] : tensor<1x12x4x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: func.func @Conv2DReshapeSoftMaxAddWithOutstandingQuant
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x3x3xf16>, [[INPUT1:%.+]]: tensor<1x12x4x3xf16>) -> tensor<1x12x4x3xf16>
func.func @Conv2DReshapeSoftMaxAddWithOutstandingQuant(%arg0: tensor<1x16x3x3xf16>, %arg1: tensor<1x12x4x3xf16>) -> tensor<1x12x4x3xf16> {
  %cst = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>

  %0 = IE.Convolution(%arg0, %cst) {
    dilations = [1, 1],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    strides = [1, 1]
  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.Reshape(%0) {
    shape_value = [1, 12, 4, 3]
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x12x4x3x!qElemType>

  %2 = IE.SoftMax(%arg1) {
    axisInd = 1
  } : tensor<1x12x4x3xf16> -> tensor<1x12x4x3xf16>

  %3 = IE.Quantize(%2) {
    dstElemType = !qElemType
  } : tensor<1x12x4x3xf16> -> tensor<1x12x4x3x!qElemType>

  %4 = IE.Add(%1, %3) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x12x4x3x!qElemType>, tensor<1x12x4x3x!qElemType> -> tensor<1x12x4x3xf16>

  return %4 : tensor<1x12x4x3xf16>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> :
  // CHECK-SAME:    tensor<16x16x1x1xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Convolution([[INPUT0]], [[CST]]) {
  // CHECK-SAME:    dilations = [1, 1],
  // CHECK-SAME:    pads_begin = [0, 0],
  // CHECK-SAME:    pads_end = [0, 0],
  // CHECK-SAME:    strides = [1, 1]
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {
  // CHECK-SAME:    shape_value = [1, 12, 4, 3]
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x12x4x3xf16>

  // CHECK:       [[VAL2:%.+]] = IE.SoftMax([[INPUT1]]) {
  // CHECK-SAME:    axisInd = 1 : i64
  // CHECK-SAME:  } : tensor<1x12x4x3xf16> -> tensor<1x12x4x3xf16>

  // CHECK-NOT:   IE.Quantize

  // CHECK:       [[VAL3:%.+]] = IE.Add([[VAL1]], [[VAL2]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x12x4x3xf16>, tensor<1x12x4x3xf16> -> tensor<1x12x4x3xf16>

  // CHECK:       return [[VAL3]] : tensor<1x12x4x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: func.func @SoftMaxReshapeTransposeSoftMaxTransposeReshapeAddWithOutstandingQuant
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x3x3xf16>, [[INPUT1:%.+]]: tensor<1x16x9xf16>) -> tensor<1x12x4x3xf16>
func.func @SoftMaxReshapeTransposeSoftMaxTransposeReshapeAddWithOutstandingQuant(%arg0: tensor<1x16x3x3xf16>, %arg1: tensor<1x16x9xf16>) -> tensor<1x12x4x3xf16> {
  %0 = IE.SoftMax(%arg0) {
    axisInd = 1
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  %1 = IE.Quantize(%0) {
    dstElemType = !qElemType
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  %2 = IE.Reshape(%1) {
    shape_value = [1, 12, 3, 4]
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x12x3x4x!qElemType>

  %3 = IE.Transpose(%2) {
    order_value = #NCWH
  } : tensor<1x12x3x4x!qElemType> -> tensor<1x12x4x3x!qElemType>

  %4 = IE.SoftMax(%arg1) {
    axisInd = 1
  } : tensor<1x16x9xf16> -> tensor<1x16x9xf16>

  %5 = IE.Quantize(%4) {
    dstElemType = !qElemType
  } : tensor<1x16x9xf16> -> tensor<1x16x9x!qElemType>

  %6 = IE.Transpose(%5) {
    order_value = #map
  } : tensor<1x16x9x!qElemType> -> tensor<1x9x16x!qElemType>

  %7 = IE.Reshape(%6) {
    shape_value = [1, 12, 4, 3]
  } : tensor<1x9x16x!qElemType> -> tensor<1x12x4x3x!qElemType>

  %8 = IE.Add(%3, %7) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x12x4x3x!qElemType>, tensor<1x12x4x3x!qElemType> -> tensor<1x12x4x3xf16>

  return %8 : tensor<1x12x4x3xf16>

  // CHECK:       [[VAL0:%.+]] = IE.SoftMax([[INPUT0]]) {
  // CHECK-SAME:    axisInd = 1 : i64
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:   IE.Quantize

  // CHECK:       [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {
  // CHECK-SAME:    shape_value = [1, 12, 3, 4]
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x12x3x4xf16>

  // CHECK:       [[VAL2:%.+]] = IE.Transpose([[VAL1]]) {
  // CHECK-SAME:    order_value = #NCWH
  // CHECK-SAME:  } : tensor<1x12x3x4xf16> -> tensor<1x12x4x3xf16>

  // CHECK:       [[VAL3:%.+]] = IE.SoftMax([[INPUT1]]) {
  // CHECK-SAME:    axisInd = 1 : i64
  // CHECK-SAME:  } : tensor<1x16x9xf16> -> tensor<1x16x9xf16>

  // CHECK-NOT:   IE.Quantize

  // CHECK:       [[VAL4:%.+]] = IE.Transpose([[VAL3]]) {
  // CHECK-SAME:    order_value = #map
  // CHECK-SAME:  } : tensor<1x16x9xf16> -> tensor<1x9x16xf16>

  // CHECK:       [[VAL5:%.+]] = IE.Reshape([[VAL4]]) {
  // CHECK-SAME:    shape_value = [1, 12, 4, 3]
  // CHECK-SAME:  } : tensor<1x9x16xf16> -> tensor<1x12x4x3xf16>

  // CHECK:       [[VAL6:%.+]] = IE.Add([[VAL2]], [[VAL5]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x12x4x3xf16>, tensor<1x12x4x3xf16> -> tensor<1x12x4x3xf16>

  // CHECK:       return [[VAL6]] : tensor<1x12x4x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.5>
!qElemType1 = !quant.uniform<u8:f16, 0.25>

// CHECK-LABEL: func.func @Conv2DQuantizeCastSoftMaxAddNotFuse
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @Conv2DQuantizeCastSoftMaxAddNotFuse(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %cst = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>

  %0 = IE.Convolution(%arg0, %cst) {
    dilations = [1, 1],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    strides = [1, 1]
  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.QuantizeCast(%0) {
    dstElemType = !qElemType1
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType1>

  %2 = IE.SoftMax(%arg0) {
    axisInd = 1
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  %3 = IE.Quantize(%2) {
    dstElemType = !qElemType1
  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>

  %4 = IE.Add(%1, %3) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x3x3x!qElemType1>, tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>

  return %4 : tensor<1x16x3x3xf16>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> :
  // CHECK-SAME:    tensor<16x16x1x1xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
  // CHECK-SAME:    dilations = [1, 1],
  // CHECK-SAME:    pads_begin = [0, 0],
  // CHECK-SAME:    pads_end = [0, 0],
  // CHECK-SAME:    strides = [1, 1]
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  // CHECK:       [[VAL1:%.+]] = IE.QuantizeCast([[VAL0]]) {
  // CHECK-SAME:    dstElemType = !qElemType1
  // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType1>

  // CHECK:       [[VAL2:%.+]] = IE.SoftMax([[INPUT]]) {
  // CHECK-SAME:    axisInd = 1 : i64
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       [[VAL3:%.+]] = IE.Quantize([[VAL2]]) {
  // CHECK-SAME:    dstElemType = !qElemType1
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>

  // CHECK:       [[VAL4:%.+]] = IE.Add([[VAL1]], [[VAL3]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType1>, tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[VAL4]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @Conv2DSoftMaxConcatAddNotFuse
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x3x3xf16>, [[INPUT1:%.+]]: tensor<1x8x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @Conv2DSoftMaxConcatAddNotFuse(%arg0: tensor<1x16x3x3xf16> ,%arg1: tensor<1x8x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %cst0 = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>
  %cst1 = const.Declare tensor<1x8x3x3x!qElemType> = dense<1.0> :
    tensor<1x8x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  %0 = IE.Convolution(%arg0, %cst0) {
    dilations = [1, 1],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    strides = [1, 1]
  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.SoftMax(%arg1) {
    axisInd = 1
  } : tensor<1x8x3x3xf16> -> tensor<1x8x3x3xf16>

  %2 = IE.Quantize(%1) {
    dstElemType = !qElemType
  } : tensor<1x8x3x3xf16> -> tensor<1x8x3x3x!qElemType>

  %3 = IE.Concat(%2, %cst1) {
    per_axis = #IE.Concat<axis = 1 : i64>
  } : tensor<1x8x3x3x!qElemType>, tensor<1x8x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType>

  %4 = IE.Add(%0, %3) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x3x3x!qElemType>, tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  return %4 : tensor<1x16x3x3xf16>

  // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> :
  // CHECK-SAME:    tensor<16x16x1x1xf16>

  // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x8x3x3x!qElemType> = dense<1.000000e+00> :
  // CHECK-SAME:    tensor<1x8x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  // CHECK:       [[VAL0:%.+]] = IE.Convolution([[INPUT0]], [[CST0]]) {
  // CHECK-SAME:    dilations = [1, 1],
  // CHECK-SAME:    pads_begin = [0, 0],
  // CHECK-SAME:    pads_end = [0, 0],
  // CHECK-SAME:    strides = [1, 1]
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  // CHECK:       [[VAL1:%.+]] = IE.SoftMax([[INPUT1]]) {
  // CHECK-SAME:    axisInd = 1 : i64
  // CHECK-SAME:  } : tensor<1x8x3x3xf16> -> tensor<1x8x3x3xf16>

  // CHECK:       [[VAL2:%.+]] = IE.Quantize([[VAL1]]) {
  // CHECK-SAME:    dstElemType = !qElemType
  // CHECK-SAME:  } : tensor<1x8x3x3xf16> -> tensor<1x8x3x3x!qElemType>

  // CHECK:       [[VAL3:%.+]] = IE.Concat([[VAL2]], [[CST1]]) {
  // CHECK-SAME:    per_axis = #IE.Concat<axis = 1 : i64>
  // CHECK-SAME:  } : tensor<1x8x3x3x!qElemType>, tensor<1x8x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType>

  // CHECK:       [[VAL4:%.+]] = IE.Add([[VAL0]], [[VAL3]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType>, tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[VAL4]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @Conv2DSplitSoftMaxAddHasOneUse
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x3x3xf16>, [[INPUT1:%.+]]: tensor<1x8x3x3xf16>) -> tensor<1x8x3x3xf16>
func.func @Conv2DSplitSoftMaxAddHasOneUse(%arg0: tensor<1x16x3x3xf16>, %arg1: tensor<1x8x3x3xf16>) -> tensor<1x8x3x3xf16> {
  %cst = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>

  %0 = IE.Convolution(%arg0, %cst) {
    dilations = [1, 1],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    strides = [1, 1]
  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  %1:2 = IE.Split(%0) {
    axis_value = 1 : i64, num_splits = 2 : i64
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x8x3x3x!qElemType>, tensor<1x8x3x3x!qElemType>

  %2 = IE.SoftMax(%arg1) {
    axisInd = 1
  } : tensor<1x8x3x3xf16> -> tensor<1x8x3x3xf16>

  %3 = IE.Quantize(%2) {
    dstElemType = !qElemType
  } : tensor<1x8x3x3xf16> -> tensor<1x8x3x3x!qElemType>

  %4 = IE.Add(%1#0, %3) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x8x3x3x!qElemType>, tensor<1x8x3x3x!qElemType> -> tensor<1x8x3x3xf16>

  return %4 : tensor<1x8x3x3xf16>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> :
  // CHECK-SAME:    tensor<16x16x1x1xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Convolution([[INPUT0]], [[CST]]) {
  // CHECK-SAME:    dilations = [1, 1],
  // CHECK-SAME:    pads_begin = [0, 0],
  // CHECK-SAME:    pads_end = [0, 0],
  // CHECK-SAME:    strides = [1, 1]
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       [[VAL1:%.+]]:2 = IE.Split([[VAL0]]) {
  // CHECK-SAME:    axis_value = 1 : i64, num_splits = 2 : i64
  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x8x3x3xf16>, tensor<1x8x3x3xf16>

  // CHECK:       [[VAL2:%.+]] = IE.SoftMax([[INPUT1]]) {
  // CHECK-SAME:    axisInd = 1 : i64
  // CHECK-SAME:  } : tensor<1x8x3x3xf16> -> tensor<1x8x3x3xf16>

  // CHECK-NOT:   IE.Quantize

  // CHECK:       [[VAL3:%.+]] = IE.Add([[VAL1]]#0, [[VAL2]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x8x3x3xf16>, tensor<1x8x3x3xf16> -> tensor<1x8x3x3xf16>

  // CHECK:       return [[VAL3]] : tensor<1x8x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @Conv2DSplitSoftMaxAddNotHasOneUseNotFuse
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x3x3xf16>, [[INPUT1:%.+]]: tensor<1x8x3x3xf16>) -> (tensor<1x8x3x3xf16>, tensor<1x8x3x3x!qElemType>)
func.func @Conv2DSplitSoftMaxAddNotHasOneUseNotFuse(%arg0: tensor<1x16x3x3xf16>, %arg1: tensor<1x8x3x3xf16>) -> (tensor<1x8x3x3xf16>, tensor<1x8x3x3x!qElemType>) {
  %cst = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>

  %0 = IE.Convolution(%arg0, %cst) {
    dilations = [1, 1],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    strides = [1, 1]
  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  %1:2 = IE.Split(%0) {
    axis_value = 1 : i64, num_splits = 2 : i64
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x8x3x3x!qElemType>, tensor<1x8x3x3x!qElemType>

  %2 = IE.SoftMax(%arg1) {
    axisInd = 1
  } : tensor<1x8x3x3xf16> -> tensor<1x8x3x3xf16>

  %3 = IE.Quantize(%2) {
    dstElemType = !qElemType
  } : tensor<1x8x3x3xf16> -> tensor<1x8x3x3x!qElemType>

  %4 = IE.Add(%1#0, %3) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x8x3x3x!qElemType>, tensor<1x8x3x3x!qElemType> -> tensor<1x8x3x3xf16>

  return %4, %1#1 : tensor<1x8x3x3xf16>, tensor<1x8x3x3x!qElemType>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> :
  // CHECK-SAME:    tensor<16x16x1x1xf16>

  // CHECK:       [[VAL0:%.+]] = IE.Convolution([[INPUT0]], [[CST]]) {
  // CHECK-SAME:    dilations = [1, 1],
  // CHECK-SAME:    pads_begin = [0, 0],
  // CHECK-SAME:    pads_end = [0, 0],
  // CHECK-SAME:    strides = [1, 1]
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  // CHECK:       [[VAL1:%.+]]:2 = IE.Split([[VAL0]]) {
  // CHECK-SAME:    axis_value = 1 : i64, num_splits = 2 : i64
  // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType> -> tensor<1x8x3x3x!qElemType>, tensor<1x8x3x3x!qElemType>

  // CHECK:       [[VAL2:%.+]] = IE.SoftMax([[INPUT1]]) {
  // CHECK-SAME:    axisInd = 1 : i64
  // CHECK-SAME:  } : tensor<1x8x3x3xf16> -> tensor<1x8x3x3xf16>

  // CHECK:       [[VAL3:%.+]] = IE.Quantize([[VAL2]]) {
  // CHECK-SAME:    dstElemType = !qElemType
  // CHECK-SAME:  } : tensor<1x8x3x3xf16> -> tensor<1x8x3x3x!qElemType>

  // CHECK:       [[VAL4:%.+]] = IE.Add([[VAL1]]#0, [[VAL3]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x8x3x3x!qElemType>, tensor<1x8x3x3x!qElemType> -> tensor<1x8x3x3xf16>

  // CHECK:       return [[VAL4]], [[VAL1]]#1 : tensor<1x8x3x3xf16>, tensor<1x8x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.049356617647058822>
!qElemType1 = !quant.uniform<u8:f16, 0.01013327205882353>
!qElemType2 = !quant.uniform<u8:f16, 0.053278186274509802>

// CHECK-LABEL: func.func @AddWithOutstandingQuantDifferentScale
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x56x56xf16>, [[INPUT1:%.+]]: tensor<1x16x56x56xf16>) -> tensor<1x16x56x56xf16>
func.func @AddWithOutstandingQuantDifferentScale(%arg0: tensor<1x16x56x56xf16>, %arg1: tensor<1x16x56x56xf16>) -> tensor<1x16x56x56xf16> {
  %0 = IE.Quantize(%arg0) {
    dstElemType = !qElemType
  } : tensor<1x16x56x56xf16> -> tensor<1x16x56x56x!qElemType>

  %1 = IE.Quantize(%arg1) {
    dstElemType = !qElemType1
  } : tensor<1x16x56x56xf16> -> tensor<1x16x56x56x!qElemType1>

  %2 = IE.Add(%0, %1) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x56x56x!qElemType>, tensor<1x16x56x56x!qElemType1> -> tensor<1x16x56x56x!qElemType2>

  %3 = IE.Dequantize(%2) {
    dstElemType = f16
  } : tensor<1x16x56x56x!qElemType2> -> tensor<1x16x56x56xf16>

  return %3 : tensor<1x16x56x56xf16>

  // CHECK-NOT:   IE.Quantize
  // CHECK-NOT:   IE.Quantize

  // CHECK:       [[VAL0:%.+]] = IE.Add([[INPUT0]], [[INPUT1]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x56x56xf16>, tensor<1x16x56x56xf16> ->
  // CHECK:       tensor<1x16x56x56x[[QELEMTYPE:!.+]]>

  // CHECK:       [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {
  // CHECK-SAME:    dstElemType = f16
  // CHECK-SAME:  } : tensor<1x16x56x56x[[QELEMTYPE]]> -> tensor<1x16x56x56xf16>

  // CHECK:       return [[VAL1]] : tensor<1x16x56x56xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @ClampAvgPoolWithOutstandingQuant
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x88x88xf16>) -> tensor<1x64x11x11xf16>
func.func @ClampAvgPoolWithOutstandingQuant(%arg0: tensor<1x64x88x88xf16>) -> tensor<1x64x11x11xf16> {
  %quantize = IE.Quantize(%arg0) {
    dstElemType = !qElemType
  } : tensor<1x64x88x88xf16> -> tensor<1x64x88x88x!qElemType>

  %clamp = IE.Clamp(%quantize) {
    max = 6.000000e+00 : f64,
    min = 0.000000e+00 : f64
  } : tensor<1x64x88x88x!qElemType> -> tensor<1x64x88x88x!qElemType>

  %avgpool = IE.AvgPool(%clamp) {
    exclude_pads,
    kernel_size = [8, 8],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    rounding_type = #IE.rounding_type<FLOOR>,
    strides = [8, 8]
  } : tensor<1x64x88x88x!qElemType> -> tensor<1x64x11x11xf16>

  return %avgpool : tensor<1x64x11x11xf16>

  // CHECK-NOT:   IE.Quantize

  // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[INPUT]]) {
  // CHECK-SAME:    max = 6.000000e+00 : f64,
  // CHECK-SAME:    min = 0.000000e+00 : f64
  // CHECK-SAME:  } : tensor<1x64x88x88xf16> -> tensor<1x64x88x88xf16>

  // CHECK:       [[AVGPOOL:%.+]] = IE.AvgPool([[CLAMP]]) {
  // CHECK-SAME:    exclude_pads,
  // CHECK-SAME:    kernel_size = [8, 8],
  // CHECK-SAME:    pads_begin = [0, 0],
  // CHECK-SAME:    pads_end = [0, 0],
  // CHECK-SAME:    rounding_type = #IE.rounding_type<FLOOR>,
  // CHECK-SAME:    strides = [8, 8]
  // CHECK-SAME:  } : tensor<1x64x88x88xf16> -> tensor<1x64x11x11xf16>

  // CHECK:       return [[AVGPOOL]] : tensor<1x64x11x11xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0038706578460394169>
!qElemType1 = !quant.uniform<u8:f16, 0.006584587517906638:150>
!qElemType2 = !quant.uniform<u8:f16, 0.0038446489502401912>
!qElemType3 = !quant.uniform<u8:f16, 0.0012826621532440186>

// CHECK-LABEL: func.func @NoFuseConvAvgPoolChainAlreadyQuantized
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x128x64x64xf16>) -> tensor<1x256x1x1xf16>
func.func @NoFuseConvAvgPoolChainAlreadyQuantized(%arg0: tensor<1x128x64x64xf16>) -> tensor<1x256x1x1xf16> {
  %cst = const.Declare tensor<256x128x3x3x!qElemType1> = dense<1.0> : tensor<256x128x3x3xf16>, [#const.CastElemType<!qElemType1>]
  %cst_0 = const.Declare tensor<1x256x1x1xf16> = dense<1.0> : tensor<1x256x1x1xf16>

  %0 = IE.Quantize(%arg0) {
    dstElemType = !qElemType2
  } : tensor<1x128x64x64xf16> -> tensor<1x128x64x64x!qElemType2>

  %1 = IE.Convolution(%0, %cst, %cst_0) {
    dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
  } : tensor<1x128x64x64x!qElemType2>, tensor<256x128x3x3x!qElemType1>, tensor<1x256x1x1xf16> -> tensor<1x256x64x64x!qElemType>

  %2 = IE.AvgPool(%1) {
    exclude_pads, kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0],
    rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 8]
  } : tensor<1x256x64x64x!qElemType> -> tensor<1x256x8x8xf16>

  %3 = IE.AvgPool(%2) {
    exclude_pads, kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0],
    rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
  } : tensor<1x256x8x8xf16> -> tensor<1x256x1x1x!qElemType3>

  %4 = IE.Dequantize(%3) {
    dstElemType = f16
  } : tensor<1x256x1x1x!qElemType3> -> tensor<1x256x1x1xf16>

  return %4 : tensor<1x256x1x1xf16>

  // CHECK:       [[CST:%.+]] = const.Declare tensor<256x128x3x3x!qElemType>
  // CHECK:       [[CST_0:%.+]] = const.Declare tensor<1x256x1x1xf16>
  // CHECK:       [[QUANT:%.+]] = IE.Quantize([[INPUT]])
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[QUANT]], [[CST]], [[CST_0]])
  // CHECK:       [[POOL1:%.+]] = IE.AvgPool([[CONV]])
  // CHECK:       [[POOL2:%.+]] = IE.AvgPool([[POOL1]])
  // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[POOL2]])
  // CHECK:       return [[DEQUANT]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @AvgPoolWithPostOpNotFuse
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x88x88xf16>) -> tensor<1x64x11x11xf16>
func.func @AvgPoolWithPostOpNotFuse(%arg0: tensor<1x64x88x88xf16>) -> tensor<1x64x11x11xf16> {
  %0 = IE.Quantize(%arg0) {
    dstElemType = !qElemType
  } : tensor<1x64x88x88xf16> -> tensor<1x64x88x88x!qElemType>

  %1 = IE.AvgPool(%0) {
    kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], strides = [8, 8],
    rounding_type = #IE.rounding_type<FLOOR>, exclude_pads,
    clamp = {min = 0.0 : f64, max = 6.0 : f64}
  } : tensor<1x64x88x88x!qElemType> -> tensor<1x64x11x11xf16>

  return %1 : tensor<1x64x11x11xf16>

  // Fusion should be blocked due to PostOp
  // CHECK:       [[VAL0:%.+]] = IE.Quantize([[INPUT]]) {
  // CHECK-SAME:    dstElemType = !qElemType
  // CHECK-SAME:  } : tensor<1x64x88x88xf16> -> tensor<1x64x88x88x!qElemType>

  // CHECK:       [[VAL1:%.+]] = IE.AvgPool([[VAL0]]) {
  // CHECK-SAME:    clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}
  // CHECK-SAME:  } : tensor<1x64x88x88x!qElemType> -> tensor<1x64x11x11xf16>

  // CHECK:       return [[VAL1]] : tensor<1x64x11x11xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: func.func @ConvWithPostOpBlocksFusion
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x56x56xf16>) -> tensor<1x16x56x56xf16>
func.func @ConvWithPostOpBlocksFusion(%arg0: tensor<1x16x56x56xf16>) -> tensor<1x16x56x56xf16> {
  %cst = const.Declare tensor<16x16x3x3x!quant.uniform<u8:f16, 0.0039215686274509803>> = dense<1.0> : tensor<16x16x3x3xf16>, [#const.CastElemType<!quant.uniform<u8:f16, 0.0039215686274509803>>]

  %0 = IE.Quantize(%arg0) {
    dstElemType = !qElemType
  } : tensor<1x16x56x56xf16> -> tensor<1x16x56x56x!qElemType>

  %1 = IE.Convolution(%0, %cst) {
    dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1],
    post_op = #IE.Relu<>
  } : tensor<1x16x56x56x!qElemType>, tensor<16x16x3x3x!quant.uniform<u8:f16, 0.0039215686274509803>> -> tensor<1x16x56x56x!qElemType>

  %2 = IE.Add(%0, %1) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x56x56x!qElemType>, tensor<1x16x56x56x!qElemType> -> tensor<1x16x56x56xf16>

  return %2 : tensor<1x16x56x56xf16>

  // PostOp blocks fusion - quantize remains
  // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x3x3x!qElemType>
  // CHECK:       [[QUANT:%.+]] = IE.Quantize([[INPUT]])
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[QUANT]], [[CST]]) {
  // CHECK-SAME:    post_op = #IE.Relu<>
  // CHECK:       [[ADD:%.+]] = IE.Add([[QUANT]], [[CONV]])
  // CHECK:       return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.022970187430288277:92>
!qElemType1 = !quant.uniform<u8:f16:0, {0.01019284023958094:128,0.010167219592075722:128,0.0096418605131261495:128}>
!qElemType2 = !quant.uniform<u8:f16, 0.0071074686798394899:128>
!qElemType3 = !quant.uniform<u8:f16, 0.016421539643231561:128>

// CHECK-LABEL: func.func @Conv2DSpaceToDepthWithOutstandingQuant
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x3x512x512xf32>, [[INPUT1:%.+]]: tensor<1x12x256x256x!qElemType>) -> tensor<1x3x512x512xf16>
func.func @Conv2DSpaceToDepthWithOutstandingQuant(%arg0: tensor<1x3x512x512xf32>, %arg1: tensor<1x12x256x256x!qElemType> ) -> tensor<1x3x512x512xf16>  {
  %cst = const.Declare tensor<3x3x3x3x!qElemType1> = dense<1.0> :
    tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]
  %cst_1 = const.Declare tensor<1x3x1x1xf16> = dense<1.0> : tensor<1x3x1x1xf16>
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x512x512xf32> -> tensor<1x3x512x512xf16>
  %1 = IE.Quantize(%0) {dstElemType = !qElemType2} : tensor<1x3x512x512xf16> -> tensor<1x3x512x512x!qElemType2>
  %2 = IE.DepthToSpace(%arg1) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x12x256x256x!qElemType> -> tensor<1x3x512x512x!qElemType>
  %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x512x512x!qElemType2>, tensor<1x3x512x512x!qElemType> -> tensor<1x3x512x512x!qElemType3>
  %4 = IE.Convolution(%3, %cst, %cst_1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.Tanh<>, strides = [1, 1]} : tensor<1x3x512x512x!qElemType3>, tensor<3x3x3x3x!qElemType1>, tensor<1x3x1x1xf16> -> tensor<1x3x512x512xf16>
  return %4 : tensor<1x3x512x512xf16>

  // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType1> = dense<1.000000e+00>
  // CHECK:       [[CST_0:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+00>
  // CHECK:       [[CONVERT:%.+]] = IE.Convert([[INPUT0]]) {dstElemType = f16}
  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType2}
  // CHECK:       [[DEPTHTOSPACE:%.+]] = IE.DepthToSpace([[INPUT1]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
  // CHECK:       [[ADD:%.+]] = IE.Add([[QUANTIZE]], [[DEPTHTOSPACE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[ADD]], [[CST]], [[CST_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.Tanh<>, strides = [1, 1]}
  // CHECK:       return [[CONV]]

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.020894650851978974:128>
!qElemType1 = !quant.uniform<u8:f16, 0.016421539643231561:128>
!qElemType2 = !quant.uniform<u8:f16:0, {0.0021634723625931084:128,0.0025575409917270434:128,0.0038781131015104406:128,0.0032656551576128192:128,0.0020378891159506403:128,0.0035840687798518759:128,0.0028856976359498267:128,0.0023135664416294474:128,0.0034765291447732961:128,0.0033485044451320874:128,0.0035138309001922609:128,0.0018750901900085749:128}>
!qElemType3 = !quant.uniform<i8:f16:0, {0.0021634723625931084,0.0025575409917270434,0.0038781131015104406,0.0032656551576128192,0.0020378891159506403,0.0035840687798518759,0.0028856976359498267,0.0023135664416294474,0.0034765291447732961,0.0033485044451320874,0.0035138309001922609,0.0018750901900085749}>
!qElemType4 = !quant.uniform<u8:f16, 0.0071074686798394899:128>
!qElemType5 = !quant.uniform<u8:f16, 0.022970187430288277:92>

// CHECK-LABEL: func.func @ConvWPostOpBeforeAddNotFused
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x3x512x512xf32>, [[INPUT1:%.+]]: tensor<1x32x256x256x!qElemType>) -> tensor<1x3x512x512x!qElemType1>
func.func @ConvWPostOpBeforeAddNotFused(%arg0: tensor<1x3x512x512xf32>, %arg1: tensor<1x32x256x256x!qElemType>) -> tensor<1x3x512x512x!qElemType1> {
  %cst = const.Declare tensor<12x32x3x3x!qElemType2> = dense<0> : tensor<12x32x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType3>, #const.ConvertElemType<!qElemType2>]
  %cst_0 = const.Declare tensor<1x12x1x1xf16> = dense<1.0> : tensor<1x12x1x1xf16>, [#const.CastElemType<f16>]
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x512x512xf32> -> tensor<1x3x512x512xf16>
  %1 = IE.Quantize(%0) {dstElemType = !qElemType4} : tensor<1x3x512x512xf16> -> tensor<1x3x512x512x!qElemType4>
  %2 = IE.Convolution(%arg1, %cst, %cst_0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.LeakyRelu<negative_slope = 0.300048828125 : f64>, strides = [1, 1]} : tensor<1x32x256x256x!qElemType>, tensor<12x32x3x3x!qElemType2>, tensor<1x12x1x1xf16> -> tensor<1x12x256x256x!qElemType5>
  %3 = IE.DepthToSpace(%2) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x12x256x256x!qElemType5> -> tensor<1x3x512x512x!qElemType5>
  %4 = IE.Add(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x512x512x!qElemType4>, tensor<1x3x512x512x!qElemType5> -> tensor<1x3x512x512x!qElemType1>
  return %4 : tensor<1x3x512x512x!qElemType1>

  // CHECK:        [[CST:%.+]] = const.Declare tensor<12x32x3x3x!qElemType2> = dense<0> : tensor<12x32x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType3>, #const.ConvertElemType<!qElemType2>]
  // CHECK:        [[CST_0:%.+]] = const.Declare tensor<1x12x1x1xf16> = dense<1.000000e+00> : tensor<1x12x1x1xf16>, [#const.CastElemType<f16>]
  // CHECK:        [[CONVERT:%.+]] = IE.Convert([[INPUT0]]) {dstElemType = f16} : tensor<1x3x512x512xf32> -> tensor<1x3x512x512xf16>
  // CHECK:        [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType4} : tensor<1x3x512x512xf16> -> tensor<1x3x512x512x!qElemType4>
  // CHECK:        [[CONV:%.+]] = IE.Convolution([[INPUT1]], [[CST]], [[CST_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.LeakyRelu<negative_slope = 0.300048828125 : f64>, strides = [1, 1]} : tensor<1x32x256x256x!qElemType>, tensor<12x32x3x3x!qElemType2>, tensor<1x12x1x1xf16> -> tensor<1x12x256x256x!qElemType5>
  // CHECK:        [[DEPTHTOSP:%.+]] = IE.DepthToSpace([[CONV]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x12x256x256x!qElemType5> -> tensor<1x3x512x512x!qElemType5>
  // CHECK:        [[ADD:%.+]] = IE.Add([[QUANTIZE]], [[DEPTHTOSP]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x512x512x!qElemType4>, tensor<1x3x512x512x!qElemType5> -> tensor<1x3x512x512x!qElemType1>
  // CHECK:        return [[ADD]] : tensor<1x3x512x512x!qElemType1>

  }

// -----

!qElemType = !quant.uniform<u8:f16, 0.020894650851978974:128>
!qElemType1 = !quant.uniform<u8:f16, 0.016421539643231561:128>
!qElemType2 = !quant.uniform<u8:f16:0, {0.0021634723625931084:128,0.0025575409917270434:128,0.0038781131015104406:128,0.0032656551576128192:128,0.0020378891159506403:128,0.0035840687798518759:128,0.0028856976359498267:128,0.0023135664416294474:128,0.0034765291447732961:128,0.0033485044451320874:128,0.0035138309001922609:128,0.0018750901900085749:128}>
!qElemType3 = !quant.uniform<i8:f16:0, {0.0021634723625931084,0.0025575409917270434,0.0038781131015104406,0.0032656551576128192,0.0020378891159506403,0.0035840687798518759,0.0028856976359498267,0.0023135664416294474,0.0034765291447732961,0.0033485044451320874,0.0035138309001922609,0.0018750901900085749}>
!qElemType4 = !quant.uniform<u8:f16, 0.0071074686798394899:128>
!qElemType5 = !quant.uniform<u8:f16, 0.022970187430288277:92>

// CHECK-LABEL: func.func @ConvPostOpBeforeAddFused
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x3x512x512xf32>, [[INPUT1:%.+]]: tensor<1x32x256x256x!qElemType>) -> tensor<1x3x512x512x!qElemType1>
func.func @ConvPostOpBeforeAddFused(%arg0: tensor<1x3x512x512xf32>, %arg1: tensor<1x32x256x256x!qElemType>) -> tensor<1x3x512x512x!qElemType1> {
  %cst = const.Declare tensor<12x32x3x3x!qElemType2> = dense<0> : tensor<12x32x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType3>, #const.ConvertElemType<!qElemType2>]
  %cst_0 = const.Declare tensor<1x12x1x1xf16> = dense<1.0> : tensor<1x12x1x1xf16>, [#const.CastElemType<f16>]
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x512x512xf32> -> tensor<1x3x512x512xf16>
  %1 = IE.Quantize(%0) {dstElemType = !qElemType4} : tensor<1x3x512x512xf16> -> tensor<1x3x512x512x!qElemType4>
  %2 = IE.Convolution(%arg1, %cst, %cst_0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x32x256x256x!qElemType>, tensor<12x32x3x3x!qElemType2>, tensor<1x12x1x1xf16> -> tensor<1x12x256x256x!qElemType5>
  %3 = IE.DepthToSpace(%2) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x12x256x256x!qElemType5> -> tensor<1x3x512x512x!qElemType5>
  %4 = IE.Add(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x512x512x!qElemType4>, tensor<1x3x512x512x!qElemType5> -> tensor<1x3x512x512x!qElemType1>
  return %4 : tensor<1x3x512x512x!qElemType1>

  // CHECK:        [[CST:%.+]] = const.Declare tensor<12x32x3x3x!qElemType2> = dense<0> : tensor<12x32x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType3>, #const.ConvertElemType<!qElemType2>]
  // CHECK:        [[CST_0:%.+]] = const.Declare tensor<1x12x1x1xf16> = dense<1.000000e+00> : tensor<1x12x1x1xf16>, [#const.CastElemType<f16>]
  // CHECK:        [[CONVERT:%.+]] = IE.Convert([[INPUT0]]) {dstElemType = f16} : tensor<1x3x512x512xf32> -> tensor<1x3x512x512xf16>
  // CHECK:        [[CONV:%.+]] = IE.Convolution([[INPUT1]], [[CST]], [[CST_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x32x256x256x!qElemType>, tensor<12x32x3x3x!qElemType2>, tensor<1x12x1x1xf16> -> tensor<1x12x256x256xf16>
  // CHECK:        [[DEPTHTOSP:%.+]] = IE.DepthToSpace([[CONV]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x12x256x256xf16> -> tensor<1x3x512x512xf16>
  // CHECK:        [[ADD:%.+]] = IE.Add([[CONVERT]], [[DEPTHTOSP]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x512x512xf16>, tensor<1x3x512x512xf16> -> tensor<1x3x512x512x!qElemType1>
  // CHECK:        return [[ADD]] : tensor<1x3x512x512x!qElemType1>
  }
