//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-outstanding-dequant %s | FileCheck %s
// REQUIRES: platform-NPU5010 || platform-NPU6010


!qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.012968034837760177:121>

// CHECK-LABEL: func.func @ConvWithoutOutstandingDequantClampLowHighNotZeroAndOutputNotQuantized
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x384x20x20xf16>) -> tensor<1x384x20x20xf16>
func.func @ConvWithoutOutstandingDequantClampLowHighNotZeroAndOutputNotQuantized(%arg0: tensor<1x384x20x20xf16>) -> tensor<1x384x20x20xf16> {
    %cst = const.Declare tensor<384x384x1x1x!quant.uniform<i8:f16, 1.000000e+00>> = dense<1> : tensor<384x384x1x1xsi8>

    %0 = IE.Convolution(%arg0, %cst) {clamp = {max = 1.671716570854187 : f64, min = -1.4173249006271362 : f64}, dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x384x20x20xf16>, tensor<384x384x1x1x!qElemType> -> tensor<1x384x20x20x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x384x20x20x!qElemType1> -> tensor<1x384x20x20xf16>
    return %1 : tensor<1x384x20x20xf16>

    // CHECK:      [[CST:%.+]] = const.Declare tensor<384x384x1x1x!qElemType> = dense<1> : tensor<384x384x1x1xsi8>

    // CHECK:      [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {
    // CHECK:          clamp = {max = 1.671716570854187 : f64, min = -1.4173249006271362 : f64}
    // CHECK-SAME:     dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME: } : tensor<1x384x20x20xf16>, tensor<384x384x1x1x!qElemType> -> tensor<1x384x20x20xf16>

    // CHECK-NOT:  IE.Dequantize

    // CHECK:      return [[CONV]] : tensor<1x384x20x20xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803:127>

// CHECK-LABEL: func.func @Conv2dSoftMaxWithOutstandingDequantClamp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @Conv2dSoftMaxWithOutstandingDequantClamp(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
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

  // CHECK:      [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
  // CHECK:        clamp = {max = 0.32276079058647156 : f64, min = -0.32023921608924866 : f64}
  // CHECK-SAME:   dilations = [1, 1],
  // CHECK-SAME:   pads_begin = [0, 0],
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:  IE.Dequantize

  // CHECK:      [[SOFTMAX:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64} :
  // CHECK-SAME:   tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[SOFTMAX]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: func.func @Conv2dSoftMaxWithOutstandingDequantPerAxesClampUnsupported
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @Conv2dSoftMaxWithOutstandingDequantPerAxesClampUnsupported(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
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

  // CHECK:      [[GROUP_CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
  // CHECK-SAME:   dilations = [1, 1],
  // CHECK-SAME:   pads_begin = [0, 0],
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  // CHECK:      [[DQ:%.+]] = IE.Dequantize([[GROUP_CONV]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  // CHECK:      [[SOFTMAX:%.+]] = IE.SoftMax([[DQ]]) {axisInd = 1 : i64} :
  // CHECK-SAME:   tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[SOFTMAX]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803:127>

// CHECK-LABEL: func.func @GroupConvSoftMaxWithOutstandingDequantClamp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @GroupConvSoftMaxWithOutstandingDequantClamp(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
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

  // CHECK:      [[GROUP_CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[CST]]) {
  // CHECK:        clamp = {max = 0.32276079058647156 : f64, min = -0.32023921608924866 : f64}
  // CHECK-SAME:   dilations = [1, 1]
  // CHECK-SAME:   groups = 16 : i64,
  // CHECK-SAME:   pads_begin = [0, 0]
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:  IE.Dequantize

  // CHECK:      [[SOFTMAX:%.+]] = IE.SoftMax([[GROUP_CONV]]) {axisInd = 1 : i64} :
  // CHECK-SAME:   tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[SOFTMAX]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: func.func @GroupConvSoftMaxWithOutstandingDequantPerAxesClampUnsupported
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @GroupConvSoftMaxWithOutstandingDequantPerAxesClampUnsupported(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
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

  // CHECK:      [[GROUP_CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[CST]]) {
  // CHECK-SAME:   dilations = [1, 1]
  // CHECK-SAME:   groups = 16 : i64,
  // CHECK-SAME:   pads_begin = [0, 0
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType>

  // CHECK:      [[DQ:%.+]] = IE.Dequantize([[GROUP_CONV]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  // CHECK:      [[SOFTMAX:%.+]] = IE.SoftMax([[DQ]]) {axisInd = 1 : i64} :
  // CHECK-SAME:   tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[SOFTMAX]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803:45>

// CHECK-LABEL: func.func @AvgPoolSoftMaxWithOutstandingDequantClamp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @AvgPoolSoftMaxWithOutstandingDequantClamp(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
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

  // CHECK:       [[AVGPOOL:%.+]] = IE.AvgPool([[INPUT]]) {
  // CHECK:         clamp = {max = 0.5295293927192688 : f64, min = -0.11347059160470963 : f64}
  // CHECK-SAME:    kernel_size = [1, 1],
  // CHECK-SAME:    pads_begin = [0, 0],
  // CHECK-SAME:    pads_end = [0, 0],
  // CHECK-SAME:    rounding_type = #IE.rounding_type<FLOOR>,
  // CHECK-SAME:    strides = [1, 1]

  // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[AVGPOOL]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[SOFTMAX]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

// CHECK-LABEL: func.func @AddSoftMaxWithOutstandingDequantClamp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @AddSoftMaxWithOutstandingDequantClamp(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.Dequantize(%0) {
    dstElemType = f16
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  return %2 : tensor<1x16x3x3xf16>

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK:         clamp = {max = 0.50196081399917603 : f64, min = -0.49803921580314636 : f64}
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[SOFTMAX]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: func.func @AddSoftMaxWithOutstandingDequantPerAxesClampUnsupported
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @AddSoftMaxWithOutstandingDequantPerAxesClampUnsupported(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  %1 = IE.Dequantize(%0) {
    dstElemType = f16
  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  return %2 : tensor<1x16x3x3xf16>

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

  // CHECK:       [[DQ:%.+]] = IE.Dequantize([[ADD]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[DQ]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[SOFTMAX]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

// CHECK-LABEL: func.func @AddAffineReshapeWithOutstandingDequantClamp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x2x48x25xf16>) -> tensor<2x48x5x5xf16>
func.func @AddAffineReshapeWithOutstandingDequantClamp(%arg0: tensor<1x2x48x25xf16>) -> tensor<2x48x5x5xf16> {
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

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK:         clamp = {max = 0.50196081399917603 : f64, min = -0.49803921580314636 : f64}
  // CHECK-SAME:  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25xf16>

  // CHECK:       [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[ADD]]) {
  // CHECK-SAME{LITERAL}: dim_mapping = [[0], [0], [1], [2, 3]],
  // CHECK-SAME:    shape_value = [2, 48, 5, 5]
  // CHECK-SAME:  } : tensor<1x2x48x25xf16> -> tensor<2x48x5x5xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[RESHAPE:%.+]] = IE.SoftMax([[AFFINE_RESHAPE]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<2x48x5x5xf16> -> tensor<2x48x5x5xf16>

  // CHECK:       return [[RESHAPE]] : tensor<2x48x5x5xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

// CHECK-LABEL: func.func @AddAffineReshapeReshapeWithOutstandingDequantClamp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x2x48x25xf16>) -> tensor<2x48x25xf16>
func.func @AddAffineReshapeReshapeWithOutstandingDequantClamp(%arg0: tensor<1x2x48x25xf16>) -> tensor<2x48x25xf16> {
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

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK:         clamp = {max = 0.50196081399917603 : f64, min = -0.49803921580314636 : f64}
  // CHECK-SAME:  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25xf16>

  // CHECK:       [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[ADD]]) {
  // CHECK-SAME{LITERAL}: dim_mapping = [[0], [0], [1], [2, 3]],
  // CHECK-SAME:    shape_value = [2, 48, 5, 5]
  // CHECK-SAME:  } : tensor<1x2x48x25xf16> -> tensor<2x48x5x5xf16>

  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[AFFINE_RESHAPE]]) {
  // CHECK-SAME:    shape_value = [2, 48, 25]
  // CHECK-SAME:  } : tensor<2x48x5x5xf16> -> tensor<2x48x25xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[RESHAPE]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<2x48x25xf16> -> tensor<2x48x25xf16>

  // CHECK:       return [[SOFTMAX]] : tensor<2x48x25xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {0.0048406556540844491:128,0.0028785893730088777:128,0.0037009462422015619:128,0.0028234841776829142:128,0.0036912409698261935:128,0.0031632916600096458:128,0.00418546340044807:128,0.0027643728489969289:128,0.0060231732387168732:128,0.0028949776116539449:128,0.0044727944860271382:128,0.0053863244898178994:128,0.0038609912582472259:128,0.0028460528336319269:128,0.0047786067513858567:128,0.0029905208185607313:128}>
!qElemType1 = !quant.uniform<i8:f16:0, {0.0048406556540844491,0.0028785893730088777,0.0037009462422015619,0.0028234841776829142,0.0036912409698261935,0.0031632916600096458,0.00418546340044807,0.0027643728489969289,0.0060231732387168732,0.0028949776116539449,0.0044727944860271382,0.0053863244898178994,0.0038609912582472259,0.0028460528336319269,0.0047786067513858567,0.0029905208185607313}>
!qElemType2 = !quant.uniform<u8:f16, 0.011894785189161114>
!qElemType3 = !quant.uniform<u8:f16, 0.0039216639948826213>
!qElemTypeDequantizeInput = !quant.uniform<u8:f16, 0.017833509632185395:90>
!qElemTypeQuantizeOutput = !quant.uniform<u8:f16, 0.019608233021754844:127>

// CHECK-LABEL:  @QuantizedConvWithImplicitClamp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16>
func.func @QuantizedConvWithImplicitClamp(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %cst = const.Declare tensor<16x16x1x1x!qElemType> = dense<0> : tensor<16x16x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>]
  %cst_0 = const.Declare tensor<1x16x1x1xf16> = dense<0.000000e+00> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]

  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType2>

  %1 = IE.Convolution(%0, %cst, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType2>, tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemTypeDequantizeInput>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemTypeDequantizeInput> -> tensor<1x16x1x1xf16>

  %3 = IE.Quantize(%2) {dstElemType = !qElemTypeQuantizeOutput} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemTypeQuantizeOutput>
  %4 = IE.QuantizeCast(%3) {dstElemType = !qElemType3} : tensor<1x16x1x1x!qElemTypeQuantizeOutput> -> tensor<1x16x1x1x!qElemType3>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x16x1x1x!qElemType3> -> tensor<1x16x1x1xf16>

  return %5 : tensor<1x16x1x1xf16>

  // CHECK:      [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> = dense<0> : tensor<16x16x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>]
  // CHECK:      [[CST_0:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<0.000000e+00> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK:      [[QNT:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType2} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType2>

  // CHECK:      [[CONV:%.+]] = IE.Convolution([[QNT]], [[CST]], [[CST_0]]) {
  // CHECK:         clamp = {max = 2.9425292015075684 : f64, min = -1.6050158739089966 : f64}
  // CHECK-SAME:    dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x1x1x!qElemType2>, tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

  // CHECK-NOT:  IE.Dequantize

  // CHECK:      [[QNT_0:%.+]] = IE.Quantize([[CONV]]) {dstElemType = !qElemType3} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType3>
  // CHECK:      [[QNT_CAST:%.+]] = IE.QuantizeCast([[QNT_0]]) {dstElemType = !qElemType4} : tensor<1x16x1x1x!qElemType3> -> tensor<1x16x1x1x!qElemType4>
  // CHECK:      [[DQ:%.+]] = IE.Dequantize([[QNT_CAST]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType4> -> tensor<1x16x1x1xf16>

  // CHECK:      return [[DQ]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {0.0048406556540844491:128,0.0028785893730088777:128,0.0037009462422015619:128,0.0028234841776829142:128,0.0036912409698261935:128,0.0031632916600096458:128,0.00418546340044807:128,0.0027643728489969289:128,0.0060231732387168732:128,0.0028949776116539449:128,0.0044727944860271382:128,0.0053863244898178994:128,0.0038609912582472259:128,0.0028460528336319269:128,0.0047786067513858567:128,0.0029905208185607313:128}>
!qElemType1 = !quant.uniform<i8:f16:0, {0.0048406556540844491,0.0028785893730088777,0.0037009462422015619,0.0028234841776829142,0.0036912409698261935,0.0031632916600096458,0.00418546340044807,0.0027643728489969289,0.0060231732387168732,0.0028949776116539449,0.0044727944860271382,0.0053863244898178994,0.0038609912582472259,0.0028460528336319269,0.0047786067513858567,0.0029905208185607313}>
!qElemType2 = !quant.uniform<u8:f16, 0.011894785189161114>
!qElemType3 = !quant.uniform<u8:f16, 0.0039216639948826213>
!qElemTypeDequantizeInput = !quant.uniform<u8:f16, 0.017833509632185395:90>
!qElemTypeQuantizeOutput = !quant.uniform<u8:f16, 0.019608233021754844:127>

// CHECK-LABEL:  @QuantizedConvWithImplicitClampUnsigned
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16>
func.func @QuantizedConvWithImplicitClampUnsigned(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %cst = const.Declare tensor<16x16x1x1x!qElemType> = dense<0> : tensor<16x16x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>]
  %cst_0 = const.Declare tensor<1x16x1x1xf16> = dense<0.000000e+00> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]

  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType2>

  %1 = IE.Convolution(%0, %cst, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType2>, tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemTypeDequantizeInput>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemTypeDequantizeInput> -> tensor<1x16x1x1xf16>

  %3 = IE.Quantize(%2) {dstElemType = !qElemTypeQuantizeOutput} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemTypeQuantizeOutput>
  %4 = IE.QuantizeCast(%3) {dstElemType = !qElemType3} : tensor<1x16x1x1x!qElemTypeQuantizeOutput> -> tensor<1x16x1x1x!qElemType3>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x16x1x1x!qElemType3> -> tensor<1x16x1x1xf16>

  return %5 : tensor<1x16x1x1xf16>

  // CHECK:      [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> = dense<0> : tensor<16x16x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>]
  // CHECK:      [[CST_0:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<0.000000e+00> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK:      [[QNT:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType2} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType2>

  // CHECK:      [[CONV:%.+]] = IE.Convolution([[QNT]], [[CST]], [[CST_0]]) {
  // CHECK:         clamp = {max = 2.9425292015075684 : f64, min = -1.6050158739089966 : f64}
  // CHECK-SAME:    dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x1x1x!qElemType2>, tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

  // CHECK-NOT:  IE.Dequantize

  // CHECK:      [[QNT_0:%.+]] = IE.Quantize([[CONV]]) {dstElemType = !qElemType3} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType3>
  // CHECK:      [[QNT_CAST:%.+]] = IE.QuantizeCast([[QNT_0]]) {dstElemType = !qElemType4} : tensor<1x16x1x1x!qElemType3> -> tensor<1x16x1x1x!qElemType4>
  // CHECK:      [[DQ:%.+]] = IE.Dequantize([[QNT_CAST]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType4> -> tensor<1x16x1x1xf16>

  // CHECK:      return [[DQ]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {0.0048406556540844491:128,0.0028785893730088777:128,0.0037009462422015619:128,0.0028234841776829142:128,0.0036912409698261935:128,0.0031632916600096458:128,0.00418546340044807:128,0.0027643728489969289:128,0.0060231732387168732:128,0.0028949776116539449:128,0.0044727944860271382:128,0.0053863244898178994:128,0.0038609912582472259:128,0.0028460528336319269:128,0.0047786067513858567:128,0.0029905208185607313:128}>
!qElemType1 = !quant.uniform<i8:f16:0, {0.0048406556540844491,0.0028785893730088777,0.0037009462422015619,0.0028234841776829142,0.0036912409698261935,0.0031632916600096458,0.00418546340044807,0.0027643728489969289,0.0060231732387168732,0.0028949776116539449,0.0044727944860271382,0.0053863244898178994,0.0038609912582472259,0.0028460528336319269,0.0047786067513858567,0.0029905208185607313}>
!qElemType2 = !quant.uniform<u8:f16, 0.011894785189161114>
!qElemType3 = !quant.uniform<i8:f16, 0.0039216639948826213>
!qElemTypeDequantizeInput = !quant.uniform<i8:f16, 0.017833509632185395>
!qElemTypeQuantizeOutput = !quant.uniform<i8:f16, 0.019608233021754844>


// CHECK-LABEL:  @QuantizedConvWithImplicitClampSigned
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16>
func.func @QuantizedConvWithImplicitClampSigned(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %cst = const.Declare tensor<16x16x1x1x!qElemType> = dense<0> : tensor<16x16x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>]

  %cst_0 = const.Declare tensor<1x16x1x1xf16> = dense<0.000000e+00> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]

  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType2>

  %1 = IE.Convolution(%0, %cst, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType2>, tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemTypeDequantizeInput>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemTypeDequantizeInput> -> tensor<1x16x1x1xf16>

  %3 = IE.Quantize(%2) {dstElemType = !qElemTypeQuantizeOutput} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemTypeQuantizeOutput>
  %4 = IE.QuantizeCast(%3) {dstElemType = !qElemType3} : tensor<1x16x1x1x!qElemTypeQuantizeOutput> -> tensor<1x16x1x1x!qElemType3>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x16x1x1x!qElemType3> -> tensor<1x16x1x1xf16>

  return %5 : tensor<1x16x1x1xf16>

  // CHECK:      [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> = dense<0> : tensor<16x16x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>]
  // CHECK:      [[CST_0:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<0.000000e+00> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK:      [[QNT:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType2} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType2>

  // CHECK:      [[CONV:%.+]] = IE.Convolution([[QNT]], [[CST]], [[CST_0]]) {
  // CHECK:         clamp = {max = 2.2648556232452393 : f64, min = -2.2826893329620361 : f64}
  // CHECK-SAME:    dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x1x1x!qElemType2>, tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

  // CHECK-NOT:  IE.Dequantize

  // CHECK:      [[QNT_0:%.+]] = IE.Quantize([[CONV]]) {dstElemType = !qElemType3} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType3>
  // CHECK:      [[QNT_CAST:%.+]] = IE.QuantizeCast([[QNT_0]]) {dstElemType = !qElemType4} : tensor<1x16x1x1x!qElemType3> -> tensor<1x16x1x1x!qElemType4>
  // CHECK:      [[DQ:%.+]] = IE.Dequantize([[QNT_CAST]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType4> -> tensor<1x16x1x1xf16>

  // CHECK:      return [[DQ]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

// CHECK-LABEL: func.func @SliceWithOutstandingDequantClamp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x48x48xf16>) -> tensor<1x16x32x32xf16>
func.func @SliceWithOutstandingDequantClamp(%arg0: tensor<1x16x48x48xf16>) -> tensor<1x16x32x32xf16> {
  %0 = IE.Add(%arg0, %arg0) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  } : tensor<1x16x48x48xf16>, tensor<1x16x48x48xf16> -> tensor<1x16x48x48x!qElemType>

  %1 = IE.Slice %0 [0, 0, 8, 8] [1, 16, 32, 32] : tensor<1x16x48x48x!qElemType> to tensor<1x16x32x32x!qElemType>

  %2 = IE.Dequantize(%1) {
    dstElemType = f16
  } : tensor<1x16x32x32x!qElemType> -> tensor<1x16x32x32xf16>

  %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

  return %3 : tensor<1x16x32x32xf16>

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK:         clamp = {max = 0.50196081399917603 : f64, min = -0.49803921580314636 : f64}
  // CHECK-SAME:  } : tensor<1x16x48x48xf16>, tensor<1x16x48x48xf16> -> tensor<1x16x48x48xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[SLICE:%.+]] = IE.Slice [[ADD]] [0, 0, 8, 8] [1, 16, 32, 32] :
  // CHECK-SAME:    tensor<1x16x48x48xf16> to tensor<1x16x32x32xf16>

  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[SLICE]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

  // CHECK:       return [[SOFTMAX]] : tensor<1x16x32x32xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

// CHECK-LABEL: func.func @ConvWithReluPostOpAndDequantizeClamp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<151x768x1x1xf16>) -> tensor<151x1536x1x1xf16>
func.func @ConvWithReluPostOpAndDequantizeClamp(%arg0: tensor<151x768x1x1xf16>) -> tensor<151x1536x1x1xf16> {
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

  // CHECK:      [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) {
  // CHECK:        clamp = {max = 0.50196081399917603 : f64, min = -0.49803921580314636 : f64}
  // CHECK-SAME:   dilations = [1, 1],
  // CHECK-SAME:   pads_begin = [0, 0],
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   post_op = #IE.Relu<>,
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<151x768x1x1xf16>, tensor<1536x768x1x1xf16>, tensor<1x1536x1x1xf16> -> tensor<151x1536x1x1xf16>

  // CHECK-NOT:  IE.Dequantize

  // CHECK:      return [[CONV]] : tensor<151x1536x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

// NCEInvariant is unsupported for Convolution, which makes Clamp unsupported on the Convolution. Ensure creation of Clamp since accuracy is important for these platforms.
// CHECK-LABEL:    func.func @ConvUnsupportedNCEExplicitClamp
// CHECK:     ([[INPUT_0:%.+]]: tensor<1x512x120x16xf16>, [[INPUT_1:%.+]]: tensor<512x512x1x1x!qElemType>) -> tensor<1x512x122x18xf16>
func.func @ConvUnsupportedNCEExplicitClamp(%input_0: tensor<1x512x120x16xf16>, %input_1: tensor<512x512x1x1x!qElemType>) -> tensor<1x512x122x18xf16> {
    %1 = IE.Convolution(%input_0, %input_1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x120x16xf16>, tensor<512x512x1x1x!qElemType> -> tensor<1x512x122x18x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x512x122x18x!qElemType> -> tensor<1x512x122x18xf16>

    return %2 : tensor<1x512x122x18xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT_0]], [[INPUT_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x120x16xf16>, tensor<512x512x1x1x!qElemType> -> tensor<1x512x122x18xf16>
    // CHECK: [[CLAMP:%.+]] = IE.Clamp([[CONV]]) {max = 0.50196081399917603 : f64, min = -0.49803921580314636 : f64} : tensor<1x512x122x18xf16> -> tensor<1x512x122x18xf16>
    // CHECK: return [[CLAMP]] : tensor<1x512x122x18xf16>

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803:127>

// NCEInvariant is unsupported for Convolution, which makes clamp attribute unsupported on the Convolution. Avoid fusing Dequantize
// CHECK-LABEL:    func.func @ConvUnsupportedNCEKeepDequantize
// CHECK:     ([[INPUT_0:%.+]]: tensor<1x512x120x16xf16>, [[INPUT_1:%.+]]: tensor<512x512x1x1x!qElemType>) -> tensor<1x512x122x18xf16>
func.func @ConvUnsupportedNCEKeepDequantize(%input_0: tensor<1x512x120x16xf16>, %input_1: tensor<512x512x1x1x!qElemType>) -> tensor<1x512x122x18xf16> {
    %1 = IE.Convolution(%input_0, %input_1) {clamp = {max = 2.9425292015075684 : f64, min = -1.6050158739089966 : f64}, dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x120x16xf16>, tensor<512x512x1x1x!qElemType> -> tensor<1x512x122x18x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x512x122x18x!qElemType> -> tensor<1x512x122x18xf16>

    return %2 : tensor<1x512x122x18xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT_0]], [[INPUT_1]]) {clamp = {max = 2.9425292015075684 : f64, min = -1.6050158739089966 : f64}, dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x120x16xf16>, tensor<512x512x1x1x!qElemType> -> tensor<1x512x122x18x!qElemType>
    // CHECK: [[DQ:%.+]] = IE.Dequantize([[CONV]]) {dstElemType = f16} : tensor<1x512x122x18x!qElemType> -> tensor<1x512x122x18xf16>
    // CHECK: return [[DQ]] : tensor<1x512x122x18xf16>

}
