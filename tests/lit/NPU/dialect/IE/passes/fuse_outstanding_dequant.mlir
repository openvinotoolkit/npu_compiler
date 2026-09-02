//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-outstanding-dequant %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

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

// CHECK-LABEL: func.func @GroupConvWithNonReLUxPostOpOutstandingDequantNoRemoval
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @GroupConvWithNonReLUxPostOpOutstandingDequantNoRemoval(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
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

// CHECK-LABEL: func.func @AddWithNonReLUxPostOpOutstandingDequantNoRemoval
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @AddWithNonReLUxPostOpOutstandingDequantNoRemoval(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
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

!qElemType = !quant.uniform<u8:f16, 0.12009554844276578:255>

// CHECK-LABEL: func.func @MaxPoolDequantizeNoRemoval
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<4x64x200x63x!qElemType>) -> tensor<4x64x1x1xf16>
func.func @MaxPoolDequantizeNoRemoval(%arg0: tensor<4x64x200x63x!qElemType>) -> tensor<4x64x1x1xf16> {
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
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {
    // CHECK-SAME:   dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME: } : tensor<1x384x20x20xf16>, tensor<384x384x1x1x!qElemType> -> tensor<1x384x20x20xf16>

    // CHECK-NOT:  IE.Dequantize
    // CHECK:  [[CLAMP:%.+]] = IE.Clamp([[CONV]]) {max = 1.671716570854187 : f64, min = 0.000000e+00 : f64} : tensor<1x384x20x20xf16> -> tensor<1x384x20x20xf16>

    // CHECK: return [[CLAMP]] : tensor<1x384x20x20xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: func.func @Conv2dSoftMaxWithOutstandingDequantWithImplicitReLUx
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @Conv2dSoftMaxWithOutstandingDequantWithImplicitReLUx(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
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
  // CHECK-SAME:   dilations = [1, 1],
  // CHECK-SAME:   pads_begin = [0, 0],
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:  IE.Dequantize
  // CHECK:      [[CLAMP:%.+]] = IE.Clamp([[CONV]]) {max = 0.64300000667572021 : f64, min = 0.000000e+00 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:      [[SOFTMAX:%.+]] = IE.SoftMax([[CLAMP]]) {axisInd = 1 : i64} :
  // CHECK-SAME:   tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[SOFTMAX]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803:120>

// CHECK-LABEL: func.func @Conv2dSoftMaxWithOutstandingDequantWithoutImplicitReLUx
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16>
func.func @Conv2dSoftMaxWithOutstandingDequantWithoutImplicitReLUx(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
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

  // CHECK:      [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
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

  // CHECK:      [[GROUPCONV:%.+]] = IE.GroupConvolution([[INPUT]], [[CST]]) {
  // CHECK-SAME:   dilations = [1, 1]
  // CHECK-SAME:   groups = 16 : i64,
  // CHECK-SAME:   pads_begin = [0, 0
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:  IE.Dequantize

  // CHECK:      [[SOFTMAX:%.+]] = IE.SoftMax([[GROUPCONV]]) {axisInd = 1 : i64} :
  // CHECK-SAME:   tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[SOFTMAX]] : tensor<1x16x3x3xf16>
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

  // CHECK:      [[GROUPCONV:%.+]] = IE.GroupConvolution([[INPUT]], [[CST]]) {
  // CHECK-SAME:   dilations = [1, 1]
  // CHECK-SAME:   groups = 16 : i64,
  // CHECK-SAME:   pads_begin = [0, 0
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:  IE.Dequantize

  // CHECK:      [[SOFTMAX:%.+]] = IE.SoftMax([[GROUPCONV]]) {axisInd = 1 : i64} :
  // CHECK-SAME:   tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:      return [[SOFTMAX]] : tensor<1x16x3x3xf16>
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

  // CHECK:       [[AVGPOOL:%.+]] = IE.AvgPool([[INPUT]]) {
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


  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[SOFTMAX]] : tensor<1x16x3x3xf16>
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

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

  // CHECK:       return [[SOFTMAX]] : tensor<1x16x3x3xf16>
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

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25xf16>

  // CHECK:       [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[ADD]]) {
  // CHECK-SAME{LITERAL}:dim_mapping = [[0], [0], [1], [2, 3]],
  // CHECK-SAME:    shape_value = [2, 48, 5, 5]
  // CHECK-SAME:  } : tensor<1x2x48x25xf16> -> tensor<2x48x5x5xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[AFFINE_RESHAPE]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<2x48x5x5xf16> -> tensor<2x48x5x5xf16>

  // CHECK:       return [[SOFTMAX]] : tensor<2x48x5x5xf16>
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

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x2x48x25xf16>, tensor<1x2x48x25xf16> -> tensor<1x2x48x25xf16>

  // CHECK:       [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[ADD]]) {
  // CHECK-SAME{LITERAL}:dim_mapping = [[0], [0], [1], [2, 3]],
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

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {
  // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>
  // CHECK-SAME:  } : tensor<1x16x48x48xf16>, tensor<1x16x48x48xf16> -> tensor<1x16x48x48xf16>

  // CHECK:       [[SLICE:%.+]] = IE.Slice [[ADD]] [0, 0, 8, 8] [1, 16, 32, 32] :
  // CHECK-SAME:    tensor<1x16x48x48xf16> to tensor<1x16x32x32xf16>

  // CHECK-NOT:   IE.Dequantize

  // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[SLICE]]) {axisInd = 1 : i64} :
  // CHECK-SAME:    tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

  // CHECK:       return [[SOFTMAX]] : tensor<1x16x32x32xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {0.0048406556540844491:128,0.0028785893730088777:128,0.0037009462422015619:128,0.0028234841776829142:128,0.0036912409698261935:128,0.0031632916600096458:128,0.00418546340044807:128,0.0027643728489969289:128,0.0060231732387168732:128,0.0028949776116539449:128,0.0044727944860271382:128,0.0053863244898178994:128,0.0038609912582472259:128,0.0028460528336319269:128,0.0047786067513858567:128,0.0029905208185607313:128}>
!qElemType1 = !quant.uniform<i8:f16:0, {0.0048406556540844491,0.0028785893730088777,0.0037009462422015619,0.0028234841776829142,0.0036912409698261935,0.0031632916600096458,0.00418546340044807,0.0027643728489969289,0.0060231732387168732,0.0028949776116539449,0.0044727944860271382,0.0053863244898178994,0.0038609912582472259,0.0028460528336319269,0.0047786067513858567,0.0029905208185607313}>
!qElemType2 = !quant.uniform<u8:f16, 0.011894785189161114>
!qElemType3 = !quant.uniform<u8:f16, 0.0039216639948826213>
!qElemTypeDequantizeInput = !quant.uniform<u8:f16, 0.017833509632185395>        // float range: [0 to 4.547544956207275]
!qElemTypeQuantizeOutput = !quant.uniform<u8:f16, 0.019608233021754844:127>     // float range: [-2.490246 to 2.509854]

// CHECK-LABEL:  @QuantizedConvWithImplicitReLUx
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16>
func.func @QuantizedConvWithImplicitReLUx(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
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
  // CHECK-SAME:    dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x1x1x!qElemType2>, tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

  // CHECK-NOT:  IE.Dequantize
  // CHECK:      [[CLAMP:%.+]] = IE.Clamp([[CONV]]) {max = 4.5475449562072754 : f64, min = 0.000000e+00 : f64} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

  // CHECK:      [[QNT_0:%.+]] = IE.Quantize([[CLAMP]]) {dstElemType = !qElemType3} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType3>
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

// CHECK-LABEL:  @QuantizedConvWithoutImplicitReLUxUnsigned
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16>
func.func @QuantizedConvWithoutImplicitReLUxUnsigned(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
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
  // CHECK-SAME:    dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x1x1x!qElemType2>, tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

  // CHECK-NOT:  IE.Dequantize
  // CHECK-NOT:  IE.ReLU

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

// CHECK-LABEL:  @QuantizedConvWithoutImplicitReLUxSigned
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16>
func.func @QuantizedConvWithoutImplicitReLUxSigned(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
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
  // CHECK-SAME:    dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
  // CHECK-SAME: } : tensor<1x16x1x1x!qElemType2>, tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

  // CHECK-NOT:  IE.Dequantize
  // CHECK-NOT:  IE.ReLU

  // CHECK:      [[QNT_0:%.+]] = IE.Quantize([[CONV]]) {dstElemType = !qElemType3} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType3>
  // CHECK:      [[QNT_CAST:%.+]] = IE.QuantizeCast([[QNT_0]]) {dstElemType = !qElemType4} : tensor<1x16x1x1x!qElemType3> -> tensor<1x16x1x1x!qElemType4>
  // CHECK:      [[DQ:%.+]] = IE.Dequantize([[QNT_CAST]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType4> -> tensor<1x16x1x1xf16>

  // CHECK:      return [[DQ]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.045763225181429994>

// CHECK-LABEL: func.func @ConvWithReLUPostOpAndDequantizeExplicitReLUx
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<151x768x1x1xf16>) -> tensor<151x1536x1x1xf16>
func.func @ConvWithReLUPostOpAndDequantizeExplicitReLUx(%arg0: tensor<151x768x1x1xf16>) -> tensor<151x1536x1x1xf16> {
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
  // CHECK-SAME:   dilations = [1, 1],
  // CHECK-SAME:   pads_begin = [0, 0],
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   post_op = #IE.Relu<>,
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<151x768x1x1xf16>, tensor<1536x768x1x1xf16>, tensor<1x1536x1x1xf16> -> tensor<151x1536x1x1xf16>

  // CHECK-NOT:  IE.Dequantize
  // CHECK:      [[CLAMP:%.+]] = IE.Clamp([[CONV]]) {max = 11.669622421264648 : f64, min = 0.000000e+00 : f64} : tensor<151x1536x1x1xf16> -> tensor<151x1536x1x1xf16>

  // CHECK:      return [[CLAMP]] : tensor<151x1536x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.061515766966576672:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.002958523058423809:128,0.00278550994162466:128,0.0080354947669833317:128,0.0042945681833753396:128,0.0053018885500290816:128,0.006951279499951531:128,0.0043906863997964299:128,0.0033814437249127557:128,0.005055825616799149:128,0.0058785987835304407:128,0.0048059178333656463:128,0.0050327572168088423:128,0.004309947116702211:128,0.003404512124903062:128,0.0031238465916876698:128,0.0041600023998933679:128,0.0052096149500678564:128,0.0075279897334528901:128,0.010842150800368365:128,0.0065591167001163254:128,0.0037678396000581627:128,0.0051096518834431968:128,0.0072242556833753396:128,0.0047559361831814636:128,0.0057824803333656463:128,0.005344180499806124:128,0.0041446234665664973:128,0.0030046598584044211:128,0.0089428518332687074:128,0.0049443282333074831:128,0.0034237359084335027:128,0.0060170091834722784:128,0.0040638841834722784:128,0.0039120171584335027:128,0.0056632934832105452:128,0.0055748647334528901:128,0.0043599285331426879:128,0.0040446603999418369:128,0.004252275999854593:128,0.0042484313833947279:128,0.0035967489083607992:128,0.0062284693998448989:128,0.0050788940167894548:128,0.0040946420501260196:128,0.0030834768332687079:128,0.0035025528832977894:128,0.0058785987835304407:128,0.0036736435749951529:128,0.006116972250096938:128,0.0047213337000678564:128,0.0082507998335595231:128,0.0029950479666392008:128,0.0059324250501744885:128,0.0039350855584238086:128,0.0078970843670414948:128,0.0064860666499418369:128,0.0047136442334044211:128,0.0046905758334141153:128,0.0048174519164889467:128,0.0053480253500096941:128,0.0028201125416101192:128,0.004913570366653742:128,0.0048482097831426879:128,0.005832461749806124:128,0.0054787462832880957:128,0.0030642531666101192:128,0.0046944204498739804:128,0.0078009656831329946:128,0.0028431809416004255:128,0.0033622200582541671:128,0.003777451374951531:128,0.0031968965249903063:128,0.003963921116847618:128,0.0032776359249563776:128,0.0037255475334092682:128,0.0039081723082299326:128,0.0058093933498158173:128,0.0053326464166828227:128,0.0069051426999709184:128,0.0050058439666149663:128,0.0079278422336952359:128,0.0079739790336758476:128,0.0051288754332299326:128,0.0029527558999903063:128,0.0052057700998642863:128,0.0027912771000581627:128,0.0055171936166052721:128,0.0066129429667603733:128,0.0083123155668670054:128,0.0035986712165907318:128,0.0060554565167894548:128,0.0029662125250872443:128,0.0030527189666149663:128,0.0061208168665568032:128,0.0053941621499903059:128,0.0047520915667215984:128,0.006320742999806124:128,0.0041600023998933679:128,0.0052403728167215984:128,0.0041100209834528901:128,0.0038851039082396263:128,0.0050442912999321436:128,0.0078086551497964299:128,0.0050673596999224494:128,0.0074434056001551011:128,0.0025951955832687079:128,0.0040638841834722784:128,0.005832461749806124:128,0.0056556040165471099:128,0.0053326464166828227:128,0.0039158616580215154:128,0.0026317206083559521:128,0.005774790866702211:128,0.0023779680915907318:128,0.003533310749951531:128,0.0053595594331329946:128,0.0077971210666731294:128,0.0042638103167215984:128,0.008673720266304764:128,0.0078970843670414948:128,0.0054056964668573121:128,0.014671506133734011:128,0.0069743478999418369:128,0.0087659938662659891:128,0.0033756766833511055:128,0.0064822220334819717:128,0.010211614066479253:128,0.0040100579168282306:128}>
!qElemType2 = !quant.uniform<u8:f16, 0.063098907470703125:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: func.func @ConvolutionWithLeakyReLUPostOpOutstandingDequantNoRemoval
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x256x13x13x!qElemType>) -> tensor<1x128x13x13xf16>
func.func @ConvolutionWithLeakyReLUPostOpOutstandingDequantNoRemoval(%arg0: tensor<1x256x13x13x!qElemType>) -> tensor<1x128x13x13xf16> {
  %cst = const.Declare tensor<128x256x1x1x!qElemType1> = dense<1> : tensor<128x256x1x1xsi8>, [#const.CastElemType<f16>]
  %cst_0 = const.Declare tensor<1x128x1x1xf16> = dense<1.0> : tensor<1x128x1x1xf16>, [#const.CastElemType<f16>]

  %0 = IE.Convolution(%arg0, %cst, %cst_0) {
    dilations = [1, 1],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>,
    strides = [1, 1]
  } : tensor<1x256x13x13x!qElemType>, tensor<128x256x1x1x!qElemType1>, tensor<1x128x1x1xf16> -> tensor<1x128x13x13x!qElemType2>

  %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x128x13x13x!qElemType2> -> tensor<1x13x13x128x!qElemType2>
  %2 = IE.Transpose(%1) {order_value = #NWCH} : tensor<1x13x13x128x!qElemType2> -> tensor<1x128x13x13x!qElemType2>

  %3 = IE.Dequantize(%2) {
    dstElemType = f16
  } : tensor<1x128x13x13x!qElemType2> -> tensor<1x128x13x13xf16>

  return %3 : tensor<1x128x13x13xf16>

  // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<128x256x1x1x!qElemType1> = dense<1> :
  // CHECK-SAME:   tensor<128x256x1x1xsi8>, [#const.CastElemType<f16>]
  // CHECK-DAG:  [[CST_0:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<1.000000e+00> :
  // CHECK-SAME:   tensor<1x128x1x1xf16>, [#const.CastElemType<f16>]

  // CHECK:      [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]], [[CST_0]]) {
  // CHECK-SAME:   dilations = [1, 1],
  // CHECK-SAME:   pads_begin = [0, 0],
  // CHECK-SAME:   pads_end = [0, 0],
  // CHECK-SAME:   post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>,
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: } : tensor<1x256x13x13x!qElemType>, tensor<128x256x1x1x!qElemType1>, tensor<1x128x1x1xf16> -> tensor<1x128x13x13x!qElemType2>

  // CHECK:      [[TRANSPOSE_1:%.+]] = IE.Transpose([[CONV]]) {order_value = #NHWC} :
  // CHECK-SAME:   tensor<1x128x13x13x!qElemType2> -> tensor<1x13x13x128x!qElemType2>

  // CHECK:      [[TRANSPOSE_2:%.+]] = IE.Transpose([[TRANSPOSE_1]]) {order_value = #NWCH} :
  // CHECK-SAME:   tensor<1x13x13x128x!qElemType2> -> tensor<1x128x13x13x!qElemType2>

  // CHECK:      [[DQ:%.+]] = IE.Dequantize([[TRANSPOSE_2]]) {
  // CHECK-SAME:   dstElemType = f16
  // CHECK-SAME: } : tensor<1x128x13x13x!qElemType2> -> tensor<1x128x13x13xf16>

  // CHECK:      return [[DQ]] : tensor<1x128x13x13xf16>
}
