//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --propagate-and-fuse-quantize-dequantize %s | FileCheck %s
// REQUIRES: platform-NPU6010

!qElemType = !quant.uniform<u16:f16, 1.5102968551005343E-4:1788>

// CHECK-LABEL: @DoNotPropagateU16Quant
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x192x40x40xf16>)
func.func @DoNotPropagateU16Quant(%arg0: tensor<1x192x40x40xf16>) -> tensor<1x576x40x40x!qElemType> {
  %0 = IE.MaxPool(%arg0) {kernel_size = [5, 5], pads_begin = [2, 2], pads_end = [2, 2], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x192x40x40xf16> -> tensor<1x192x40x40xf16>
  %1 = IE.MaxPool(%arg0) {kernel_size = [9, 9], pads_begin = [4, 4], pads_end = [4, 4], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x192x40x40xf16> -> tensor<1x192x40x40xf16>
  %2 = IE.MaxPool(%arg0) {kernel_size = [13, 13], pads_begin = [6, 6], pads_end = [6, 6], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x192x40x40xf16> -> tensor<1x192x40x40xf16>
  %3 = IE.Concat(%0, %1, %2) {static_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0]]} : tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16> -> tensor<1x576x40x40xf16>
  %4 = IE.Quantize(%3) {dstElemType = !qElemType} : tensor<1x576x40x40xf16> -> tensor<1x576x40x40x!qElemType>
  return %4 : tensor<1x576x40x40x!qElemType>

  // CHECK: [[MAXPOOL_0:%.+]] = IE.MaxPool([[INPUT]]) {kernel_size = [5, 5], pads_begin = [2, 2], pads_end = [2, 2], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x192x40x40xf16> -> tensor<1x192x40x40xf16>
  // CHECK: [[MAXPOOL_1:%.+]] = IE.MaxPool([[INPUT]]) {kernel_size = [9, 9], pads_begin = [4, 4], pads_end = [4, 4], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x192x40x40xf16> -> tensor<1x192x40x40xf16>
  // CHECK: [[MAXPOOL_2:%.+]] = IE.MaxPool([[INPUT]]) {kernel_size = [13, 13], pads_begin = [6, 6], pads_end = [6, 6], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x192x40x40xf16> -> tensor<1x192x40x40xf16>

  // CHECK: [[CONCAT:%.+]] = IE.Concat([[MAXPOOL_0]], [[MAXPOOL_1]], [[MAXPOOL_2]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0]]} : tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16> -> tensor<1x576x40x40xf16>
  // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[CONCAT]]) {dstElemType = !qElemType} : tensor<1x576x40x40xf16> -> tensor<1x576x40x40x!qElemType>
  // CHECK: return [[QUANTIZE]] : tensor<1x576x40x40x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @PropagateDynamicDequantTransposeWithQuantizeDimChanged
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<256x128x1x1xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x128x1x4x!qElemType>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<1x128x1x1xf16>
func.func @PropagateDynamicDequantTransposeWithQuantizeDimChanged(%arg0: tensor<256x128x1x1xf16>, %arg1: tensor<1x128x1x4x!qElemType>, %arg2: tensor<1x128x1x1xf16>) -> tensor<256x4x1x1xf16> {
  %1 = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<1x128x1x4x!qElemType>, tensor<1x128x1x1xf16> -> tensor<1x128x1x4xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 128, 4]} : tensor<1x128x1x4xf16> -> tensor<1x1x128x4xf16>
  %3 = IE.Transpose(%2) {order_value = #NCWH} : tensor<1x1x128x4xf16> -> tensor<1x1x4x128xf16>
  %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 128, 1, 1]} : tensor<1x1x4x128xf16> -> tensor<4x128x1x1xf16>
  %5 = IE.Convolution(%arg0, %4) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<4x128x1x1xf16> -> tensor<256x4x1x1xf16>
  return %5 : tensor<256x4x1x1xf16>


  //CHECK: [[AFFINE_RESHAPE_SCALE_1:%.+]] = IE.AffineReshape([[ARG_2]])
  //CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 128, 1]} : tensor<1x128x1x1xf16> -> tensor<1x1x128x1xf16>
  //CHECK: [[TRANSPOSE_SCALE:%.+]] = IE.Transpose([[AFFINE_RESHAPE_SCALE_1]]) {order_value = #NCWH} : tensor<1x1x128x1xf16> -> tensor<1x1x1x128xf16>
  //CHECK: [[AFFINE_RESHAPE_SCALE_2:%.+]] = IE.AffineReshape([[TRANSPOSE_SCALE]])
  //CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 128, 1, 1]} : tensor<1x1x1x128xf16> -> tensor<1x128x1x1xf16>

  //CHECK: [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[ARG_1]])
  //CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 128, 4]} : tensor<1x128x1x4x!qElemType> -> tensor<1x1x128x4x!qElemType>
  //CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[AFFINE_RESHAPE_1]]) {order_value = #NCWH} : tensor<1x1x128x4x!qElemType> -> tensor<1x1x4x128x!qElemType>
  //CHECK: [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[TRANSPOSE]])
  //CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 128, 1, 1]} : tensor<1x1x4x128x!qElemType> -> tensor<4x128x1x1x!qElemType>

  //CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[AFFINE_RESHAPE_2]], [[AFFINE_RESHAPE_SCALE_2]]) {dstElemType = f16} : tensor<4x128x1x1x!qElemType>, tensor<1x128x1x1xf16> -> tensor<4x128x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[DYN_DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<4x128x1x1xf16> -> tensor<256x4x1x1xf16>
  //CHECK: return [[CONV]] : tensor<256x4x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @PropagateDynamicDequantTranspose
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<256x128x1x1xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x4x128x128x!qElemType>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<1x1x128x1xf16>
func.func @PropagateDynamicDequantTranspose(%arg0: tensor<256x128x1x1xf16>, %arg1: tensor<1x4x128x128x!qElemType>, %arg2: tensor<1x1x128x1xf16>) -> tensor<256x512x1x1xf16> {
  %0 = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<1x4x128x128x!qElemType>, tensor<1x1x128x1xf16> -> tensor<1x4x128x128xf16>
  %1 = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x4x128x128xf16> -> tensor<1x4x128x128xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [512, 128, 1, 1]} : tensor<1x4x128x128xf16> -> tensor<512x128x1x1xf16>
  %3 = IE.Convolution(%arg0, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<512x128x1x1xf16> -> tensor<256x512x1x1xf16>
  return %3 : tensor<256x512x1x1xf16>


  //CHECK: [[TRANSPOSE_SCALE:%.+]] = IE.Transpose([[ARG_2]]) {order_value = #NCWH} : tensor<1x1x128x1xf16> -> tensor<1x1x1x128xf16>
  //CHECK: [[AFFINE_RESHAPE_SCALE:%.+]] = IE.AffineReshape([[TRANSPOSE_SCALE]])
  //CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 128, 1, 1]} : tensor<1x1x1x128xf16> -> tensor<1x128x1x1xf16>

  //CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[ARG_1]]) {order_value = #NCWH} : tensor<1x4x128x128x!qElemType> -> tensor<1x4x128x128x!qElemType>
  //CHECK: [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
  //CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [512, 128, 1, 1]} : tensor<1x4x128x128x!qElemType> -> tensor<512x128x1x1x!qElemType>

  //CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[AFFINE_RESHAPE]], [[AFFINE_RESHAPE_SCALE]]) {dstElemType = f16} : tensor<512x128x1x1x!qElemType>, tensor<1x128x1x1xf16> -> tensor<512x128x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[DYN_DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<512x128x1x1xf16> -> tensor<256x512x1x1xf16>
  //CHECK: return [[CONV]] : tensor<256x512x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @DoNotPropagateDynamicDequantTranspose
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<256x128x1x1xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x4x128x128x!qElemType>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<1x4x128x1xf16>
func.func @DoNotPropagateDynamicDequantTranspose(%arg0: tensor<256x128x1x1xf16>, %arg1: tensor<1x4x128x128x!qElemType>, %arg2: tensor<1x4x128x1xf16>) -> tensor<256x512x1x1xf16> {
  %0 = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<1x4x128x128x!qElemType>, tensor<1x4x128x1xf16> -> tensor<1x4x128x128xf16>
  %1 = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x4x128x128xf16> -> tensor<1x4x128x128xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [512, 128, 1, 1]} : tensor<1x4x128x128xf16> -> tensor<512x128x1x1xf16>
  %3 = IE.Convolution(%arg0, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<512x128x1x1xf16> -> tensor<256x512x1x1xf16>
  return %3 : tensor<256x512x1x1xf16>

  //CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[ARG_1]], [[ARG_2]]) {dstElemType = f16} : tensor<1x4x128x128x!qElemType>, tensor<1x4x128x1xf16> -> tensor<1x4x128x128xf16>

  //CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[DYN_DEQUANT]]) {order_value = #NCWH} : tensor<1x4x128x128xf16> -> tensor<1x4x128x128xf16>

  //CHECK: [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
  //CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [512, 128, 1, 1]} : tensor<1x4x128x128xf16> -> tensor<512x128x1x1xf16>

  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[AFFINE_RESHAPE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<512x128x1x1xf16> -> tensor<256x512x1x1xf16>
  //CHECK: return [[CONV]] : tensor<256x512x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @PropagateDynamicDequantAffineReshape
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<256x128x1x1xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x4x128x128x!qElemType>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<1x4x128x1xf16>
func.func @PropagateDynamicDequantAffineReshape(%arg0: tensor<256x128x1x1xf16>, %arg1: tensor<1x4x128x128x!qElemType>, %arg2: tensor<1x4x128x1xf16>) -> tensor<256x512x1x1xf16> {
  %1 = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<1x4x128x128x!qElemType>, tensor<1x4x128x1xf16> -> tensor<1x4x128x128xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [512, 128, 1, 1]} : tensor<1x4x128x128xf16> -> tensor<512x128x1x1xf16>
  %3 = IE.Convolution(%arg0, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<512x128x1x1xf16> -> tensor<256x512x1x1xf16>
  return %3 : tensor<256x512x1x1xf16>


  //CHECK: [[AFFINE_RESHAPE_SCALE_1:%.+]] = IE.AffineReshape([[ARG_2]])
  //CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [512, 1, 1, 1]} : tensor<1x4x128x1xf16> -> tensor<512x1x1x1xf16>

  //CHECK: [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[ARG_1]])
  //CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [512, 128, 1, 1]} : tensor<1x4x128x128x!qElemType> -> tensor<512x128x1x1x!qElemType>

  //CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[AFFINE_RESHAPE_1]], [[AFFINE_RESHAPE_SCALE_1]]) {dstElemType = f16} : tensor<512x128x1x1x!qElemType>, tensor<512x1x1x1xf16> -> tensor<512x128x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[DYN_DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<512x128x1x1xf16> -> tensor<256x512x1x1xf16>
  //CHECK: return [[CONV]] : tensor<256x512x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @PropagateDynamicDequantAffineReshapeStaticScale
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<256x128x1x1xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x4x128x128x!qElemType>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<1x1x1x1xf16>
func.func @PropagateDynamicDequantAffineReshapeStaticScale(%arg0: tensor<256x128x1x1xf16>, %arg1: tensor<1x4x128x128x!qElemType>, %arg2: tensor<1x1x1x1xf16>) -> tensor<256x512x1x1xf16> {
  %1 = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<1x4x128x128x!qElemType>, tensor<1x1x1x1xf16> -> tensor<1x4x128x128xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [512, 128, 1, 1]} : tensor<1x4x128x128xf16> -> tensor<512x128x1x1xf16>
  %3 = IE.Convolution(%arg0, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<512x128x1x1xf16> -> tensor<256x512x1x1xf16>
  return %3 : tensor<256x512x1x1xf16>

  //CHECK: [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[ARG_1]])
  //CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [512, 128, 1, 1]} : tensor<1x4x128x128x!qElemType> -> tensor<512x128x1x1x!qElemType>

  //CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[AFFINE_RESHAPE_1]], [[ARG_2]]) {dstElemType = f16} : tensor<512x128x1x1x!qElemType>, tensor<1x1x1x1xf16> -> tensor<512x128x1x1xf16>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[DYN_DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<512x128x1x1xf16> -> tensor<256x512x1x1xf16>
  //CHECK: return [[CONV]] : tensor<256x512x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @DoNotPropagateDynamicDequantZP
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<256x128x1x1xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x4x128x128x!qElemType>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<1x1x128x1xf16>
// CHECK-SAME:      [[ARG_3:%[^:]+]]: tensor<1x1x128x1xui2>
func.func @DoNotPropagateDynamicDequantZP(%arg0: tensor<256x128x1x1xf16>, %arg1: tensor<1x4x128x128x!qElemType>, %arg2: tensor<1x1x128x1xf16>, %arg3: tensor<1x1x128x1xui2>) -> tensor<256x512x1x1xf16> {
  %0 = IE.DynamicDequantize(%arg1, %arg2, %arg3) {dstElemType = f16} : tensor<1x4x128x128x!qElemType>, tensor<1x1x128x1xf16>, tensor<1x1x128x1xui2> -> tensor<1x4x128x128xf16>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [512, 128, 1, 1]} : tensor<1x4x128x128xf16> -> tensor<512x128x1x1xf16>
  %2 = IE.Convolution(%arg0, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<512x128x1x1xf16> -> tensor<256x512x1x1xf16>
  return %2 : tensor<256x512x1x1xf16>

  //CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[ARG_1]], [[ARG_2]], [[ARG_3]]) {dstElemType = f16} : tensor<1x4x128x128x!qElemType>, tensor<1x1x128x1xf16>, tensor<1x1x128x1xui2> -> tensor<1x4x128x128xf16>

  //CHECK: [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[DYN_DEQUANT]])
  //CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [512, 128, 1, 1]} : tensor<1x4x128x128xf16> -> tensor<512x128x1x1xf16>

  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[AFFINE_RESHAPE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<512x128x1x1xf16> -> tensor<256x512x1x1xf16>
  //CHECK: return [[CONV]] : tensor<256x512x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0>

// Propagation is blocked because the scale dim 2 (size 1) cannot be split
// using the original sub-dim ratio tail=3 (origOutputShape[2]=3): 1 % 3 != 0.
//
// Scale path would require: inDim2==1 split into [2,3] → 1 % 3 != 0 → std::nullopt → abort propagation.

// CHECK-LABEL: @DoNotPropagateDynamicDequantReshapeSplitNotDivisibleOnScale
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x2x5x1xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x6x6x1x!qElemType>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<1x6x1x1xf16>
func.func @DoNotPropagateDynamicDequantReshapeSplitNotDivisibleOnScale(
        %arg0: tensor<1x2x5x1xf16>,
        %arg1: tensor<1x6x6x1x!qElemType>,
        %arg2: tensor<1x6x1x1xf16>) -> tensor<1x6x3x1xf16> {
  %0 = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<1x6x6x1x!qElemType>, tensor<1x6x1x1xf16> -> tensor<1x6x6x1xf16>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [6, 2, 3, 1]} : tensor<1x6x6x1xf16> -> tensor<6x2x3x1xf16>
  %2 = IE.Convolution(%arg0, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2x5x1xf16>, tensor<6x2x3x1xf16> -> tensor<1x6x3x1xf16>
  return %2 : tensor<1x6x3x1xf16>

  // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[ARG_1]], [[ARG_2]]) {dstElemType = f16} : tensor<1x6x6x1x!qElemType>, tensor<1x6x1x1xf16> -> tensor<1x6x6x1xf16>
  // CHECK: [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[DYN_DEQUANT]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [6, 2, 3, 1]} : tensor<1x6x6x1xf16> -> tensor<6x2x3x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[AFFINE_RESHAPE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2x5x1xf16>, tensor<6x2x3x1xf16> -> tensor<1x6x3x1xf16>
  // CHECK: return [[CONV]] : tensor<1x6x3x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0>

// The AffineReshapeOp uses dim_mapping [[0],[0],[1],[1,2,3]]: inDims 0+1 merge into outDim 0,
// while inDim 2 (4 groups) and inDim 3 (group_size=16) both map to outDim 1.
// When computing the new scale shape the merge contribution from inDim 2 (value 4) is
// overwritten by the split path for inDim 3, yielding [8,1,1,1] (8 elements) instead of
// [8,4,1,1] (32 elements).  The element-count mismatch reveals that the reshape would fold
// group dimensions into the output channel, which is invalid for per-group DynamicDequantize.
// Propagation must be blocked.

// CHECK-LABEL: @DoNotPropagateDynamicDequantAffineReshapeMergesGroupDims
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x64x1x1xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x8x4x16x!qElemType>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<1x8x4x1xf16>
func.func @DoNotPropagateDynamicDequantAffineReshapeMergesGroupDims(
        %arg0: tensor<1x64x1x1xf16>,
        %arg1: tensor<1x8x4x16x!qElemType>,
        %arg2: tensor<1x8x4x1xf16>) -> tensor<1x8x1x1xf16> {
  %0 = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<1x8x4x16x!qElemType>, tensor<1x8x4x1xf16> -> tensor<1x8x4x16xf16>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [8, 64, 1, 1]} : tensor<1x8x4x16xf16> -> tensor<8x64x1x1xf16>
  %2 = IE.Convolution(%arg0, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<8x64x1x1xf16> -> tensor<1x8x1x1xf16>
  return %2 : tensor<1x8x1x1xf16>

  // Propagation is blocked: DynamicDequantize and AffineReshape are unchanged.
  // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[ARG_1]], [[ARG_2]]) {dstElemType = f16} : tensor<1x8x4x16x!qElemType>, tensor<1x8x4x1xf16> -> tensor<1x8x4x16xf16>
  // CHECK: [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[DYN_DEQUANT]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [8, 64, 1, 1]} : tensor<1x8x4x16xf16> -> tensor<8x64x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[AFFINE_RESHAPE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<8x64x1x1xf16> -> tensor<1x8x1x1xf16>
  // CHECK: return [[CONV]] : tensor<1x8x1x1xf16>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
!qElemType = !quant.uniform<u16:f16, 1.822021157242954E-4:32337>

// CHECK-LABEL: @DoNotSplitU16QuantizeAcrossConcatBeforeTranspose
// CHECK-SAME:     ([[ARG_0:%.+]]: tensor<1x1x64x3072xf16>, [[ARG_1:%.+]]: tensor<1x1x64x3072xf16>)
func.func @DoNotSplitU16QuantizeAcrossConcatBeforeTranspose(
        %arg0: tensor<1x1x64x3072xf16>,
        %arg1: tensor<1x1x64x3072xf16>) -> tensor<1x6144x1x64x!qElemType> {
  %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 3072]]} : tensor<1x1x64x3072xf16>, tensor<1x1x64x3072xf16> -> tensor<1x1x64x6144xf16>
  %1 = IE.Transpose(%0) {order_value = #NWCH} : tensor<1x1x64x6144xf16> -> tensor<1x6144x1x64xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x6144x1x64xf16> -> tensor<1x6144x1x64x!qElemType>
  return %2 : tensor<1x6144x1x64x!qElemType>

  // CHECK: [[CONCAT:%.+]] = IE.Concat([[ARG_0]], [[ARG_1]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 3072]]} : tensor<1x1x64x3072xf16>, tensor<1x1x64x3072xf16> -> tensor<1x1x64x6144xf16>
  // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[CONCAT]]) {dstElemType = !qElemType} : tensor<1x1x64x6144xf16> -> tensor<1x1x64x6144x!qElemType>
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[QUANTIZE]]) {order_value = #NWCH} : tensor<1x1x64x6144x!qElemType> -> tensor<1x6144x1x64x!qElemType>
  // CHECK: return [[TRANSPOSE]] : tensor<1x6144x1x64x!qElemType>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
!qElemType = !quant.uniform<u16:f16, 1.822021157242954E-4:32337>

// One operand traces back to a matching Dequantize, so splitting is beneficial here.
// CHECK-LABEL: @SplitU16QuantizeAcrossConcatWhenDequantizeReachable
// CHECK-SAME:     ([[ARG_0:%.+]]: tensor<1x1x64x3072x!qElemType>, [[ARG_1:%.+]]: tensor<1x1x64x3072xf16>)
func.func @SplitU16QuantizeAcrossConcatWhenDequantizeReachable(
        %arg0: tensor<1x1x64x3072x!qElemType>,
        %arg1: tensor<1x1x64x3072xf16>) -> tensor<1x6144x1x64x!qElemType> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x64x3072x!qElemType> -> tensor<1x1x64x3072xf16>
  %1 = IE.Concat(%0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 3072]]} : tensor<1x1x64x3072xf16>, tensor<1x1x64x3072xf16> -> tensor<1x1x64x6144xf16>
  %2 = IE.Transpose(%1) {order_value = #NWCH} : tensor<1x1x64x6144xf16> -> tensor<1x6144x1x64xf16>
  %3 = IE.Quantize(%2) {dstElemType = !qElemType} : tensor<1x6144x1x64xf16> -> tensor<1x6144x1x64x!qElemType>
  return %3 : tensor<1x6144x1x64x!qElemType>

  // Dequantize -> Quantize pair folds away, so Concat uses [[ARG_0]] directly.
  // CHECK-NOT: IE.Dequantize
  // CHECK: [[QUANTIZE_1:%.+]] = IE.Quantize([[ARG_1]]) {dstElemType = !qElemType} : tensor<1x1x64x3072xf16> -> tensor<1x1x64x3072x!qElemType>
  // CHECK: [[CONCAT:%.+]] = IE.Concat([[ARG_0]], [[QUANTIZE_1]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 3072]]} : tensor<1x1x64x3072x!qElemType>, tensor<1x1x64x3072x!qElemType> -> tensor<1x1x64x6144x!qElemType>
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[CONCAT]]) {order_value = #NWCH} : tensor<1x1x64x6144x!qElemType> -> tensor<1x6144x1x64x!qElemType>
  // CHECK: return [[TRANSPOSE]] : tensor<1x6144x1x64x!qElemType>
}

// -----

!qElemType = !quant.uniform<u16:f16, 1.822021157242954E-4:32337>

// Only 1 of 5 Concat operands traces back to a matching Dequantize (a minority), so splitting
// Quantize across all 5 operands would add 4 dangling, non-fusable Quantize ops for only 1 op
// saved (the folded Dequantize/Quantize pair). Not beneficial: keep the single shared Quantize.
// CHECK-LABEL: @DoNotSplitU16QuantizeAcrossConcatWhenMinorityHaveDequantize
// CHECK-SAME:     ([[ARG_0:%.+]]: tensor<1x192x40x40xf16>, [[ARG_1:%.+]]: tensor<1x192x40x40x!qElemType>, [[ARG_2:%.+]]: tensor<1x192x40x40xf16>, [[ARG_3:%.+]]: tensor<1x192x40x40xf16>, [[ARG_4:%.+]]: tensor<1x192x40x40xf16>)
func.func @DoNotSplitU16QuantizeAcrossConcatWhenMinorityHaveDequantize(
        %arg0: tensor<1x192x40x40xf16>,
        %arg1: tensor<1x192x40x40x!qElemType>,
        %arg2: tensor<1x192x40x40xf16>,
        %arg3: tensor<1x192x40x40xf16>,
        %arg4: tensor<1x192x40x40xf16>) -> tensor<1x960x40x40x!qElemType> {
  %0 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x192x40x40x!qElemType> -> tensor<1x192x40x40xf16>
  %1 = IE.Concat(%arg0, %0, %arg2, %arg3, %arg4) {
      static_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0], [0, 768, 0, 0]]
  } : tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16> -> tensor<1x960x40x40xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x960x40x40xf16> -> tensor<1x960x40x40x!qElemType>
  return %2 : tensor<1x960x40x40x!qElemType>

  // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[ARG_1]]) {dstElemType = f16} : tensor<1x192x40x40x!qElemType> -> tensor<1x192x40x40xf16>
  // CHECK: [[CONCAT:%.+]] = IE.Concat([[ARG_0]], [[DEQUANTIZE]], [[ARG_2]], [[ARG_3]], [[ARG_4]])
  // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[CONCAT]]) {dstElemType = !qElemType} : tensor<1x960x40x40xf16> -> tensor<1x960x40x40x!qElemType>
  // CHECK: return [[QUANTIZE]] : tensor<1x960x40x40x!qElemType>
}

// -----

!qElemType = !quant.uniform<u16:f16, 1.822021157242954E-4:32337>

// A majority (3 of 5) Concat operands trace back to a matching Dequantize, so splitting pays off:
// 3 Dequantize/Quantize pairs fold away and only 2 new, unfused Quantize ops are added.
// CHECK-LABEL: @SplitU16QuantizeAcrossConcatWhenMajorityHaveDequantize
// CHECK-SAME:     ([[ARG_0:%.+]]: tensor<1x192x40x40x!qElemType>, [[ARG_1:%.+]]: tensor<1x192x40x40xf16>, [[ARG_2:%.+]]: tensor<1x192x40x40x!qElemType>, [[ARG_3:%.+]]: tensor<1x192x40x40xf16>, [[ARG_4:%.+]]: tensor<1x192x40x40x!qElemType>)
func.func @SplitU16QuantizeAcrossConcatWhenMajorityHaveDequantize(
        %arg0: tensor<1x192x40x40x!qElemType>,
        %arg1: tensor<1x192x40x40xf16>,
        %arg2: tensor<1x192x40x40x!qElemType>,
        %arg3: tensor<1x192x40x40xf16>,
        %arg4: tensor<1x192x40x40x!qElemType>) -> tensor<1x960x40x40x!qElemType> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x192x40x40x!qElemType> -> tensor<1x192x40x40xf16>
  %1 = IE.Dequantize(%arg2) {dstElemType = f16} : tensor<1x192x40x40x!qElemType> -> tensor<1x192x40x40xf16>
  %2 = IE.Dequantize(%arg4) {dstElemType = f16} : tensor<1x192x40x40x!qElemType> -> tensor<1x192x40x40xf16>
  %3 = IE.Concat(%0, %arg1, %1, %arg3, %2) {
      static_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0], [0, 768, 0, 0]]
  } : tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16> -> tensor<1x960x40x40xf16>
  %4 = IE.Quantize(%3) {dstElemType = !qElemType} : tensor<1x960x40x40xf16> -> tensor<1x960x40x40x!qElemType>
  return %4 : tensor<1x960x40x40x!qElemType>

  // CHECK-NOT: IE.Dequantize
  // CHECK: [[QUANTIZE_1:%.+]] = IE.Quantize([[ARG_1]]) {dstElemType = !qElemType} : tensor<1x192x40x40xf16> -> tensor<1x192x40x40x!qElemType>
  // CHECK: [[QUANTIZE_3:%.+]] = IE.Quantize([[ARG_3]]) {dstElemType = !qElemType} : tensor<1x192x40x40xf16> -> tensor<1x192x40x40x!qElemType>
  // CHECK: [[CONCAT:%.+]] = IE.Concat([[ARG_0]], [[QUANTIZE_1]], [[ARG_2]], [[QUANTIZE_3]], [[ARG_4]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0], [0, 768, 0, 0]]} : tensor<1x192x40x40x!qElemType>, tensor<1x192x40x40x!qElemType>, tensor<1x192x40x40x!qElemType>, tensor<1x192x40x40x!qElemType>, tensor<1x192x40x40x!qElemType> -> tensor<1x960x40x40x!qElemType>
  // CHECK: return [[CONCAT]] : tensor<1x960x40x40x!qElemType>
}

// -----

!qElemType = !quant.uniform<u16:f16, 1.822021157242954E-4:32337>

// CHECK-LABEL: @DoNotSplitU16QuantizeAcrossConcatWhenDequantizeHasOtherUses
// CHECK-SAME:     ([[ARG_0:%.+]]: tensor<1x192x40x40x!qElemType>, [[ARG_1:%.+]]: tensor<1x192x40x40xf16>)
func.func @DoNotSplitU16QuantizeAcrossConcatWhenDequantizeHasOtherUses(
        %arg0: tensor<1x192x40x40x!qElemType>,
        %arg1: tensor<1x192x40x40xf16>) -> (tensor<1x384x40x40x!qElemType>, tensor<1x192x40x40xf16>) {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x192x40x40x!qElemType> -> tensor<1x192x40x40xf16>
  %1 = IE.Concat(%0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 192, 0, 0]]} : tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16> -> tensor<1x384x40x40xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x384x40x40xf16> -> tensor<1x384x40x40x!qElemType>
  return %2, %0 : tensor<1x384x40x40x!qElemType>, tensor<1x192x40x40xf16>

  // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[ARG_0]]) {dstElemType = f16} : tensor<1x192x40x40x!qElemType> -> tensor<1x192x40x40xf16>
  // CHECK: [[CONCAT:%.+]] = IE.Concat([[DEQUANTIZE]], [[ARG_1]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 192, 0, 0]]} : tensor<1x192x40x40xf16>, tensor<1x192x40x40xf16> -> tensor<1x384x40x40xf16>
  // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[CONCAT]]) {dstElemType = !qElemType} : tensor<1x384x40x40xf16> -> tensor<1x384x40x40x!qElemType>
  // CHECK: return [[QUANTIZE]], [[DEQUANTIZE]] : tensor<1x384x40x40x!qElemType>, tensor<1x192x40x40xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.0>

// DynamicDequantize (scale-only, no zp) feeds a chain of
// AffineReshape -> Slice -> AffineReshape before the Convolution. The scale must be propagated
// through all 3 hops, chaining each cloned scale-path op onto the previous one's output (not onto
// the original un-reshaped scale), so the final DynamicDequantize's scale/input ranks stay in sync.

// CHECK-LABEL: @PropagateDynamicDequantAffineReshapeSliceAffineReshape
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x5120x1x1xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x35840x1x5120x!qElemType>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<1x35840x1x1xf16>
func.func @PropagateDynamicDequantAffineReshapeSliceAffineReshape(%arg0: tensor<1x5120x1x1xf16>, %arg1: tensor<1x35840x1x5120x!qElemType>, %arg2: tensor<1x35840x1x1xf16>) -> tensor<1x17920x1x1xf16> {
  %0 = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<1x35840x1x5120x!qElemType>, tensor<1x35840x1x1xf16> -> tensor<1x35840x1x5120xf16>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1]], shape_value = [35840, 5120]} : tensor<1x35840x1x5120xf16> -> tensor<35840x5120xf16>
  %2 = IE.Slice %1 [0, 0] [17920, 5120] : tensor<35840x5120xf16> to tensor<17920x5120xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1, 2, 3]], shape_value = [17920, 5120, 1, 1]} : tensor<17920x5120xf16> -> tensor<17920x5120x1x1xf16>
  %4 = IE.Convolution(%arg0, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x5120x1x1xf16>, tensor<17920x5120x1x1xf16> -> tensor<1x17920x1x1xf16>
  return %4 : tensor<1x17920x1x1xf16>

  // CHECK: [[AFFINE_RESHAPE_SCALE_1:%.+]] = IE.AffineReshape([[ARG_2]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [35840, 1]} : tensor<1x35840x1x1xf16> -> tensor<35840x1xf16>
  // CHECK: [[SLICE_SCALE:%.+]] = IE.Slice [[AFFINE_RESHAPE_SCALE_1]] [0, 0] [17920, 1] : tensor<35840x1xf16> to tensor<17920x1xf16>
  // CHECK: [[AFFINE_RESHAPE_SCALE_2:%.+]] = IE.AffineReshape([[SLICE_SCALE]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [17920, 1, 1, 1]} : tensor<17920x1xf16> -> tensor<17920x1x1x1xf16>

  // CHECK: [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[ARG_1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [35840, 5120]} : tensor<1x35840x1x5120x!qElemType> -> tensor<35840x5120x!qElemType>
  // CHECK: [[SLICE:%.+]] = IE.Slice [[AFFINE_RESHAPE_1]] [0, 0] [17920, 5120] : tensor<35840x5120x!qElemType> to tensor<17920x5120x!qElemType>
  // CHECK: [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[SLICE]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [17920, 5120, 1, 1]} : tensor<17920x5120x!qElemType> -> tensor<17920x5120x1x1x!qElemType>

  // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[AFFINE_RESHAPE_2]], [[AFFINE_RESHAPE_SCALE_2]]) {dstElemType = f16} : tensor<17920x5120x1x1x!qElemType>, tensor<17920x1x1x1xf16> -> tensor<17920x5120x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[DYN_DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x5120x1x1xf16>, tensor<17920x5120x1x1xf16> -> tensor<1x17920x1x1xf16>
  // CHECK: return [[CONV]] : tensor<1x17920x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// PermuteCastOp swaps dims 1 and 2 (the group-size dim, 4, and a size-1 spacer dim), which
// is a trivial (view-like) permutation since one of the swapped dims has extent 1: the
// group-size dim is never reordered relative to the num-groups dim (dim3=128, matching the
// scale's only non-unit dim), so no data movement is required and the scale (which is 1 at
// every position except the num-groups dim) is unaffected by the swap. AffineReshape then
// merges the group-size dim into the output-channel dim and splits the num-groups dim into
// the input-channel dim. Both hops should be recognized as valid to propagate through,
// moving DynamicDequantize right before the Convolution.

// CHECK-LABEL: @PropagateDynamicDequantPermuteCastAffineReshape
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<256x128x1x1xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x4x1x128x!qElemType>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<1x1x1x128xf16>
func.func @PropagateDynamicDequantPermuteCastAffineReshape(
        %arg0: tensor<256x128x1x1xf16>,
        %arg1: tensor<1x4x1x128x!qElemType>,
        %arg2: tensor<1x1x1x128xf16>) -> tensor<256x4x1x1xf16> {
  %0 = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<1x4x1x128x!qElemType>, tensor<1x1x1x128xf16> -> tensor<1x4x1x128xf16>
  %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x4x1x128xf16> -> tensor<1x1x4x128xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 128, 1, 1]} : tensor<1x1x4x128xf16> -> tensor<4x128x1x1xf16>
  %3 = IE.Convolution(%arg0, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<4x128x1x1xf16> -> tensor<256x4x1x1xf16>
  return %3 : tensor<256x4x1x1xf16>

  // CHECK: [[PERMUTECAST_SCALE:%.+]] = IE.PermuteCast([[ARG_2]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x1x1x128xf16> -> tensor<1x1x1x128xf16>
  // CHECK: [[AFFINE_RESHAPE_SCALE:%.+]] = IE.AffineReshape([[PERMUTECAST_SCALE]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 128, 1, 1]} : tensor<1x1x1x128xf16> -> tensor<1x128x1x1xf16>

  // CHECK: [[PERMUTECAST:%.+]] = IE.PermuteCast([[ARG_1]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x4x1x128x!qElemType> -> tensor<1x1x4x128x!qElemType>
  // CHECK: [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[PERMUTECAST]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 128, 1, 1]} : tensor<1x1x4x128x!qElemType> -> tensor<4x128x1x1x!qElemType>

  // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[AFFINE_RESHAPE]], [[AFFINE_RESHAPE_SCALE]]) {dstElemType = f16} : tensor<4x128x1x1x!qElemType>, tensor<1x128x1x1xf16> -> tensor<4x128x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[DYN_DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<4x128x1x1xf16> -> tensor<256x4x1x1xf16>
  // CHECK: return [[CONV]] : tensor<256x4x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0>

// IE.ShapeCastOp only carries the target activation shape (a plain I64ArrayAttr), with no
// per-dim mapping between input and output dims like AffineReshapeOp's dim_mapping. There is
// therefore no way to derive how the scale's own (much smaller, per-group) shape should be
// reshaped alongside the quantized activation, so computeNextScaleShape must reject this hop
// and propagation has to stop before ShapeCastOp.

// CHECK-LABEL: @DoNotPropagateDynamicDequantShapeCast
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x64x1x1xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x8x4x16x!qElemType>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<1x8x4x1xf16>
func.func @DoNotPropagateDynamicDequantShapeCast(
        %arg0: tensor<1x64x1x1xf16>,
        %arg1: tensor<1x8x4x16x!qElemType>,
        %arg2: tensor<1x8x4x1xf16>) -> tensor<1x8x1x1xf16> {
  %0 = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<1x8x4x16x!qElemType>, tensor<1x8x4x1xf16> -> tensor<1x8x4x16xf16>
  %1 = IE.ShapeCast {shape = [8, 64, 1, 1]} inputs(%0 : tensor<1x8x4x16xf16>) -> tensor<8x64x1x1xf16>
  %2 = IE.Convolution(%arg0, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<8x64x1x1xf16> -> tensor<1x8x1x1xf16>
  return %2 : tensor<1x8x1x1xf16>

  // Propagation is blocked: DynamicDequantize and ShapeCast are unchanged.
  // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[ARG_1]], [[ARG_2]]) {dstElemType = f16} : tensor<1x8x4x16x!qElemType>, tensor<1x8x4x1xf16> -> tensor<1x8x4x16xf16>
  // CHECK: [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [8, 64, 1, 1]} inputs([[DYN_DEQUANT]] : tensor<1x8x4x16xf16>) -> tensor<8x64x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[SHAPE_CAST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<8x64x1x1xf16> -> tensor<1x8x1x1xf16>
  // CHECK: return [[CONV]] : tensor<1x8x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// DynamicDequantize's result feeds two direct users (PermuteCast and AffineReshape), taking the
// "multiple users" branch of matchAndRewrite instead of the single-use hop-by-hop walk. The two
// branches use DIFFERENT hop ops so that any state leaking from one branch into the other
// (wrong shape or wrong already-cloned scale value) would make the sibling branch's own propagation
// validation/rewrite fail or produce a mismatched result.

// CHECK-LABEL: @PropagateDynamicDequantMultipleUsers
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x4x1x128x!qElemType>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x1x1x128xf16>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<256x128x1x1xf16>
// CHECK-SAME:      [[ARG_3:%[^:]+]]: tensor<512x128x1x1xf16>
func.func @PropagateDynamicDequantMultipleUsers(
        %arg0: tensor<1x4x1x128x!qElemType>,
        %arg1: tensor<1x1x1x128xf16>,
        %arg2: tensor<256x128x1x1xf16>,
        %arg3: tensor<512x128x1x1xf16>) -> (tensor<256x4x1x1xf16>, tensor<512x4x1x1xf16>) {
  %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x4x1x128x!qElemType>, tensor<1x1x1x128xf16> -> tensor<1x4x1x128xf16>
  %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x4x1x128xf16> -> tensor<1x1x4x128xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 128, 1, 1]} : tensor<1x1x4x128xf16> -> tensor<4x128x1x1xf16>
  %3 = IE.Convolution(%arg2, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<4x128x1x1xf16> -> tensor<256x4x1x1xf16>

  %4 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 128, 1, 1]} : tensor<1x4x1x128xf16> -> tensor<4x128x1x1xf16>
  %5 = IE.Convolution(%arg3, %4) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<512x128x1x1xf16>, tensor<4x128x1x1xf16> -> tensor<512x4x1x1xf16>

  return %3, %5 : tensor<256x4x1x1xf16>, tensor<512x4x1x1xf16>

  // Branch 1 (PermuteCast -> AffineReshape -> Convolution): its scale-path clones must start
  // from [[ARG_1]] (the original scale), not from anything left over by branch 2's processing.
  // CHECK: [[SCALE_PERMUTECAST:%.+]] = IE.PermuteCast([[ARG_1]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x1x1x128xf16> -> tensor<1x1x1x128xf16>

  // Branch 2 (AffineReshape -> Convolution): its scale-path clone must also start from
  // [[ARG_1]] directly, unaffected by branch 1's PermuteCast clone above.
  // CHECK: [[SCALE_AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[ARG_1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 128, 1, 1]} : tensor<1x1x1x128xf16> -> tensor<1x128x1x1xf16>

  // CHECK: [[ACT_PERMUTECAST:%.+]] = IE.PermuteCast([[ARG_0]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x4x1x128x!qElemType> -> tensor<1x1x4x128x!qElemType>

  // CHECK: [[SCALE_AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[SCALE_PERMUTECAST]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 128, 1, 1]} : tensor<1x1x1x128xf16> -> tensor<1x128x1x1xf16>
  // CHECK: [[ACT_AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[ACT_PERMUTECAST]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 128, 1, 1]} : tensor<1x1x4x128x!qElemType> -> tensor<4x128x1x1x!qElemType>
  // CHECK: [[DYN_DEQUANT_1:%.+]] = IE.DynamicDequantize([[ACT_AFFINE_RESHAPE_1]], [[SCALE_AFFINE_RESHAPE_1]]) {dstElemType = f16} : tensor<4x128x1x1x!qElemType>, tensor<1x128x1x1xf16> -> tensor<4x128x1x1xf16>
  // CHECK: [[CONV_1:%.+]] = IE.Convolution([[ARG_2]], [[DYN_DEQUANT_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<4x128x1x1xf16> -> tensor<256x4x1x1xf16>

  // CHECK: [[ACT_AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[ARG_0]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 128, 1, 1]} : tensor<1x4x1x128x!qElemType> -> tensor<4x128x1x1x!qElemType>
  // CHECK: [[DYN_DEQUANT_2:%.+]] = IE.DynamicDequantize([[ACT_AFFINE_RESHAPE_2]], [[SCALE_AFFINE_RESHAPE_2]]) {dstElemType = f16} : tensor<4x128x1x1x!qElemType>, tensor<1x128x1x1xf16> -> tensor<4x128x1x1xf16>
  // CHECK: [[CONV_2:%.+]] = IE.Convolution([[ARG_3]], [[DYN_DEQUANT_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<512x128x1x1xf16>, tensor<4x128x1x1xf16> -> tensor<512x4x1x1xf16>

  // CHECK: return [[CONV_1]], [[CONV_2]] : tensor<256x4x1x1xf16>, tensor<512x4x1x1xf16>
}
