//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --run-initial-low-precision-transformations-rewriters="enable-dynamic-quantization-for-static-case" --initial-low-precision-transformations %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU5010

// CHECK-LABEL: @UI2WeightsDequantizeToDynamicDequantize
func.func @UI2WeightsDequantizeToDynamicDequantize(%arg0: tensor<1x1x8192xf16>) -> tensor<1x1x3072xf32> {
  %zero_point = const.Declare tensor<3072x128x1xf32> = dense<1> : tensor<3072x128x1xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f32>]
  %weights = const.Declare tensor<3072x128x64xf32> = dense<0> : tensor<3072x128x64xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f32>]
  %scale = const.Declare tensor<3072x128x1xf32> = dense<2.0> : tensor<3072x128x1xf32>
  %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x1x8192xf16> -> tensor<1x1x8192xf32>
  %1 = IE.Subtract(%weights, %zero_point) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3072x128x64xf32>, tensor<3072x128x1xf32> -> tensor<3072x128x64xf32>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3072x128x64xf32>, tensor<3072x128x1xf32> -> tensor<3072x128x64xf32>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [3072, 8192]} : tensor<3072x128x64xf32> -> tensor<3072x8192xf32>
  %4 = IE.Reshape(%0) {shape_value = [1, 8192]} : tensor<1x1x8192xf32> -> tensor<1x8192xf32>
  %5 = IE.FullyConnected(%4, %3) : tensor<1x8192xf32>, tensor<3072x8192xf32> -> tensor<1x3072xf32>
  %6 = IE.Reshape(%5) {shape_value = [1, 1, 3072]} : tensor<1x3072xf32> -> tensor<1x1x3072xf32>
  return %6 : tensor<1x1x3072xf32>

  // CHECK-DAG:  [[CST_SCALE_RESHAPED:%.+]] = const.Declare tensor<3072x128xf32> = dense<2.000000e+00> : tensor<3072x128x1xf32>, [#const.Reshape<[3072, 128]>]
  // CHECK-DAG:  [[CST_SCALE:%.+]] = const.Declare tensor<3072x128x1xf32> = dense<2.000000e+00> : tensor<3072x128x1xf32>
  // CHECK-DAG:  [[CST_WEIGHTS:%.+]] = const.Declare tensor<3072x128x64x!qElemType> = dense<0> : tensor<3072x128x64xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f32>, #const.CastElemType<!qElemType>]

  // DynamicDequantizeOp
  // CHECK:      IE.DynamicDequantize([[CST_WEIGHTS]], [[CST_SCALE]]) {dstElemType = f32} : tensor<3072x128x64x!qElemType>, tensor<3072x128x1xf32> -> tensor<3072x128x64xf32>
  // CHECK-NEXT: IE.Convert
  // CHECK-NEXT: IE.AffineReshape
  // CHECK-NEXT: IE.Reshape
  // CHECK-NEXT: IE.FullyConnected
  // CHECK-NEXT: IE.Reshape
  // CHECK-NEXT: IE.ReduceSum
  // CHECK-NEXT: IE.FullyConnected
  // CHECK-NEXT: IE.Subtract
  // CHECK-NEXT: IE.Reshape
  // CHECK-NEXT: return {{[^:*]+}} : tensor<1x1x3072xf32>

}

// -----

// CHECK-LABEL: @UI2WeightsMultiplyToDynamicDequantize
func.func @UI2WeightsMultiplyToDynamicDequantize(%arg0: tensor<1x1x128xf16>) -> tensor<1x1x64xf32> {
  %weights = const.Declare tensor<64x16x8xf32> = dense<0> : tensor<64x16x8xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f32>]
  %scale = const.Declare tensor<64x16x1xf32> = dense<1.5> : tensor<64x16x1xf32>
  %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x1x128xf16> -> tensor<1x1x128xf32>
  %1 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<64x16x8xf32>, tensor<64x16x1xf32> -> tensor<64x16x8xf32>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1]], shape_value = [64, 128]} : tensor<64x16x8xf32> -> tensor<64x128xf32>
  %3 = IE.Reshape(%0) {shape_value = [1, 128]} : tensor<1x1x128xf32> -> tensor<1x128xf32>
  %4 = IE.FullyConnected(%3, %2) : tensor<1x128xf32>, tensor<64x128xf32> -> tensor<1x64xf32>
  %5 = IE.Reshape(%4) {shape_value = [1, 1, 64]} : tensor<1x64xf32> -> tensor<1x1x64xf32>
  return %5 : tensor<1x1x64xf32>

  // CHECK-DAG:  [[CST_WEIGHTS:%.+]] = const.Declare tensor<64x16x8x!qElemType> = dense<0> : tensor<64x16x8xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f32>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:  [[CST_SCALE:%.+]] = const.Declare tensor<64x16x1xf32> = dense<1.500000e+00> : tensor<64x16x1xf32>

  // DynamicDequantizeOp
  // CHECK:      IE.DynamicDequantize([[CST_WEIGHTS]], [[CST_SCALE]]) {dstElemType = f32} : tensor<64x16x8x!qElemType>, tensor<64x16x1xf32> -> tensor<64x16x8xf32>
  // CHECK-NEXT: IE.Convert
  // CHECK-NEXT: IE.AffineReshape
  // CHECK-NEXT: IE.Reshape
  // CHECK-NEXT: IE.FullyConnected
  // CHECK-NEXT: IE.Reshape
  // CHECK-NEXT: return {{[^:*]+}} : tensor<1x1x64xf32>

}
