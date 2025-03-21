//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-unaligned-qdq-seq %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @OptimizeQuantDequantSequence
func.func @OptimizeQuantDequantSequence(%arg0 : tensor<1x40x1x1xf16>, %arg1 : tensor<512x40x1x1xf16>) -> tensor<1x64x1x8xf16> {
  %cst_0 = const.Declare tensor<f16> = dense<0.0> : tensor<f16>
  %cst_1 = const.Declare tensor<f16> = dense<1.0> : tensor<f16>
  %1 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x40x1x1xf16>, tensor<512x40x1x1xf16> -> tensor<1x512x1x1xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 512]} : tensor<1x512x1x1xf16> -> tensor<1x1x1x512xf16>
  %3 = IE.FakeQuantize(%2, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x512xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x1x1x512xf16>
  %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1, 8, 64]} : tensor<1x1x1x512xf16> -> tensor<1x1x8x64xf16>
  %5 = IE.Transpose(%4) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x1x8x64xf16> -> tensor<1x64x1x8xf16>
  return %5 : tensor<1x64x1x8xf16>

  // CHECK:  [[VAL1:%.*]] = IE.Convolution(%arg0, %arg1)
  // CHECK:  [[VAL2:%.*]] = IE.FakeQuantize([[VAL1]]
  // CHECK-SAME: -> tensor<1x512x1x1xf16>
  // CHECK:  [[VAL3:%.*]] = IE.AffineReshape([[VAL2]])
  // CHECK:  IE.AffineReshape([[VAL3]])
}

// CHECK-LABEL: @NoOptimizeQuantDequantSequence
func.func @NoOptimizeQuantDequantSequence(%arg0 : tensor<1xf16>) -> tensor<1x1x1x1xf16> {
  %cst_0 = const.Declare tensor<f16> = dense<0.0> : tensor<f16>
  %cst_1 = const.Declare tensor<f16> = dense<1.0> : tensor<f16>
  %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 1, 1, 1]} : tensor<1xf16> -> tensor<1x1x1x1xf16>
  %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x1x1x1xf16>
  return %2 : tensor<1x1x1x1xf16>

  // CHECK:  [[VAL1:%.*]] = IE.AffineReshape(%arg0)
  // CHECK-SAME: tensor<1xf16> -> tensor<1x1x1x1xf16>
  // CHECK:  [[VAL2:%.*]] = IE.FakeQuantize([[VAL1]]
  // CHECK-SAME: -> tensor<1x1x1x1xf16>
}
