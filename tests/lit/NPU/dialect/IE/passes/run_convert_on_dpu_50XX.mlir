//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --run-f16-to-f32-convert-on-dpu %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FoldConvertIntoInterpolate
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x16x512x1024xf16, {order = #NHWC}>)
func.func @FoldConvertIntoInterpolate(%arg0: tensor<1x16x512x1024xf16, {order = #NHWC}>) -> tensor<1x16x1024x2048xf32, {order = #NHWC}> {
  %0 = IE.Interpolate(%arg0)
      {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
      operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 352]
      } : tensor<1x16x512x1024xf16, {order = #NHWC}> -> tensor<1x16x1024x2048xf16, {order = #NHWC}>
  %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x16x1024x2048xf16, {order = #NHWC}> -> tensor<1x16x1024x2048xf32, {order = #NHWC}>
  return %1 : tensor<1x16x1024x2048xf32, {order = #NHWC}>

  // CHECK: [[INTERP:%.+]] = IE.Interpolate([[INPUT]])
  // CHECK-SAME:    : tensor<1x16x512x1024xf16, {order = #NHWC}> -> tensor<1x16x1024x2048xf32, {order = #NHWC}>
  // CHECK-NOT:   IE.Convert
  // CHECK:       return [[INTERP]]
}
