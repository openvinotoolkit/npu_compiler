//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-convert-with-sw-op %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NoSwapInterpolate
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x8x128x128xf16, {order = #NHWC}>)
func.func @NoSwapInterpolate(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2048x1x256xf32> {
  %0 = const.Declare tensor<4xsi64> = dense<[1, 8, 256, 256]> : tensor<4xsi64>
  %1 = const.Declare tensor<4xf32>  = dense<[1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>
  %2 = IE.Interpolate(%arg0, %0, %1) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, operandSegmentSizes = array<i32: 1, 1, 1, 0>} :
    tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<4xsi64>, tensor<4xf32> -> tensor<1x8x256x256xf16, {order = #NHWC}>
  %3 = IE.Reshape(%2) { shape_value = [1, 2048, 1, 256] } : tensor<1x8x256x256xf16, {order = #NHWC}> -> tensor<1x2048x1x256xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<1x2048x1x256xf16> -> tensor<1x2048x1x256xf32>
  return %4 : tensor<1x2048x1x256xf32>

  // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[INPUT]]
  // CHECK-SAME:    -> tensor<1x8x256x256xf16, {order = #NHWC}>
  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INTERPOLATE]])
  // CHECK:       [[RET:%.+]] = IE.Convert([[RESHAPE]]) {dstElemType = f32}
  // CHECK-SAME:    -> tensor<1x2048x1x256xf32>
  // CHECK:       return [[RET]]
}
