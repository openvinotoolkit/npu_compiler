//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --split-se-ops="se-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateNearestNeedSplitWithLargeSize([[INPUT_DATA:%.+]]: tensor<1x16x160x160xf16, {order = #NHWC}>) -> tensor<1x16x640x640xf16, {order = #NHWC}> {
func.func @InterpolateNearestNeedSplitWithLargeSize(%arg0: tensor<1x16x160x160xf16, {order = #NHWC}>) -> tensor<1x16x640x640xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <PYTORCH_HALF_PIXEL>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <NEAREST>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [4.000000e+00, 4.000000e+00],
            sizes_attr = [640, 640],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x160x160xf16, {order = #NHWC}> -> tensor<1x16x640x640xf16, {order = #NHWC}>

    return %0 : tensor<1x16x640x640xf16, {order = #NHWC}>

    // CHECK:       [[INTERPOLATE_W:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK-SAME:          {attr = #IE.Interpolate<mode = <NEAREST>,
    // CHECK-SAME:           shape_calc_mode = <SCALES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
    // CHECK-SAME:           antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:           axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
    // CHECK-SAME:           scales_attr = [1.000000e+00, 4.000000e+00], sizes_attr = [640, 640]}
    // CHECK:           : tensor<1x16x160x160xf16, {order = #NHWC}> -> tensor<1x16x160x640xf16, {order = #NHWC}>
    // CHECK:       [[INTERPOLATE_H:%.+]] = VPU.Interpolate([[INTERPOLATE_W]])
    // CHECK-SAME:          {attr = #IE.Interpolate<mode = <NEAREST>,
    // CHECK-SAME:           shape_calc_mode = <SCALES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
    // CHECK-SAME:           antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:           axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
    // CHECK-SAME:           scales_attr = [4.000000e+00, 1.000000e+00], sizes_attr = [640, 640]}
    // CHECK:           : tensor<1x16x160x640xf16, {order = #NHWC}> -> tensor<1x16x640x640xf16, {order = #NHWC}>

    // CHECK:       return [[INTERPOLATE_H]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateNearestSizeModeNeedSplitWithLargeSize([[INPUT_DATA:%.+]]: tensor<1x16x160x160xf16, {order = #NHWC}>) -> tensor<1x16x640x640xf16, {order = #NHWC}> {
func.func @InterpolateNearestSizeModeNeedSplitWithLargeSize(%arg0: tensor<1x16x160x160xf16, {order = #NHWC}>) -> tensor<1x16x640x640xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <PYTORCH_HALF_PIXEL>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <NEAREST>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SIZES>>,
            axes_attr = [0, 1, 2, 3],
            scales_attr = [1.000000e+00,1.000000e+00,4.000000e+00, 4.000000e+00],
            sizes_attr = [1, 16, 640, 640],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x160x160xf16, {order = #NHWC}> -> tensor<1x16x640x640xf16, {order = #NHWC}>

    return %0 : tensor<1x16x640x640xf16, {order = #NHWC}>

    // CHECK:       [[INTERPOLATE_W:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK-SAME:          {attr = #IE.Interpolate<mode = <NEAREST>,
    // CHECK-SAME:           shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
    // CHECK-SAME:           antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:           axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
    // CHECK-SAME:           scales_attr = [1.000000e+00, 1.000000e+00, 4.000000e+00, 4.000000e+00], sizes_attr = [1, 16, 160, 640]}
    // CHECK:           : tensor<1x16x160x160xf16, {order = #NHWC}> -> tensor<1x16x160x640xf16, {order = #NHWC}>
    // CHECK:       [[INTERPOLATE_H:%.+]] = VPU.Interpolate([[INTERPOLATE_W]])
    // CHECK-SAME:          {attr = #IE.Interpolate<mode = <NEAREST>,
    // CHECK-SAME:           shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
    // CHECK-SAME:           antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:           axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
    // CHECK-SAME:           scales_attr = [1.000000e+00, 1.000000e+00, 4.000000e+00, 4.000000e+00], sizes_attr = [1, 16, 640, 640]}
    // CHECK:           : tensor<1x16x160x640xf16, {order = #NHWC}> -> tensor<1x16x640x640xf16, {order = #NHWC}>

    // CHECK:       return [[INTERPOLATE_H]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateBilinearNeedSplitWithLargeSize([[INPUT_DATA:%.+]]: tensor<1x16x160x160xf16, {order = #NHWC}>) -> tensor<1x16x320x320xf16, {order = #NHWC}> {
func.func @InterpolateBilinearNeedSplitWithLargeSize(%arg0: tensor<1x16x160x160xf16, {order = #NHWC}>) -> tensor<1x16x320x320xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <PYTORCH_HALF_PIXEL>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [0, 1, 2, 3],
            scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
            sizes_attr = [1, 16, 320, 320],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x160x160xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>

    return %0 : tensor<1x16x320x320xf16, {order = #NHWC}>

    // CHECK:       [[INTERPOLATE_W:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK-SAME:          {attr = #IE.Interpolate<mode = <LINEAR>,
    // CHECK-SAME:           shape_calc_mode = <SCALES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
    // CHECK-SAME:           antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:           axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
    // CHECK-SAME:           scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00], sizes_attr = [1, 16, 320, 320]}
    // CHECK:           : tensor<1x16x160x160xf16, {order = #NHWC}> -> tensor<1x16x160x320xf16, {order = #NHWC}>
    // CHECK:       [[INTERPOLATE_H:%.+]] = VPU.Interpolate([[INTERPOLATE_W]])
    // CHECK-SAME:          {attr = #IE.Interpolate<mode = <LINEAR>,
    // CHECK-SAME:           shape_calc_mode = <SCALES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
    // CHECK-SAME:           antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:           axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
    // CHECK-SAME:           scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00], sizes_attr = [1, 16, 320, 320]}
    // CHECK:           : tensor<1x16x160x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>

    // CHECK:       return [[INTERPOLATE_H]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateBilinearNotSplitWithSmallSize([[INPUT_DATA:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x32x32xf16, {order = #NHWC}> {
func.func @InterpolateBilinearNotSplitWithSmallSize(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x32x32xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <PYTORCH_HALF_PIXEL>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [32, 32],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    return %0 : tensor<1x16x32x32xf16, {order = #NHWC}>

    // CHECK:       [[INTERPOLATE_W:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK-SAME:          {attr = #IE.Interpolate<mode = <LINEAR>,
    // CHECK-SAME:           shape_calc_mode = <SCALES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
    // CHECK-SAME:           antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:           axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
    // CHECK-SAME:           scales_attr = [1.000000e+00, 2.000000e+00], sizes_attr = [32, 32]}
    // CHECK:           : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x32xf16, {order = #NHWC}>
    // CHECK:       [[INTERPOLATE_H:%.+]] = VPU.Interpolate([[INTERPOLATE_W]])
    // CHECK-SAME:          {attr = #IE.Interpolate<mode = <LINEAR>,
    // CHECK-SAME:           shape_calc_mode = <SCALES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
    // CHECK-SAME:           antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:           axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
    // CHECK-SAME:           scales_attr = [2.000000e+00, 1.000000e+00], sizes_attr = [32, 32]}
    // CHECK:           : tensor<1x16x16x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    // CHECK:       return [[INTERPOLATE_H]]
}
