//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-bilinear-to-strided-concat-and-conv --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @NotConvertIfChannelIs3For40XXPlus
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x3x80x80xf16>
func.func @NotConvertIfChannelIs3For40XXPlus(%arg0: tensor<1x3x80x80xf16>) -> tensor<1x3x160x160xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [160, 160]
         } : tensor<1x3x80x80xf16> -> tensor<1x3x160x160xf16>

    return %0 : tensor<1x3x160x160xf16>

    // CHECK:           [[INTERPOLATE:%.+]] = IE.Interpolate([[INPUT]]) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [160, 160]} : tensor<1x3x80x80xf16> -> tensor<1x3x160x160xf16>
    // CHECK:           return [[INTERPOLATE]] : tensor<1x3x160x160xf16>
}

// -----

// CHECK-LABEL: @ConvertIfChannelIs3ButNotExactly2xUpscale
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x3x80x80xf16>
func.func @ConvertIfChannelIs3ButNotExactly2xUpscale(%arg0: tensor<1x3x80x80xf16>) -> tensor<1x3x240x240xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [3.000000e+00, 3.000000e+00], sizes_attr = [240, 240]
         } : tensor<1x3x80x80xf16> -> tensor<1x3x240x240xf16>

    return %0 : tensor<1x3x240x240xf16>

    // CHECK-NOT:       IE.Interpolate
    // CHECK:           [[INPUTREORDER:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NHWC} : tensor<1x3x80x80xf16> -> tensor<1x3x80x80xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @ConvertIfDifferentChannelCount
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x80x80xf16>
func.func @ConvertIfDifferentChannelCount(%arg0: tensor<1x4x80x80xf16>) -> tensor<1x4x160x160xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [160, 160]
         } : tensor<1x4x80x80xf16> -> tensor<1x4x160x160xf16>

    return %0 : tensor<1x4x160x160xf16>

    // CHECK-NOT:       IE.Interpolate
    // CHECK:           [[INPUTREORDER:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NHWC} : tensor<1x4x80x80xf16> -> tensor<1x4x80x80xf16, {order = #NHWC}>
}
