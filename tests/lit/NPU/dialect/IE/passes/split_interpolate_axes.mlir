//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-interpolate-axes --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @SplitInterpolateAxes
func.func @SplitInterpolateAxes(%arg0: tensor<1x2x2x2x2xf16>) -> tensor<1x2x4x4x4xf16> {
    %0 = IE.Reshape(%arg0) {shape_value = [1, 2, 4, 2]} : tensor<1x2x2x2x2xf16> -> tensor<1x2x4x2xf16>
    %1 = IE.Reshape(%arg0) {shape_value = [2, 2, 2, 2]} : tensor<1x2x2x2x2xf16> -> tensor<2x2x2x2xf16>
    %2 = IE.Interpolate(%1) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0], pads_end = [0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00, 2.000000e+00], sizes_attr = [4, 4, 4]} : tensor<2x2x2x2xf16> -> tensor<2x4x4x4xf16>
    %3 = IE.Reshape(%2) {shape_value = [1, 2, 4, 4, 4]} : tensor<2x4x4x4xf16> -> tensor<1x2x4x4x4xf16>
    return %3 : tensor<1x2x4x4x4xf16>


    // CHECK: [[INPUT_RESHAPE:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:           {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [2, 2, 2, 2]} : tensor<1x2x2x2x2xf16> -> tensor<2x2x2x2xf16>
    // CHECK: [[Interpolate:%.+]] = IE.Interpolate([[INPUT_RESHAPE]])
    // CHECK:   {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>,
    // CHECK-SAME:       antialias = false,
    // CHECK-SAME:       pads_begin = [0, 0, 0], pads_end = [0, 0, 0],
    // CHECK:            cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:       axes_attr = [1, 2],
    // CHECK:            operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK:            scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [4, 4]} :
    // CHECK:       tensor<2x2x2x2xf16> -> tensor<2x4x4x2xf16>
    // CHECK: [[Interpolate2:%.+]] = IE.Interpolate([[Interpolate]])
    // CHECK:   {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>,
    // CHECK-SAME:       antialias = false,
    // CHECK-SAME:       pads_begin = [0, 0, 0], pads_end = [0, 0, 0],
    // CHECK:            cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:       axes_attr = [3],
    // CHECK:            operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK:            scales_attr = [2.000000e+00], sizes_attr = [4]} :
    // CHECK:       tensor<2x4x4x2xf16> -> tensor<2x4x4x4xf16>

    // CHECK: [[OUTPUT_RESHAPE:%.+]] = IE.AffineReshape([[Interpolate2]])
    // CHECK-SAME{LITERAL}:            {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 2, 4, 4, 4]} : tensor<2x4x4x4xf16> -> tensor<1x2x4x4x4xf16>
    // CHECK: return [[OUTPUT_RESHAPE]] : tensor<1x2x4x4x4xf16>

}
