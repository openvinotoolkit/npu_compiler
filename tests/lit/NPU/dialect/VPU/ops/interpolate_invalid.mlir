//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @InterpolateDynamicInputOffsetsWrongLen(
        %input: tensor<1x32x64x64xf16, {order = #NHWC}>,
        %dyn_in_offsets: tensor<3xi64>) -> tensor<1x32x128x128xf16, {order = #NHWC}> {
    // expected-error@+1 {{'dynamic_input_offsets' must have 4 elements, got 3}}
    %0 = VPU.Interpolate(%input, %dyn_in_offsets) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
            mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0],
            pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        initial_input_dims_attr = [1, 32, 64, 64],
        initial_input_offset_attr = [0, 0, -9223372036854775808, 0],
        initial_output_dims_attr = [1, 32, 128, 128],
        initial_output_offset_attr = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 1, 0>,
        scales_attr = [2.0, 2.0],
        sizes_attr = [128, 128],
        tile_offset_attr = [0.0, 0.0, 0.0, 0.0]
    } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<3xi64> -> tensor<1x32x128x128xf16, {order = #NHWC}>
    return %0 : tensor<1x32x128x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @InterpolateDynamicInputOffsetsMissingSentinel(
        %input: tensor<1x32x64x64xf16, {order = #NHWC}>,
        %dyn_in_offsets: tensor<4xi64>) -> tensor<1x32x128x128xf16, {order = #NHWC}> {
    // expected-error@+1 {{'dynamic_input_offsets' is provided, but 'initial_input_offset_attr' does not contain dynamic sentinel values}}
    %0 = VPU.Interpolate(%input, %dyn_in_offsets) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
            mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0],
            pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        initial_input_dims_attr = [1, 32, 64, 64],
        initial_input_offset_attr = [0, 0, 0, 0],
        initial_output_dims_attr = [1, 32, 128, 128],
        initial_output_offset_attr = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 1, 0>,
        scales_attr = [2.0, 2.0],
        sizes_attr = [128, 128],
        tile_offset_attr = [0.0, 0.0, 0.0, 0.0]
    } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<4xi64> -> tensor<1x32x128x128xf16, {order = #NHWC}>
    return %0 : tensor<1x32x128x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @InterpolateDynamicOutputOffsetsWithoutAttr(
        %input: tensor<1x32x64x64xf16, {order = #NHWC}>,
        %dyn_out_offsets: tensor<4xi64>) -> tensor<1x32x128x128xf16, {order = #NHWC}> {
    // expected-error@+1 {{'dynamic_output_offsets' requires 'initial_output_offset_attr' with dynamic sentinels to describe dynamic dimensions}}
    %0 = VPU.Interpolate(%input, %dyn_out_offsets) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
            mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0],
            pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        initial_input_dims_attr = [1, 32, 64, 64],
        initial_input_offset_attr = [0, 0, 0, 0],
        initial_output_dims_attr = [1, 32, 128, 128],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 1>,
        scales_attr = [2.0, 2.0],
        sizes_attr = [128, 128],
        tile_offset_attr = [0.0, 0.0, 0.0, 0.0]
    } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<4xi64> -> tensor<1x32x128x128xf16, {order = #NHWC}>
    return %0 : tensor<1x32x128x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @InterpolateStaticOffsetsWithDynamicSentinelWithoutTensor(
        %input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x128x128xf16, {order = #NHWC}> {
    // expected-error@+1 {{'initial_input_offset_attr' contains dynamic sentinel values but 'dynamic_input_offsets' is not provided}}
    %0 = VPU.Interpolate(%input) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
            mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0],
            pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        initial_input_dims_attr = [1, 32, 64, 64],
        initial_input_offset_attr = [0, 0, -9223372036854775808, 0],
        initial_output_dims_attr = [1, 32, 128, 128],
        initial_output_offset_attr = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>,
        scales_attr = [2.0, 2.0],
        sizes_attr = [128, 128],
        tile_offset_attr = [0.0, 0.0, 0.0, 0.0]
    } : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x128x128xf16, {order = #NHWC}>
    return %0 : tensor<1x32x128x128xf16, {order = #NHWC}>
}
