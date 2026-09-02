//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateStaticShape
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x100x80xf16, {order = #NHWC}>
func.func @InterpolateStaticShape(%arg0: tensor<1x32x100x80xf16, {order = #NHWC}>) -> tensor<1x32x400x320xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        scales_attr = [4.0, 4.0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>
    } : tensor<1x32x100x80xf16, {order = #NHWC}> -> tensor<1x32x400x320xf16, {order = #NHWC}>

    return %0 : tensor<1x32x400x320xf16, {order = #NHWC}>

    // CHECK:       [[INTERP:%.+]] = VPU.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>
    // CHECK-SAME:      nearest_mode = <ROUND_PREFER_FLOOR>
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x400x320xf16, {order = #NHWC}>
    // CHECK:       return [[INTERP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateDynamicShapeBounded
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16
func.func @InterpolateDynamicShapeBounded(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}> {

    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        scales_attr = [4.0, 4.0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[INTERP:%.+]] = VPU.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>
    // CHECK-SAME:      nearest_mode = <ROUND_PREFER_FLOOR>
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[INTERP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateDynamicHalfPixelLinear
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16
func.func @InterpolateDynamicHalfPixelLinear(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}> {

    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        scales_attr = [4.0, 4.0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[INTERP:%.+]] = VPU.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <LINEAR>
    // CHECK-SAME:      coord_mode = <HALF_PIXEL>
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[INTERP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateDynamicAlignCorners
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16
func.func @InterpolateDynamicAlignCorners(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}> {

    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        scales_attr = [4.0, 4.0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[INTERP:%.+]] = VPU.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <LINEAR>
    // CHECK-SAME:      coord_mode = <ALIGN_CORNERS>
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[INTERP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateDynamicDownscale
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16
func.func @InterpolateDynamicDownscale(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}> {

    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        scales_attr = [0.25, 0.25],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[INTERP:%.+]] = VPU.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>
    // CHECK-SAME:      nearest_mode = <FLOOR>
    // CHECK-SAME:      scales_attr = [2.500000e-01, 2.500000e-01]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[INTERP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateDynamicCubic
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16
func.func @InterpolateDynamicCubic(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}> {

    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <CUBIC>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        scales_attr = [4.0, 4.0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[INTERP:%.+]] = VPU.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <CUBIC>
    // CHECK-SAME:      coord_mode = <PYTORCH_HALF_PIXEL>
    // CHECK-SAME:      cube_coeff = -7.500000e-01
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[INTERP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateDynamicTfHalfPixel
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16
func.func @InterpolateDynamicTfHalfPixel(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}> {

    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <TF_HALF_PIXEL_FOR_NN>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <CEIL>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        scales_attr = [4.0, 4.0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[INTERP:%.+]] = VPU.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>
    // CHECK-SAME:      coord_mode = <TF_HALF_PIXEL_FOR_NN>
    // CHECK-SAME:      nearest_mode = <CEIL>
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[INTERP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateDynamicSingleAxis
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x80xf16
func.func @InterpolateDynamicSingleAxis(
        %arg0: tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>)
            -> tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 80]> : tensor<4xsi64>, order = #NHWC}> {

    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2],
        scales_attr = [4.0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>
    } : tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
        -> tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 80]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 80]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[INTERP:%.+]] = VPU.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>
    // CHECK-SAME:      axes_attr = [2]
    // CHECK-SAME:      scales_attr = [4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 80]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[INTERP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateDynamicHeightStaticWidth
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x80xf16
func.func @InterpolateDynamicHeightStaticWidth(
        %arg0: tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>)
            -> tensor<1x32x?x320xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}> {

    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        scales_attr = [4.0, 4.0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>
    } : tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
        -> tensor<1x32x?x320xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x32x?x320xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[INTERP:%.+]] = VPU.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>
    // CHECK-SAME:      axes_attr = [2, 3]
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x320xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @InterpolateDynamicLinearOnnx
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16
func.func @InterpolateDynamicLinearOnnx(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}> {

    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        scales_attr = [4.0, 4.0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    // CHECK:       [[INTERP:%.+]] = VPU.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <LINEAR_ONNX>
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>
    // CHECK:       return [[INTERP]]
}
