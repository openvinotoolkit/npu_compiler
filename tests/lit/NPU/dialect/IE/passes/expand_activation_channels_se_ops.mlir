//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true enable-se-ptrs-operations=true" --expand-activation-channels --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandInterpolateNearestChannels
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x20x30x30xf16>
func.func @ExpandInterpolateNearestChannels(%arg0: tensor<1x20x30x30xf16>) -> tensor<1x20x60x60xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [60, 60]
         } : tensor<1x20x30x30xf16> -> tensor<1x20x60x60xf16>

    return %0 : tensor<1x20x60x60xf16>
}

// CHECK:       [[PAD:%.+]] = IE.Expand([[ARG_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
// CHECK-SAME:      -> tensor<1x32x30x30xf16>

// CHECK:       [[INTERP:%.+]] = IE.Interpolate([[PAD]])
// CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>,
// CHECK-SAME:                             shape_calc_mode = <SCALES>,
// CHECK-SAME:                             coord_mode = <ASYMMETRIC>,
// CHECK-SAME:                             nearest_mode = <FLOOR>,
// CHECK-SAME:                             antialias = false,
// CHECK-SAME:                             pads_begin = [0, 0, 0, 0],
// CHECK-SAME:                             pads_end = [0, 0, 0, 0],
// CHECK-SAME:                             cube_coeff = -7.500000e-01 : f64>,
// CHECK-SAME:      axes_attr = [2, 3],
// CHECK-SAME:      scales_attr = [2.000000e+00, 2.000000e+00],
// CHECK-SAME:      sizes_attr = [60, 60]
// CHECK-SAME:      -> tensor<1x32x60x60xf16>

// CHECK:       [[OUT:%.+]] = IE.Slice [[INTERP]] [0, 0, 0, 0] [1, 20, 60, 60]
// CHECK-SAME:      to tensor<1x20x60x60xf16>

// CHECK:       return [[OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandInterpolateLinearChannels
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x20x30x30xf16>
func.func @ExpandInterpolateLinearChannels(%arg0: tensor<1x20x30x30xf16>) -> tensor<1x20x60x60xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [60, 60]
         } : tensor<1x20x30x30xf16> -> tensor<1x20x60x60xf16>

    return %0 : tensor<1x20x60x60xf16>
}

// CHECK:       [[PAD:%.+]] = IE.Expand([[ARG_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
// CHECK-SAME:      -> tensor<1x32x30x30xf16>

// CHECK:       [[INTERP:%.+]] = IE.Interpolate([[PAD]])
// CHECK-SAME:      attr = #IE.Interpolate<mode = <LINEAR>,
// CHECK-SAME:                             shape_calc_mode = <SCALES>,
// CHECK-SAME:                             coord_mode = <ASYMMETRIC>,
// CHECK-SAME:                             nearest_mode = <FLOOR>,
// CHECK-SAME:                             antialias = false,
// CHECK-SAME:                             pads_begin = [0, 0, 0, 0],
// CHECK-SAME:                             pads_end = [0, 0, 0, 0],
// CHECK-SAME:                             cube_coeff = -7.500000e-01 : f64>,
// CHECK-SAME:      axes_attr = [2, 3],
// CHECK-SAME:      scales_attr = [2.000000e+00, 2.000000e+00],
// CHECK-SAME:      sizes_attr = [60, 60]
// CHECK-SAME:      -> tensor<1x32x60x60xf16>

// CHECK:       [[OUT:%.+]] = IE.Slice [[INTERP]] [0, 0, 0, 0] [1, 20, 60, 60]
// CHECK-SAME:      to tensor<1x20x60x60xf16>

// CHECK:       return [[OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandInterpolateChannelsWithShapeCalcModeSizesAttrInput
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x20x30x30xf16>
func.func @ExpandInterpolateChannelsWithShapeCalcModeSizesAttrInput(%arg0: tensor<1x20x30x30xf16>) -> tensor<1x20x60x60xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 20, 60, 60]
         } : tensor<1x20x30x30xf16> -> tensor<1x20x60x60xf16>

    return %0 : tensor<1x20x60x60xf16>
}

// CHECK:       [[PAD:%.+]] = IE.Expand([[ARG_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
// CHECK-SAME:      -> tensor<1x32x30x30xf16>

// CHECK:       [[INTERP:%.+]] = IE.Interpolate([[PAD]])
// CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>,
// CHECK-SAME:                             shape_calc_mode = <SIZES>,
// CHECK-SAME:                             coord_mode = <ASYMMETRIC>,
// CHECK-SAME:                             nearest_mode = <FLOOR>,
// CHECK-SAME:                             antialias = false,
// CHECK-SAME:                             pads_begin = [0, 0, 0, 0],
// CHECK-SAME:                             pads_end = [0, 0, 0, 0],
// CHECK-SAME:                             cube_coeff = -7.500000e-01 : f64>,
// CHECK-SAME:      axes_attr = [0, 1, 2, 3],
// CHECK-SAME:      scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00],
// CHECK-SAME:      sizes_attr = [1, 32, 60, 60]
// CHECK-SAME:      -> tensor<1x32x60x60xf16>

// CHECK:       [[OUT:%.+]] = IE.Slice [[INTERP]] [0, 0, 0, 0] [1, 20, 60, 60]
// CHECK-SAME:      to tensor<1x20x60x60xf16>

// CHECK:       return [[OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandInterpolateChannelsWithShapeCalcModeSizesTensorInput
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x20x30x30xf16>
func.func @ExpandInterpolateChannelsWithShapeCalcModeSizesTensorInput(%arg0: tensor<1x20x30x30xf16>) -> tensor<1x20x60x60xf16> {
    %0 = const.Declare tensor<4xsi64> = dense<[1, 20, 60, 60]> : tensor<4xsi64>
    %1 = IE.Interpolate(%arg0, %0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 1, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
         } : tensor<1x20x30x30xf16>, tensor<4xsi64> -> tensor<1x20x60x60xf16>

    return %1 : tensor<1x20x60x60xf16>
}

// CHECK:       [[PAD:%.+]] = IE.Expand([[ARG_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
// CHECK-SAME:      -> tensor<1x32x30x30xf16>

// CHECK:       [[INTERP:%.+]] = IE.Interpolate([[PAD]])
// CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>,
// CHECK-SAME:                             shape_calc_mode = <SIZES>,
// CHECK-SAME:                             coord_mode = <ASYMMETRIC>,
// CHECK-SAME:                             nearest_mode = <FLOOR>,
// CHECK-SAME:                             antialias = false,
// CHECK-SAME:                             pads_begin = [0, 0, 0, 0],
// CHECK-SAME:                             pads_end = [0, 0, 0, 0],
// CHECK-SAME:                             cube_coeff = -7.500000e-01 : f64>,
// CHECK-SAME:      axes_attr = [0, 1, 2, 3],
// CHECK-SAME:      scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00],
// CHECK-SAME:      sizes_attr = [1, 32, 60, 60]
// CHECK-SAME:      -> tensor<1x32x60x60xf16>

// CHECK:       [[OUT:%.+]] = IE.Slice [[INTERP]] [0, 0, 0, 0] [1, 20, 60, 60]
// CHECK-SAME:      to tensor<1x20x60x60xf16>

// CHECK:       return [[OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotExpandPadOp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x20x23x30xf16, {order = #NHWC}>)
func.func @DoNotExpandPadOp(%input: tensor<1x20x23x30xf16, {order = #NHWC}>) -> tensor<1x20x26x33xf16, {order = #NHWC}> {
    %0 = IE.Pad(%input) {
                mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64,
                pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]
            } : tensor<1x20x23x30xf16, {order = #NHWC}> -> tensor<1x20x26x33xf16, {order = #NHWC}>

    return %0 : tensor<1x20x26x33xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = IE.Pad([[INPUT]]) {
    // CHECK-SAME:          mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]
    // CHECK-SAME:      } : tensor<1x20x23x30xf16, {order = #NHWC}> -> tensor<1x20x26x33xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandPadOp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x40x23x30xf16, {order = #NHWC}>)
func.func @ExpandPadOp(%input: tensor<1x40x23x30xf16, {order = #NHWC}>) -> tensor<1x40x26x33xf16, {order = #NHWC}> {
    %0 = IE.Pad(%input) {
                mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64,
                pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]
            } : tensor<1x40x23x30xf16, {order = #NHWC}> -> tensor<1x40x26x33xf16, {order = #NHWC}>

    return %0 : tensor<1x40x26x33xf16, {order = #NHWC}>

    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]}
    // CHECK-SAME:      : tensor<1x40x23x30xf16, {order = #NHWC}> -> tensor<1x48x23x30xf16, {order = #NHWC}>
    // CHECK:       [[PAD:%.+]] = IE.Pad([[EXPAND]]) {
    // CHECK-SAME:          mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]
    // CHECK-SAME:      } : tensor<1x48x23x30xf16, {order = #NHWC}> -> tensor<1x48x26x33xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_SLICE:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 0] [1, 40, 26, 33]
    // CHECK-SAME:      : tensor<1x48x26x33xf16, {order = #NHWC}> to tensor<1x40x26x33xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT_SLICE]]
}
