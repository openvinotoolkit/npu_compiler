//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --adjust-layouts="se-ops-enabled=true" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @AdjustInterpolateNearestLayout
module @AdjustInterpolateNearestLayout {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x30x30xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x60x60xf16>
    }

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x16x30x30xf16>) -> tensor<1x16x60x60xf16> {
func.func @main(%arg0: tensor<1x16x30x30xf16>) -> tensor<1x16x60x60xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [60, 60]
         } : tensor<1x16x30x30xf16> -> tensor<1x16x60x60xf16>

    return %0 : tensor<1x16x60x60xf16>

    // CHECK:       [[INPUT_REORDERED:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NHWC} : tensor<1x16x30x30xf16> -> tensor<1x16x30x30xf16, {order = #NHWC}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[INPUT_REORDERED]])
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
    // CHECK-SAME:      -> tensor<1x16x60x60xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.*]] = IE.Reorder([[INTERP]]) {dstOrder = #NCHW} : tensor<1x16x60x60xf16, {order = #NHWC}> -> tensor<1x16x60x60xf16>

    // CHECK        return [[OUTPUT]]
}
}

// -----

// CHECK-LABEL: @AdjustInterpolateLinearLayout
module @AdjustInterpolateLinearLayout {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x30x30xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x60x60xf16>
    }

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x16x30x30xf16>) -> tensor<1x16x60x60xf16> {
func.func @main(%arg0: tensor<1x16x30x30xf16>) -> tensor<1x16x60x60xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [60, 60]
         } : tensor<1x16x30x30xf16> -> tensor<1x16x60x60xf16>

    return %0 : tensor<1x16x60x60xf16>

    // CHECK:       [[INPUT_REORDERED:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NHWC} : tensor<1x16x30x30xf16> -> tensor<1x16x30x30xf16, {order = #NHWC}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[INPUT_REORDERED]])
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
    // CHECK-SAME:      -> tensor<1x16x60x60xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.*]] = IE.Reorder([[INTERP]]) {dstOrder = #NCHW} : tensor<1x16x60x60xf16, {order = #NHWC}> -> tensor<1x16x60x60xf16>

    // CHECK        return [[OUTPUT]]
}
}

// -----

// CHECK-LABEL: @AdjustTransposedConvolutionLayout
module @AdjustTransposedConvolutionLayout {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x23x30xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x46x60xf16>
    }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x16x23x30xf16>) -> tensor<1x16x46x60xf16> {
func.func @main(%input: tensor<1x16x23x30xf16>) -> tensor<1x16x46x60xf16> {
    %weights = const.Declare tensor<16x16x2x2xf16> = dense<1.000000e+00> : tensor<16x16x2x2xf16>
    %output = IE.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x16x23x30xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x46x60xf16>
    return %output : tensor<1x16x46x60xf16>

    // CHECK:       [[WEIGHTS_REORDERED:%.+]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x2x2xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[INPUT_REORDERED:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NHWC} : tensor<1x16x23x30xf16> -> tensor<1x16x23x30xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = IE.TransposedConvolution([[INPUT_REORDERED]], [[WEIGHTS_REORDERED]]) {
    // CHECK-SAME:          dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
    // CHECK-SAME:      -> tensor<1x16x46x60xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.*]] = IE.Reorder([[CONV]]) {dstOrder = #NCHW} : tensor<1x16x46x60xf16, {order = #NHWC}> -> tensor<1x16x46x60xf16>

    // CHECK        return [[OUTPUT]]

}
}
