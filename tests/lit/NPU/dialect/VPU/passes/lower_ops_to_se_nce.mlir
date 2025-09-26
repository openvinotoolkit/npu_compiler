//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=-1 --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --lower-ops-to-se-nce="se-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @NotConvertTransposedConvolutionWithNegativePadding([[INPUT_DATA:%.+]]: tensor<1x16x1x8xf16, {order = #NHWC}>) -> tensor<1x16x1x18xf16, {order = #NHWC}> {
func.func @NotConvertTransposedConvolutionWithNegativePadding(%input: tensor<1x16x1x8xf16, {order = #NHWC}>) -> tensor<1x16x1x18xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x3xf16, {order = #NHWC}>

    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, -1], strides = [1, 2]
        } : tensor<1x16x1x8xf16, {order = #NHWC}>, tensor<16x16x1x3xf16, {order = #NHWC}> -> tensor<1x16x1x18xf16, {order = #NHWC}>

    return %output : tensor<1x16x1x18xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x3xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT:%.+]] = VPU.TransposedConvolution([[INPUT_DATA]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:      pads_begin = [0, 0], pads_end = [0, -1], spatial_output_padding = [0, 0], strides = [1, 2]} : tensor<1x16x1x8xf16, {order = #NHWC}>, tensor<16x16x1x3xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x1x18xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotLowerInterpolateWithBatch
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x16x3x3xf16, {order = #NHWC}>
func.func @DoNotLowerInterpolateWithBatch(%arg0: tensor<4x16x3x3xf16, {order = #NHWC}>) -> tensor<4x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <NEAREST>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<4x16x3x3xf16, {order = #NHWC}> -> tensor<4x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<4x16x6x6xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearAlignCorners([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearAlignCorners(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ALIGN_CORNERS>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearTFHALFPIXELFORNN([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearTFHALFPIXELFORNN(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <TF_HALF_PIXEL_FOR_NN>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearFloatScales([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearFloatScales(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.999999e+00, 2.999999e+00],
            sizes_attr = [8, 8],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>

    return %0 : tensor<1x16x8x8xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearAsymmetricLargeKernel([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x48x48xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearAsymmetricLargeKernel(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x48x48xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [16.000000e+00, 16.0000000e+00],
            sizes_attr = [36, 36],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x48x48xf16, {order = #NHWC}>

    return %0 : tensor<1x16x48x48xf16, {order = #NHWC}>

    // kernel size is: [16, 16]
    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearHalfPixelLargeKernel([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x60x60xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearHalfPixelLargeKernel(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x60x60xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <HALF_PIXEL>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [20.000000e+00, 20.0000000e+00],
            sizes_attr = [60, 60],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x60x60xf16, {order = #NHWC}>

    return %0 : tensor<1x16x60x60xf16, {order = #NHWC}>

    // kernel size is: [21, 21]
    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearAlignCornersWithIllegalScales([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearAlignCornersWithIllegalScales(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ALIGN_CORNERS>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3],
            scales_attr = [1.000000e+00, 1.0000000e+00],
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // Scales: (output_size - 1) / (input_size - 1) is not an integer
    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}
