//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --lower-ops-to-se-nce="se-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpolateNearestLargeChannels
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x144x3x3xf16, {order = #NHWC}>)
func.func @InterpolateNearestLargeChannels(%input: tensor<1x144x3x3xf16, {order = #NHWC}>) -> tensor<1x144x6x6xf16, {order = #NHWC}> {
    %output = VPU.Interpolate(%input) {
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
        } : tensor<1x144x3x3xf16, {order = #NHWC}> -> tensor<1x144x6x6xf16, {order = #NHWC}>

    return %output : tensor<1x144x6x6xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 144, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 144 : i64}
    // CHECK-SAME:      -> tensor<1x1x6x6xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x144x6x6xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x144x6x6xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x144x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x144x6x6xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x6x6xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                            nearest_mode = <FLOOR>>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<144x144x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<144x144x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<144x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:       rawFilterShape = [144, 144, 1, 1],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x144x6x6xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpolateBilinearAsymmetricLargeChannels
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x144x3x3xf16, {order = #NHWC}>)
func.func @InterpolateBilinearAsymmetricLargeChannels(%input: tensor<1x144x3x3xf16, {order = #NHWC}>) -> tensor<1x144x6x6xf16, {order = #NHWC}> {
    %output = VPU.Interpolate(%input) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
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
        } : tensor<1x144x3x3xf16, {order = #NHWC}> -> tensor<1x144x6x6xf16, {order = #NHWC}>

    return %output : tensor<1x144x6x6xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 144, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 144 : i64}
    // CHECK-SAME:      -> tensor<1x1x7x7xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x144x7x7xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x144x7x7xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x144x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x144x7x7xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x7x7xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<144x144x2x2xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<144x144x2x2xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<144x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:       rawFilterShape = [144, 144, 2, 2],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x144x6x6xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpolateBilinearHalfPixelLargeChannels
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x144x3x3xf16, {order = #NHWC}>)
func.func @InterpolateBilinearHalfPixelLargeChannels(%input: tensor<1x144x3x3xf16, {order = #NHWC}>) -> tensor<1x144x6x6xf16, {order = #NHWC}> {
    %output = VPU.Interpolate(%input) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <HALF_PIXEL>,
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
        } : tensor<1x144x3x3xf16, {order = #NHWC}> -> tensor<1x144x6x6xf16, {order = #NHWC}>

    return %output : tensor<1x144x6x6xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 144, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 144 : i64}
    // CHECK-SAME:      -> tensor<1x1x8x8xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x144x8x8xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x144x8x8xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x144x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x144x8x8xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x8x8xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<144x144x3x3xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<144x144x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<144x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:       rawFilterShape = [144, 144, 3, 3],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x144x6x6xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpolateBilinearPytorchHalfPixelLargeChannels
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x144x3x3xf16, {order = #NHWC}>)
func.func @InterpolateBilinearPytorchHalfPixelLargeChannels(%input: tensor<1x144x3x3xf16, {order = #NHWC}>) -> tensor<1x144x6x6xf16, {order = #NHWC}> {
    %output = VPU.Interpolate(%input) {
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
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x144x3x3xf16, {order = #NHWC}> -> tensor<1x144x6x6xf16, {order = #NHWC}>

    return %output : tensor<1x144x6x6xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 144, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 144 : i64}
    // CHECK-SAME:      -> tensor<1x1x8x8xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x144x8x8xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x144x8x8xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x144x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x144x8x8xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x8x8xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<144x144x3x3xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<144x144x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<144x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:       rawFilterShape = [144, 144, 3, 3],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x144x6x6xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpolateBilinearAlignCornersWithLargeChannels
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x144x3x3xf16, {order = #NHWC}>)
func.func @InterpolateBilinearAlignCornersWithLargeChannels(%input: tensor<1x144x3x3xf16, {order = #NHWC}>) -> tensor<1x144x7x7xf16, {order = #NHWC}> {
    %output = VPU.Interpolate(%input) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ALIGN_CORNERS>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3],
            scales_attr = [1.000000e+00, 1.000000e+00],
            sizes_attr = [7, 7],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x144x3x3xf16, {order = #NHWC}> -> tensor<1x144x7x7xf16, {order = #NHWC}>

    return %output : tensor<1x144x7x7xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 144, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:               initial_input_shape = [1, 144, 3, 3], initial_output_shape = [1, 144, 7, 7]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 144 : i64}
    // CHECK-SAME:      -> tensor<1x1x9x9xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x144x9x9xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x144x9x9xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:                initial_input_shape = [1, 144, 3, 3], initial_output_shape = [1, 144, 7, 7]>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x144x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x144x9x9xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x9x9xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:                                            initial_input_shape = [1, 144, 3, 3], initial_output_shape = [1, 144, 7, 7]>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<144x144x3x3xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<144x144x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<144x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:       rawFilterShape = [144, 144, 3, 3],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x144x7x7xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]

}
