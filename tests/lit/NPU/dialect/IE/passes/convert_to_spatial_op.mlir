//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-spatial-op="se-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertToSpatialInterpolation
func.func @ConvertToSpatialInterpolation(%arg0: tensor<1x16x16x64xf16>) -> tensor<1x32x32x64xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [1, 2],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [32, 32]} :
         tensor<1x16x16x64xf16> -> tensor<1x32x32x64xf16>

    return %0 : tensor<1x32x32x64xf16>

    // CHECK:       [[INPUT_TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #NWCH} : tensor<1x16x16x64xf16> -> tensor<1x64x16x16xf16>
    // CHECK:       [[INTERPOLATE:%.*]] = IE.Interpolate([[INPUT_TRANSPOSE]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [32, 32]} :
    // CHECK-SAME:                        tensor<1x64x16x16xf16> -> tensor<1x64x32x32xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.*]] = IE.Transpose([[INTERPOLATE]]) {order_value = #NHWC} : tensor<1x64x32x32xf16> -> tensor<1x32x32x64xf16>

    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x32x32x64xf16>
}

// -----

// CHECK-LABEL: @BypassSpatialInterpolation
func.func @BypassSpatialInterpolation(%arg0: tensor<1x64x16x16xf16>) -> tensor<1x64x32x32xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [32, 32]} :
         tensor<1x64x16x16xf16> -> tensor<1x64x32x32xf16>

    return %0 : tensor<1x64x32x32xf16>

    // CHECK:       [[INTERPOLATE:%.*]] = IE.Interpolate(%arg0)
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [32, 32]} :
    // CHECK-SAME:                        tensor<1x64x16x16xf16> -> tensor<1x64x32x32xf16>

    // CHECK:       return [[INTERPOLATE]] : tensor<1x64x32x32xf16>
}

// -----
// CHECK-LABEL: @ConvertToSpatialInterpolationOnSingleDim
func.func @ConvertToSpatialInterpolationOnSingleDim(%arg0: tensor<1x16x16x64xf16>) -> tensor<1x32x16x64xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 32, 16, 64]} :
         tensor<1x16x16x64xf16> -> tensor<1x32x16x64xf16>

    return %0 : tensor<1x32x16x64xf16>

    // CHECK:       [[INPUT_TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #NWCH} : tensor<1x16x16x64xf16> -> tensor<1x64x16x16xf16>
    // CHECK:       [[INTERPOLATE:%.*]] = IE.Interpolate([[INPUT_TRANSPOSE]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00], sizes_attr = [1, 64, 32, 16]} :
    // CHECK-SAME:                        tensor<1x64x16x16xf16> -> tensor<1x64x32x16xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.*]] = IE.Transpose([[INTERPOLATE]]) {order_value = #NHWC} : tensor<1x64x32x16xf16> -> tensor<1x32x16x64xf16>

    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x32x16x64xf16>
}

// -----

// CHECK-LABEL: @BypassSpatialInterpolationOnSingleDim
func.func @BypassSpatialInterpolationOnSingleDim(%arg0: tensor<1x8x64x2xf16>) -> tensor<1x8x64x4xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00], sizes_attr = [1, 8, 64, 4]} :
         tensor<1x8x64x2xf16> -> tensor<1x8x64x4xf16>


    return %0 : tensor<1x8x64x4xf16>

    // CHECK:       [[INTERPOLATE:%.*]] = IE.Interpolate(%arg0)
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00], sizes_attr = [1, 8, 64, 4]} :
    // CHECK-SAME:                        tensor<1x8x64x2xf16> -> tensor<1x8x64x4xf16>

    // CHECK:       return [[INTERPOLATE]] : tensor<1x8x64x4xf16>
}

// -----

// CHECK-LABEL: @ConvertToSpatialInterpolationWithFullDimsAttr_ShapeCalcMode_SCALES
func.func @ConvertToSpatialInterpolationWithFullDimsAttr_ShapeCalcMode_SCALES(%arg0: tensor<1x4x32x4xf16>) -> tensor<1x8x64x4xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 2.000000e+00, 2.000000e+00, 1.000000e+00], sizes_attr = [1, 8, 64, 4]} :
         tensor<1x4x32x4xf16> -> tensor<1x8x64x4xf16>

    return %0 : tensor<1x8x64x4xf16>

    // CHECK:       [[INPUT_TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #NWCH} : tensor<1x4x32x4xf16> -> tensor<1x4x4x32xf16>
    // CHECK:       [[INTERPOLATE:%.*]] = IE.Interpolate([[INPUT_TRANSPOSE]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                        sizes_attr = [1, 4, 8, 64]} : tensor<1x4x4x32xf16> -> tensor<1x4x8x64xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.*]] = IE.Transpose([[INTERPOLATE]]) {order_value = #NHWC} : tensor<1x4x8x64xf16> -> tensor<1x8x64x4xf16>

    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x8x64x4xf16>
}

// -----

// CHECK-LABEL: @ConvertToSpatialInterpolationWithFullDimsAttr_ShapeCalcMode_SIZES
func.func @ConvertToSpatialInterpolationWithFullDimsAttr_ShapeCalcMode_SIZES(%arg0: tensor<1x4x32x4xf16>) -> tensor<1x8x64x4xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 8, 64, 4]} :
         tensor<1x4x32x4xf16> -> tensor<1x8x64x4xf16>

    return %0 : tensor<1x8x64x4xf16>

    // CHECK:       [[INPUT_TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #NWCH} : tensor<1x4x32x4xf16> -> tensor<1x4x4x32xf16>
    // CHECK:       [[INTERPOLATE:%.*]] = IE.Interpolate([[INPUT_TRANSPOSE]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00],
    // CHECK-SAME:                        sizes_attr = [1, 4, 8, 64]} : tensor<1x4x4x32xf16> -> tensor<1x4x8x64xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.*]] = IE.Transpose([[INTERPOLATE]]) {order_value = #NHWC} : tensor<1x4x8x64xf16> -> tensor<1x8x64x4xf16>

    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x8x64x4xf16>
}

// -----

// CHECK-LABEL: @ConvertToSpatialRollAtChannelAndHeight
// CHECK-SAME: ([[INPUT_DATA:%.+]]: tensor<1x7x9x64xf16>)
func.func @ConvertToSpatialRollAtChannelAndHeight(%arg0: tensor<1x7x9x64xf16>) -> tensor<1x7x9x64xf16> {
    %shift = const.Declare tensor<2xsi32> = dense<[5, 4]> : tensor<2xsi32>
    %axes = const.Declare tensor<2xsi32> = dense<[1, 2]> : tensor<2xsi32>
    %roll = IE.Roll(%arg0, %shift, %axes) : tensor<1x7x9x64xf16>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x7x9x64xf16>
    return %roll : tensor<1x7x9x64xf16>

    // CHECK-DAG:   [[AXES:%.+]] = const.Declare tensor<2xsi32> = dense<[2, 3]> : tensor<2xsi32>
    // CHECK-DAG:   [[SHIFTS:%.+]] = const.Declare tensor<2xsi32> = dense<[5, 4]> : tensor<2xsi32>
    // CHECK:       [[INPUT_TRANSPOSE:%.*]] = IE.Transpose([[INPUT_DATA]]) {order_value = #NWCH}
    // CHECK-SAME:                 tensor<1x7x9x64xf16> -> tensor<1x64x7x9xf16>
    // CHECK:       [[ROLL:%.+]] = IE.Roll([[INPUT_TRANSPOSE]], [[SHIFTS]], [[AXES]])
    // CHECK-SAME:                 tensor<1x64x7x9xf16>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x64x7x9xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.*]] = IE.Transpose([[ROLL]]) {order_value = #NHWC}
    // CHECK-SAME:                 tensor<1x64x7x9xf16> -> tensor<1x7x9x64xf16>
    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x7x9x64xf16>
}

// -----

// CHECK-LABEL: @ConvertToSpatialRollAtChannel
// CHECK-SAME: ([[INPUT_DATA:%.+]]: tensor<1x7x9x64xf16>)
func.func @ConvertToSpatialRollAtChannel(%arg0: tensor<1x7x9x64xf16>) -> tensor<1x7x9x64xf16> {
    %shift = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    %axes = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %roll = IE.Roll(%arg0, %shift, %axes) : tensor<1x7x9x64xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x7x9x64xf16>
    return %roll : tensor<1x7x9x64xf16>

    // CHECK-DAG:   [[AXES:%.+]] = const.Declare tensor<1xsi32> = dense<2> : tensor<1xsi32>
    // CHECK-DAG:   [[SHIFTS:%.+]] = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    // CHECK:       [[INPUT_TRANSPOSE:%.*]] = IE.Transpose([[INPUT_DATA]]) {order_value = #NWCH}
    // CHECK-SAME:                 tensor<1x7x9x64xf16> -> tensor<1x64x7x9xf16>
    // CHECK:       [[ROLL:%.+]] = IE.Roll([[INPUT_TRANSPOSE]], [[SHIFTS]], [[AXES]])
    // CHECK-SAME:                 tensor<1x64x7x9xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x64x7x9xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.*]] = IE.Transpose([[ROLL]]) {order_value = #NHWC}
    // CHECK-SAME:                 tensor<1x64x7x9xf16> -> tensor<1x7x9x64xf16>
    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x7x9x64xf16>
}

// -----

// CHECK-LABEL: @NotConvertToSpatialRollDueToWidthNotAligned
// CHECK-SAME: ([[INPUT_DATA:%.+]]: tensor<1x64x9x10xf16>)
func.func @NotConvertToSpatialRollDueToWidthNotAligned(%arg0: tensor<1x64x9x10xf16>) -> tensor<1x64x9x10xf16> {
    %shift = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    %axes = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %roll = IE.Roll(%arg0, %shift, %axes) : tensor<1x64x9x10xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x64x9x10xf16>
    return %roll : tensor<1x64x9x10xf16>

    // CHECK-DAG:   [[AXES:%.+]] = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    // CHECK-DAG:   [[SHIFTS:%.+]] = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    // CHECK:       [[ROLL:%.+]] = IE.Roll([[INPUT_DATA]], [[SHIFTS]], [[AXES]])
    // CHECK-SAME:                 tensor<1x64x9x10xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x64x9x10xf16>
    // CHECK:       return [[ROLL]] : tensor<1x64x9x10xf16>
}

// -----

// CHECK-LABEL: @NotConvertToSpatialRollAtChannelAndWidth
// CHECK-SAME: ([[INPUT_DATA:%.+]]: tensor<1x7x9x64xf16>)
func.func @NotConvertToSpatialRollAtChannelAndWidth(%arg0: tensor<1x7x9x64xf16>) -> tensor<1x7x9x64xf16> {
    %shift = const.Declare tensor<2xsi32> = dense<[5, 4]> : tensor<2xsi32>
    %axes = const.Declare tensor<2xsi32> = dense<[1, 3]> : tensor<2xsi32>
    %roll = IE.Roll(%arg0, %shift, %axes) : tensor<1x7x9x64xf16>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x7x9x64xf16>
    return %roll : tensor<1x7x9x64xf16>

    // CHECK-DAG:   [[AXES:%.+]] = const.Declare tensor<2xsi32> = dense<[1, 3]> : tensor<2xsi32>
    // CHECK-DAG:   [[SHIFTS:%.+]] = const.Declare tensor<2xsi32> = dense<[5, 4]> : tensor<2xsi32>
    // CHECK:       [[ROLL:%.+]] = IE.Roll([[INPUT_DATA]], [[SHIFTS]], [[AXES]])
    // CHECK-SAME:                 tensor<1x7x9x64xf16>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x7x9x64xf16>
    // CHECK:       return [[ROLL]] : tensor<1x7x9x64xf16>
}
