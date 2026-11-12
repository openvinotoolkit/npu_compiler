//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true enable-se-ptrs-operations=true" --convert-to-spatial-op --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @ConvertToSpatialInterpolation
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x16x64xf16>
func.func @ConvertToSpatialInterpolation(%arg0: tensor<1x16x16x64xf16>) -> tensor<1x32x32x64xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [1, 2],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [32, 32]} :
         tensor<1x16x16x64xf16> -> tensor<1x32x32x64xf16>

    return %0 : tensor<1x32x32x64xf16>

    // CHECK:       [[INPUT_TRANSPOSE:%.+]] = IE.Transpose([[ARG_0]]) {order_value = #NWCH} : tensor<1x16x16x64xf16> -> tensor<1x64x16x16xf16>
    // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[INPUT_TRANSPOSE]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [32, 32]} :
    // CHECK-SAME:                        tensor<1x64x16x16xf16> -> tensor<1x64x32x32xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.+]] = IE.Transpose([[INTERPOLATE]]) {order_value = #NHWC} : tensor<1x64x32x32xf16> -> tensor<1x32x32x64xf16>

    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x32x32x64xf16>
}

// -----

// CHECK-LABEL: @BypassSpatialInterpolation
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x64x16x16xf16>
func.func @BypassSpatialInterpolation(%arg0: tensor<1x64x16x16xf16>) -> tensor<1x64x32x32xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [32, 32]} :
         tensor<1x64x16x16xf16> -> tensor<1x64x32x32xf16>

    return %0 : tensor<1x64x32x32xf16>

    // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [32, 32]} :
    // CHECK-SAME:                        tensor<1x64x16x16xf16> -> tensor<1x64x32x32xf16>

    // CHECK:       return [[INTERPOLATE]] : tensor<1x64x32x32xf16>
}

// -----

// CHECK-LABEL: @ConvertToSpatialInterpolationOnSingleDim
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x16x64xf16>
func.func @ConvertToSpatialInterpolationOnSingleDim(%arg0: tensor<1x16x16x64xf16>) -> tensor<1x32x16x64xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 32, 16, 64]} :
         tensor<1x16x16x64xf16> -> tensor<1x32x16x64xf16>

    return %0 : tensor<1x32x16x64xf16>

    // CHECK:       [[INPUT_TRANSPOSE:%.+]] = IE.Transpose([[ARG_0]]) {order_value = #NWCH} : tensor<1x16x16x64xf16> -> tensor<1x64x16x16xf16>
    // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[INPUT_TRANSPOSE]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00], sizes_attr = [1, 64, 32, 16]} :
    // CHECK-SAME:                        tensor<1x64x16x16xf16> -> tensor<1x64x32x16xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.+]] = IE.Transpose([[INTERPOLATE]]) {order_value = #NHWC} : tensor<1x64x32x16xf16> -> tensor<1x32x16x64xf16>

    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x32x16x64xf16>
}

// -----

// CHECK-LABEL: @BypassSpatialInterpolationOnSingleDim
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x8x64x2xf16>
func.func @BypassSpatialInterpolationOnSingleDim(%arg0: tensor<1x8x64x2xf16>) -> tensor<1x8x64x4xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00], sizes_attr = [1, 8, 64, 4]} :
         tensor<1x8x64x2xf16> -> tensor<1x8x64x4xf16>

    return %0 : tensor<1x8x64x4xf16>

    // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00], sizes_attr = [1, 8, 64, 4]} :
    // CHECK-SAME:                        tensor<1x8x64x2xf16> -> tensor<1x8x64x4xf16>

    // CHECK:       return [[INTERPOLATE]] : tensor<1x8x64x4xf16>
}

// -----

// CHECK-LABEL: @ConvertToSpatialInterpolationWithFullDimsAttr_ShapeCalcMode_SCALES
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x32x4xf16>) -> tensor<1x8x64x4xf16>
func.func @ConvertToSpatialInterpolationWithFullDimsAttr_ShapeCalcMode_SCALES(%arg0: tensor<1x4x32x4xf16>) -> tensor<1x8x64x4xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 2.000000e+00, 2.000000e+00, 1.000000e+00], sizes_attr = [1, 8, 64, 4]} :
         tensor<1x4x32x4xf16> -> tensor<1x8x64x4xf16>

    return %0 : tensor<1x8x64x4xf16>

    // CHECK:       [[INPUT_TRANSPOSE:%.+]] = IE.Transpose([[ARG_0]]) {order_value = #NWCH} : tensor<1x4x32x4xf16> -> tensor<1x4x4x32xf16>
    // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[INPUT_TRANSPOSE]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                        sizes_attr = [1, 4, 8, 64]} : tensor<1x4x4x32xf16> -> tensor<1x4x8x64xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.+]] = IE.Transpose([[INTERPOLATE]]) {order_value = #NHWC} : tensor<1x4x8x64xf16> -> tensor<1x8x64x4xf16>

    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x8x64x4xf16>
}

// -----

// CHECK-LABEL: @ConvertToSpatialInterpolationWithFullDimsAttr_ShapeCalcMode_SIZES
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x32x4xf16>) -> tensor<1x8x64x4xf16>
func.func @ConvertToSpatialInterpolationWithFullDimsAttr_ShapeCalcMode_SIZES(%arg0: tensor<1x4x32x4xf16>) -> tensor<1x8x64x4xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 8, 64, 4]} :
         tensor<1x4x32x4xf16> -> tensor<1x8x64x4xf16>

    return %0 : tensor<1x8x64x4xf16>

    // CHECK:       [[INPUT_TRANSPOSE:%.+]] = IE.Transpose([[ARG_0]]) {order_value = #NWCH} : tensor<1x4x32x4xf16> -> tensor<1x4x4x32xf16>
    // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate([[INPUT_TRANSPOSE]])
    // CHECK-SAME:                        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:                        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
    // CHECK-SAME:                        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00],
    // CHECK-SAME:                        sizes_attr = [1, 4, 8, 64]} : tensor<1x4x4x32xf16> -> tensor<1x4x8x64xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.+]] = IE.Transpose([[INTERPOLATE]]) {order_value = #NHWC} : tensor<1x4x8x64xf16> -> tensor<1x8x64x4xf16>

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
    // CHECK:       [[INPUT_TRANSPOSE:%.+]] = IE.Transpose([[INPUT_DATA]]) {order_value = #NWCH}
    // CHECK-SAME:                 tensor<1x7x9x64xf16> -> tensor<1x64x7x9xf16>
    // CHECK:       [[ROLL:%.+]] = IE.Roll([[INPUT_TRANSPOSE]], [[SHIFTS]], [[AXES]])
    // CHECK-SAME:                 tensor<1x64x7x9xf16>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x64x7x9xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.+]] = IE.Transpose([[ROLL]]) {order_value = #NHWC}
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
    // CHECK:       [[INPUT_TRANSPOSE:%.+]] = IE.Transpose([[INPUT_DATA]]) {order_value = #NWCH}
    // CHECK-SAME:                 tensor<1x7x9x64xf16> -> tensor<1x64x7x9xf16>
    // CHECK:       [[ROLL:%.+]] = IE.Roll([[INPUT_TRANSPOSE]], [[SHIFTS]], [[AXES]])
    // CHECK-SAME:                 tensor<1x64x7x9xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x64x7x9xf16>
    // CHECK:       [[OUTPUT_TRANSPOSE:%.+]] = IE.Transpose([[ROLL]]) {order_value = #NHWC}
    // CHECK-SAME:                 tensor<1x64x7x9xf16> -> tensor<1x7x9x64xf16>
    // CHECK:       return [[OUTPUT_TRANSPOSE]] : tensor<1x7x9x64xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatSingleAxis
// CHECK-SAME: ([[INPUT_DATA:%.+]]: tensor<1x64x9x10xf16>)
func.func @ConvertToSliceConcatSingleAxis(%arg0: tensor<1x64x9x10xf16>) -> tensor<1x64x9x10xf16> {
    %shift = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    %axes = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %roll = IE.Roll(%arg0, %shift, %axes) : tensor<1x64x9x10xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x64x9x10xf16>
    return %roll : tensor<1x64x9x10xf16>

    // CHECK:       [[SLICE_TAIL:%.+]] = IE.Slice [[INPUT_DATA]] [0, 59, 0, 0] [1, 5, 9, 10]
    // CHECK-SAME:      : tensor<1x64x9x10xf16> to tensor<1x5x9x10xf16>
    // CHECK:       [[SLICE_HEAD:%.+]] = IE.Slice [[INPUT_DATA]] [0, 0, 0, 0] [1, 59, 9, 10]
    // CHECK-SAME:      : tensor<1x64x9x10xf16> to tensor<1x59x9x10xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SLICE_TAIL]], [[SLICE_HEAD]])
    // CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 5, 0, 0]]}
    // CHECK-SAME:      : tensor<1x5x9x10xf16>, tensor<1x59x9x10xf16> -> tensor<1x64x9x10xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x64x9x10xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatMultiAxis
// CHECK-SAME: ([[INPUT_DATA:%.+]]: tensor<1x7x9x64xf16>)
func.func @ConvertToSliceConcatMultiAxis(%arg0: tensor<1x7x9x64xf16>) -> tensor<1x7x9x64xf16> {
    %shift = const.Declare tensor<2xsi32> = dense<[5, 4]> : tensor<2xsi32>
    %axes = const.Declare tensor<2xsi32> = dense<[1, 3]> : tensor<2xsi32>
    %roll = IE.Roll(%arg0, %shift, %axes) : tensor<1x7x9x64xf16>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x7x9x64xf16>
    return %roll : tensor<1x7x9x64xf16>

    // CHECK:       [[SLICE_TAIL_0:%.+]] = IE.Slice [[INPUT_DATA]] [0, 2, 0, 0] [1, 5, 9, 64]
    // CHECK-SAME:      : tensor<1x7x9x64xf16> to tensor<1x5x9x64xf16>
    // CHECK:       [[SLICE_HEAD_0:%.+]] = IE.Slice [[INPUT_DATA]] [0, 0, 0, 0] [1, 2, 9, 64]
    // CHECK-SAME:      : tensor<1x7x9x64xf16> to tensor<1x2x9x64xf16>
    // CHECK:       [[CONCAT_0:%.+]] = IE.Concat([[SLICE_TAIL_0]], [[SLICE_HEAD_0]])
    // CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 5, 0, 0]]}
    // CHECK-SAME:      : tensor<1x5x9x64xf16>, tensor<1x2x9x64xf16> -> tensor<1x7x9x64xf16>

    // CHECK:       [[SLICE_TAIL_1:%.+]] = IE.Slice [[CONCAT_0]] [0, 0, 0, 60] [1, 7, 9, 4]
    // CHECK-SAME:      : tensor<1x7x9x64xf16> to tensor<1x7x9x4xf16>
    // CHECK:       [[SLICE_HEAD_1:%.+]] = IE.Slice [[CONCAT_0]] [0, 0, 0, 0] [1, 7, 9, 60]
    // CHECK-SAME:      : tensor<1x7x9x64xf16> to tensor<1x7x9x60xf16>
    // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[SLICE_TAIL_1]], [[SLICE_HEAD_1]])
    // CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 4]]}
    // CHECK-SAME:      : tensor<1x7x9x4xf16>, tensor<1x7x9x60xf16> -> tensor<1x7x9x64xf16>
    // CHECK:       return [[CONCAT_1]] : tensor<1x7x9x64xf16>
}

// -----

// All axes have shift=0
// CHECK-LABEL: @NotConvertToSliceConcatAllZeroShift
// CHECK-SAME: ([[INPUT_DATA:%.+]]: tensor<1x7x9x64xf16>)
func.func @NotConvertToSliceConcatAllZeroShift(%arg0: tensor<1x7x9x64xf16>) -> tensor<1x7x9x64xf16> {
    %shift = const.Declare tensor<2xsi32> = dense<[0, 0]> : tensor<2xsi32>
    %axes = const.Declare tensor<2xsi32> = dense<[2, 3]> : tensor<2xsi32>
    %roll = IE.Roll(%arg0, %shift, %axes) : tensor<1x7x9x64xf16>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x7x9x64xf16>
    return %roll : tensor<1x7x9x64xf16>

    // CHECK:      [[ROLL:%.+]] = IE.Roll
    // CHECK：      return [[ROLL]] : tensor<1x7x9x64xf16>
}
