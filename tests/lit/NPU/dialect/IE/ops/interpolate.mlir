//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertConstToAttr
func.func @ConvertConstToAttr(%arg0: tensor<1x3x10x10xf32>) -> tensor<1x3x20x15xf32> {
    %0 = const.Declare tensor<2xsi64> = dense<[20, 15]> : tensor<2xsi64>
    %1 = const.Declare tensor<2xf32>  = dense<[2.000000e+00, 1.500000e+00]> : tensor<2xf32>
    %2 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    // CHECK-NOT:   const.Declare
    %3 = IE.Interpolate(%arg0, %0, %1, %2) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<1x3x10x10xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x3x20x15xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME: axes_attr = [2, 3],
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK-SAME: scales_attr = [2.000000e+00, 1.500000e+00],
    // CHECK-SAME: sizes_attr = [20, 15]}
    // CHECK-SAME: tensor<1x3x10x10xf32> -> tensor<1x3x20x15xf32>

    return %3 : tensor<1x3x20x15xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertConstToAttr3InputsSizes
func.func @ConvertConstToAttr3InputsSizes(%arg0: tensor<1x3x10x10xf32>) -> tensor<1x3x20x15xf32> {
    %0 = const.Declare tensor<4xsi64> = dense<[1, 3, 20, 15]> : tensor<4xsi64>
    %1 = const.Declare tensor<4xf32>  = dense<[1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>
    // CHECK-NOT:   const.Declare
    %2 = IE.Interpolate(%arg0, %0, %1) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, operandSegmentSizes = array<i32: 1, 1, 1, 0>} : tensor<1x3x10x10xf32>, tensor<4xsi64>, tensor<4xf32> -> tensor<1x3x20x15xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME: axes_attr = [2, 3],
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK-SAME: scales_attr = [2.000000e+00, 1.500000e+00],
    // CHECK-SAME: sizes_attr = [20, 15]}
    // CHECK-SAME: tensor<1x3x10x10xf32> -> tensor<1x3x20x15xf32>

    return %2 : tensor<1x3x20x15xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertConstToAttr3InputsScales
func.func @ConvertConstToAttr3InputsScales(%arg0: tensor<1x3x10x10xf32>) -> tensor<1x3x20x15xf32> {
    %0 = const.Declare tensor<4xsi64> = dense<[1, 1, 1, 1]> : tensor<4xsi64>
    %1 = const.Declare tensor<4xf32>  = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 1.500000e+00]> : tensor<4xf32>
    // CHECK-NOT:   const.Declare
    %2 = IE.Interpolate(%arg0, %0, %1) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, operandSegmentSizes = array<i32: 1, 1, 1, 0>} : tensor<1x3x10x10xf32>, tensor<4xsi64>, tensor<4xf32> -> tensor<1x3x20x15xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME: axes_attr = [2, 3],
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK-SAME: scales_attr = [2.000000e+00, 1.500000e+00],
    // CHECK-SAME: sizes_attr = [20, 15]}
    // CHECK-SAME: tensor<1x3x10x10xf32> -> tensor<1x3x20x15xf32>

    return %2 : tensor<1x3x20x15xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @InferOutputShapeWithFloatScales
func.func @InferOutputShapeWithFloatScales(%arg0: tensor<1x128x3x3xf32>) -> tensor<1x128x5x5xf32> {
    %0 = const.Declare tensor<4xsi32> = dense<1> : tensor<4xsi32>
    %1 = const.Declare tensor<4xf32> = dense<[1.000000e+00, 1.000000e+00, 1.6666666269302368, 1.6666666269302368]> : tensor<4xf32>
    %2 = IE.Interpolate(%arg0, %0, %1) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, operandSegmentSizes = array<i32: 1, 1, 1, 0>} : tensor<1x128x3x3xf32>, tensor<4xsi32>, tensor<4xf32> -> tensor<1x128x5x5xf32>

    return %2 : tensor<1x128x5x5xf32>

    // CHECK:       [[INTERP:%.*]] = IE.Interpolate(%arg0)
    // CHECK-SAME:      mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>
    // CHECK-SAME:      antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:      axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.6666666666666667, 1.6666666666666667], sizes_attr = [5, 5]} : tensor<1x128x3x3xf32> -> tensor<1x128x5x5xf32>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @Fold
func.func @Fold(%arg0: tensor<1x3x512x512xf16>) -> tensor<1x3x512x512xf16> {
        %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [512, 512]
         } : tensor<1x3x512x512xf16> -> tensor<1x3x512x512xf16>

    return %0 : tensor<1x3x512x512xf16>

    // CHECK-NOT:    IE.Interpolate
    // CHECK:       return %arg0 : tensor<1x3x512x512xf16>
}

// -----

// CHECK-LABEL: @FoldCubicInterpolateWithSplatConstInput
func.func @FoldCubicInterpolateWithSplatConstInput() -> tensor<1x384x27x27xf16> {
        %cst = const.Declare tensor<1x384x37x37xf16> = dense<1.0> : tensor<1x384x37x37xf16>
        %0 = IE.Interpolate(%cst)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <CUBIC>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [0.72972972972972971, 0.72972972972972971], sizes_attr = [27, 27]
         } : tensor<1x384x37x37xf16> -> tensor<1x384x27x27xf16>

    return %0 : tensor<1x384x27x27xf16>

    // CHECK-NOT:    IE.Interpolate
    // CHECK:       return %cst : tensor<1x384x27x27xf16>
}

// -----

// CHECK-LABEL: @FoldCubicInterpolateWithNonSplatConstInput
func.func @FoldCubicInterpolateWithNonSplatConstInput() -> tensor<1x1x3x3xf16> {
        %cst = const.Declare tensor<1x1x2x2xf16> = dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf16>
        %0 = IE.Interpolate(%cst)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <CUBIC>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.5, 1.5], sizes_attr = [3, 3]
         } : tensor<1x1x2x2xf16> -> tensor<1x1x3x3xf16>

    return %0 : tensor<1x1x3x3xf16>

    // CHECK-NOT:    IE.Interpolate
    // CHECK:       return %cst : tensor<1x1x3x3xf16>
}

// -----

// CHECK-LABEL: @ConvertToNearestWithSIZESMode
func.func @ConvertToNearestWithSIZESMode(%arg0: tensor<1x96x1x1xf32>) -> tensor<1x96x33x33xf32> {
    %cst = const.Declare tensor<2xsi64> = dense<33> : tensor<2xsi64>
    %cst_0 = const.Declare tensor<2xf32> = dense<3.300000e+01> : tensor<2xf32>
    %cst_1 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    %0 = IE.Interpolate(%arg0, %cst, %cst_0, %cst_1) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<1x96x1x1xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x96x33x33xf32>
    return %0 : tensor<1x96x33x33xf32>

    // CHECK:       [[INTERP:%.*]] = IE.Interpolate(%arg0)
    // CHECK-SAME:      mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>
    // CHECK-SAME:      antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:      axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [3.300000e+01, 3.300000e+01], sizes_attr = [33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @NotConvertToNearestWithSIZESMode
func.func @NotConvertToNearestWithSIZESMode(%arg0: tensor<1x96x3x1xf32>) -> tensor<1x96x33x33xf32> {
    %cst = const.Declare tensor<2xsi64> = dense<33> : tensor<2xsi64>
    %cst_0 = const.Declare tensor<2xf32> = dense<[1.100000e+01, 3.300000e+01]> : tensor<2xf32>
    %cst_1 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    %0 = IE.Interpolate(%arg0, %cst, %cst_0, %cst_1) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<1x96x3x1xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x96x33x33xf32>
    return %0 : tensor<1x96x33x33xf32>

    // CHECK:       [[INTERP:%.*]] = IE.Interpolate(%arg0)
    // CHECK-SAME:      mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @ConvertToNearestWithSCALESMode
func.func @ConvertToNearestWithSCALESMode(%arg0: tensor<1x96x1x1xf32>) -> tensor<1x96x33x33xf32> {
    %cst = const.Declare tensor<2xsi64> = dense<33> : tensor<2xsi64>
    %cst_0 = const.Declare tensor<2xf32> = dense<3.300000e+01> : tensor<2xf32>
    %cst_1 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    %0 = IE.Interpolate(%arg0, %cst, %cst_0, %cst_1) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<1x96x1x1xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x96x33x33xf32>
    return %0 : tensor<1x96x33x33xf32>

    // CHECK:       [[INTERP:%.*]] = IE.Interpolate(%arg0)
    // CHECK-SAME:      mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>
    // CHECK-SAME:      antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:      axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [3.300000e+01, 3.300000e+01], sizes_attr = [33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @NotConvertToNearestWithSCALESMode
func.func @NotConvertToNearestWithSCALESMode(%arg0: tensor<1x96x3x1xf32>) -> tensor<1x96x33x33xf32> {
    %cst = const.Declare tensor<2xsi64> = dense<33> : tensor<2xsi64>
    %cst_0 = const.Declare tensor<2xf32> = dense<[1.100000e+01, 3.300000e+01]> : tensor<2xf32>
    %cst_1 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    %0 = IE.Interpolate(%arg0, %cst, %cst_0, %cst_1) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<1x96x3x1xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x96x33x33xf32>
    return %0 : tensor<1x96x33x33xf32>

    // CHECK:       [[INTERP:%.*]] = IE.Interpolate(%arg0)
    // CHECK-SAME:      mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>
    // CHECK-SAME:      antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:      axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.100000e+01, 3.300000e+01], sizes_attr = [33, 33]} : tensor<1x96x3x1xf32> -> tensor<1x96x33x33xf32>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @ConvertHalfPixelToAsymmetric
func.func @ConvertHalfPixelToAsymmetric(%arg0: tensor<1x3x160x160xf32>) -> tensor<1x3x320x320xf32> {
    %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_CEIL>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [320, 320]} : tensor<1x3x160x160xf32> -> tensor<1x3x320x320xf32>
    return %0 : tensor<1x3x320x320xf32>

    // CHECK:       [[INTERP:%.*]] = IE.Interpolate({{[^:]+}})
    // CHECK-SAME:      {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK-SAME:      scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [320, 320]} : tensor<1x3x160x160xf32> -> tensor<1x3x320x320xf32>

    // CHECK:       return [[INTERP]] : tensor<1x3x320x320xf32>

}
