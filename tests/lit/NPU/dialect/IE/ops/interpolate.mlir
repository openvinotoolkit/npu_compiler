//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ConvertConstToAttr
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x10x10xf32>)
func.func @ConvertConstToAttr(%arg0: tensor<1x3x10x10xf32>) -> tensor<1x3x20x15xf32> {
    %0 = const.Declare tensor<2xsi64> = dense<[20, 15]> : tensor<2xsi64>
    %1 = const.Declare tensor<2xf32>  = dense<[2.000000e+00, 1.500000e+00]> : tensor<2xf32>
    %2 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    // CHECK-NOT:   const.Declare
    %3 = IE.Interpolate(%arg0, %0, %1, %2) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<1x3x10x10xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x3x20x15xf32>
    // CHECK:       [[VAL0:%.+]] = IE.Interpolate([[ARG_0]]) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME: axes_attr = [2, 3],
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK-SAME: scales_attr = [2.000000e+00, 1.500000e+00],
    // CHECK-SAME: sizes_attr = [20, 15]}
    // CHECK-SAME: tensor<1x3x10x10xf32> -> tensor<1x3x20x15xf32>

    return %3 : tensor<1x3x20x15xf32>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertConstToAttr3InputsSizes
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x10x10xf32>)
func.func @ConvertConstToAttr3InputsSizes(%arg0: tensor<1x3x10x10xf32>) -> tensor<1x3x20x15xf32> {
    %0 = const.Declare tensor<4xsi64> = dense<[1, 3, 20, 15]> : tensor<4xsi64>
    %1 = const.Declare tensor<4xf32>  = dense<[1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>
    // CHECK-NOT:   const.Declare
    %2 = IE.Interpolate(%arg0, %0, %1) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, operandSegmentSizes = array<i32: 1, 1, 1, 0>} : tensor<1x3x10x10xf32>, tensor<4xsi64>, tensor<4xf32> -> tensor<1x3x20x15xf32>
    // CHECK:       [[VAL0:%.+]] = IE.Interpolate([[ARG_0]]) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME: axes_attr = [2, 3],
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK-SAME: scales_attr = [2.000000e+00, 1.500000e+00],
    // CHECK-SAME: sizes_attr = [20, 15]}
    // CHECK-SAME: tensor<1x3x10x10xf32> -> tensor<1x3x20x15xf32>

    return %2 : tensor<1x3x20x15xf32>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertConstToAttr3InputsScales
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x10x10xf32>)
func.func @ConvertConstToAttr3InputsScales(%arg0: tensor<1x3x10x10xf32>) -> tensor<1x3x20x15xf32> {
    %0 = const.Declare tensor<4xsi64> = dense<[1, 1, 1, 1]> : tensor<4xsi64>
    %1 = const.Declare tensor<4xf32>  = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 1.500000e+00]> : tensor<4xf32>
    // CHECK-NOT:   const.Declare
    %2 = IE.Interpolate(%arg0, %0, %1) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, operandSegmentSizes = array<i32: 1, 1, 1, 0>} : tensor<1x3x10x10xf32>, tensor<4xsi64>, tensor<4xf32> -> tensor<1x3x20x15xf32>
    // CHECK:       [[VAL0:%.+]] = IE.Interpolate([[ARG_0]]) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME: axes_attr = [2, 3],
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK-SAME: scales_attr = [2.000000e+00, 1.500000e+00],
    // CHECK-SAME: sizes_attr = [20, 15]}
    // CHECK-SAME: tensor<1x3x10x10xf32> -> tensor<1x3x20x15xf32>

    return %2 : tensor<1x3x20x15xf32>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @InferOutputShapeWithFloatScales
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x128x3x3xf32>)
func.func @InferOutputShapeWithFloatScales(%arg0: tensor<1x128x3x3xf32>) -> tensor<1x128x5x5xf32> {
    %0 = const.Declare tensor<4xsi32> = dense<1> : tensor<4xsi32>
    %1 = const.Declare tensor<4xf32> = dense<[1.000000e+00, 1.000000e+00, 1.6666666269302368, 1.6666666269302368]> : tensor<4xf32>
    %2 = IE.Interpolate(%arg0, %0, %1) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, operandSegmentSizes = array<i32: 1, 1, 1, 0>} : tensor<1x128x3x3xf32>, tensor<4xsi32>, tensor<4xf32> -> tensor<1x128x5x5xf32>

    return %2 : tensor<1x128x5x5xf32>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>
    // CHECK-SAME:      antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:      axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.6666666269302368, 1.6666666269302368], sizes_attr = [5, 5]} : tensor<1x128x3x3xf32> -> tensor<1x128x5x5xf32>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @Fold
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x512x512xf16>)
func.func @Fold(%arg0: tensor<1x3x512x512xf16>) -> tensor<1x3x512x512xf16> {
        %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [512, 512]
         } : tensor<1x3x512x512xf16> -> tensor<1x3x512x512xf16>

    return %0 : tensor<1x3x512x512xf16>

    // CHECK-NOT:    IE.Interpolate
    // CHECK:       return [[ARG_0]] : tensor<1x3x512x512xf16>
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

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x384x27x27xf16>
    // CHECK-NOT:    IE.Interpolate
    // CHECK:       return [[CST]] : tensor<1x384x27x27xf16>
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

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x3x3xf16>
    // CHECK-NOT:    IE.Interpolate
    // CHECK:       return [[CST]] : tensor<1x1x3x3xf16>
}

// -----

// CHECK-LABEL: @ConvertToNearestWithSIZESMode
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x96x1x1xf32>
func.func @ConvertToNearestWithSIZESMode(%arg0: tensor<1x96x1x1xf32>) -> tensor<1x96x33x33xf32> {
    %cst = const.Declare tensor<2xsi64> = dense<33> : tensor<2xsi64>
    %cst_0 = const.Declare tensor<2xf32> = dense<3.300000e+01> : tensor<2xf32>
    %cst_1 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    %0 = IE.Interpolate(%arg0, %cst, %cst_0, %cst_1) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<1x96x1x1xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x96x33x33xf32>
    return %0 : tensor<1x96x33x33xf32>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>
    // CHECK-SAME:      antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:      axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [3.300000e+01, 3.300000e+01], sizes_attr = [33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @NotConvertToNearestWithSIZESMode
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x96x3x1xf32>
func.func @NotConvertToNearestWithSIZESMode(%arg0: tensor<1x96x3x1xf32>) -> tensor<1x96x33x33xf32> {
    %cst = const.Declare tensor<2xsi64> = dense<33> : tensor<2xsi64>
    %cst_0 = const.Declare tensor<2xf32> = dense<[1.100000e+01, 3.300000e+01]> : tensor<2xf32>
    %cst_1 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    %0 = IE.Interpolate(%arg0, %cst, %cst_0, %cst_1) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<1x96x3x1xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x96x33x33xf32>
    return %0 : tensor<1x96x33x33xf32>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @ConvertToNearestWithSCALESMode
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x96x1x1xf32>
func.func @ConvertToNearestWithSCALESMode(%arg0: tensor<1x96x1x1xf32>) -> tensor<1x96x33x33xf32> {
    %cst = const.Declare tensor<2xsi64> = dense<33> : tensor<2xsi64>
    %cst_0 = const.Declare tensor<2xf32> = dense<3.300000e+01> : tensor<2xf32>
    %cst_1 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    %0 = IE.Interpolate(%arg0, %cst, %cst_0, %cst_1) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<1x96x1x1xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x96x33x33xf32>
    return %0 : tensor<1x96x33x33xf32>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>
    // CHECK-SAME:      antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:      axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [3.300000e+01, 3.300000e+01], sizes_attr = [33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @NotConvertToNearestWithSCALESMode
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x96x3x1xf32>
func.func @NotConvertToNearestWithSCALESMode(%arg0: tensor<1x96x3x1xf32>) -> tensor<1x96x33x33xf32> {
    %cst = const.Declare tensor<2xsi64> = dense<33> : tensor<2xsi64>
    %cst_0 = const.Declare tensor<2xf32> = dense<[1.100000e+01, 3.300000e+01]> : tensor<2xf32>
    %cst_1 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    %0 = IE.Interpolate(%arg0, %cst, %cst_0, %cst_1) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<1x96x3x1xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x96x33x33xf32>
    return %0 : tensor<1x96x33x33xf32>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>
    // CHECK-SAME:      antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:      axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.100000e+01, 3.300000e+01], sizes_attr = [33, 33]} : tensor<1x96x3x1xf32> -> tensor<1x96x33x33xf32>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @ConvertHalfPixelToAsymmetric
func.func @ConvertHalfPixelToAsymmetric(%arg0: tensor<1x3x160x160xf32>) -> tensor<1x3x320x320xf32> {
    %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_CEIL>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [320, 320]} : tensor<1x3x160x160xf32> -> tensor<1x3x320x320xf32>
    return %0 : tensor<1x3x320x320xf32>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate({{[^:]+}})
    // CHECK-SAME:      {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK-SAME:      scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [320, 320]} : tensor<1x3x160x160xf32> -> tensor<1x3x320x320xf32>

    // CHECK:       return [[INTERP]] : tensor<1x3x320x320xf32>

}

// -----

// CHECK-LABEL: @InterpolateDynamicShapeBounded
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
func.func @InterpolateDynamicShapeBounded(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}> {

    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [4.000000e+00, 4.000000e+00],
        sizes_attr = [400, 320]
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>
    // CHECK-SAME:      nearest_mode = <ROUND_PREFER_FLOOR>
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @InterpolateDynamicHalfPixelLinear
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
func.func @InterpolateDynamicHalfPixelLinear(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}> {

    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR>, nearest_mode = <FLOOR>,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [4.000000e+00, 4.000000e+00],
        sizes_attr = [400, 320]
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <LINEAR>
    // CHECK-SAME:      coord_mode = <HALF_PIXEL>
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @InterpolateDynamicAlignCorners
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
func.func @InterpolateDynamicAlignCorners(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}> {

    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR>, nearest_mode = <FLOOR>,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [4.000000e+00, 4.000000e+00],
        sizes_attr = [400, 320]
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <LINEAR>
    // CHECK-SAME:      coord_mode = <ALIGN_CORNERS>
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @InterpolateDynamicDownscale
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>
func.func @InterpolateDynamicDownscale(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}> {

    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [2.500000e-01, 2.500000e-01],
        sizes_attr = [100, 80]
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>
    // CHECK-SAME:      nearest_mode = <FLOOR>
    // CHECK-SAME:      scales_attr = [2.500000e-01, 2.500000e-01]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @InterpolateDynamicCubic
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
func.func @InterpolateDynamicCubic(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}> {

    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <CUBIC>, nearest_mode = <FLOOR>,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [4.000000e+00, 4.000000e+00],
        sizes_attr = [400, 320]
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <CUBIC>
    // CHECK-SAME:      coord_mode = <PYTORCH_HALF_PIXEL>
    // CHECK-SAME:      cube_coeff = -7.500000e-01
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @InterpolateDynamicTfHalfPixel
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
func.func @InterpolateDynamicTfHalfPixel(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}> {

    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <TF_HALF_PIXEL_FOR_NN>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <CEIL>,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [4.000000e+00, 4.000000e+00],
        sizes_attr = [400, 320]
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>
    // CHECK-SAME:      coord_mode = <TF_HALF_PIXEL_FOR_NN>
    // CHECK-SAME:      nearest_mode = <CEIL>
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @InterpolateDynamicSingleAxis
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
func.func @InterpolateDynamicSingleAxis(
        %arg0: tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>)
            -> tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 80]> : tensor<4xsi64>}> {

    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [4.000000e+00],
        sizes_attr = [400]
    } : tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
        -> tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 80]> : tensor<4xsi64>}>

    return %0 : tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 80]> : tensor<4xsi64>}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>
    // CHECK-SAME:      axes_attr = [2]
    // CHECK-SAME:      scales_attr = [4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 80]> : tensor<4xsi64>}>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @InterpolateDynamicHeightStaticWidth
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
func.func @InterpolateDynamicHeightStaticWidth(
        %arg0: tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>)
            -> tensor<1x32x?x320xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}> {

    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [4.000000e+00, 4.000000e+00],
        sizes_attr = [400, 320]
    } : tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
        -> tensor<1x32x?x320xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    return %0 : tensor<1x32x?x320xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>
    // CHECK-SAME:      axes_attr = [2, 3]
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x320xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @InterpolateDynamicLinearOnnx
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
func.func @InterpolateDynamicLinearOnnx(
        %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}> {

    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [4.000000e+00, 4.000000e+00],
        sizes_attr = [400, 320]
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <LINEAR_ONNX>
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>
    // CHECK:       return [[INTERP]]
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InterpolateDynamicConvertInputsToAttr
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x2x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 720, 1280]> : tensor<4xsi64>, order = #NCHW}>
func.func @InterpolateDynamicConvertInputsToAttr(%arg0: tensor<1x2x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 720, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x2x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 1440, 2560]> : tensor<4xsi64>, order = #NCHW}> {
    %cst_0 = const.Declare tensor<4xsi32> = dense<1> : tensor<4xsi32>
    %cst_1 = const.Declare tensor<4xf32> = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>

    %0 = IE.Interpolate(%arg0, %cst_0, %cst_1) {
        attr = #IE.Interpolate<mode = <NEAREST>,
        shape_calc_mode = <SCALES>,
        coord_mode = <ASYMMETRIC>,
        nearest_mode = <FLOOR>,
        antialias = false,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 0, 0, 0],
        cube_coeff = -7.500000e-01 : f64>,
        operandSegmentSizes = array<i32: 1, 1, 1, 0>
    } : tensor<1x2x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 720, 1280]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi32>, tensor<4xf32>
        -> tensor<1x2x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 1440, 2560]> : tensor<4xsi64>, order = #NCHW}>

    return %0 : tensor<1x2x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 1440, 2560]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-NOT:   const.Declare
    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>,
    // CHECK-SAME:      shape_calc_mode = <SCALES>,
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>,
    // CHECK-SAME:      nearest_mode = <FLOOR>,
    // CHECK-SAME:      antialias = false,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 0, 0],
    // CHECK-SAME:      cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:      axes_attr = [2, 3],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK-SAME:      scales_attr = [2.000000e+00, 2.000000e+00],
    // CHECK-SAME:      sizes_attr = [1440, 2560]
    // CHECK-SAME:      -> tensor<1x2x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2, 1440, 2560]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @ConvertAxesToAttrScaleAsParam
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x10x10xf16>, [[ARG_1:%[^:]+]]: tensor<2xf32>)
func.func @ConvertAxesToAttrScaleAsParam(%arg0: tensor<1x3x10x10xf16>, %arg1: tensor<2xf32>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 80, 80]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> {
    %0 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    %1 = IE.Interpolate(%arg0, %arg1, %0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        operandSegmentSizes = array<i32: 1, 0, 1, 1>
    } : tensor<1x3x10x10xf16>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 80, 80]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>

    return %1 : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 80, 80]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>

    // CHECK-NOT:   const.Declare
    // CHECK:       [[VAL0:%.+]] = IE.Interpolate([[ARG_0]], [[ARG_1]]) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:      axes_attr = [2, 3],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 0, 1, 0>, sizes_attr = []}
    // CHECK-SAME:      tensor<1x3x10x10xf16>, tensor<2xf32> -> tensor<1x3x?x?xf16
    // CHECK:       return [[VAL0]]
}
// -----

// CHECK-LABEL: @InterpolateDynamicChannelsAndHeight
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
func.func @InterpolateDynamicChannelsAndHeight(
    %arg0: tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>)
            -> tensor<1x128x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 400, 80]> : tensor<4xsi64>}> {

    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [1, 2],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [4.000000e+00, 4.000000e+00],
        sizes_attr = [128, 400]
    } : tensor<1x32x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
        -> tensor<1x128x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 400, 80]> : tensor<4xsi64>}>

    return %0 : tensor<1x128x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 400, 80]> : tensor<4xsi64>}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>
    // CHECK-SAME:      axes_attr = [1, 2]
    // CHECK-SAME:      scales_attr = [4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x128x?x80xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 400, 80]> : tensor<4xsi64>}>
    // CHECK:       return [[INTERP]]
}

// -----

// CHECK-LABEL: @InterpolateDynamicAllAxes
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
func.func @InterpolateDynamicAllAxes(
    %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}> {

    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [0, 1, 2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [1.000000e+00, 1.000000e+00, 4.000000e+00, 4.000000e+00],
        sizes_attr = [1, 32, 400, 320]
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]])
    // CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>
    // CHECK-SAME:      coord_mode = <ASYMMETRIC>
    // CHECK-SAME:      axes_attr = [0, 1, 2, 3]
    // CHECK-SAME:      scales_attr = [1.000000e+00, 1.000000e+00, 4.000000e+00, 4.000000e+00]
    // CHECK-SAME:      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>}>
    // CHECK:       return [[INTERP]]
}
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InterpolateLargeScaleAsParamInputCapsReturnTypeBound
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x1081x1920xf16>, [[ARG_1:%[^:]+]]: tensor<2xf16>)
func.func @InterpolateLargeScaleAsParamInputCapsReturnTypeBound(%arg0: tensor<1x3x1081x1920xf16>, %arg1: tensor<2xf16>)
        -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3838]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.Interpolate(%arg0, %arg1) {
        attr = #IE.Interpolate<mode = <LINEAR>,
                               shape_calc_mode = <SCALES>,
                               coord_mode = <HALF_PIXEL>,
                               nearest_mode = <FLOOR>,
                               antialias = false,
                               pads_begin = [0, 0, 0, 0],
                               pads_end = [0, 0, 0, 0],
                               cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        sizes_attr = []}
        : tensor<1x3x1081x1920xf16>, tensor<2xf16>
            -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3838]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3838]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[ARG_0]], [[ARG_1]])
    // CHECK-SAME:      axes_attr = [2, 3]
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 0, 1, 0>
    // CHECK-SAME:      -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3838]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       return [[INTERP]]
}
