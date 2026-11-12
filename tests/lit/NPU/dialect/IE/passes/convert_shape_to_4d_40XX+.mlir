//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-shape-to-4d --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @Convert2dTopKPositiveAxis
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<80x77xsi32>
func.func @Convert2dTopKPositiveAxis(%arg0: tensor<80x77xsi32>) -> (tensor<80x1xsi32>, tensor<80x1xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<80x77xsi32>, tensor<si32> -> tensor<80x1xsi32>, tensor<80x1xsi32>

    return %output_values, %target_shape : tensor<80x1xsi32>, tensor<80x1xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG_0]]) {
    // CHECK-SAME:         shape_value = [1, 1, 80, 77]
    // CHECK-SAME:     } : tensor<80x77xsi32> -> tensor<1x1x80x77xsi32>

    // CHECK:   [[VALUE:%.+]], [[SHAPE:%.+]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>}
    // CHECK-SAME:         tensor<1x1x80x77xsi32> -> tensor<1x1x80x1xsi32>, tensor<1x1x80x1xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.+]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:                    } : tensor<1x1x80x1xsi32> -> tensor<80x1xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.+]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:                    } : tensor<1x1x80x1xsi32> -> tensor<80x1xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]

}

// -----

// CHECK-LABEL: @Convert2dTopKNegativeAxis
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<80x77xsi32>
func.func @Convert2dTopKNegativeAxis(%arg0: tensor<80x77xsi32>) -> (tensor<1x77xsi32>, tensor<1x77xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = -2 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<80x77xsi32>, tensor<si32> -> tensor<1x77xsi32>, tensor<1x77xsi32>

    return %output_values, %target_shape : tensor<1x77xsi32>, tensor<1x77xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG_0]]) {
    // CHECK-SAME:         shape_value = [1, 1, 80, 77]
    // CHECK-SAME:     } : tensor<80x77xsi32> -> tensor<1x1x80x77xsi32>

    // CHECK:   [[VALUE:%.+]], [[SHAPE:%.+]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 2 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
    // CHECK-SAME:         tensor<1x1x80x77xsi32> -> tensor<1x1x1x77xsi32>, tensor<1x1x1x77xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.+]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:                    } : tensor<1x1x1x77xsi32> -> tensor<1x77xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.+]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:                    } : tensor<1x1x1x77xsi32> -> tensor<1x77xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}

// -----

// CHECK-LABEL: @Convert3dTopKPositiveAxis
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<60x80x77xsi32>
func.func @Convert3dTopKPositiveAxis(%arg0: tensor<60x80x77xsi32>) -> (tensor<60x1x77xsi32>, tensor<60x1x77xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<60x80x77xsi32>, tensor<si32> -> tensor<60x1x77xsi32>, tensor<60x1x77xsi32>

    return %output_values, %target_shape : tensor<60x1x77xsi32>, tensor<60x1x77xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG_0]]) {
    // CHECK-SAME{LITERAL}:   shape_value = [1, 60, 80, 77]} : tensor<60x80x77xsi32> -> tensor<1x60x80x77xsi32>

    // CHECK:   [[VALUE:%.+]], [[SHAPE:%.+]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 2 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
    // CHECK-SAME:         tensor<1x60x80x77xsi32> -> tensor<1x60x1x77xsi32>, tensor<1x60x1x77xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.+]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:         shape_value = [60, 1, 77]} : tensor<1x60x1x77xsi32> -> tensor<60x1x77xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.+]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:         shape_value = [60, 1, 77]} : tensor<1x60x1x77xsi32> -> tensor<60x1x77xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}

// -----

// CHECK-LABEL: @Convert3dTopKFirstAxis
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<60x80x77xsi32>
func.func @Convert3dTopKFirstAxis(%arg0: tensor<60x80x77xsi32>) -> (tensor<1x80x77xsi32>, tensor<1x80x77xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 0 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<60x80x77xsi32>, tensor<si32> -> tensor<1x80x77xsi32>, tensor<1x80x77xsi32>

    return %output_values, %target_shape : tensor<1x80x77xsi32>, tensor<1x80x77xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG_0]]) {
    // CHECK-SAME{LITERAL}:   shape_value = [1, 1, 60, 6160]} : tensor<60x80x77xsi32> -> tensor<1x1x60x6160xsi32>

    // CHECK:   [[VALUE:%.+]], [[SHAPE:%.+]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 2 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
    // CHECK-SAME:         tensor<1x1x60x6160xsi32> -> tensor<1x1x1x6160xsi32>, tensor<1x1x1x6160xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.+]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:         shape_value = [1, 80, 77]} : tensor<1x1x1x6160xsi32> -> tensor<1x80x77xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.+]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:         shape_value = [1, 80, 77]} : tensor<1x1x1x6160xsi32> -> tensor<1x80x77xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}

// -----

// CHECK-LABEL: @Convert3dTopKLastAxis
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<60x80x77xsi32>
func.func @Convert3dTopKLastAxis(%arg0: tensor<60x80x77xsi32>) -> (tensor<60x80x1xsi32>, tensor<60x80x1xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 2 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<60x80x77xsi32>, tensor<si32> -> tensor<60x80x1xsi32>, tensor<60x80x1xsi32>

    return %output_values, %target_shape : tensor<60x80x1xsi32>, tensor<60x80x1xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.Reshape([[ARG_0]]) {
    // CHECK-SAME{LITERAL}:   shape_value = [1, 1, 4800, 77]} : tensor<60x80x77xsi32> -> tensor<1x1x4800x77xsi32>

    // CHECK:   [[VALUE:%.+]], [[SHAPE:%.+]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
    // CHECK-SAME:         tensor<1x1x4800x77xsi32> -> tensor<1x1x4800x1xsi32>, tensor<1x1x4800x1xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.+]] = IE.Reshape([[VALUE]]) {
    // CHECK-SAME:         shape_value = [60, 80, 1]} : tensor<1x1x4800x1xsi32> -> tensor<60x80x1xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.+]] = IE.Reshape([[SHAPE]]) {
    // CHECK-SAME:         shape_value = [60, 80, 1]} : tensor<1x1x4800x1xsi32> -> tensor<60x80x1xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}

// -----

// CHECK-LABEL: @RMSNorm
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x32x6xf16>
func.func @RMSNorm(%arg0: tensor<1x32x6xf16>) -> tensor<1x32x6xf16> {
  %cst = const.Declare tensor<1x1x6xf16> = dense<1.000000e+00> : tensor<1x1x6xf16>
  %0 = IE.RMS(%arg0, %cst) {eps = 9.9999997473787516E-6 : f64} : tensor<1x32x6xf16>, tensor<1x1x6xf16> -> tensor<1x32x6xf16>
  return %0 : tensor<1x32x6xf16>

    // CHECK:           [[CST:%.+]] = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00> : tensor<1x1x6xf16>, [#const.Reshape<[1, 1, 1, 6]>]
    // CHECK:           [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 32, 6]} : tensor<1x32x6xf16> -> tensor<1x1x32x6xf16>
    // CHECK:           [[RMS:%.+]] = IE.RMS([[RESHAPE_IN]], [[CST]]) {eps = 9.9999997473787516E-6 : f64} : tensor<1x1x32x6xf16>, tensor<1x1x1x6xf16> -> tensor<1x1x32x6xf16>
    // CHECK:           [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[RMS]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 32, 6]} : tensor<1x1x32x6xf16> -> tensor<1x32x6xf16>
    // CHECK:           return [[RESHAPE_OUT]] : tensor<1x32x6xf16>
}

// -----

// CHECK-LABEL: @RMSNorm5D
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x3x1x256x1152xf16>
func.func @RMSNorm5D(%arg0: tensor<1x3x1x256x1152xf16>) -> tensor<1x3x1x256x1152xf16> {
  %cst = const.Declare tensor<1152xf16> = dense<1.000000e+00> : tensor<1152xf16>
  %0 = IE.RMS(%arg0, %cst) {eps = 9.9999997473787516E-6 : f64} : tensor<1x3x1x256x1152xf16>, tensor<1152xf16> -> tensor<1x3x1x256x1152xf16>
  return %0 : tensor<1x3x1x256x1152xf16>

    // CHECK:           [[CST:%.+]] = const.Declare tensor<1x1x1x1152xf16> = dense<1.000000e+00> : tensor<1152xf16>, [#const.Reshape<[1, 1, 1, 1152]>]
    // CHECK:           [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 3, 256, 1152]} : tensor<1x3x1x256x1152xf16> -> tensor<1x3x256x1152xf16>
    // CHECK:           [[RMS:%.+]] = IE.RMS([[RESHAPE_IN]], [[CST]]) {eps = 9.9999997473787516E-6 : f64} : tensor<1x3x256x1152xf16>, tensor<1x1x1x1152xf16> -> tensor<1x3x256x1152xf16>
    // CHECK:           [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[RMS]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 3, 1, 256, 1152]} : tensor<1x3x256x1152xf16> -> tensor<1x3x1x256x1152xf16>
    // CHECK:           return [[RESHAPE_OUT]] : tensor<1x3x1x256x1152xf16>
}

// -----

// Outer dims are split into DimC and DimH with sizes as close as possible: 2x3x4 -> 24, 64 -> 64.

// CHECK-LABEL: @RMSNorm5DBalancedSplit
// CHECK-SAME:    [[ARG0:%.+]]: tensor<2x3x4x64x1152xf16>
func.func @RMSNorm5DBalancedSplit(%arg0: tensor<2x3x4x64x1152xf16>) -> tensor<2x3x4x64x1152xf16> {
  %cst = const.Declare tensor<1152xf16> = dense<1.000000e+00> : tensor<1152xf16>
  %0 = IE.RMS(%arg0, %cst) {eps = 9.9999997473787516E-6 : f64} : tensor<2x3x4x64x1152xf16>, tensor<1152xf16> -> tensor<2x3x4x64x1152xf16>
  return %0 : tensor<2x3x4x64x1152xf16>

    // CHECK:           [[CST:%.+]] = const.Declare tensor<1x1x1x1152xf16> = dense<1.000000e+00> : tensor<1152xf16>, [#const.Reshape<[1, 1, 1, 1152]>]
    // CHECK:           [[RESHAPE_IN:%.+]] = IE.Reshape([[ARG0]]) {
    // CHECK-SAME{LITERAL}:     shape_value = [1, 24, 64, 1152]} : tensor<2x3x4x64x1152xf16> -> tensor<1x24x64x1152xf16>
    // CHECK:           [[RMS:%.+]] = IE.RMS([[RESHAPE_IN]], [[CST]]) {eps = 9.9999997473787516E-6 : f64} : tensor<1x24x64x1152xf16>, tensor<1x1x1x1152xf16> -> tensor<1x24x64x1152xf16>
    // CHECK:           [[RESHAPE_OUT:%.+]] = IE.Reshape([[RMS]]) {
    // CHECK-SAME{LITERAL}:     shape_value = [2, 3, 4, 64, 1152]} : tensor<1x24x64x1152xf16> -> tensor<2x3x4x64x1152xf16>
    // CHECK:           return [[RESHAPE_OUT]] : tensor<2x3x4x64x1152xf16>
}

// -----

// CHECK-LABEL: @DynamicGelu1DTo4D
// CHECK-SAME:      [[INPUT:%.+]]: tensor<?xf16, {bounds = #const.OpaqueI64Elements<[256]> : tensor<1xsi64>, order = #C}>
func.func @DynamicGelu1DTo4D(%arg0: tensor<?xf16, {bounds = #const.OpaqueI64Elements<[256]> : tensor<1xsi64>, order = affine_map<(d0) -> (d0)>}>) -> tensor<?xf16, {bounds = #const.OpaqueI64Elements<[256]> : tensor<1xsi64>, order = affine_map<(d0) -> (d0)>}> {
    %cst = const.Declare tensor<4xsi32> = dense<[1, 1, 1, -1]> : tensor<4xsi32>
    %0 = IE.DynamicReshape(%arg0, %cst) {output_bounds = [1, 1, 1, 256], output_shape = [1, 1, 1, -9223372036854775808]} : tensor<?xf16, {bounds = #const.OpaqueI64Elements<[256]> : tensor<1xsi64>, order = affine_map<(d0) -> (d0)>}>, tensor<4xsi32> -> tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 256]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    %1 = IE.Gelu(%0) : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 256]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> -> tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 256]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    %cst_0 = const.Declare tensor<1xsi32> = dense<-1> : tensor<1xsi32>
    %2 = IE.DynamicReshape(%1, %cst_0) {output_bounds = [256], output_shape = [-9223372036854775808]} : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 256]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>, tensor<1xsi32> -> tensor<?xf16, {bounds = #const.OpaqueI64Elements<[256]> : tensor<1xsi64>, order = affine_map<(d0) -> (d0)>}>
    return %2 : tensor<?xf16, {bounds = #const.OpaqueI64Elements<[256]> : tensor<1xsi64>, order = affine_map<(d0) -> (d0)>}>

    // CHECK-DAG:   [[SHAPE_TO_4D:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 1, 1, -1]> : tensor<4xsi32>
    // CHECK-DAG:   [[SHAPE_TO_1D:%.+]] = const.Declare tensor<1xsi32> = dense<-1> : tensor<1xsi32>

    // CHECK:       [[RESHAPE_TO_4D:%.+]] = IE.DynamicReshape([[INPUT]], [[SHAPE_TO_4D]]) {output_bounds = [1, 1, 1, 256], output_shape = [1, 1, 1, -9223372036854775808]} : tensor<?xf16, {bounds = #const.OpaqueI64Elements<[256]> : tensor<1xsi64>, order = #C}>, tensor<4xsi32> -> tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 256]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[GELU:%.+]] = IE.Gelu([[RESHAPE_TO_4D]]) : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 256]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 256]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[RESHAPE_TO_1D:%.+]] = IE.DynamicReshape([[GELU]], [[SHAPE_TO_1D]]) {output_bounds = [256], output_shape = [-9223372036854775808]} : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 256]> : tensor<4xsi64>, order = #NCHW}>, tensor<1xsi32> -> tensor<?xf16, {bounds = #const.OpaqueI64Elements<[256]> : tensor<1xsi64>, order = #C}>

    // CHECK:       return [[RESHAPE_TO_1D]] : tensor<?xf16, {bounds = #const.OpaqueI64Elements<[256]> : tensor<1xsi64>, order = #C}>
}

// -----

// CHECK-LABEL: @DynamicGelu2DTo4D
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = #NC}>
func.func @DynamicGelu2DTo4D(%arg0: tensor<1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = affine_map<(d0, d1) -> (d0, d1)>}>) -> tensor<1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = affine_map<(d0, d1) -> (d0, d1)>}> {
    %cst = const.Declare tensor<4xsi32> = dense<[1, 1, 1, -1]> : tensor<4xsi32>
    %0 = IE.DynamicReshape(%arg0, %cst) {output_bounds = [1, 1, 1, 64], output_shape = [1, 1, 1, -9223372036854775808]} : tensor<1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = affine_map<(d0, d1) -> (d0, d1)>}>, tensor<4xsi32> -> tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    %1 = IE.Gelu(%0) : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> -> tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    %cst_0 = const.Declare tensor<2xsi32> = dense<[1, -1]> : tensor<2xsi32>
    %2 = IE.DynamicReshape(%1, %cst_0) {output_bounds = [1, 64], output_shape = [1, -9223372036854775808]} : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>, tensor<2xsi32> -> tensor<1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = affine_map<(d0, d1) -> (d0, d1)>}>
    return %2 : tensor<1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = affine_map<(d0, d1) -> (d0, d1)>}>

    // CHECK-DAG:   [[SHAPE_TO_4D:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 1, 1, -1]> : tensor<4xsi32>
    // CHECK-DAG:   [[SHAPE_TO_2D:%.+]] = const.Declare tensor<2xsi32> = dense<[1, -1]> : tensor<2xsi32>

    // CHECK:       [[RESHAPE_TO_4D:%.+]] = IE.DynamicReshape([[INPUT]], [[SHAPE_TO_4D]]) {output_bounds = [1, 1, 1, 64], output_shape = [1, 1, 1, -9223372036854775808]} : tensor<1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = #NC}>, tensor<4xsi32> -> tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[GELU:%.+]] = IE.Gelu([[RESHAPE_TO_4D]]) : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[RESHAPE_TO_2D:%.+]] = IE.DynamicReshape([[GELU]], [[SHAPE_TO_2D]]) {output_bounds = [1, 64], output_shape = [1, -9223372036854775808]} : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<2xsi32> -> tensor<1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = #NC}>

    // CHECK:       return [[RESHAPE_TO_2D]] : tensor<1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64]> : tensor<2xsi64>, order = #NC}>
}

// -----

// CHECK-LABEL: @DynamicGelu3DTo4D
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 3072]> : tensor<3xsi64>, order = #CHW}>
func.func @DynamicGelu3DTo4D(%arg0: tensor<1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 3072]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>) -> tensor<1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 3072]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> {
    %cst = const.Declare tensor<4xsi32> = dense<[1, 1, -1, 3072]> : tensor<4xsi32>
    %0 = IE.DynamicReshape(%arg0, %cst) {output_bounds = [1, 1, 8, 3072], output_shape = [1, 1, -9223372036854775808, 3072]} : tensor<1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 3072]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>, tensor<4xsi32> -> tensor<1x1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 8, 3072]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    %1 = IE.Gelu(%0) : tensor<1x1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 8, 3072]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> -> tensor<1x1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 8, 3072]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    %cst_0 = const.Declare tensor<3xsi32> = dense<[1, -1, 3072]> : tensor<3xsi32>
    %2 = IE.DynamicReshape(%1, %cst_0) {output_bounds = [1, 8, 3072], output_shape = [1, -9223372036854775808, 3072]} : tensor<1x1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 8, 3072]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>, tensor<3xsi32> -> tensor<1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 3072]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
    return %2 : tensor<1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 3072]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>

    // CHECK-DAG:   [[SHAPE_TO_4D:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 1, -1, 3072]> : tensor<4xsi32>
    // CHECK-DAG:   [[SHAPE_TO_3D:%.+]] = const.Declare tensor<3xsi32> = dense<[1, -1, 3072]> : tensor<3xsi32>

    // CHECK:       [[RESHAPE_TO_4D:%.+]] = IE.DynamicReshape([[INPUT]], [[SHAPE_TO_4D]]) {output_bounds = [1, 1, 8, 3072], output_shape = [1, 1, -9223372036854775808, 3072]} : tensor<1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 3072]> : tensor<3xsi64>, order = #CHW}>, tensor<4xsi32> -> tensor<1x1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 8, 3072]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[GELU:%.+]] = IE.Gelu([[RESHAPE_TO_4D]]) : tensor<1x1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 8, 3072]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 8, 3072]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[RESHAPE_TO_3D:%.+]] = IE.DynamicReshape([[GELU]], [[SHAPE_TO_3D]]) {output_bounds = [1, 8, 3072], output_shape = [1, -9223372036854775808, 3072]} : tensor<1x1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 8, 3072]> : tensor<4xsi64>, order = #NCHW}>, tensor<3xsi32> -> tensor<1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 3072]> : tensor<3xsi64>, order = #CHW}>

    // CHECK:       return [[RESHAPE_TO_3D]] : tensor<1x?x3072xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 3072]> : tensor<3xsi64>, order = #CHW}>
}

// -----

// An unbounded shape of rank > 4 cannot be reified to merge its axes, so it is kept unchanged.

// CHECK-LABEL: @UnboundedGelu5DKept
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x?x4x8xf16>
func.func @UnboundedGelu5DKept(%arg0: tensor<1x2x?x4x8xf16>) -> tensor<1x2x?x4x8xf16> {
    %0 = IE.Gelu(%arg0) : tensor<1x2x?x4x8xf16> -> tensor<1x2x?x4x8xf16>
    return %0 : tensor<1x2x?x4x8xf16>

    // CHECK:       [[GELU:%.+]] = IE.Gelu([[INPUT]]) : tensor<1x2x?x4x8xf16> -> tensor<1x2x?x4x8xf16>
    // CHECK:       return [[GELU]] : tensor<1x2x?x4x8xf16>
}

// -----

// An unbounded shape of rank < 4 is extended to 4D, keeping the unbounded dim.

// CHECK-LABEL: @UnboundedGelu3DTo4D
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x?x64xf16>
func.func @UnboundedGelu3DTo4D(%arg0: tensor<1x?x64xf16>) -> tensor<1x?x64xf16> {
    %0 = IE.Gelu(%arg0) : tensor<1x?x64xf16> -> tensor<1x?x64xf16>
    return %0 : tensor<1x?x64xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, -9223372036854775808, 64]} : tensor<1x?x64xf16> -> tensor<1x1x?x64xf16>
    // CHECK:       [[GELU:%.+]] = IE.Gelu([[RESHAPE_IN]]) : tensor<1x1x?x64xf16> -> tensor<1x1x?x64xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[GELU]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, -9223372036854775808, 64]} : tensor<1x1x?x64xf16> -> tensor<1x?x64xf16>
    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x?x64xf16>
}

// -----

// An unbounded eltwise operand of rank < 4 is extended to 4D without merging axes.

// CHECK-LABEL: @UnboundedMultiply3DTo4D
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x?x64xf16>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x1x64xf16>
func.func @UnboundedMultiply3DTo4D(%arg0: tensor<1x?x64xf16>, %arg1: tensor<1x1x64xf16>) -> tensor<1x?x64xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x?x64xf16>, tensor<1x1x64xf16> -> tensor<1x?x64xf16>
    return %0 : tensor<1x?x64xf16>

    // CHECK-DAG:   [[RESHAPE_IN2:%.+]] = IE.AffineReshape([[INPUT2]]) {{.*}} : tensor<1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK-DAG:   [[RESHAPE_IN1:%.+]] = IE.AffineReshape([[INPUT1]]) {{.*}} : tensor<1x?x64xf16> -> tensor<1x1x?x64xf16>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[RESHAPE_IN1]], [[RESHAPE_IN2]])
    // CHECK-SAME:      : tensor<1x1x?x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x?x64xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MULTIPLY]]) {{.*}} : tensor<1x1x?x64xf16> -> tensor<1x?x64xf16>
    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x?x64xf16>
}

// -----

// An unbounded eltwise operand of rank > 4 cannot be reified to merge its axes, so the op is kept unchanged.

// CHECK-LABEL: @UnboundedMultiply5DKept
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x2x?x4x8xf16>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x2x1x4x8xf16>
func.func @UnboundedMultiply5DKept(%arg0: tensor<1x2x?x4x8xf16>, %arg1: tensor<1x2x1x4x8xf16>) -> tensor<1x2x?x4x8xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x?x4x8xf16>, tensor<1x2x1x4x8xf16> -> tensor<1x2x?x4x8xf16>
    return %0 : tensor<1x2x?x4x8xf16>

    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT1]], [[INPUT2]])
    // CHECK-SAME:      : tensor<1x2x?x4x8xf16>, tensor<1x2x1x4x8xf16> -> tensor<1x2x?x4x8xf16>
    // CHECK:       return [[MULTIPLY]] : tensor<1x2x?x4x8xf16>
}

// -----

// An unbounded FakeQuantize input of rank < 4 is extended to 4D without merging axes.

// CHECK-LABEL: @UnboundedFakeQuantize3DTo4D
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x?x64xf16>
func.func @UnboundedFakeQuantize3DTo4D(%arg0: tensor<1x?x64xf16>) -> tensor<1x?x64xf16> {
    %low = const.Declare tensor<1x1x1xf16> = dense<0.0> : tensor<1x1x1xf16>
    %high = const.Declare tensor<1x1x1xf16> = dense<255.0> : tensor<1x1x1xf16>
    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256} : tensor<1x?x64xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<1x?x64xf16>
    return %0 : tensor<1x?x64xf16>

    // CHECK-DAG:   [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG:   [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]]) {{.*}} : tensor<1x?x64xf16> -> tensor<1x1x?x64xf16>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[RESHAPE_IN]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:      -> tensor<1x1x?x64xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[FQ]]) {{.*}} : tensor<1x1x?x64xf16> -> tensor<1x?x64xf16>
    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x?x64xf16>
}

// -----

// An unbounded FakeQuantize input of rank > 4 cannot be reified to merge its axes, so the op is kept unchanged.

// CHECK-LABEL: @UnboundedFakeQuantize5DKept
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x?x4x8xf16>
func.func @UnboundedFakeQuantize5DKept(%arg0: tensor<1x2x?x4x8xf16>) -> tensor<1x2x?x4x8xf16> {
    %low = const.Declare tensor<1x1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1x1xf16>
    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256} : tensor<1x2x?x4x8xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x2x?x4x8xf16>
    return %0 : tensor<1x2x?x4x8xf16>

    // CHECK-DAG:   [[LOW:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<0.000000e+00>
    // CHECK-DAG:   [[HIGH:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<2.550000e+02>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:      -> tensor<1x2x?x4x8xf16>
    // CHECK:       return [[FQ]] : tensor<1x2x?x4x8xf16>
}
