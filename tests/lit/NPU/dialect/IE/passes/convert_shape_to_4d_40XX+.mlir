//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-shape-to-4d --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX

func.func @Convert2dTopKPositiveAxis(%arg0: tensor<80x77xsi32>) -> (tensor<80x1xsi32>, tensor<80x1xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<80x77xsi32>, tensor<si32> -> tensor<80x1xsi32>, tensor<80x1xsi32>

    return %output_values, %target_shape : tensor<80x1xsi32>, tensor<80x1xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:         shape_value = [1, 1, 80, 77]
    // CHECK-SAME:     } : tensor<80x77xsi32> -> tensor<1x1x80x77xsi32>

    // CHECK:   [[VALUE:%.*]], [[SHAPE:%.*]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>}
    // CHECK-SAME:         tensor<1x1x80x77xsi32> -> tensor<1x1x80x1xsi32>, tensor<1x1x80x1xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.*]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:                    } : tensor<1x1x80x1xsi32> -> tensor<80x1xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.*]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:                    } : tensor<1x1x80x1xsi32> -> tensor<80x1xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]

}

// -----

// CHECK-LABEL: @Convert2dTopKNegativeAxis
func.func @Convert2dTopKNegativeAxis(%arg0: tensor<80x77xsi32>) -> (tensor<1x77xsi32>, tensor<1x77xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = -2 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<80x77xsi32>, tensor<si32> -> tensor<1x77xsi32>, tensor<1x77xsi32>

    return %output_values, %target_shape : tensor<1x77xsi32>, tensor<1x77xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:         shape_value = [1, 1, 80, 77]
    // CHECK-SAME:     } : tensor<80x77xsi32> -> tensor<1x1x80x77xsi32>

    // CHECK:   [[VALUE:%.*]], [[SHAPE:%.*]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 2 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
    // CHECK-SAME:         tensor<1x1x80x77xsi32> -> tensor<1x1x1x77xsi32>, tensor<1x1x1x77xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.*]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:                    } : tensor<1x1x1x77xsi32> -> tensor<1x77xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.*]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:                    } : tensor<1x1x1x77xsi32> -> tensor<1x77xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}

// -----

// CHECK-LABEL: @Convert3dTopKPositiveAxis
func.func @Convert3dTopKPositiveAxis(%arg0: tensor<60x80x77xsi32>) -> (tensor<60x1x77xsi32>, tensor<60x1x77xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<60x80x77xsi32>, tensor<si32> -> tensor<60x1x77xsi32>, tensor<60x1x77xsi32>

    return %output_values, %target_shape : tensor<60x1x77xsi32>, tensor<60x1x77xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}:   shape_value = [1, 60, 80, 77]} : tensor<60x80x77xsi32> -> tensor<1x60x80x77xsi32>

    // CHECK:   [[VALUE:%.*]], [[SHAPE:%.*]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 2 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
    // CHECK-SAME:         tensor<1x60x80x77xsi32> -> tensor<1x60x1x77xsi32>, tensor<1x60x1x77xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.*]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:         shape_value = [60, 1, 77]} : tensor<1x60x1x77xsi32> -> tensor<60x1x77xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.*]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:         shape_value = [60, 1, 77]} : tensor<1x60x1x77xsi32> -> tensor<60x1x77xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}

// -----

// CHECK-LABEL: @Convert3dTopKFirstAxis
func.func @Convert3dTopKFirstAxis(%arg0: tensor<60x80x77xsi32>) -> (tensor<1x80x77xsi32>, tensor<1x80x77xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 0 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<60x80x77xsi32>, tensor<si32> -> tensor<1x80x77xsi32>, tensor<1x80x77xsi32>

    return %output_values, %target_shape : tensor<1x80x77xsi32>, tensor<1x80x77xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}:   shape_value = [1, 1, 60, 6160]} : tensor<60x80x77xsi32> -> tensor<1x1x60x6160xsi32>

    // CHECK:   [[VALUE:%.*]], [[SHAPE:%.*]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 2 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
    // CHECK-SAME:         tensor<1x1x60x6160xsi32> -> tensor<1x1x1x6160xsi32>, tensor<1x1x1x6160xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.*]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:         shape_value = [1, 80, 77]} : tensor<1x1x1x6160xsi32> -> tensor<1x80x77xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.*]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:         shape_value = [1, 80, 77]} : tensor<1x1x1x6160xsi32> -> tensor<1x80x77xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}

// -----

// CHECK-LABEL: @Convert3dTopKLastAxis
func.func @Convert3dTopKLastAxis(%arg0: tensor<60x80x77xsi32>) -> (tensor<60x80x1xsi32>, tensor<60x80x1xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 2 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<60x80x77xsi32>, tensor<si32> -> tensor<60x80x1xsi32>, tensor<60x80x1xsi32>

    return %output_values, %target_shape : tensor<60x80x1xsi32>, tensor<60x80x1xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {
    // CHECK-SAME{LITERAL}:   shape_value = [1, 1, 4800, 77]} : tensor<60x80x77xsi32> -> tensor<1x1x4800x77xsi32>

    // CHECK:   [[VALUE:%.*]], [[SHAPE:%.*]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
    // CHECK-SAME:         tensor<1x1x4800x77xsi32> -> tensor<1x1x4800x1xsi32>, tensor<1x1x4800x1xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.*]] = IE.Reshape([[VALUE]]) {
    // CHECK-SAME:         shape_value = [60, 80, 1]} : tensor<1x1x4800x1xsi32> -> tensor<60x80x1xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.*]] = IE.Reshape([[SHAPE]]) {
    // CHECK-SAME:         shape_value = [60, 80, 1]} : tensor<1x1x4800x1xsi32> -> tensor<60x80x1xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}

// -----

// CHECK-LABEL: @RMSNorm
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x32x6xf16>
func.func @RMSNorm(%arg0: tensor<1x32x6xf16>) -> tensor<1x32x6xf16> {
  %cst = const.Declare tensor<1x1x6xf16> = dense<1.000000e+00> : tensor<1x1x6xf16>
  %0 = IE.RMS(%arg0, %cst) {epsilon = 9.9999997473787516E-6 : f64} : tensor<1x32x6xf16>, tensor<1x1x6xf16> -> tensor<1x32x6xf16>
  return %0 : tensor<1x32x6xf16>

    // CHECK:           [[CST:%.*]] = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00> : tensor<1x1x6xf16>, [#const.Reshape<[1, 1, 1, 6]>]
    // CHECK:           [[RESHAPE_IN:%.*]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 32, 6]} : tensor<1x32x6xf16> -> tensor<1x1x32x6xf16>
    // CHECK:           [[RMS:%.*]] = IE.RMS([[RESHAPE_IN]], [[CST]]) {epsilon = 9.9999997473787516E-6 : f64} : tensor<1x1x32x6xf16>, tensor<1x1x1x6xf16> -> tensor<1x1x32x6xf16>
    // CHECK:           [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[RMS]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 32, 6]} : tensor<1x1x32x6xf16> -> tensor<1x32x6xf16>
    // CHECK:           return [[RESHAPE_OUT]] : tensor<1x32x6xf16>
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
