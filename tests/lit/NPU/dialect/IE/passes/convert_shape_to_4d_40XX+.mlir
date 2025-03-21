//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-shape-to-4d --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX
func.func @Convert2dTopKPositiveAxis(%arg0: tensor<80x77xsi32>) -> (tensor<80x1xsi32>, tensor<80x1xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<80x77xsi32>, tensor<si32> -> tensor<80x1xsi32>, tensor<80x1xsi32>

    return %output_values, %target_shape : tensor<80x1xsi32>, tensor<80x1xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SMAE:         shape_value = [1, 1, 80, 77]
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
    // CHECK-SMAE:         shape_value = [1, 1, 80, 77]
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
