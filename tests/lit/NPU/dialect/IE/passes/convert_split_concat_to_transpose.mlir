//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-split-concat-to-transpose %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3, d4)>

// CHECK-LABEL: @ConvertSplitAffineReshapeConcatToTranspose
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x2x120x96x49xf16>
func.func @ConvertSplitAffineReshapeConcatToTranspose(%arg0: tensor<1x2x120x96x49xf16>) -> tensor<1x120x2x96x49xf16> {
    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x2x120x96x49xf16> -> tensor<1x1x120x96x49xf16>, tensor<1x1x120x96x49xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    %2 = IE.AffineReshape(%0#1) {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]} : tensor<1x120x1x96x49xf16>, tensor<1x120x1x96x49xf16> -> tensor<1x120x2x96x49xf16>

    return %3 : tensor<1x120x2x96x49xf16>

   // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #map} : tensor<1x2x120x96x49xf16> -> tensor<1x120x2x96x49xf16>
   // CHECK:       return [[TRANSPOSE]] : tensor<1x120x2x96x49xf16>
}

// -----

// CHECK-LABEL: @NotConvertSplitAffineReshapeConcatToTransposeAsSplitShape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x96x2x48xf16>
func.func @NotConvertSplitAffineReshapeConcatToTransposeAsSplitShape(%arg0: tensor<1x96x2x48xf16>) -> tensor<1x2x48x96xf16> {
    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x96x2x48xf16> -> tensor<1x48x2x48xf16>, tensor<1x48x2x48xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 48, 96]} : tensor<1x48x2x48xf16> -> tensor<1x1x48x96xf16>
    %2 = IE.AffineReshape(%0#1) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 48, 96]} : tensor<1x48x2x48xf16> -> tensor<1x1x48x96xf16>
    %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x48x96xf16>, tensor<1x1x48x96xf16> -> tensor<1x2x48x96xf16>

    return %3 : tensor<1x2x48x96xf16>

    // CHECK:       [[SPLIT:%.+]]:2 = IE.Split([[INPUT]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x96x2x48xf16> -> tensor<1x48x2x48xf16>, tensor<1x48x2x48xf16>
    // CHECK:       [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[SPLIT]]#0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 48, 96]} : tensor<1x48x2x48xf16> -> tensor<1x1x48x96xf16>
    // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[SPLIT]]#1)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 48, 96]} : tensor<1x48x2x48xf16> -> tensor<1x1x48x96xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x48x96xf16>, tensor<1x1x48x96xf16> -> tensor<1x2x48x96xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x2x48x96xf16>
}

// -----

// CHECK-LABEL: @NotConvertSplitAffineReshapeConcatToTransposeAsDimMapping
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x120x96x48xf16>
func.func @NotConvertSplitAffineReshapeConcatToTransposeAsDimMapping(%arg0: tensor<1x2x120x96x48xf16>) -> tensor<1x120x2x96x48xf16> {
    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x2x120x96x48xf16> -> tensor<1x1x120x96x48xf16>, tensor<1x1x120x96x48xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0], [0], [1], [2, 3], [4]], shape_value = [1, 120, 2, 48, 48]} : tensor<1x1x120x96x48xf16> -> tensor<1x120x2x48x48xf16>
    %2 = IE.AffineReshape(%0#1) {dim_mapping = [[0], [0], [1], [2, 3], [4]], shape_value = [1, 120, 2, 48, 48]} : tensor<1x1x120x96x48xf16> -> tensor<1x120x2x48x48xf16>
    %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 48, 0]]} : tensor<1x120x2x48x48xf16>, tensor<1x120x2x48x48xf16> -> tensor<1x120x2x96x48xf16>

    return %3 : tensor<1x120x2x96x48xf16>

    // CHECK:       [[SPLIT:%.+]]:2 = IE.Split([[INPUT]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x120x96x48xf16> -> tensor<1x1x120x96x48xf16>, tensor<1x1x120x96x48xf16>
    // CHECK:       [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[SPLIT]]#0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2, 3], [4]], shape_value = [1, 120, 2, 48, 48]} : tensor<1x1x120x96x48xf16> -> tensor<1x120x2x48x48xf16>
    // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[SPLIT]]#1)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2, 3], [4]], shape_value = [1, 120, 2, 48, 48]} : tensor<1x1x120x96x48xf16> -> tensor<1x120x2x48x48xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 48, 0]]} : tensor<1x120x2x48x48xf16>, tensor<1x120x2x48x48xf16> -> tensor<1x120x2x96x48xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x120x2x96x48xf16>
}

// -----

// CHECK-LABEL: @NotConvertSplitAffineReshapeConcatToTransposeAsDimSize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x120x96x49xf16>
func.func @NotConvertSplitAffineReshapeConcatToTransposeAsDimSize(%arg0: tensor<1x2x120x96x49xf16>) -> tensor<1x120x3x96x49xf16> {
    %cst = const.Declare tensor<1x120x1x96x49xf16> = dense<0.000000e+00> : tensor<1x120x1x96x49xf16>
    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x2x120x96x49xf16> -> tensor<1x1x120x96x49xf16>, tensor<1x1x120x96x49xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    %2 = IE.AffineReshape(%0#1) {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    %3 = IE.Concat(%1, %2, %cst) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0]]} : tensor<1x120x1x96x49xf16>, tensor<1x120x1x96x49xf16>, tensor<1x120x1x96x49xf16> -> tensor<1x120x3x96x49xf16>

    return %3 : tensor<1x120x3x96x49xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare
    // CHECK:       [[SPLIT:%.+]]:2 = IE.Split([[INPUT]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x120x96x49xf16> -> tensor<1x1x120x96x49xf16>, tensor<1x1x120x96x49xf16>
    // CHECK:       [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[SPLIT]]#0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[SPLIT]]#1)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]], [[CST]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0]]} : tensor<1x120x1x96x49xf16>, tensor<1x120x1x96x49xf16>, tensor<1x120x1x96x49xf16> -> tensor<1x120x3x96x49xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x120x3x96x49xf16>
}

// -----

// CHECK-LABEL: @ConvertSplitConcatToAffineReshape
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x2x120x96x49xf16>
func.func @ConvertSplitConcatToAffineReshape(%arg0: tensor<1x2x120x96x49xf16>) -> tensor<1x1x240x96x49xf16> {
    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x2x120x96x49xf16> -> tensor<1x1x120x96x49xf16>, tensor<1x1x120x96x49xf16>
    %1 = IE.Concat(%0#0, %0#1) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 120, 0, 0]]} : tensor<1x1x120x96x49xf16>, tensor<1x1x120x96x49xf16> -> tensor<1x1x240x96x49xf16>

    return %1 : tensor<1x1x240x96x49xf16>

   // CHECK:       [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
   // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [2], [3], [4]], shape_value = [1, 1, 240, 96, 49]} : tensor<1x2x120x96x49xf16> -> tensor<1x1x240x96x49xf16>
   // CHECK:       return [[AFFINERESHAPE]] : tensor<1x1x240x96x49xf16>
}

// -----

// CHECK-LABEL: @ComposeSplitAffineReshapeConv
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x3x150x2xf16>
func.func @ComposeSplitAffineReshapeConv(%arg0: tensor<1x3x150x2xf16>) -> tensor<1x32x1x150xf16> {
    %cst_0 = const.Declare tensor<16x1x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>]

    %0:2 = IE.Split(%arg0) {axis_value = 3 : i64, num_splits = 2 : i64} : tensor<1x3x150x2xf16> -> tensor<1x3x150x1xf16>, tensor<1x3x150x1xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3, 150]} :
        tensor<1x3x150x1xf16> -> tensor<1x1x3x150xf16>
    %2 = IE.Convolution(%1, %cst_1) {
            dilations = [1, 1],
            pads_begin = [0, 1],
            pads_end = [0, 1],
            strides = [1, 1]
    } : tensor<1x1x3x150xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x150xf16>

    %3 = IE.AffineReshape(%0#1) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3, 150]} :
        tensor<1x3x150x1xf16> -> tensor<1x1x3x150xf16>
    %4 = IE.Convolution(%3, %cst_0) {
            dilations = [1, 1],
            pads_begin = [0, 1],
            pads_end = [0, 1],
            strides = [1, 1]
    } : tensor<1x1x3x150xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x150xf16>

    %5 = IE.Concat(%2, %4) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x1x150xf16>, tensor<1x16x1x150xf16> -> tensor<1x32x1x150xf16>

    return %5 : tensor<1x32x1x150xf16>

    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<16x2x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 1, 0, 0], [0, 0, 0, 0]>]
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<16x2x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>]

    // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x3x150x2xf16> -> tensor<1x2x3x150xf16>
    // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CST_1]], [[CST_0]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<16x2x3x3xf16>, tensor<16x2x3x3xf16> -> tensor<32x2x3x3xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_IN]], [[CONCAT_1]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 1],
    // CHECK-SAME:      pads_end = [0, 1],
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x2x3x150xf16>, tensor<32x2x3x3xf16> -> tensor<1x32x1x150xf16>

    // CHECK:       return [[CONV]] : tensor<1x32x1x150xf16>
}

// -----

// CHECK-LABEL: @ComposeSplitAffineReshapeConvWithBias
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x3x150x2xf16>
func.func @ComposeSplitAffineReshapeConvWithBias(%arg0: tensor<1x3x150x2xf16>) -> tensor<1x32x1x150xf16> {
    %cst_0 = const.Declare tensor<16x1x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>]

    %bias_0 = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>, [#const.CastElemType<f16>]
    %bias_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.CastElemType<f16>]

    %0:2 = IE.Split(%arg0) {axis_value = 3 : i64, num_splits = 2 : i64} : tensor<1x3x150x2xf16> -> tensor<1x3x150x1xf16>, tensor<1x3x150x1xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3, 150]} :
        tensor<1x3x150x1xf16> -> tensor<1x1x3x150xf16>
    %2 = IE.Convolution(%1, %cst_1, %bias_1) {
            dilations = [1, 1],
            pads_begin = [0, 1],
            pads_end = [0, 1],
            strides = [1, 1]
    } : tensor<1x1x3x150xf16>, tensor<16x1x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x150xf16>

    %3 = IE.AffineReshape(%0#1) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3, 150]} :
        tensor<1x3x150x1xf16> -> tensor<1x1x3x150xf16>
    %4 = IE.Convolution(%3, %cst_0, %bias_0) {
            dilations = [1, 1],
            pads_begin = [0, 1],
            pads_end = [0, 1],
            strides = [1, 1]
    } : tensor<1x1x3x150xf16>, tensor<16x1x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x150xf16>

    %5 = IE.Concat(%2, %4) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x1x150xf16>, tensor<1x16x1x150xf16> -> tensor<1x32x1x150xf16>

    return %5 : tensor<1x32x1x150xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x2x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 1, 0, 0], [0, 0, 0, 0]>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<16x2x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>]
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>, [#const.CastElemType<f16>]

    // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x3x150x2xf16> -> tensor<1x2x3x150xf16>
    // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CST_0]], [[CST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<16x2x3x3xf16>, tensor<16x2x3x3xf16> -> tensor<32x2x3x3xf16>
    // CHECK:       [[CONCAT_2:%.+]] = IE.Concat([[CST_1]], [[CST_2]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x1xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_IN]], [[CONCAT_1]], [[CONCAT_2]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 1],
    // CHECK-SAME:      pads_end = [0, 1],
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x2x3x150xf16>, tensor<32x2x3x3xf16>, tensor<1x2x1x1xf16> -> tensor<1x32x1x150xf16>

    // CHECK:       return [[CONV]] : tensor<1x32x1x150xf16>
}

// -----

// CHECK-LABEL: @ComposeSplitAffineReshapeConvWithOneBias
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x3x150x2xf16>
func.func @ComposeSplitAffineReshapeConvWithOneBias(%arg0: tensor<1x3x150x2xf16>) -> tensor<1x32x1x150xf16> {
    %cst_0 = const.Declare tensor<16x1x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>]

    %bias_0 = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>, [#const.CastElemType<f16>]

    %0:2 = IE.Split(%arg0) {axis_value = 3 : i64, num_splits = 2 : i64} : tensor<1x3x150x2xf16> -> tensor<1x3x150x1xf16>, tensor<1x3x150x1xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3, 150]} :
        tensor<1x3x150x1xf16> -> tensor<1x1x3x150xf16>
    %2 = IE.Convolution(%1, %cst_1) {
            dilations = [1, 1],
            pads_begin = [0, 1],
            pads_end = [0, 1],
            strides = [1, 1]
    } : tensor<1x1x3x150xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x150xf16>

    %3 = IE.AffineReshape(%0#1) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 3, 150]} :
        tensor<1x3x150x1xf16> -> tensor<1x1x3x150xf16>
    %4 = IE.Convolution(%3, %cst_0, %bias_0) {
            dilations = [1, 1],
            pads_begin = [0, 1],
            pads_end = [0, 1],
            strides = [1, 1]
    } : tensor<1x1x3x150xf16>, tensor<16x1x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x1x150xf16>

    %5 = IE.Concat(%2, %4) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x1x150xf16>, tensor<1x16x1x150xf16> -> tensor<1x32x1x150xf16>

    return %5 : tensor<1x32x1x150xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<16x2x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 1, 0, 0], [0, 0, 0, 0]>]
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<16x2x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>]
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>, [#const.CastElemType<f16>]

    // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x3x150x2xf16> -> tensor<1x2x3x150xf16>
    // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CST_1]], [[CST_0]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<16x2x3x3xf16>, tensor<16x2x3x3xf16> -> tensor<32x2x3x3xf16>
    // CHECK:       [[CONCAT_2:%.+]] = IE.Concat([[CST]], [[CST_2]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_IN]], [[CONCAT_1]], [[CONCAT_2]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 1],
    // CHECK-SAME:      pads_end = [0, 1],
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x2x3x150xf16>, tensor<32x2x3x3xf16>, tensor<1x2x1x1xf16> -> tensor<1x32x1x150xf16>

    // CHECK:       return [[CONV]] : tensor<1x32x1x150xf16>
}

// -----

// CHECK-LABEL: @ComposeSplitAffineReshapeConvWithMultiUsers
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x1x32x2xf16>
func.func @ComposeSplitAffineReshapeConvWithMultiUsers(%arg0: tensor<1x1x32x2xf16>) -> (tensor<1x2x1x16xf16>, tensor<1x2x1x32xf16>) {
    %cst_0 = const.Declare tensor<1x1x1x3xf16> = dense<0.000000e+00> : tensor<1x1x1x3xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x3xf16> = dense<0.000000e+00> : tensor<1x1x1x3xf32>, [#const.CastElemType<f16>]

    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.CastElemType<f16>]

    %0:2 = IE.Split(%arg0) {axis_value = 3 : i64, num_splits = 2 : i64} : tensor<1x1x32x2xf16> -> tensor<1x1x32x1xf16>, tensor<1x1x32x1xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 32]} :
        tensor<1x1x32x1xf16> -> tensor<1x1x1x32xf16>
    %2 = IE.Convolution(%1, %cst_1) {
            dilations = [1, 1],
            pads_begin = [0, 1],
            pads_end = [0, 1],
            strides = [1, 2]
    } : tensor<1x1x1x32xf16>, tensor<1x1x1x3xf16> -> tensor<1x1x1x16xf16>

    %3 = IE.AffineReshape(%0#1) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 1, 32]} :
        tensor<1x1x32x1xf16> -> tensor<1x1x1x32xf16>

    %4 = IE.Convolution(%3, %cst_0) {
            dilations = [1, 1],
            pads_begin = [0, 1],
            pads_end = [0, 1],
            strides = [1, 2]
    } : tensor<1x1x1x32xf16>, tensor<1x1x1x3xf16> -> tensor<1x1x1x16xf16>

    %5 = IE.Concat(%2, %4) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x16xf16>, tensor<1x1x1x16xf16> -> tensor<1x2x1x16xf16>

    %6 = IE.Convolution(%1, %cst_2) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
    } : tensor<1x1x1x32xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x32xf16>

    %7 = IE.Convolution(%3, %cst_3) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
    } : tensor<1x1x1x32xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x32xf16>

    %8 = IE.Concat(%6, %7) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x2x1x32xf16>

    return %5, %8 : tensor<1x2x1x16xf16>, tensor<1x2x1x32xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x2x1x3xf16> = dense<0.000000e+00> : tensor<1x1x1x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 1, 0, 0], [0, 0, 0, 0]>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x2x1x3xf16> = dense<0.000000e+00> : tensor<1x1x1x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>]
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 1, 0, 0], [0, 0, 0, 0]>]
    // CHECK-DAG:   [[CST_3:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>]

    // CHECK:       [[TRANSPOSE_IN1:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x1x32x2xf16> -> tensor<1x2x1x32xf16>
    // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[CST_3]], [[CST_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16> -> tensor<2x2x1x1xf16>
    // CHECK:       [[CONV1:%.+]] = IE.Convolution([[TRANSPOSE_IN1]], [[CONCAT1]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x2x1x32xf16>, tensor<2x2x1x1xf16> -> tensor<1x2x1x32xf16>

    // CHECK:       [[TRANSPOSE_IN2:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x1x32x2xf16> -> tensor<1x2x1x32xf16>
    // CHECK:       [[CONCAT5:%.+]] = IE.Concat([[CST_0]], [[CST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x2x1x3xf16>, tensor<1x2x1x3xf16> -> tensor<2x2x1x3xf16>
    // CHECK:       [[CONV2:%.+]] = IE.Convolution([[TRANSPOSE_IN2]], [[CONCAT5]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 1],
    // CHECK-SAME:      pads_end = [0, 1],
    // CHECK-SAME:      strides = [1, 2]} : tensor<1x2x1x32xf16>, tensor<2x2x1x3xf16> -> tensor<1x2x1x16xf16>

    // CHECK:       return [[CONV2]], [[CONV1]] : tensor<1x2x1x16xf16>, tensor<1x2x1x32xf16>
}

// -----

// CHECK-LABEL: @ComposeSplitAffineReshapeConvWithDimWNoChange
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x3x2x150xf16>
func.func @ComposeSplitAffineReshapeConvWithDimWNoChange(%arg0: tensor<1x3x2x150xf16>) -> tensor<1x32x1x150xf16> {
    %cst_0 = const.Declare tensor<16x1x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>]

    %0:2 = IE.Split(%arg0) {axis_value = 2 : i64, num_splits = 2 : i64} : tensor<1x3x2x150xf16> -> tensor<1x3x1x150xf16>, tensor<1x3x1x150xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 3, 150]} :
        tensor<1x3x1x150xf16> -> tensor<1x1x3x150xf16>
    %2 = IE.Convolution(%1, %cst_1) {
            dilations = [1, 1],
            pads_begin = [0, 1],
            pads_end = [0, 1],
            strides = [1, 1]
    } : tensor<1x1x3x150xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x150xf16>

    %3 = IE.AffineReshape(%0#1) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 3, 150]} :
        tensor<1x3x1x150xf16> -> tensor<1x1x3x150xf16>
    %4 = IE.Convolution(%3, %cst_0) {
            dilations = [1, 1],
            pads_begin = [0, 1],
            pads_end = [0, 1],
            strides = [1, 1]
    } : tensor<1x1x3x150xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x150xf16>

    %5 = IE.Concat(%2, %4) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x1x150xf16>, tensor<1x16x1x150xf16> -> tensor<1x32x1x150xf16>

    return %5 : tensor<1x32x1x150xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x2x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 1, 0, 0], [0, 0, 0, 0]>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<16x2x3x3xf16> = dense<0.000000e+00> : tensor<16x1x3x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>]

    // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NHCW} : tensor<1x3x2x150xf16> -> tensor<1x2x3x150xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[CST_0]], [[CST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<16x2x3x3xf16>, tensor<16x2x3x3xf16> -> tensor<32x2x3x3xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_IN]], [[CONCAT]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 1],
    // CHECK-SAME:      pads_end = [0, 1],
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x2x3x150xf16>, tensor<32x2x3x3xf16> -> tensor<1x32x1x150xf16>

    // CHECK:       return [[CONV]] : tensor<1x32x1x150xf16>
}

// -----

// CHECK-LABEL: @ComposeSplitAffineReshapeTransConv
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x1x8x2xf16>
func.func @ComposeSplitAffineReshapeTransConv(%arg0: tensor<1x1x8x2xf16>) -> tensor<1x2x1x18xf16> {
    %cst_0 = const.Declare tensor<1x1x1x3xf16> = dense<0.000000e+00> : tensor<1x1x1x3xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x3xf16> = dense<0.000000e+00> : tensor<1x1x1x3xf32>, [#const.CastElemType<f16>]

    %0:2 = IE.Split(%arg0) {axis_value = 3 : i64, num_splits = 2 : i64} : tensor<1x1x8x2xf16> -> tensor<1x1x8x1xf16>, tensor<1x1x8x1xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 8]} :
        tensor<1x1x8x1xf16> -> tensor<1x1x1x8xf16>
    %2 = IE.TransposedConvolution(%1, %cst_1) {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0>,
            spatial_output_padding = [0, 0],
            pads_begin = [0, 0],
            pads_end = [0, -1],
            strides = [1, 2]
    } : tensor<1x1x1x8xf16>, tensor<1x1x1x3xf16> -> tensor<1x1x1x18xf16>

    %3 = IE.AffineReshape(%0#1) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 8]} :
        tensor<1x1x8x1xf16> -> tensor<1x1x1x8xf16>
    %4 = IE.TransposedConvolution(%3, %cst_0) {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0>,
            spatial_output_padding = [0, 0],
            pads_begin = [0, 0],
            pads_end = [0, -1],
            strides = [1, 2]
    } : tensor<1x1x1x8xf16>, tensor<1x1x1x3xf16> -> tensor<1x1x1x18xf16>

    %5 = IE.Concat(%2, %4) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x18xf16>, tensor<1x1x1x18xf16> -> tensor<1x2x1x18xf16>

    return %5 : tensor<1x2x1x18xf16>

    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x2x1x3xf16> = dense<0.000000e+00> : tensor<1x1x1x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 1, 0, 0], [0, 0, 0, 0]>]
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x2x1x3xf16> = dense<0.000000e+00> : tensor<1x1x1x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>]

    // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x1x8x2xf16> -> tensor<1x2x1x8xf16>
    // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CST_1]], [[CST_0]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x2x1x3xf16>, tensor<1x2x1x3xf16> -> tensor<2x2x1x3xf16>

    // CHECK:       [[CONV:%.+]] = IE.TransposedConvolution([[TRANSPOSE_IN]], [[CONCAT_1]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, -1],
    // CHECK-SAME:      spatial_output_padding = [0, 0],
    // CHECK-SAME:      strides = [1, 2]} : tensor<1x2x1x8xf16>, tensor<2x2x1x3xf16> -> tensor<1x2x1x18xf16>

    // CHECK:       return [[CONV]] : tensor<1x2x1x18xf16>
}

// -----

// CHECK-LABEL: @ComposeSplitAffineReshapeTransConvWithBias
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x1x8x2xf16>
func.func @ComposeSplitAffineReshapeTransConvWithBias(%arg0: tensor<1x1x8x2xf16>) -> tensor<1x2x1x18xf16> {
    %cst_0 = const.Declare tensor<1x1x1x3xf16> = dense<0.000000e+00> : tensor<1x1x1x3xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x3xf16> = dense<0.000000e+00> : tensor<1x1x1x3xf32>, [#const.CastElemType<f16>]

    %bias_0 = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %bias_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    %0:2 = IE.Split(%arg0) {axis_value = 3 : i64, num_splits = 2 : i64} : tensor<1x1x8x2xf16> -> tensor<1x1x8x1xf16>, tensor<1x1x8x1xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 8]} :
        tensor<1x1x8x1xf16> -> tensor<1x1x1x8xf16>
    %2 = IE.TransposedConvolution(%1, %cst_1, %bias_1) {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1>,
            spatial_output_padding = [0, 0],
            pads_begin = [0, 0],
            pads_end = [0, -1],
            strides = [1, 2]
    } : tensor<1x1x1x8xf16>, tensor<1x1x1x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x18xf16>

    %3 = IE.AffineReshape(%0#1) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 8]} :
        tensor<1x1x8x1xf16> -> tensor<1x1x1x8xf16>
    %4 = IE.TransposedConvolution(%3, %cst_0, %bias_0) {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1>,
            spatial_output_padding = [0, 0],
            pads_begin = [0, 0],
            pads_end = [0, -1],
            strides = [1, 2]
    } : tensor<1x1x1x8xf16>, tensor<1x1x1x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x18xf16>

    %5 = IE.Concat(%2, %4) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x18xf16>, tensor<1x1x1x18xf16> -> tensor<1x2x1x18xf16>

    return %5 : tensor<1x2x1x18xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x2x1x3xf16> = dense<0.000000e+00> : tensor<1x1x1x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 1, 0, 0], [0, 0, 0, 0]>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x2x1x3xf16> = dense<0.000000e+00> : tensor<1x1x1x3xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>]
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    // CHECK:       [[TRANSPOSE_IN:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x1x8x2xf16> -> tensor<1x2x1x8xf16>
    // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CST_0]], [[CST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x2x1x3xf16>, tensor<1x2x1x3xf16> -> tensor<2x2x1x3xf16>
    // CHECK:       [[CONCAT_2:%.+]] = IE.Concat([[CST_2]], [[CST_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x1xf16>

    // CHECK:       [[CONV:%.+]] = IE.TransposedConvolution([[TRANSPOSE_IN]], [[CONCAT_1]], [[CONCAT_2]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 1>,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, -1],
    // CHECK-SAME:      spatial_output_padding = [0, 0],
    // CHECK-SAME:      strides = [1, 2]} : tensor<1x2x1x8xf16>, tensor<2x2x1x3xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x1x18xf16>

    // CHECK:       return [[CONV]] : tensor<1x2x1x18xf16>
}
