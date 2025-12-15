//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% enable-auto-padding-odu" --adjust-input-shape --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK-LABEL: @ExpandAddToShapeCastAddWithTwoExpands
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x32x32xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @ExpandAddToShapeCastAddWithTwoExpands(%arg0: tensor<1x3x32x32xf16>, %arg1: tensor<1x3x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %1 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, input_padding = [0, 13, 0, 0], output_padding = [0, 0, 0, 0]} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x3x32x32xf16>
    %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    return %3 : tensor<1x16x32x32xf16>

    // CHECK-NOT:   IE.Expand
    // CHECK-DAG:   [[CAST1:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT1]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK-DAG:   [[CAST2:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT2]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>

    // CHECK:       [[ADD:%.+]] = IE.Add([[CAST1]], [[CAST2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x12xf16>, tensor<1x16x16x12xf16> -> tensor<1x16x16x12xf16>
    // CHECK:       [[CAST_OUTPUT:%.+]] = IE.ShapeCast {shape = [1, 3, 32, 32]} inputs([[ADD]] : tensor<1x16x16x12xf16>) -> tensor<1x3x32x32xf16>
    // CHECK:       [[EXPAND_OUTPUT:%.+]] = IE.Expand([[CAST_OUTPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       return [[EXPAND_OUTPUT]]
}

// -----

// CHECK-LABEL: @AdjustAvgPoolingToShapeCastAvgPooling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x2048xf16>
func.func @AdjustAvgPoolingToShapeCastAvgPooling(%arg0: tensor<1x1x1x2048xf16>) -> tensor<1x16x1x2048xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1], input_padding = [0, 15, 0, 0], output_padding = [0, 0, 0, 0]} : tensor<1x16x1x2048xf16> -> tensor<1x1x1x2048xf16>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    return %2 : tensor<1x16x1x2048xf16>

    // ExpandPoolingRewriter will be used instead of ExpandSingleChannelPoolingRewriter due to benefit level setting
    // CHECK:   [[SHAPECAST0:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 8]} inputs([[INPUT]] : tensor<1x1x1x2048xf16>) -> tensor<1x16x16x8xf16>
    // CHECK:   [[POOLING:%.+]] = IE.AvgPool([[SHAPECAST0]])
    // CHECK:   [[SHAPECAST0:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 2048]} inputs([[POOLING]] : tensor<1x16x16x8xf16>) -> tensor<1x1x1x2048xf16>
    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[SHAPECAST0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    // CHECK:       return [[EXPAND]] : tensor<1x16x1x2048xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DoNotPropagateShapeCastAfterPaddedEltwise
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x48x512x32xf16, {order = #NHWC}>
func.func @DoNotPropagateShapeCastAfterPaddedEltwise(%input: tensor<1x48x512x32xf16, {order = #NHWC}>)
            -> tensor<1x16x256x192xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%input, %weights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
        : tensor<1x48x512x32xf16, {order = #NHWC}>, tensor<48x48x3x3xf16, {order = #NHWC}> -> tensor<1x48x512x32xf16, {order = #NHWC}>
    %shapecast = IE.ShapeCast {shape = [1, 16, 256, 192]} inputs(%conv : tensor<1x48x512x32xf16, {order = #NHWC}>) -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %eltwise = IE.Add(%shapecast, %shapecast) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, input_padding = [0, 13, 0, 0], output_padding = [0, 13, 0, 0]}
        : tensor<1x16x256x192xf16, {order = #NHWC}>, tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    return %eltwise : tensor<1x16x256x192xf16, {order = #NHWC}>

    // CHECK:  [[WEIGHTS:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}>
    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]])
    // CHECK:  [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 16, 256, 192]} inputs([[CONV]]
    // CHECK:  [[ADD:%.+]] = IE.Add([[SHAPE_CAST]], [[SHAPE_CAST]]
    // CHECK:  return [[ADD]]
}
