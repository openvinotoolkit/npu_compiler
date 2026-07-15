//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --adjust-convolution-input-shape="preferred-spatial-alignment=8" %s | FileCheck %s
// REQUIRES: platform-NPU5010

// CHECK-LABEL: @ReshapeInputFor1x1Conv
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1280x4096x1xf16>
func.func @ReshapeInputFor1x1Conv(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x4096x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4096x1xf16>
    return %0 : tensor<1x320x4096x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 512, 8]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x512x8xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x512x8xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x512x8xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4096, 1]} : tensor<1x320x512x8xf16> -> tensor<1x320x4096x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x4096x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeInputFor1x1GroupConv
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x320x4096x1xf16>
func.func @ReshapeInputFor1x1GroupConv(%arg0: tensor<1x320x4096x1xf16>) -> tensor<1x320x4096x1xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4096x1xf16>
    return %0 : tensor<1x320x4096x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 320, 512, 8]} : tensor<1x320x4096x1xf16> -> tensor<1x320x512x8xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x512x8xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x512x8xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4096, 1]} : tensor<1x320x512x8xf16> -> tensor<1x320x4096x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x4096x1xf16>
}

// -----

// CHECK: @ReshapeInputFor1x1ConvHeight1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1280x1x4096xf16>)
func.func @ReshapeInputFor1x1ConvHeight1(%arg0: tensor<1x1280x1x4096xf16>) -> tensor<1x320x1x4096xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1x4096xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1x4096xf16>
    return %0 : tensor<1x320x1x4096xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>

    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 1280, 512, 8]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x512x8xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x1280x512x8xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x512x8xf16>

    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 320, 1, 4096]} : tensor<1x320x512x8xf16> -> tensor<1x320x1x4096xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x1x4096xf16>
}

// -----

// CHECK: @ReshapeInputFor1x1GroupConvHeight1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x320x1x4096xf16>)
func.func @ReshapeInputFor1x1GroupConvHeight1(%arg0: tensor<1x320x1x4096xf16>) -> tensor<1x320x1x4096xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x1x4096xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1x4096xf16>
    return %0 : tensor<1x320x1x4096xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 8, 512]} : tensor<1x320x1x4096xf16> -> tensor<1x320x8x512xf16>

    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[RESHAPE0]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:        {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:     : tensor<1x320x8x512xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x8x512xf16>

    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 320, 1, 4096]} : tensor<1x320x8x512xf16> -> tensor<1x320x1x4096xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x1x4096xf16>
}

// -----

// Conv connected to SoftMax — W=8 override should NOT apply, falls back to default alignment (4).
// inputShapeToAlign = 4096, divisible by 8, but SoftMax user prevents W=8 override.

// CHECK-LABEL: @NoW8OverrideWhenConnectedToSoftMax
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1280x4096x1xf16>
func.func @NoW8OverrideWhenConnectedToSoftMax(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x4096x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16> -> tensor<1x320x4096x1xf16>
    %1 = IE.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x320x4096x1xf16> -> tensor<1x320x4096x1xf16>
    return %1 : tensor<1x320x4096x1xf16>

    // W=8 override is skipped due to SoftMax connection. Falls back to alignment=4.
    // 4096 / 4 = 1024, so reshape to [1, 1280, 1024, 4]
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 1024, 4]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x1024x4xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]])
    // CHECK-SAME:      : tensor<1x1280x1024x4xf16>, tensor<320x1280x1x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[RESHAPE1]])
    // CHECK:       return [[SOFTMAX]] : tensor<1x320x4096x1xf16>
}

// -----

// Input dimension not divisible by 8 — should fall back to default alignment (4).
// inputShapeToAlign = 100, not divisible by 8, so W=8 override does not apply.

// CHECK-LABEL: @FallbackWhenNotDivisibleBy8
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1280x100x1xf16>
func.func @FallbackWhenNotDivisibleBy8(%arg0: tensor<1x1280x100x1xf16>) -> tensor<1x320x100x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x100x1xf16>, tensor<320x1280x1x1xf16> -> tensor<1x320x100x1xf16>
    return %0 : tensor<1x320x100x1xf16>

    // 100 is not divisible by 8, so falls back to alignment=4. 100/4 = 25.
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 25, 4]} : tensor<1x1280x100x1xf16> -> tensor<1x1280x25x4xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]])
    // CHECK-SAME:      : tensor<1x1280x25x4xf16>, tensor<320x1280x1x1xf16> -> tensor<1x320x25x4xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 100, 1]} : tensor<1x320x25x4xf16> -> tensor<1x320x100x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x100x1xf16>
}

// -----

// Input dimension / 8 < 8 (guard condition fails) — should fall back to default alignment.
// inputShapeToAlign = 48, divisible by 8, but 48/8 = 6 < 8, so W=8 override does not apply.

// CHECK-LABEL: @FallbackWhenQuotientTooSmall
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1280x48x1xf16>
func.func @FallbackWhenQuotientTooSmall(%arg0: tensor<1x1280x48x1xf16>) -> tensor<1x320x48x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x48x1xf16>, tensor<320x1280x1x1xf16> -> tensor<1x320x48x1xf16>
    return %0 : tensor<1x320x48x1xf16>

    // 48 / 8 = 6 < 8, guard fails. Falls back to alignment=4. 48/4 = 12.
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 12, 4]} : tensor<1x1280x48x1xf16> -> tensor<1x1280x12x4xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]])
    // CHECK-SAME:      : tensor<1x1280x12x4xf16>, tensor<320x1280x1x1xf16> -> tensor<1x320x12x4xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 48, 1]} : tensor<1x320x12x4xf16> -> tensor<1x320x48x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x48x1xf16>
}

// -----

// Conv -> Add -> SoftMax pattern (attention mask): W=8 override should NOT apply.
// The pass looks through AddOp to detect SoftMax connectivity.

// CHECK-LABEL: @NoW8OverrideWhenConnectedToSoftMaxThroughAdd
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1280x4096x1xf16>
func.func @NoW8OverrideWhenConnectedToSoftMaxThroughAdd(%arg0: tensor<1x1280x4096x1xf16>, %arg1: tensor<1x320x4096x1xf16>) -> tensor<1x320x4096x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16> -> tensor<1x320x4096x1xf16>
    %1 = IE.Add(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x320x4096x1xf16>, tensor<1x320x4096x1xf16> -> tensor<1x320x4096x1xf16>
    %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x320x4096x1xf16> -> tensor<1x320x4096x1xf16>
    return %2 : tensor<1x320x4096x1xf16>

    // W=8 override skipped due to SoftMax detected through Add. Falls back to alignment=4.
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 1024, 4]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x1024x4xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]])
    // CHECK-SAME:      : tensor<1x1280x1024x4xf16>, tensor<320x1280x1x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK:       [[ADD:%.+]] = IE.Add([[RESHAPE1]]
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]])
    // CHECK:       return [[SOFTMAX]] : tensor<1x320x4096x1xf16>
}
