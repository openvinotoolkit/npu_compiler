//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --adjust-convolution-input-shape %s | FileCheck %s --check-prefixes=CHECK,CHECK-%platform%
// REQUIRES: platform-NPU5010 || platform-NPU6010

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

// -----

// Conv -> channel-aligned Expand/Slice -> Add -> SoftMax. The architecture
// determines whether the SoftMax guard is retained.

// CHECK-LABEL: @SoftMaxConnectedThroughExpandSlice
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x1280x4096x1xf16>
func.func @SoftMaxConnectedThroughExpandSlice(%arg0: tensor<1x1280x4096x1xf16>, %arg1: tensor<1x320x4096x1xf16>) -> tensor<1x320x4096x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16> -> tensor<1x320x4096x1xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x320x4096x1xf16> -> tensor<1x324x4096x1xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 320, 4096, 1] : tensor<1x324x4096x1xf16> to tensor<1x320x4096x1xf16>
    %3 = IE.Add(%2, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x320x4096x1xf16>, tensor<1x320x4096x1xf16> -> tensor<1x320x4096x1xf16>
    %4 = IE.SoftMax(%3) {axisInd = 1 : i64} : tensor<1x320x4096x1xf16> -> tensor<1x320x4096x1xf16>
    return %4 : tensor<1x320x4096x1xf16>

    // CHECK-NPU5010:       {{%.+}} = IE.AffineReshape([[ARG_0]])
    // CHECK-NPU5010-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 512, 8]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x512x8xf16>
    // CHECK-NPU5010:       {{%.+}} = IE.Convolution({{%.+}}, {{.*}})
    // CHECK-NPU5010-SAME:      : tensor<1x1280x512x8xf16>, tensor<320x1280x1x1xf16> -> tensor<1x320x512x8xf16>
    // CHECK-NPU6010:       {{%.+}} = IE.AffineReshape([[ARG_0]])
    // CHECK-NPU6010-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 1024, 4]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x1024x4xf16>
    // CHECK-NPU6010:       {{%.+}} = IE.Convolution({{%.+}}, {{.*}})
    // CHECK-NPU6010-SAME:      : tensor<1x1280x1024x4xf16>, tensor<320x1280x1x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       {{%.+}} = IE.AffineReshape({{%.+}})
    // CHECK:       {{%.+}} = IE.Expand({{%.+}})
    // CHECK:       {{%.+}} = IE.Slice {{%.+}}
    // CHECK:       {{%.+}} = IE.Add({{%.+}}
    // CHECK:       {{%.+}} = IE.SoftMax({{%.+}})
    // CHECK:       return {{%.+}} : tensor<1x320x4096x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Two convs share one input; each output goes to an NCHW MemPermute with C = 1500.
// 375 rows do not split equally between 4 tiles, so both convs get 500x3 instead.

// CHECK-LABEL: @AlignHeightToTilesForConvsToMemPermute
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x512x1500x1xf16, {order = #NHWC}>
func.func @AlignHeightToTilesForConvsToMemPermute(%arg0: tensor<1x512x1500x1xf16, {order = #NHWC}>)
        -> (tensor<1x8x1500x64xf16>, tensor<1x8x1500x64xf16>) {
    %filter0 = const.Declare tensor<512x512x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x512x1x1xf16>, [#const.Reorder<#NHWC>]
    %filter1 = const.Declare tensor<512x512x1x1xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<512x512x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %filter0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1500x1xf16, {order = #NHWC}>, tensor<512x512x1x1xf16, {order = #NHWC}> -> tensor<1x512x1500x1xf16, {order = #NHWC}>
    %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x512x1500x1xf16, {order = #NHWC}> -> tensor<1500x512x1x1xf16>
    %2 = IE.Reshape(%1) {shape_value = [1, 1500, 8, 64]} : tensor<1500x512x1x1xf16> -> tensor<1x1500x8x64xf16>
    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf16> -> tensor<1x8x1500x64xf16>
    %4 = IE.Convolution(%arg0, %filter1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1500x1xf16, {order = #NHWC}>, tensor<512x512x1x1xf16, {order = #NHWC}> -> tensor<1x512x1500x1xf16, {order = #NHWC}>
    %5 = IE.PermuteCast(%4) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x512x1500x1xf16, {order = #NHWC}> -> tensor<1500x512x1x1xf16>
    %6 = IE.Reshape(%5) {shape_value = [1, 1500, 8, 64]} : tensor<1500x512x1x1xf16> -> tensor<1x1500x8x64xf16>
    %7 = IE.MemPermute(%6) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1500x8x64xf16> -> tensor<1x8x1500x64xf16>
    return %3, %7 : tensor<1x8x1500x64xf16>, tensor<1x8x1500x64xf16>

    // CHECK:       [[IN_RESHAPE0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-NPU5010-SAME:      -> tensor<1x512x375x4xf16, {order = #NHWC}>
    // CHECK-NPU6010-SAME:      -> tensor<1x512x500x3xf16, {order = #NHWC}>
    // CHECK:       [[CONV0:%.+]] = IE.Convolution([[IN_RESHAPE0]]
    // CHECK-NPU5010-SAME:      -> tensor<1x512x375x4xf16, {order = #NHWC}>
    // CHECK-NPU6010-SAME:      -> tensor<1x512x500x3xf16, {order = #NHWC}>
    // CHECK:       [[OUT_RESHAPE0:%.+]] = IE.AffineReshape([[CONV0]])
    // CHECK-SAME:      -> tensor<1x512x1500x1xf16, {order = #NHWC}>
    // CHECK:       [[CAST0:%.+]] = IE.PermuteCast([[OUT_RESHAPE0]])
    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[CAST0]])
    // CHECK:       [[PERMUTE0:%.+]] = IE.MemPermute([[RESHAPE0]])

    // CHECK:       [[IN_RESHAPE1:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-NPU5010-SAME:      -> tensor<1x512x375x4xf16, {order = #NHWC}>
    // CHECK-NPU6010-SAME:      -> tensor<1x512x500x3xf16, {order = #NHWC}>
    // CHECK:       [[CONV1:%.+]] = IE.Convolution([[IN_RESHAPE1]]
    // CHECK-NPU5010-SAME:      -> tensor<1x512x375x4xf16, {order = #NHWC}>
    // CHECK-NPU6010-SAME:      -> tensor<1x512x500x3xf16, {order = #NHWC}>
    // CHECK:       [[OUT_RESHAPE1:%.+]] = IE.AffineReshape([[CONV1]])
    // CHECK-SAME:      -> tensor<1x512x1500x1xf16, {order = #NHWC}>
    // CHECK:       [[CAST1:%.+]] = IE.PermuteCast([[OUT_RESHAPE1]])
    // CHECK:       [[RESHAPE1:%.+]] = IE.Reshape([[CAST1]])
    // CHECK:       [[PERMUTE1:%.+]] = IE.MemPermute([[RESHAPE1]])

    // CHECK:       return [[PERMUTE0]], [[PERMUTE1]] : tensor<1x8x1500x64xf16>, tensor<1x8x1500x64xf16>
}
