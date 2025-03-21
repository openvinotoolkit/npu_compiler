//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-fake-quant-with-reshape-and-strided-slice %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @SwapFakeQuantReshape
func.func @SwapFakeQuantReshape(
        %input: tensor<1x1x40xf16>,
        %weights: tensor<512x40x1x1xf16>)
            -> tensor<1x512x1x1xf16> {
    %cst_0 = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
    %cst_1 = const.Declare tensor<f16> = dense<1.000000e+00> : tensor<f16>
    %1 = IE.SoftMax(%input) {axisInd = 2} : tensor<1x1x40xf16> -> tensor<1x1x40xf16>
    %2 = IE.AffineReshape(%1) {shape_value = [1, 1, 1, 40], dim_mapping = [[0], [1, 2], [3]]} : tensor<1x1x40xf16> -> tensor<1x1x1x40xf16>
    %3 = IE.FakeQuantize(%2, %cst_0, %cst_1, %cst_0, %cst_1) {
                     auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                     levels = 256 : i64
                 } : tensor<1x1x1x40xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x1x1x40xf16>
    %4 = IE.AffineReshape(%3) {shape_value = [1, 40, 1, 1], dim_mapping = [[0], [0], [0], [1, 2, 3]]} : tensor<1x1x1x40xf16> -> tensor<1x40x1x1xf16>
    %5 = IE.Convolution(%4, %weights) {
                    strides = [1, 1],
                    pads_begin = [0, 0],
                    pads_end = [0, 0],
                    dilations = [1, 1]
                } : tensor<1x40x1x1xf16>, tensor<512x40x1x1xf16> -> tensor<1x512x1x1xf16>

    return %5 : tensor<1x512x1x1xf16>

    // CHECK-DAG:       %[[FQ_MAX:.*]] = const.Declare tensor<f16> = dense<1.000000e+00> : tensor<f16>
    // CHECK-DAG:       %[[FQ_MIN:.*]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
    // CHECK:       %[[SOFTMAX:.*]] = IE.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x1x40xf16> -> tensor<1x1x40xf16>
    // CHECK:       %[[AFFINERESHAPE1:.*]] = IE.AffineReshape(%[[SOFTMAX]])
    // CHECK-SAME:      tensor<1x1x40xf16> -> tensor<1x1x1x40xf16>
    // CHECK:       %[[AFFINERESHAPE2:.*]] = IE.AffineReshape(%[[AFFINERESHAPE1]])
    // CHECK-SAME:      tensor<1x1x1x40xf16> -> tensor<1x40x1x1xf16>
    // CHECK:       %[[FQ:.*]] = IE.FakeQuantize(%[[AFFINERESHAPE2]], %[[FQ_MIN]], %[[FQ_MAX]], %[[FQ_MIN]], %[[FQ_MAX]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK-SAME:      tensor<1x40x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16>
    // CHECK-SAME:         -> tensor<1x40x1x1xf16>
    // CHECK:       %[[CONV:.*]] = IE.Convolution(%[[FQ]], %arg1)
    // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:      tensor<1x40x1x1xf16>, tensor<512x40x1x1xf16> -> tensor<1x512x1x1xf16>
}

// -----

// CHECK-LABEL: @SwapFakeQuantWithStridedSlice
func.func @SwapFakeQuantWithStridedSlice(%arg0: tensor<1x3x640x640xf16>) -> tensor<1x6x320x320xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.996688663> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x640x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x640x640xf16>
    %2 = IE.StridedSlice(%1) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    %3 = IE.StridedSlice(%2) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    %4 = IE.StridedSlice(%1) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    %5 = IE.StridedSlice(%4) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    %6 = IE.Concat(%3, %5) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x320x320xf16>, tensor<1x3x320x320xf16> -> tensor<1x6x320x320xf16>

    return %6 : tensor<1x6x320x320xf16>

    // CHECK:      [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:      [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.996688663> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:      [[STRIDEDSLICE0:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    // CHECK:      [[STRIDEDSLICE1:%.*]] = IE.StridedSlice([[STRIDEDSLICE0]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    // CHECK:      [[FQ0:%.*]] = IE.FakeQuantize([[STRIDEDSLICE1]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x320x320xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x320x320xf16>

    // CHECK:      [[STRIDEDSLICE2:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    // CHECK:      [[STRIDEDSLICE3:%.*]] = IE.StridedSlice([[STRIDEDSLICE2]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    // CHECK:      [[FQ1:%.*]] = IE.FakeQuantize([[STRIDEDSLICE3]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x320x320xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x320x320xf16>

    // CHECK:      [[CONCAT:%.*]] = IE.Concat([[FQ0]], [[FQ1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x320x320xf16>, tensor<1x3x320x320xf16> -> tensor<1x6x320x320xf16>

    // CHECK:      return [[CONCAT]] : tensor<1x6x320x320xf16>
}

// CHECK-LABEL: @SwapPerChannelFakeQuantWithReshape
func.func @SwapPerChannelFakeQuantWithReshape(%arg0: tensor<1x128x768x1xf16> , %arg2: tensor<1x768x1x1xf16>) -> tensor<1x768x1x1xf16> {
    %cst_input_fq = const.Declare tensor<1x768x1x768xf16> = dense<0.882825195> : tensor<768x768xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 768, 1, 768]>]
    %cst_output_high = const.Declare tensor<1x768x1x1xf16> = dense<0.882825195> : tensor<768x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 768, 1, 1]>]
    %cst_output_low = const.Declare tensor<1x768x1x1xf16> = dense<0.882825195> : tensor<768x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 768, 1, 1]>]
    %cst_output_high_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.882825195> : tensor<1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
    %cst_output_low_1 = const.Declare tensor<1x1x1x1xf16> = dense<-0.889776587> : tensor<1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
    %cst_input_high_1 = const.Declare tensor<1x1x1x768xf16> = dense<0.882825195> : tensor<1x768xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 768]>]
    %cst_input_low_1 = const.Declare tensor<1x1x1x768xf16> = dense<0.882825195> : tensor<1x768xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 768]>]
    %cst_input_high = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
    %cst_input_low = const.Declare tensor<1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
    %bias = const.Declare tensor<1x768x1x1xf16> = dense<-1.270000e+02> : tensor<1x768xf32>, [#const.Reshape<[1, 768, 1, 1]>, #const.CastElemType<f16>]
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 128, 768]} : tensor<1x128x768x1xf16> -> tensor<1x128x768xf16>
    %1 = IE.Slice %0 [0, 0, 0] [1, 1, 768] : tensor<1x128x768xf16> to tensor<1x1x768xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 768]} : tensor<1x1x768xf16> -> tensor<1x1x1x768xf16>
    %3 = IE.FakeQuantize(%2, %cst_input_low_1, %cst_input_high_1, %cst_output_low_1, %cst_output_high_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x768xf16>, tensor<1x1x1x768xf16>, tensor<1x1x1x768xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x768xf16>
    %4 = IE.FakeQuantize(%cst_input_fq, %cst_input_low, %cst_input_high, %cst_output_low, %cst_output_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x768x1x768xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x768x1x1xf16>, tensor<1x768x1x1xf16> -> tensor<1x768x1x768xf16>
    %5 = IE.AffineReshape(%3) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 768, 1, 1]} : tensor<1x1x1x768xf16> -> tensor<1x768x1x1xf16>
    %6 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [768, 768, 1, 1]} : tensor<1x768x1x768xf16> -> tensor<768x768x1x1xf16>
    %7 = IE.Convolution(%5, %6, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x768x1x1xf16>, tensor<768x768x1x1xf16>, tensor<1x768x1x1xf16> -> tensor<1x768x1x1xf16>

    return %7 : tensor<1x768x1x1xf16>

    // CHECK:      [[cst_FQ1_input_reshaped:%.*]] = const.Declare tensor<1x768x1x1xf16> = dense<0.882825195> : tensor<1x768xf32>, [#const.Reshape<[1, 768, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:      [[cst_FQ0_input:%.*]] = const.Declare tensor<1x768x1x768xf16> = dense<0.882825195> : tensor<768x768xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 768, 1, 768]>]
    // CHECK:      [[cst_FQ0_output_high_low:%.*]] = const.Declare tensor<1x768x1x1xf16> = dense<0.882825195> : tensor<768x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 768, 1, 1]>]
    // CHECK:      [[cst_FQ1_input_high:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.882825195> : tensor<1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
    // CHECK:      [[cst_FQ1_input_low:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-0.889776587> : tensor<1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
    // CHECK:      [[cst_FQ0_input_high:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
    // CHECK:      [[cst_FQ0_input_low:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
    // CHECK:      [[bias:%.*]] = const.Declare tensor<1x768x1x1xf16> = dense<-1.270000e+02> : tensor<1x768xf32>, [#const.Reshape<[1, 768, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:      [[AffineReshape:%.*]] = IE.AffineReshape(%arg0)
    // CHECK{LITERAL}:      {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 128, 768]} : tensor<1x128x768x1xf16> -> tensor<1x128x768xf16>

    // CHECK:      [[SLICE:%.*]] = IE.Slice [[AffineReshape:%.*]] [0, 0, 0] [1, 1, 768] : tensor<1x128x768xf16> to tensor<1x1x768xf16>
    // CHECK:      [[AffineReshape0:%.*]] = IE.AffineReshape([[SLICE]])
    // CHECK{LITERAL}:      {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 768]} : tensor<1x1x768xf16> -> tensor<1x1x1x768xf16>

    // CHECK:      [[FQ0:%.*]] = IE.FakeQuantize([[cst_FQ0_input]], [[cst_FQ0_input_low]], [[cst_FQ0_input_high]], [[cst_FQ0_output_high_low]], [[cst_FQ0_output_high_low]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x768x1x768xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x768x1x1xf16>, tensor<1x768x1x1xf16> -> tensor<1x768x1x768xf16>
    // CHECK:      [[AffineReshape1:%.*]] = IE.AffineReshape([[AffineReshape0]])
    // CHECK{LITERAL}:      {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 768, 1, 1]} : tensor<1x1x1x768xf16> -> tensor<1x768x1x1xf16>
    // CHECK:      [[AffineReshape2:%.*]] = IE.AffineReshape([[FQ0]])
    // CHECK{LITERAL}:      {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [768, 768, 1, 1]} : tensor<1x768x1x768xf16> -> tensor<768x768x1x1xf16>
    // CHECK:      [[FQ1:%.*]] = IE.FakeQuantize([[AffineReshape1]], [[cst_FQ1_input_reshaped]], [[cst_FQ1_input_reshaped]], [[cst_FQ1_input_low]], [[cst_FQ1_input_high]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x768x1x1xf16>, tensor<1x768x1x1xf16>, tensor<1x768x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x768x1x1xf16>

    // CHECK:      [[CONVOLUTION:%.*]] = IE.Convolution([[FQ1]], [[AffineReshape2]], [[bias]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x768x1x1xf16>, tensor<768x768x1x1xf16>, tensor<1x768x1x1xf16> -> tensor<1x768x1x1xf16>

    // CHECK:      return [[CONVOLUTION]] : tensor<1x768x1x1xf16>
}
