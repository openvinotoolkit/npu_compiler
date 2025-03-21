//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-strided-slice-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvertStridedSlice2ConvOnH
func.func @ConvertStridedSlice2ConvOnH(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}> {
    %1 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16, {order = #NHWC}> -> tensor<1x3x320x640xf16, {order = #NHWC}>
    return %1 : tensor<1x3x320x640xf16, {order = #NHWC}>

    // CHECK-NOT: IE.StridedSlice
    // CHECK:            [[CST:%.*]] = const.Declare tensor<3x3x1x1xf16> = dense<
    // CHECK-SAME{LITERAL}: [[[[1.000000e+00]], [[0.000000e+00]], [[0.000000e+00]]],
    // CHECK-SAME{LITERAL}: [[[0.000000e+00]], [[1.000000e+00]], [[0.000000e+00]]],
    // CHECK-SAME{LITERAL}: [[[0.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]]]> : tensor<3x3x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[CONV:%.*]] = IE.Convolution(%arg0, [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<3x3x1x1xf16> -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK: return [[CONV]] : tensor<1x3x320x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertForUnalignedChannels
func.func @ConvertForUnalignedChannels(%arg0: tensor<1x3x320x640xf16, {order = #NHWC}>) -> tensor<1x3x320x320xf16, {order = #NHWC}> {
    %1 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x3x320x320xf16, {order = #NHWC}>
    return %1 : tensor<1x3x320x320xf16, {order = #NHWC}>

    // CHECK:           [[CST0:%.*]] = const.Declare tensor<3x3x1x1xf16> = dense<
    // CHECK-SAME{LITERAL}: [[[[1.000000e+00]], [[0.000000e+00]], [[0.000000e+00]]],
    // CHECK-SAME{LITERAL}: [[[0.000000e+00]], [[1.000000e+00]], [[0.000000e+00]]],
    // CHECK-SAME{LITERAL}: [[[0.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]]]> : tensor<3x3x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[SLICE:%.*]] = IE.Slice %arg0 [0, 0, 0, 1] [1, 3, 320, 639] : tensor<1x3x320x640xf16, {order = #NHWC}> to tensor<1x3x320x639xf16, {order = #NHWC}>
    // CHECK: [[CONV:%.*]] = IE.Convolution([[SLICE]], [[CST0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]} : tensor<1x3x320x639xf16, {order = #NHWC}>, tensor<3x3x1x1xf16> -> tensor<1x3x320x320xf16, {order = #NHWC}>
    // CHECK: return [[CONV]] : tensor<1x3x320x320xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @ConvertStridedSlice2ConvWithStridesOnWH
func.func @ConvertStridedSlice2ConvWithStridesOnWH(%arg0: tensor<1x3x416x416xf16>) -> tensor<1x3x208x208xf16> {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %1 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16> -> tensor<1x3x208x208xf16>

    return %2 : tensor<1x3x208x208xf16>

    // CHECK:            [[CST0:%.*]] = const.Declare tensor<3x3x1x1xf16> = dense<
    // CHECK-SAME{LITERAL}: [[[[1.000000e+00]], [[0.000000e+00]], [[0.000000e+00]]],
    // CHECK-SAME{LITERAL}: [[[0.000000e+00]], [[1.000000e+00]], [[0.000000e+00]]],
    // CHECK-SAME{LITERAL}: [[[0.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]]]> : tensor<3x3x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[CONV_0:%.*]] = IE.Convolution(%arg0, [[CST0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x416x416xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x208x208xf16>
    // CHECK: [[SLICE:%.*]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 3, 415, 416] : tensor<1x3x416x416xf16> to tensor<1x3x415x416xf16>
    // CHECK: [[CONV_1:%.*]] = IE.Convolution([[SLICE]], [[CST0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x415x416xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x208x208xf16>
    // CHECK: [[ADD:%.*]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16> -> tensor<1x3x208x208xf16>
    // CHECK: return [[ADD]] : tensor<1x3x208x208xf16>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertStridedSliceWithFQ2Conv
func.func @ConvertStridedSliceWithFQ2Conv(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<3.18300e+00> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<3.068850e+00> : tensor<1x1x1x1xf16>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x640x640xf16, {order = #NHWC}>
    %1 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16, {order = #NHWC}> -> tensor<1x3x320x640xf16, {order = #NHWC}>
    return %1 : tensor<1x3x320x640xf16, {order = #NHWC}>

    // CHECK-NOT: IE.StridedSlice
    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<3x3x1x1xf16> = dense<{{\[\[\[\[}}1.000000e+00]], {{\[\[}}0.000000e+00]], {{\[\[}}0.000000e+00]]], {{\[\[\[}}0.000000e+00]], {{\[\[}}1.000000e+00]], {{\[\[}}0.000000e+00]]], {{\[\[\[}}0.000000e+00]], {{\[\[}}0.000000e+00]], {{\[\[}}1.000000e+00]]]]> : tensor<3x3x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
    // CHECK-DAG: [[CST1:%.+]] = const.Declare tensor<f16> = dense<2.540000e+02> : tensor<f16>
    // CHECK-DAG: [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.183590e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG: [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.068360e+00> : tensor<1x1x1x1xf16>
    // CHECK: [[FQ_0:%.+]] = IE.FakeQuantize(%arg0, [[CST3]], [[CST2]], [[CST3]], [[CST2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x640x640xf16, {order = #NHWC}>
    // CHECK: [[FQ_1:%.+]] = IE.FakeQuantize([[CST]], [[CST0]], [[CST1]], [[CST0]], [[CST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<3x3x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<3x3x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[FQ_0]], [[FQ_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<3x3x1x1xf16> -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK: return [[CONV]] : tensor<1x3x320x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotConvertStridedSliceIfInputTypeSI32
func.func @NotConvertStridedSliceIfInputTypeSI32(%arg0: tensor<1x3x640x640xsi32, {order = #NHWC}>) -> tensor<1x3x320x640xsi32, {order = #NHWC}> {
    %1 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xsi32, {order = #NHWC}> -> tensor<1x3x320x640xsi32, {order = #NHWC}>
    return %1 : tensor<1x3x320x640xsi32, {order = #NHWC}>

    // CHECK: [[SLICE:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xsi32, {order = #NHWC}> -> tensor<1x3x320x640xsi32, {order = #NHWC}>
    // CHECK: return [[SLICE]] : tensor<1x3x320x640xsi32, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @NotConvertStridedSlice2ConvIfStridesExceeding
// CHECK-SAME:       ([[ARG0:%.+]]: tensor<1x320x33x1xf16>)
func.func @NotConvertStridedSlice2ConvIfStridesExceeding(%arg0: tensor<1x320x33x1xf16>) -> tensor<1x320x3x1xf16> {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 320, 33, 1], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 16, 1]} : tensor<1x320x33x1xf16> -> tensor<1x320x3x1xf16>
    return %0 : tensor<1x320x3x1xf16>

    // CHECK: [[SLICE:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 320, 33, 1], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 16, 1]} : tensor<1x320x33x1xf16> -> tensor<1x320x3x1xf16>
    // CHECK: return [[SLICE]] : tensor<1x320x3x1xf16>

}

// -----

// CHECK-LABEL: @ConvertParallelStridedSlicesToConv
func.func @ConvertParallelStridedSlicesToConv(%arg0: tensor<1x3x416x416xf16>) -> tensor<1x12x208x208xf16> {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %1 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %2 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %3 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %4 = IE.Concat(%0, %1, %2, %3) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]} : tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16> -> tensor<1x12x208x208xf16>
    return %4 : tensor<1x12x208x208xf16>

    // CHECK:            [[CST:%.*]] = const.Declare tensor<12x3x2x2xf16> = dense<
    // CHECK-SAME{LITERAL}: "0x0000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F00000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F00000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F00000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F">
    // CHECK-SAME:               tensor<12x3x2x2xf32>, [#const.CastElemType<f16>]
    // CHECK: [[CONV:%.+]] = IE.Convolution(%arg0, [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x416x416xf16>, tensor<12x3x2x2xf16> -> tensor<1x12x208x208xf16>
    // CHECK: return [[CONV]] : tensor<1x12x208x208xf16>
}

// -----

// CHECK-LABEL: @ConvertParallelStridedSlicesToConvWithFQ
func.func @ConvertParallelStridedSlicesToConvWithFQ(%arg0: tensor<1x3x416x416xf16>) -> tensor<1x12x208x208xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<2.541950e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x416x416xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x416x416xf16>
    %1 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %2 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %3 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %4 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %5 = IE.Concat(%1, %2, %3, %4) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]} : tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16> -> tensor<1x12x208x208xf16>
    return %5 : tensor<1x12x208x208xf16>

    // CHECK-DAG:            [[CST:%.*]] = const.Declare tensor<12x3x2x2xf16> = dense<"0x0000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F00000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F00000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F00000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F">
    // CHECK-DAG-SAME:                          tensor<12x3x2x2xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:            [[CST_0:%.*]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
    // CHECK-DAG:            [[CST_1:%.*]] = const.Declare tensor<f16> = dense<2.540000e+02> : tensor<f16>
    // CHECK-DAG:            [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.541950e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:            [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[FQ_0:%.+]] = IE.FakeQuantize(%arg0, [[CST_3]], [[CST_2]], [[CST_3]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x416x416xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x416x416xf16>
    // CHECK: [[FQ_1:%.+]] = IE.FakeQuantize([[CST]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<12x3x2x2xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<12x3x2x2xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[FQ_0]], [[FQ_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x416x416xf16>, tensor<12x3x2x2xf16> -> tensor<1x12x208x208xf16>
    // CHECK: return [[CONV]] : tensor<1x12x208x208xf16>
}


// -----

// CHECK-LABEL: @NotConvertParallelStridedSlicesToConv
func.func @NotConvertParallelStridedSlicesToConv(%arg0: tensor<1x3x416x416xf16>) -> tensor<1x3x208x832xf16> {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %1 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %2 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %3 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %4 = IE.Concat(%0, %1, %2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 208], [0, 0, 0, 416], [0, 0, 0, 624]]} : tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16> -> tensor<1x3x208x832xf16>
    return %4 : tensor<1x3x208x832xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<3x3x1x1xf16> = dense<
    // CHECK-SAME{LITERAL}: [[[[1.000000e+00]], [[0.000000e+00]], [[0.000000e+00]]],
    // CHECK-SAME{LITERAL}: [[[0.000000e+00]], [[1.000000e+00]], [[0.000000e+00]]],
    // CHECK-SAME{LITERAL}: [[[0.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]]]> : tensor<3x3x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[CONV_0:%.*]] = IE.Convolution(%arg0, [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x416x416xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x208x208xf16>
    // CHECK: [[SLICE_1:%.*]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 3, 415, 416] : tensor<1x3x416x416xf16> to tensor<1x3x415x416xf16>
    // CHECK: [[CONV_1:%.*]] = IE.Convolution([[SLICE_1]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x415x416xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x208x208xf16>
    // CHECK: [[SLICE_2:%.*]] = IE.Slice %arg0 [0, 0, 0, 1] [1, 3, 416, 415] : tensor<1x3x416x416xf16> to tensor<1x3x416x415xf16>
    // CHECK: [[CONV_2:%.*]] = IE.Convolution([[SLICE_2]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x416x415xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x208x208xf16>
    // CHECK: [[SLICE_3:%.*]] = IE.Slice %arg0 [0, 0, 1, 1] [1, 3, 415, 415] : tensor<1x3x416x416xf16> to tensor<1x3x415x415xf16>
    // CHECK: [[CONV_3:%.*]] = IE.Convolution([[SLICE_3]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x415x415xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x208x208xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat([[CONV_0]], [[CONV_1]], [[CONV_2]], [[CONV_3]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 208], [0, 0, 0, 416], [0, 0, 0, 624]]} : tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16> -> tensor<1x3x208x832xf16>

    // CHECK: return [[CONCAT]] : tensor<1x3x208x832xf16>
}

// -----

// CHECK-LABEL: @NotConvertParallelStridedSlicesToConvWitnInputTypeSI32
func.func @NotConvertParallelStridedSlicesToConvWitnInputTypeSI32(%arg0: tensor<1x3x416x416xsi32>) -> tensor<1x12x208x208xsi32> {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xsi32> -> tensor<1x3x208x208xsi32>
    %1 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xsi32> -> tensor<1x3x208x208xsi32>
    %2 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xsi32> -> tensor<1x3x208x208xsi32>
    %3 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xsi32> -> tensor<1x3x208x208xsi32>
    %4 = IE.Concat(%0, %1, %2, %3) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]} : tensor<1x3x208x208xsi32>, tensor<1x3x208x208xsi32>, tensor<1x3x208x208xsi32>, tensor<1x3x208x208xsi32> -> tensor<1x12x208x208xsi32>
    return %4 : tensor<1x12x208x208xsi32>

    // CHECK: IE.StridedSlice
    // CHECK: IE.StridedSlice
    // CHECK: IE.StridedSlice
    // CHECK: IE.StridedSlice
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @ConvertForOutOfBoundEndAttrs
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x16x40x30xf16>
func.func @ConvertForOutOfBoundEndAttrs(%arg0: tensor<1x16x40x30xf16>)
        -> tensor<1x16x40x10xf16> {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [], end_mask = [0, 0, 0, 0], ends_attr = [1000000000, 1000000000, 1000000000, 1000000000], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 3]} : tensor<1x16x40x30xf16> -> tensor<1x16x40x10xf16>

    return %0 : tensor<1x16x40x10xf16>

    // CHECK-NOT: IE.StridedSlice

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<"0x0000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F"> : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 3]} : tensor<1x16x40x30xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x40x10xf16>
    // CHECK:       return [[CONV]] : tensor<1x16x40x10xf16>
}

// -----

// CHECK-LABEL: @NotConvertForNegativeEndAttrs
func.func @NotConvertForNegativeEndAttrs(%arg0: tensor<1x8x40x30xf16>)
        -> tensor<1x8x40x10xf16> {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [], end_mask = [0, 0, 0, 0], ends_attr = [1000000000, 1000000000, 1000000000, -1], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 3]} : tensor<1x8x40x30xf16> -> tensor<1x8x40x10xf16>

    return %0 : tensor<1x8x40x10xf16>
    // CHECK: IE.StridedSlice
}


// -----

// CHECK-LABEL: @NotConvertTooSmallSize
func.func @NotConvertTooSmallSize(%arg0: tensor<1x16x1x128xf16>) -> tensor<1x16x1x64xf16> {
    %1 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 16, 1, 128], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x16x1x128xf16> -> tensor<1x16x1x64xf16>
    return %1 : tensor<1x16x1x64xf16>

    // CHECK: IE.StridedSlice
}
