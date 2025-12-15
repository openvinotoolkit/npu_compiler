//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-fake-quant-with-reshape-and-strided-slice %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @SwapFakeQuantReshapeF8E4M3FN
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x1x40xf16>
// CHECK-SAME:   [[WEIGHTS:%.+]]: tensor<512x40x1x1xf16>
func.func @SwapFakeQuantReshapeF8E4M3FN(%input: tensor<1x1x40xf16>, %weights: tensor<512x40x1x1xf16>) -> tensor<1x512x1x1xf16> {
    %low = const.Declare tensor<f16> = dense<-4.480000e+02> : tensor<f16>
    %high = const.Declare tensor<f16> = dense<4.480000e+02> : tensor<f16>

    %0 = IE.SoftMax(%input) {axisInd = 2} : tensor<1x1x40xf16> -> tensor<1x1x40xf16>
    %1 = IE.AffineReshape(%0) {shape_value = [1, 1, 1, 40], dim_mapping = [[0], [1, 2], [3]]} : tensor<1x1x40xf16> -> tensor<1x1x1x40xf16>

    %2 = IE.FakeQuantize(%1, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x1x1x40xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x1x1x40xf16>

    %3 = IE.AffineReshape(%2) {shape_value = [1, 40, 1, 1], dim_mapping = [[0], [0], [0], [1, 2, 3]]} : tensor<1x1x1x40xf16> -> tensor<1x40x1x1xf16>
    %4 = IE.Convolution(%3, %weights) {strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]} : tensor<1x40x1x1xf16>, tensor<512x40x1x1xf16> -> tensor<1x512x1x1xf16>

    return %4 : tensor<1x512x1x1xf16>

    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<f16> = dense<-4.480000e+02> : tensor<f16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<f16> = dense<4.480000e+02> : tensor<f16>

    // CHECK:    [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 2 : i64} : tensor<1x1x40xf16> -> tensor<1x1x40xf16>

    // CHECK:    [[RESHAPE_0:%.+]] = IE.AffineReshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 40]} : tensor<1x1x40xf16> -> tensor<1x1x1x40xf16>
    // CHECK:    [[RESHAPE_1:%.+]] = IE.AffineReshape([[RESHAPE_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 40, 1, 1]} : tensor<1x1x1x40xf16> -> tensor<1x40x1x1xf16>

    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[RESHAPE_1]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x40x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x40x1x1xf16>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[FQ]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x40x1x1xf16>, tensor<512x40x1x1xf16> -> tensor<1x512x1x1xf16>

    // CHECK:    return [[CONV]] : tensor<1x512x1x1xf16>
}

// -----

// CHECK-LABEL: @SwapFakeQuantWithStridedSliceF8E5M2
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x3x640x640xf16>
func.func @SwapFakeQuantWithStridedSliceF8E5M2(%input: tensor<1x3x640x640xf16>) -> tensor<1x6x320x320xf16> {
    %low = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%input, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x3x640x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x640x640xf16>

    %1 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    %2 = IE.StridedSlice(%1) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    %3 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    %4 = IE.StridedSlice(%3) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    %5 = IE.Concat(%2, %4) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x320x320xf16>, tensor<1x3x320x320xf16> -> tensor<1x6x320x320xf16>

    return %5 : tensor<1x6x320x320xf16>

    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf16>

    // CHECK:    [[SLICE_0:%.+]] = IE.StridedSlice([[INPUT]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    // CHECK:    [[SLICE_1:%.+]] = IE.StridedSlice([[SLICE_0]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    // CHECK:    [[FQ_0:%.+]] = IE.FakeQuantize([[SLICE_1]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x3x320x320xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x320x320xf16>

    // CHECK:    [[SLICE_2:%.+]] = IE.StridedSlice([[INPUT]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    // CHECK:    [[SLICE_3:%.+]] = IE.StridedSlice([[SLICE_2]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    // CHECK:    [[FQ_1:%.+]] = IE.FakeQuantize([[SLICE_3]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x3x320x320xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x320x320xf16>

    // CHECK:    [[CONCAT:%.+]] = IE.Concat([[FQ_0]], [[FQ_1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x320x320xf16>, tensor<1x3x320x320xf16> -> tensor<1x6x320x320xf16>

    // CHECK:    return [[CONCAT]] : tensor<1x6x320x320xf16>
}
