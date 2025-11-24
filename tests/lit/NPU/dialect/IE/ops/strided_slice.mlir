//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @ConvertConstToAttr
func.func @ConvertConstToAttr(%arg0: tensor<1x10x20x30xf16>) -> tensor<1x10x10x30xf16> {
    %begins = const.Declare tensor<4xsi64> = dense<[0, 0, 0, 0]> : tensor<4xsi64>
    %ends = const.Declare tensor<4xsi64> = dense<[1, 5, 10, 20]> : tensor<4xsi64>
    %strides = const.Declare tensor<4xsi64> = dense<[1, 1, 1, 1]> : tensor<4xsi64>

    %0 = IE.StridedSlice(%arg0, %begins, %ends, %strides) {
        begin_mask = [0, 1, 1, 0],
        end_mask = [0, 1, 0, 1],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 1, 1, 1>
    } : tensor<1x10x20x30xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<4xsi64> -> tensor<1x10x10x30xf16>

    return %0 : tensor<1x10x10x30xf16>
    // CHECK:       %[[VAL0:.*]] = IE.StridedSlice(%arg0)
    // CHECK-SAME:  begins_attr = [0, 0, 0, 0]
    // CHECK-SAME:  ends_attr = [1, 5, 10, 20]
    // CHECK-SAME:  strides_attr = [1, 1, 1, 1]
}

// -----

//CHECK-LABEL: @NoComposeOnDifferentStrides
func.func @NoComposeOnDifferentStrides(%arg0: tensor<1x3x640x640xf16>) -> tensor<1x3x320x320xf16> {
    %begins1 = const.Declare tensor<4xsi64> = dense<[0, 0, 0, 0]> : tensor<4xsi64>
    %ends1 = const.Declare tensor<4xsi64> = dense<[0, 0, 2147483647, 0]> : tensor<4xsi64>
    %strides1 = const.Declare tensor<4xsi64> = dense<[1, 1, 2, 1]> : tensor<4xsi64>
    %begins2 = const.Declare tensor<4xsi64> = dense<[0, 0, 0, 0]> : tensor<4xsi64>
    %ends2 = const.Declare tensor<4xsi64> = dense<[0, 0, 0, 2147483647]> : tensor<4xsi64>
    %strides2 = const.Declare tensor<4xsi64> = dense<[1, 1, 1, 2]> : tensor<4xsi64>

    %0 = IE.StridedSlice(%arg0, %begins1, %ends1, %strides1) {
        begin_mask = [1, 1, 1, 1],
        end_mask = [1, 1, 0, 1],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 1, 1, 1>
    } : tensor<1x3x640x640xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<4xsi64> -> tensor<1x3x320x640xf16>
    %1 = IE.StridedSlice(%0, %begins2, %ends2, %strides2) {
        begin_mask = [1, 1, 1, 1],
        end_mask = [1, 1, 1, 0],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 1, 1, 1>
    } : tensor<1x3x320x640xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<4xsi64> -> tensor<1x3x320x320xf16>

    return %1 : tensor<1x3x320x320xf16>
    // CHECK:       [[VAL0:%.*]] = IE.StridedSlice(%arg0)
    // CHECK:       [[VAL1:%.*]] = IE.StridedSlice([[VAL0]])
}

// -----

// For 2 level StridedSlice graph as show below. The StridedSlice could be composed into 1 level.
//
//          -----------------------(act)--------------------
//         |                |                 |                 |
//  (StridedSlice)   (StridedSlice)    (StridedSlice)    (StridedSlice)
//         |                |                 |                 |
//  (StridedSlice)   (StridedSlice)    (StridedSlice)    (StridedSlice)
//         |                |                 |                 |
//          -----------------------(concat)--------------------
// Compose to:
//
//          -----------------------(act)--------------------
//         |                |                 |                 |
//  (StridedSlice)   (StridedSlice)    (StridedSlice)    (StridedSlice)
//         |                |                 |                 |
//          -----------------------(concat)--------------------
//
// The result graph could be handled by subsequent optimizations.

//CHECK-LABEL: @ComposeStridedSlices
func.func @ComposeStridedSlices(%arg0: tensor<1x3x640x640xf16>) -> tensor<1x12x320x320xf16> {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    %1 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    %2 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    %3 = IE.StridedSlice(%2) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    %4 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    %5 = IE.StridedSlice(%4) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    %6 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    %7 = IE.StridedSlice(%6) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    %8 = IE.Concat(%1, %3, %5, %7) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]} : tensor<1x3x320x320xf16>, tensor<1x3x320x320xf16>, tensor<1x3x320x320xf16>, tensor<1x3x320x320xf16> -> tensor<1x12x320x320xf16>

    return %8 : tensor<1x12x320x320xf16>

    // CHECK:     [[SLICE_1:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x320xf16>
    // CHECK:     [[SLICE_2:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 641, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x320xf16>
    // CHECK:     [[SLICE_3:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x320xf16>
    // CHECK:     [[SLICE_4:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 641, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x320xf16>
    // CHECK:     [[CONCAT:%.*]] = IE.Concat([[SLICE_1]], [[SLICE_2]], [[SLICE_3]], [[SLICE_4]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]} : tensor<1x3x320x320xf16>, tensor<1x3x320x320xf16>, tensor<1x3x320x320xf16>, tensor<1x3x320x320xf16> -> tensor<1x12x320x320xf16>

    // CHECK: return [[CONCAT]] : tensor<1x12x320x320xf16>
}

// -----

// For 2 level StridedSlice graph as show below. The StridedSlice could be composed into 1 level.
//
//          -----------------------(act)--------------------
//                 |                                   |
//           (StridedSlice)                     (StridedSlice)
//            |        |                         |         |
//  (StridedSlice)   (StridedSlice)    (StridedSlice)    (StridedSlice)
//         |                |                 |                 |
//          -----------------------(concat)--------------------
// Compose to:
//
//          -----------------------(act)--------------------
//         |                |                 |                 |
//  (StridedSlice)   (StridedSlice)    (StridedSlice)    (StridedSlice)
//         |                |                 |                 |
//          -----------------------(concat)--------------------
//
// The result graph could be handled by subsequent optimizations.

//CHECK-LABEL: @ComposeStridedSlicesOneStridedSliceConnectToTwoStridedSlice
func.func @ComposeStridedSlicesOneStridedSliceConnectToTwoStridedSlice(%arg0: tensor<1x3x416x416xf16>) -> tensor<1x12x208x208xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<2.541950e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x416x416xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x416x416xf16>
    %1 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x416xf16>
    %2 = IE.StridedSlice(%1) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 208, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x208x416xf16> -> tensor<1x3x208x208xf16>
    %3 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x416xf16>
    %4 = IE.StridedSlice(%3) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 208, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x208x416xf16> -> tensor<1x3x208x208xf16>
    %5 = IE.StridedSlice(%1) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 208, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x208x416xf16> -> tensor<1x3x208x208xf16>
    %6 = IE.StridedSlice(%3) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 208, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x208x416xf16> -> tensor<1x3x208x208xf16>
    %7 = IE.Concat(%2, %4, %5, %6) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]} : tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16> -> tensor<1x12x208x208xf16>

    return %7 : tensor<1x12x208x208xf16>

    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.541950e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:     [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x416x416xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x416x416xf16>
    // CHECK:     [[SLICE_1:%.*]] = IE.StridedSlice([[FQ]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    // CHECK:     [[SLICE_2:%.*]] = IE.StridedSlice([[FQ]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 417, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    // CHECK:     [[SLICE_3:%.*]] = IE.StridedSlice([[FQ]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    // CHECK:     [[SLICE_4:%.*]] = IE.StridedSlice([[FQ]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 417, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>

    // CHECK: [[CONCAT:%.*]] = IE.Concat([[SLICE_1]], [[SLICE_2]], [[SLICE_3]], [[SLICE_4]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]} : tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16> -> tensor<1x12x208x208xf16>
    // CHECK: return [[CONCAT]] : tensor<1x12x208x208xf16>

}

// -----

// CHECK-LABEL: @ConvertNegStrideStridedSlice2Reverse
// CHECK: [[INPUT:%.+]]: tensor<1x3x50x128xf16>
func.func @ConvertNegStrideStridedSlice2Reverse(%arg0: tensor<1x3x50x128xf16>) -> tensor<1x3x50x128xf16> {
    %cst = const.Declare tensor<4xsi64> = dense<[0, 0, -1, 0]> : tensor<4xsi64>
    %cst_0 = const.Declare tensor<4xsi64> = dense<[0, 3, -9223372036854775807, 128]> : tensor<4xsi64>
    %cst_1 = const.Declare tensor<4xsi64> = dense<[1, 1, -1, 1]> : tensor<4xsi64>
    %slice = IE.StridedSlice(%arg0, %cst, %cst_0, %cst_1) {
            begin_mask = [1, 0, 0, 0],
            ellipsis_mask = [],
            end_mask = [1, 0, 0, 0],
            new_axis_mask = [],
            operandSegmentSizes = array<i32: 1, 1, 1, 1>,
            shrink_axis_mask = []
        } : tensor<1x3x50x128xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<4xsi64> -> tensor<1x3x50x128xf16>
    return %slice : tensor<1x3x50x128xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xsi32> = dense<50> : tensor<1xsi64>, [#const.CastElemType<si32>]
    // CHECK:     [[REVERSE_SEQ:%.+]] = IE.ReverseSequence([[INPUT]], [[CST]]) {batch_axis = 0 : i64, seq_axis = 2 : i64} : tensor<1x3x50x128xf16>, tensor<1xsi32> -> tensor<1x3x50x128xf16>

    // CHECK:     return [[REVERSE_SEQ]] : tensor<1x3x50x128xf16>
}

// -----

// CHECK-LABEL: @NoReverseForFirstNonOneAxis
// CHECK: [[INPUT:%.+]]: tensor<1x3x128x128xf16>
func.func @NoReverseForFirstNonOneAxis(%arg0: tensor<1x3x128x128xf16>) -> tensor<1x3x128x128xf16> {
    %begins = const.Declare tensor<4xsi64> = dense<[0, -1, 0, 0]> : tensor<4xsi64>
    %ends = const.Declare tensor<4xsi64> = dense<[0, -9223372036854775807, 128, 128]> : tensor<4xsi64>
    %strides = const.Declare tensor<4xsi64> = dense<[1, -1, 1, 1]> : tensor<4xsi64>

    %slice = IE.StridedSlice(%arg0, %begins, %ends, %strides) {
        begin_mask = [1, 0, 0, 0],
        end_mask = [1, 0, 0, 0],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 1, 1, 1>
    } : tensor<1x3x128x128xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<4xsi64> -> tensor<1x3x128x128xf16>
    return %slice : tensor<1x3x128x128xf16>

    // CHECK: [[STRIDED_SLICE:%.+]] = IE.StridedSlice([[INPUT]]) {
    // CHECK-SAME: begin_mask = [1, 0, 0, 0],
    // CHECK-SAME: begins_attr = [0, -1, 0, 0],
    // CHECK-SAME: ellipsis_mask = [0, 0, 0, 0],
    // CHECK-SAME: end_mask = [1, 0, 0, 0],
    // CHECK-SAME: ends_attr = [0, -9223372036854775807, 128, 128],
    // CHECK-SAME: new_axis_mask = [0, 0, 0, 0],
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    // CHECK-SAME: shrink_axis_mask = [0, 0, 0, 0],
    // CHECK-SAME: strides_attr = [1, -1, 1, 1]
    // CHECK-SAME: } : tensor<1x3x128x128xf16> -> tensor<1x3x128x128xf16>

    // CHECK: return [[STRIDED_SLICE]] : tensor<1x3x128x128xf16>
}
