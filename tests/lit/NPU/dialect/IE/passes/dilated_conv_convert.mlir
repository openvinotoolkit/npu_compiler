//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dilated-conv-convert %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: func.func @DilatedGroupConvConvert
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1344x33x33xf32>)
func.func @DilatedGroupConvConvert(%arg0: tensor<1x1344x33x33xf32>) -> tensor<1x1344x33x33xf32> {
    %cst = const.Declare tensor<1344x1x3x3xf32> = dense<0.100000e+00> : tensor<1344x1x1x3x3xf32>, [#const.Reshape<[1344, 1, 3, 3]>]
    %0 = IE.SpaceToBatch(%arg0) {
            block_shape_value = [1, 1, 2, 2],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>,
            pads_begin_value = [0, 0, 2, 2],
            pads_end_value = [0, 0, 3, 3]
        } : tensor<1x1344x33x33xf32> -> tensor<4x1344x19x19xf32>
    %1 = IE.GroupConvolution(%0, %cst) {
            dilations = [1, 1], groups = 1344 : i64,
            pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<4x1344x19x19xf32>, tensor<1344x1x3x3xf32> -> tensor<4x1344x17x17xf32>
    %2 = IE.BatchToSpace(%1) {
            block_shape_value = [1, 1, 2, 2],
            crops_begin_value = [0, 0, 0, 0],
            crops_end_value = [0, 0, 1, 1],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>
        } : tensor<4x1344x17x17xf32> -> tensor<1x1344x33x33xf32>

    return %2 : tensor<1x1344x33x33xf32>

    // CHECK-NOT:   IE.SpaceToBatch

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1344x1x3x3xf32> = dense<1.000000e-01> : tensor<1344x1x1x3x3xf32>, [#const.Reshape<[1344, 1, 3, 3]>]
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[INPUT_DATA]], [[CST]]) {
    // CHECK-SAME:          dilations = [2, 2], groups = 1344 : i64,
    // CHECK-SAME:          pads_begin = [2, 2], pads_end = [2, 2], strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x1344x33x33xf32>, tensor<1344x1x3x3xf32> -> tensor<1x1344x33x33xf32>

    // CHECK:       return [[GROUP_CONV]] : tensor<1x1344x33x33xf32>
}

//
// -----
//

// CHECK-LABEL: func.func @DilatedGroupConvConvertWithQuantized
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1344x33x33xf32>)
func.func @DilatedGroupConvConvertWithQuantized(%arg0: tensor<1x1344x33x33xf32>) -> tensor<1x1344x33x33xf32> {
    %cst = const.Declare tensor<1x1344x1x1xf32> = dense<0.000000e+00> : tensor<1x1344x1x1xf32>
    %cst_0 = const.Declare tensor<1x1344x1x1xf32> = dense<1.000000e+00> : tensor<1x1344x1x1xf32>
    %cst_1 = const.Declare tensor<1344x1x3x3xf32> = dense<0.100000e+00> : tensor<1344x1x3x3xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_4 = const.Declare tensor<1344x1x1x1xf32> = dense<0.000000e+00> : tensor<1344x1x1x1xf32>
    %cst_5 = const.Declare tensor<1344x1x1x1xf32> = dense<1.000000e+00> : tensor<1344x1x1x1xf32>

    %0 = IE.SpaceToBatch(%arg0) {
            block_shape_value = [1, 1, 2, 2],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>,
            pads_begin_value = [0, 0, 2, 2],
            pads_end_value = [0, 0, 3, 3]
        } : tensor<1x1344x33x33xf32> -> tensor<4x1344x19x19xf32>
    %1 = IE.FakeQuantize(%0, %cst, %cst_0, %cst, %cst_0) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
        } : tensor<4x1344x19x19xf32>, tensor<1x1344x1x1xf32>, tensor<1x1344x1x1xf32>, tensor<1x1344x1x1xf32>, tensor<1x1344x1x1xf32> -> tensor<4x1344x19x19xf32>
    %2 = IE.FakeQuantize(%cst_1, %cst_2, %cst_3, %cst_4, %cst_5) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
        } : tensor<1344x1x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1344x1x1x1xf32>, tensor<1344x1x1x1xf32> -> tensor<1344x1x3x3xf32>
    %3 = IE.GroupConvolution(%1, %2) {
            dilations = [1, 1], groups = 1344 : i64,
            pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<4x1344x19x19xf32>, tensor<1344x1x3x3xf32> -> tensor<4x1344x17x17xf32>
    %4 = IE.BatchToSpace(%3) {
            block_shape_value = [1, 1, 2, 2],
            crops_begin_value = [0, 0, 0, 0],
            crops_end_value = [0, 0, 1, 1],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>
        } : tensor<4x1344x17x17xf32> -> tensor<1x1344x33x33xf32>


    return %4 : tensor<1x1344x33x33xf32>

    // CHECK-NOT:   IE.SpaceToBatch

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1344x1x1xf32> = dense<0.000000e+00> : tensor<1x1344x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1344x1x1xf32> = dense<1.000000e+00> : tensor<1x1344x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1344x1x3x3xf32> = dense<1.000000e-01> : tensor<1344x1x3x3xf32>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_4:%.+]] = const.Declare tensor<1344x1x1x1xf32> = dense<0.000000e+00> : tensor<1344x1x1x1xf32>
    // CHECK-DAG:   [[CST_5:%.+]] = const.Declare tensor<1344x1x1x1xf32> = dense<1.000000e+00> : tensor<1344x1x1x1xf32>

    // CHECK:       [[INPUT_FQ:%.+]] = IE.FakeQuantize([[INPUT_DATA]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      } : tensor<1x1344x33x33xf32>, tensor<1x1344x1x1xf32>, tensor<1x1344x1x1xf32>, tensor<1x1344x1x1xf32>, tensor<1x1344x1x1xf32> -> tensor<1x1344x33x33xf32>
    // CHECK:       [[WEIGHTS_FQ:%.+]] = IE.FakeQuantize([[CST_1]], [[CST_2]], [[CST_3]], [[CST_4]], [[CST_5]]) {
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:      } : tensor<1344x1x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1344x1x1x1xf32>, tensor<1344x1x1x1xf32> -> tensor<1344x1x3x3xf32>

    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[INPUT_FQ]], [[WEIGHTS_FQ]]) {
    // CHECK-SAME:          dilations = [2, 2], groups = 1344 : i64,
    // CHECK-SAME:          pads_begin = [2, 2], pads_end = [2, 2], strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x1344x33x33xf32>, tensor<1344x1x3x3xf32> -> tensor<1x1344x33x33xf32>

    // CHECK:       return [[GROUP_CONV]] : tensor<1x1344x33x33xf32>
}

//
// -----
//

// CHECK-LABEL: func.func @DilatedConvConvert
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x33x33xf32>)
func.func @DilatedConvConvert(%arg0: tensor<1x16x33x33xf32>) -> tensor<1x16x33x33xf32> {
    %cst = const.Declare tensor<16x16x3x3xf32> = dense<0.100000e+00> : tensor<16x16x3x3xf32>
    %0 = IE.SpaceToBatch(%arg0) {
            block_shape_value = [1, 1, 2, 2],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>,
            pads_begin_value = [0, 0, 2, 2],
            pads_end_value = [0, 0, 3, 3]
        } : tensor<1x16x33x33xf32> -> tensor<4x16x19x19xf32>
    %1 = IE.Convolution(%0, %cst) {
            pads_begin = [0, 0], pads_end = [0, 0],
            dilations = [1, 1], strides = [1, 1]
        } : tensor<4x16x19x19xf32>, tensor<16x16x3x3xf32> -> tensor<4x16x17x17xf32>
    %2 = IE.BatchToSpace(%1) {
            block_shape_value = [1, 1, 2, 2],
            crops_begin_value = [0, 0, 0, 0],
            crops_end_value = [0, 0, 1, 1],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>
        } : tensor<4x16x17x17xf32> -> tensor<1x16x33x33xf32>

    return %2 : tensor<1x16x33x33xf32>

    // CHECK-NOT:   IE.SpaceToBatch

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x16x3x3xf32> = dense<1.000000e-01> : tensor<16x16x3x3xf32>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT_DATA]], [[CST]]) {
    // CHECK-SAME:          dilations = [2, 2], pads_begin = [2, 2],
    // CHECK-SAME:          pads_end = [2, 2], strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x16x33x33xf32>, tensor<16x16x3x3xf32> -> tensor<1x16x33x33xf32>

    // CHECK:       return [[CONV]] : tensor<1x16x33x33xf32>
}

//
// -----
//

// CHECK-LABEL: func.func @DilatedConvConvertWithQuantized
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x33x33xf32>)
func.func @DilatedConvConvertWithQuantized(%arg0: tensor<1x16x33x33xf32>) -> tensor<1x16x33x33xf32> {
    %cst = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x16x1x1xf32>
    %cst_0 = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>
    %cst_1 = const.Declare tensor<16x16x3x3xf32> = dense<0.100000e+00> : tensor<16x16x3x3xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_4 = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
    %cst_5 = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>

    %0 = IE.SpaceToBatch(%arg0) {
            block_shape_value = [1, 1, 2, 2],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>,
            pads_begin_value = [0, 0, 2, 2],
            pads_end_value = [0, 0, 3, 3]
        } : tensor<1x16x33x33xf32> -> tensor<4x16x19x19xf32>
    %1 = IE.FakeQuantize(%0, %cst, %cst_0, %cst, %cst_0) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
        } : tensor<4x16x19x19xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<4x16x19x19xf32>
    %2 = IE.FakeQuantize(%cst_1, %cst_2, %cst_3, %cst_4, %cst_5) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
        } : tensor<16x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32> -> tensor<16x16x3x3xf32>
    %3 = IE.Convolution(%1, %2) {
            dilations = [1, 1], strides = [1, 1],
            pads_begin = [0, 0], pads_end = [0, 0]
        } : tensor<4x16x19x19xf32>, tensor<16x16x3x3xf32> -> tensor<4x16x17x17xf32>
    %4 = IE.BatchToSpace(%3) {
            block_shape_value = [1, 1, 2, 2],
            crops_begin_value = [0, 0, 0, 0],
            crops_end_value = [0, 0, 1, 1],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>
        } : tensor<4x16x17x17xf32> -> tensor<1x16x33x33xf32>


    return %4 : tensor<1x16x33x33xf32>

    // CHECK-NOT:   IE.SpaceToBatch

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<16x16x3x3xf32> = dense<1.000000e-01> : tensor<16x16x3x3xf32>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_4:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK-DAG:   [[CST_5:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[INPUT_FQ:%.+]] = IE.FakeQuantize([[INPUT_DATA]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      } : tensor<1x16x33x33xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x33x33xf32>
    // CHECK:       [[WEIGHTS_FQ:%.+]] = IE.FakeQuantize([[CST_1]], [[CST_2]], [[CST_3]], [[CST_4]], [[CST_5]]) {
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:      } : tensor<16x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32> -> tensor<16x16x3x3xf32>

    // CHECK:       [[GROUP_CONV:%.+]] = IE.Convolution([[INPUT_FQ]], [[WEIGHTS_FQ]]) {
    // CHECK-SAME:          dilations = [2, 2], pads_begin = [2, 2],
    // CHECK-SAME:          pads_end = [2, 2], strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x16x33x33xf32>, tensor<16x16x3x3xf32> -> tensor<1x16x33x33xf32>

    // CHECK:       return [[GROUP_CONV]] : tensor<1x16x33x33xf32>
}
