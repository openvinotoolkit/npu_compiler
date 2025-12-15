//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-branches-concat-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @OptimizeGroupConvConcat
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x144x144xf16>
func.func @OptimizeGroupConvConcat(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2 : tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[NEW_WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16>

    // CHECK:           [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[NEW_CONV:%.+]] = IE.Convolution([[CONV]], [[NEW_WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x144x144xf16>

    // CHECK:           return [[NEW_CONV]] : tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @OptimizeGroupConvConcat_SwapConcatInput
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x144x144xf16>
func.func @OptimizeGroupConvConcat_SwapConcatInput(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Concat(%1, %0) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2 : tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[NEW_WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16>

    // CHECK:           [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[NEW_CONV:%.+]] = IE.Convolution([[CONV]], [[NEW_WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x144x144xf16>

    // CHECK:           return [[NEW_CONV]] : tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @OptimizeGroupConvConcatWithReLu
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x144x144xf16>
func.func @OptimizeGroupConvConcatWithReLu(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.Relu<>,
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        post_op = #IE.Relu<>,
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2 : tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[NEW_WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16>

    // CHECK:           [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[NEW_CONV:%.+]] = IE.Convolution([[CONV]], [[NEW_WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x144x144xf16>

    // CHECK:           return [[NEW_CONV]] : tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @NotConvertIfRootTensorIsNotReLu
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x144x144xf16>
func.func @NotConvertIfRootTensorIsNotReLu(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        post_op = #IE.Relu<>,
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2 : tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[WEIGHTS_0:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    // CHECK-DAG:       [[WEIGHTS_1:%.+]] = const.Declare tensor<16x1x3x3xf16> = dense<1.000000e+00> : tensor<16x1x3x3xf16>

    // CHECK:           [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[GROUP_CONV:%.+]] = IE.GroupConvolution([[CONV]], [[WEIGHTS_1]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>

    // CHECK:           [[CONCAT:%.+]] = IE.Concat([[CONV]], [[GROUP_CONV]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16

    // CHECK:           return [[CONCAT]] : tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @NotConvertIfPostOpIsNotReLu
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x144x144xf16>
func.func @NotConvertIfPostOpIsNotReLu(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.Sigmoid<>,
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        post_op = #IE.Sigmoid<>,
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2 : tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[WEIGHTS_0:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    // CHECK-DAG:       [[WEIGHTS_1:%.+]] = const.Declare tensor<16x1x3x3xf16> = dense<1.000000e+00> : tensor<16x1x3x3xf16>

    // CHECK:           [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Sigmoid<>, strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[GROUP_CONV:%.+]] = IE.GroupConvolution([[CONV]], [[WEIGHTS_1]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.Sigmoid<>, strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>

    // CHECK:           [[CONCAT:%.+]] = IE.Concat([[CONV]], [[GROUP_CONV]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16

    // CHECK:           return [[CONCAT]] : tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @NotConvertIfConcatInputsDoNotHaveSameRoot
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x16x144x144xf16>,
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<1x16x144x144xf16>
func.func @NotConvertIfConcatInputsDoNotHaveSameRoot(%arg0: tensor<1x16x144x144xf16>, %arg1: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%arg1, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2 : tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[WEIGHTS_0:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    // CHECK-DAG:       [[WEIGHTS_1:%.+]] = const.Declare tensor<16x1x3x3xf16> = dense<1.000000e+00> : tensor<16x1x3x3xf16>

    // CHECK:           [[CONV:%.+]] = IE.Convolution([[INPUT0]], [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[GROUP_CONV:%.+]] = IE.GroupConvolution([[INPUT1]], [[WEIGHTS_1]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>

    // CHECK:           [[CONCAT:%.+]] = IE.Concat([[CONV]], [[GROUP_CONV]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16

    // CHECK:           return [[CONCAT]] : tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @OptimizeGroupConvConcatWhenRootTensorHasExtraUser
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x144x144xf16>
func.func @OptimizeGroupConvConcatWhenRootTensorHasExtraUser(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x16x144x144xf16>, tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Tanh(%0) : tensor<1x16x144x144xf16> -> tensor<1x16x144x144xf16>
    %3 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2, %3 : tensor<1x16x144x144xf16>, tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[NEW_WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16>

    // CHECK:           [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[TANH:%.+]] = IE.Tanh([[CONV]]) : tensor<1x16x144x144xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[NEW_CONV:%.+]] = IE.Convolution([[CONV]], [[NEW_WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x144x144xf16>

    // CHECK:           return [[TANH]], [[NEW_CONV]] : tensor<1x16x144x144xf16>, tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @NotConvertForConcatWithLargeChannelNum
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x256x36x36xf16>
func.func @NotConvertForConcatWithLargeChannelNum(%arg0: tensor<1x256x36x36xf16>) -> (tensor<1x512x36x36xf16>) {
    %cst_0 = const.Declare tensor<256x256x1x1xf16> = dense<1.0> : tensor<256x256x1x1xf16>
    %cst_1 = const.Declare tensor<256x1x3x3xf16> = dense<1.0> : tensor<256x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x256x36x36xf16>, tensor<256x256x1x1xf16> -> tensor<1x256x36x36xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 256 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x256x36x36xf16>, tensor<256x1x3x3xf16> -> tensor<1x256x36x36xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x36x36xf16>, tensor<1x256x36x36xf16> -> tensor<1x512x36x36xf16>

    return %2 : tensor<1x512x36x36xf16>

    // CHECK-DAG:       [[WEIGHTS_0:%.+]] = const.Declare tensor<256x256x1x1xf16> = dense<1.000000e+00> : tensor<256x256x1x1xf16>
    // CHECK-DAG:       [[WEIGHTS_1:%.+]] = const.Declare tensor<256x1x3x3xf16> = dense<1.000000e+00> : tensor<256x1x3x3xf16>

    // CHECK:           [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x36x36xf16>, tensor<256x256x1x1xf16> -> tensor<1x256x36x36xf16>
    // CHECK:           [[GROUP_CONV:%.+]] = IE.GroupConvolution([[CONV]], [[WEIGHTS_1]]) {dilations = [1, 1], groups = 256 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x256x36x36xf16>, tensor<256x1x3x3xf16> -> tensor<1x256x36x36xf16>

    // CHECK:           [[CONCAT:%.+]] = IE.Concat([[CONV]], [[GROUP_CONV]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x36x36xf16>, tensor<1x256x36x36xf16> -> tensor<1x512x36x36xf16>

    // CHECK:           return [[CONCAT]] : tensor<1x512x36x36xf16>
}

// -----

// CHECK-LABEL: @OptimizeGroupConvConcatWithBias
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x144x144xf16>
func.func @OptimizeGroupConvConcatWithBias(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x2x144x144xf16>) {
    %cst_0 = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<1x16x1x1xf16> -> tensor<1x1x144x144xf16>
    %1 = IE.GroupConvolution(%0, %cst_1, %cst_2) {
        dilations = [1, 1],
        groups = 1 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x1x144x144xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x144x144xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x144x144xf16>, tensor<1x1x144x144xf16> -> tensor<1x2x144x144xf16>

    return %2 : tensor<1x2x144x144xf16>

    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.PadWithZero<[0, 1, 0, 0], [0, 0, 0, 0]>]
    // CHECK-DAG:       [[NEW_WEIGHTS:%.+]] = const.Declare tensor<2x1x1x1xf16>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x16x1x1xf16>

    // CHECK:           [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<1x16x1x1xf16> -> tensor<1x1x144x144xf16>
    // CHECK:           [[NEW_CONV:%.+]] = IE.Convolution([[CONV]], [[NEW_WEIGHTS]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x144x144xf16>, tensor<2x1x1x1xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x144x144xf16>

    // CHECK:           return [[NEW_CONV]] : tensor<1x2x144x144xf16>
}

// -----

// CHECK-LABEL: @OptimizeConvConcatWithBias
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x512x512xf16>
func.func @OptimizeConvConcatWithBias(%arg0: tensor<1x16x512x512xf16>) -> (tensor<1x4x512x512xf16>) {
    %cst = const.Declare tensor<3x16x1x1xf16> = dense<1.0> : tensor<3x16x1x1xf16>
    %cst_0 = const.Declare tensor<1x3x1x1xf16> = dense<1.0> : tensor<1x3x1x1xf16>
    %cst_1 = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x512x512xf16>, tensor<3x16x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x512x512xf16>
    %1 = IE.Convolution(%arg0, %cst_1, %cst_2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x512x512xf16>, tensor<1x16x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x512x512xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x512x512xf16>, tensor<1x1x512x512xf16> -> tensor<1x4x512x512xf16>

    return %2 : tensor<1x4x512x512xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<4x16x1x1xf16> = dense<1.000000e+00> : tensor<4x16x1x1xf16>
    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x4x1x1xf16> = dense<1.000000e+00> : tensor<1x4x1x1xf16>

    // CHECK:           [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x512x512xf16>, tensor<4x16x1x1xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x512x512xf16>
    // CHECK:           return [[CONV]] : tensor<1x4x512x512xf16>
}

// -----

// CHECK-LABEL: @OptimizeConvConcatWithoutBias
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x512x512xf16>
func.func @OptimizeConvConcatWithoutBias(%arg0: tensor<1x16x512x512xf16>) -> (tensor<1x4x512x512xf16>) {
    %cst = const.Declare tensor<3x16x1x1xf16> = dense<1.0> : tensor<3x16x1x1xf16>
    %cst_0 = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x512x512xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x512x512xf16>
    %1 = IE.Convolution(%arg0, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x512x512xf16>, tensor<1x16x1x1xf16> -> tensor<1x1x512x512xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x512x512xf16>, tensor<1x1x512x512xf16> -> tensor<1x4x512x512xf16>

    return %2 : tensor<1x4x512x512xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<4x16x1x1xf16> = dense<1.000000e+00> : tensor<4x16x1x1xf16>

    // CHECK:           [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x512x512xf16>, tensor<4x16x1x1xf16> -> tensor<1x4x512x512xf16>
    // CHECK:           return [[CONV]] : tensor<1x4x512x512xf16>
}

// -----

// CHECK-LABEL: @OptimizeConvConcatOneHasBias
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x512x512xf16>
func.func @OptimizeConvConcatOneHasBias(%arg0: tensor<1x16x512x512xf16>) -> (tensor<1x4x512x512xf16>) {
    %cst_0 = const.Declare tensor<3x16x1x1xf16> = dense<1.0> : tensor<3x16x1x1xf16>
    %cst_1 = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x512x512xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x512x512xf16>
    %1 = IE.Convolution(%arg0, %cst_1, %cst_2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x512x512xf16>, tensor<1x16x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x512x512xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x512x512xf16>, tensor<1x1x512x512xf16> -> tensor<1x4x512x512xf16>

    return %2 : tensor<1x4x512x512xf16>

    // CHECK-DAG:           [[WEIGHTS:%.+]] = const.Declare tensor<4x16x1x1xf16> = dense<1.000000e+00> : tensor<4x16x1x1xf16>
    // CHECK-DAG:           [[BIAS:%.+]] = const.Declare tensor<1x4x1x1xf16> =
    // CHECK-SAME{LITERAL}:     dense<[[[[0.000000e+00]], [[0.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]]]> : tensor<1x4x1x1xf16>

    // CHECK:               [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x512x512xf16>, tensor<4x16x1x1xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x512x512xf16>
    // CHECK:               return [[CONV]] : tensor<1x4x512x512xf16>
}

// -----

// CHECK-LABEL: @NotOptimizeConvConcat
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x20x20xf16>
func.func @NotOptimizeConvConcat(%arg0: tensor<1x4x20x20xf16>) -> (tensor<1x10x18x18xf16>) {
    %cst_0 = const.Declare tensor<5x2x3x3xf16> = dense<1.0> : tensor<5x2x3x3xf16>
    %cst_1 = const.Declare tensor<5x2x3x3xf16> = dense<2.0> : tensor<5x2x3x3xf16>

    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x4x20x20xf16> -> tensor<1x2x20x20xf16>, tensor<1x2x20x20xf16>
    %1 = IE.Convolution(%0#0, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x2x20x20xf16>, tensor<5x2x3x3xf16> -> tensor<1x5x18x18xf16>
    %2 = IE.Convolution(%0#1, %cst_1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x2x20x20xf16>, tensor<5x2x3x3xf16> -> tensor<1x5x18x18xf16>
    %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 5, 0, 0]]} : tensor<1x5x18x18xf16>, tensor<1x5x18x18xf16> -> tensor<1x10x18x18xf16>

    return %3 : tensor<1x10x18x18xf16>

    // CHECK-DAG:           [[WEIGHTS_0:%.+]] = const.Declare tensor<5x2x3x3xf16> = dense<1.000000e+00> : tensor<5x2x3x3xf16>
    // CHECK-DAG:           [[WEIGHTS_1:%.+]] = const.Declare tensor<5x2x3x3xf16> = dense<2.000000e+00> : tensor<5x2x3x3xf16>

    // CHECK:               [[SPLIT:%.+]]:2 = IE.Split([[INPUT]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x4x20x20xf16> -> tensor<1x2x20x20xf16>, tensor<1x2x20x20xf16>
    // CHECK:               [[CONV_0:%.+]] = IE.Convolution([[SPLIT]]#0, [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x2x20x20xf16>, tensor<5x2x3x3xf16> -> tensor<1x5x18x18xf16>
    // CHECK:               [[CONV_1:%.+]] = IE.Convolution([[SPLIT]]#1, [[WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x2x20x20xf16>, tensor<5x2x3x3xf16> -> tensor<1x5x18x18xf16>
    // CHECK:               [[CONCAT:%.+]] = IE.Concat([[CONV_0]], [[CONV_1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 5, 0, 0]]} : tensor<1x5x18x18xf16>, tensor<1x5x18x18xf16> -> tensor<1x10x18x18xf16>

    // CHECK:               return [[CONCAT]] : tensor<1x10x18x18xf16>
}

// -----

// CHECK-LABEL: @OptimizeSliceMultiplyConcat
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x24x1x64xf16>
func.func @OptimizeSliceMultiplyConcat(%arg0: tensor<1x24x1x64xf16>) -> (tensor<1x24x1x64xf16>) {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    %0 = IE.Slice %arg0 [0, 0, 0, 32] [1, 24, 1, 32] : tensor<1x24x1x64xf16> to tensor<1x24x1x32xf16>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x24x1x32xf16>, tensor<1x1x1x1xf16> -> tensor<1x24x1x32xf16>
    %2 = IE.Slice %arg0 [0, 0, 0, 0] [1, 24, 1, 32] : tensor<1x24x1x64xf16> to tensor<1x24x1x32xf16>
    %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32]]} : tensor<1x24x1x32xf16>, tensor<1x24x1x32xf16> -> tensor<1x24x1x64xf16>

    return %3 : tensor<1x24x1x64xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<64x64x1x1xf16>

    // CHECK:           [[PERMUTE_CAST_IN:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x24x1x64xf16> -> tensor<1x64x24x1xf16, {order = #NHWC}>
    // CHECK:           [[CONV:%.+]] = IE.Convolution([[PERMUTE_CAST_IN]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x24x1xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x24x1xf16, {order = #NHWC}>
    // CHECK:           [[PERMUTE_CAST_OUT:%.+]] = IE.PermuteCast([[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x24x1xf16, {order = #NHWC}> -> tensor<1x24x1x64xf16>

    // CHECK:           return [[PERMUTE_CAST_OUT]] : tensor<1x24x1x64xf16>
}

// -----

// CHECK-LABEL: @OptimizeOverlappedSliceMultiplyConcat
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x24x1x64xf16>
func.func @OptimizeOverlappedSliceMultiplyConcat(%arg0: tensor<1x24x1x64xf16>) -> (tensor<1x24x1x80xf16>) {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    %0 = IE.Slice %arg0 [0, 0, 0, 16] [1, 24, 1, 48] : tensor<1x24x1x64xf16> to tensor<1x24x1x48xf16>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x24x1x48xf16>, tensor<1x1x1x1xf16> -> tensor<1x24x1x48xf16>
    %2 = IE.Slice %arg0 [0, 0, 0, 0] [1, 24, 1, 32] : tensor<1x24x1x64xf16> to tensor<1x24x1x32xf16>
    %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 48]]} : tensor<1x24x1x48xf16>, tensor<1x24x1x32xf16> -> tensor<1x24x1x80xf16>

    return %3 : tensor<1x24x1x80xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<80x64x1x1xf16>

    // CHECK:           [[PERMUTE_CAST_IN:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x24x1x64xf16> -> tensor<1x64x24x1xf16, {order = #NHWC}>
    // CHECK:           [[CONV:%.+]] = IE.Convolution([[PERMUTE_CAST_IN]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x24x1xf16, {order = #NHWC}>, tensor<80x64x1x1xf16> -> tensor<1x80x24x1xf16, {order = #NHWC}>
    // CHECK:           [[PERMUTE_CAST_OUT:%.+]] = IE.PermuteCast([[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x80x24x1xf16, {order = #NHWC}> -> tensor<1x24x1x80xf16>

    // CHECK:           return [[PERMUTE_CAST_OUT]] : tensor<1x24x1x80xf16>
}

// -----

// CHECK-LABEL: @NotOptimizeParallelSliceOps
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x24x1x64xf16>
func.func @NotOptimizeParallelSliceOps(%arg0: tensor<1x24x1x64xf16>) -> (tensor<1x24x1x80xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 16] [1, 24, 1, 48] : tensor<1x24x1x64xf16> to tensor<1x24x1x48xf16>
    %1 = IE.Slice %arg0 [0, 0, 0, 0] [1, 24, 1, 32] : tensor<1x24x1x64xf16> to tensor<1x24x1x32xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 48]]} : tensor<1x24x1x48xf16>, tensor<1x24x1x32xf16> -> tensor<1x24x1x80xf16>

    return %2 : tensor<1x24x1x80xf16>

    // CHECK:           [[SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 16] [1, 24, 1, 48] : tensor<1x24x1x64xf16> to tensor<1x24x1x48xf16>
    // CHECK:           [[SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 24, 1, 32] : tensor<1x24x1x64xf16> to tensor<1x24x1x32xf16>
    // CHECK:           [[CONCAT:%.+]] = IE.Concat([[SLICE_0]], [[SLICE_1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 48]]} : tensor<1x24x1x48xf16>, tensor<1x24x1x32xf16> -> tensor<1x24x1x80xf16>

    // CHECK:           return [[CONCAT]] : tensor<1x24x1x80xf16>
}

// -----

// CHECK-LABEL: @NotOptimizeSliceMultiplyConcatForNonSplatScale
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x24x1x64xf16>
func.func @NotOptimizeSliceMultiplyConcatForNonSplatScale(%arg0: tensor<1x24x1x64xf16>) -> (tensor<1x24x1x64xf16>) {
    %cst = const.Declare tensor<1x1x1x32xf16> = dense<-1.000000e+00> : tensor<1x1x1x32xf32>, [#const.CastElemType<f16>]

    %0 = IE.Slice %arg0 [0, 0, 0, 32] [1, 24, 1, 32] : tensor<1x24x1x64xf16> to tensor<1x24x1x32xf16>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x24x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x24x1x32xf16>
    %2 = IE.Slice %arg0 [0, 0, 0, 0] [1, 24, 1, 32] : tensor<1x24x1x64xf16> to tensor<1x24x1x32xf16>
    %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32]]} : tensor<1x24x1x32xf16>, tensor<1x24x1x32xf16> -> tensor<1x24x1x64xf16>

    return %3 : tensor<1x24x1x64xf16>

    // CHECK-DAG:       [[SCALE:%.+]] = const.Declare tensor<1x1x1x32xf16>

    // CHECK:           [[SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 32] [1, 24, 1, 32] : tensor<1x24x1x64xf16> to tensor<1x24x1x32xf16>
    // CHECK:           [[MULTIPLY:%.+]] = IE.Multiply([[SLICE_0]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x24x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x24x1x32xf16>
    // CHECK:           [[SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 24, 1, 32] : tensor<1x24x1x64xf16> to tensor<1x24x1x32xf16>
    // CHECK:           [[CONCAT:%.+]] = IE.Concat([[MULTIPLY]], [[SLICE_1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32]]} : tensor<1x24x1x32xf16>, tensor<1x24x1x32xf16> -> tensor<1x24x1x64xf16>

    // CHECK:           return [[CONCAT]] : tensor<1x24x1x64xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 3.1906749834032623E-5>
!qElemType1 = !quant.uniform<u8:f16:0, {0.04918184093400544:128,0.072342521069096583:128,0.056963582132376879:128}>
!qElemType2 = !quant.uniform<u8:f16:0, {0.064222440532609532:128,0.063976377599379602:128,0.057332676532221773:128}>
!qElemType3 = !quant.uniform<i8:f16:1, {0.04918184093400544,0.072342521069096583,0.056963582132376879}>
!qElemType4 = !quant.uniform<i8:f16:0, {0.04918184093400544,0.072342521069096583,0.056963582132376879}>
!qElemType5 = !quant.uniform<i8:f16:1, {0.064222440532609532,0.063976377599379602,0.057332676532221773}>
!qElemType6 = !quant.uniform<i8:f16:0, {0.064222440532609532,0.063976377599379602,0.057332676532221773}>

// CHECK-LABEL: func.func @PropagateOutputTypeOfInputTensorIntoOutputConvType
// CHECK-SAME:        [[INPUT:%.+]]:  tensor<1x512x1x1x!qElemType>
func.func @PropagateOutputTypeOfInputTensorIntoOutputConvType(%arg0: tensor<1x512x1x1x!qElemType>) -> tensor<1x6x1x1xf16> {
  %cst = const.Declare tensor<3x512x1x1x!qElemType1> = dense<1> : tensor<3x512xsi8>, [#const.Reshape<[1, 3, 1, 512]>, #const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType3>, #const.ChangeShapeAndElemType<[3, 512, 1, 1], !qElemType4>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.280000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]
  %cst_0 = const.Declare tensor<3x512x1x1x!qElemType2> = dense<1> : tensor<3x512xsi8>, [#const.Reshape<[1, 3, 1, 512]>, #const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType5>, #const.ChangeShapeAndElemType<[3, 512, 1, 1], !qElemType6>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.280000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>]
  %cst_1 = const.Declare tensor<1x3x1x1xf16> = dense<1.0> : tensor<1x3xf32>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]
  %cst_2 = const.Declare tensor<1x3x1x1xf16> = dense<1.0> : tensor<1x3xf32>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>]
  %0 = IE.Convolution(%arg0, %cst_0, %cst_2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1x!qElemType>, tensor<3x512x1x1x!qElemType2>, tensor<1x3x1x1xf16> -> tensor<1x3x1x1xf16>
  %1 = IE.Convolution(%arg0, %cst, %cst_1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1x!qElemType>, tensor<3x512x1x1x!qElemType1>, tensor<1x3x1x1xf16> -> tensor<1x3x1x1xf16>
  %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x6x1x1xf16>
  return %2 : tensor<1x6x1x1xf16>

  // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<6x512x1x1x!qElemType1> = dense<129> : tensor<6x512x1x1xui8>, [#const.CastElemType<!qElemType1>]
  // CHECK-DAG:       [[CST_0:%.+]] =  const.Declare tensor<1x6x1x1xf16> = dense<1.000000e+00> : tensor<1x6x1x1xf16>

  // CHECK:    [[CONVOLUTION:%.+]] = IE.Convolution([[INPUT]], [[CST]], [[CST_0]]) {
  // CHECK-SAME: dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
  // CHECK-SAME: } : tensor<1x512x1x1x!qElemType>, tensor<6x512x1x1x!qElemType1>, tensor<1x6x1x1xf16> -> tensor<1x6x1x1xf16>
  // CHECK:    return [[CONVOLUTION]] : tensor<1x6x1x1xf16>
}
