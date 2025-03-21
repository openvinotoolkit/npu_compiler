//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//


// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-conv-with-slice %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @FuseConvWithSliceSingleUser
func.func @FuseConvWithSliceSingleUser(%arg0: tensor<1x16x80x80xf16>) -> tensor<1x32x80x80xf16> {
    %weights = const.Declare tensor<64x16x3x3xf16> = dense<1.0> : tensor<64x16x3x3xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %0 = IE.Convolution(%arg0, %weights, %bias) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>} : tensor<1x16x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 32, 80, 80] : tensor<1x64x80x80xf16> to tensor<1x32x80x80xf16>
    return %1 : tensor<1x32x80x80xf16>

    // CHECK-DAG:       [[WEIGHTS:%.*]] = const.Declare tensor<32x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [32, 16, 3, 3]>]
    // CHECK-DAG:       [[BIAS:%.*]] = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 32, 1, 1]>]
    // CHECK:           [[CONV0:%.*]] = IE.Convolution(%arg0, [[WEIGHTS]], [[BIAS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 1]}
    // CHECK-SAME:              : tensor<1x16x80x80xf16>, tensor<32x16x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x80x80xf16>
    // CHECK:           return      [[CONV0]]
}


// -----

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>
!qElemType2 = !quant.uniform<u8:f32, 3.000000e+00>
!qElemType1 = !quant.uniform<u8:f32, 2.000000e+00>
// CHECK-LABEL: @FuseConvWithSliceSingleUserQuantizeType
func.func @FuseConvWithSliceSingleUserQuantizeType(%arg0: tensor<1x16x80x80x!qElemType>) -> tensor<1x32x80x80x!qElemType1> {
    %weights = const.Declare tensor<64x16x3x3x!qElemType2> = dense<1.000000e+00> : tensor<64x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>]
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %0 = IE.Convolution(%arg0, %weights, %bias) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>} : tensor<1x16x80x80x!qElemType>, tensor<64x16x3x3x!qElemType2>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80x!qElemType1>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 32, 80, 80] : tensor<1x64x80x80x!qElemType1> to tensor<1x32x80x80x!qElemType1>
    return %1 : tensor<1x32x80x80x!qElemType1>

    // CHECK-DAG:       [[WEIGHTS:%.*]] = const.Declare tensor<32x16x3x3x!qElemType2> = dense<1.000000e+00> : tensor<64x16x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [32, 16, 3, 3]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>]
    // CHECK-DAG:       [[BIAS:%.*]] = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 32, 1, 1]>]
    // CHECK:           [[CONV0:%.*]] = IE.Convolution(%arg0, [[WEIGHTS]], [[BIAS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 1]}
    // CHECK-SAME:              : tensor<1x16x80x80x!qElemType>, tensor<32x16x3x3x!qElemType2>, tensor<1x32x1x1xf16> -> tensor<1x32x80x80x!qElemType1>
    // CHECK:           return      [[CONV0]]
}

// -----

// CHECK-LABEL: @FuseConvWithSliceTwoUsers
func.func @FuseConvWithSliceTwoUsers(%arg0: tensor<1x16x80x80xf16>) -> (tensor<1x16x80x80xf16>, tensor<1x32x80x80xf16>) {
    %weights = const.Declare tensor<64x16x3x3xf16> = dense<1.0> : tensor<64x16x3x3xf16>
    %0 = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x80x80xf16>, tensor<64x16x3x3xf16> -> tensor<1x64x80x80xf16>
    %1 = IE.Slice %0 [0, 32, 0, 0] [1, 16, 80, 80] : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    %2 = IE.Slice %0 [0, 0, 0, 0] [1, 32, 80, 80] : tensor<1x64x80x80xf16> to tensor<1x32x80x80xf16>
    return %1, %2 : tensor<1x16x80x80xf16>, tensor<1x32x80x80xf16>

    // CHECK-DAG:       [[WEIGHTS0:%.*]] = const.Declare tensor<32x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [32, 16, 3, 3]>]
    // CHECK-DAG:       [[WEIGHTS1:%.*]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>, [#const.SubView<[32, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK:           [[CONV0:%.*]] = IE.Convolution(%arg0, [[WEIGHTS0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:               : tensor<1x16x80x80xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x80x80xf16>
    // CHECK:           [[CONV1:%.*]] = IE.Convolution(%arg0, [[WEIGHTS1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:               : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16> -> tensor<1x16x80x80xf16>
    // CHECK:           return      [[CONV1]], [[CONV0]]
}


// -----

// CHECK-LABEL: @NotFuseWithNonConstWeights
func.func @NotFuseWithNonConstWeights(%arg0: tensor<1x16x80x80xf16>, %arg1: tensor<64x16x3x3xf16>) -> tensor<1x32x80x80xf16> {
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %0 = IE.Convolution(%arg0, %arg1, %bias) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>} : tensor<1x16x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 32, 80, 80] : tensor<1x64x80x80xf16> to tensor<1x32x80x80xf16>
    return %1 : tensor<1x32x80x80xf16>

    // CHECK-DAG:       [[BIAS:%.*]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>
    // CHECK:           [[CONV:%.*]] = IE.Convolution(%arg0, %arg1, [[BIAS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 1]} : tensor<1x16x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>
    // CHECK:           [[SLICE:%.*]] =  IE.Slice [[CONV]] [0, 0, 0, 0] [1, 32, 80, 80] : tensor<1x64x80x80xf16> to tensor<1x32x80x80xf16>
    // CHECK:           return      [[SLICE]]
}


// -----

// CHECK-LABEL: @NotFuseWithNonChannelSlice
func.func @NotFuseWithNonChannelSlice(%arg0: tensor<1x16x80x80xf16>) -> tensor<1x64x32x80xf16> {
    %weights = const.Declare tensor<64x16x3x3xf16> = dense<1.0> : tensor<64x16x3x3xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %0 = IE.Convolution(%arg0, %weights, %bias) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>} : tensor<1x16x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 64, 32, 80] : tensor<1x64x80x80xf16> to tensor<1x64x32x80xf16>
    return %1 : tensor<1x64x32x80xf16>

    // CHECK-DAG:       [[WEIGHTS:%.*]] = const.Declare tensor<64x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-DAG:       [[BIAS:%.*]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>
    // CHECK:           [[CONV:%.*]] = IE.Convolution(%arg0, [[WEIGHTS]], [[BIAS]])
    // CHECK:           [[SLICE:%.*]] = IE.Slice [[CONV]]
    // CHECK:           return      [[SLICE]]
}

// -----

// CHECK-LABEL: @NotFuseWithUnAlignedSlice
func.func @NotFuseWithUnAlignedSlice(%arg0: tensor<1x16x80x80xf16>) -> tensor<1x33x80x80xf16> {
    %weights = const.Declare tensor<64x16x3x3xf16> = dense<1.0> : tensor<64x16x3x3xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %0 = IE.Convolution(%arg0, %weights, %bias) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>} : tensor<1x16x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 33, 80, 80] : tensor<1x64x80x80xf16> to tensor<1x33x80x80xf16>
    return %1 : tensor<1x33x80x80xf16>

    // CHECK-DAG:       [[WEIGHTS:%.*]] = const.Declare tensor<64x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-DAG:       [[BIAS:%.*]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>
    // CHECK:           [[CONV:%.*]] = IE.Convolution(%arg0, [[WEIGHTS]], [[BIAS]])
    // CHECK:           [[SLICE:%.*]] = IE.Slice [[CONV]]
    // CHECK:           return      [[SLICE]]
}


// -----

// CHECK-LABEL: @NotFuseWithSliceOverlap
func.func @NotFuseWithSliceOverlap(%arg0: tensor<1x16x80x80xf16>) -> (tensor<1x16x80x80xf16>, tensor<1x32x80x80xf16>) {
    %weights = const.Declare tensor<64x16x3x3xf16> = dense<1.0> : tensor<64x16x3x3xf16>
    %0 = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x80x80xf16>, tensor<64x16x3x3xf16> -> tensor<1x64x80x80xf16>
    %1 = IE.Slice %0 [0, 31, 0, 0] [1, 16, 80, 80] : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    %2 = IE.Slice %0 [0, 0, 0, 0] [1, 32, 80, 80] : tensor<1x64x80x80xf16> to tensor<1x32x80x80xf16>
    return %1, %2 : tensor<1x16x80x80xf16>, tensor<1x32x80x80xf16>

    // CHECK-DAG:       [[WEIGHTS:%.*]] = const.Declare tensor<64x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK:           [[CONV0:%.*]] = IE.Convolution(%arg0, [[WEIGHTS]])
    // CHECK:           [[SLICE0:%.*]] = IE.Slice [[CONV0]] [0, 31, 0, 0] [1, 16, 80, 80] : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK:           [[SLICE1:%.*]] = IE.Slice [[CONV0]] [0, 0, 0, 0] [1, 32, 80, 80] : tensor<1x64x80x80xf16> to tensor<1x32x80x80xf16>

    // CHECK:           return      [[SLICE0]],  [[SLICE1]]
}


// -----

// CHECK-LABEL: @NotFuseWithNonSliceUser
func.func @NotFuseWithNonSliceUser(%arg0: tensor<1x16x80x80xf16>) -> (tensor<1x64x80x80xf16>, tensor<1x32x80x80xf16>) {
    %weights = const.Declare tensor<64x16x3x3xf16> = dense<1.0> : tensor<64x16x3x3xf16>
    %0 = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x80x80xf16>, tensor<64x16x3x3xf16> -> tensor<1x64x80x80xf16>
    %1 = IE.AvgPool(%0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x64x80x80xf16> -> tensor<1x64x80x80xf16>
    %2 = IE.Slice %0 [0, 0, 0, 0] [1, 32, 80, 80] : tensor<1x64x80x80xf16> to tensor<1x32x80x80xf16>
    return %1, %2 : tensor<1x64x80x80xf16>, tensor<1x32x80x80xf16>

    // CHECK-DAG:       [[WEIGHTS:%.*]] = const.Declare tensor<64x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK:           [[CONV:%.*]] = IE.Convolution(%arg0, [[WEIGHTS]])
    // CHECK:           [[AVGPOOL:%.*]] = IE.AvgPool([[CONV]])
    // CHECK:           [[SLICE:%.*]] = IE.Slice [[CONV]]
    // CHECK:           return      [[AVGPOOL]],  [[SLICE]]
}
