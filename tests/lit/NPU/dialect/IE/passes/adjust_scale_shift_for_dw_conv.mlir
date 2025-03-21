//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-scale-shift-for-dw-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @AdjustScaleShiftForDWConvWithInputIsConst
func.func @AdjustScaleShiftForDWConvWithInputIsConst(%arg0: tensor<1x77x1x1xf16>) -> tensor<77x77x3x3xf16> {
    %input_const = const.Declare tensor<77x77x3x3xf16> = dense<1.000000e+00> : tensor<77x77x3x3xf16>
    %result = IE.ScaleShift(%input_const, %arg0) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<77x77x3x3xf16>, tensor<1x77x1x1xf16> -> tensor<77x77x3x3xf16>
    return %result : tensor<77x77x3x3xf16>

    // CHECK-DAG:   [[INPUT_CONST:%.*]] = const.Declare tensor<1x5929x3x3xf16> = dense<1.000000e+00> :
    // CHECK-SAME:            tensor<77x77x3x3xf16>, [#const.Reshape<[1, 5929, 3, 3]>]
    // CHECK-DAG:   [[BROADCAST_SHAPE:%.*]] = const.Declare tensor<4xsi32> = dense<[77, 77, 1, 1]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[BROADCAST:%.*]] = IE.Broadcast(%arg0, [[BROADCAST_SHAPE]]) {mode = #IE.broadcast_type<NUMPY>} :
    // CHECK-SAME:            tensor<1x77x1x1xf16>, tensor<4xsi32> -> tensor<77x77x1x1xf16>
    // CHECK:       [[INPUT_RESHAPE:%.*]] = IE.Reshape([[BROADCAST]]) {shape_value = [1, 5929, 1, 1]} :
    // CHECK-SAME:            tensor<77x77x1x1xf16> -> tensor<1x5929x1x1xf16>
    // CHECK:       [[SCALESHIFT:%.*]] = IE.ScaleShift([[INPUT_CONST]], [[INPUT_RESHAPE]]) {operandSegmentSizes = array<i32: 1, 1, 0>} :
    // CHECK-SAME:            tensor<1x5929x3x3xf16>, tensor<1x5929x1x1xf16> -> tensor<1x5929x3x3xf16>
    // CHECK:       [[RESULT:%.*]] = IE.Reshape([[SCALESHIFT]]) {shape_value = [77, 77, 3, 3]} : tensor<1x5929x3x3xf16> -> tensor<77x77x3x3xf16>
    // CHECK:       return [[RESULT]] : tensor<77x77x3x3xf16>
}

// -----

// CHECK-LABEL: @AdjustScaleShiftForDWConvWithLargeN
func.func @AdjustScaleShiftForDWConvWithLargeN(%arg0: tensor<1x77x77x9xf16>) -> tensor<77x77x3x3xf16> {
    %cst = const.Declare tensor<1x77x1x1xf16> = dense<-1.39928699> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 77 : i64>]
    %reshape = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [77, 77, 3, 3]} : tensor<1x77x77x9xf16> -> tensor<77x77x3x3xf16>
    %result = IE.ScaleShift(%reshape, %cst) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<77x77x3x3xf16>, tensor<1x77x1x1xf16> -> tensor<77x77x3x3xf16>

    return %result : tensor<77x77x3x3xf16>

    // CHECK-DAG:   [[CST1:%.*]] = const.Declare tensor<1x5929x1x1xf16> = dense<-1.39928699> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 77 : i64>, #const.Broadcast<0 : i64, 77 : i64>, #const.Reshape<[1, 5929, 1, 1]>]
    // CHECK:       [[AFFINERESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [77, 77, 3, 3]} : tensor<1x77x77x9xf16> -> tensor<77x77x3x3xf16>
    // CHECK:       [[RESHAPE:%.*]] = IE.Reshape([[AFFINERESHAPE]]) {shape_value = [1, 5929, 3, 3]} : tensor<77x77x3x3xf16> -> tensor<1x5929x3x3xf16>
    // CHECK:       [[SCALESHIFT:%.*]] = IE.ScaleShift([[RESHAPE]], [[CST1]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x5929x3x3xf16>, tensor<1x5929x1x1xf16> -> tensor<1x5929x3x3xf16>
    // CHECK:       [[RESULT:%.*]] = IE.Reshape([[SCALESHIFT]]) {shape_value = [77, 77, 3, 3]} : tensor<1x5929x3x3xf16> -> tensor<77x77x3x3xf16>
    // CHECK:       return [[RESULT]] : tensor<77x77x3x3xf16>
}

// -----

// CHECK-LABEL: @NotAdjustScaleShiftForDWConvWith1N
func.func @NotAdjustScaleShiftForDWConvWith1N(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
    %weights = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>
    %bias = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    %0 = IE.ScaleShift(%arg0, %weights, %bias) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x3x224x224xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>

    return %0 : tensor<1x3x224x224xf16>

    // CHECK-NOT:   IE.Reshape
    // CHECK-NOT:   IE.Broadcast
}

// -----

// CHECK-LABEL: @AdjustScaleShiftForDWConvWithConstantSplatWeight
func.func @AdjustScaleShiftForDWConvWithConstantSplatWeight(%arg0: tensor<8x2x64x1xf16>) -> tensor<8x2x64x1xf16> {
    %weights = const.Declare tensor<1x2x1x1xf16> = dense<-1.500000e+00> : tensor<1x2x1x1xf16>
    %0 = IE.ScaleShift(%arg0, %weights) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<8x2x64x1xf16>, tensor<1x2x1x1xf16> -> tensor<8x2x64x1xf16>

    return %0 : tensor<8x2x64x1xf16>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<-1.500000e+00> : tensor<1x2x1x1xf16>, [#const.Broadcast<0 : i64, 8 : i64>, #const.Reshape<[1, 16, 1, 1]>]
    // CHECK:       [[RESHAPE:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 64, 1]} : tensor<8x2x64x1xf16> -> tensor<1x16x64x1xf16>
    // CHECK:       [[SCALESHIFT:%.*]] = IE.ScaleShift([[RESHAPE]], [[WEIGHTS]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x16x64x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x64x1xf16>
    // CHECK:       [[RESULT:%.*]] = IE.Reshape([[SCALESHIFT]]) {shape_value = [8, 2, 64, 1]} : tensor<1x16x64x1xf16> -> tensor<8x2x64x1xf16>
    // CHECK:       return [[RESULT]] : tensor<8x2x64x1xf16>
}

// -----

// CHECK-LABEL: @NotAdjustScaleShiftForDWConvWithConstantNotSplatWeight
func.func @NotAdjustScaleShiftForDWConvWithConstantNotSplatWeight(%arg0: tensor<8x2x64x1xf16>) -> tensor<8x2x64x1xf16> {
    %weights = const.Declare tensor<1x2x1x1xf16> = dense<[[[[-1.500000e+00]],[[-1.600000e+00]]]]> : tensor<1x2x1x1xf16>
    %0 = IE.ScaleShift(%arg0, %weights) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<8x2x64x1xf16>, tensor<1x2x1x1xf16> -> tensor<8x2x64x1xf16>

    return %0 : tensor<8x2x64x1xf16>

    // CHECK:   const.Declare
    // CHECK:   IE.ScaleShift
}

// -----

// CHECK-LABEL: @NotAdjustScaleShiftForDWConvWithTensorWeightsSmallBatch
func.func @NotAdjustScaleShiftForDWConvWithTensorWeightsSmallBatch(%arg0: tensor<8x2x64x1xf16>, %arg1: tensor<1x2x1x1xf16>) -> tensor<8x2x64x1xf16> {
    %0 = IE.ScaleShift(%arg0, %arg1) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<8x2x64x1xf16>, tensor<1x2x1x1xf16> -> tensor<8x2x64x1xf16>

    return %0 : tensor<8x2x64x1xf16>

    // CHECK:   IE.ScaleShift
}

// -----

// CHECK-LABEL: @AdjustScaleShiftForDWConvWithConstantSplatBiases
func.func @AdjustScaleShiftForDWConvWithConstantSplatBiases(%arg0: tensor<8x2x64x1xf16>) -> tensor<8x2x64x1xf16> {
    %biases = const.Declare tensor<1x2x1x1xf16> = dense<-1.500000e+00> : tensor<1x2x1x1xf16>
    %0 = IE.ScaleShift(%arg0, %biases) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<8x2x64x1xf16>, tensor<1x2x1x1xf16> -> tensor<8x2x64x1xf16>

    return %0 : tensor<8x2x64x1xf16>

    // CHECK-DAG:   [[BIASES:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<-1.500000e+00> : tensor<1x2x1x1xf16>, [#const.Broadcast<0 : i64, 8 : i64>, #const.Reshape<[1, 16, 1, 1]>]
    // CHECK:       [[RESHAPE:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 64, 1]} : tensor<8x2x64x1xf16> -> tensor<1x16x64x1xf16>
    // CHECK:       [[SCALESHIFT:%.*]] = IE.ScaleShift([[RESHAPE]], [[BIASES]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x16x64x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x64x1xf16>
    // CHECK:       [[RESULT:%.*]] = IE.Reshape([[SCALESHIFT]]) {shape_value = [8, 2, 64, 1]} : tensor<1x16x64x1xf16> -> tensor<8x2x64x1xf16>
    // CHECK:       return [[RESULT]] : tensor<8x2x64x1xf16>
}

// -----

// CHECK-LABEL: @NotAdjustScaleShiftForDWConvWithConstantNotSplatBiases
func.func @NotAdjustScaleShiftForDWConvWithConstantNotSplatBiases(%arg0: tensor<8x2x64x1xf16>) -> tensor<8x2x64x1xf16> {
    %biases = const.Declare tensor<1x2x1x1xf16> = dense<[[[[-1.500000e+00]],[[-1.600000e+00]]]]> : tensor<1x2x1x1xf16>
    %0 = IE.ScaleShift(%arg0, %biases) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<8x2x64x1xf16>, tensor<1x2x1x1xf16> -> tensor<8x2x64x1xf16>

    return %0 : tensor<8x2x64x1xf16>

    // CHECK:   const.Declare
    // CHECK:   IE.ScaleShift
}
