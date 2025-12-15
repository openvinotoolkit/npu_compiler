//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --scaleshift-processing %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ScaleShiftProcessingWithConstInput
func.func @ScaleShiftProcessingWithConstInput(%arg0: tensor<1x77x1x1xf16>) -> tensor<77x77x3x3xf16> {
    %input_const = const.Declare tensor<77x77x3x3xf16> = dense<1.000000e+00> : tensor<77x77x3x3xf16>
    %result = IE.ScaleShift(%input_const, %arg0) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<77x77x3x3xf16>, tensor<1x77x1x1xf16> -> tensor<77x77x3x3xf16>
    return %result : tensor<77x77x3x3xf16>

    // CHECK-DAG:   [[INPUT_CONST:%.*]] = const.Declare tensor<1x5929x3x3xf16> = dense<1.000000e+00> : tensor<77x77x3x3xf16>, [#const.Reshape<[1, 5929, 3, 3]>]
    // CHECK:       [[TILE:%.*]] = IE.Tile(%arg0) {repeats_values = [77, 1, 1, 1]} : tensor<1x77x1x1xf16> -> tensor<77x77x1x1xf16>
    // CHECK:       [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[TILE]]) {
    // CHECK-SAME{LITERAL}:          dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [5929, 1, 1, 1]} :
    // CHECK-SAME:           tensor<77x77x1x1xf16> -> tensor<5929x1x1x1xf16>
    // CHECK:       [[CONV:%.*]] = IE.GroupConvolution([[INPUT_CONST]], [[AFFINERESHAPE]])
    // CHECK-SAME:         dilations = [1, 1], groups = 5929 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:         tensor<1x5929x3x3xf16>, tensor<5929x1x1x1xf16> -> tensor<1x5929x3x3xf16>
    // CHECK:       [[RESULT:%.*]] = IE.Reshape([[CONV]]) {shape_value = [77, 77, 3, 3]} : tensor<1x5929x3x3xf16> -> tensor<77x77x3x3xf16>
    // CHECK:       return [[RESULT]] : tensor<77x77x3x3xf16>
}

// -----

// CHECK-LABEL: @ScaleShiftProcessingWith1NInput
func.func @ScaleShiftProcessingWith1NInput(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
    %weights = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>
    %bias = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    %0 = IE.ScaleShift(%arg0, %weights, %bias) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x3x224x224xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>

    return %0 : tensor<1x3x224x224xf16>

    // CHECK-DAG:   [[CONST1:%.*]] = const.Declare tensor<3x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>, [#const.Reshape<[3, 1, 1, 1]>]
    // CHECK-DAG:   [[CONST2:%.*]] = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    // CHECK:       [[RESULT:%.*]] = IE.GroupConvolution(%arg0, [[CONST1]], [[CONST2]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x3x224x224xf16>, tensor<3x1x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>
    // CHECK:       return [[RESULT]] : tensor<1x3x224x224xf16>
}
