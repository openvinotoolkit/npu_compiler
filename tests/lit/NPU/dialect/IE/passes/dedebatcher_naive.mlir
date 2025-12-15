//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --de-debatcher="debatching-inlining-method=naive" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

func.func private @SingleInputSingleOutputNonBatched_Batch1(%arg0: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
    %1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
    %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    return %2 : tensor<1x48x60x60xf32>
}

// CHECK-LABEL: @SingleInputSingleOutputNonBatched
func.func @SingleInputSingleOutputNonBatched(%arg0: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
    %0 = call @SingleInputSingleOutputNonBatched_Batch1(%arg0) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
    return %0 : tensor<1x48x60x60xf32>

    // CHECK: func.func @SingleInputSingleOutputNonBatched([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {

    // CHECK: [[FUNC_0:%.*]] = call @SingleInputSingleOutputNonBatched_Batch1([[ARG0]]) :
    // CHECK-SAME:  (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>

    // CHECK: return [[FUNC_0]] : tensor<1x48x60x60xf32>
}

// -----


func.func private @MultipleInputSingleOutputDeBatched_Batch1(%arg0: tensor<1x3x62x62xf32>, %arg1: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
    %1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
    %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    %3 = IE.Add(%2, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    %4 = IE.SoftMax(%3) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    return %4 : tensor<1x48x60x60xf32>
}

// CHECK-LABEL: @MultipleInputSingleOutputDeBatched
// Unlike similar `dedebatcher_attributed.mlir`, which is supposed to be executed
// using a different option '--de-debatcher="debatching-inlining-method="', this one
// doesn't inject an attribute "debatched = []" to debatched functions call, thus
// this is all the difference between these LIT tests.
// The absence of the attribute is the only condition we test here
func.func @MultipleInputSingleOutputDeBatched(%arg0: tensor<3x3x62x62xf32>, %arg1: tensor<3x48x60x60xf32>) -> tensor<3x48x60x60xf32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
    %1 = builtin.unrealized_conversion_cast %arg1 : tensor<3x48x60x60xf32> to tensor<1x48x60x60xf32>
    %2 = call @MultipleInputSingleOutputDeBatched_Batch1(%0, %1) : (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    %3 = builtin.unrealized_conversion_cast %2: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
    return %3 : tensor<3x48x60x60xf32>

    // CHECK: func.func @MultipleInputSingleOutputDeBatched([[ARG0:%.+]]: tensor<3x3x62x62xf32>, [[ARG1:%.+]]: tensor<3x48x60x60xf32>) -> tensor<3x48x60x60xf32> {

    // CHECK: [[SLICE_00:%.*]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 3, 62, 62] :
    // CHECK-SAME:  tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
    // CHECK: [[SLICE_01:%.*]] = IE.Slice [[ARG1]] [0, 0, 0, 0] [1, 48, 60, 60] :
    // CHECK-SAME:  tensor<3x48x60x60xf32> to tensor<1x48x60x60xf32>

    // CHECK: [[FUNC_0:%.*]] = call @MultipleInputSingleOutputDeBatched_Batch1([[SLICE_00]], [[SLICE_01]]) {debatched = [0, 3]} :
    // CHECK-SAME:  (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>

    // CHECK: [[SLICE_10:%.*]] = IE.Slice [[ARG0]] [1, 0, 0, 0] [1, 3, 62, 62] :
    // CHECK-SAME:  tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
    // CHECK: [[SLICE_11:%.*]] = IE.Slice [[ARG1]] [1, 0, 0, 0] [1, 48, 60, 60] :
    // CHECK-SAME:  tensor<3x48x60x60xf32> to tensor<1x48x60x60xf32>

    // CHECK: [[FUNC_1:%.*]] = call @MultipleInputSingleOutputDeBatched_Batch1([[SLICE_10]], [[SLICE_11]]) {debatched = [1, 3]} :
    // CHECK-SAME:  (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>

    // CHECK: [[SLICE_20:%.*]] = IE.Slice [[ARG0]] [2, 0, 0, 0] [1, 3, 62, 62] :
    // CHECK-SAME:  tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
    // CHECK: [[SLICE_21:%.*]] = IE.Slice [[ARG1]] [2, 0, 0, 0] [1, 48, 60, 60] :
    // CHECK-SAME:  tensor<3x48x60x60xf32> to tensor<1x48x60x60xf32>

    // CHECK: [[FUNC_2:%.*]] = call @MultipleInputSingleOutputDeBatched_Batch1([[SLICE_20]], [[SLICE_21]]) {debatched = [2, 3]} :
    // CHECK-SAME:  (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>

    // CHECK: [[CONCAT:%.*]] = IE.Concat([[FUNC_0]], [[FUNC_1]], [[FUNC_2]]) {
    // CHECK-SAME:  per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<3x48x60x60xf32>

    // CHECK: return [[CONCAT]] : tensor<3x48x60x60xf32>
}
