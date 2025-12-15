//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --de-debatcher="debatching-inlining-method=reordering" %s | FileCheck %s
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

    // CHECK: [[FUNC_0:%.*]] = call @MultipleInputSingleOutputDeBatched_Batch1([[SLICE_00]], [[SLICE_01]]) {debatched = [0, 3], reordering = 1 : i64} :
    // CHECK-SAME:  (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>

    // CHECK: [[SLICE_10:%.*]] = IE.Slice [[ARG0]] [1, 0, 0, 0] [1, 3, 62, 62] :
    // CHECK-SAME:  tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
    // CHECK: [[SLICE_11:%.*]] = IE.Slice [[ARG1]] [1, 0, 0, 0] [1, 48, 60, 60] :
    // CHECK-SAME:  tensor<3x48x60x60xf32> to tensor<1x48x60x60xf32>

    // CHECK: [[FUNC_1:%.*]] = call @MultipleInputSingleOutputDeBatched_Batch1([[SLICE_10]], [[SLICE_11]]) {debatched = [1, 3], reordering = 1 : i64} :
    // CHECK-SAME:  (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>

    // CHECK: [[SLICE_20:%.*]] = IE.Slice [[ARG0]] [2, 0, 0, 0] [1, 3, 62, 62] :
    // CHECK-SAME:  tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
    // CHECK: [[SLICE_21:%.*]] = IE.Slice [[ARG1]] [2, 0, 0, 0] [1, 48, 60, 60] :
    // CHECK-SAME:  tensor<3x48x60x60xf32> to tensor<1x48x60x60xf32>

    // CHECK: [[FUNC_2:%.*]] = call @MultipleInputSingleOutputDeBatched_Batch1([[SLICE_20]], [[SLICE_21]]) {debatched = [2, 3], reordering = 1 : i64} :
    // CHECK-SAME:  (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>

    // CHECK: [[CONCAT:%.*]] = IE.Concat([[FUNC_0]], [[FUNC_1]], [[FUNC_2]]) {
    // CHECK-SAME:  per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<3x48x60x60xf32>

    // CHECK: return [[CONCAT]] : tensor<3x48x60x60xf32>
}

// -----

func.func private @MultipleInputDifferentRanksSingleOutputDeBatched_Batch1(%arg0: tensor<1x3x62x62xf32>, %arg1: tensor<1x48x3600xf32>) -> tensor<1x48x60x60xf32> {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
    %1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
    %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    %3 = VPU.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 48, 60, 60]} :
        tensor<1x48x3600xf32> -> tensor<1x48x60x60xf32>
    %4 = IE.Add(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    %5 = IE.SoftMax(%3) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    return %5 : tensor<1x48x60x60xf32>
}

// CHECK-LABEL: @MultipleInputDifferentRanksSingleOutputDeBatched
func.func @MultipleInputDifferentRanksSingleOutputDeBatched(%arg0: tensor<3x3x62x62xf32>, %arg1: tensor<3x48x3600xf32>) -> tensor<3x48x60x60xf32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
    %1 = builtin.unrealized_conversion_cast %arg1 : tensor<3x48x3600xf32> to tensor<1x48x3600xf32>
    %2 = call @MultipleInputDifferentRanksSingleOutputDeBatched_Batch1(%0, %1) : (tensor<1x3x62x62xf32>, tensor<1x48x3600xf32>) -> tensor<1x48x60x60xf32>
    %3 = builtin.unrealized_conversion_cast %2: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
    return %3 : tensor<3x48x60x60xf32>

    // CHECK: func.func @MultipleInputDifferentRanksSingleOutputDeBatched([[ARG0:%.+]]: tensor<3x3x62x62xf32>, [[ARG1:%.+]]: tensor<3x48x3600xf32>) -> tensor<3x48x60x60xf32> {

    // CHECK: [[SLICE_00:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 3, 62, 62] :
    // CHECK-SAME:  tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
    // CHECK: [[SLICE_01:%.+]] = IE.Slice [[ARG1]] [0, 0, 0] [1, 48, 3600] :
    // CHECK-SAME:  tensor<3x48x3600xf32> to tensor<1x48x3600xf32>

    // CHECK: [[FUNC_0:%.+]] = call @MultipleInputDifferentRanksSingleOutputDeBatched_Batch1([[SLICE_00]], [[SLICE_01]]) {debatched = [0, 3], reordering = 1 : i64} :
    // CHECK-SAME:  (tensor<1x3x62x62xf32>, tensor<1x48x3600xf32>) -> tensor<1x48x60x60xf32>

    // CHECK: [[SLICE_10:%.+]] = IE.Slice [[ARG0]] [1, 0, 0, 0] [1, 3, 62, 62] :
    // CHECK-SAME:  tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
    // CHECK: [[SLICE_11:%.+]] = IE.Slice [[ARG1]] [1, 0, 0] [1, 48, 3600] :
    // CHECK-SAME:  tensor<3x48x3600xf32> to tensor<1x48x3600xf32>

    // CHECK: [[FUNC_1:%.+]] = call @MultipleInputDifferentRanksSingleOutputDeBatched_Batch1([[SLICE_10]], [[SLICE_11]]) {debatched = [1, 3], reordering = 1 : i64} :
    // CHECK-SAME:  (tensor<1x3x62x62xf32>, tensor<1x48x3600xf32>) -> tensor<1x48x60x60xf32>

    // CHECK: [[SLICE_20:%.+]] = IE.Slice [[ARG0]] [2, 0, 0, 0] [1, 3, 62, 62] :
    // CHECK-SAME:  tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
    // CHECK: [[SLICE_21:%.+]] = IE.Slice [[ARG1]] [2, 0, 0] [1, 48, 3600] :
    // CHECK-SAME:  tensor<3x48x3600xf32> to tensor<1x48x3600xf32>

    // CHECK: [[FUNC_2:%.+]] = call @MultipleInputDifferentRanksSingleOutputDeBatched_Batch1([[SLICE_20]], [[SLICE_21]]) {debatched = [2, 3], reordering = 1 : i64} :
    // CHECK-SAME:  (tensor<1x3x62x62xf32>, tensor<1x48x3600xf32>) -> tensor<1x48x60x60xf32>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[FUNC_0]], [[FUNC_1]], [[FUNC_2]]) {
    // CHECK-SAME:  per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<3x48x60x60xf32>

    // CHECK: return [[CONCAT]] : tensor<3x48x60x60xf32>
}

// -----

func.func private @SingleInputMultipleOutputDeBatched_Batch1(%arg0: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
    %1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
    %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    %4 = IE.SoftMax(%3) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    return %4, %1 : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
}

// CHECK-LABEL: @SingleInputMultipleOutputDeBatched
func.func @SingleInputMultipleOutputDeBatched(%arg0: tensor<3x3x62x62xf32>) -> (tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
    %1:2 = call @SingleInputMultipleOutputDeBatched_Batch1(%0) : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
    %2 = builtin.unrealized_conversion_cast %1#0: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
    %3 = builtin.unrealized_conversion_cast %1#1: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
    return %2, %3 : tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>


    // CHECK: func.func @SingleInputMultipleOutputDeBatched([[ARG0:%.+]]: tensor<3x3x62x62xf32>) -> (tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>) {
    // CHECK:   [[SLICE_0:%.*]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 3, 62, 62] :
    // CHECK-SAME:      tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>

    // CHECK:   [[FUNC_0:%.*]]:2 = call @SingleInputMultipleOutputDeBatched_Batch1([[SLICE_0]]) {debatched = [0, 3], reordering = 1 : i64} : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)

    // CHECK:   [[SLICE_1:%.*]] = IE.Slice [[ARG0]] [1, 0, 0, 0] [1, 3, 62, 62] :
    // CHECK-SAME:      tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>

    // CHECK:   [[FUNC_1:%.*]]:2 = call @SingleInputMultipleOutputDeBatched_Batch1([[SLICE_1]]) {debatched = [1, 3], reordering = 1 : i64} : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)

    // CHECK:   [[SLICE_2:%.*]] = IE.Slice [[ARG0]] [2, 0, 0, 0] [1, 3, 62, 62] :
    // CHECK-SAME:      tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>

    // CHECK:   [[FUNC_2:%.*]]:2 = call @SingleInputMultipleOutputDeBatched_Batch1([[SLICE_2]]) {debatched = [2, 3], reordering = 1 : i64} : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)

    // CHECK:   [[CONCAT_0:%.*]] = IE.Concat([[FUNC_0]]#0, [[FUNC_1]]#0, [[FUNC_2]]#0) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<3x48x60x60xf32>

    // CHECK:   [[CONCAT_1:%.*]] = IE.Concat([[FUNC_0]]#1, [[FUNC_1]]#1, [[FUNC_2]]#1) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<3x48x60x60xf32>

    // CHECK:   return [[CONCAT_0]], [[CONCAT_1]] : tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>
}

// -----

func.func private @SingleInputSingleOutputDeBatchedTo2_Batch2(%arg0: tensor<2x3x62x62xf32>) -> tensor<2x48x60x60xf32> {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
    %1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<2x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<2x48x60x60xf32>
    %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<2x48x60x60xf32> -> tensor<2x48x60x60xf32>
    return %2 : tensor<2x48x60x60xf32>
}

// CHECK-LABEL: @SingleInputSingleOutputDeBatchedTo2
func.func @SingleInputSingleOutputDeBatchedTo2(%arg0: tensor<6x3x62x62xf32>) -> tensor<6x48x60x60xf32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<6x3x62x62xf32> to tensor<2x3x62x62xf32>
    %1 = call @SingleInputSingleOutputDeBatchedTo2_Batch2(%0) : (tensor<2x3x62x62xf32>) -> tensor<2x48x60x60xf32>
    %2 = builtin.unrealized_conversion_cast %1: tensor<2x48x60x60xf32> to tensor<6x48x60x60xf32>
    return %2 : tensor<6x48x60x60xf32>

    // CHECK: func.func @SingleInputSingleOutputDeBatchedTo2([[ARG0:%.+]]: tensor<6x3x62x62xf32>) -> tensor<6x48x60x60xf32> {

    // CHECK: [[SLICE_0:%.*]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [2, 3, 62, 62] :
    // CHECK-SAME:  tensor<6x3x62x62xf32> to tensor<2x3x62x62xf32>
    // CHECK: [[FUNC_0:%.*]] = call @SingleInputSingleOutputDeBatchedTo2_Batch2([[SLICE_0]]) {debatched = [0, 3], reordering = 1 : i64} :
    // CHECK-SAME:  (tensor<2x3x62x62xf32>) -> tensor<2x48x60x60xf32>

    // CHECK: [[SLICE_1:%.*]] = IE.Slice [[ARG0]] [2, 0, 0, 0] [2, 3, 62, 62] :
    // CHECK-SAME:  tensor<6x3x62x62xf32> to tensor<2x3x62x62xf32>
    // CHECK: [[FUNC_1:%.*]] = call @SingleInputSingleOutputDeBatchedTo2_Batch2([[SLICE_1]]) {debatched = [1, 3], reordering = 1 : i64} :
    // CHECK-SAME:  (tensor<2x3x62x62xf32>) -> tensor<2x48x60x60xf32>

    // CHECK: [[SLICE_2:%.*]] = IE.Slice [[ARG0]] [4, 0, 0, 0] [2, 3, 62, 62] :
    // CHECK-SAME:  tensor<6x3x62x62xf32> to tensor<2x3x62x62xf32>
    // CHECK: [[FUNC_2:%.*]] = call @SingleInputSingleOutputDeBatchedTo2_Batch2([[SLICE_2]]) {debatched = [2, 3], reordering = 1 : i64} :
    // CHECK-SAME:  (tensor<2x3x62x62xf32>) -> tensor<2x48x60x60xf32>

    // CHECK: [[CONCAT:%.*]] = IE.Concat([[FUNC_0]], [[FUNC_1]], [[FUNC_2]]) {
    // CHECK-SAME:  per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<2x48x60x60xf32>, tensor<2x48x60x60xf32>, tensor<2x48x60x60xf32> -> tensor<6x48x60x60xf32>

    // CHECK: return [[CONCAT]] : tensor<6x48x60x60xf32>
}
