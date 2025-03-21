//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --override-tile-executor-num="override-to-tiles-per-batch-mode=revert" %s | FileCheck %s
// REQUIRES: arch-NPU40XX
module @OverrideToTilesPerBatchForNonBatchedCase {
    IE.TileResource 4 of @NCE at 1.700000e+03 MHz
    IE.CNNNetwork entryPoint : @SingleInputSingleOutputNonBatched
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    // CHECK: IE.TileResource 4 of @NCE at 1.700000e+03 MHz

    func.func private @SingleInputSingleOutputNonBatched_Batch1(%arg0: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        %1 = IE.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>
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
}

// -----

module @OverrideToTilesPerBatchForBatchedCase {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz
    IE.CNNNetwork entryPoint : @MultipleInputSingleOutputDeBatched
    inputsInfo : {
        DataInfo "input1" : tensor<3x3x62x62xf32>
        DataInfo "input2" : tensor<3x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<3x48x60x60xf32>
    }

    // CHECK: IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    func.func private @MultipleInputSingleOutputDeBatched_Batch1(%arg0: tensor<1x3x62x62xf32>, %arg1: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        %1 = IE.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %2 = IE.Add(%1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %3 = IE.SoftMax(%2) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %3 : tensor<1x48x60x60xf32>
    }

    // CHECK-LABEL: @MultipleInputSingleOutputDeBatched
    func.func @MultipleInputSingleOutputDeBatched(%arg0: tensor<3x3x62x62xf32>, %arg1: tensor<3x48x60x60xf32>) -> tensor<3x48x60x60xf32> {
        %0 = builtin.unrealized_conversion_cast %arg0 : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
        %1 = builtin.unrealized_conversion_cast %arg1 : tensor<3x48x60x60xf32> to tensor<1x48x60x60xf32>
        %2 = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 62, 62] : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
        %3 = IE.Slice %arg1 [0, 0, 0, 0] [1, 48, 60, 60] : tensor<3x48x60x60xf32> to tensor<1x48x60x60xf32>
        %4 = call @MultipleInputSingleOutputDeBatched_Batch1(%2, %3) {available_tiles = 6 : i64, debatched = [0, 3]} : (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        %5 = IE.Slice %arg0 [1, 0, 0, 0] [1, 3, 62, 62] : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
        %6 = IE.Slice %arg1 [1, 0, 0, 0] [1, 48, 60, 60] : tensor<3x48x60x60xf32> to tensor<1x48x60x60xf32>
        %7 = call @MultipleInputSingleOutputDeBatched_Batch1(%5, %6) {available_tiles = 6 : i64, debatched = [1, 3]} : (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        %8 = IE.Slice %arg0 [2, 0, 0, 0] [1, 3, 62, 62] : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
        %9 = IE.Slice %arg1 [2, 0, 0, 0] [1, 48, 60, 60] : tensor<3x48x60x60xf32> to tensor<1x48x60x60xf32>
        %10 = call @MultipleInputSingleOutputDeBatched_Batch1(%8, %9) {available_tiles = 6 : i64, debatched = [2, 3]} : (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        %11 = IE.Concat(%4, %7, %10) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<3x48x60x60xf32>
        %12 = call @MultipleInputSingleOutputDeBatched_Batch1(%0, %1) {available_tiles = 6 : i64, debatched = [1, 3]} : (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        %13 = builtin.unrealized_conversion_cast %12 : tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
        return %11 : tensor<3x48x60x60xf32>

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
}
