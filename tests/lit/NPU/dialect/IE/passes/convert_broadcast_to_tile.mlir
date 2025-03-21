//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-broadcast-to-tile %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @ConvertBroadcastNumpyToTile
// CHECK-SAME: [[INPUT:%.+]]: tensor<3x1xf16>
func.func @ConvertBroadcastNumpyToTile(%arg0: tensor<3x1xf16>) -> tensor<2x3x6xf16> {
    %cst = const.Declare tensor<3xsi64> = dense<[2, 3, 6]> : tensor<3xsi64>
    %0 = IE.Broadcast(%arg0, %cst) {mode = #IE.broadcast_type<NUMPY>} : tensor<3x1xf16>, tensor<3xsi64> -> tensor<2x3x6xf16>
    return %0 : tensor<2x3x6xf16>

    // CHECK-NOT:           IE.Broadcast
    // CHECK:               [[RESHAPE:%.*]] = IE.Reshape([[INPUT]]) {shape_value = [1, 3, 1]} : tensor<3x1xf16> -> tensor<1x3x1xf16>
    // CHECK:               [[TILE:%.*]] = IE.Tile([[RESHAPE]]) {repeats_values = [2, 1, 6]} : tensor<1x3x1xf16> -> tensor<2x3x6xf16>
    // CHECK:               return [[TILE]]
}

// CHECK-LABEL: @ConvertBroadcastBidirectionalToTile
// CHECK-SAME: [[INPUT:%.+]]: tensor<4x1xf16>
func.func @ConvertBroadcastBidirectionalToTile(%arg0: tensor<4x1xf16>) -> tensor<2x4x4xf16> {
    %cst = const.Declare tensor<3xsi64> = dense<[2, 1, 4]> : tensor<3xsi64>
    %0 = IE.Broadcast(%arg0, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<4x1xf16>, tensor<3xsi64> -> tensor<2x4x4xf16>
    return %0 : tensor<2x4x4xf16>

    // CHECK-NOT:           IE.Broadcast
    // CHECK:               [[RESHAPE:%.*]] = IE.Reshape([[INPUT]]) {shape_value = [1, 4, 1]} : tensor<4x1xf16> -> tensor<1x4x1xf16>
    // CHECK:               [[TILE:%.*]] = IE.Tile(%0) {repeats_values = [2, 1, 4]} : tensor<1x4x1xf16> -> tensor<2x4x4xf16>
    // CHECK:               return [[TILE]]
}

// CHECK-LABEL: @ConvertBroadcastExplicitToTile
// CHECK-SAME: [[INPUT:%.+]]: tensor<2x4xf16>
func.func @ConvertBroadcastExplicitToTile(%arg0: tensor<2x4xf16>) -> tensor<2x3x4xf16> {
    %cst = const.Declare tensor<3xsi32> = dense<[2, 3, 4]> : tensor<3xsi64>
    %cst_0 = const.Declare tensor<2xsi32> = dense<[0, 2]> : tensor<2xsi64>
    %0 = IE.Broadcast(%arg0, %cst, %cst_0) {mode = #IE.broadcast_type<EXPLICIT>} : tensor<2x4xf16>, tensor<3xsi32>, tensor<2xsi32> -> tensor<2x3x4xf16>
    return %0 : tensor<2x3x4xf16>

    // CHECK-NOT:           IE.Broadcast
    // CHECK:               [[RESHAPE:%.*]] = IE.Reshape([[INPUT]]) {shape_value = [2, 1, 4]} : tensor<2x4xf16> -> tensor<2x1x4xf16>
    // CHECK:               [[TILE:%.*]] = IE.Tile(%0) {repeats_values = [1, 3, 1]} : tensor<2x1x4xf16> -> tensor<2x3x4xf16>
    // CHECK:               return [[TILE]]
}
