//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --split-input-file --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @FoldTile
func.func @FoldTile(%arg0: tensor<3x4x2xf32>) -> tensor<3x4x2xf32> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1]} : tensor<3x4x2xf32> -> tensor<3x4x2xf32>
    // CHECK-NOT:   IE.Tile
    return %0 : tensor<3x4x2xf32>
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @FoldTileWithBroadcast
func.func @FoldTileWithBroadcast() -> tensor<4x6xsi64> {
    %0 = const.Declare tensor<2x2xsi64> = dense<[[0, 1], [2, 3]]> : tensor<2x2xsi64>
    // CHECK-NOT:   IE.Tile
    %1 = IE.Tile(%0) {repeats_values = [2, 3]} : tensor<2x2xsi64> -> tensor<4x6xsi64>
    // CHECK:       [[VAL0:%.+]] = const.Declare tensor<4x6xsi64> =
    // CHECK-SAME{LITERAL}:          dense<[[0, 1], [2, 3]]> : tensor<2x2xsi64>,
    // CHECK-SAME:                  [#const.Broadcast<0 : i64, 4 : i64>, #const.Broadcast<1 : i64, 6 : i64>]
    return %1 : tensor<4x6xsi64>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @InsertUnsqueezeBeforedTile
// CHECK-SAME: [[INPUT:%.+]]: tensor<2x3xf32>
func.func @InsertUnsqueezeBeforedTile(%arg0: tensor<2x3xf32>) -> tensor<1x6x15xf32> {
    // CHECK:       [[VAL0:%.+]] = IE.Unsqueeze([[INPUT]]) {axes_value = [0]} : tensor<2x3xf32> -> tensor<1x2x3xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 3, 5]} : tensor<2x3xf32> -> tensor<1x6x15xf32>
    // CHECK:       [[VAL1:%.+]] = IE.Tile([[VAL0]]) {repeats_values = [1, 3, 5]} : tensor<1x2x3xf32> -> tensor<1x6x15xf32>

    return %0 : tensor<1x6x15xf32>
    // CHECK:       return [[VAL1]]
}

// -----

// CHECK-LABEL: @FuseTwoTiles
// CHECK-SAME: [[INPUT:%.+]]: tensor<2x3x4xf32>
func.func @FuseTwoTiles(%arg0: tensor<2x3x4xf32>) -> tensor<8x9x32xf32> {
    %cst = const.Declare tensor<3xsi64> = dense<[2, 1, 2]> : tensor<3xsi64>
    %1 = IE.Tile(%arg0) {repeats_values = [2, 3, 4]} : tensor<2x3x4xf32> -> tensor<4x9x16xf32>
    %2 = IE.Tile(%1, %cst) : tensor<4x9x16xf32>, tensor<3xsi64> -> tensor<8x9x32xf32>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[INPUT]]) {
    // CHECK-SAME:      repeats_values = [4, 3, 8]} : tensor<2x3x4xf32> -> tensor<8x9x32xf32>

    return %2 : tensor<8x9x32xf32>
    // CHECK:       return [[TILE]] : tensor<8x9x32xf32>
}

// -----

// CHECK-LABEL: @FuseTwoTilesWithDiffDimSmallToLarge
// CHECK-SAME: [[INPUT:%.+]]: tensor<2x3x4xf32>
func.func @FuseTwoTilesWithDiffDimSmallToLarge(%arg0: tensor<2x3x4xf32>) -> tensor<2x8x27x64xf32> {
    // CHECK:       [[UNSQUEEZE:%.+]] = IE.Unsqueeze([[INPUT]]) {axes_value = [0]} : tensor<2x3x4xf32> -> tensor<1x2x3x4xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [2, 3, 4]} : tensor<2x3x4xf32> -> tensor<4x9x16xf32>
    %1 = IE.Tile(%0) {repeats_values = [2, 2, 3, 4]} : tensor<4x9x16xf32> -> tensor<2x8x27x64xf32>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[UNSQUEEZE]]) {
    // CHECK-SAME:      repeats_values = [2, 4, 9, 16]} : tensor<1x2x3x4xf32> -> tensor<2x8x27x64xf32>

    return %1 : tensor<2x8x27x64xf32>
    // CHECK:       return [[TILE]] : tensor<2x8x27x64xf32>
}

// -----

// CHECK-LABEL: @FuseTwoTilesWithDiffDimLargeToSmall
// CHECK-SAME: [[INPUT:%.+]]: tensor<2x3x4xf32>
func.func @FuseTwoTilesWithDiffDimLargeToSmall(%arg0: tensor<2x3x4xf32>) -> tensor<2x8x27x64xf32> {
    // CHECK:       [[UNSQUEEZE:%.+]] = IE.Unsqueeze([[INPUT]]) {axes_value = [0]} : tensor<2x3x4xf32> -> tensor<1x2x3x4xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [2, 2, 3, 4]} : tensor<2x3x4xf32> -> tensor<2x4x9x16xf32>
    %1 = IE.Tile(%0) {repeats_values = [2, 3, 4]} : tensor<2x4x9x16xf32> -> tensor<2x8x27x64xf32>
    // CHECK:       [[TILE:%.+]] = IE.Tile([[UNSQUEEZE]]) {
    // CHECK-SAME:      repeats_values = [2, 4, 9, 16]} : tensor<1x2x3x4xf32> -> tensor<2x8x27x64xf32>

    return %1 : tensor<2x8x27x64xf32>
    // CHECK:       return [[TILE]] : tensor<2x8x27x64xf32>
}
