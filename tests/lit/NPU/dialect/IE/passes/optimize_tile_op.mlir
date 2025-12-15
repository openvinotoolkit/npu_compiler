//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-tile-op %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

func.func @FoldTileBeforeMultiply(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
    %cst_0 = const.Declare tensor<1x1x4x4xf32> = dense<1.0> : tensor<1x1x4x4xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 16]} : tensor<1x1x1x1xf32> -> tensor<1x1x1x16xf32>
    %1 = IE.Reshape(%0) { shape_value = [1, 1, 4, 4] } : tensor<1x1x1x16xf32> -> tensor<1x1x4x4xf32>
    %2 = IE.Multiply(%1, %cst_0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>

    return %2 : tensor<1x1x4x4xf32>

    // CHECK-NOT:    IE.Tile(
    // CHECK:        IE.Multiply
}

func.func @FoldTileBeforeMultiplyWith3DInput(%arg0: tensor<1x1x1xf32>) -> tensor<1x1x4x4xf32> {
    %cst_0 = const.Declare tensor<1x1x4x4xf32> = dense<1.0> : tensor<1x1x4x4xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 16]} : tensor<1x1x1xf32> -> tensor<1x1x16xf32>
    %1 = IE.Reshape(%0) { shape_value = [1, 1, 4, 4] } : tensor<1x1x16xf32> -> tensor<1x1x4x4xf32>
    %2 = IE.Multiply(%1, %cst_0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>

    return %2 : tensor<1x1x4x4xf32>

    // CHECK-NOT:    IE.Tile(
    // CHECK:        IE.Reshape
    // CHECK-SAME:       {shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:        IE.Multiply
}

func.func @FoldTileBeforeAdd(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
    %cst_0 = const.Declare tensor<1x1x4x4xf32> = dense<1.0> : tensor<1x1x4x4xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 16]} : tensor<1x1x1x1xf32> -> tensor<1x1x1x16xf32>
    %1 = IE.Reshape(%0) { shape_value = [1, 1, 4, 4] } : tensor<1x1x1x16xf32> -> tensor<1x1x4x4xf32>
    %2 = IE.Add(%1, %cst_0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>

    return %2 : tensor<1x1x4x4xf32>

    // CHECK-NOT:    IE.Tile(
    // CHECK:        IE.Add
}

func.func @FoldTileBeforeAddWith3DInput(%arg0: tensor<1x1x1xf32>) -> tensor<1x1x4x4xf32> {
    %cst_0 = const.Declare tensor<1x1x4x4xf32> = dense<1.0> : tensor<1x1x4x4xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 16]} : tensor<1x1x1xf32> -> tensor<1x1x16xf32>
    %1 = IE.Reshape(%0) { shape_value = [1, 1, 4, 4] } : tensor<1x1x16xf32> -> tensor<1x1x4x4xf32>
    %2 = IE.Add(%1, %cst_0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>

    return %2 : tensor<1x1x4x4xf32>

    // CHECK-NOT:    IE.Tile(
    // CHECK:        IE.Reshape
    // CHECK-SAME:       {shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:        IE.Add
}

//
// -----
//

// CHECK-LABEL: @FoldTileBeforeAddWith4DInput
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x1x1024x1024xf16>
func.func @FoldTileBeforeAddWith4DInput(%arg0: tensor<1x1x1024x1024xf16>) -> tensor<1x16x1024x1024xf16> {
    %cst_0 = const.Declare tensor<1x16x1024x1024xf16> = dense<1.0> : tensor<1x16x1024x1024xf16>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 16, 1, 1]} : tensor<1x1x1024x1024xf16> -> tensor<1x16x1024x1024xf16>
    %1 = IE.Add(%0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1024x1024xf16>, tensor<1x16x1024x1024xf16> -> tensor<1x16x1024x1024xf16>
    return %1 : tensor<1x16x1024x1024xf16>
    // CHECK:        [[CST:%.+]] = const.Declare tensor<1x16x1024x1024xf16>
    // CHECK-NOT:    IE.Tile
    // CHECK:        [[ADD:%.+]] = IE.Add([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:                     tensor<1x1x1024x1024xf16>, tensor<1x16x1024x1024xf16> -> tensor<1x16x1024x1024xf16>
    // CHECK:        return [[ADD]] : tensor<1x16x1024x1024xf16>
}

// -----

// CHECK-LABEL: @NotFoldForInOutDifferentPrecision
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1024x1024xf16>
func.func @NotFoldForInOutDifferentPrecision(%arg0: tensor<1x1x1024x1024xf16>) -> tensor<1x16x1024x1024xf32> {
    %cst_0 = const.Declare tensor<1x16x1024x1024xf16> = dense<1.0> : tensor<1x16x1024x1024xf16>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 16, 1, 1]} : tensor<1x1x1024x1024xf16> -> tensor<1x16x1024x1024xf16>
    %1 = IE.Add(%0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1024x1024xf16>, tensor<1x16x1024x1024xf16> -> tensor<1x16x1024x1024xf32>
    return %1 : tensor<1x16x1024x1024xf32>

    // CHECK:        [[CST:%.+]] = const.Declare
    // CHECK:        [[TILE:%.+]] = IE.Tile
    // CHECK:        [[ADD:%.+]] = IE.Add
    // CHECK:        return [[ADD]] : tensor<1x16x1024x1024xf32>
}
