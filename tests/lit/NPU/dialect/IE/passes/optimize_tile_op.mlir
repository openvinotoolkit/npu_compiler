//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-tile-op %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

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

func.func @FoldTileBeforeMultiplyWithTransposedReshape(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x4x1x1xf32> {
    %cst_0 = const.Declare tensor<1x4x1x1xf32> = dense<1.0> : tensor<1x4x1x1xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 4]} : tensor<1x1x1x1xf32> -> tensor<1x1x1x4xf32>
    %1 = IE.Reshape(%0) { shape_value = [1, 4, 1, 1] } : tensor<1x1x1x4xf32> -> tensor<1x4x1x1xf32>
    %2 = IE.Multiply(%1, %cst_0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x1x1xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x1x1xf32>

    return %2 : tensor<1x4x1x1xf32>

    // CHECK-NOT:    IE.Tile(
    // CHECK-NOT:    IE.Reshape
    // CHECK:        IE.Multiply
    // CHECK-SAME:       tensor<1x1x1x1xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x1x1xf32>
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
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1024x1024xf16>
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

// CHECK-LABEL: @NotFoldLargeSingleChannelTileWhenBroadcastOutputChanges
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1024x1024xf16>
func.func @NotFoldLargeSingleChannelTileWhenBroadcastOutputChanges(%arg0: tensor<1x1x1024x1024xf16>) -> tensor<1x16x1024x1024xf16> {
    %cst_0 = const.Declare tensor<1x1x1024x1024xf16> = dense<1.0> : tensor<1x1x1024x1024xf16>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 16, 1, 1]} : tensor<1x1x1024x1024xf16> -> tensor<1x16x1024x1024xf16>
    %1 = IE.Add(%0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1024x1024xf16>, tensor<1x1x1024x1024xf16> -> tensor<1x16x1024x1024xf16>
    return %1 : tensor<1x16x1024x1024xf16>

    // CHECK:        [[TILE:%.+]] = IE.Tile([[INPUT]]) {repeats_values = [1, 16, 1, 1]}
    // CHECK-SAME:       tensor<1x1x1024x1024xf16> -> tensor<1x16x1024x1024xf16>
    // CHECK:        IE.Add([[TILE]], {{%.+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:       tensor<1x16x1024x1024xf16>, tensor<1x1x1024x1024xf16> -> tensor<1x16x1024x1024xf16>
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

// -----

// CHECK-LABEL: @NotFoldScalarTileWhenBroadcastOutputChanges
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x1xf32>
func.func @NotFoldScalarTileWhenBroadcastOutputChanges(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
    %cst = const.Declare tensor<1x1x1x4xf32> = dense<1.0> : tensor<1x1x1x4xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 4, 1]} : tensor<1x1x1x1xf32> -> tensor<1x1x4x1xf32>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x1xf32>, tensor<1x1x1x4xf32> -> tensor<1x1x4x4xf32>
    return %1 : tensor<1x1x4x4xf32>

    // CHECK:        [[TILE:%.+]] = IE.Tile([[INPUT]]) {repeats_values = [1, 1, 4, 1]}
    // CHECK-SAME:       tensor<1x1x1x1xf32> -> tensor<1x1x4x1xf32>
    // CHECK:        IE.Multiply([[TILE]], {{%.+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:       tensor<1x1x4x1xf32>, tensor<1x1x1x4xf32> -> tensor<1x1x4x4xf32>
}

// -----

// CHECK-LABEL: @FuseTileConvert
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x512xsi64>
func.func @FuseTileConvert(%input: tensor<1x1x1x512xsi64>) -> tensor<1x1x512x512xf16> {
    %0 = IE.Convert(%input) {dstElemType = si32} : tensor<1x1x1x512xsi64> -> tensor<1x1x1x512xsi32>
    %1 = IE.Tile(%0) {repeats_values = [1, 1, 512, 1]} : tensor<1x1x1x512xsi32> -> tensor<1x1x512x512xsi32>
    %2 = IE.Convert(%1) {dstElemType = f16} : tensor<1x1x512x512xsi32> -> tensor<1x1x512x512xf16>
    return %2 : tensor<1x1x512x512xf16>

    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f16}
    // CHECK-SAME:                       : tensor<1x1x1x512xsi64> -> tensor<1x1x1x512xf16>

    // CHECK:       [[TILE:%.+]] = IE.Tile([[CONVERT]]) {repeats_values = [1, 1, 512, 1]}
    // CHECK-SAME:                       : tensor<1x1x1x512xf16> -> tensor<1x1x512x512xf16>

    // CHECK:       return [[TILE]] : tensor<1x1x512x512xf16>
}

// -----

// CHECK-LABEL: @NotFuseTileConvertWhenDataSizeIncreases
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x512xf32>
func.func @NotFuseTileConvertWhenDataSizeIncreases(%input: tensor<1x1x1x512xf32>) -> tensor<1x1x512x512xf32> {
    %0 = IE.Convert(%input) {dstElemType = f16} : tensor<1x1x1x512xf32> -> tensor<1x1x1x512xf16>
    %1 = IE.Tile(%0) {repeats_values = [1, 1, 512, 1]} : tensor<1x1x1x512xf16> -> tensor<1x1x512x512xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf32>
    return %2 : tensor<1x1x512x512xf32>

    // CHECK:       [[CONVERT1:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f16}
    // CHECK-SAME:                       : tensor<1x1x1x512xf32> -> tensor<1x1x1x512xf16>

    // CHECK:       [[TILE:%.+]] = IE.Tile([[CONVERT1]]) {repeats_values = [1, 1, 512, 1]}
    // CHECK-SAME:                       : tensor<1x1x1x512xf16> -> tensor<1x1x512x512xf16>

    // CHECK:       [[CONVERT2:%.+]] = IE.Convert([[TILE]]) {dstElemType = f32}
    // CHECK-SAME:                       : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf32>

    // CHECK:       return [[CONVERT2]] : tensor<1x1x512x512xf32>
}
