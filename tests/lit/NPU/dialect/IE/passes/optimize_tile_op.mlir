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

// -----

// CHECK-LABEL: @FoldSpatialBroadcastTileBeforeAdd
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x128x64x64xf16>
func.func @FoldSpatialBroadcastTileBeforeAdd(%feat: tensor<1x128x64x64xf16>) -> tensor<1x128x64x64xf16> {
    %bias = const.Declare tensor<1x128x1x1xf16> = dense<1.0> : tensor<1x128x1x1xf16>
    %0 = IE.Tile(%bias) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Add(%feat, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x128x1x1xf16>
    // CHECK-NOT:   IE.Tile
    // CHECK:       [[ADD:%.+]] = IE.Add([[FEAT]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:                   : tensor<1x128x64x64xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       return [[ADD]] : tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: @FoldSpatialBroadcastTileBeforeMultiply
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x128x64x64xf16>
func.func @FoldSpatialBroadcastTileBeforeMultiply(%feat: tensor<1x128x64x64xf16>) -> tensor<1x128x64x64xf16> {
    %scale = const.Declare tensor<1x128x1x1xf16> = dense<1.0> : tensor<1x128x1x1xf16>
    %0 = IE.Tile(%scale) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Multiply(%feat, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<1x128x1x1xf16>
    // CHECK-NOT:   IE.Tile
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[FEAT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x128x64x64xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       return [[MUL]] : tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: @FoldSpatialBroadcastTileBeforeAddWithPostOp
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x128x64x64xf16>
func.func @FoldSpatialBroadcastTileBeforeAddWithPostOp(%feat: tensor<1x128x64x64xf16>) -> tensor<1x128x64x64xf16> {
    %bias = const.Declare tensor<1x128x1x1xf16> = dense<1.0> : tensor<1x128x1x1xf16>
    %0 = IE.Tile(%bias) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Add(%feat, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>} : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x128x1x1xf16>
    // CHECK-NOT:   IE.Tile
    // CHECK:       [[ADD:%.+]] = IE.Add([[FEAT]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>}
    // CHECK-SAME:                   : tensor<1x128x64x64xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       return [[ADD]] : tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: @NotFoldSpatialBroadcastTileWhenChannelTiled
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x64x28x28xf16>, [[BIAS:%.+]]: tensor<1x32x1x1xf16>
func.func @NotFoldSpatialBroadcastTileWhenChannelTiled(%feat: tensor<1x64x28x28xf16>, %bias: tensor<1x32x1x1xf16>) -> tensor<1x64x28x28xf16> {
    %0 = IE.Tile(%bias) {repeats_values = [1, 2, 28, 28]} : tensor<1x32x1x1xf16> -> tensor<1x64x28x28xf16>
    %1 = IE.Add(%feat, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x28x28xf16>, tensor<1x64x28x28xf16> -> tensor<1x64x28x28xf16>
    return %1 : tensor<1x64x28x28xf16>

    // CHECK:       [[TILE:%.+]] = IE.Tile([[BIAS]]) {repeats_values = [1, 2, 28, 28]}
    // CHECK:       [[ADD:%.+]] = IE.Add([[FEAT]], [[TILE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:       return [[ADD]] : tensor<1x64x28x28xf16>
}

// -----

// CHECK-LABEL: @NotFoldSpatialBroadcastTileWhenOutputIsOneDSpatial
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x2304x1x576xf16>, [[BIAS:%.+]]: tensor<1x2304x1x1xf16>
// Bias [1x2304x1x1] tiled only in W -> output [1x2304x1x576] has H==1.
// SHAVE broadcast Add on a 1D (H=1) tensor is slower than the original Tile + NCE Eltwise.
func.func @NotFoldSpatialBroadcastTileWhenOutputIsOneDSpatial(
        %feat: tensor<1x2304x1x576xf16>, %bias: tensor<1x2304x1x1xf16>) -> tensor<1x2304x1x576xf16> {
    %0 = IE.Tile(%bias) {repeats_values = [1, 1, 1, 576]} : tensor<1x2304x1x1xf16> -> tensor<1x2304x1x576xf16>
    %1 = IE.Add(%feat, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2304x1x576xf16>, tensor<1x2304x1x576xf16> -> tensor<1x2304x1x576xf16>
    return %1 : tensor<1x2304x1x576xf16>

    // Tile must be kept: output H==1 means SHAVE broadcast Add would be slower than NCE Eltwise.
    // CHECK:       [[TILE:%.+]] = IE.Tile([[BIAS]]) {repeats_values = [1, 1, 1, 576]}
    // CHECK:       [[ADD:%.+]] = IE.Add([[FEAT]], [[TILE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x2304x1x576xf16>, tensor<1x2304x1x576xf16> -> tensor<1x2304x1x576xf16>
    // CHECK:       return [[ADD]] : tensor<1x2304x1x576xf16>
}

// -----

// CHECK-LABEL: @NotFoldSpatialBroadcastTileWhenBothOperandsBroadcasted
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x512x1x1xf16>, [[INPUT2:%.+]]: tensor<1x512x1x1xf16>
// Use a large tensor (1x512x64x64 ~ 4MB) so the size threshold is met and the fold would be
// attempted -- only the output-shape guard (broadcast(1x512x1x1, 1x512x1x1) = 1x512x1x1 != 1x512x64x64)
// prevents it.
func.func @NotFoldSpatialBroadcastTileWhenBothOperandsBroadcasted(
        %arg0: tensor<1x512x1x1xf16>, %arg1: tensor<1x512x1x1xf16>) -> tensor<1x512x64x64xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 64, 64]} : tensor<1x512x1x1xf16> -> tensor<1x512x64x64xf16>
    %1 = IE.Add(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x64x64xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x64x64xf16>
    return %1 : tensor<1x512x64x64xf16>

    // Tile must be kept: removing it changes output shape from 1x512x64x64 to 1x512x1x1.
    // CHECK:       [[TILE:%.+]] = IE.Tile([[INPUT1]]) {repeats_values = [1, 1, 64, 64]}
    // CHECK:       [[ADD:%.+]] = IE.Add([[TILE]], [[INPUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x512x64x64xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x64x64xf16>
    // CHECK:       return [[ADD]] : tensor<1x512x64x64xf16>
}

// -----

// CHECK-LABEL: @FuseGroupConvWithBiasAdd
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x4x4xf16>
func.func @FuseGroupConvWithBiasAdd(%arg0: tensor<1x4x4x4xf16>) -> tensor<1x4x4x4xf16> {
    %filter = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00> : tensor<4x1x1x1xf16>
    %bias   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01> : tensor<1x4x1x1xf16>
    %tiled  = IE.Tile(%bias) {repeats_values = [1, 1, 4, 4]} : tensor<1x4x1x1xf16> -> tensor<1x4x4x4xf16>
    %gconv  = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
              : tensor<1x4x4x4xf16>, tensor<4x1x1x1xf16> -> tensor<1x4x4x4xf16>
    %add    = IE.Add(%gconv, %tiled) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
              : tensor<1x4x4x4xf16>, tensor<1x4x4x4xf16> -> tensor<1x4x4x4xf16>
    return %add : tensor<1x4x4x4xf16>

    // Static filter + static bias: FoldTileOpRewriter force-folds the Tile via isSpatialBroadcastFromGroupConv,
    // exposing the 1xCx1x1 bias directly; FuseGroupConvWithBiasAdd then fuses it into GroupConvolutionOp.
    // CHECK-DAG: [[FILTER:%.+]] = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-DAG: [[BIAS:%.+]]   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01>
    // CHECK-NOT: IE.Tile
    // CHECK:     [[FUSED:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:    {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:    : tensor<1x4x4x4xf16>, tensor<4x1x1x1xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x4x4xf16>
    // CHECK-NOT: IE.Add
    // CHECK:     return [[FUSED]] : tensor<1x4x4x4xf16>
}

// -----

// CHECK-LABEL: @NotForceFoldTileWhenGroupConvHasPostOp
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x4x4xf16>
func.func @NotForceFoldTileWhenGroupConvHasPostOp(%arg0: tensor<1x4x4x4xf16>) -> tensor<1x4x4x4xf16> {
    %filter = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00> : tensor<4x1x1x1xf16>
    %bias   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01> : tensor<1x4x1x1xf16>
    %tiled  = IE.Tile(%bias) {repeats_values = [1, 1, 4, 4]} : tensor<1x4x1x1xf16> -> tensor<1x4x4x4xf16>
    %gconv  = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0],
                                                    post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>, strides = [1, 1]}
              : tensor<1x4x4x4xf16>, tensor<4x1x1x1xf16> -> tensor<1x4x4x4xf16>
    %add    = IE.Add(%gconv, %tiled) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
              : tensor<1x4x4x4xf16>, tensor<1x4x4x4xf16> -> tensor<1x4x4x4xf16>
    return %add : tensor<1x4x4x4xf16>

    // Regression guard: GroupConvolution already has a post-op (e.g. ReLU6 on a MobileNet-style
    // depthwise conv), so FuseGroupConvWithBiasAdd will reject fusion. isSpatialBroadcastFromGroupConv
    // must therefore NOT force-fold the Tile here either -- otherwise the Tile would be stripped
    // (assuming fusion completes) while fusion actually fails, leaving a slow de-tiled SW
    // broadcast Add on a small tensor instead of the original Tile + equal-shape NCE.Eltwise.
    // CHECK:     [[TILE:%.+]] = IE.Tile({{%.+}})
    // CHECK:     [[GCONV:%.+]] = IE.GroupConvolution([[INPUT]], {{%.+}})
    // CHECK-SAME:    post_op = #IE.LeakyRelu
    // CHECK:     IE.Add([[GCONV]], [[TILE]])
}

// -----

// CHECK-LABEL: @NotForceFoldTileWhenGroupConvHasMultipleUses
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x4x4xf16>
func.func @NotForceFoldTileWhenGroupConvHasMultipleUses(%arg0: tensor<1x4x4x4xf16>) -> (tensor<1x4x4x4xf16>, tensor<1x4x4x4xf16>) {
    %filter = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00> : tensor<4x1x1x1xf16>
    %bias   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01> : tensor<1x4x1x1xf16>
    %tiled  = IE.Tile(%bias) {repeats_values = [1, 1, 4, 4]} : tensor<1x4x1x1xf16> -> tensor<1x4x4x4xf16>
    %gconv  = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
              : tensor<1x4x4x4xf16>, tensor<4x1x1x1xf16> -> tensor<1x4x4x4xf16>
    %add    = IE.Add(%gconv, %tiled) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
              : tensor<1x4x4x4xf16>, tensor<1x4x4x4xf16> -> tensor<1x4x4x4xf16>
    return %gconv, %add : tensor<1x4x4x4xf16>, tensor<1x4x4x4xf16>

    // Regression guard: GroupConvolution output feeds both the Add and a skip connection (the
    // function return), so FuseGroupConvWithBiasAdd will reject fusion (hasOneUse() is false).
    // isSpatialBroadcastFromGroupConv must therefore NOT force-fold the Tile here either.
    // CHECK:     [[TILE:%.+]] = IE.Tile({{%.+}})
    // CHECK:     [[GCONV:%.+]] = IE.GroupConvolution([[INPUT]], {{%.+}})
    // CHECK:     [[ADD:%.+]] = IE.Add([[GCONV]], [[TILE]])
    // CHECK:     return [[GCONV]], [[ADD]]
}

// -----

// CHECK-LABEL: @FuseGroupConvWithBiasAddDirectBias
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x4x4xf16>
func.func @FuseGroupConvWithBiasAddDirectBias(%arg0: tensor<1x4x4x4xf16>) -> tensor<1x4x4x4xf16> {
    %filter = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00> : tensor<4x1x1x1xf16>
    %bias   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01> : tensor<1x4x1x1xf16>
    %gconv  = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
              : tensor<1x4x4x4xf16>, tensor<4x1x1x1xf16> -> tensor<1x4x4x4xf16>
    %add    = IE.Add(%gconv, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
              : tensor<1x4x4x4xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x4x4xf16>
    return %add : tensor<1x4x4x4xf16>

    // No Tile present -- bias is already 1xCx1x1; fuse directly.
    // CHECK-DAG: [[FILTER:%.+]] = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-DAG: [[BIAS:%.+]]   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01>
    // CHECK:     [[FUSED:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-NOT: IE.Add
    // CHECK:     return [[FUSED]] : tensor<1x4x4x4xf16>
}

// -----

// CHECK-LABEL: @FuseGroupConvWithBiasAddBiasOnLhs
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x4x4xf16>
func.func @FuseGroupConvWithBiasAddBiasOnLhs(%arg0: tensor<1x4x4x4xf16>) -> tensor<1x4x4x4xf16> {
    %filter = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00> : tensor<4x1x1x1xf16>
    %bias   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01> : tensor<1x4x1x1xf16>
    %gconv  = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
              : tensor<1x4x4x4xf16>, tensor<4x1x1x1xf16> -> tensor<1x4x4x4xf16>
    %add    = IE.Add(%bias, %gconv) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
              : tensor<1x4x1x1xf16>, tensor<1x4x4x4xf16> -> tensor<1x4x4x4xf16>
    return %add : tensor<1x4x4x4xf16>

    // GroupConvolution on rhs, bias on lhs -- symmetry test.
    // CHECK-DAG: [[FILTER:%.+]] = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-DAG: [[BIAS:%.+]]   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01>
    // CHECK:     [[FUSED:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-NOT: IE.Add
    // CHECK:     return [[FUSED]] : tensor<1x4x4x4xf16>
}

// -----

// CHECK-LABEL: @FuseGroupConvWithBiasAddPostOpTransferred
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x4x4xf16>
func.func @FuseGroupConvWithBiasAddPostOpTransferred(%arg0: tensor<1x4x4x4xf16>) -> tensor<1x4x4x4xf16> {
    %filter = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00> : tensor<4x1x1x1xf16>
    %bias   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01> : tensor<1x4x1x1xf16>
    %gconv  = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
              : tensor<1x4x4x4xf16>, tensor<4x1x1x1xf16> -> tensor<1x4x4x4xf16>
    %add    = IE.Add(%gconv, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                                     post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>}
              : tensor<1x4x4x4xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x4x4xf16>
    return %add : tensor<1x4x4x4xf16>

    // Add's LeakyRelu post-op is transferred to the fused GroupConvolution.
    // CHECK-DAG: [[FILTER:%.+]] = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-DAG: [[BIAS:%.+]]   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01>
    // CHECK:     [[FUSED:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:    post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
    // CHECK-NOT: IE.Add
    // CHECK:     return [[FUSED]] : tensor<1x4x4x4xf16>
}

// -----

// CHECK-LABEL: @NotFuseGroupConvWithBiasAddWhenAddHasStaticScale
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x4x4xf16>
func.func @NotFuseGroupConvWithBiasAddWhenAddHasStaticScale(%arg0: tensor<1x4x4x4xf16>) -> tensor<1x4x4x4xf16> {
    %filter = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00> : tensor<4x1x1x1xf16>
    %bias   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01> : tensor<1x4x1x1xf16>
    %tiled  = IE.Tile(%bias) {repeats_values = [1, 1, 4, 4]} : tensor<1x4x1x1xf16> -> tensor<1x4x4x4xf16>
    %gconv  = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
              : tensor<1x4x4x4xf16>, tensor<4x1x1x1xf16> -> tensor<1x4x4x4xf16>
    %add    = IE.Add(%gconv, %tiled) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, static_scale = 2.000000e+00 : f32}
              : tensor<1x4x4x4xf16>, tensor<1x4x4x4xf16> -> tensor<1x4x4x4xf16>
    return %add : tensor<1x4x4x4xf16>

    // Add carries a static_scale attribute -- fusing into GroupConvolution's bias slot would
    // silently drop the scale, so FuseGroupConvWithBiasAdd must not fuse (independent Tile-folding
    // by FoldTileOpRewriter still removes the Tile, exposing the bias directly).
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-DAG:   [[BIAS:%.+]]   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01>
    // CHECK-NOT:   IE.Tile
    // CHECK:       [[GC_OUT:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]])
    // CHECK:       IE.Add([[GC_OUT]], [[BIAS]])
    // CHECK-SAME:      static_scale
}

// -----

// CHECK-LABEL: @NotFuseGroupConvWithBiasAddMultipleUses
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x4x4xf16>
func.func @NotFuseGroupConvWithBiasAddMultipleUses(%arg0: tensor<1x4x4x4xf16>) -> (tensor<1x4x4x4xf16>, tensor<1x4x4x4xf16>) {
    %filter = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00> : tensor<4x1x1x1xf16>
    %bias   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01> : tensor<1x4x1x1xf16>
    %gconv  = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
              : tensor<1x4x4x4xf16>, tensor<4x1x1x1xf16> -> tensor<1x4x4x4xf16>
    %add    = IE.Add(%gconv, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
              : tensor<1x4x4x4xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x4x4xf16>
    return %gconv, %add : tensor<1x4x4x4xf16>, tensor<1x4x4x4xf16>

    // GroupConvolution output is used by both Add and the function return -- hasOneUse() is
    // false, so FuseGroupConvWithBiasAdd must not fuse.
    // CHECK: [[GCONV:%.+]] = IE.GroupConvolution([[INPUT]], {{%.+}})
    // CHECK: IE.Add([[GCONV]], {{%.+}})
    // CHECK: return [[GCONV]]
}

// -----

// CHECK-LABEL: @NotFuseGroupConvWithBiasAddWhenGroupConvHasPostOp
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x4x4xf16>
func.func @NotFuseGroupConvWithBiasAddWhenGroupConvHasPostOp(%arg0: tensor<1x4x4x4xf16>) -> tensor<1x4x4x4xf16> {
    %filter = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00> : tensor<4x1x1x1xf16>
    %bias   = const.Declare tensor<1x4x1x1xf16> = dense<5.000000e-01> : tensor<1x4x1x1xf16>
    %gconv  = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0],
                                                    post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>, strides = [1, 1]}
              : tensor<1x4x4x4xf16>, tensor<4x1x1x1xf16> -> tensor<1x4x4x4xf16>
    %add    = IE.Add(%gconv, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
              : tensor<1x4x4x4xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x4x4xf16>
    return %add : tensor<1x4x4x4xf16>

    // GroupConvolution already has a post-op (PPE) -- hasPPE() guard rejects fusion.
    // CHECK:     [[GCONV:%.+]] = IE.GroupConvolution([[INPUT]], {{%.+}})
    // CHECK-SAME:    post_op = #IE.LeakyRelu
    // CHECK:     IE.Add([[GCONV]], {{%.+}})
}

// -----

// CHECK-LABEL: @NotFuseGroupConvWithBiasAddBiasWrongShape
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x4x4xf16>
func.func @NotFuseGroupConvWithBiasAddBiasWrongShape(%arg0: tensor<1x4x4x4xf16>) -> tensor<1x4x4x4xf16> {
    %filter  = const.Declare tensor<4x1x1x1xf16> = dense<1.000000e+00> : tensor<4x1x1x1xf16>
    %bias    = const.Declare tensor<1x4x4x4xf16> = dense<5.000000e-01> : tensor<1x4x4x4xf16>
    %gconv   = IE.GroupConvolution(%arg0, %filter) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
              : tensor<1x4x4x4xf16>, tensor<4x1x1x1xf16> -> tensor<1x4x4x4xf16>
    %add     = IE.Add(%gconv, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
              : tensor<1x4x4x4xf16>, tensor<1x4x4x4xf16> -> tensor<1x4x4x4xf16>
    return %add : tensor<1x4x4x4xf16>

    // Direct bias with shape 1x4x4x4 (H=4, W=4 != 1): the 1xCx1x1 shape guard in
    // FuseGroupConvWithBiasAdd rejects fusion.
    // CHECK: [[GCONV:%.+]] = IE.GroupConvolution([[INPUT]], {{%.+}})
    // CHECK: IE.Add([[GCONV]], {{%.+}})
}

// -----

// CHECK-LABEL: @NotFuseGroupConvWithBiasAddWhenGroupConvHasBias
func.func @NotFuseGroupConvWithBiasAddWhenGroupConvHasBias(%input: tensor<1x4x4x4xf16>, %filter: tensor<4x1x1x1xf16>, %bias: tensor<1x4x1x1xf16>) -> tensor<1x4x4x4xf16> {
    %cst_bias = const.Declare tensor<1x4x1x1xf16> = dense<0.0> : tensor<1x4x1x1xf16>
    %0 = IE.GroupConvolution(%input, %filter, %cst_bias) {
            dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x4x4x4xf16>, tensor<4x1x1x1xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x4x4xf16>
    %1 = IE.Tile(%bias) {repeats_values = [1, 1, 4, 4]} : tensor<1x4x1x1xf16> -> tensor<1x4x4x4xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x4x4xf16>, tensor<1x4x4x4xf16> -> tensor<1x4x4x4xf16>
    return %2 : tensor<1x4x4x4xf16>

    // GroupConvolution already carries a bias -- cannot fuse another one.
    // CHECK:       IE.GroupConvolution
    // CHECK:       IE.Tile
    // CHECK:       IE.Add
}

// -----

// CHECK-LABEL: @FuseGroupConvWithBiasAddLargeTensor
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x128x64x64xf16>, [[BIAS:%.+]]: tensor<1x128x1x1xf16>
func.func @FuseGroupConvWithBiasAddLargeTensor(%input: tensor<1x128x64x64xf16>, %bias: tensor<1x128x1x1xf16>) -> tensor<1x128x64x64xf16> {
    %filter = const.Declare tensor<128x1x1x1xf16> = dense<1.0> : tensor<128x1x1x1xf16>
    %0 = IE.GroupConvolution(%input, %filter) {
            dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x128x64x64xf16>, tensor<128x1x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Tile(%bias) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %2 : tensor<1x128x64x64xf16>

    // Static filter + non-constant bias: FuseGroupConvWithBiasAdd skips (non-constant bias is not a
    // Const::DeclareOp). FoldTileOpRewriter also skips: the other Add input comes from
    // IE.GroupConvolution with a static (const) filter, so the NCE-chain guard prevents folding
    // (keeping the equal-shape NCE Add preserves DPU eltwise scheduling chained with the upstream GroupConv).
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<128x1x1x1xf16>
    // CHECK:       [[GC_OUT:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]]
    // CHECK:       [[TILED:%.+]] = IE.Tile([[BIAS]]) {repeats_values = [1, 1, 64, 64]}
    // CHECK:       IE.Add([[GC_OUT]], [[TILED]])
}

// -----

// CHECK-LABEL: @NotFoldSpatialBroadcastTileWhenOtherAddInputIsConv
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x128x64x64xf16>, [[PROJ:%.+]]: tensor<1x128x1x1xf16>
// Stable Diffusion time-embedding pattern: runtime 1xCx1x1 bias tiled to HxW, added to NCE
// Convolution output. The tiled output (1x128x64x64, ~1MB) exceeds the size threshold, but the
// fold is suppressed because the other Add input comes from IE.Convolution. Keeping the equal-shape
// NCE Add preserves DPU eltwise scheduling chained with the upstream conv.
func.func @NotFoldSpatialBroadcastTileWhenOtherAddInputIsConv(
        %input: tensor<1x128x64x64xf16>, %proj: tensor<1x128x1x1xf16>) -> tensor<1x128x64x64xf16> {
    %weight = const.Declare tensor<128x128x1x1xf16> = dense<1.0> : tensor<128x128x1x1xf16>
    %conv = IE.Convolution(%input, %weight) {
            dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x128x64x64xf16>, tensor<128x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %0 = IE.Tile(%proj) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Add(%conv, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // CHECK-DAG:   [[WEIGHT:%.+]] = const.Declare tensor<128x128x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHT]])
    // CHECK:       [[TILED:%.+]] = IE.Tile([[PROJ]]) {repeats_values = [1, 1, 64, 64]}
    // CHECK-SAME:      tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       IE.Add([[CONV]], [[TILED]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: @FoldSpatialBroadcastTileBeforeAdd_NonConstantInput
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x128x64x64xf16>, [[PROJ:%.+]]: tensor<1x128x1x1xf16>
// AdaIN conditioner: runtime 1xCx1x1 bias (from Multiply/Slice chain, not Convolution)
// tiled to HxW. The other Add input is not from an NCE Convolution so the fold fires and
// removes the explicit Tile, letting the Add use NUMPY broadcast directly.
func.func @FoldSpatialBroadcastTileBeforeAdd_NonConstantInput(
        %feat: tensor<1x128x64x64xf16>, %proj: tensor<1x128x1x1xf16>) -> tensor<1x128x64x64xf16> {
    %0 = IE.Tile(%proj) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Add(%feat, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // CHECK-NOT:   IE.Tile
    // CHECK:       IE.Add([[FEAT]], [[PROJ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x128x64x64xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: @FoldSpatialBroadcastTileBeforeAdd_NonConstantFilterGroupConv
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x128x64x64xf16>, [[SCALE:%.+]]: tensor<128x1x1x1xf16>, [[PROJ:%.+]]: tensor<1x128x1x1xf16>
// AdaIN per-channel scale pattern: non-constant-filter GroupConv (depthwise multiply) → Tile → Add.
// The GroupConv filter is a runtime value, so the GroupConv is NOT an NCE conv that chains
// with a downstream NCE Add. The guard does not fire and the Tile is folded.
func.func @FoldSpatialBroadcastTileBeforeAdd_NonConstantFilterGroupConv(
        %feat: tensor<1x128x64x64xf16>, %scale: tensor<128x1x1x1xf16>,
        %proj: tensor<1x128x1x1xf16>) -> tensor<1x128x64x64xf16> {
    %gc = IE.GroupConvolution(%feat, %scale) {
            dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x128x64x64xf16>, tensor<128x1x1x1xf16> -> tensor<1x128x64x64xf16>
    %0 = IE.Tile(%proj) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Add(%gc, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // CHECK:       [[GC_OUT:%.+]] = IE.GroupConvolution([[FEAT]], [[SCALE]])
    // CHECK-NOT:   IE.Tile
    // CHECK:       IE.Add([[GC_OUT]], [[PROJ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x128x64x64xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: @FoldSpatialBroadcastTileNonConstantBiasWithPostOpStripped
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x128x64x64xf16>, [[SCALE:%.+]]: tensor<128x1x1x1xf16>, [[PROJ:%.+]]: tensor<1x128x1x1xf16>
// AdaIN pattern: dynamic-filter GroupConv → Tile(non-constant bias) → Add(post_op=LeakyRelu).
// The Tile must be folded for performance (avoids large HxW DMA), but the post_op is stripped
// from the Add and replaced with a standalone IE.LeakyRelu so that RunAddBroadcastOnDPU can
// convert the clean broadcast Add to NCEMaxPool (per-channel weight-table bias) without
// silently dropping the activation.
func.func @FoldSpatialBroadcastTileNonConstantBiasWithPostOpStripped(
        %feat: tensor<1x128x64x64xf16>, %scale: tensor<128x1x1x1xf16>,
        %proj: tensor<1x128x1x1xf16>) -> tensor<1x128x64x64xf16> {
    %gc = IE.GroupConvolution(%feat, %scale) {
            dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x128x64x64xf16>, tensor<128x1x1x1xf16> -> tensor<1x128x64x64xf16>
    %0 = IE.Tile(%proj) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Add(%gc, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                          post_op = #IE.LeakyRelu<negative_slope = 1.000000e-02 : f64>}
             : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // Tile is folded; Add has post_op stripped; standalone LeakyRelu inserted after Add.
    // CHECK:       [[GC_OUT:%.+]] = IE.GroupConvolution([[FEAT]], [[SCALE]])
    // CHECK-NOT:   IE.Tile
    // CHECK:       [[ADD:%.+]] = IE.Add([[GC_OUT]], [[PROJ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x128x64x64xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       [[LRELU:%.+]] = IE.LeakyRelu([[ADD]]) {negative_slope = 1.000000e-02 : f64}
    // CHECK-SAME:      tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       return [[LRELU]] : tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: @FoldSpatialBroadcastTileNonConstantBiasWithPostOpStripped_TileOnLhs
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x128x64x64xf16>, [[SCALE:%.+]]: tensor<128x1x1x1xf16>, [[PROJ:%.+]]: tensor<1x128x1x1xf16>
// AdaIN: same as FoldSpatialBroadcastTileNonConstantBiasWithPostOpStripped but Tile is input1
// (lhsIsTileOp=true). Verifies that newAdd uses proj as input1 and gc as input2.
func.func @FoldSpatialBroadcastTileNonConstantBiasWithPostOpStripped_TileOnLhs(
        %feat: tensor<1x128x64x64xf16>, %scale: tensor<128x1x1x1xf16>,
        %proj: tensor<1x128x1x1xf16>) -> tensor<1x128x64x64xf16> {
    %gc = IE.GroupConvolution(%feat, %scale) {
            dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x128x64x64xf16>, tensor<128x1x1x1xf16> -> tensor<1x128x64x64xf16>
    %0 = IE.Tile(%proj) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Add(%0, %gc) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                          post_op = #IE.LeakyRelu<negative_slope = 1.000000e-02 : f64>}
             : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // CHECK:       [[GC_OUT:%.+]] = IE.GroupConvolution([[FEAT]], [[SCALE]])
    // CHECK-NOT:   IE.Tile
    // CHECK:       [[ADD:%.+]] = IE.Add([[PROJ]], [[GC_OUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x128x1x1xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       [[LRELU:%.+]] = IE.LeakyRelu([[ADD]]) {negative_slope = 1.000000e-02 : f64}
    // CHECK-SAME:      tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       return [[LRELU]] : tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: @FoldSpatialBroadcastTileNonConstantBiasWithReLUPostOpStripped
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x128x64x64xf16>, [[SCALE:%.+]]: tensor<128x1x1x1xf16>, [[PROJ:%.+]]: tensor<1x128x1x1xf16>
// AdaIN pattern with post_op = IE.Relu (not LeakyRelu). Verifies the IE.Relu branch of the
// strip-post_op block: a standalone IE.ReLU is inserted after the clean broadcast Add.
func.func @FoldSpatialBroadcastTileNonConstantBiasWithReLUPostOpStripped(
        %feat: tensor<1x128x64x64xf16>, %scale: tensor<128x1x1x1xf16>,
        %proj: tensor<1x128x1x1xf16>) -> tensor<1x128x64x64xf16> {
    %gc = IE.GroupConvolution(%feat, %scale) {
            dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x128x64x64xf16>, tensor<128x1x1x1xf16> -> tensor<1x128x64x64xf16>
    %0 = IE.Tile(%proj) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Add(%gc, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                          post_op = #IE.Relu<>}
             : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // CHECK:       [[GC_OUT:%.+]] = IE.GroupConvolution([[FEAT]], [[SCALE]])
    // CHECK-NOT:   IE.Tile
    // CHECK:       [[ADD:%.+]] = IE.Add([[GC_OUT]], [[PROJ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x128x64x64xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[ADD]])
    // CHECK-SAME:      tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       return [[RELU]] : tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: @NotFoldSpatialBroadcastTileNonConstantBiasBelowSizeThreshold
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x128x16x16xf16>, [[SCALE:%.+]]: tensor<128x1x1x1xf16>, [[PROJ:%.+]]: tensor<1x128x1x1xf16>
// Small tensor: 1x128x16x16xf16 = 65,536 bytes. Below isLargeEnoughForDPUOverSHAVE threshold on
// all supported platforms (most restrictive: NPU4000 threshold = ceil(1,571,840 / 6) = 261,974 B).
// isFoldBeneficial=false => hasSpatiallyBroadcastedInput=false. Tile and Add with its LeakyRelu
// post_op must be preserved so the equal-shape NCE Eltwise can apply the activation via PPE.
func.func @NotFoldSpatialBroadcastTileNonConstantBiasBelowSizeThreshold(
        %feat: tensor<1x128x16x16xf16>, %scale: tensor<128x1x1x1xf16>,
        %proj: tensor<1x128x1x1xf16>) -> tensor<1x128x16x16xf16> {
    %gc = IE.GroupConvolution(%feat, %scale) {
            dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x128x16x16xf16>, tensor<128x1x1x1xf16> -> tensor<1x128x16x16xf16>
    %0 = IE.Tile(%proj) {repeats_values = [1, 1, 16, 16]} : tensor<1x128x1x1xf16> -> tensor<1x128x16x16xf16>
    %1 = IE.Add(%gc, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                          post_op = #IE.LeakyRelu<negative_slope = 1.000000e-02 : f64>}
             : tensor<1x128x16x16xf16>, tensor<1x128x16x16xf16> -> tensor<1x128x16x16xf16>
    return %1 : tensor<1x128x16x16xf16>

    // CHECK:       [[GC_OUT:%.+]] = IE.GroupConvolution([[FEAT]], [[SCALE]])
    // CHECK:       [[TILE:%.+]] = IE.Tile([[PROJ]]) {repeats_values = [1, 1, 16, 16]}
    // CHECK-SAME:      tensor<1x128x1x1xf16> -> tensor<1x128x16x16xf16>
    // CHECK:       IE.Add([[GC_OUT]], [[TILE]])
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 1.000000e-02 : f64>
    // CHECK-SAME:      tensor<1x128x16x16xf16>, tensor<1x128x16x16xf16> -> tensor<1x128x16x16xf16>
}

// -----

// CHECK-LABEL: @FoldSpatialBroadcastTileNonConstantBiasWithClampPostOpStripped
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x128x64x64xf16>, [[SCALE:%.+]]: tensor<128x1x1x1xf16>, [[PROJ:%.+]]: tensor<1x128x1x1xf16>
// AdaIN pattern with clamp-only (no post_op). Verifies the clamp-only branch of the
// strip-post_op block: a standalone IE.Clamp is inserted after the clean broadcast Add.
// Also guards against the previous bug where an unrecognized post_op would fall through to
// the clamp branch instead of aborting the fold.
func.func @FoldSpatialBroadcastTileNonConstantBiasWithClampPostOpStripped(
        %feat: tensor<1x128x64x64xf16>, %scale: tensor<128x1x1x1xf16>,
        %proj: tensor<1x128x1x1xf16>) -> tensor<1x128x64x64xf16> {
    %gc = IE.GroupConvolution(%feat, %scale) {
            dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x128x64x64xf16>, tensor<128x1x1x1xf16> -> tensor<1x128x64x64xf16>
    %0 = IE.Tile(%proj) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Add(%gc, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                          clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}}
             : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // Tile is folded; Add has clamp stripped; standalone IE.Clamp inserted after Add.
    // CHECK:       [[GC_OUT:%.+]] = IE.GroupConvolution([[FEAT]], [[SCALE]])
    // CHECK-NOT:   IE.Tile
    // CHECK:       [[ADD:%.+]] = IE.Add([[GC_OUT]], [[PROJ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x128x64x64xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[ADD]]) {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}
    // CHECK-SAME:      tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       return [[CLAMP]] : tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: @FoldSpatialBroadcastTileNonConstantBiasWithReluAndClampStripped
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x128x64x64xf16>, [[SCALE:%.+]]: tensor<128x1x1x1xf16>, [[PROJ:%.+]]: tensor<1x128x1x1xf16>
// post_op and clamp co-exist (ReLU6 pattern: post_op=Relu + clamp={min=0,max=6}).
// Both must be preserved: the fold strips both from the Add, inserts a standalone IE.ReLU
// first, then chains a standalone IE.Clamp after it (the new postOpAttr+clampAttr code path).
func.func @FoldSpatialBroadcastTileNonConstantBiasWithReluAndClampStripped(
        %feat: tensor<1x128x64x64xf16>, %scale: tensor<128x1x1x1xf16>,
        %proj: tensor<1x128x1x1xf16>) -> tensor<1x128x64x64xf16> {
    %gc = IE.GroupConvolution(%feat, %scale) {
            dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x128x64x64xf16>, tensor<128x1x1x1xf16> -> tensor<1x128x64x64xf16>
    %0 = IE.Tile(%proj) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Add(%gc, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                          post_op = #IE.Relu<>,
                          clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}}
             : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // Tile folded; Add stripped of both post_op and clamp; ReLU then Clamp inserted in sequence.
    // CHECK:       [[GC_OUT:%.+]] = IE.GroupConvolution([[FEAT]], [[SCALE]])
    // CHECK-NOT:   IE.Tile
    // CHECK:       [[ADD:%.+]] = IE.Add([[GC_OUT]], [[PROJ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x128x64x64xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[ADD]])
    // CHECK-SAME:      tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[RELU]]) {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}
    // CHECK-SAME:      tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       return [[CLAMP]] : tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: @NotFoldSpatialBroadcastTileNonConstantBiasUnrecognizedPostOp
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x128x64x64xf16>, [[SCALE:%.+]]: tensor<128x1x1x1xf16>, [[PROJ:%.+]]: tensor<1x128x1x1xf16>
// Unrecognized post_op (#IE.Sigmoid<>) on the Add: the fold must abort and leave the
// original Tile and Add+postOp untouched to avoid silently dropping the activation.
func.func @NotFoldSpatialBroadcastTileNonConstantBiasUnrecognizedPostOp(
        %feat: tensor<1x128x64x64xf16>, %scale: tensor<128x1x1x1xf16>,
        %proj: tensor<1x128x1x1xf16>) -> tensor<1x128x64x64xf16> {
    %gc = IE.GroupConvolution(%feat, %scale) {
            dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x128x64x64xf16>, tensor<128x1x1x1xf16> -> tensor<1x128x64x64xf16>
    %0 = IE.Tile(%proj) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Add(%gc, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                          post_op = #IE.Sigmoid<>}
             : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // Unrecognized post_op: fold aborts; Tile and Add+Sigmoid are preserved unchanged.
    // CHECK:       IE.GroupConvolution([[FEAT]], [[SCALE]])
    // CHECK:       [[TILE:%.+]] = IE.Tile([[PROJ]]) {repeats_values = [1, 1, 64, 64]}
    // CHECK-SAME:      tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       IE.Add({{%.+}}, [[TILE]])
    // CHECK-SAME:      post_op = #IE.Sigmoid<>
}

// -----

// CHECK-LABEL: @NotFoldSpatialBroadcastTileNonConstantBiasUnrecognizedPostOpWithClamp
// CHECK-SAME:  [[FEAT:%.+]]: tensor<1x128x64x64xf16>, [[SCALE:%.+]]: tensor<128x1x1x1xf16>, [[PROJ:%.+]]: tensor<1x128x1x1xf16>
// Unrecognized post_op with clamp both set: fold must abort. Previously the unrecognized
// post_op would be silently dropped and only the clamp applied — a correctness bug. Fixed:
// any unrecognized post_op (regardless of clamp) now aborts to preserve the original semantics.
func.func @NotFoldSpatialBroadcastTileNonConstantBiasUnrecognizedPostOpWithClamp(
        %feat: tensor<1x128x64x64xf16>, %scale: tensor<128x1x1x1xf16>,
        %proj: tensor<1x128x1x1xf16>) -> tensor<1x128x64x64xf16> {
    %gc = IE.GroupConvolution(%feat, %scale) {
            dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x128x64x64xf16>, tensor<128x1x1x1xf16> -> tensor<1x128x64x64xf16>
    %0 = IE.Tile(%proj) {repeats_values = [1, 1, 64, 64]} : tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    %1 = IE.Add(%gc, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                          post_op = #IE.Sigmoid<>,
                          clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}}
             : tensor<1x128x64x64xf16>, tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // Unrecognized post_op with clamp: fold aborts; Tile and Add+Sigmoid+clamp preserved.
    // CHECK:       IE.GroupConvolution([[FEAT]], [[SCALE]])
    // CHECK:       [[TILE:%.+]] = IE.Tile([[PROJ]]) {repeats_values = [1, 1, 64, 64]}
    // CHECK-SAME:      tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    // CHECK:       IE.Add({{%.+}}, [[TILE]])
    // CHECK-SAME:      clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}
    // CHECK-SAME:      post_op = #IE.Sigmoid<>
}
