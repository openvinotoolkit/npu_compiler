//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-op-slice %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @NoChangesAcrossInputsAtHeight
func.func @NoChangesAcrossInputsAtHeight(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 2, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

    // CHECK: [[VAR0:%.+]] = IE.Concat(%arg0, %arg1)
    // CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} :
    // CHECK-SAME:            tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    // CHECK:        [[VAR1:%.+]] = IE.Slice [[VAR0]]
    // CHECK-SAME:          [0, 0, 2, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    // CHECK: return [[VAR1]] : tensor<1x16x3x4xf16>
}

// -----

// CHECK-LABEL: @ChangesInInput0AtHeight
func.func @ChangesInInput0AtHeight(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 1, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

   // CHECK: [[VAR0:%.+]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 16, 3, 4] : tensor<1x16x4x4xf16> to tensor<1x16x3x4xf16>
   // CHECK: return [[VAR0]] : tensor<1x16x3x4xf16>
}

// -----

// CHECK-LABEL: @ChangesInInput1SameShapeAtHeight
func.func @ChangesInInput1SameShapeAtHeight(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 4, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

   // CHECK: return %arg1 : tensor<1x16x3x4xf16>
}

// -----

// CHECK-LABEL: @ChangesAtHeightTwoUsers
func.func @ChangesAtHeightTwoUsers(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> (tensor<1x16x3x4xf16>, tensor<1x16x3x4xf16>) {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 1, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    %2 = IE.Slice %0 [0, 0, 2, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1, %2 : tensor<1x16x3x4xf16>, tensor<1x16x3x4xf16>

   // CHECK: [[VAR0:%.+]] = IE.Concat(%arg0, %arg1)
   // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} :
   // CHECK-SAME:            tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
   // CHECK: [[VAR1:%.+]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 16, 3, 4] : tensor<1x16x4x4xf16> to tensor<1x16x3x4xf16>
   // CHECK: [[VAR2:%.+]] = IE.Slice [[VAR0]]
   // CHECK-SAME:           [0, 0, 2, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
   // CHECK:  return [[VAR1]], [[VAR2:%.+]] : tensor<1x16x3x4xf16>, tensor<1x16x3x4xf16>

}

// -----

// CHECK-LABEL: @NoChangesNoOffsetAttribute
func.func @NoChangesNoOffsetAttribute(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) { per_axis = #IE.Concat<axis = 2 : i64> } : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 4, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

    // CHECK: [[VAR0:%.+]] = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    // CHECK: [[VAR1:%.+]] = IE.Slice [[VAR0]] [0, 0, 4, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    // CHECK: return [[VAR1:%.+]] : tensor<1x16x3x4xf16>

}

// -----

// CHECK-LABEL: @NoChangesAtHeightWidth
func.func @NoChangesAtHeightWidth(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>, %arg2 : tensor<1x16x7x3xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16>, tensor<1x16x7x3xf16> -> tensor<1x16x7x7xf16>
    %1 = IE.Slice %0 [0, 0, 2, 2] [1, 16, 3, 4] : tensor<1x16x7x7xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

   // CHECK: [[VAR0:%.+]] = IE.Concat(%arg0, %arg1, %arg2)
   // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]} :
   // CHECK-SAME:            tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16>, tensor<1x16x7x3xf16> -> tensor<1x16x7x7xf16>
   // CHECK: [[VAR1:%.+]] = IE.Slice [[VAR0]]
   // CHECK-SAME:           [0, 0, 2, 2] [1, 16, 3, 4] : tensor<1x16x7x7xf16> to tensor<1x16x3x4xf16>
   // CHECK:  return [[VAR1:%.+]] : tensor<1x16x3x4xf16>

}

// -----

// CHECK-LABEL: @ChangesAtHeightWidth
func.func @ChangesAtHeightWidth(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>, %arg2 : tensor<1x16x7x3xf16>) -> tensor<1x16x2x2xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16>, tensor<1x16x7x3xf16> -> tensor<1x16x7x7xf16>
    %1 = IE.Slice %0 [0, 0, 5, 1] [1, 16, 2, 2] : tensor<1x16x7x7xf16> to tensor<1x16x2x2xf16>
    return %1 : tensor<1x16x2x2xf16>

    // CHECK: [[VAR0:%.+]] = IE.Slice %arg1 [0, 0, 1, 1] [1, 16, 2, 2] : tensor<1x16x3x4xf16> to tensor<1x16x2x2xf16>
    // CHECK: return [[VAR0]] : tensor<1x16x2x2xf16>

}

// -----

// CHECK-LABEL: @ChangesAtChannelHeight
func.func @ChangesAtChannelHeight(%arg0: tensor<1x3x4x4xf16>, %arg1 : tensor<1x3x3x4xf16>, %arg2 : tensor<1x2x7x4xf16>) -> tensor<1x2x2x2xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 3, 0, 0]]} : tensor<1x3x4x4xf16>, tensor<1x3x3x4xf16>, tensor<1x2x7x4xf16> -> tensor<1x5x7x4xf16>
    %1 = IE.Slice %0 [0, 1, 5, 1] [1, 2, 2, 2] : tensor<1x5x7x4xf16> to tensor<1x2x2x2xf16>
    return %1 : tensor<1x2x2x2xf16>

    // CHECK: [[VAR0:%.+]] = IE.Slice %arg1 [0, 1, 1, 1] [1, 2, 2, 2] : tensor<1x3x3x4xf16> to tensor<1x2x2x2xf16>
    // CHECK: return [[VAR0]] : tensor<1x2x2x2xf16>

}

// -----

// CHECK-LABEL: @NoChangesAcrossInputsAtHeightWithAffinReshape
func.func @NoChangesAcrossInputsAtHeightWithAffinReshape(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x1x12xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 16, 1, 28]} : tensor<1x16x7x4xf16> -> tensor<1x16x1x28xf16>
    %2 = IE.Slice %1 [0, 0, 0, 8] [1, 16, 1, 12] : tensor<1x16x1x28xf16> to tensor<1x16x1x12xf16>
    return %2 : tensor<1x16x1x12xf16>

    // CHECK: [[CONCAT:%.*]] = IE.Concat({{[^:]+}}, {{[^:]+}})
    // CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    // CHECK:        [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 16, 1, 28]} : tensor<1x16x7x4xf16> -> tensor<1x16x1x28xf16>
    // CHECK:        [[SLICE:%.*]] = IE.Slice [[AFFINERESHAPE]] [0, 0, 0, 8] [1, 16, 1, 12] : tensor<1x16x1x28xf16> to tensor<1x16x1x12xf16>
    // CHECK:        return [[SLICE]] : tensor<1x16x1x12xf16>

}

// -----

// CHECK-LABEL: @ChangesInInput0AtHeightWithAffinReshape
func.func @ChangesInInput0AtHeightWithAffinReshape(%arg0: tensor<1x1x1x1xf16>, %arg1: tensor<1x1x1x1xf16>, %arg2: tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16> {
    %3 = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x3xf16>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 3, 1, 1]} : tensor<1x1x1x3xf16> -> tensor<1x3x1x1xf16>
    %5 = IE.Slice %4 [0, 1, 0, 0] [1, 1, 1, 1] : tensor<1x3x1x1xf16> to tensor<1x1x1x1xf16>
    return %5 : tensor<1x1x1x1xf16>

    // CHECK: return {{[^:]+}} : tensor<1x1x1x1xf16>

}

// -----

// CHECK-LABEL: @TileAffineReshapeSliceOpt
func.func @TileAffineReshapeSliceOpt(%arg0: tensor<1x128x3x1xf16>) -> tensor<1x128x249xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 128, 300]} : tensor<1x128x3x100xf16> -> tensor<1x128x300xf16>
    %2 = IE.Slice %1 [0, 0, 0] [1, 128, 249] : tensor<1x128x300xf16> to tensor<1x128x249xf16>

    return %2 : tensor<1x128x249xf16>

    // CHECK:        [[TILE:%.*]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x83xf16>
    // CHECK:        [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[TILE]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 128, 249]} : tensor<1x128x3x83xf16> -> tensor<1x128x249xf16>
    // CHECK:        return [[AFFINERESHAPE]] : tensor<1x128x249xf16>

}

// -----

// The SliceOp output shape is 1x128x248, the data on dim 2 with shape "248" is sliced from TileOp output 3x100. 248 is not divisible by 3.
// So the pattern should not be optimized.
// CHECK-LABEL: @TileAffineReshapeSliceNotOpt
func.func @TileAffineReshapeSliceNotOpt(%arg0: tensor<1x128x3x1xf16>) -> tensor<1x128x248xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 128, 300]} : tensor<1x128x3x100xf16> -> tensor<1x128x300xf16>
    %2 = IE.Slice %1 [0, 0, 0] [1, 128, 248] : tensor<1x128x300xf16> to tensor<1x128x248xf16>

    return %2 : tensor<1x128x248xf16>

    // CHECK:        [[TILE:%.*]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    // CHECK:        [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[TILE]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 128, 300]} : tensor<1x128x3x100xf16> -> tensor<1x128x300xf16>
    // CHECK:        [[SLICE:%.*]] = IE.Slice [[AFFINERESHAPE]] [0, 0, 0] [1, 128, 248] : tensor<1x128x300xf16> to tensor<1x128x248xf16>
    // CHECK:        return [[SLICE]] : tensor<1x128x248xf16>

}

// -----

// CHECK-LABEL: @TileSliceOpt
func.func @TileSliceOpt(%arg0: tensor<1x128x3x1xf16>) -> tensor<1x128x3x83xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 128, 3, 83] : tensor<1x128x3x100xf16> to tensor<1x128x3x83xf16>

    return %1 : tensor<1x128x3x83xf16>

    // CHECK:        [[TILE:%.*]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x83xf16>
    // CHECK:        return [[TILE]] : tensor<1x128x3x83xf16>

}

// -----

// The slice performs on 2 Axes, so the pattern should not be optimized.
// CHECK-LABEL: @TileSliceNotOpt
func.func @TileSliceNotOpt(%arg0: tensor<1x128x3x1xf16>) -> tensor<1x128x2x83xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 128, 2, 83] : tensor<1x128x3x100xf16> to tensor<1x128x2x83xf16>

    return %1 : tensor<1x128x2x83xf16>

    // CHECK:        [[TILE:%.*]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    // CHECK:        [[SLICE:%.*]] = IE.Slice [[TILE]] [0, 0, 0, 0] [1, 128, 2, 83] : tensor<1x128x3x100xf16> to tensor<1x128x2x83xf16>
    // CHECK:        return [[SLICE]] : tensor<1x128x2x83xf16>

}

// -----

// CHECK-LABEL: @TileSliceNotOptForMultiUser
func.func @TileSliceNotOptForMultiUser(%arg0: tensor<1x128x3x1xf16>) -> tensor<1x64x6x100xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 64, 3, 100] : tensor<1x128x3x100xf16> to tensor<1x64x3x100xf16>
    %2 = IE.Slice %0 [0, 64, 0, 0] [1, 64, 3, 100] : tensor<1x128x3x100xf16> to tensor<1x64x3x100xf16>
    %3 = IE.Concat(%1, %2) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x64x3x100xf16>, tensor<1x64x3x100xf16> -> tensor<1x64x6x100xf16>

    return %3 : tensor<1x64x6x100xf16>

    // CHECK:        [[TILE:%.*]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    // CHECK:        [[SLICE0:%.*]] = IE.Slice [[TILE]] [0, 0, 0, 0] [1, 64, 3, 100] : tensor<1x128x3x100xf16> to tensor<1x64x3x100xf16>
    // CHECK:        [[SLICE1:%.*]] = IE.Slice [[TILE]] [0, 64, 0, 0] [1, 64, 3, 100] : tensor<1x128x3x100xf16> to tensor<1x64x3x100xf16>
    // CHECK:        [[CONCAT:%.*]] = IE.Concat([[SLICE0]], [[SLICE1]])
    // CHECK:        return [[CONCAT]] : tensor<1x64x6x100xf16>

}

// -----

// we should not optimize the case when there is no reshape, slice and tile axis are different
// CHECK-LABEL: @TileSliceNotOptForDifferentAxis
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x128x3x1xf16>
func.func @TileSliceNotOptForDifferentAxis(%arg0: tensor<1x128x3x1xf16>) -> tensor<1x128x2x100xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 128, 2, 100] : tensor<1x128x3x100xf16> to tensor<1x128x2x100xf16>

    return %1 : tensor<1x128x2x100xf16>

    // CHECK:        [[TILE:%.+]] = IE.Tile
    // CHECK:        [[SLICE:%.+]] = IE.Slice
    // CHECK:        return [[SLICE]] : tensor<1x128x2x100xf16>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseSliceConcat
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<4x128x3x6xf16, {order = #NHWC}>
func.func @FuseSliceConcat(%arg0: tensor<4x128x3x6xf16, {order = #NHWC}>) -> tensor<7x128x3x6xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [1, 0, 0, 0] [2, 128, 3, 6] : tensor<4x128x3x6xf16, {order = #NHWC}> to tensor<2x128x3x6xf16, {order = #NHWC}>
    %1 = IE.Slice %arg0 [3, 0, 0, 0] [1, 128, 3, 6] : tensor<4x128x3x6xf16, {order = #NHWC}> to tensor<1x128x3x6xf16, {order = #NHWC}>

    %2 = IE.Concat(%0, %1, %arg0) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<2x128x3x6xf16, {order = #NHWC}>, tensor<1x128x3x6xf16, {order = #NHWC}>, tensor<4x128x3x6xf16, {order = #NHWC}> -> tensor<7x128x3x6xf16, {order = #NHWC}>
    return %2 : tensor<7x128x3x6xf16, {order = #NHWC}>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[INPUT0]]
    // CHECK-SAME{LITERAL}:          [1, 0, 0, 0] [3, 128, 3, 6] : tensor<4x128x3x6xf16, {order = #NHWC}> to tensor<3x128x3x6xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SLICE]], [[INPUT0]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<3x128x3x6xf16, {order = #NHWC}>, tensor<4x128x3x6xf16, {order = #NHWC}> -> tensor<7x128x3x6xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]] : tensor<7x128x3x6xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseSliceConcatWithPermuteCast
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<4x128x1x1xf16, {order = #NHWC}>
func.func @FuseSliceConcatWithPermuteCast(%arg0: tensor<4x128x1x1xf16, {order = #NHWC}>) -> tensor<1x128x2x1xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [2, 0, 0, 0] [1, 128, 1, 1] : tensor<4x128x1x1xf16, {order = #NHWC}> to tensor<1x128x1x1xf16, {order = #NHWC}>
    %1 = IE.Slice %arg0 [3, 0, 0, 0] [1, 128, 1, 1] : tensor<4x128x1x1xf16, {order = #NHWC}> to tensor<1x128x1x1xf16, {order = #NHWC}>

    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x128x1x1xf16, {order = #NHWC}>, tensor<1x128x1x1xf16, {order = #NHWC}> -> tensor<1x128x2x1xf16, {order = #NHWC}>
    return %2 : tensor<1x128x2x1xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #NHWC, mem_perm = #map} : tensor<4x128x1x1xf16, {order = #NHWC}> -> tensor<1x128x4x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[PERMUTECAST]]
    // CHECK-SAME{LITERAL}:             [0, 0, 2, 0] [1, 128, 2, 1] : tensor<1x128x4x1xf16, {order = #NHWC}> to tensor<1x128x2x1xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE]] : tensor<1x128x2x1xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFuseSliceIsNotContinuous
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<4x128x3x6xf16, {order = #NHWC}>
func.func @NotFuseSliceIsNotContinuous(%arg0: tensor<4x128x3x6xf16, {order = #NHWC}>) -> tensor<3x128x3x6xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [2, 128, 3, 6] : tensor<4x128x3x6xf16, {order = #NHWC}> to tensor<2x128x3x6xf16, {order = #NHWC}>
    %1 = IE.Slice %arg0 [3, 0, 0, 0] [1, 128, 3, 6] : tensor<4x128x3x6xf16, {order = #NHWC}> to tensor<1x128x3x6xf16, {order = #NHWC}>

    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<2x128x3x6xf16, {order = #NHWC}>, tensor<1x128x3x6xf16, {order = #NHWC}> -> tensor<3x128x3x6xf16, {order = #NHWC}>
    return %2 : tensor<3x128x3x6xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[INPUT0]]
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[INPUT0]]
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SLICE_0]], [[SLICE_1]])
    // CHECK:       return  [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFuseConcatIsNotContinuous
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<4x128x3x6xf16, {order = #NHWC}>
func.func @NotFuseConcatIsNotContinuous(%arg0: tensor<4x128x3x6xf16, {order = #NHWC}>) -> tensor<3x128x3x6xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [1, 0, 0, 0] [2, 128, 3, 6] : tensor<4x128x3x6xf16, {order = #NHWC}> to tensor<2x128x3x6xf16, {order = #NHWC}>
    %1 = IE.Slice %arg0 [3, 0, 0, 0] [1, 128, 3, 6] : tensor<4x128x3x6xf16, {order = #NHWC}> to tensor<1x128x3x6xf16, {order = #NHWC}>

    %2 = IE.Concat(%1, %0) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x128x3x6xf16, {order = #NHWC}>, tensor<2x128x3x6xf16, {order = #NHWC}> -> tensor<3x128x3x6xf16, {order = #NHWC}>
    return %2 : tensor<3x128x3x6xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[INPUT0]]
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[INPUT0]]
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SLICE_1]], [[SLICE_0]])
    // CHECK:       return  [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFuseCouldNotInsertPermuteCast
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<4x128x3x6xf16, {order = #NHWC}>
func.func @NotFuseCouldNotInsertPermuteCast(%arg0: tensor<4x128x3x6xf16, {order = #NHWC}>) -> tensor<1x128x6x6xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [2, 0, 0, 0] [1, 128, 3, 6] : tensor<4x128x3x6xf16, {order = #NHWC}> to tensor<1x128x3x6xf16, {order = #NHWC}>
    %1 = IE.Slice %arg0 [3, 0, 0, 0] [1, 128, 3, 6] : tensor<4x128x3x6xf16, {order = #NHWC}> to tensor<1x128x3x6xf16, {order = #NHWC}>

    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x128x3x6xf16, {order = #NHWC}>, tensor<1x128x3x6xf16, {order = #NHWC}> -> tensor<1x128x6x6xf16, {order = #NHWC}>
    return %2 : tensor<1x128x6x6xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[INPUT0]]
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[INPUT0]]
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SLICE_0]], [[SLICE_1]])
    // CHECK:       return  [[CONCAT]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFuseMultiDimSlice
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<4x128x3x6xf16, {order = #NHWC}>
func.func @NotFuseMultiDimSlice(%arg0: tensor<4x128x3x6xf16, {order = #NHWC}>) -> tensor<3x128x2x6xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [2, 128, 2, 6] : tensor<4x128x3x6xf16, {order = #NHWC}> to tensor<2x128x2x6xf16, {order = #NHWC}>
    %1 = IE.Slice %arg0 [3, 0, 0, 0] [1, 128, 2, 6] : tensor<4x128x3x6xf16, {order = #NHWC}> to tensor<1x128x2x6xf16, {order = #NHWC}>

    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<2x128x2x6xf16, {order = #NHWC}>, tensor<1x128x2x6xf16, {order = #NHWC}> -> tensor<3x128x2x6xf16, {order = #NHWC}>
    return %2 : tensor<3x128x2x6xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[INPUT0]]
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[INPUT0]]
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SLICE_0]], [[SLICE_1]])
    // CHECK:       return  [[CONCAT]]
}

// -----

// CHECK-LABEL: @OptConcatAndSliceWithAffineReshapeAndSplitConcatAxis
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x1x1x1xf16>
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x1x1x2xf16>
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x1x1x3xf16>
func.func @OptConcatAndSliceWithAffineReshapeAndSplitConcatAxis(
                %arg0: tensor<1x1x1x1xf16>, %arg1: tensor<1x1x1x2xf16>, %arg2: tensor<1x1x1x3xf16>) -> tensor<1x2x1x1xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {
            static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 3]]
        } : tensor<1x1x1x1xf16>, tensor<1x1x1x2xf16>, tensor<1x1x1x3xf16> -> tensor<1x1x1x6xf16>
    %1 = IE.AffineReshape(%0) {
            dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 6, 1, 1]
        } : tensor<1x1x1x6xf16> -> tensor<1x6x1x1xf16>
    %2 = IE.Slice %1 [0, 4, 0, 0] [1, 2, 1, 1] : tensor<1x6x1x1xf16> to tensor<1x2x1x1xf16>

    return %2 : tensor<1x2x1x1xf16>

    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[INPUT2]]) {
    // CHECK-SAME{LITERAL}:       dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 3, 1, 1]} : tensor<1x1x1x3xf16> -> tensor<1x3x1x1xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[RESHAPE]] [0, 1, 0, 0] [1, 2, 1, 1] : tensor<1x3x1x1xf16> to tensor<1x2x1x1xf16>

    // CHECK:       return [[SLICE]] : tensor<1x2x1x1xf16>
}

// -----

// CHECK-LABEL: func.func @Opt1DConcatAndSliceWithAffineReshapeAndSplitConcatAxis
// CHECK-SAME:      ([[INPUT0:%.+]]: tensor<6xf16>, [[INPUT1:%.+]]: tensor<6xf16>, [[INPUT2:%.+]]: tensor<6xf16>)
func.func @Opt1DConcatAndSliceWithAffineReshapeAndSplitConcatAxis(
            %arg0: tensor<6xf16>, %arg1: tensor<6xf16>, %arg2: tensor<6xf16>) -> tensor<1x1x2x3xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {
            static_offsets = [[0], [6], [12]]
        } : tensor<6xf16>, tensor<6xf16>, tensor<6xf16> -> tensor<18xf16>
    %1 = IE.AffineReshape(%0) {
            dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 3, 2, 3]
        } : tensor<18xf16> -> tensor<1x3x2x3xf16>
    %2 = IE.Slice %1 [0, 1, 0, 0] [1, 1, 2, 3] : tensor<1x3x2x3xf16> to tensor<1x1x2x3xf16>

    return %2 : tensor<1x1x2x3xf16>

    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[INPUT1]]) {
    // CHECK-SAME{LITERAL}:       dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 2, 3]} : tensor<6xf16> -> tensor<1x1x2x3xf16>

    // CHECK:       return [[RESHAPE]] : tensor<1x1x2x3xf16>
}

// -----

// CHECK-LABEL: @NotOptConcatAndSliceWithAffineReshapeAndSplitConcatAxis
// CHECK-SAME:      ([[INPUT0:%.+]]: tensor<2x1x3x4xf16>, [[INPUT1:%.+]]: tensor<2x1x3x4xf16>, [[INPUT2:%.+]]: tensor<2x1x3x4xf16>)
func.func @NotOptConcatAndSliceWithAffineReshapeAndSplitConcatAxis(
                %arg0: tensor<2x1x3x4xf16>, %arg1: tensor<2x1x3x4xf16>, %arg2: tensor<2x1x3x4xf16>) -> tensor<1x3x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {
            static_offsets = [[0, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]]
        } : tensor<2x1x3x4xf16>, tensor<2x1x3x4xf16>, tensor<2x1x3x4xf16> -> tensor<6x1x3x4xf16>
    %1 = IE.AffineReshape(%0) {
            dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [2, 3, 3, 4]
        } : tensor<6x1x3x4xf16> -> tensor<2x3x3x4xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 3, 3, 4] : tensor<2x3x3x4xf16> to tensor<1x3x3x4xf16>

    return %2 : tensor<1x3x3x4xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]], [[INPUT2]]) {
    // CHECK-SAME{LITERAL}:       static_offsets = [[0, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]]} : tensor<2x1x3x4xf16>, tensor<2x1x3x4xf16>, tensor<2x1x3x4xf16> -> tensor<6x1x3x4xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]]) {
    // CHECK-SAME{LITERAL}:       dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [2, 3, 3, 4]} : tensor<6x1x3x4xf16> -> tensor<2x3x3x4xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[RESHAPE]] [0, 0, 0, 0] [1, 3, 3, 4] : tensor<2x3x3x4xf16> to tensor<1x3x3x4xf16>

    // CHECK:       return [[SLICE]] : tensor<1x3x3x4xf16>
}

// -----

// CHECK-LABEL: @NotOptConcatAndSliceWithAffineReshapeAndMergeConcatAxis
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x2x3x4xf16>
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x2x4x4xf16>
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x2x5x4xf16>
func.func @NotOptConcatAndSliceWithAffineReshapeAndMergeConcatAxis(
                %arg0: tensor<1x2x3x4xf16>, %arg1: tensor<1x2x4x4xf16>, %arg2: tensor<1x2x5x4xf16>) -> tensor<1x6x1x4xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {
            static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 7, 0]]
        } : tensor<1x2x3x4xf16>, tensor<1x2x4x4xf16>, tensor<1x2x5x4xf16> -> tensor<1x2x12x4xf16>
    %1 = IE.AffineReshape(%0) {
            dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 24, 1, 4]
        } : tensor<1x2x12x4xf16> -> tensor<1x24x1x4xf16>
    %2 = IE.Slice %1 [0, 7, 0, 0] [1, 6, 1, 4] : tensor<1x24x1x4xf16> to tensor<1x6x1x4xf16>

    return %2 : tensor<1x6x1x4xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]], [[INPUT2]]) {
    // CHECK-SAME{LITERAL}:       static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 7, 0]]} : tensor<1x2x3x4xf16>, tensor<1x2x4x4xf16>, tensor<1x2x5x4xf16> -> tensor<1x2x12x4xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]]) {
    // CHECK-SAME{LITERAL}:       dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 24, 1, 4]} : tensor<1x2x12x4xf16> -> tensor<1x24x1x4xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[RESHAPE]] [0, 7, 0, 0] [1, 6, 1, 4] : tensor<1x24x1x4xf16> to tensor<1x6x1x4xf16>

    // CHECK:       return [[SLICE]] : tensor<1x6x1x4xf16>
}
