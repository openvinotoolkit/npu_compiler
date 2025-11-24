//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --run-adjust-for-vpu-rewriters="rewriter=merge-tile-with-slice-set" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


// -----

// CHECK-LABEL: @MergeTileOnFirstDim
// CHECK-SAME:      [[INPUT:%.*]]: tensor<1x2x16x32xf16>
func.func @MergeTileOnFirstDim(%arg0: tensor<1x2x16x32xf16>) -> (tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [2, 1, 1, 1]} : tensor<1x2x16x32xf16> -> tensor<2x2x16x32xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 4, 16, 32]} : tensor<2x2x16x32xf16> -> tensor<1x4x16x32xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 16, 32] : tensor<1x4x16x32xf16> to tensor<1x1x16x32xf16>
    %3 = IE.Slice %1 [0, 1, 0, 0] [1, 1, 16, 32] : tensor<1x4x16x32xf16> to tensor<1x1x16x32xf16>
    %4 = IE.Slice %1 [0, 2, 0, 0] [1, 1, 16, 32] : tensor<1x4x16x32xf16> to tensor<1x1x16x32xf16>
    %5 = IE.Slice %1 [0, 3, 0, 0] [1, 1, 16, 32] : tensor<1x4x16x32xf16> to tensor<1x1x16x32xf16>

    return  %2, %3, %4, %5 : tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>

    // CHECK:   [[SLICE0:%.*]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 16, 32] : tensor<1x2x16x32xf16> to tensor<1x1x16x32xf16>
    // CHECK:   [[SLICE1:%.*]] = IE.Slice [[INPUT]] [0, 1, 0, 0] [1, 1, 16, 32] : tensor<1x2x16x32xf16> to tensor<1x1x16x32xf16>
    // CHECK:   return      [[SLICE0]], [[SLICE1]], [[SLICE0]], [[SLICE1]]
}


// -----

// CHECK-LABEL: @MergeTileOnSecondDim
// CHECK-SAME:      [[INPUT:%.*]]: tensor<2x1x16x32xf16>
func.func @MergeTileOnSecondDim(%arg0: tensor<2x1x16x32xf16>) -> (tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 1, 1]} : tensor<2x1x16x32xf16> -> tensor<2x2x16x32xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 4, 16, 32]} : tensor<2x2x16x32xf16> -> tensor<1x4x16x32xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 16, 32] : tensor<1x4x16x32xf16> to tensor<1x1x16x32xf16>
    %3 = IE.Slice %1 [0, 1, 0, 0] [1, 1, 16, 32] : tensor<1x4x16x32xf16> to tensor<1x1x16x32xf16>
    %4 = IE.Slice %1 [0, 2, 0, 0] [1, 1, 16, 32] : tensor<1x4x16x32xf16> to tensor<1x1x16x32xf16>
    %5 = IE.Slice %1 [0, 3, 0, 0] [1, 1, 16, 32] : tensor<1x4x16x32xf16> to tensor<1x1x16x32xf16>

    return  %2, %3, %4, %5 : tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>

    // CHECK:   [[SLICE0:%.*]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 16, 32] : tensor<2x1x16x32xf16> to tensor<1x1x16x32xf16>
    // CHECK:   [[SLICE1:%.*]] = IE.Slice [[INPUT]] [1, 0, 0, 0] [1, 1, 16, 32] : tensor<2x1x16x32xf16> to tensor<1x1x16x32xf16>
    // CHECK:   return      [[SLICE0]], [[SLICE0]], [[SLICE1]], [[SLICE1]]
}


// -----

// CHECK-LABEL: @NotMergeTileHasMoreUser
// CHECK-SAME:      [[INPUT:%.*]]: tensor<2x1x16x32xf16>
func.func @NotMergeTileHasMoreUser(%arg0: tensor<2x1x16x32xf16>) -> (tensor<2x2x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 1, 1]} : tensor<2x1x16x32xf16> -> tensor<2x2x16x32xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 4, 16, 32]} : tensor<2x2x16x32xf16> -> tensor<1x4x16x32xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 16, 32] : tensor<1x4x16x32xf16> to tensor<1x1x16x32xf16>
    %3 = IE.Slice %1 [0, 1, 0, 0] [1, 1, 16, 32] : tensor<1x4x16x32xf16> to tensor<1x1x16x32xf16>

    return %0, %2, %3 : tensor<2x2x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>

    // CHECK:    [[TILE:%.*]] = IE.Tile([[INPUT]])
    // CHECK:    [[RESHAPE:%.*]] = IE.Reshape([[TILE]])
    // CHECK:    [[SLICE0:%.*]] = IE.Slice [[RESHAPE]]
    // CHECK:    [[SLICE1:%.*]] = IE.Slice [[RESHAPE]]
    // CHECK:    return      [[TILE]], [[SLICE0]], [[SLICE1]]
}


// -----

// CHECK-LABEL: @NotMergeTileOnTwoDim
// CHECK-SAME:      [[INPUT:%.*]]: tensor<2x1x16x32xf16>
func.func @NotMergeTileOnTwoDim(%arg0: tensor<2x1x16x32xf16>) -> (tensor<1x1x32x32xf16>, tensor<1x1x32x32xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 2, 1]} : tensor<2x1x16x32xf16> -> tensor<2x2x32x32xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 4, 32, 32]} : tensor<2x2x32x32xf16> -> tensor<1x4x32x32xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 32, 32] : tensor<1x4x32x32xf16> to tensor<1x1x32x32xf16>
    %3 = IE.Slice %1 [0, 1, 0, 0] [1, 1, 32, 32] : tensor<1x4x32x32xf16> to tensor<1x1x32x32xf16>

    return  %2, %3 : tensor<1x1x32x32xf16>, tensor<1x1x32x32xf16>

    // CHECK:    [[TILE:%.*]] = IE.Tile([[INPUT]])
    // CHECK:    [[RESHAPE:%.*]] = IE.Reshape([[TILE]])
    // CHECK:    [[SLICE0:%.*]] = IE.Slice [[RESHAPE]]
    // CHECK:    [[SLICE1:%.*]] = IE.Slice [[RESHAPE]]
    // CHECK:    return      [[SLICE0]], [[SLICE1]]
}


// -----

// CHECK-LABEL: @NotMergeReshapeDimNotMeetRequirement
// CHECK-SAME:      [[INPUT:%.*]]: tensor<2x1x16x32xf16>
func.func @NotMergeReshapeDimNotMeetRequirement(%arg0: tensor<2x1x16x32xf16>) -> (tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 1, 1]} : tensor<2x1x16x32xf16> -> tensor<2x2x16x32xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 8, 16, 16]} : tensor<2x2x16x32xf16> -> tensor<1x8x16x16xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 16, 16] : tensor<1x8x16x16xf16> to tensor<1x1x16x16xf16>
    %3 = IE.Slice %1 [0, 1, 0, 0] [1, 1, 16, 16] : tensor<1x8x16x16xf16> to tensor<1x1x16x16xf16>

    return  %2, %3 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>

    // CHECK:    [[TILE:%.*]] = IE.Tile([[INPUT]])
    // CHECK:    [[RESHAPE:%.*]] = IE.Reshape([[TILE]])
    // CHECK:    [[SLICE0:%.*]] = IE.Slice [[RESHAPE]]
    // CHECK:    [[SLICE1:%.*]] = IE.Slice [[RESHAPE]]
    // CHECK:    return      [[SLICE0]], [[SLICE1]]
}

// -----

// CHECK-LABEL: @NotMergeSliceSizeMoreThanOne
// CHECK-SAME:      [[INPUT:%.*]]: tensor<2x1x16x32xf16>
func.func @NotMergeSliceSizeMoreThanOne(%arg0: tensor<2x1x16x32xf16>) -> (tensor<1x2x16x32xf16>, tensor<1x2x16x32xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 1, 1]} : tensor<2x1x16x32xf16> -> tensor<2x2x16x32xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 4, 16, 32]} : tensor<2x2x16x32xf16> -> tensor<1x4x16x32xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 2, 16, 32] : tensor<1x4x16x32xf16> to tensor<1x2x16x32xf16>
    %3 = IE.Slice %1 [0, 1, 0, 0] [1, 2, 16, 32] : tensor<1x4x16x32xf16> to tensor<1x2x16x32xf16>

    return  %2, %3 : tensor<1x2x16x32xf16>, tensor<1x2x16x32xf16>

    // CHECK:    [[TILE:%.*]] = IE.Tile([[INPUT]])
    // CHECK:    [[RESHAPE:%.*]] = IE.Reshape([[TILE]])
    // CHECK:    [[SLICE0:%.*]] = IE.Slice [[RESHAPE]]
    // CHECK:    [[SLICE1:%.*]] = IE.Slice [[RESHAPE]]
    // CHECK:    return      [[SLICE0]], [[SLICE1]]
}


// -----

// CHECK-LABEL: @NotMergeCouldNotSlice
// CHECK-SAME:      [[INPUT:%.*]]: tensor<2x1x16x32xf16>
func.func @NotMergeCouldNotSlice(%arg0: tensor<2x1x16x32xf16>) -> (tensor<1x1x8x32xf16>, tensor<1x1x8x32xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 1, 1]} : tensor<2x1x16x32xf16> -> tensor<2x2x16x32xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 4, 16, 32]} : tensor<2x2x16x32xf16> -> tensor<1x4x16x32xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1, 8, 32] : tensor<1x4x16x32xf16> to tensor<1x1x8x32xf16>
    %3 = IE.Slice %1 [0, 1, 0, 0] [1, 1, 8, 32] : tensor<1x4x16x32xf16> to tensor<1x1x8x32xf16>

    return  %2, %3 : tensor<1x1x8x32xf16>, tensor<1x1x8x32xf16>

    // CHECK:    [[TILE:%.*]] = IE.Tile([[INPUT]])
    // CHECK:    [[RESHAPE:%.*]] = IE.Reshape([[TILE]])
    // CHECK:    [[SLICE0:%.*]] = IE.Slice [[RESHAPE]]
    // CHECK:    [[SLICE1:%.*]] = IE.Slice [[RESHAPE]]
    // CHECK:    return      [[SLICE0]], [[SLICE1]]
}


// -----

// CHECK-LABEL: @MergeTileSliceMoreThanOne
// CHECK-SAME:      [[INPUT:%.*]]: tensor<1x3x16x32xf16>
func.func @MergeTileSliceMoreThanOne(%arg0: tensor<1x3x16x32xf16>) -> (tensor<1x2x16x32xf16>, tensor<1x2x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [2, 1, 1, 1]} : tensor<1x3x16x32xf16> -> tensor<2x3x16x32xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 6, 16, 32]} : tensor<2x3x16x32xf16> -> tensor<1x6x16x32xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 2, 16, 32] : tensor<1x6x16x32xf16> to tensor<1x2x16x32xf16>
    %3 = IE.Slice %1 [0, 1, 0, 0] [1, 2, 16, 32] : tensor<1x6x16x32xf16> to tensor<1x2x16x32xf16>
    %4 = IE.Slice %1 [0, 2, 0, 0] [1, 1, 16, 32] : tensor<1x6x16x32xf16> to tensor<1x1x16x32xf16>
    %5 = IE.Slice %1 [0, 3, 0, 0] [1, 1, 16, 32] : tensor<1x6x16x32xf16> to tensor<1x1x16x32xf16>

    return  %2, %3, %4, %5 : tensor<1x2x16x32xf16>, tensor<1x2x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>

    // CHECK:   [[SLICE0:%.*]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 2, 16, 32] : tensor<1x3x16x32xf16> to tensor<1x2x16x32xf16>
    // CHECK:   [[SLICE1:%.*]] = IE.Slice [[INPUT]] [0, 1, 0, 0] [1, 2, 16, 32] : tensor<1x3x16x32xf16> to tensor<1x2x16x32xf16>
    // CHECK:   [[SLICE2:%.*]] = IE.Slice [[INPUT]] [0, 2, 0, 0] [1, 1, 16, 32] : tensor<1x3x16x32xf16> to tensor<1x1x16x32xf16>
    // CHECK:   [[SLICE3:%.*]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 16, 32] : tensor<1x3x16x32xf16> to tensor<1x1x16x32xf16>

    // CHECK:   return      [[SLICE0]], [[SLICE1]], [[SLICE2]], [[SLICE3]]
}


// -----

// CHECK-LABEL: @MergeTileOnFirstDimWithTranspose
// CHECK-SAME:      [[INPUT:%.*]]: tensor<1x2x16x32xf16>
func.func @MergeTileOnFirstDimWithTranspose(%arg0: tensor<1x2x16x32xf16>) -> (tensor<1x1x32x16xf16>, tensor<1x1x32x16xf16>, tensor<1x1x32x16xf16>, tensor<1x1x32x16xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [2, 1, 1, 1]} : tensor<1x2x16x32xf16> -> tensor<2x2x16x32xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 4, 16, 32]} : tensor<2x2x16x32xf16> -> tensor<1x4x16x32xf16>
    %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x4x16x32xf16> -> tensor<1x4x32x16xf16>
    %3 = IE.Slice %2 [0, 0, 0, 0] [1, 1, 32, 16] : tensor<1x4x32x16xf16> to tensor<1x1x32x16xf16>
    %4 = IE.Slice %2 [0, 1, 0, 0] [1, 1, 32, 16] : tensor<1x4x32x16xf16> to tensor<1x1x32x16xf16>
    %5 = IE.Slice %2 [0, 2, 0, 0] [1, 1, 32, 16] : tensor<1x4x32x16xf16> to tensor<1x1x32x16xf16>
    %6 = IE.Slice %2 [0, 3, 0, 0] [1, 1, 32, 16] : tensor<1x4x32x16xf16> to tensor<1x1x32x16xf16>

    return  %3, %4, %5, %6 : tensor<1x1x32x16xf16>, tensor<1x1x32x16xf16>, tensor<1x1x32x16xf16>, tensor<1x1x32x16xf16>

    // CHECK:   [[TRANSPOSE:%.*]] = IE.Transpose([[INPUT]]) {order_value = #NCWH} : tensor<1x2x16x32xf16> -> tensor<1x2x32x16xf16>
    // CHECK:   [[SLICE0:%.*]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [1, 1, 32, 16] : tensor<1x2x32x16xf16> to tensor<1x1x32x16xf16>
    // CHECK:   [[SLICE1:%.*]] = IE.Slice [[TRANSPOSE]] [0, 1, 0, 0] [1, 1, 32, 16] : tensor<1x2x32x16xf16> to tensor<1x1x32x16xf16>

    // CHECK:   return      [[SLICE0]], [[SLICE1]], [[SLICE0]], [[SLICE1]]
}

// -----

// CHECK-LABEL: @NotMergeWithInvalidTranspose
// CHECK-SAME:      [[INPUT:%.*]]: tensor<1x2x16x32xf16>
func.func @NotMergeWithInvalidTranspose(%arg0: tensor<1x2x16x32xf16>) -> (tensor<1x32x1x16xf16>, tensor<1x32x1x16xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [2, 1, 1, 1]} : tensor<1x2x16x32xf16> -> tensor<2x2x16x32xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 4, 16, 32]} : tensor<2x2x16x32xf16> -> tensor<1x4x16x32xf16>
    %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x4x16x32xf16> -> tensor<1x32x4x16xf16>
    %3 = IE.Slice %2 [0, 0, 0, 0] [1, 32, 1, 16] : tensor<1x32x4x16xf16> to tensor<1x32x1x16xf16>
    %4 = IE.Slice %2 [0, 0, 1, 0] [1, 32, 1, 16] : tensor<1x32x4x16xf16> to tensor<1x32x1x16xf16>


    return  %3, %4 : tensor<1x32x1x16xf16>, tensor<1x32x1x16xf16>

    // CHECK:    [[TILE:%.*]] = IE.Tile([[INPUT]])
    // CHECK:    [[RESHAPE:%.*]] = IE.Reshape([[TILE]])
    // CHECK:    [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE]])
    // CHECK:    [[SLICE0:%.*]] = IE.Slice [[TRANSPOSE]]
    // CHECK:    [[SLICE1:%.*]] = IE.Slice [[TRANSPOSE]]
    // CHECK:    return      [[SLICE0]], [[SLICE1]]
}

// -----

// CHECK-LABEL: @MergeTileWithSlice
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x16x32xf16>
func.func @MergeTileWithSlice(%arg0: tensor<1x1x16x32xf16>) -> (tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 1, 1]} : tensor<1x1x16x32xf16> -> tensor<1x2x16x32xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 1, 16, 32] : tensor<1x2x16x32xf16> to tensor<1x1x16x32xf16>
    %2 = IE.Slice %0 [0, 1, 0, 0] [1, 1, 16, 32] : tensor<1x2x16x32xf16> to tensor<1x1x16x32xf16>

    return  %1, %2 : tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>

    // CHECK-NOT:    IE.Tile
    // CHECK-NOT:    IE.Slice
    // CHECK:    return      [[INPUT]], [[INPUT]]
}

// -----

// CHECK-LABEL: @NotMergeTileWithSliceDueToDifferentRepeats
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x16x32xf16>
func.func @NotMergeTileWithSliceDueToDifferentRepeats(%arg0: tensor<1x1x16x32xf16>) -> (tensor<1x1x16x32xf16>, tensor<1x2x16x16xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 1, 1]} : tensor<1x1x16x32xf16> -> tensor<1x2x16x32xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 1, 16, 32] : tensor<1x2x16x32xf16> to tensor<1x1x16x32xf16>
    %2 = IE.Slice %0 [0, 0, 0, 0] [1, 2, 16, 16] : tensor<1x2x16x32xf16> to tensor<1x2x16x16xf16>

    return  %1, %2 : tensor<1x1x16x32xf16>, tensor<1x2x16x16xf16>

    // CHECK:    [[TILE:%.+]] = IE.Tile([[INPUT]])
    // CHECK:    [[SLICE_1:%.+]] = IE.Slice [[TILE]]
    // CHECK:    [[SLICE_2:%.+]] = IE.Slice [[TILE]]

    // CHECK:    return      [[SLICE_1]], [[SLICE_2]]
}

// -----

// CHECK-LABEL: @NotMergeTileWithSliceWhenSliceNotInOriginalInput
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x16x32xf16>
func.func @NotMergeTileWithSliceWhenSliceNotInOriginalInput(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 1, 1]} : tensor<1x2x16x32xf16> -> tensor<1x4x16x32xf16>
    %1 = IE.Slice %0 [0, 1, 0, 0] [1, 2, 16, 32] : tensor<1x4x16x32xf16> to tensor<1x2x16x32xf16>

    return  %1 : tensor<1x2x16x32xf16>

    // CHECK:    [[TILE:%.+]] = IE.Tile([[INPUT]])
    // CHECK:    [[SLICE_1:%.+]] = IE.Slice [[TILE]]

    // CHECK:    return      [[SLICE_1]]
}
