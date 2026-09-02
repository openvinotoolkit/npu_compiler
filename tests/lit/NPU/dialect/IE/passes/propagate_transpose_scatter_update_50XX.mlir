//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --propagate-transpose %s | FileCheck %s
// REQUIRES: platform-NPU5010

// These tests verify that MoveThroughScatterUpdate fires when newAxisValue != 0.
// The DMA path (LayerWithDmaInterface) is required for non-zero axis, which is
// only registered for NPU50XX.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MoveTransposeThroughScatterUpdate
// CHECK-SAME:      ([[INPUT:%arg[0-9]]]: tensor<1x4x8x16xf16>, [[INDICES:%arg[0-9]]]: tensor<2xsi32>, [[UPDATES:%arg[0-9]]]: tensor<1x2x16x4xf16>)
func.func @MoveTransposeThroughScatterUpdate(%arg0: tensor<1x4x8x16xf16>, %arg1: tensor<2xsi32>, %arg2: tensor<1x2x16x4xf16>) -> tensor<1x4x8x16xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x4x8x16xf16> -> tensor<1x8x16x4xf16>
    %1 = IE.ScatterUpdate(%0, %arg1, %arg2) {axis_value = 1 : i64} : tensor<1x8x16x4xf16>, tensor<2xsi32>, tensor<1x2x16x4xf16> -> tensor<1x8x16x4xf16>
    %2 = IE.Transpose(%1) {order_value = #NWCH} : tensor<1x8x16x4xf16> -> tensor<1x4x8x16xf16>
    return %2 : tensor<1x4x8x16xf16>

    // CHECK:       [[UPDATES_T:%.+]] = IE.Transpose([[UPDATES]]) {order_value = #NWCH} : tensor<1x2x16x4xf16> -> tensor<1x4x2x16xf16>
    // CHECK:       [[SCATTER:%.+]] = IE.ScatterUpdate([[INPUT]], [[INDICES]], [[UPDATES_T]]) {axis_value = 2 : i64} : tensor<1x4x8x16xf16>, tensor<2xsi32>, tensor<1x4x2x16xf16> -> tensor<1x4x8x16xf16>
    // CHECK:       return [[SCATTER]] : tensor<1x4x8x16xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MoveTransposeThroughScatterUpdateHigherIndicesRank
// CHECK-SAME:      ([[INPUT:%arg[0-9]]]: tensor<1x4x8x16xf16>, [[INDICES:%arg[0-9]]]: tensor<2x3xsi32>, [[UPDATES:%arg[0-9]]]: tensor<1x2x3x16x4xf16>)
func.func @MoveTransposeThroughScatterUpdateHigherIndicesRank(%arg0: tensor<1x4x8x16xf16>, %arg1: tensor<2x3xsi32>, %arg2: tensor<1x2x3x16x4xf16>) -> tensor<1x4x8x16xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x4x8x16xf16> -> tensor<1x8x16x4xf16>
    %1 = IE.ScatterUpdate(%0, %arg1, %arg2) {axis_value = 1 : i64} : tensor<1x8x16x4xf16>, tensor<2x3xsi32>, tensor<1x2x3x16x4xf16> -> tensor<1x8x16x4xf16>
    %2 = IE.Transpose(%1) {order_value = #NWCH} : tensor<1x8x16x4xf16> -> tensor<1x4x8x16xf16>
    return %2 : tensor<1x4x8x16xf16>

    // CHECK:       [[UPDATES_T:%.+]] = IE.Transpose([[UPDATES]]) {order_value = #map} : tensor<1x2x3x16x4xf16> -> tensor<1x4x2x3x16xf16>
    // CHECK:       [[SCATTER:%.+]] = IE.ScatterUpdate([[INPUT]], [[INDICES]], [[UPDATES_T]]) {axis_value = 2 : i64} : tensor<1x4x8x16xf16>, tensor<2x3xsi32>, tensor<1x4x2x3x16xf16> -> tensor<1x4x8x16xf16>
    // CHECK:       return [[SCATTER]] : tensor<1x4x8x16xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MoveTransposeThroughScatterUpdateNegativeAxis
// CHECK-SAME:      ([[INPUT:%arg[0-9]]]: tensor<1x4x8x16xf16>, [[INDICES:%arg[0-9]]]: tensor<2xsi32>, [[UPDATES:%arg[0-9]]]: tensor<1x2x16x4xf16>)
func.func @MoveTransposeThroughScatterUpdateNegativeAxis(%arg0: tensor<1x4x8x16xf16>, %arg1: tensor<2xsi32>, %arg2: tensor<1x2x16x4xf16>) -> tensor<1x4x8x16xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x4x8x16xf16> -> tensor<1x8x16x4xf16>
    %1 = IE.ScatterUpdate(%0, %arg1, %arg2) {axis_value = -3 : i64} : tensor<1x8x16x4xf16>, tensor<2xsi32>, tensor<1x2x16x4xf16> -> tensor<1x8x16x4xf16>
    %2 = IE.Transpose(%1) {order_value = #NWCH} : tensor<1x8x16x4xf16> -> tensor<1x4x8x16xf16>
    return %2 : tensor<1x4x8x16xf16>

    // axis_value = -3 normalizes to 1 (rank 4); same result as positive-axis test.
    // CHECK:       [[UPDATES_T:%.+]] = IE.Transpose([[UPDATES]]) {order_value = #NWCH} : tensor<1x2x16x4xf16> -> tensor<1x4x2x16xf16>
    // CHECK:       [[SCATTER:%.+]] = IE.ScatterUpdate([[INPUT]], [[INDICES]], [[UPDATES_T]]) {axis_value = 2 : i64} : tensor<1x4x8x16xf16>, tensor<2xsi32>, tensor<1x4x2x16xf16> -> tensor<1x4x8x16xf16>
    // CHECK:       return [[SCATTER]] : tensor<1x4x8x16xf16>
}
