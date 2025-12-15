//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-gather-elements-to-gather %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertGatherElementsOnHAndTileToGather
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x1x5376x80xf16>, [[INPUT1:%.+]]: tensor<1x1x300x1xsi32>)
func.func @ConvertGatherElementsOnHAndTileToGather(%arg0: tensor<1x1x5376x80xf16>,
                                                   %arg1: tensor<1x1x300x1xsi32>)
                                                   -> (tensor<1x1x300x80xf16>) {
    %0 = IE.Tile(%arg1) {repeats_values = [1, 1, 1, 80]} : tensor<1x1x300x1xsi32> -> tensor<1x1x300x80xsi32>
    %1 = IE.GatherElements(%arg0, %0) {axis = 2 : i64} : tensor<1x1x5376x80xf16>, tensor<1x1x300x80xsi32> -> tensor<1x1x300x80xf16>

    return %1 : tensor<1x1x300x80xf16>

    // CHECK-NOT:    IE.Tile
    // CHECK-NOT:    IE.GatherElements

    // CHECK:        [[SQUEEZE:%.+]] = IE.Squeeze([[INPUT1]]) {axes_value = [0, 1, 3]} : tensor<1x1x300x1xsi32> -> tensor<300xsi32>
    // CHECK:        [[GATHER:%.+]] = IE.Gather([[INPUT0]], [[SQUEEZE]]) {axis_value = 2 : i64, batch_dims = 0 : i64} : tensor<1x1x5376x80xf16>, tensor<300xsi32> -> tensor<1x1x300x80xf16>

    // CHECK:        return [[GATHER]] : tensor<1x1x300x80xf16>
}

// -----

// CHECK-LABEL: @ConvertGatherElementsOnWAndTileToGather
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x1x300x200xf16>, [[INPUT1:%.+]]: tensor<1x1x1x80xsi32>)
func.func @ConvertGatherElementsOnWAndTileToGather(%arg0: tensor<1x1x300x200xf16>,
                                                   %arg1: tensor<1x1x1x80xsi32>)
                                                   -> (tensor<1x1x300x80xf16>) {
    %0 = IE.Tile(%arg1) {repeats_values = [1, 1, 300, 1]} : tensor<1x1x1x80xsi32> -> tensor<1x1x300x80xsi32>
    %1 = IE.GatherElements(%arg0, %0) {axis = 3 : i64} : tensor<1x1x300x200xf16>, tensor<1x1x300x80xsi32> -> tensor<1x1x300x80xf16>

    return %1 : tensor<1x1x300x80xf16>

    // CHECK-NOT:    IE.Tile
    // CHECK-NOT:    IE.GatherElements

    // CHECK:        [[SQUEEZE:%.+]] = IE.Squeeze([[INPUT1]]) {axes_value = [0, 1, 2]} : tensor<1x1x1x80xsi32> -> tensor<80xsi32>
    // CHECK:        [[GATHER:%.+]] = IE.Gather([[INPUT0]], [[SQUEEZE]]) {axis_value = 3 : i64, batch_dims = 0 : i64} : tensor<1x1x300x200xf16>, tensor<80xsi32> -> tensor<1x1x300x80xf16>

    // CHECK:        return [[GATHER]] : tensor<1x1x300x80xf16>
}

// -----

// CHECK-LABEL: @NotConvertGatherElementsToGatherAsDifferentAxis
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x1x5376x80xf16>, [[INPUT1:%.+]]: tensor<1x1x1x80xsi32>)
func.func @NotConvertGatherElementsToGatherAsDifferentAxis(%arg0: tensor<1x1x5376x80xf16>,
                                                           %arg1: tensor<1x1x1x80xsi32>)
                                                           -> (tensor<1x1x300x80xf16>) {
    %0 = IE.Tile(%arg1) {repeats_values = [1, 1, 300, 1]} : tensor<1x1x1x80xsi32> -> tensor<1x1x300x80xsi32>
    %1 = IE.GatherElements(%arg0, %0) {axis = 2 : i64} : tensor<1x1x5376x80xf16>, tensor<1x1x300x80xsi32> -> tensor<1x1x300x80xf16>

    return %1 : tensor<1x1x300x80xf16>

    // CHECK:        [[TILE:%.+]] = IE.Tile([[INPUT1]]) {repeats_values = [1, 1, 300, 1]} : tensor<1x1x1x80xsi32> -> tensor<1x1x300x80xsi32>
    // CHECK:        [[GATHERELEMENTS:%.+]] = IE.GatherElements([[INPUT0]], [[TILE]]) {axis = 2 : i64} : tensor<1x1x5376x80xf16>, tensor<1x1x300x80xsi32> -> tensor<1x1x300x80xf16>

    // CHECK:        return [[GATHERELEMENTS]] : tensor<1x1x300x80xf16>
}

// -----

// CHECK-LABEL: @NotConvertGatherElementsToGatherAsTileNonOneDimsSize
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x5376x80xf16>, [[INPUT1:%.+]]: tensor<1x16x300x1xsi32>)
func.func @NotConvertGatherElementsToGatherAsTileNonOneDimsSize(%arg0: tensor<1x16x5376x80xf16>,
                                                                %arg1: tensor<1x16x300x1xsi32>)
                                                                -> (tensor<1x16x300x80xf16>) {
    %0 = IE.Tile(%arg1) {repeats_values = [1, 1, 1, 80]} : tensor<1x16x300x1xsi32> -> tensor<1x16x300x80xsi32>
    %1 = IE.GatherElements(%arg0, %0) {axis = 2 : i64} : tensor<1x16x5376x80xf16>, tensor<1x16x300x80xsi32> -> tensor<1x16x300x80xf16>

    return %1 : tensor<1x16x300x80xf16>

    // CHECK:        [[TILE:%.+]] = IE.Tile([[INPUT1]]) {repeats_values = [1, 1, 1, 80]} : tensor<1x16x300x1xsi32> -> tensor<1x16x300x80xsi32>
    // CHECK:        [[GATHERELEMENTS:%.+]] = IE.GatherElements([[INPUT0]], [[TILE]]) {axis = 2 : i64} : tensor<1x16x5376x80xf16>, tensor<1x16x300x80xsi32> -> tensor<1x16x300x80xf16>

    // CHECK:        return [[GATHERELEMENTS]] : tensor<1x16x300x80xf16>
}
