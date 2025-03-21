//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tile-gather %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// -----

// CHECK-LABEL: @TileGatherElement
// CHECK-SAME: ([[ARG0:%.+]]: tensor<12x4096xf16>, [[ARG1:%.+]]:  tensor<1x1xsi32>)

func.func @TileGatherElement(%arg0: tensor<12x4096xf16>, %arg1: tensor<1x1xsi32>) -> tensor<1x1x4096xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x4096xf16>, tensor<1x1xsi32> -> tensor<1x1x4096xf16>
    return %0 :  tensor<1x1x4096xf16>

    // CHECK:       [[TILE0:%.+]] = VPU.Slice [[ARG0]] [0, 0] [12, 2048] : tensor<12x4096xf16> to tensor<12x2048xf16>
    // CHECK:       [[GATHER0:%.+]] = VPU.Gather([[TILE0]], [[ARG1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x2048xf16>, tensor<1x1xsi32> -> tensor<1x1x2048xf16>
    // CHECK:       [[TILE1:%.+]] = VPU.Slice [[ARG0]] [0, 2048] [12, 2048] : tensor<12x4096xf16> to tensor<12x2048xf16>
    // CHECK:       [[GATHER1:%.+]] = VPU.Gather([[TILE1]], [[ARG1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x2048xf16>, tensor<1x1xsi32> -> tensor<1x1x2048xf16>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GATHER0]], [[GATHER1]])
    // CHECK-SAME{LITERAL}              {static_offsets = [[0, 0, 0], [0, 0, 2048]]} : tensor<1x1x2048xf16>, tensor<1x1x2048xf16> -> tensor<1x1x4096xf16>
    // CHECK:       return      [[CONCAT]] : tensor<1x1x4096xf16>
}


// -----

// CHECK-LABEL: @TileGatherElementMoreTile
// CHECK-SAME: ([[ARG0:%.+]]: tensor<12x4097xf16>, [[ARG1:%.+]]:  tensor<1x1xsi32>)

func.func @TileGatherElementMoreTile(%arg0: tensor<12x4097xf16>, %arg1: tensor<1x1xsi32>) -> tensor<1x1x4097xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x4097xf16>, tensor<1x1xsi32> -> tensor<1x1x4097xf16>
    return %0 :  tensor<1x1x4097xf16>

    // CHECK:       [[TILE0:%.+]] = VPU.Slice [[ARG0]] [0, 0] [12, 1366] : tensor<12x4097xf16> to tensor<12x1366xf16>
    // CHECK:       [[GATHER0:%.+]] = VPU.Gather([[TILE0]], [[ARG1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x1366xf16>, tensor<1x1xsi32> -> tensor<1x1x1366xf16>
    // CHECK:       [[TILE1:%.+]] = VPU.Slice [[ARG0]] [0, 1366] [12, 1366] : tensor<12x4097xf16> to tensor<12x1366xf16>
    // CHECK:       [[GATHER1:%.+]] = VPU.Gather([[TILE1]], [[ARG1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x1366xf16>, tensor<1x1xsi32> -> tensor<1x1x1366xf16>
    // CHECK:       [[TILE2:%.+]] = VPU.Slice [[ARG0]] [0, 2732] [12, 1365] : tensor<12x4097xf16> to tensor<12x1365xf16>
    // CHECK:       [[GATHER2:%.+]] = VPU.Gather([[TILE2]], [[ARG1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x1365xf16>, tensor<1x1xsi32> -> tensor<1x1x1365xf16>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GATHER0]], [[GATHER1]], [[GATHER2]])
    // CHECK-SAME{LITERAL}          {static_offsets = [[0, 0, 0], [0, 0, 1366], [0, 0, 2732]]} : tensor<1x1x1366xf16>, tensor<1x1x1366xf16>, tensor<1x1x1365xf16> -> tensor<1x1x4097xf16>
    // CHECK:       return      [[CONCAT]] : tensor<1x1x4097xf16>
}


// -----

// CHECK-LABEL: @TileGatherIndices
// CHECK-SAME: ([[ARG0:%.+]]: tensor<12x1xf16>, [[ARG1:%.+]]:  tensor<1x100000xsi32>)

func.func @TileGatherIndices(%arg0: tensor<12x1xf16>, %arg1: tensor<1x100000xsi32>) -> tensor<1x100000x1xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x1xf16>, tensor<1x100000xsi32> -> tensor<1x100000x1xf16>
    return %0 :  tensor<1x100000x1xf16>

    // CHECK:       [[TILE0:%.+]] = VPU.Slice [[ARG1]] [0, 0] [1, 50000] : tensor<1x100000xsi32> to tensor<1x50000xsi32>
    // CHECK:       [[GATHER0:%.+]] = VPU.Gather([[ARG0]], [[TILE0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x1xf16>, tensor<1x50000xsi32> -> tensor<1x50000x1xf16>
    // CHECK:       [[TILE1:%.+]] = VPU.Slice [[ARG1]] [0, 50000] [1, 50000] : tensor<1x100000xsi32> to tensor<1x50000xsi32>
    // CHECK:       [[GATHER1:%.+]] = VPU.Gather([[ARG0]], [[TILE1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x1xf16>, tensor<1x50000xsi32> -> tensor<1x50000x1xf16>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GATHER0]], [[GATHER1]])
    // CHECK-SAME{LITERAL}              {static_offsets = [[0, 0, 0], [0, 50000, 0]]} : tensor<1x50000x1xf16>, tensor<1x50000x1xf16> -> tensor<1x100000x1xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x100000x1xf16>
}

// -----

// CHECK-LABEL: @Tile4DGatherElement
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x12x4096xf16>, [[ARG1:%.+]]:  tensor<1x1x1x1xsi32>)

func.func @Tile4DGatherElement(%arg0: tensor<1x1x12x4096xf16>, %arg1: tensor<1x1x1x1xsi32>) -> tensor<1x1x1x4096xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x12x4096xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x4096xf16>
    return %0 :  tensor<1x1x1x4096xf16>

    // CHECK:       [[TILE0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 1, 12, 2048] : tensor<1x1x12x4096xf16> to tensor<1x1x12x2048xf16>
    // CHECK:       [[GATHER0:%.+]] = VPU.Gather([[TILE0]], [[ARG1]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x12x2048xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x2048xf16>
    // CHECK:       [[TILE1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 2048] [1, 1, 12, 2048] : tensor<1x1x12x4096xf16> to tensor<1x1x12x2048xf16>
    // CHECK:       [[GATHER1:%.+]] = VPU.Gather([[TILE1]], [[ARG1]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x12x2048xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x2048xf16>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GATHER0]], [[GATHER1]])
    // CHECK-SAME{LITERAL}              {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 2048]]} : tensor<1x1x1x2048xf16>, tensor<1x1x1x2048xf16> -> tensor<1x1x1x4096xf16>
    // CHECK:       return      [[CONCAT]] : tensor<1x1x1x4096xf16>
}

// -----

// CHECK-LABEL: @Tile4DGatherIndices
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x12x1xf16>, [[ARG1:%.+]]:  tensor<1x100000x1x1xsi32>

func.func @Tile4DGatherIndices(%arg0: tensor<1x1x12x1xf16>, %arg1: tensor<1x100000x1x1xsi32>) -> tensor<1x1x100000x1xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x12x1xf16>, tensor<1x100000x1x1xsi32> -> tensor<1x1x100000x1xf16>
    return %0 :  tensor<1x1x100000x1xf16>

    // CHECK:       [[TILE0:%.+]] = VPU.Slice [[ARG1]] [0, 0, 0, 0] [1, 50000, 1, 1] : tensor<1x100000x1x1xsi32> to tensor<1x50000x1x1xsi32>
    // CHECK:       [[GATHER0:%.+]] = VPU.Gather([[ARG0]], [[TILE0]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x12x1xf16>, tensor<1x50000x1x1xsi32> -> tensor<1x1x50000x1xf16>
    // CHECK:       [[TILE1:%.+]] = VPU.Slice [[ARG1]] [0, 50000, 0, 0] [1, 50000, 1, 1] : tensor<1x100000x1x1xsi32> to tensor<1x50000x1x1xsi32>
    // CHECK:       [[GATHER1:%.+]] = VPU.Gather([[ARG0]], [[TILE1]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x12x1xf16>, tensor<1x50000x1x1xsi32> -> tensor<1x1x50000x1xf16>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GATHER0]], [[GATHER1]])
    // CHECK-SAME{LITERAL}              {static_offsets = [[0, 0, 0, 0], [0, 0, 50000, 0]]} : tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16> -> tensor<1x1x100000x1xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x1x100000x1xf16>
}

// -----

// CHECK-LABEL: @NotTileGatherForSmallSize
// CHECK-SAME: ([[ARG0:%.+]]: tensor<12x2048xf16>, [[ARG1:%.+]]:  tensor<1x1xsi32>)

func.func @NotTileGatherForSmallSize(%arg0: tensor<12x2048xf16>, %arg1: tensor<1x1xsi32>) -> tensor<1x1x2048xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x2048xf16>, tensor<1x1xsi32> -> tensor<1x1x2048xf16>
    return %0 :  tensor<1x1x2048xf16>

    // CHECK:       [[GATHER:%.+]] = VPU.Gather([[ARG0]], [[ARG1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x2048xf16>, tensor<1x1xsi32> -> tensor<1x1x2048xf16>
    // CHECK:       return      [[GATHER]] : tensor<1x1x2048xf16>
}

// -----

// CHECK-LABEL: @NotTileGatherForCouldNotConverToGatherDMA
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x12x4096xf16>, [[ARG1:%.+]]:  tensor<1x1xsi32>)

func.func @NotTileGatherForCouldNotConverToGatherDMA(%arg0: tensor<3x12x4096xf16>, %arg1: tensor<1x1xsi32>) -> tensor<3x1x1x4096xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<3x12x4096xf16>, tensor<1x1xsi32> -> tensor<3x1x1x4096xf16>
    return %0 :  tensor<3x1x1x4096xf16>

    // CHECK:       [[GATHER:%.+]] = VPU.Gather([[ARG0]], [[ARG1]]) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<3x12x4096xf16>, tensor<1x1xsi32> -> tensor<3x1x1x4096xf16>
    // CHECK:       return      [[GATHER]] : tensor<3x1x1x4096xf16>
}
