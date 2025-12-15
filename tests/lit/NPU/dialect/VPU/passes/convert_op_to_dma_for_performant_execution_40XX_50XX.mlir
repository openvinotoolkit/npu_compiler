//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-op-to-dma-for-performant-execution %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @TileGatherElement
// CHECK-SAME: ([[ARG0:%.+]]: tensor<12x4096xf16>, [[ARG1:%.+]]:  tensor<1x1xsi32>)
func.func @TileGatherElement(%arg0: tensor<12x4096xf16>, %arg1: tensor<1x1xsi32>) -> tensor<1x1x4096xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x4096xf16>, tensor<1x1xsi32> -> tensor<1x1x4096xf16>
    return %0 :  tensor<1x1x4096xf16>

    // CHECK:       [[TILE0:%.+]] = VPU.Slice [[ARG0]] [0, 0] [12, 2048] : tensor<12x4096xf16> to tensor<12x2048xf16>
    // CHECK:       [[RESHAPE0:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1]} : tensor<1x1xsi32> -> tensor<1x1xsi32>
    // CHECK:       [[INDICES0:%.+]] = VPU.Convert([[RESHAPE0]]) {dstElemType = i64} : tensor<1x1xsi32> -> tensor<1x1xi64>
    // CHECK:       [[GATHER0:%.+]] = VPU.GatherDMA([[TILE0]], [[INDICES0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x2048xf16>, tensor<1x1xi64> -> tensor<1x2048xf16>
    // CHECK:       [[OUT_RESHAPE0:%.+]] = VPU.Reshape([[GATHER0]]) {shape_value = [1, 1, 2048]} : tensor<1x2048xf16> -> tensor<1x1x2048xf16>
    // CHECK:       [[TILE1:%.+]] = VPU.Slice [[ARG0]] [0, 2048] [12, 2048] : tensor<12x4096xf16> to tensor<12x2048xf16>
    // CHECK:       [[RESHAPE1:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1]} : tensor<1x1xsi32> -> tensor<1x1xsi32>
    // CHECK:       [[INDICES1:%.+]] = VPU.Convert([[RESHAPE1]]) {dstElemType = i64} : tensor<1x1xsi32> -> tensor<1x1xi64>
    // CHECK:       [[GATHER1:%.+]] = VPU.GatherDMA([[TILE1]], [[INDICES1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x2048xf16>, tensor<1x1xi64> -> tensor<1x2048xf16>
    // CHECK:       [[OUT_RESHAPE1:%.+]] = VPU.Reshape([[GATHER1]]) {shape_value = [1, 1, 2048]} : tensor<1x2048xf16> -> tensor<1x1x2048xf16>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUT_RESHAPE0]], [[OUT_RESHAPE1]])
    // CHECK-SAME{LITERAL}:             {static_offsets = [[0, 0, 0], [0, 0, 2048]]} : tensor<1x1x2048xf16>, tensor<1x1x2048xf16> -> tensor<1x1x4096xf16>
    // CHECK:       return      [[CONCAT]] : tensor<1x1x4096xf16>
}

// -----

// CHECK-LABEL: @TileGatherElementMoreTile
// CHECK-SAME: ([[ARG0:%.+]]: tensor<12x4097xf16>, [[ARG1:%.+]]:  tensor<1x1xsi32>)
func.func @TileGatherElementMoreTile(%arg0: tensor<12x4097xf16>, %arg1: tensor<1x1xsi32>) -> tensor<1x1x4097xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x4097xf16>, tensor<1x1xsi32> -> tensor<1x1x4097xf16>
    return %0 :  tensor<1x1x4097xf16>

    // CHECK:       [[TILE0:%.+]] = VPU.Slice [[ARG0]] [0, 0] [12, 1366] : tensor<12x4097xf16> to tensor<12x1366xf16>
    // CHECK:       [[RESHAPE0:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1]} : tensor<1x1xsi32> -> tensor<1x1xsi32>
    // CHECK:       [[INDICES0:%.+]] = VPU.Convert([[RESHAPE0]]) {dstElemType = i64} : tensor<1x1xsi32> -> tensor<1x1xi64>
    // CHECK:       [[GATHER0:%.+]] = VPU.GatherDMA([[TILE0]], [[INDICES0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x1366xf16>, tensor<1x1xi64> -> tensor<1x1366xf16>
    // CHECK:       [[OUT_RESHAPE0:%.+]] = VPU.Reshape([[GATHER0]]) {shape_value = [1, 1, 1366]} : tensor<1x1366xf16> -> tensor<1x1x1366xf16>
    // CHECK:       [[TILE1:%.+]] = VPU.Slice [[ARG0]] [0, 1366] [12, 1366] : tensor<12x4097xf16> to tensor<12x1366xf16>
    // CHECK:       [[RESHAPE1:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1]} : tensor<1x1xsi32> -> tensor<1x1xsi32>
    // CHECK:       [[INDICES1:%.+]] = VPU.Convert([[RESHAPE1]]) {dstElemType = i64} : tensor<1x1xsi32> -> tensor<1x1xi64>
    // CHECK:       [[GATHER1:%.+]] = VPU.GatherDMA([[TILE1]], [[INDICES1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x1366xf16>, tensor<1x1xi64> -> tensor<1x1366xf16>
    // CHECK:       [[OUT_RESHAPE1:%.+]] = VPU.Reshape([[GATHER1]]) {shape_value = [1, 1, 1366]} : tensor<1x1366xf16> -> tensor<1x1x1366xf16>
    // CHECK:       [[TILE2:%.+]] = VPU.Slice [[ARG0]] [0, 2732] [12, 1365] : tensor<12x4097xf16> to tensor<12x1365xf16>
    // CHECK:       [[RESHAPE2:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1]} : tensor<1x1xsi32> -> tensor<1x1xsi32>
    // CHECK:       [[INDICES2:%.+]] = VPU.Convert([[RESHAPE2]]) {dstElemType = i64} : tensor<1x1xsi32> -> tensor<1x1xi64>
    // CHECK:       [[GATHER2:%.+]] = VPU.GatherDMA([[TILE2]], [[INDICES2]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x1365xf16>, tensor<1x1xi64> -> tensor<1x1365xf16>
    // CHECK:       [[OUT_RESHAPE2:%.+]] = VPU.Reshape([[GATHER2]]) {shape_value = [1, 1, 1365]} : tensor<1x1365xf16> -> tensor<1x1x1365xf16>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUT_RESHAPE0]], [[OUT_RESHAPE1]], [[OUT_RESHAPE2]])
    // CHECK-SAME{LITERAL}:             {static_offsets = [[0, 0, 0], [0, 0, 1366], [0, 0, 2732]]} : tensor<1x1x1366xf16>, tensor<1x1x1366xf16>, tensor<1x1x1365xf16> -> tensor<1x1x4097xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x1x4097xf16>
}

// -----

// CHECK-LABEL: @Tile4DGatherElement
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x12x4096xf16>, [[ARG1:%.+]]:  tensor<1x1x1x1xsi32>)
func.func @Tile4DGatherElement(%arg0: tensor<1x1x12x4096xf16>, %arg1: tensor<1x1x1x1xsi32>) -> tensor<1x1x1x4096xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x12x4096xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x4096xf16>
    return %0 :  tensor<1x1x1x4096xf16>

    // CHECK:       [[TILE0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 1, 12, 2048] : tensor<1x1x12x4096xf16> to tensor<1x1x12x2048xf16>
    // CHECK:       [[RESHAPE0:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1, 1, 1]} : tensor<1x1x1x1xsi32> -> tensor<1x1x1x1xsi32>
    // CHECK:       [[INDICES0:%.+]] = VPU.Convert([[RESHAPE0]]) {dstElemType = i64} : tensor<1x1x1x1xsi32> -> tensor<1x1x1x1xi64>
    // CHECK:       [[GATHER0:%.+]] = VPU.GatherDMA([[TILE0]], [[INDICES0]]) {addressing_mode = 1 : i64, axis_value = 2 : i64, batch_dims = 1 : i64} : tensor<1x1x12x2048xf16>, tensor<1x1x1x1xi64> -> tensor<1x1x1x2048xf16>
    // CHECK:       [[OUT_RESHAPE0:%.+]] = VPU.Reshape([[GATHER0]]) {shape_value = [1, 1, 1, 2048]} : tensor<1x1x1x2048xf16> -> tensor<1x1x1x2048xf16>
    // CHECK:       [[TILE1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 2048] [1, 1, 12, 2048] : tensor<1x1x12x4096xf16> to tensor<1x1x12x2048xf16>
    // CHECK:       [[RESHAPE1:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1, 1, 1]} : tensor<1x1x1x1xsi32> -> tensor<1x1x1x1xsi32>
    // CHECK:       [[INDICES1:%.+]] = VPU.Convert([[RESHAPE1]]) {dstElemType = i64} : tensor<1x1x1x1xsi32> -> tensor<1x1x1x1xi64>
    // CHECK:       [[GATHER1:%.+]] = VPU.GatherDMA([[TILE1]], [[INDICES1]]) {addressing_mode = 1 : i64, axis_value = 2 : i64, batch_dims = 1 : i64} : tensor<1x1x12x2048xf16>, tensor<1x1x1x1xi64> -> tensor<1x1x1x2048xf16>
    // CHECK:       [[OUT_RESHAPE1:%.+]] = VPU.Reshape([[GATHER1]]) {shape_value = [1, 1, 1, 2048]} : tensor<1x1x1x2048xf16> -> tensor<1x1x1x2048xf16>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUT_RESHAPE0]], [[OUT_RESHAPE1]])
    // CHECK-SAME{LITERAL}:             {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 2048]]} : tensor<1x1x1x2048xf16>, tensor<1x1x1x2048xf16> -> tensor<1x1x1x4096xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x1x1x4096xf16>
}
