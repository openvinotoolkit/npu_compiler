//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-op-to-dma-for-performant-execution %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// A big-element Gather (row 4096xf16 = 8192B > 4096B limit) that divides evenly is kept as a single
// chunked GatherDMA instead of being element-tiled.

// CHECK-LABEL: @SplitBigElementToChunkedIndices
// CHECK-SAME: ([[ARG0:%.+]]: tensor<12x4096xf16>, [[ARG1:%.+]]:  tensor<1x1xsi32>)
func.func @SplitBigElementToChunkedIndices(%arg0: tensor<12x4096xf16>, %arg1: tensor<1x1xsi32>) -> tensor<1x1x4096xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x4096xf16>, tensor<1x1xsi32> -> tensor<1x1x4096xf16>
    return %0 :  tensor<1x1x4096xf16>

    // chunkElems = 4096B*8/16 = 2048, chunksPerRow = 4096/2048 = 2
    // CHECK-NOT:   VPU.Concat
    // CHECK:       [[IN_RESHAPE:%.+]] = VPU.Reshape([[ARG0]]) {shape_value = [24, 2048]} : tensor<12x4096xf16> -> tensor<24x2048xf16>
    // CHECK:       [[IDX_RESHAPE:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1, 1, 1]} : tensor<1x1xsi32> -> tensor<1x1x1x1xsi32>
    // CHECK:       [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<2>
    // CHECK:       [[SCALED:%.+]] = VPU.Multiply([[IDX_RESHAPE]], [[SCALE]]) {{.*}} -> tensor<1x1x1x1xsi32>
    // CHECK:       [[IOTA:%.+]] = const.Declare tensor<1x1x1x2xsi32> = dense<{{\[\[\[}}[0, 1]]]]>
    // CHECK:       [[EXPANDED:%.+]] = VPU.Add([[SCALED]], [[IOTA]]) {{.*}} : tensor<1x1x1x1xsi32>, tensor<1x1x1x2xsi32> -> tensor<1x1x1x2xsi32>
    // CHECK:       [[IDX_FLAT:%.+]] = VPU.Reshape([[EXPANDED]]) {shape_value = [2]} : tensor<1x1x1x2xsi32> -> tensor<2xsi32>
    // CHECK:       [[IDX_2D:%.+]] = VPU.Reshape([[IDX_FLAT]]) {shape_value = [2, 1]} : tensor<2xsi32> -> tensor<2x1xsi32>
    // CHECK:       [[IDX_I64:%.+]] = VPU.Convert([[IDX_2D]]) {dstElemType = i64} : tensor<2x1xsi32> -> tensor<2x1xi64>
    // CHECK:       [[GATHER:%.+]] = VPU.GatherDMA([[IN_RESHAPE]], [[IDX_I64]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<24x2048xf16>, tensor<2x1xi64> -> tensor<2x2048xf16>
    // CHECK-NOT:   VPU.Concat
    // CHECK:       [[RESHAPE_GATHER:%.+]] = VPU.Reshape([[GATHER]]) {shape_value = [2, 2048]} : tensor<2x2048xf16> -> tensor<2x2048xf16>
    // CHECK:       [[OUT:%.+]] = VPU.Reshape([[RESHAPE_GATHER]]) {shape_value = [1, 1, 4096]} : tensor<2x2048xf16> -> tensor<1x1x4096xf16>
    // CHECK:       return [[OUT]] : tensor<1x1x4096xf16>
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

// -----

// A whole-row Gather (2048x1024xf16 row) dividing evenly into 1024 chunks is kept as a
// single 1024-entry chunked GatherDMA.

// CHECK-LABEL: @SplitBigElementHeadFuse
// CHECK-SAME: ([[ARG0:%.+]]: tensor<15x2048x1024xf16>, [[ARG1:%.+]]: tensor<1x1xsi32>)
func.func @SplitBigElementHeadFuse(%arg0: tensor<15x2048x1024xf16>, %arg1: tensor<1x1xsi32>) -> tensor<1x1x2048x1024xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<15x2048x1024xf16>, tensor<1x1xsi32> -> tensor<1x1x2048x1024xf16>
    return %0 : tensor<1x1x2048x1024xf16>

    // chunkElems = 4096B*8/16 = 2048, chunksPerRow = (2048*1024)/2048 = 1024, expanded indices = 1*1024
    // CHECK-NOT:   VPU.Concat
    // CHECK:       [[IN_RESHAPE:%.+]] = VPU.Reshape([[ARG0]]) {shape_value = [15360, 2048]} : tensor<15x2048x1024xf16> -> tensor<15360x2048xf16>
    // CHECK:       [[IDX_RESHAPE:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1, 1, 1]} : tensor<1x1xsi32> -> tensor<1x1x1x1xsi32>
    // CHECK:       [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<1024> : tensor<1x1x1x1xsi32>
    // CHECK:       [[SCALED:%.+]] = VPU.Multiply([[IDX_RESHAPE]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xsi32>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x1xsi32>
    // CHECK:       [[IOTA:%.+]] = const.Declare tensor<1x1x1x1024xsi32>
    // CHECK:       [[EXPANDED:%.+]] = VPU.Add([[SCALED]], [[IOTA]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xsi32>, tensor<1x1x1x1024xsi32> -> tensor<1x1x1x1024xsi32>
    // CHECK:       [[IDX_FLAT:%.+]] = VPU.Reshape([[EXPANDED]]) {shape_value = [1024]} : tensor<1x1x1x1024xsi32> -> tensor<1024xsi32>
    // CHECK:       [[IDX_2D:%.+]] = VPU.Reshape([[IDX_FLAT]]) {shape_value = [1024, 1]} : tensor<1024xsi32> -> tensor<1024x1xsi32>
    // CHECK:       [[IDX_I64:%.+]] = VPU.Convert([[IDX_2D]]) {dstElemType = i64} : tensor<1024x1xsi32> -> tensor<1024x1xi64>
    // CHECK:       [[GATHER:%.+]] = VPU.GatherDMA([[IN_RESHAPE]], [[IDX_I64]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<15360x2048xf16>, tensor<1024x1xi64> -> tensor<1024x2048xf16>
    // CHECK-NOT:   VPU.Concat
    // CHECK:       [[RESHAPE_GATHER:%.+]] = VPU.Reshape([[GATHER]]) {shape_value = [1024, 2048]} : tensor<1024x2048xf16> -> tensor<1024x2048xf16>
    // CHECK:       [[OUT:%.+]] = VPU.Reshape([[RESHAPE_GATHER]]) {shape_value = [1, 1, 2048, 1024]} : tensor<1024x2048xf16> -> tensor<1x1x2048x1024xf16>
    // CHECK:       return [[OUT]] : tensor<1x1x2048x1024xf16>
}

// -----

// CHECK-LABEL: @SplitBigElementSi64Indices
// CHECK-SAME: ([[ARG0:%.+]]: tensor<12x4096xf16>, [[ARG1:%.+]]: tensor<1x1xsi64>)
func.func @SplitBigElementSi64Indices(%arg0: tensor<12x4096xf16>, %arg1: tensor<1x1xsi64>) -> tensor<1x1x4096xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x4096xf16>, tensor<1x1xsi64> -> tensor<1x1x4096xf16>
    return %0 : tensor<1x1x4096xf16>

    // CHECK-NOT:   VPU.Concat
    // CHECK:       [[IN_RESHAPE:%.+]] = VPU.Reshape([[ARG0]]) {shape_value = [24, 2048]} : tensor<12x4096xf16> -> tensor<24x2048xf16>
    // CHECK:       [[IDX_RESHAPE:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1, 1, 1]} : tensor<1x1xsi64> -> tensor<1x1x1x1xsi64>
    // CHECK:       [[IDX_SI32:%.+]] = VPU.Convert([[IDX_RESHAPE]]) {dstElemType = si32} : tensor<1x1x1x1xsi64> -> tensor<1x1x1x1xsi32>
    // CHECK:       [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<2> : tensor<1x1x1x1xsi32>
    // CHECK:       [[SCALED:%.+]] = VPU.Multiply([[IDX_SI32]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xsi32>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x1xsi32>
    // CHECK:       [[IOTA:%.+]] = const.Declare tensor<1x1x1x2xsi32> = dense<{{\[\[\[}}[0, 1]]]]> : tensor<1x1x1x2xsi32>
    // CHECK:       [[EXPANDED:%.+]] = VPU.Add([[SCALED]], [[IOTA]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xsi32>, tensor<1x1x1x2xsi32> -> tensor<1x1x1x2xsi32>
    // CHECK:       [[IDX_FLAT:%.+]] = VPU.Reshape([[EXPANDED]]) {shape_value = [2]} : tensor<1x1x1x2xsi32> -> tensor<2xsi32>
    // CHECK:       [[IDX_2D:%.+]] = VPU.Reshape([[IDX_FLAT]]) {shape_value = [2, 1]} : tensor<2xsi32> -> tensor<2x1xsi32>
    // CHECK:       [[IDX_I64:%.+]] = VPU.Convert([[IDX_2D]]) {dstElemType = i64} : tensor<2x1xsi32> -> tensor<2x1xi64>
    // CHECK:       [[GATHER:%.+]] = VPU.GatherDMA([[IN_RESHAPE]], [[IDX_I64]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<24x2048xf16>, tensor<2x1xi64> -> tensor<2x2048xf16>
    // CHECK-NOT:   VPU.Concat
    // CHECK:       [[RESHAPE_GATHER:%.+]] = VPU.Reshape([[GATHER]]) {shape_value = [2, 2048]} : tensor<2x2048xf16> -> tensor<2x2048xf16>
    // CHECK:       [[OUT:%.+]] = VPU.Reshape([[RESHAPE_GATHER]]) {shape_value = [1, 1, 4096]} : tensor<2x2048xf16> -> tensor<1x1x4096xf16>
    // CHECK:       return [[OUT]] : tensor<1x1x4096xf16>
}

// -----

// Constant negative indices must not take the chunked path (the index arithmetic would defeat the
// negative-index normalization in MoveToDMAGather); fall back to element-tiling instead.

// CHECK-LABEL: @SplitBigElementNegativeConstIndicesFallback
// CHECK-SAME: ([[ARG0:%.+]]: tensor<12x4096xf16>)
func.func @SplitBigElementNegativeConstIndicesFallback(%arg0: tensor<12x4096xf16>) -> tensor<1x1x4096xf16> {
    %indices = const.Declare tensor<1x1xsi32> = dense<-1> : tensor<1x1xsi32>
    %0 = VPU.Gather(%arg0, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x4096xf16>, tensor<1x1xsi32> -> tensor<1x1x4096xf16>
    return %0 : tensor<1x1x4096xf16>

    // No index arithmetic is emitted; the element-tiling fallback produces a Concat instead.
    // CHECK-NOT:   VPU.Multiply
    // CHECK-NOT:   VPU.Add
}

// -----

// CHECK-LABEL: @ExpandIndicesForDimsBeforeAxisGreaterThanOne
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1024x35x256xf32>, [[ARG1:%.+]]: tensor<1x3x1x1xsi32>)
func.func @ExpandIndicesForDimsBeforeAxisGreaterThanOne(%arg0: tensor<1x1024x35x256xf32>, %arg1: tensor<1x3x1x1xsi32>) -> tensor<1x1024x3x256xf32> {
    %0 = VPU.Gather(%arg0, %arg1) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1024x35x256xf32>, tensor<1x3x1x1xsi32> -> tensor<1x1024x3x256xf32>
    return %0 : tensor<1x1024x3x256xf32>

    // CHECK:       [[IN_RESHAPE:%.+]] = VPU.Reshape([[ARG0]]) {shape_value = [35840, 256]} : tensor<1x1024x35x256xf32> -> tensor<35840x256xf32>
    // CHECK:       [[IDX_RESHAPE:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1, 1, 3]} : tensor<1x3x1x1xsi32> -> tensor<1x1x1x3xsi32>
    // CHECK:       [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<1>
    // CHECK:       [[SCALED:%.+]] = VPU.Multiply([[IDX_RESHAPE]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x3xsi32>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x3xsi32>
    // CHECK:       [[IOTA:%.+]] = const.Declare tensor<1x1x1024x1xsi32>
    // CHECK:       [[EXPANDED:%.+]] = VPU.Add([[SCALED]], [[IOTA]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x3xsi32>, tensor<1x1x1024x1xsi32> -> tensor<1x1x1024x3xsi32>
    // CHECK:       [[IDX_FLAT:%.+]] = VPU.Reshape([[EXPANDED]]) {shape_value = [3072]} : tensor<1x1x1024x3xsi32> -> tensor<3072xsi32>
    // CHECK:       [[IDX_2D:%.+]] = VPU.Reshape([[IDX_FLAT]]) {shape_value = [3072, 1]} : tensor<3072xsi32> -> tensor<3072x1xsi32>
    // CHECK:       [[IDX_I64:%.+]] = VPU.Convert([[IDX_2D]]) {dstElemType = i64} : tensor<3072x1xsi32> -> tensor<3072x1xi64>
    // CHECK:       [[GATHER:%.+]] = VPU.GatherDMA([[IN_RESHAPE]], [[IDX_I64]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<35840x256xf32>, tensor<3072x1xi64> -> tensor<3072x256xf32>
    // CHECK:       [[RESHAPE_GATHER:%.+]] = VPU.Reshape([[GATHER]]) {shape_value = [3072, 256]} : tensor<3072x256xf32> -> tensor<3072x256xf32>
    // CHECK:       [[OUT:%.+]] = VPU.Reshape([[RESHAPE_GATHER]]) {shape_value = [1, 1024, 3, 256]} : tensor<3072x256xf32> -> tensor<1x1024x3x256xf32>
    // CHECK:       return [[OUT]] : tensor<1x1024x3x256xf32>
}

// -----


// CHECK-LABEL: @NotOptBatchDimSizeGreaterThanOne
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x1024x35x256xf32>, [[ARG1:%.+]]: tensor<2x1x1x1xsi32>)
func.func @NotOptBatchDimSizeGreaterThanOne(%arg0: tensor<2x1024x35x256xf32>, %arg1: tensor<2x1x1x1xsi32>) -> tensor<2x1024x1x256xf32> {
    %0 = VPU.Gather(%arg0, %arg1) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<2x1024x35x256xf32>, tensor<2x1x1x1xsi32> -> tensor<2x1024x1x256xf32>
    return %0 : tensor<2x1024x1x256xf32>

    // CHECK-NOT:   VPU.GatherDMA
    // CHECK:       VPU.Gather([[ARG0]], [[ARG1]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64}
}

// -----

// CHECK-LABEL: @NotTileGatherForCouldNotConvertToGatherDMA
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x12x4096xf16>, [[ARG1:%.+]]: tensor<1x1xsi32>)
func.func @NotTileGatherForCouldNotConvertToGatherDMA(%arg0: tensor<3x12x4096xf16>, %arg1: tensor<1x1xsi32>) -> tensor<3x1x1x4096xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<3x12x4096xf16>, tensor<1x1xsi32> -> tensor<3x1x1x4096xf16>
    return %0 : tensor<3x1x1x4096xf16>

    // CHECK-NOT:   VPU.GatherDMA
    // CHECK:       VPU.Gather([[ARG0]], [[ARG1]]) {axis_value = 1 : i64, batch_dims = 0 : i64}
}
