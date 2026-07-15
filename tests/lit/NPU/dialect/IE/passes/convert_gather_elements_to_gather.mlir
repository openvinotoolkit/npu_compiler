//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-gather-elements-to-gather %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

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

// -----

// Test GatherElementsFlatConverter: same shape data/indices, axis=2, trailing unit dim, totalElements > 32768
// Shape: [1, 128, 512, 1], axis=2 -> batchSize=128, gatherSize=512, totalElements=65536
// rowsPerChunk=64, numChunks=2

// CHECK-LABEL: @ConvertGatherElementsFlatToChunkedGather
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x128x512x1xf16>, [[INPUT1:%.+]]: tensor<1x128x512x1xsi32>)
func.func @ConvertGatherElementsFlatToChunkedGather(%arg0: tensor<1x128x512x1xf16>,
                                                    %arg1: tensor<1x128x512x1xsi32>)
                                                    -> (tensor<1x128x512x1xf16>) {
    %0 = IE.GatherElements(%arg0, %arg1) {axis = 2 : i64} : tensor<1x128x512x1xf16>, tensor<1x128x512x1xsi32> -> tensor<1x128x512x1xf16>

    return %0 : tensor<1x128x512x1xf16>

    // CHECK-NOT:    IE.GatherElements

    // Reshape data and indices to 2D [128, 512]
    // CHECK:        [[DATA_2D:%.+]] = IE.Reshape([[INPUT0]]) {shape_value = [128, 512]} : tensor<1x128x512x1xf16> -> tensor<128x512xf16>
    // CHECK:        [[INDICES_2D:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [128, 512]} : tensor<1x128x512x1xsi32> -> tensor<128x512xsi32>

    // Row offsets constant for full chunk (64*512 = 32768 elements)
    // CHECK:        [[OFFSETS:%.+]] = const.Declare tensor<32768xsi32>

    // Chunk 0: rows [0, 64)
    // CHECK:        [[DATA_CHUNK0:%.+]] = IE.Slice [[DATA_2D]] [0, 0] [64, 512] : tensor<128x512xf16> to tensor<64x512xf16>
    // CHECK:        [[IDX_CHUNK0:%.+]] = IE.Slice [[INDICES_2D]] [0, 0] [64, 512] : tensor<128x512xsi32> to tensor<64x512xsi32>
    // CHECK:        [[IDX_FLAT0:%.+]] = IE.Reshape([[IDX_CHUNK0]]) {shape_value = [32768]} : tensor<64x512xsi32> -> tensor<32768xsi32>
    // CHECK:        [[IDX_ABS0:%.+]] = IE.Add([[IDX_FLAT0]], [[OFFSETS]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32768xsi32>, tensor<32768xsi32> -> tensor<32768xsi32>
    // CHECK:        [[DATA_FLAT0:%.+]] = IE.Reshape([[DATA_CHUNK0]]) {shape_value = [32768]} : tensor<64x512xf16> -> tensor<32768xf16>
    // CHECK:        [[GATHER0:%.+]] = IE.Gather([[DATA_FLAT0]], [[IDX_ABS0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<32768xf16>, tensor<32768xsi32> -> tensor<32768xf16>
    // CHECK:        [[CHUNK_2D0:%.+]] = IE.Reshape([[GATHER0]]) {shape_value = [64, 512]} : tensor<32768xf16> -> tensor<64x512xf16>

    // Chunk 1: rows [64, 128)
    // CHECK:        [[DATA_CHUNK1:%.+]] = IE.Slice [[DATA_2D]] [64, 0] [64, 512] : tensor<128x512xf16> to tensor<64x512xf16>
    // CHECK:        [[IDX_CHUNK1:%.+]] = IE.Slice [[INDICES_2D]] [64, 0] [64, 512] : tensor<128x512xsi32> to tensor<64x512xsi32>
    // CHECK:        [[IDX_FLAT1:%.+]] = IE.Reshape([[IDX_CHUNK1]]) {shape_value = [32768]} : tensor<64x512xsi32> -> tensor<32768xsi32>
    // CHECK:        [[IDX_ABS1:%.+]] = IE.Add([[IDX_FLAT1]], [[OFFSETS]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32768xsi32>, tensor<32768xsi32> -> tensor<32768xsi32>
    // CHECK:        [[DATA_FLAT1:%.+]] = IE.Reshape([[DATA_CHUNK1]]) {shape_value = [32768]} : tensor<64x512xf16> -> tensor<32768xf16>
    // CHECK:        [[GATHER1:%.+]] = IE.Gather([[DATA_FLAT1]], [[IDX_ABS1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<32768xf16>, tensor<32768xsi32> -> tensor<32768xf16>
    // CHECK:        [[CHUNK_2D1:%.+]] = IE.Reshape([[GATHER1]]) {shape_value = [64, 512]} : tensor<32768xf16> -> tensor<64x512xf16>

    // Concat chunks and reshape back
    // CHECK:        [[CONCAT:%.+]] = IE.Concat([[CHUNK_2D0]], [[CHUNK_2D1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<64x512xf16>, tensor<64x512xf16> -> tensor<128x512xf16>
    // CHECK:        [[RESULT:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 128, 512, 1]} : tensor<128x512xf16> -> tensor<1x128x512x1xf16>
    // CHECK:        return [[RESULT]] : tensor<1x128x512x1xf16>
}

// -----

// Test GatherElementsFlatConverter with partial last chunk:
// Shape: [1, 100, 512, 1], axis=2 -> batchSize=100, gatherSize=512, totalElements=51200
// rowsPerChunk=64, numChunks=2 (chunk0=64 rows, chunk1=36 rows - partial)

// CHECK-LABEL: @ConvertGatherElementsFlatPartialChunk
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x100x512x1xf16>, [[INPUT1:%.+]]: tensor<1x100x512x1xsi32>)
func.func @ConvertGatherElementsFlatPartialChunk(%arg0: tensor<1x100x512x1xf16>,
                                                  %arg1: tensor<1x100x512x1xsi32>)
                                                  -> (tensor<1x100x512x1xf16>) {
    %0 = IE.GatherElements(%arg0, %arg1) {axis = 2 : i64} : tensor<1x100x512x1xf16>, tensor<1x100x512x1xsi32> -> tensor<1x100x512x1xf16>

    return %0 : tensor<1x100x512x1xf16>

    // CHECK-NOT:    IE.GatherElements

    // CHECK:        [[DATA_2D:%.+]] = IE.Reshape([[INPUT0]]) {shape_value = [100, 512]} : tensor<1x100x512x1xf16> -> tensor<100x512xf16>
    // CHECK:        [[INDICES_2D:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [100, 512]} : tensor<1x100x512x1xsi32> -> tensor<100x512xsi32>

    // Full chunk offset constant (64*512 = 32768)
    // CHECK:        [[OFFSETS_FULL:%.+]] = const.Declare tensor<32768xsi32>

    // Chunk 0: 64 full rows
    // CHECK:        [[DATA_CHUNK0:%.+]] = IE.Slice [[DATA_2D]] [0, 0] [64, 512] : tensor<100x512xf16> to tensor<64x512xf16>
    // CHECK:        [[IDX_CHUNK0:%.+]] = IE.Slice [[INDICES_2D]] [0, 0] [64, 512] : tensor<100x512xsi32> to tensor<64x512xsi32>
    // CHECK:        [[IDX_FLAT0:%.+]] = IE.Reshape([[IDX_CHUNK0]]) {shape_value = [32768]} : tensor<64x512xsi32> -> tensor<32768xsi32>
    // CHECK:        [[IDX_ABS0:%.+]] = IE.Add([[IDX_FLAT0]], [[OFFSETS_FULL]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32768xsi32>, tensor<32768xsi32> -> tensor<32768xsi32>
    // CHECK:        [[DATA_FLAT0:%.+]] = IE.Reshape([[DATA_CHUNK0]]) {shape_value = [32768]} : tensor<64x512xf16> -> tensor<32768xf16>
    // CHECK:        [[GATHER0:%.+]] = IE.Gather([[DATA_FLAT0]], [[IDX_ABS0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<32768xf16>, tensor<32768xsi32> -> tensor<32768xf16>
    // CHECK:        [[CHUNK_2D0:%.+]] = IE.Reshape([[GATHER0]]) {shape_value = [64, 512]} : tensor<32768xf16> -> tensor<64x512xf16>

    // Chunk 1: 36 partial rows (36*512 = 18432), uses separate smaller offset constant
    // CHECK:        [[DATA_CHUNK1:%.+]] = IE.Slice [[DATA_2D]] [64, 0] [36, 512] : tensor<100x512xf16> to tensor<36x512xf16>
    // CHECK:        [[IDX_CHUNK1:%.+]] = IE.Slice [[INDICES_2D]] [64, 0] [36, 512] : tensor<100x512xsi32> to tensor<36x512xsi32>
    // CHECK:        [[IDX_FLAT1:%.+]] = IE.Reshape([[IDX_CHUNK1]]) {shape_value = [18432]} : tensor<36x512xsi32> -> tensor<18432xsi32>
    // CHECK:        [[OFFSETS_PARTIAL:%.+]] = const.Declare tensor<18432xsi32>
    // CHECK:        [[IDX_ABS1:%.+]] = IE.Add([[IDX_FLAT1]], [[OFFSETS_PARTIAL]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<18432xsi32>, tensor<18432xsi32> -> tensor<18432xsi32>
    // CHECK:        [[DATA_FLAT1:%.+]] = IE.Reshape([[DATA_CHUNK1]]) {shape_value = [18432]} : tensor<36x512xf16> -> tensor<18432xf16>
    // CHECK:        [[GATHER1:%.+]] = IE.Gather([[DATA_FLAT1]], [[IDX_ABS1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<18432xf16>, tensor<18432xsi32> -> tensor<18432xf16>
    // CHECK:        [[CHUNK_2D1:%.+]] = IE.Reshape([[GATHER1]]) {shape_value = [36, 512]} : tensor<18432xf16> -> tensor<36x512xf16>

    // Concat and reshape back
    // CHECK:        [[CONCAT:%.+]] = IE.Concat([[CHUNK_2D0]], [[CHUNK_2D1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<64x512xf16>, tensor<36x512xf16> -> tensor<100x512xf16>
    // CHECK:        [[RESULT:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 100, 512, 1]} : tensor<100x512xf16> -> tensor<1x100x512x1xf16>
    // CHECK:        return [[RESULT]] : tensor<1x100x512x1xf16>
}

// -----

// Negative test: total elements within safe limit (128*128=16384 <= 32768), should NOT be converted by FlatConverter
// But also no Tile pattern, so remains as GatherElements

// CHECK-LABEL: @NotConvertGatherElementsFlatSmallSize
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x128x128x1xsi32>, [[INPUT1:%.+]]: tensor<1x128x128x1xsi32>)
func.func @NotConvertGatherElementsFlatSmallSize(%arg0: tensor<1x128x128x1xsi32>,
                                                 %arg1: tensor<1x128x128x1xsi32>)
                                                 -> (tensor<1x128x128x1xsi32>) {
    %0 = IE.GatherElements(%arg0, %arg1) {axis = 2 : i64} : tensor<1x128x128x1xsi32>, tensor<1x128x128x1xsi32> -> tensor<1x128x128x1xsi32>

    return %0 : tensor<1x128x128x1xsi32>

    // CHECK:        [[GATHERELEMENTS:%.+]] = IE.GatherElements([[INPUT0]], [[INPUT1]]) {axis = 2 : i64} : tensor<1x128x128x1xsi32>, tensor<1x128x128x1xsi32> -> tensor<1x128x128x1xsi32>
    // CHECK:        return [[GATHERELEMENTS]] : tensor<1x128x128x1xsi32>
}

// -----

// Negative test: non-unit trailing dimension after axis, should NOT be converted by FlatConverter

// CHECK-LABEL: @NotConvertGatherElementsFlatNonUnitTrailingDim
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x128x512x4xsi32>, [[INPUT1:%.+]]: tensor<1x128x512x4xsi32>)
func.func @NotConvertGatherElementsFlatNonUnitTrailingDim(%arg0: tensor<1x128x512x4xsi32>,
                                                          %arg1: tensor<1x128x512x4xsi32>)
                                                          -> (tensor<1x128x512x4xsi32>) {
    %0 = IE.GatherElements(%arg0, %arg1) {axis = 2 : i64} : tensor<1x128x512x4xsi32>, tensor<1x128x512x4xsi32> -> tensor<1x128x512x4xsi32>

    return %0 : tensor<1x128x512x4xsi32>

    // CHECK:        [[GATHERELEMENTS:%.+]] = IE.GatherElements([[INPUT0]], [[INPUT1]]) {axis = 2 : i64} : tensor<1x128x512x4xsi32>, tensor<1x128x512x4xsi32> -> tensor<1x128x512x4xsi32>
    // CHECK:        return [[GATHERELEMENTS]] : tensor<1x128x512x4xsi32>
}

// -----

// Negative test: data and indices shapes differ, should NOT be converted by FlatConverter

// CHECK-LABEL: @NotConvertGatherElementsFlatDifferentShapes
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x128x512x1xsi32>, [[INPUT1:%.+]]: tensor<1x64x512x1xsi32>)
func.func @NotConvertGatherElementsFlatDifferentShapes(%arg0: tensor<1x128x512x1xsi32>,
                                                       %arg1: tensor<1x64x512x1xsi32>)
                                                       -> (tensor<1x64x512x1xsi32>) {
    %0 = IE.GatherElements(%arg0, %arg1) {axis = 2 : i64} : tensor<1x128x512x1xsi32>, tensor<1x64x512x1xsi32> -> tensor<1x64x512x1xsi32>

    return %0 : tensor<1x64x512x1xsi32>

    // CHECK:        [[GATHERELEMENTS:%.+]] = IE.GatherElements([[INPUT0]], [[INPUT1]]) {axis = 2 : i64} : tensor<1x128x512x1xsi32>, tensor<1x64x512x1xsi32> -> tensor<1x64x512x1xsi32>
    // CHECK:        return [[GATHERELEMENTS]] : tensor<1x64x512x1xsi32>
}

// -----

// Test GatherElementsFlatConverter with si64 indices: verifies row-offset constant matches indices type
// Shape: [1, 128, 512, 1], axis=2 -> batchSize=128, gatherSize=512, totalElements=65536

// CHECK-LABEL: @ConvertGatherElementsFlatSI64Indices
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x128x512x1xf16>, [[INPUT1:%.+]]: tensor<1x128x512x1xsi64>)
func.func @ConvertGatherElementsFlatSI64Indices(%arg0: tensor<1x128x512x1xf16>,
                                                 %arg1: tensor<1x128x512x1xsi64>)
                                                 -> (tensor<1x128x512x1xf16>) {
    %0 = IE.GatherElements(%arg0, %arg1) {axis = 2 : i64} : tensor<1x128x512x1xf16>, tensor<1x128x512x1xsi64> -> tensor<1x128x512x1xf16>

    return %0 : tensor<1x128x512x1xf16>

    // CHECK-NOT:    IE.GatherElements

    // CHECK:        [[DATA_2D:%.+]] = IE.Reshape([[INPUT0]]) {shape_value = [128, 512]} : tensor<1x128x512x1xf16> -> tensor<128x512xf16>
    // CHECK:        [[INDICES_2D:%.+]] = IE.Reshape([[INPUT1]]) {shape_value = [128, 512]} : tensor<1x128x512x1xsi64> -> tensor<128x512xsi64>

    // Row offsets constant must be si64 to match indices type
    // CHECK:        [[OFFSETS:%.+]] = const.Declare tensor<32768xsi64>

    // CHECK:        [[DATA_CHUNK0:%.+]] = IE.Slice [[DATA_2D]] [0, 0] [64, 512] : tensor<128x512xf16> to tensor<64x512xf16>
    // CHECK:        [[IDX_CHUNK0:%.+]] = IE.Slice [[INDICES_2D]] [0, 0] [64, 512] : tensor<128x512xsi64> to tensor<64x512xsi64>
    // CHECK:        [[IDX_FLAT0:%.+]] = IE.Reshape([[IDX_CHUNK0]]) {shape_value = [32768]} : tensor<64x512xsi64> -> tensor<32768xsi64>
    // CHECK:        [[IDX_ABS0:%.+]] = IE.Add([[IDX_FLAT0]], [[OFFSETS]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32768xsi64>, tensor<32768xsi64> -> tensor<32768xsi64>
    // CHECK:        [[DATA_FLAT0:%.+]] = IE.Reshape([[DATA_CHUNK0]]) {shape_value = [32768]} : tensor<64x512xf16> -> tensor<32768xf16>
    // CHECK:        [[GATHER0:%.+]] = IE.Gather([[DATA_FLAT0]], [[IDX_ABS0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<32768xf16>, tensor<32768xsi64> -> tensor<32768xf16>
    // CHECK:        [[CHUNK_2D0:%.+]] = IE.Reshape([[GATHER0]]) {shape_value = [64, 512]} : tensor<32768xf16> -> tensor<64x512xf16>

    // CHECK:        [[DATA_CHUNK1:%.+]] = IE.Slice [[DATA_2D]] [64, 0] [64, 512] : tensor<128x512xf16> to tensor<64x512xf16>
    // CHECK:        [[IDX_CHUNK1:%.+]] = IE.Slice [[INDICES_2D]] [64, 0] [64, 512] : tensor<128x512xsi64> to tensor<64x512xsi64>
    // CHECK:        [[IDX_FLAT1:%.+]] = IE.Reshape([[IDX_CHUNK1]]) {shape_value = [32768]} : tensor<64x512xsi64> -> tensor<32768xsi64>
    // CHECK:        [[IDX_ABS1:%.+]] = IE.Add([[IDX_FLAT1]], [[OFFSETS]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32768xsi64>, tensor<32768xsi64> -> tensor<32768xsi64>
    // CHECK:        [[DATA_FLAT1:%.+]] = IE.Reshape([[DATA_CHUNK1]]) {shape_value = [32768]} : tensor<64x512xf16> -> tensor<32768xf16>
    // CHECK:        [[GATHER1:%.+]] = IE.Gather([[DATA_FLAT1]], [[IDX_ABS1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<32768xf16>, tensor<32768xsi64> -> tensor<32768xf16>
    // CHECK:        [[CHUNK_2D1:%.+]] = IE.Reshape([[GATHER1]]) {shape_value = [64, 512]} : tensor<32768xf16> -> tensor<64x512xf16>

    // CHECK:        [[CONCAT:%.+]] = IE.Concat([[CHUNK_2D0]], [[CHUNK_2D1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<64x512xf16>, tensor<64x512xf16> -> tensor<128x512xf16>
    // CHECK:        [[RESULT:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 128, 512, 1]} : tensor<128x512xf16> -> tensor<1x128x512x1xf16>
    // CHECK:        return [[RESULT]] : tensor<1x128x512x1xf16>
}
