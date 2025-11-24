//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-op-to-dma-for-performant-execution %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: @GatherMoveToDMA
// CHECK-SAME:  [[ARG0:%.+]]: tensor<30522x21xf16>, [[ARG1:%.+]]: tensor<1x512xsi32>
func.func @GatherMoveToDMA(%arg0: tensor<30522x21xf16>, %arg1: tensor<1x512xsi32>) -> tensor<1x512x21xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64
        } : tensor<30522x21xf16>, tensor<1x512xsi32> -> tensor<1x512x21xf16>

    return %0 :  tensor<1x512x21xf16>

    // CHECK:   [[RESHAPE_IN:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [512, 1]} : tensor<1x512xsi32> -> tensor<512x1xsi32>
    // CHECK:   [[CONVERT:%.+]] = VPU.Convert([[RESHAPE_IN]]) {dstElemType = i64} : tensor<512x1xsi32> -> tensor<512x1xi64>
    // CHECK:   [[GATHER_DMA:%.+]] = VPU.GatherDMA([[ARG0]], [[CONVERT]]) {
    // CHECK-SAME:      axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<30522x21xf16>, tensor<512x1xi64> -> tensor<512x21xf16>
    // CHECK:   [[RESHAPE_OUT:%.+]] = VPU.Reshape([[GATHER_DMA]]) {shape_value = [1, 512, 21]} : tensor<512x21xf16> -> tensor<1x512x21xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x512x21xf16>
}

// -----

// CHECK-LABEL: @GatherMoveToDMAWithMultipleIndices
// CHECK-SAME:  [[ARG0:%.+]]: tensor<8404x512xf16>, [[ARG1:%.+]]: tensor<100x10xsi32>
func.func @GatherMoveToDMAWithMultipleIndices(%arg0: tensor<8404x512xf16>, %arg1: tensor<100x10xsi32>) -> tensor<100x10x512xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64
        } : tensor<8404x512xf16>, tensor<100x10xsi32> -> tensor<100x10x512xf16>

    return %0 :  tensor<100x10x512xf16>

    // CHECK:   [[RESHAPE_IN:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1000, 1]} : tensor<100x10xsi32> -> tensor<1000x1xsi32>
    // CHECK:   [[CONVERT:%.+]] = VPU.Convert([[RESHAPE_IN]]) {dstElemType = i64} : tensor<1000x1xsi32> -> tensor<1000x1xi64>
    // CHECK:   [[GATHER_DMA:%.+]] = VPU.GatherDMA([[ARG0]], [[CONVERT]]) {
    // CHECK-SAME:      axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<8404x512xf16>, tensor<1000x1xi64> -> tensor<1000x512xf16>
    // CHECK:   [[RESHAPE_OUT:%.+]] = VPU.Reshape([[GATHER_DMA]]) {shape_value = [100, 10, 512]} : tensor<1000x512xf16> -> tensor<100x10x512xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<100x10x512xf16>
}

// -----

// CHECK-LABEL: @Gather4DMoveToDMA
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x30522x21xf16>, [[ARG1:%.+]]: tensor<1x512x1x1xsi32>
func.func @Gather4DMoveToDMA(%arg0: tensor<1x1x30522x21xf16>, %arg1: tensor<1x512x1x1xsi32>) -> tensor<1x1x512x21xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
        } : tensor<1x1x30522x21xf16>, tensor<1x512x1x1xsi32> -> tensor<1x1x512x21xf16>

    return %0 :  tensor<1x1x512x21xf16>

    // CHECK:   [[RESHAPE_IN:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1, 512, 1]} : tensor<1x512x1x1xsi32> -> tensor<1x1x512x1xsi32>
    // CHECK:   [[CONVERT:%.+]] = VPU.Convert([[RESHAPE_IN]]) {dstElemType = i64} : tensor<1x1x512x1xsi32> -> tensor<1x1x512x1xi64>
    // CHECK:   [[GATHER_DMA:%.+]] = VPU.GatherDMA([[ARG0]], [[CONVERT]]) {
    // CHECK-SAME:      addressing_mode = 1 : i64, axis_value = 2 : i64, batch_dims = 1 : i64} : tensor<1x1x30522x21xf16>, tensor<1x1x512x1xi64> -> tensor<1x1x512x21xf16>
    // CHECK:   [[RESHAPE_OUT:%.+]] = VPU.Reshape([[GATHER_DMA]]) {shape_value = [1, 1, 512, 21]} : tensor<1x1x512x21xf16> -> tensor<1x1x512x21xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x1x512x21xf16>
}

// -----

// CHECK-LABEL: @Gather4DMoveToDMAWithMultipleIndices
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x8404x512xf16>, [[ARG1:%.+]]: tensor<1x1000x1x1xsi32>
func.func @Gather4DMoveToDMAWithMultipleIndices(%arg0: tensor<1x1x8404x512xf16>, %arg1: tensor<1x1000x1x1xsi32>) -> tensor<1x1x1000x512xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
        } : tensor<1x1x8404x512xf16>, tensor<1x1000x1x1xsi32> -> tensor<1x1x1000x512xf16>

    return %0 :  tensor<1x1x1000x512xf16>

    // CHECK:   [[RESHAPE_IN:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1, 1000, 1]} : tensor<1x1000x1x1xsi32> -> tensor<1x1x1000x1xsi32>
    // CHECK:   [[CONVERT:%.+]] = VPU.Convert([[RESHAPE_IN]]) {dstElemType = i64} : tensor<1x1x1000x1xsi32> -> tensor<1x1x1000x1xi64>
    // CHECK:   [[GATHER_DMA:%.+]] = VPU.GatherDMA([[ARG0]], [[CONVERT]]) {
    // CHECK-SAME:      addressing_mode = 1 : i64, axis_value = 2 : i64, batch_dims = 1 : i64} : tensor<1x1x8404x512xf16>, tensor<1x1x1000x1xi64> -> tensor<1x1x1000x512xf16>
    // CHECK:   [[RESHAPE_OUT:%.+]] = VPU.Reshape([[GATHER_DMA]]) {shape_value = [1, 1, 1000, 512]} : tensor<1x1x1000x512xf16> -> tensor<1x1x1000x512xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x1x1000x512xf16>
}

// -----

// CHECK-LABEL: @GatherMoveToDMAWithNegativeIndices
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x8404x512xf16>
func.func @GatherMoveToDMAWithNegativeIndices(%arg0: tensor<1x1x8404x512xf16>) -> tensor<1x1x6x512xf16> {
    %0 = const.Declare tensor<1x6x1x1xsi32> = dense<[[[[0]], [[-10]], [[8]], [[100]], [[-20]], [[4]]]]> : tensor<1x6x1x1xsi32>
    %1 = VPU.Gather(%arg0, %0) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
        } : tensor<1x1x8404x512xf16>, tensor<1x6x1x1xsi32> -> tensor<1x1x6x512xf16>

    return %1 :  tensor<1x1x6x512xf16>

    // CHECK:   const.Declare tensor<1x1x6x1xi64>
    // CHECK:   [[NEW_INDICES:%.+]] = const.Declare tensor<1x1x6x1xi64> = dense<
    // CHECK-SAME{LITERAL}:         [[[[0]], [[8394]], [[8]], [[100]], [[8384]], [[4]]]]> : tensor<1x6x1x1xi64>, [#const.Reshape<[1, 1, 6, 1]>, #const.CastElemType<i64>]

    // CHECK:   [[GATHER_DMA:%.+]] = VPU.GatherDMA([[ARG0]], [[NEW_INDICES]]) {
    // CHECK-SAME:      addressing_mode = 1 : i64, axis_value = 2 : i64, batch_dims = 1 : i64} : tensor<1x1x8404x512xf16>, tensor<1x1x6x1xi64> -> tensor<1x1x6x512xf16>
    // CHECK:   [[RESHAPE_OUT:%.+]] = VPU.Reshape([[GATHER_DMA]]) {shape_value = [1, 1, 6, 512]} : tensor<1x1x6x512xf16> -> tensor<1x1x6x512xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x1x6x512xf16>
}

// -----

// CHECK-LABEL: @TileGatherIndices
// CHECK-SAME: ([[ARG0:%.+]]: tensor<12x1xf16>, [[ARG1:%.+]]:  tensor<1x100000xsi32>)
func.func @TileGatherIndices(%arg0: tensor<12x1xf16>, %arg1: tensor<1x100000xsi32>) -> tensor<1x100000x1xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x1xf16>, tensor<1x100000xsi32> -> tensor<1x100000x1xf16>
    return %0 :  tensor<1x100000x1xf16>

    // CHECK:       [[TILE0:%.+]] = VPU.Slice [[ARG1]] [0, 0] [1, 50000] : tensor<1x100000xsi32> to tensor<1x50000xsi32>
    // CHECK:       [[RESHAPE0:%.+]] = VPU.Reshape([[TILE0]]) {shape_value = [50000, 1]} : tensor<1x50000xsi32> -> tensor<50000x1xsi32>
    // CHECK:       [[INDICES0:%.+]] = VPU.Convert([[RESHAPE0]]) {dstElemType = i64} : tensor<50000x1xsi32> -> tensor<50000x1xi64>
    // CHECK:       [[GATHER0:%.+]] = VPU.GatherDMA([[ARG0]], [[INDICES0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x1xf16>, tensor<50000x1xi64> -> tensor<50000x1xf16>
    // CHECK:       [[OUT_RESHAPE0:%.+]] = VPU.Reshape([[GATHER0]]) {shape_value = [1, 50000, 1]} : tensor<50000x1xf16> -> tensor<1x50000x1xf16>
    // CHECK:       [[TILE1:%.+]] = VPU.Slice [[ARG1]] [0, 50000] [1, 50000] : tensor<1x100000xsi32> to tensor<1x50000xsi32>
    // CHECK:       [[RESHAPE1:%.+]] = VPU.Reshape([[TILE1]]) {shape_value = [50000, 1]} : tensor<1x50000xsi32> -> tensor<50000x1xsi32>
    // CHECK:       [[INDICES1:%.+]] = VPU.Convert([[RESHAPE1]]) {dstElemType = i64} : tensor<50000x1xsi32> -> tensor<50000x1xi64>
    // CHECK:       [[GATHER1:%.+]] = VPU.GatherDMA([[ARG0]], [[INDICES1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x1xf16>, tensor<50000x1xi64> -> tensor<50000x1xf16>
    // CHECK:       [[OUT_RESHAPE1:%.+]] = VPU.Reshape([[GATHER1]]) {shape_value = [1, 50000, 1]} : tensor<50000x1xf16> -> tensor<1x50000x1xf16>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUT_RESHAPE0]], [[OUT_RESHAPE1]])
    // CHECK-SAME{LITERAL}:             {static_offsets = [[0, 0, 0], [0, 50000, 0]]} : tensor<1x50000x1xf16>, tensor<1x50000x1xf16> -> tensor<1x100000x1xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x100000x1xf16>
}

// -----

// CHECK-LABEL: @Tile4DGatherIndices
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x12x1xf16>, [[ARG1:%.+]]:  tensor<1x100000x1x1xsi32>
func.func @Tile4DGatherIndices(%arg0: tensor<1x1x12x1xf16>, %arg1: tensor<1x100000x1x1xsi32>) -> tensor<1x1x100000x1xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x12x1xf16>, tensor<1x100000x1x1xsi32> -> tensor<1x1x100000x1xf16>
    return %0 :  tensor<1x1x100000x1xf16>

    // CHECK:       [[TILE0:%.+]] = VPU.Slice [[ARG1]] [0, 0, 0, 0] [1, 50000, 1, 1] : tensor<1x100000x1x1xsi32> to tensor<1x50000x1x1xsi32>
    // CHECK:       [[RESHAPE0:%.+]] = VPU.Reshape([[TILE0]]) {shape_value = [1, 1, 50000, 1]} : tensor<1x50000x1x1xsi32> -> tensor<1x1x50000x1xsi32>
    // CHECK:       [[INDICES0:%.+]] = VPU.Convert([[RESHAPE0]]) {dstElemType = i64} : tensor<1x1x50000x1xsi32> -> tensor<1x1x50000x1xi64>
    // CHECK:       [[GATHER0:%.+]] = VPU.GatherDMA([[ARG0]], [[INDICES0]]) {axis_value = 2 : i64, batch_dims = 1 : i64} : tensor<1x1x12x1xf16>, tensor<1x1x50000x1xi64> -> tensor<1x1x50000x1xf16>
    // CHECK:       [[OUT_RESHAPE0:%.+]] = VPU.Reshape([[GATHER0]]) {shape_value = [1, 1, 50000, 1]} : tensor<1x1x50000x1xf16> -> tensor<1x1x50000x1xf16>
    // CHECK:       [[TILE1:%.+]] = VPU.Slice [[ARG1]] [0, 50000, 0, 0] [1, 50000, 1, 1] : tensor<1x100000x1x1xsi32> to tensor<1x50000x1x1xsi32>
    // CHECK:       [[RESHAPE1:%.+]] = VPU.Reshape([[TILE1]]) {shape_value = [1, 1, 50000, 1]} : tensor<1x50000x1x1xsi32> -> tensor<1x1x50000x1xsi32>
    // CHECK:       [[INDICES1:%.+]] = VPU.Convert([[RESHAPE1]]) {dstElemType = i64} : tensor<1x1x50000x1xsi32> -> tensor<1x1x50000x1xi64>
    // CHECK:       [[GATHER1:%.+]] = VPU.GatherDMA([[ARG0]], [[INDICES1]]) {axis_value = 2 : i64, batch_dims = 1 : i64} : tensor<1x1x12x1xf16>, tensor<1x1x50000x1xi64> -> tensor<1x1x50000x1xf16>
    // CHECK:       [[OUT_RESHAPE1:%.+]] = VPU.Reshape([[GATHER1]]) {shape_value = [1, 1, 50000, 1]} : tensor<1x1x50000x1xf16> -> tensor<1x1x50000x1xf16>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUT_RESHAPE0]], [[OUT_RESHAPE1]])
    // CHECK-SAME{LITERAL}:             {static_offsets = [[0, 0, 0, 0], [0, 0, 50000, 0]]} : tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16> -> tensor<1x1x100000x1xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x1x100000x1xf16>
}

// -----

// CHECK-LABEL: @NotTileGatherForSmallSize
// CHECK-SAME: ([[ARG0:%.+]]: tensor<12x2048xf16>, [[ARG1:%.+]]:  tensor<1x1xsi32>)
func.func @NotTileGatherForSmallSize(%arg0: tensor<12x2048xf16>, %arg1: tensor<1x1xsi32>) -> tensor<1x1x2048xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x2048xf16>, tensor<1x1xsi32> -> tensor<1x1x2048xf16>
    return %0 :  tensor<1x1x2048xf16>

    // CHECK:       [[RESHAPE:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1]} : tensor<1x1xsi32> -> tensor<1x1xsi32>
    // CHECK:       [[INDICES:%.+]] = VPU.Convert([[RESHAPE]]) {dstElemType = i64} : tensor<1x1xsi32> -> tensor<1x1xi64>
    // CHECK:       [[GATHER:%.+]] = VPU.GatherDMA([[ARG0]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<12x2048xf16>, tensor<1x1xi64> -> tensor<1x2048xf16>
    // CHECK:       [[OUT_RESHAPE:%.+]] = VPU.Reshape([[GATHER]]) {shape_value = [1, 1, 2048]} : tensor<1x2048xf16> -> tensor<1x1x2048xf16>
    // CHECK:       return [[OUT_RESHAPE]] : tensor<1x1x2048xf16>
}

// -----

// CHECK-LABEL: @Gather4BitsTiling
// CHECK-SAME: ([[ARG0:%.+]]: tensor<645632x224xsi4>, [[ARG1:%.+]]:  tensor<1x1024xsi32>)
func.func @Gather4BitsTiling(%arg0: tensor<645632x224xsi4>, %arg1: tensor<1x1024xsi32>) -> tensor<1x1024x224xsi4> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<645632x224xsi4>, tensor<1x1024xsi32> -> tensor<1x1024x224xsi4>
    return %0 :  tensor<1x1024x224xsi4>

    // CHECK:       [[RESHAPE:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1024, 1]} : tensor<1x1024xsi32> -> tensor<1024x1xsi32>
    // CHECK:       [[INDICES:%.+]] = VPU.Convert([[RESHAPE]]) {dstElemType = i64} : tensor<1024x1xsi32> -> tensor<1024x1xi64>
    // CHECK:       [[GATHER:%.+]] = VPU.GatherDMA([[ARG0]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<645632x224xsi4>, tensor<1024x1xi64> -> tensor<1024x224xsi4>
    // CHECK:       [[OUT_RESHAPE:%.+]] = VPU.Reshape([[GATHER]]) {shape_value = [1, 1024, 224]} : tensor<1024x224xsi4> -> tensor<1x1024x224xsi4>
    // CHECK:       return [[OUT_RESHAPE]] : tensor<1x1024x224xsi4>
}

// -----

// CHECK-LABEL: @NotTileGatherForCouldNotConvertToGatherDMA
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x12x4096xf16>, [[ARG1:%.+]]:  tensor<1x1xsi32>)
func.func @NotTileGatherForCouldNotConvertToGatherDMA(%arg0: tensor<3x12x4096xf16>, %arg1: tensor<1x1xsi32>) -> tensor<3x1x1x4096xf16> {
    %0 =  VPU.Gather(%arg0, %arg1) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<3x12x4096xf16>, tensor<1x1xsi32> -> tensor<3x1x1x4096xf16>
    return %0 :  tensor<3x1x1x4096xf16>

    // CHECK:       [[GATHER:%.+]] = VPU.Gather([[ARG0]], [[ARG1]]) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<3x12x4096xf16>, tensor<1x1xsi32> -> tensor<3x1x1x4096xf16>
    // CHECK:       return      [[GATHER]] : tensor<3x1x1x4096xf16>
}
