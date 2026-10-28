//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: env IE_NPU_LOG_FILTER="dump-statistics-of-task-ops" vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --dump-statistics-of-task-ops -o /dev/null %s 2>&1 | FileCheck %s
// REQUIRES: platform-NPU4000
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e-01>

module @DumpOpsStatisticsTestTwoCompressionModes {

    func.func @CompressedWeightsAndNoUncompressedWeights(%arg0: memref<1x512x3x3x!qElemType>, %arg1: memref<1x512x3x3x!qElemType>) -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]> {
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>

        %cst_1 = const.Declare memref<7600x1x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<1.0> : tensor<7600x1x1x1xf16>
        %2 = VPURT.DeclareBuffer <CMX_NN> <1605632> -> !VPUIP.DistributedBuffer<25088x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        VPURT.Task attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64, isTrailingSWLayer = false} {
            %4 = VPUIP.DecompressDMAOp inputs(%cst_1 : memref<7600x1x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>) outputs(%2 : !VPUIP.DistributedBuffer<25088x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<25088x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        }

        %cst_2 = const.Declare memref<1408x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<1> : tensor<1408x1x1x1xui8>
        %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
        VPURT.Task attributes {cycleBegin = 2 : i64, cycleEnd = 3 : i64, isTrailingSWLayer = false} {
            %4 = VPUIP.DecompressDMAOp inputs(%cst_2 : memref<1408x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>) outputs(%3 : memref<4608x1x1x1xui8, [@CMX_NN, 0]>) -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
        }
        return %0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
    } // func
}

// CHECK:   Input size - 4.50 KB Output size - 4.50 KB
// CHECK:   DDR heap size - 0 bytes
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 2 ops
// CHECK:     VPUIP.DecompressDMAOp - 2 ops
// CHECK:       DDR2CMX - 2 ops : Size - 53.50 KB
// CHECK:   Weights statistics
// CHECK:     Total weights - count: 2, size: 53.50 KB, compressed size: 16.21 KB, (30.32%)
// CHECK:     Compressed weights - count: 2, size: 53.50 KB, compressed size: 16.21 KB, (30.32%)
// CHECK:       F16 - count: 1, size: 49.00 KB, compressed size: 14.84 KB, (30.29%)
// CHECK:       Int8 - count: 1, size: 4.50 KB, compressed size: 1.37 KB, (30.56%)
// CHECK:   Const swizzling statistics:
// CHECK:     Swizzled constants     - count: 0, size: 0 bytes
// CHECK:     Not swizzled constants - count: 2, size: 16.21 KB

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e-01>

module @DumpOpsStatisticsTestF16compressionModeOnly {

    func.func @CompressedWeightsAndNoUncompressedWeights(%arg0: memref<1x512x3x3x!qElemType>, %arg1: memref<1x512x3x3x!qElemType>) -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]> {
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>

        %cst_1 = const.Declare memref<7600x1x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<1.0> : tensor<7600x1x1x1xf16>
        %2 = VPURT.DeclareBuffer <CMX_NN> <1605632> -> !VPUIP.DistributedBuffer<25088x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        VPURT.Task attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64, isTrailingSWLayer = false} {
            %4 = VPUIP.DecompressDMAOp inputs(%cst_1 : memref<7600x1x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>) outputs(%2 : !VPUIP.DistributedBuffer<25088x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<25088x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        }

        return %0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
    } // func
}

// CHECK:   Input size - 4.50 KB Output size - 4.50 KB
// CHECK:   DDR heap size - 0 bytes
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 1 ops
// CHECK:     VPUIP.DecompressDMAOp - 1 ops
// CHECK:       DDR2CMX - 1 ops : Size - 49.00 KB
// CHECK:   Weights statistics
// CHECK:     Total weights - count: 1, size: 49.00 KB, compressed size: 14.84 KB, (30.29%)
// CHECK:     Compressed weights - count: 1, size: 49.00 KB, compressed size: 14.84 KB, (30.29%)
// CHECK:       F16 - count: 1, size: 49.00 KB, compressed size: 14.84 KB, (30.29%)
// CHECK:   Const swizzling statistics:
// CHECK:     Swizzled constants     - count: 0, size: 0 bytes
// CHECK:     Not swizzled constants - count: 1, size: 14.84 KB
