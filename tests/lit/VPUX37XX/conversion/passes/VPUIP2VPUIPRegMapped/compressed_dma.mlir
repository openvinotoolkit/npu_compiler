//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIP-to-VPUIPRegMapped %s | FileCheck %s

 module @compressedDMA {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Parameter_143" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "Convolution_145" : tensor<8x1x1x1xui8>
  }
func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<8x1x1x1xui8, @DDR>) -> memref<8x1x1x1xui8, @DDR> {
  %cst = const.Declare memref<8x1x1x1xui8> = dense<"0xDEADBEEFDEADBEEF"> : tensor<8x1x1x1xui8>
  // CHECK: %[[CST:.*]] = const.Declare memref<8x1x1x1xui8>

  %1 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x16x16x16xf16, @DDR>
  // CHECK: %[[IN:.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x16x16x16xf16, @DDR>

  %2 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<8x1x1x1xui8, @DDR>
  // CHECK: %[[OUT:.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<8x1x1x1xui8, @DDR>

  %3 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<8x1x1x1xui8, [@CMX_NN, 0]>
  // CHECK: %[[COMPRESSED_WEIGHTS:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<8x1x1x1xui8, [@CMX_NN, 0]>

  VPURT.Task attributes {cycleBegin = 1410 : i64, cycleEnd = 1613 : i64, isTrailingSWLayer = false} {
    %16 = VPUIP.CompressedDMAOp {port = 0 : i64} inputs(%cst : memref<8x1x1x1xui8>) outputs(%3 : memref<8x1x1x1xui8, [@CMX_NN, 0]>) -> memref<8x1x1x1xui8, [@CMX_NN, 0]>
  }
  // CHECK-NOT: VPURT.Task
  // CHECK: %[[DMA0:.*]] = VPUIPRegMapped.NNDMA {compression, port = 0 : i64} inputs(%[[CST]] : memref<8x1x1x1xui8>) outputs(%[[COMPRESSED_WEIGHTS]] : memref<8x1x1x1xui8, [@CMX_NN, 0]>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>

  VPURT.Task attributes {cycleBegin = 2104 : i64, cycleEnd = 2170 : i64, isTrailingSWLayer = false} {
    %16 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<8x1x1x1xui8, [@CMX_NN, 0]>) outputs(%2 : memref<8x1x1x1xui8, @DDR>) -> memref<8x1x1x1xui8, @DDR>
  }
  // CHECK-NOT: VPURT.Task
  // CHECK: %[[DMA1:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[COMPRESSED_WEIGHTS]] : memref<8x1x1x1xui8, [@CMX_NN, 0]>) outputs(%[[OUT]] : memref<8x1x1x1xui8, @DDR>) previousDMA(%[[DMA0]] : !VPUIPRegMapped.Index<0>)  start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>

  return %arg1 : memref<8x1x1x1xui8, @DDR>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
 module @compressedDMA_2 {
func @UnrollDistributedCompressedDMAOutput(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<64x32x1x1xf16, @DDR>) -> memref<64x32x1x1xf16, @DDR> {
  %cst = const.Declare memref<64x32x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK: %[[CST:.*]] = const.Declare memref<64x32x1x1xf16, #NHWC, @DDR>

  %3 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  // CHECK: %[[COMPRESSED_WEIGHTS:.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

  VPURT.Task attributes {cycleBegin = 1410 : i64, cycleEnd = 1613 : i64, isTrailingSWLayer = false} {
    %16 = VPUIP.CompressedDMAOp {port = 0 : i64} inputs(%cst : memref<64x32x1x1xf16, #NHWC, @DDR>) outputs(%3 : !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  }
  // CHECK: %[[BUFF_TILE_0:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK: %[[BUFF_TILE_1:.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
  // CHECK-NOT: VPURT.Task
  // CHECK: %[[DMA0:.*]] = VPUIPRegMapped.NNDMA {compression, port = 0 : i64} inputs(%[[CST]] : memref<64x32x1x1xf16, #NHWC, @DDR>) outputs(%[[BUFF_TILE_0]], %[[BUFF_TILE_1]] : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>, memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>

  return %arg1 : memref<64x32x1x1xf16, @DDR>
}
}
