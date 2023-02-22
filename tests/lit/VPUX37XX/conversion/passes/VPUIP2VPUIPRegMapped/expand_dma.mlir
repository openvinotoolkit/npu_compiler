//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIP-to-VPUIPRegMapped %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @expandDMA attributes {VPU.compilationMode = "DefaultHW"} {
  func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, #NHWC, @DDR>) -> memref<1x1x16x256xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x64x7x7xf16, #NHWC, @DDR>
    %1 = VPURT.DeclareBuffer "DDR" <3136> -> memref<1x64x7x16xf16, #NHWC, @DDR>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) -> memref<1x64x7x7xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL0:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) -> memref<1x64x7x7xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL1:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) previousDMA(%[[VAL0]] : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %5 = VPUIP.ExpandDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2048 : i64, dstWidth = 896 : i64, len = 6272 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 6272 : i64, srcWidth = 6272 : i64}, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) -> memref<1x64x7x16xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.ExpandDMA
    // CHECK: %[[VAL2:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2048 : i64, dstWidth = 896 : i64, len = 6272 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 6272 : i64, srcWidth = 6272 : i64}, port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) previousDMA(%[[VAL1]] : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>
    
    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %6 = VPUIP.ExpandDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2048 : i64, dstWidth = 896 : i64, len = 6272 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 6272 : i64, srcWidth = 6272 : i64}, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) -> memref<1x64x7x16xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.ExpandDMA
    // CHECK: %[[VAL3:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2048 : i64, dstWidth = 896 : i64, len = 6272 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 6272 : i64, srcWidth = 6272 : i64}, port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) previousDMA(%[[VAL2]] : !VPUIPRegMapped.Index<2>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<3>
    
    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) -> memref<1x64x7x7xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL4:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) previousDMA(%[[VAL3]] : !VPUIPRegMapped.Index<3>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<4>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) -> memref<1x64x7x7xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL5:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) previousDMA(%[[VAL4]] : !VPUIPRegMapped.Index<4>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<5>

    return %arg1 : memref<1x1x16x256xf16, #NHWC, @DDR>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @expandDMA attributes {VPU.compilationMode = "DefaultHW"} {
  func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, #NHWC, @DDR>) -> memref<1x1x16x256xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x64x7x7xf16, #NHWC, @DDR>
    %1 = VPURT.DeclareBuffer "DDR" <3136> -> memref<1x64x7x16xf16, #NHWC, @DDR>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %2 = VPUIP.ExpandDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2048 : i64, dstWidth = 896 : i64, len = 6272 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 6272 : i64, srcWidth = 6272 : i64}, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i644} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) -> memref<1x64x7x16xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.ExpandDMA
    // CHECK: %[[VAL0:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2048 : i64, dstWidth = 896 : i64, len = 6272 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 6272 : i64, srcWidth = 6272 : i64}, port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %3 = VPUIP.ExpandDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2048 : i64, dstWidth = 896 : i64, len = 6272 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 6272 : i64, srcWidth = 6272 : i64}, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i644} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) -> memref<1x64x7x16xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.ExpandDMA
    // CHECK: %[[VAL1:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2048 : i64, dstWidth = 896 : i64, len = 6272 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 6272 : i64, srcWidth = 6272 : i64}, port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) previousDMA(%2 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) -> memref<1x64x7x7xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL2:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) previousDMA(%[[VAL1]] : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %5 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) -> memref<1x64x7x7xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL3:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) previousDMA(%[[VAL2]] : !VPUIPRegMapped.Index<2>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<3>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %6 = VPUIP.ExpandDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2048 : i64, dstWidth = 896 : i64, len = 6272 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 6272 : i64, srcWidth = 6272 : i64}, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) -> memref<1x64x7x16xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.ExpandDMA
    // CHECK: %[[VAL4:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2048 : i64, dstWidth = 896 : i64, len = 6272 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 6272 : i64, srcWidth = 6272 : i64}, port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) previousDMA(%[[VAL3]] : !VPUIPRegMapped.Index<3>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<4>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %7 = VPUIP.ExpandDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2048 : i64, dstWidth = 896 : i64, len = 6272 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 6272 : i64, srcWidth = 6272 : i64}, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) -> memref<1x64x7x16xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.ExpandDMA
    // CHECK: %[[VAL5:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2048 : i64, dstWidth = 896 : i64, len = 6272 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 6272 : i64, srcWidth = 6272 : i64}, port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) previousDMA(%[[VAL4]] : !VPUIPRegMapped.Index<4>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<5>

    return %arg1 : memref<1x1x16x256xf16, #NHWC, @DDR>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @expandDMA {
func @UnrollDistributedExpandDMAOutput(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<64x32x1x1xf16, @DDR>) -> memref<64x32x1x1xf16, @DDR> {
  %cst = const.Declare memref<64x32x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK: %[[CST:.*]] = const.Declare memref<64x32x1x1xf16, #NHWC, @DDR>

  %3 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  // CHECK: %[[WEIGHTS:.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

  VPURT.Task attributes {cycleBegin = 1410 : i64, cycleEnd = 1613 : i64, isTrailingSWLayer = false} {
    %16 = VPUIP.ExpandDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 64 : i64, dstWidth = 64 : i64, len = 4096 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 4096 : i64, srcWidth = 4096 : i64}, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i64} inputs(%cst : memref<64x32x1x1xf16, #NHWC, @DDR>) outputs(%3 : !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  }
  // CHECK: %[[BUFF_TILE_0:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK: %[[BUFF_TILE_1:.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
  // CHECK-NOT: VPURT.Task
  // CHECK: %[[DMA0:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 64 : i64, dstWidth = 64 : i64, len = 4096 : i64, numPlanes = 1 : i64, srcPlaneStride = 0 : i64, srcStride = 4096 : i64, srcWidth = 4096 : i64}, port = 0 : i64} inputs(%[[CST]] : memref<64x32x1x1xf16, #NHWC, @DDR>) outputs(%[[BUFF_TILE_0]], %[[BUFF_TILE_1]] : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>, memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
  return %arg1 : memref<64x32x1x1xf16, @DDR>
}
}
