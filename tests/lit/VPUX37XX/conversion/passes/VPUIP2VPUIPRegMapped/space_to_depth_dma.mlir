//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIP-to-VPUIPRegMapped %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @spaceToDepth attributes {VPU.compilationMode = "DefaultHW"} {
  func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, #NHWC, @DDR>) -> memref<1x1x16x256xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL0:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL1:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL0]] : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %5 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, mode = "BLOCKS_FIRST", port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.SpaceToDepth
    // CHECK: %[[VAL2:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL1]] : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %6 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, mode = "BLOCKS_FIRST", port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.SpaceToDepth
    // CHECK: %[[VAL3:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL2]] : !VPUIPRegMapped.Index<2>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<3>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL4:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL3]] : !VPUIPRegMapped.Index<3>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<4>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL5:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL4]] : !VPUIPRegMapped.Index<4>)  start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<5>

    return %arg1 : memref<1x1x16x256xf16, #NHWC, @DDR>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @spaceToDepth attributes {VPU.compilationMode = "DefaultHW"} {
  func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, #NHWC, @DDR>) -> memref<1x1x16x256xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %2 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, mode = "BLOCKS_FIRST", port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.SpaceToDepth
    // CHECK: %[[VAL0:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %3 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, mode = "BLOCKS_FIRST", port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.SpaceToDepth
    // CHECK: %[[VAL1:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL0]] : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL2:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL1]] : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %5 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL3:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL2]] : !VPUIPRegMapped.Index<2>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<3>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %6 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, mode = "BLOCKS_FIRST", port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.SpaceToDepth
    // CHECK: %[[VAL4:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL3]] : !VPUIPRegMapped.Index<3>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<4>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %7 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, mode = "BLOCKS_FIRST", port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.SpaceToDepth
    // CHECK: %[[VAL5:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL4]] : !VPUIPRegMapped.Index<4>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<5>

    return %arg1 : memref<1x1x16x256xf16, #NHWC, @DDR>
  }
}
