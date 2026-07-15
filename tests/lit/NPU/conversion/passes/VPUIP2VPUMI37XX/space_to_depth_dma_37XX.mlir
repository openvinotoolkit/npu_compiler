//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @spaceToDepth {
  func.func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, {order = #NHWC}, @DDR>) -> memref<1x1x16x256xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %3 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.SpaceToDepthDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %6 = VPUIP.SpaceToDepthDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %7 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    return %arg1 : memref<1x1x16x256xf16, {order = #NHWC}, @DDR>
  }
}

// CHECK: [[BUF_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
// CHECK: [[BUF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
// CHECK-NOT: VPUIP.NNDMA
// CHECK: [[VAL5:%.+]] = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:5>
// CHECK-NOT: VPUIP.NNDMA
// CHECK: [[VAL4:%.+]] = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx([[VAL5]] : !VPURegMapped.Index<0:0:5>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>
// CHECK-NOT: VPUIP.SpaceToDepthDMA
// CHECK: [[VAL3:%.+]] = VPUMI37XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, port = 0 : i64}> inputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_1]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx([[VAL4]] : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
// CHECK-NOT: VPUIP.SpaceToDepthDMA
// CHECK: [[VAL2:%.+]] = VPUMI37XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, port = 0 : i64}> inputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_1]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx([[VAL3]] : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
// CHECK-NOT: VPUIP.NNDMA
// CHECK: [[VAL1:%.+]] = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx([[VAL2]] : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
// CHECK-NOT: VPUIP.NNDMA
// CHECK: [[VAL0:%.+]] = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx([[VAL1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @spaceToDepth {
  func.func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, {order = #NHWC}, @DDR>) -> memref<1x1x16x256xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %2 = VPUIP.SpaceToDepthDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %3 = VPUIP.SpaceToDepthDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %6 = VPUIP.SpaceToDepthDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %7 = VPUIP.SpaceToDepthDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    return %arg1 : memref<1x1x16x256xf16, {order = #NHWC}, @DDR>
  }
}

// CHECK: [[BUF_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
// CHECK: [[BUF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
// CHECK-NOT: VPUIP.SpaceToDepthDMA
// CHECK: [[VAL5:%.+]] = VPUMI37XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, port = 0 : i64}> inputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_1]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:5>
// CHECK-NOT: VPUIP.SpaceToDepthDMA
// CHECK: [[VAL4:%.+]] = VPUMI37XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, port = 0 : i64}> inputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_1]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx([[VAL5]] : !VPURegMapped.Index<0:0:5>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>
// CHECK-NOT: VPUIP.NNDMA
// CHECK: [[VAL3:%.+]] = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx([[VAL4]] : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
// CHECK-NOT: VPUIP.NNDMA
// CHECK: [[VAL2:%.+]] = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx([[VAL3]] : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
// CHECK-NOT: VPUIP.SpaceToDepthDMA
// CHECK: [[VAL1:%.+]] = VPUMI37XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, port = 0 : i64}> inputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_1]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx([[VAL2]] : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
// CHECK-NOT: VPUIP.SpaceToDepthDMA
// CHECK: [[VAL0:%.+]] = VPUMI37XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, port = 0 : i64}> inputs([[BUF_0]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_1]] : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx([[VAL1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
