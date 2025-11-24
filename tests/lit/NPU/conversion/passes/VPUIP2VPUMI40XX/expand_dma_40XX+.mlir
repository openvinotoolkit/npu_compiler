//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @expandDMA {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x1x16x256xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x1x16x256xf16>
  }
  func.func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, #NHWC, @DDR>) -> memref<1x1x16x256xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x64x7x7xf16, #NHWC, @DDR>
    %1 = VPURT.DeclareBuffer <DDR> <3136> -> memref<1x64x7x16xf16, #NHWC, @DDR>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) -> memref<1x64x7x7xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[DMA0:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>)
    // CHECK:                dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x64x7x7xf16, #NHWC, @DDR>, outputType = memref<1x64x7x7xf16, #NHWC, @DDR>>){{.*}}-> !VPURegMapped.Index<0:0:0>


    VPURT.Task attributes {isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) -> memref<1x64x7x7xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[DMA1:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>)
    // CHECK:                outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) previousDMA(%[[DMA0]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>)
    // CHECK:                dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x64x7x7xf16, #NHWC, @DDR>, outputType = memref<1x64x7x7xf16, #NHWC, @DDR>>){{.*}}-> !VPURegMapped.Index<0:0:1>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 6272 : i64, srcWidth = 6272 : i64, srcStride = 6272 : i64, srcPlaneStride = 0 : i64, dstWidth = 896 : i64, dstStride = 2048 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) -> memref<1x64x7x16xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.ExpandDMA
    // CHECK: %[[DMA2:.*]] = VPUMI40XX.NNDMA {allow_different_in_out_shapes, port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>)
    //CHECK:                 outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) previousDMA(%[[DMA1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>)
    //CHECK:                 dma_transaction(#VPUMI40XX.ExpandDMATransaction<inputType = memref<1x64x7x7xf16, #NHWC, @DDR>, outputType = memref<1x64x7x16xf16, #NHWC, @DDR>, padsBegin = [0, 0, 0, 0], padsEnd = [0, 0, 0, 9]>){{.*}}-> !VPURegMapped.Index<0:0:2>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %6 = VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 6272 : i64, srcWidth = 6272 : i64, srcStride = 6272 : i64, srcPlaneStride = 0 : i64, dstWidth = 896 : i64, dstStride = 2048 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) -> memref<1x64x7x16xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.ExpandDMA
    // CHECK: %[[DMA3:.*]] = VPUMI40XX.NNDMA {allow_different_in_out_shapes, port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>)
    // CHECK:                outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) previousDMA(%[[DMA2]] : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>)
    // CHECK:                dma_transaction(#VPUMI40XX.ExpandDMATransaction<inputType = memref<1x64x7x7xf16, #NHWC, @DDR>, outputType = memref<1x64x7x16xf16, #NHWC, @DDR>, padsBegin = [0, 0, 0, 0], padsEnd = [0, 0, 0, 9]>){{.*}}-> !VPURegMapped.Index<0:0:3>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) -> memref<1x64x7x7xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[DMA4:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>)
    // CHECK:                outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) previousDMA(%[[DMA3]] : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>)
    // CHECK:                dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x64x7x7xf16, #NHWC, @DDR>, outputType = memref<1x64x7x7xf16, #NHWC, @DDR>>){{.*}}-> !VPURegMapped.Index<0:0:4>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) -> memref<1x64x7x7xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[DMA5:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>)
    // CHECK:                outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) previousDMA(%[[DMA4]] : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>)
    // CHECK:                dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x64x7x7xf16, #NHWC, @DDR>, outputType = memref<1x64x7x7xf16, #NHWC, @DDR>>){{.*}} -> !VPURegMapped.Index<0:0:5>

    return %arg1 : memref<1x1x16x256xf16, #NHWC, @DDR>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @expandDMA {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x1x16x256xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x1x16x256xf16>
  }
  func.func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, #NHWC, @DDR>) -> memref<1x1x16x256xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x64x7x7xf16, #NHWC, @DDR>
    %1 = VPURT.DeclareBuffer <DDR> <3136> -> memref<1x64x7x16xf16, #NHWC, @DDR>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %2 = VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 6272 : i64, srcWidth = 6272 : i64, srcStride = 6272 : i64, srcPlaneStride = 0 : i64, dstWidth = 896 : i64, dstStride = 2048 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) -> memref<1x64x7x16xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.ExpandDMA
    // CHECK: %[[DMA0:.*]] = VPUMI40XX.NNDMA {allow_different_in_out_shapes, port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>)
    // CHECK: outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>)
    // CHECK: dma_transaction(#VPUMI40XX.ExpandDMATransaction<inputType = memref<1x64x7x7xf16, #NHWC, @DDR>, outputType = memref<1x64x7x16xf16, #NHWC, @DDR>, padsBegin = [0, 0, 0, 0], padsEnd = [0, 0, 0, 9]>){{.*}}-> !VPURegMapped.Index<0:0:0>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %3 = VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 6272 : i64, srcWidth = 6272 : i64, srcStride = 6272 : i64, srcPlaneStride = 0 : i64, dstWidth = 896 : i64, dstStride = 2048 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) -> memref<1x64x7x16xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.ExpandDMA
    // CHECK: %[[DMA1:.*]] = VPUMI40XX.NNDMA {allow_different_in_out_shapes, port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>)
    // CHECK:  outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) previousDMA(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>)
    // CHECK:  dma_transaction(#VPUMI40XX.ExpandDMATransaction<inputType = memref<1x64x7x7xf16, #NHWC, @DDR>, outputType = memref<1x64x7x16xf16, #NHWC, @DDR>, padsBegin = [0, 0, 0, 0], padsEnd = [0, 0, 0, 9]>){{.*}}-> !VPURegMapped.Index<0:0:1>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) -> memref<1x64x7x7xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[DMA2:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>)
    // CHECK: outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) previousDMA(%[[DMA1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.*}}-> !VPURegMapped.Index<0:0:2>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) -> memref<1x64x7x7xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[DMA3:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) previousDMA(%[[DMA2]] : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.*}}-> !VPURegMapped.Index<0:0:3>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %6 = VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 6272 : i64, srcWidth = 6272 : i64, srcStride = 6272 : i64, srcPlaneStride = 0 : i64, dstWidth = 896 : i64, dstStride = 2048 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) -> memref<1x64x7x16xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.ExpandDMA
    // CHECK: %[[DMA4:.*]] = VPUMI40XX.NNDMA {allow_different_in_out_shapes, port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>)
    // CHECK: outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) previousDMA(%[[DMA3]] : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>)
    // CHECK: dma_transaction(#VPUMI40XX.ExpandDMATransaction<inputType = memref<1x64x7x7xf16, #NHWC, @DDR>, outputType = memref<1x64x7x16xf16, #NHWC, @DDR>, padsBegin = [0, 0, 0, 0], padsEnd = [0, 0, 0, 9]>){{.*}}-> !VPURegMapped.Index<0:0:4>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %7 = VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 6272 : i64, srcWidth = 6272 : i64, srcStride = 6272 : i64, srcPlaneStride = 0 : i64, dstWidth = 896 : i64, dstStride = 2048 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>) outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) -> memref<1x64x7x16xf16, #NHWC, @DDR>
    }

    // CHECK-NOT: VPUIP.ExpandDMA
    // CHECK: %[[DMA5:.*]] = VPUMI40XX.NNDMA {allow_different_in_out_shapes, port = 0 : i64} inputs(%0 : memref<1x64x7x7xf16, #NHWC, @DDR>)
    // CHECK: outputs(%1 : memref<1x64x7x16xf16, #NHWC, @DDR>) previousDMA(%[[DMA4]] : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>)
    // CHECK: dma_transaction(#VPUMI40XX.ExpandDMATransaction<inputType = memref<1x64x7x7xf16, #NHWC, @DDR>, outputType = memref<1x64x7x16xf16, #NHWC, @DDR>, padsBegin = [0, 0, 0, 0], padsEnd = [0, 0, 0, 9]>){{.*}}-> !VPURegMapped.Index<0:0:5>

    return %arg1 : memref<1x1x16x256xf16, #NHWC, @DDR>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @expandDMA {
net.NetworkInfo entryPoint : @UnrollDistributedExpandDMAOutput inputsInfo : {
  DataInfo "input_0" : tensor<1x16x16x16xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<64x32x1x1xf16>
}
func.func @UnrollDistributedExpandDMAOutput(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<64x32x1x1xf16, @DDR>) -> memref<64x32x1x1xf16, @DDR> {
  %cst = const.Declare memref<64x32x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG: %[[CST:.*]] = const.Declare memref<64x32x1x1xf16, #NHWC, @DDR>

  %3 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK-NOT: VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>

  VPURT.Task attributes {isTrailingSWLayer = false} {
    %16 = VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 4096 : i64, srcWidth = 4096 : i64, srcStride = 4096 : i64, srcPlaneStride = 0 : i64, dstWidth = 64 : i64, dstStride = 64 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i64} inputs(%cst : memref<64x32x1x1xf16, #NHWC, @DDR>) outputs(%3 : !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>) -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>
  }
  // CHECK: %[[BUFF_TILE_0:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK: %[[BUFF_TILE_1:.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
  // CHECK-NOT: VPURT.Task
  // CHECK: %[[DMA0:.*]] = VPUMI40XX.NNDMA {allow_different_in_out_shapes, port = 0 : i64}
  // CHECK:            inputs(%[[CST]] : memref<64x32x1x1xf16, #NHWC, @DDR>)
  // CHECK:            outputs(%[[BUFF_TILE_0]], %[[BUFF_TILE_1]] : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>, memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>)
  // CHECK:            start_after(0) clean_after(0) acceleration_mode(<DISABLE>)
  // CHECK: dma_transaction(#VPUMI40XX.ExpandDMATransaction<inputType = memref<64x32x1x1xf16, #NHWC, @DDR>, outputType = !VPUIP.DistributedBuffer<64x32x1x1xf16,
  // CHECK: {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>, padsBegin = [0, 0, 0, 0], padsEnd = [0, 0, 0, 9]>) {{.*}}-> !VPURegMapped.Index<0:0:0>

  return %arg1 : memref<64x32x1x1xf16, @DDR>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @expandDMA {
  net.NetworkInfo entryPoint : @UnrollDuplicatedExpandDMAOutput inputsInfo : {
    DataInfo "input_0" : tensor<1x1x2x3xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x2x3xf16>
  }
  func.func @UnrollDuplicatedExpandDMAOutput(%arg0: memref<1x1x2x3xf16, @DDR>, %arg1: memref<1x16x2x3xf16, @DDR>) -> memref<1x16x2x3xf16, @DDR> attributes {inliner_dispatch = #VPUIP.VPUIPInlinerDispatch} {
    %6 = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3] <0> -> !VPUIP.DistributedBuffer<1x16x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3] <192> -> !VPUIP.DistributedBuffer<1x16x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    %20 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %21 = VPURT.DeclareBuffer <NetworkInput> [1] <0> -> memref<1x1x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>

    VPURT.Task {
      %25 = VPUIP.ExpandDMA {is_out_of_order, pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0], port = 0 : i64}
            inputs(%20 : memref<1x1x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
            outputs(%6 : !VPUIP.DistributedBuffer<1x16x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64,
            uniform_distributed_segments, compute_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            memory_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
            !VPUIP.DistributedBuffer<1x16x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
            compute_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]],
            memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    }
    VPURT.Task {
      %25 = VPUIP.ExpandDMA {is_out_of_order, pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0], port = 0 : i64}
          inputs(%21 : memref<1x1x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
          outputs(%11 : !VPUIP.DistributedBuffer<1x16x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
          compute_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
          memory_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
          !VPUIP.DistributedBuffer<1x16x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]],
          compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    }

    %24 = VPUMI40XX.PlatformInfo -> <0:0:0>
    return %arg1 : memref<1x16x2x3xf16, @DDR>
  }

    // CHECK: VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x2x3xf16, #NHWC, @DDR>
    // CHECK: VPURT.DeclareBuffer <NetworkInput> [1] <0> -> memref<1x1x2x3xf16, #NHWC, @DDR>
    // CHECK: VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 1]>
    // CHECK: VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 2]>
    // CHECK: VPURT.DeclareBuffer <CMX_NN> [3] <0> -> memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 3]>
    // CHECK: VPUMI40XX.NNDMA {allow_different_in_out_shapes, is_out_of_order, port = 0 : i64}
    // CHECK: inputs({{%.+}} : memref<1x1x2x3xf16, #NHWC, @DDR>)
    // CHECK: outputs({{%.+}}, {{%.+}}, {{%.+}}, {{%.+}} : memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 1]>, memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 2]>, memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK: start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.ExpandDMATransaction<inputType = memref<1x1x2x3xf16, #NHWC, @DDR>, outputType = !VPUIP.DistributedBuffer<1x16x2x3xf16, #NHWC, @CMX_NN,
    // CHECK{LITERAL}: {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}: memory_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
    // CHECK: padsBegin = [0, 0, 0, 0], padsEnd = [0, 15, 0, 0]>) -> !VPURegMapped.Index<0:0:0>
    // CHECK: VPURT.DeclareBuffer <CMX_NN> [0] <192> -> memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: VPURT.DeclareBuffer <CMX_NN> [1] <192> -> memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 1]>
    // CHECK: VPURT.DeclareBuffer <CMX_NN> [2] <192> -> memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 2]>
    // CHECK: VPURT.DeclareBuffer <CMX_NN> [3] <192> -> memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 3]>
    // CHECK: VPUMI40XX.NNDMA {allow_different_in_out_shapes, is_out_of_order, port = 0 : i64}
    // CHECK: inputs(%1 : memref<1x1x2x3xf16, #NHWC, @DDR>)
    // CHECK: outputs(%7, %8, %9, %10 : memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 1]>, memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 2]>, memref<1x16x2x3xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK: previousDMA(%6 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>)
    // CHECK: dma_transaction(#VPUMI40XX.ExpandDMATransaction<inputType = memref<1x1x2x3xf16, #NHWC, @DDR>, outputType = !VPUIP.DistributedBuffer<1x16x2x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64,
    // CHECK{LITERAL}: uniform_distributed_segments, compute_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK{LITERAL}: memory_shapes = [[1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3], [1, 16, 2, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
    // CHECK: padsBegin = [0, 0, 0, 0], padsEnd = [0, 15, 0, 0]>) -> !VPURegMapped.Index<0:0:1>
}
