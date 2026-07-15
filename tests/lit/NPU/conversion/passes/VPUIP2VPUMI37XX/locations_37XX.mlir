//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s
// REQUIRES: platform-NPU3720

module @basicDMA {
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    VPURT.Task {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> loc("BasicDMA")
    }
    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
}
// CHECK: VPUMI37XX.NNDMA
// CHECK-SAME: loc([[LOC_OP:#.+]])
// CHECK: [[LOC_OP]] = loc("BasicDMA")

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @expandDMA {
  func.func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, {order = #NHWC}, @DDR>) -> memref<1x1x16x256xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x64x7x7xf16, {order = #NHWC}, @DDR>
    %1 = VPURT.DeclareBuffer <DDR> <3136> -> memref<1x64x7x16xf16, {order = #NHWC}, @DDR>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.ExpandDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 6272 : i64, srcWidth = 6272 : i64, srcStride = 6272 : i64, srcPlaneStride = 0 : i64, dstWidth = 896 : i64, dstStride = 2048 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9], port = 0 : i64}> inputs(%0 : memref<1x64x7x7xf16, {order = #NHWC}, @DDR>) outputs(%1 : memref<1x64x7x16xf16, {order = #NHWC}, @DDR>) -> memref<1x64x7x16xf16, {order = #NHWC}, @DDR> loc("ExpandDMA")
    }
    return %arg1 : memref<1x1x16x256xf16, {order = #NHWC}, @DDR>
  }
}
// CHECK-NOT: VPUIP.ExpandDMA
// CHECK: VPUMI37XX.NNDMA
// CHECK-SAME: loc([[LOC_OP:#.+]])
// CHECK: [[LOC_OP]] = loc("ExpandDMA")

// -----

module @permuteDMA {
  func.func @main(%arg0: memref<16x256xf16, @DDR>, %arg1: memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<16x256xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<256x16xf16, @DDR>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.PermuteDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64}> inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<256x16xf16, @DDR>) -> memref<256x16xf16, @DDR> loc("PermuteDMA")
    }
    return %arg1 : memref<16x256xf16, @DDR>
  }
}
// CHECK-NOT: VPUIP.PermuteDMA
// CHECK: VPUMI37XX.NNDMA
// CHECK-SAME: loc([[LOC_OP:#.+]])
// CHECK: [[LOC_OP]] = loc("PermuteDMA")

// -----

module @upsamplingDMA {
  func.func @main(%arg0: memref<16x256xf16, @DDR>, %arg1: memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<16x256xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<16x512xf16, @DDR>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.UpsamplingDMAOp <{upsampling_factor = [1, 2], dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64}> inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x512xf16, @DDR>) -> memref<16x512xf16, @DDR> loc("UpsamplingDMA")
    }
    return %arg1 : memref<16x256xf16, @DDR>
  }
}
// CHECK-NOT: VPUIP.UpsamplingDMAOp
// CHECK: VPUMI37XX.NNDMA
// CHECK-SAME: loc([[LOC_OP:#.+]])
// CHECK: [[LOC_OP]] = loc("UpsamplingDMA")

// -----

module @perAxisTileDMA {
  func.func @main(%arg0: memref<16x256xf16, @DDR>, %arg1: memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<16x256xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<16x256xf16, @DDR>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.PerAxisTileDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64}> inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR> loc("PerAxisTileDMA")
    }
    return %arg1 : memref<16x256xf16, @DDR>
  }
}
// CHECK-NOT: VPUIP.PerAxisTileDMA
// CHECK: VPUMI37XX.NNDMA
// CHECK-SAME: loc([[LOC_OP:#.+]])
// CHECK: [[LOC_OP]] = loc("PerAxisTileDMA")

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

 module @compressedDMA {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_143" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "Convolution_145" : tensor<8x1x1x1xui8>
  }
func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<8x1x1x1xui8, @DDR>) -> memref<8x1x1x1xui8, @DDR> {
  %cst = const.Declare memref<8x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<"0xDEADBEEFDEADBEEF"> : tensor<8x1x1x1xui8>
  %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
  %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<8x1x1x1xui8, @DDR>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<8x1x1x1xui8, [@CMX_NN, 0]>
  VPURT.Task attributes {isTrailingSWLayer = false} {
    %16 = VPUIP.DecompressDMAOp <{port = 0 : i64}> inputs(%cst : memref<8x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>) outputs(%3 : memref<8x1x1x1xui8, [@CMX_NN, 0]>) -> memref<8x1x1x1xui8, [@CMX_NN, 0]> loc("DecompressDMA")
  }
  return %arg1 : memref<8x1x1x1xui8, @DDR>
}
}
// CHECK-NOT: VPURT.Task
// CHECK: VPUMI37XX.NNDMA
// CHECK-SAME: loc([[LOC_OP:#.+]])
// CHECK: [[LOC_OP]] = loc("DecompressDMA")

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @spaceToDepth {
  func.func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, {order = #NHWC}, @DDR>) -> memref<1x1x16x256xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.SpaceToDepthDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]> loc("SpaceToDepthDMA")
    }
    return %arg1 : memref<1x1x16x256xf16, {order = #NHWC}, @DDR>
  }
}
// CHECK-NOT: VPUIP.SpaceToDepth
// CHECK: VPUMI37XX.NNDMA
// CHECK-SAME: loc([[LOC_OP:#.+]])
// CHECK: [[LOC_OP]] = loc("SpaceToDepthDMA")

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @depthToSpace {
  func.func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, {order = #NHWC}, @DDR>) -> memref<1x1x16x256xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.DepthToSpaceDMA <{block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64}> inputs(%0 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, {order = #NHWC}, [@CMX_NN, 0]> loc("DepthToSpaceDMA")
    }
    return %arg1 : memref<1x1x16x256xf16, {order = #NHWC}, @DDR>
  }
}
// CHECK-NOT: VPUIP.DepthToSpace
// CHECK: VPUMI37XX.NNDMA
// CHECK-SAME: loc([[LOC_OP:#.+]])
// CHECK: [[LOC_OP]] = loc("DepthToSpaceDMA")

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @nceMaxPool {
net.NetworkInfo entryPoint : @maxpool_f16_f16 inputsInfo : {
DataInfo "input_0" : tensor<1x64x16x16xf16>
} outputsInfo : {
DataInfo "output_0" : tensor<1x64x8x8xf16>
}

func.func nested @maxpool_f16_f16(%arg0: memref<1x64x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x64x8x8xf16, {order = #NHWC}, @DDR>) -> memref<1x64x8x8xf16, {order = #NHWC}, @DDR> {
  %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
  %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

  %cst = const.Declare memref<64x1x1x4xsi32, {order = #NHWC}, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]
  %cst_0 = const.Declare memref<1x1x1x16xui8, {order = #NHWC}, @DDR> = dense<[[[[3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]

  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %6 = VPURT.DeclareBuffer <CMX_NN> [0] <40960> -> memref<1x1x1x16xui8, {order = #NHWC}, [@CMX_NN, 0]>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <40976> -> memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>

  VPURT.Task updates(%0 : !VPURT.Barrier) {
      %8 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x64x16x16xf16, {order = #NHWC}, @DDR>) outputs(%2 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%0 : !VPURT.Barrier) {
      %8 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst_0 : memref<1x1x1x16xui8, {order = #NHWC}, @DDR>) outputs(%6 : memref<1x1x1x16xui8, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x1x1x16xui8, {order = #NHWC}, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%0 : !VPURT.Barrier) {
      %8 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst : memref<64x1x1x4xsi32, {order = #NHWC}, @DDR>) outputs(%7 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
  }
  VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %8 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>}> input(%2 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%4 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%5 : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
      DPUTask {outEnd = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
      } PPE : { PPETask { ppe = #VPU.PPEStub<> } } loc("NCEClusterTask")
  }
  VPURT.Task waits(%1 : !VPURT.Barrier) {
      %8 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%3 : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x64x8x8xf16, {order = #NHWC}, @DDR>) -> memref<1x64x8x8xf16, {order = #NHWC}, @DDR>
  }
  return %arg1 : memref<1x64x8x8xf16, {order = #NHWC}, @DDR>
}
}

// CHECK-LABEL: @maxpool_f16_f16

// CHECK-NOT: VPURT.Task
// CHECK: DPUInvariant
// CHECK: VPUMI37XX.PPETask
// CHECK: loc([[LOC_PPE:#.+]])
// CHECK: loc([[LOC_OP:#.+]])

// CHECK: VPUMI37XX.DPUVariant

// CHECK-NOT: VPURT.Task
// CHECK: VPUMI37XX.NNDMA
// CHECK-NOT: VPURT.Task
// CHECK: VPUMI37XX.NNDMA

// CHECK: [[LOC_OP]] = loc("NCEClusterTask")
