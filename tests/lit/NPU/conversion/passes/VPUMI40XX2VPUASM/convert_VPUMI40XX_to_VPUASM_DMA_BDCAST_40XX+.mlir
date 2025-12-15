//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-VPUMI40XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @dma_broadcast inputsInfo : {
    DataInfo "input_0" : tensor<16x32x1x1xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output_0" : tensor<16x32x1x1xf16, {order = #NHWC}>
    DataInfo "output_1" : tensor<16x32x1x1xf16, {order = #NHWC}>
    DataInfo "output_2" : tensor<16x32x1x1xf16, {order = #NHWC}>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<16x32x1x1xf16, #NHWC, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<16x32x1x1xf16, #NHWC, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @output_1_buffDecl !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<16x32x1x1xf16, #NHWC, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @output_2_buffDecl !VPUASM.Buffer< "NetworkOutput"[2] <0> : memref<16x32x1x1xf16, #NHWC, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @dma_broadcast() {
    %320 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %352 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:1:0>
    %353 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:1:1>
    %354 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:1:2>

    %2304 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<16x32x1x1xf16, #NHWC, @DDR>
    %2305 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<16x32x1x1xf16, #NHWC, @DDR>
    %2306 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> {swizzlingKey = 0 : i64} -> memref<16x32x1x1xf16, #NHWC, @DDR>
    %2307 = VPURT.DeclareBuffer <NetworkOutput> [2] <0> {swizzlingKey = 0 : i64} -> memref<16x32x1x1xf16, #NHWC, @DDR>
    %2308 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %2309 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 2]>
    %2310 = VPURT.DeclareBuffer <CMX_NN> [4] <0> -> memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 4]>
    %2311 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %2312 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 2]>
    %2313 = VPURT.DeclareBuffer <CMX_NN> [4] <0> -> memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 4]>
    %2315 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%320 : !VPURegMapped.Index<0:0:0>) inputs(%2304 : memref<16x32x1x1xf16, #NHWC, @DDR>) outputs(%2308, %2309, %2310 : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>, memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 2]>, memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 4]>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %2316 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%352 : !VPURegMapped.Index<0:1:0>) inputs(%2311 : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2305 : memref<16x32x1x1xf16, #NHWC, @DDR>) start_after(1) clean_after(1) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %2317 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%353 : !VPURegMapped.Index<0:1:1>) inputs(%2312 : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 2]>) outputs(%2306 : memref<16x32x1x1xf16, #NHWC, @DDR>) previousDMA(%2316 : !VPURegMapped.Index<0:1:0>) start_after(1) clean_after(1) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>
    %2318 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%354 : !VPURegMapped.Index<0:1:2>) inputs(%2313 : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 4]>) outputs(%2307 : memref<16x32x1x1xf16, #NHWC, @DDR>) previousDMA(%2317 : !VPURegMapped.Index<0:1:1>) start_after(1) clean_after(1) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:2>
    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}


//CHECK:    VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_0_0) input(@io.NetworkInput.0::@DeclareBuffer_0) outputs([@buffer.CMX_NN.0::@DeclareBuffer_4, @buffer.CMX_NN.2::@DeclareBuffer_5, @buffer.CMX_NN.4::@DeclareBuffer_6]) waits([]) updates([]) start_after(1) clean_after(0) dma_descriptor(<numPlanes = 0 : i32, len = 1024 : i32, srcWidth = 1024 : i32, srcStride = 1024 : i32, srcPlaneStride = 0 : i32, dstWidth = 1024 : i32, dstStride = 1024 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>) tile_indexes([0, 2, 4])
//CHECK:    VPUASM.NNDMA @NNDMA_0_1_0 idx(!VPURegMapped.Index<0:1:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_0) links(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_1) input(@buffer.CMX_NN.0::@DeclareBuffer_7) outputs([@io.NetworkOutput.0::@DeclareBuffer_1]) waits([]) updates([]) start_after(1) clean_after(1) dma_descriptor(<numPlanes = 0 : i32, len = 1024 : i32, srcWidth = 1024 : i32, srcStride = 1024 : i32, srcPlaneStride = 0 : i32, dstWidth = 1024 : i32, dstStride = 1024 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)
//CHECK:    VPUASM.NNDMA @NNDMA_0_1_1 idx(!VPURegMapped.Index<0:1:1>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_1) links(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_2) input(@buffer.CMX_NN.2::@DeclareBuffer_8) outputs([@io.NetworkOutput.1::@DeclareBuffer_2]) waits([]) updates([]) start_after(1) clean_after(1) dma_descriptor(<numPlanes = 0 : i32, len = 1024 : i32, srcWidth = 1024 : i32, srcStride = 1024 : i32, srcPlaneStride = 0 : i32, dstWidth = 1024 : i32, dstStride = 1024 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)
//CHECK:    VPUASM.NNDMA @NNDMA_0_1_2 idx(!VPURegMapped.Index<0:1:2>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_2) input(@buffer.CMX_NN.4::@DeclareBuffer_9) outputs([@io.NetworkOutput.2::@DeclareBuffer_3]) waits([]) updates([]) start_after(1) clean_after(1) dma_descriptor(<numPlanes = 0 : i32, len = 1024 : i32, srcWidth = 1024 : i32, srcStride = 1024 : i32, srcPlaneStride = 0 : i32, dstWidth = 1024 : i32, dstStride = 1024 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)
