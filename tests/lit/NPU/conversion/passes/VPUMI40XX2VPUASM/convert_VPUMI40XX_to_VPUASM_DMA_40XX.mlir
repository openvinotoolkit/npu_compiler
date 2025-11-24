//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-VPUMI40XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @nndma_4d_to_4d_with_single_shape inputsInfo : {
    DataInfo "input" : tensor<2x2x2x2xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<2x2x2x2xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<2x2x2x2xf16, {order = #NCHW, strides = [256, 64, 16, 1]}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<2x2x2x2xf16, {order = #NCHW, strides = [256, 64, 16, 1]}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_4d_to_4d_with_single_shape() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<2x2x2x2xf16, {order = #NCHW, strides = [256, 64, 16, 1]}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<2x2x2x2xf16, {order = #NCHW, strides = [256, 64, 16, 1]}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<2x2x2x2xf16, {order = #NCHW, strides = [256, 64, 16, 1]}, @DDR>)
        outputs(%2 : memref<2x2x2x2xf16, {order = #NCHW, strides = [256, 64, 16, 1]}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>)
        dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<2x2x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 64, 16, 1]}, @DDR>, outputType = memref<2x2x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 64, 16, 1]}, @DDR>>)
        -> !VPURegMapped.Index<0:0:0>

    // CHECK:       ELF.CreateSection @task.dma.0.0
      // CHECK:       VPUASM.NNDMA
      // CHECK-SAME:  dma_transaction
      // CHECK-NOT:   dma_descriptor

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {config.arch = #config.arch_kind<NPU40XX>} {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @ConvertDMAWithF32ToF16 inputsInfo : {
    DataInfo "input" : tensor<1x320x3x103xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x320x3x103xf16>
  }

  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x320x3x103xf32, {order = #NHWC, strides = [2764800, 1, 921600, 1280]}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x320x3x103xf16, #NHWC, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }

  func.func private @ConvertDMAWithF32ToF16() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x320x3x103xf32, {order = #NHWC, strides = [2764800, 1, 921600, 1280]}, @DDR>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x320x3x103xf16, #NHWC, [@CMX_NN, 0]>

    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
         inputs(%1 : memref<1x320x3x103xf32, {order = #NHWC, strides = [2764800, 1, 921600, 1280]}, @DDR>)
         outputs(%2 : memref<1x320x3x103xf16, #NHWC, [@CMX_NN, 0]>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges

    // CHECK:       ELF.CreateSection @task.dma.0.0
    // CHECK:       VPUASM.NNDMA @NNDMA_0_0_0
    // CHECK-SAME:  dma_descriptor(<
    // CHECK-SAME:      numPlanes = 3 : i32, len = 131840 : i32
    // CHECK-SAME:      srcWidth = 1280 : i32, srcStride = 5120 : i32, srcPlaneStride = 3686400 : i32
    // CHECK-SAME:      dstWidth = 65920 : i32, dstStride = 65920 : i32, dstPlaneStride = 65920 : i32>
  }
}
