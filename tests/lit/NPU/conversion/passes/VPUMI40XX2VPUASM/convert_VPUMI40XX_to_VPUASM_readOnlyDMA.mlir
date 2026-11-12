//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUMI40XX-to-VPUASM %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
  net.NetworkInfo entryPoint : @nndma_1d_to_1d inputsInfo : {
    DataInfo "input" : tensor<1x84x90x53xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x84x90x53xf16, {order = #NHWC}>
  }
  VPUASM.InputBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  }
  VPUASM.OutputBindings outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  }
  VPUASM.ProfilingBindings profilingDeclarations : {
  }
  func.func @nndma_1d_to_1d() {
    %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC}, @DDR>
    %3 = VPUMI40XX.NNDMA <{port = 0 : i64}> taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x84x90x53xf16, {order = #NHWC}, @DDR>)   start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    //CHECK: dma_descriptor(<numPlanes = 0 : i32, len = 801360 : i32, srcWidth = 801360 : i32, srcStride = 801360 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>)

    %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>
    %mi = VPUMI40XX.MappedInference dmas((%3) : (!VPURegMapped.Index<0:0:0>)) dmaCount([[1, 0]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) barrierCount(0) bootstrapBarriersCount(0) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>

    ELF.ABIVersion
    VPUMI40XX.OpRanges
  }
}
