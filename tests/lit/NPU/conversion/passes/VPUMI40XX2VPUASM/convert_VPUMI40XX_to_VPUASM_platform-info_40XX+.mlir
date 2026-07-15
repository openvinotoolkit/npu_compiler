//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-VPUMI40XX-to-VPUASM %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
config.ExecutorResource 1 of @DMA_NN
config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<16x32x1x1xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output_0" : tensor<16x32x1x1xf16, {order = #NHWC}>
    DataInfo "output_1" : tensor<16x32x1x1xf16, {order = #NHWC}>
    DataInfo "output_2" : tensor<16x32x1x1xf16, {order = #NHWC}>
  }
  VPUASM.InputBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<16x32x1x1xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  }
  VPUASM.OutputBindings outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<16x32x1x1xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @output_1_buffDecl !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<16x32x1x1xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @output_2_buffDecl !VPUASM.Buffer< "NetworkOutput"[2] <0> : memref<16x32x1x1xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  }
  VPUASM.ProfilingBindings profilingDeclarations : {
  }
  func.func nested @main() {
    %2 = VPUMI40XX.PlatformInfo -> <0:0:0>
    %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>
    %mi = VPUMI40XX.MappedInference dmaCount([[0, 0]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(0) bootstrapBarriersCount(0) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
    ELF.ABIVersion
    VPUMI40XX.OpRanges
  }
}

// CHECK: VPUASM.PlatformInfo
