//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-VPUMI40XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
IE.ExecutorResource 1 of @DMA_NN
IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @main inputsInfo : {
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
  func.func private @main() {
    %2 = VPUMI40XX.PlatformInfo -> <0:0:0>
    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

// CHECK: VPUASM.PlatformInfo
