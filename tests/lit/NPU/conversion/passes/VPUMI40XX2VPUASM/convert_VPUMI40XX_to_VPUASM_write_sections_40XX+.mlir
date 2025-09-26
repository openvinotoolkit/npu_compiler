//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-VPUMI40XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @Test attributes {config.arch = #config.arch_kind<NPU40XX>} {
  config.ExecutorResource 1 of @DMA_NN
  config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "softmax" : tensor<1x1000xf16>
  }
  module @VPU.SW {
    func.func private @builtin_softmax(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x1x1000xf16> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @softmax_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1x1x1000xf16> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    %tb_dma_00 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %tb_dma_10 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:1:0>
    %tb_actkerrange_0 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:0>
    %tb_actkerinv_0 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:0>
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x1x1x1000xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x1x1x1000xf16, @DDR>
    %2 = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 0 : i64} -> memref<1048576xi8, @DDR>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %5 = VPUMI40XX.DeclareKernelText kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %6 = VPUMI40XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %7 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %8 = VPUMI40XX.KernelParams inputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%4 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([0, 0, 32, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %11 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%tb_dma_00 : !VPURegMapped.Index<0:0:0>) inputs(%0 : memref<1x1x1x1000xf16, @DDR>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %12 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%tb_dma_10 : !VPURegMapped.Index<0:1:0>) inputs(%4 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x1x1000xf16, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %13 = VPUMI40XX.ActKernelRange taskLocation(%tb_actkerrange_0 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%5 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%7 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%6 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %14 = VPUMI40XX.ActKernelInvocation taskLocation(%tb_actkerinv_0 : !VPURegMapped.Index<0:0:0>) range_index(%13 : <0:0:0>) kernel_params(%8 : <0:0:0>) waits(%9 : !VPURegMapped.Index<0:0:0>) updates(%10 : !VPURegMapped.Index<0:0:1>) tile(0) start_after(2) clean_after(1) -> !VPURegMapped.Index<0:0:0>

    %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>

    VPUMI40XX.MappedInference dmas((%11, %12) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) actKernelRanges((%13): (!VPURegMapped.Index<0:0:0>)) actKernelInvocations((%14): (!VPURegMapped.Index<0:0:0>)) barriers(%9: !VPURegMapped.Index<0:0:0>) dmaCount([[1, 1]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([[1, 0]]) actKernelInvocationsCount([[1, 0]]) mediaCount(0) barrierCount(2) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    ELF.CompilerHash("0123456789abcdef0123456789abcdef01234567") {sym_name = "CompilerHash"}
    VPUMI40XX.OpRanges
  }
}

// CHECK: ELF.CreateLogicalSection @buffer.DDR.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC")
// CHECK: ELF.CreateSection @shave.data aligned(64) secType(SHT_PROGBITS) secFlags("SHF_WRITE|SHF_ALLOC")
