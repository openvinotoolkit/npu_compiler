//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --update-ELFNPU37XX-section-flags %s | FileCheck %s
// REQUIRES: platform-NPU3720

module @Test {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "softmax" : tensor<1x1000xf16>
  }
  module @VPU.SW {
    func.func nested @builtin_softmax(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
  }
  func.func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %2 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %3 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %5 = VPUMI37XX.DeclareKernelText kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %6 = VPUMI37XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %7 = VPUMI37XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %8 = VPUMI37XX.ActKernelRange kernel_text_index(%5 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%7 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.ActKernelInvocation range_index(%8 : <0:0:0>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%3 : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(1) -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI37XX.KernelParams inputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : vector<80xui8>) -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16>) waits(%3 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %4 = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) nextDMAIdx(%11 : !VPURegMapped.Index<0:0:1>) updates(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %12 = VPUMI37XX.MappedInference dmas(%4 : !VPURegMapped.Index<0:0:0>) actKernelRanges(%8 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%9 : !VPURegMapped.Index<0:0:0>) barriers(%2 : !VPURegMapped.Index<0:0:0>) dmaCount([2, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(2) -> !VPURegMapped.Index<0:0:0>
    %13 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %4 : !VPURegMapped.Index<0:0:0>
      ELFNPU37XX.PutOpInSection %11 : !VPURegMapped.Index<0:0:1>
    }
    // CHECK:  [[VAL1:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELFNPU37XX.Section {
    %14 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %2 : !VPURegMapped.Index<0:0:0>
      ELFNPU37XX.PutOpInSection %3 : !VPURegMapped.Index<0:0:1>
    }
    // CHECK:  [[VAL2:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELFNPU37XX.Section {
    %15 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelText"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %5 : !VPURegMapped.Index<0:0:0>
    }
    // CHECK:  [[VAL3:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelText"} -> !ELFNPU37XX.Section {
    %16 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelData"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %6 : !VPURegMapped.Index<0:0:0>
    }
    // CHECK:  [[VAL4:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelData"} -> !ELFNPU37XX.Section {
    %17 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelParams"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %10 : !VPURegMapped.Index<0:0:0>
    }
    // CHECK:  [[VAL5:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelParams"} -> !ELFNPU37XX.Section {
    %18 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelRanges"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %8 : !VPURegMapped.Index<0:0:0>
    }
    // CHECK:  [[VAL6:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelRanges"} -> !ELFNPU37XX.Section {
    %19 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelInvocations"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %9 : !VPURegMapped.Index<0:0:0>
    }
    // CHECK:  [[VAL7:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelInvocations"} -> !ELFNPU37XX.Section {
    %20 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %12 : !VPURegMapped.Index<0:0:0>
    }
    // CHECK:  [[VAL8:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELFNPU37XX.Section {
    %21 = ELFNPU37XX.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELFNPU37XX.Section {
      %55 = VPUMI37XX.NetworkMetadata -> !VPURegMapped.Index<0:0:0>
    }
    %22 = ELFNPU37XX.Symbol %13 name("sym_dmaSection0") : !ELFNPU37XX.Section
    %23 = ELFNPU37XX.Symbol %14 name("sym_barrierSection") : !ELFNPU37XX.Section
    %24 = ELFNPU37XX.Symbol %18 name("sym_actKernelRangeSection") : !ELFNPU37XX.Section
    %25 = ELFNPU37XX.Symbol %19 name("sym_actKernelInvo") : !ELFNPU37XX.Section
    %26 = ELFNPU37XX.Symbol %15 name("sym_kernelTextSection") : !ELFNPU37XX.Section
    %27 = ELFNPU37XX.Symbol %16 name("sym_kernelDataSection") : !ELFNPU37XX.Section
    %28 = ELFNPU37XX.Symbol %17 name("sym_kernelParamsSection") : !ELFNPU37XX.Section
    %29 = ELFNPU37XX.Symbol %arg0 name("input") size(2000) : memref<1x1x1x1000xf16>
    %30 = ELFNPU37XX.Symbol %arg1 name("softmax") size(2000) : memref<1x1x1x1000xf16>
    %31 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %29 : !ELFNPU37XX.Symbol
    }
    %32 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %30 : !ELFNPU37XX.Symbol
    }
    %c0_i8 = arith.constant 0 : i8
    %33 = ELFNPU37XX.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") <{isBuiltin}> : i8
    %c1_i8 = arith.constant 1 : i8
    %34 = ELFNPU37XX.Symbol %c1_i8 name("VPU_NNRD_SYM_RTM_IVAR") <{isBuiltin}> : i8
    %c2_i8 = arith.constant 2 : i8
    %35 = ELFNPU37XX.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT") <{isBuiltin}> : i8
    %c3_i8 = arith.constant 3 : i8
    %36 = ELFNPU37XX.Symbol %c3_i8 name("VPU_NNRD_SYM_RTM_DMA0") <{isBuiltin}> : i8
    %c4_i8 = arith.constant 4 : i8
    %37 = ELFNPU37XX.Symbol %c4_i8 name("VPU_NNRD_SYM_RTM_DMA1") <{isBuiltin}> : i8
    %c5_i8 = arith.constant 5 : i8
    %38 = ELFNPU37XX.Symbol %c5_i8 name("VPU_NNRD_SYM_FIFO_BASE") <{isBuiltin}> : i8
    %c6_i8 = arith.constant 6 : i8
    %39 = ELFNPU37XX.Symbol %c6_i8 name("VPU_NNRD_SYM_BARRIERS_START") <{isBuiltin}> : i8
    %40 = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") <{isBuiltin}> -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %33 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %34 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %35 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %36 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %37 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %38 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %39 : !ELFNPU37XX.Symbol
    }
    %41 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %22 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %23 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %26 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %27 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %28 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %24 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %25 : !ELFNPU37XX.Symbol
      %55 = ELFNPU37XX.Symbol %12 name("MappedInference_entry") type(<VPU_STT_ENTRY>) : !VPURegMapped.Index<0:0:0>
    }
    %42 = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetInput") sourceSymbolTableSection(%31) targetSection(%13) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_64> %29 0
    }
    %43 = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput") sourceSymbolTableSection(%32) targetSection(%13) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%11 : !VPURegMapped.Index<0:0:1>) offset(24) <R_VPU_64> %30 0
    }
    %44 = ELFNPU37XX.CreateRelocationSection secName(".rlt.dmaIO_CMX") sourceSymbolTableSection(%40) targetSection(%13) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(24) <R_VPU_64> %33 0
      ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(0) <R_VPU_32_RTM> %36 192
      ELFNPU37XX.Reloc baseOp(%11 : !VPURegMapped.Index<0:0:1>) offset(16) <R_VPU_64> %33 2000
    }
    %45 = ELFNPU37XX.CreateRelocationSection secName(".rlt.KernelParams") sourceSymbolTableSection(%41) targetSection(%17) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(12) <R_VPU_32> %28 80
      ELFNPU37XX.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(48) <R_VPU_32> %28 96
      ELFNPU37XX.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_32> %28 112
      ELFNPU37XX.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(52) <R_VPU_32> %28 144
    }
    %46 = ELFNPU37XX.CreateRelocationSection secName(".rlt.KernelParamsIO_CMX") sourceSymbolTableSection(%40) targetSection(%17) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(0) <R_VPU_32> %33 0
      ELFNPU37XX.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(36) <R_VPU_32> %33 2000
    }
    %47 = ELFNPU37XX.CreateRelocationSection secName(".rlt.ActKernelRange") sourceSymbolTableSection(%41) targetSection(%18) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%8 : !VPURegMapped.Index<0:0:0>) offset(8) <R_VPU_32> %26 0
    }
    %48 = ELFNPU37XX.CreateRelocationSection secName(".rlt.ActKernelInvo") sourceSymbolTableSection(%41) targetSection(%19) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%9 : !VPURegMapped.Index<0:0:0>) offset(0) <R_VPU_32> %24 0
      ELFNPU37XX.Reloc baseOp(%9 : !VPURegMapped.Index<0:0:0>) offset(8) <R_VPU_32> %27 0
      ELFNPU37XX.Reloc baseOp(%9 : !VPURegMapped.Index<0:0:0>) offset(4) <R_VPU_32> %28 0
    }
    %49 = VPURT.DeclareBuffer <DDR> <64> -> memref<262144xi32, @DDR>
    %50 = ELFNPU37XX.CreateLogicalSection secType(SHT_NOBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actKernelRtConfigSec"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %49 : memref<262144xi32, @DDR>
    }
    %51 = ELFNPU37XX.Symbol %50 name("sym_actKernelRtConfigsSec") : !ELFNPU37XX.Section
    %52 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.actKernelRtConfig") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %51 : !ELFNPU37XX.Symbol
    }
    %53 = ELFNPU37XX.CreateRelocationSection secName(".rlt.MI_AKRtConfig") sourceSymbolTableSection(%52) targetSection(%20) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%12 : !VPURegMapped.Index<0:0:0>) offset(828) <R_VPU_32> %51 0
    }
    %54 = ELFNPU37XX.CreateRelocationSection secName(".rlt.MappedInference") sourceSymbolTableSection(%41) targetSection(%20) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%12 : !VPURegMapped.Index<0:0:0>) offset(0) <R_VPU_64> %22 0
      ELFNPU37XX.Reloc baseOp(%12 : !VPURegMapped.Index<0:0:0>) offset(72) <R_VPU_64> %23 0
      ELFNPU37XX.Reloc baseOp(%12 : !VPURegMapped.Index<0:0:0>) offset(768) <R_VPU_64> %24 0
      ELFNPU37XX.Reloc baseOp(%12 : !VPURegMapped.Index<0:0:0>) offset(784) <R_VPU_64> %25 0
    }
    return %arg1 : memref<1x1x1x1000xf16>
  }
}

// -----

module @Test {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter" : tensor<1x100xui32>
  } outputsInfo : {
    DataInfo "Convert" : tensor<1x100xui32>
  }
  func.func @main(%arg0: memref<1x100xui32, @DDR>, %arg1: memref<1x100xui32, @DDR>) -> memref<1x100xui32, @DDR> {
    %0 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x100xui32, @DDR>
    %cst = const.Declare memref<1x100xui32> = dense<1> : tensor<1x100xui32>
    %3 = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs(%1 : memref<1x100xui32, @DDR>) outputs(%arg1 : memref<1x100xui32, @DDR>) waits(%0 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %2 = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs(%cst : memref<1x100xui32>) outputs(%1 : memref<1x100xui32, @DDR>) nextDMAIdx(%3 : !VPURegMapped.Index<0:0:1>) updates(%0 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %4 = VPUMI37XX.MappedInference dmas(%2 : !VPURegMapped.Index<0:0:0>) barriers(%0 : !VPURegMapped.Index<0:0:0>) dmaCount([2, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(1) -> !VPURegMapped.Index<0:0:0>
    %5 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %2 : !VPURegMapped.Index<0:0:0>
      ELFNPU37XX.PutOpInSection %3 : !VPURegMapped.Index<0:0:1>
    }
    // CHECK:  [[VAL1:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELFNPU37XX.Section {
    %6 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %0 : !VPURegMapped.Index<0:0:0>
    }
    // CHECK:  [[VAL2:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELFNPU37XX.Section {
    %7 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %4 : !VPURegMapped.Index<0:0:0>
    }
    // CHECK:  [[VAL3:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELFNPU37XX.Section {
    %8 = ELFNPU37XX.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELFNPU37XX.Section {
      %38 = VPUMI37XX.NetworkMetadata -> !VPURegMapped.Index<0:0:0>
    }
    %9 = ELFNPU37XX.Symbol %5 name("sym_dmaSection0") : !ELFNPU37XX.Section
    %10 = ELFNPU37XX.Symbol %6 name("sym_barrierSection") : !ELFNPU37XX.Section
    %11 = ELFNPU37XX.Symbol %arg0 name("Parameter") size(400) : memref<1x100xui32, @DDR>
    %12 = ELFNPU37XX.Symbol %arg1 name("Convert") size(400) : memref<1x100xui32, @DDR>
    %13 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %11 : !ELFNPU37XX.Symbol
    }
    %14 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %12 : !ELFNPU37XX.Symbol
    }
    %15 = ELFNPU37XX.CreateLogicalSection secType(SHT_NOBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.BuffersIO"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %1 : memref<1x100xui32, @DDR>
    }
    %16 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.ConstIO"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %cst : memref<1x100xui32>
    }
    // CHECK:  [[VAL4:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.ConstIO"} -> !ELFNPU37XX.Section {
    %17 = ELFNPU37XX.Symbol %cst name("sym_const0") : memref<1x100xui32>
    %18 = ELFNPU37XX.Symbol %15 name("sym_bufferSection") : !ELFNPU37XX.Section
    %19 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.buffers") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %18 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %17 : !ELFNPU37XX.Symbol
    }
    %c0_i8 = arith.constant 0 : i8
    %20 = ELFNPU37XX.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") <{isBuiltin}> : i8
    %c1_i8 = arith.constant 1 : i8
    %21 = ELFNPU37XX.Symbol %c1_i8 name("VPU_NNRD_SYM_RTM_IVAR") <{isBuiltin}> : i8
    %c2_i8 = arith.constant 2 : i8
    %22 = ELFNPU37XX.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT") <{isBuiltin}> : i8
    %c3_i8 = arith.constant 3 : i8
    %23 = ELFNPU37XX.Symbol %c3_i8 name("VPU_NNRD_SYM_RTM_DMA0") <{isBuiltin}> : i8
    %c4_i8 = arith.constant 4 : i8
    %24 = ELFNPU37XX.Symbol %c4_i8 name("VPU_NNRD_SYM_RTM_DMA1") <{isBuiltin}> : i8
    %c5_i8 = arith.constant 5 : i8
    %25 = ELFNPU37XX.Symbol %c5_i8 name("VPU_NNRD_SYM_FIFO_BASE") <{isBuiltin}> : i8
    %c6_i8 = arith.constant 6 : i8
    %26 = ELFNPU37XX.Symbol %c6_i8 name("VPU_NNRD_SYM_BARRIERS_START") <{isBuiltin}> : i8
    %27 = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") <{isBuiltin}> -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %20 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %21 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %22 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %23 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %24 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %25 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %26 : !ELFNPU37XX.Symbol
    }
    %28 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %9 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %10 : !ELFNPU37XX.Symbol
      %38 = ELFNPU37XX.Symbol %4 name("MappedInference_entry") type(<VPU_STT_ENTRY>) : !VPURegMapped.Index<0:0:0>
    }
    %29 = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput") sourceSymbolTableSection(%14) targetSection(%5) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%3 : !VPURegMapped.Index<0:0:1>) offset(24) <R_VPU_64> %12 0
    }
    %30 = ELFNPU37XX.CreateRelocationSection secName(".rlt.dmaIO") sourceSymbolTableSection(%19) targetSection(%5) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%2 : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_64> %17 0
      ELFNPU37XX.Reloc baseOp(%2 : !VPURegMapped.Index<0:0:0>) offset(24) <R_VPU_64> %18 0
      ELFNPU37XX.Reloc baseOp(%3 : !VPURegMapped.Index<0:0:1>) offset(16) <R_VPU_64> %18 0
    }
    %31 = ELFNPU37XX.CreateRelocationSection secName(".rlt.dmaIO_CMX") sourceSymbolTableSection(%27) targetSection(%5) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%2 : !VPURegMapped.Index<0:0:0>) offset(0) <R_VPU_32_RTM> %23 192
    }
    %32 = VPURT.DeclareBuffer <DDR> <64> -> memref<262144xi32, @DDR>
    %33 = ELFNPU37XX.CreateLogicalSection secType(SHT_NOBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actKernelRtConfigSec"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %32 : memref<262144xi32, @DDR>
    }
    %34 = ELFNPU37XX.Symbol %33 name("sym_actKernelRtConfigsSec") : !ELFNPU37XX.Section
    %35 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.actKernelRtConfig") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %34 : !ELFNPU37XX.Symbol
    }
    %36 = ELFNPU37XX.CreateRelocationSection secName(".rlt.MI_AKRtConfig") sourceSymbolTableSection(%35) targetSection(%7) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(828) <R_VPU_32> %34 0
    }
    %37 = ELFNPU37XX.CreateRelocationSection secName(".rlt.MappedInference") sourceSymbolTableSection(%28) targetSection(%7) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(0) <R_VPU_64> %9 0
      ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(72) <R_VPU_64> %10 0
    }
    return %arg1 : memref<1x100xui32, @DDR>
  }
}
