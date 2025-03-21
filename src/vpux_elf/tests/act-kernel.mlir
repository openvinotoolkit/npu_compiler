//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

module @Test attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.MemoryResource 2097152 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  IE.ExecutorResource 1 of @DMA_NN
  IE.ExecutorResource 1 of @SHAVE_ACT
  IE.TileResource 1 of @NCE {
    IE.ExecutorResource 1 of @DPU
  }
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "hswish" : tensor<1x1000xf16>
  }
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
  module @VPU.SW {
    func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  func.func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %2 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %3 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %4 = VPUMI37XX.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) updates(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %5 = VPUMI37XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
    %6 = VPUMI37XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
    %7 = VPUMI37XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
    %8 = VPUMI37XX.ActKernelRange kernel_text_index(%5 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%7 : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.ActKernelInvocation range_index(%8 : <0:0:0>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%3 : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI37XX.KernelParams inputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI37XX.NNDMA inputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16>) previousDMA(%4 : !VPURegMapped.Index<0:0:0>) waits(%3 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %12 = VPURT.DeclareBuffer <DDR> <0> -> memref<8192xi8, @DDR>
    %13 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_0"} -> !ELF.Section {
      ELF.PutOpInSection %12 : memref<8192xi8, @DDR>
    }
    %14 = ELF.Symbol %13 name("sym_actShaveStack_0") : !ELF.Section
    %15 = VPURT.DeclareBuffer <DDR> <0> -> memref<8192xi8, @DDR>
    %16 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_1"} -> !ELF.Section {
      ELF.PutOpInSection %15 : memref<8192xi8, @DDR>
    }
    %17 = ELF.Symbol %16 name("sym_actShaveStack_1") : !ELF.Section
    %18 = VPURT.DeclareBuffer <DDR> <0> -> memref<8192xi8, @DDR>
    %19 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_2"} -> !ELF.Section {
      ELF.PutOpInSection %18 : memref<8192xi8, @DDR>
    }
    %20 = ELF.Symbol %19 name("sym_actShaveStack_2") : !ELF.Section
    %21 = VPURT.DeclareBuffer <DDR> <0> -> memref<8192xi8, @DDR>
    %22 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_3"} -> !ELF.Section {
      ELF.PutOpInSection %21 : memref<8192xi8, @DDR>
    }
    %23 = ELF.Symbol %22 name("sym_actShaveStack_3") : !ELF.Section
    %24 = VPUMI37XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
    %25 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.actKernelRtConfigSec"} -> !ELF.Section {
      ELF.PutOpInSection %24 : !VPURegMapped.Index<0:0:0>
    }
    %26 = ELF.Symbol %25 name("sym_actKernelRtConfigsSec") : !ELF.Section
    %27 = ELF.CreateSymbolTableSection secName(".symtab.actKernelRtConfig") secFlags("SHF_NONE") -> !ELF.Section {
      ELF.PutOpInSection %26 : !ELF.Symbol
      ELF.PutOpInSection %14 : !ELF.Symbol
      ELF.PutOpInSection %17 : !ELF.Symbol
      ELF.PutOpInSection %20 : !ELF.Symbol
      ELF.PutOpInSection %23 : !ELF.Symbol
    }
    %28 = VPUMI37XX.MappedInference dmas(%4 : !VPURegMapped.Index<0:0:0>) actKernelRanges(%8 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%9 : !VPURegMapped.Index<0:0:0>) barriers(%2 : !VPURegMapped.Index<0:0:0>) actShaveRt(%24 : !VPURegMapped.Index<0:0:0>) actShaveStacks(%12, %15, %18, %21 : memref<8192xi8, @DDR>, memref<8192xi8, @DDR>, memref<8192xi8, @DDR>, memref<8192xi8, @DDR>) dmaCount([2, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(2) -> !VPURegMapped.Index<0:0:0>
    %29 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELF.Section {
      ELF.PutOpInSection %4 : !VPURegMapped.Index<0:0:0>
      ELF.PutOpInSection %11 : !VPURegMapped.Index<0:0:1>
    }
    %30 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELF.Section {
      ELF.PutOpInSection %2 : !VPURegMapped.Index<0:0:0>
      ELF.PutOpInSection %3 : !VPURegMapped.Index<0:0:1>
    }
    %31 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.KernelText"} -> !ELF.Section {
      ELF.PutOpInSection %5 : !VPURegMapped.Index<0:0:0>
    }
    %32 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.KernelData"} -> !ELF.Section {
      ELF.PutOpInSection %6 : !VPURegMapped.Index<0:0:0>
    }
    %33 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.KernelParams"} -> !ELF.Section {
      ELF.PutOpInSection %10 : !VPURegMapped.Index<0:0:0>
    }
    %34 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelRanges"} -> !ELF.Section {
      ELF.PutOpInSection %8 : !VPURegMapped.Index<0:0:0>
    }
    %35 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelInvocations"} -> !ELF.Section {
      ELF.PutOpInSection %9 : !VPURegMapped.Index<0:0:0>
    }
    %36 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELF.Section {
      ELF.PutOpInSection %28 : !VPURegMapped.Index<0:0:0>
    }
    %37 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.DPUInvariants"} -> !ELF.Section {
    }
    %38 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.DPUVariants"} -> !ELF.Section {
    }
    %39 = ELF.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 8 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELF.Section {
      %78 = VPUMI37XX.NetworkMetadata -> !VPURegMapped.Index<0:0:0>
    }
    %40 = ELF.Symbol %29 name("sym_dmaSection0") : !ELF.Section
    %41 = ELF.Symbol %30 name("sym_barrierSection") : !ELF.Section
    %42 = ELF.Symbol %34 name("sym_actKernelRangeSection") : !ELF.Section
    %43 = ELF.Symbol %35 name("sym_actKernelInvo") : !ELF.Section
    %44 = ELF.Symbol %31 name("sym_kernelTextSection") : !ELF.Section
    %45 = ELF.Symbol %32 name("sym_kernelDataSection") : !ELF.Section
    %46 = ELF.Symbol %33 name("sym_kernelParamsSection") : !ELF.Section
    %47 = ELF.Symbol %37 name("sym_inVariantsSection") : !ELF.Section
    %48 = ELF.Symbol %38 name("sym_variantsSection") : !ELF.Section
    %49 = ELF.Symbol %arg0 name("input") size(2000) : memref<1x1x1x1000xf16>
    %50 = ELF.Symbol %arg1 name("hswish") size(2000) : memref<1x1x1x1000xf16>
    %51 = ELF.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section {
      ELF.PutOpInSection %49 : !ELF.Symbol
    }
    %52 = ELF.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section {
      ELF.PutOpInSection %50 : !ELF.Symbol
    }
    %53 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.BuffersIO"} -> !ELF.Section {
    }
    %54 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.ConstIO"} -> !ELF.Section {
    }
    %55 = ELF.Symbol %54 name("sym_constSection") : !ELF.Section
    %56 = ELF.Symbol %53 name("sym_bufferSection") : !ELF.Section
    %57 = ELF.CreateSymbolTableSection secName(".symtab.buffers") secFlags("SHF_NONE") -> !ELF.Section {
      ELF.PutOpInSection %56 : !ELF.Symbol
      ELF.PutOpInSection %55 : !ELF.Symbol
    }
    %c0_i8 = arith.constant 0 : i8
    %58 = ELF.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
    %c1_i8 = arith.constant 1 : i8
    %59 = ELF.Symbol %c1_i8 name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
    %c2_i8 = arith.constant 2 : i8
    %60 = ELF.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
    %c3_i8 = arith.constant 3 : i8
    %61 = ELF.Symbol %c3_i8 name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
    %c4_i8 = arith.constant 4 : i8
    %62 = ELF.Symbol %c4_i8 name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
    %c5_i8 = arith.constant 5 : i8
    %63 = ELF.Symbol %c5_i8 name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
    %c6_i8 = arith.constant 6 : i8
    %64 = ELF.Symbol %c6_i8 name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
    %65 = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELF.Section {
      ELF.PutOpInSection %58 : !ELF.Symbol
      ELF.PutOpInSection %59 : !ELF.Symbol
      ELF.PutOpInSection %60 : !ELF.Symbol
      ELF.PutOpInSection %61 : !ELF.Symbol
      ELF.PutOpInSection %62 : !ELF.Symbol
      ELF.PutOpInSection %63 : !ELF.Symbol
      ELF.PutOpInSection %64 : !ELF.Symbol
    }
    %66 = ELF.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELF.Section {
      ELF.PutOpInSection %40 : !ELF.Symbol
      ELF.PutOpInSection %41 : !ELF.Symbol
      ELF.PutOpInSection %44 : !ELF.Symbol
      ELF.PutOpInSection %45 : !ELF.Symbol
      ELF.PutOpInSection %46 : !ELF.Symbol
      ELF.PutOpInSection %42 : !ELF.Symbol
      ELF.PutOpInSection %43 : !ELF.Symbol
      ELF.PutOpInSection %47 : !ELF.Symbol
      ELF.PutOpInSection %48 : !ELF.Symbol
      %77 = ELF.Symbol %28 name("MappedInference_entry") type(<VPU_STT_ENTRY>) : !VPURegMapped.Index<0:0:0>
    }
    %67 = ELF.CreateRelocationSection secName(".rlt.DMA_NetInput") sourceSymbolTableSection(%51) targetSection(%29) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_64> %49 0
    }
    %68 = ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput") sourceSymbolTableSection(%52) targetSection(%29) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%11 : !VPURegMapped.Index<0:0:1>) offset(24) <R_VPU_64> %50 0
    }
    %69 = ELF.CreateRelocationSection secName(".rlt.text.dmaTasks") sourceSymbolTableSection(%65) targetSection(%29) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offsetOf(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) <R_VPU_64> %58 0
      ELF.RelocImmOffset baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(0) <R_VPU_32_RTM> %61 128
      ELF.Reloc baseOp(%11 : !VPURegMapped.Index<0:0:1>) offsetOf(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) <R_VPU_64> %58 2000
    }
    %70 = ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%66) targetSection(%33) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(12) <R_VPU_32> %46 72
      ELF.RelocImmOffset baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_32> %46 88
      ELF.RelocImmOffset baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(48) <R_VPU_32> %46 120
      ELF.RelocImmOffset baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(52) <R_VPU_32> %46 136
    }
    %71 = ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%65) targetSection(%33) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offsetOf(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) <R_VPU_32> %58 0
      ELF.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offsetOf(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) <R_VPU_32> %58 2000
    }
    %72 = ELF.CreateRelocationSection secName(".rlt.text.ActKernelRanges") sourceSymbolTableSection(%66) targetSection(%34) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.Reloc baseOp(%8 : !VPURegMapped.Index<0:0:0>) offsetOf(%5 : !VPURegMapped.Index<0:0:0>) <R_VPU_32> %44 0
    }
    %73 = ELF.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%65) targetSection(%35) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.Reloc baseOp(%9 : !VPURegMapped.Index<0:0:0>) offsetOf(%8 : !VPURegMapped.Index<0:0:0>) <R_VPU_32_RTM> %60 24
    }
    %74 = ELF.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%66) targetSection(%35) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%9 : !VPURegMapped.Index<0:0:0>) offset(8) <R_VPU_32> %45 0
      ELF.RelocImmOffset baseOp(%9 : !VPURegMapped.Index<0:0:0>) offset(4) <R_VPU_32> %46 0
    }
    %75 = ELF.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%27) targetSection(%36) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%28 : !VPURegMapped.Index<0:0:0>) offset(148) <R_VPU_32> %26 0
      ELF.Reloc baseOp(%28 : !VPURegMapped.Index<0:0:0>) offsetOf(%12 : memref<8192xi8, @DDR>) <R_VPU_32> %14 8192
      ELF.Reloc baseOp(%28 : !VPURegMapped.Index<0:0:0>) offsetOf(%15 : memref<8192xi8, @DDR>) <R_VPU_32> %17 8192
      ELF.Reloc baseOp(%28 : !VPURegMapped.Index<0:0:0>) offsetOf(%18 : memref<8192xi8, @DDR>) <R_VPU_32> %20 8192
      ELF.Reloc baseOp(%28 : !VPURegMapped.Index<0:0:0>) offsetOf(%21 : memref<8192xi8, @DDR>) <R_VPU_32> %23 8192
    }
    %76 = ELF.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%66) targetSection(%36) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.Reloc baseOp(%28 : !VPURegMapped.Index<0:0:0>) offsetOf(%4 : !VPURegMapped.Index<0:0:0>) <R_VPU_64> %40 0
      ELF.Reloc baseOp(%28 : !VPURegMapped.Index<0:0:0>) offsetOf(%2 : !VPURegMapped.Index<0:0:0>) <R_VPU_64> %41 0
      ELF.Reloc baseOp(%28 : !VPURegMapped.Index<0:0:0>) offsetOf(%8 : !VPURegMapped.Index<0:0:0>) <R_VPU_64> %42 0
      ELF.Reloc baseOp(%28 : !VPURegMapped.Index<0:0:0>) offsetOf(%9 : !VPURegMapped.Index<0:0:0>) <R_VPU_64> %43 0
    }
    return %arg1 : memref<1x1x1x1000xf16>
  }
}
