//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.Resources 2 of @NCE at 1.300000e+03 MHz {
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @SHAVE_ACT
  config.ExecutorResource 1 of @SHAVE_NN
  config.ExecutorResource 2 of @DMA_NN
  config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
  config.MemoryResource 524288000 bytes of @DDR {config.bandwidth = 8 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @race_condition_dma_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16, {order = #NHWC}>
    DataInfo "output_1" : tensor<1x16x16x16xf16, {order = #NHWC}>
  }
  func.func private @race_condition_dma_f16_f16(%arg0: memref<1x16x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x16x16x16xf16, #NHWC, @DDR>, %arg2: memref<1x16x16x16xf16, #NHWC, @DDR>) -> (memref<1x16x16x16xf16, #NHWC, @DDR>, memref<1x16x16x16xf16, #NHWC, @DDR>) {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>
    %2 = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %3 = VPUMI37XX.NNDMA inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) updates(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %4 = VPUMI37XX.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) updates(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %5 = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %6 = VPUMI37XX.NNDMA inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%3 : !VPURegMapped.Index<0:0:0>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %7 = VPUMI37XX.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) previousDMA(%4 : !VPURegMapped.Index<0:0:0>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %8 = VPUMI37XX.NNDMA inputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x16x16xf16, #NHWC, @DDR>) previousDMA(%6 : !VPURegMapped.Index<0:0:1>) waits(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
    %9 = VPUMI37XX.NNDMA {port = 1 : i64} inputs(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) outputs(%arg2 : memref<1x16x16x16xf16, #NHWC, @DDR>) previousDMA(%7 : !VPURegMapped.Index<0:0:1>) waits(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
    %10 = VPUMI37XX.MappedInference dmas(%3, %4 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>) barriers(%2 : !VPURegMapped.Index<0:0:0>) dmaCount([3, 3]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPURegMapped.Index<0:0:0>
    %11 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELF.Section {
      ELF.PutOpInSection %3 : !VPURegMapped.Index<0:0:0>
      ELF.PutOpInSection %6 : !VPURegMapped.Index<0:0:1>
      ELF.PutOpInSection %8 : !VPURegMapped.Index<0:0:2>
    }
    %12 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks1"} -> !ELF.Section {
      ELF.PutOpInSection %4 : !VPURegMapped.Index<0:0:0>
      ELF.PutOpInSection %7 : !VPURegMapped.Index<0:0:1>
      ELF.PutOpInSection %9 : !VPURegMapped.Index<0:0:2>
    }
    %13 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELF.Section {
      ELF.PutOpInSection %2 : !VPURegMapped.Index<0:0:0>
      ELF.PutOpInSection %5 : !VPURegMapped.Index<0:0:1>
    }
    %14 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELF.Section {
      ELF.PutOpInSection %10 : !VPURegMapped.Index<0:0:0>
    }
    %15 = ELF.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 8 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELF.Section {
      %40 = VPUMI37XX.NetworkMetadata -> !VPURegMapped.Index<0:0:0>
    }
    %16 = ELF.Symbol %11 name("sym_dmaSection0") : !ELF.Section
    %17 = ELF.Symbol %12 name("sym_dmaSection1") : !ELF.Section
    %18 = ELF.Symbol %13 name("sym_barrierSection") : !ELF.Section
    %19 = ELF.Symbol %arg0 name("input_0") size(8192) : memref<1x16x16x16xf16, #NHWC, @DDR>
    %20 = ELF.Symbol %arg1 name("output_0") size(8192) : memref<1x16x16x16xf16, #NHWC, @DDR>
    %21 = ELF.Symbol %arg2 name("output_1") size(8192) : memref<1x16x16x16xf16, #NHWC, @DDR>
    %22 = ELF.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section {
      ELF.PutOpInSection %19 : !ELF.Symbol
    }
    %23 = ELF.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section {
      ELF.PutOpInSection %20 : !ELF.Symbol
      ELF.PutOpInSection %21 : !ELF.Symbol
    }
    %c0_i8 = arith.constant 0 : i8
    %24 = ELF.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
    %c1_i8 = arith.constant 1 : i8
    %25 = ELF.Symbol %c1_i8 name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
    %c2_i8 = arith.constant 2 : i8
    %26 = ELF.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
    %c3_i8 = arith.constant 3 : i8
    %27 = ELF.Symbol %c3_i8 name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
    %c4_i8 = arith.constant 4 : i8
    %28 = ELF.Symbol %c4_i8 name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
    %c5_i8 = arith.constant 5 : i8
    %29 = ELF.Symbol %c5_i8 name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
    %c6_i8 = arith.constant 6 : i8
    %30 = ELF.Symbol %c6_i8 name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
    %31 = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELF.Section {
      ELF.PutOpInSection %24 : !ELF.Symbol
      ELF.PutOpInSection %25 : !ELF.Symbol
      ELF.PutOpInSection %26 : !ELF.Symbol
      ELF.PutOpInSection %27 : !ELF.Symbol
      ELF.PutOpInSection %28 : !ELF.Symbol
      ELF.PutOpInSection %29 : !ELF.Symbol
      ELF.PutOpInSection %30 : !ELF.Symbol
    }
    %32 = ELF.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELF.Section {
      ELF.PutOpInSection %16 : !ELF.Symbol
      ELF.PutOpInSection %17 : !ELF.Symbol
      ELF.PutOpInSection %18 : !ELF.Symbol
      %40 = ELF.Symbol %10 name("MappedInference_entry") type(<VPU_STT_ENTRY>) : !VPURegMapped.Index<0:0:0>
    }
    %33 = ELF.CreateRelocationSection secName(".rlt.DMA_NetInput0") sourceSymbolTableSection(%22) targetSection(%11) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%3 : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_64> %19 0
      ELF.RelocImmOffset baseOp(%6 : !VPURegMapped.Index<0:0:1>) offset(16) <R_VPU_64> %19 0
    }
    %34 = ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput0") sourceSymbolTableSection(%23) targetSection(%11) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%8 : !VPURegMapped.Index<0:0:2>) offset(24) <R_VPU_64> %20 0
    }
    %35 = ELF.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection(%31) targetSection(%11) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.Reloc baseOp(%3 : !VPURegMapped.Index<0:0:0>) offsetOf(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) <R_VPU_64> %24 0
      ELF.RelocImmOffset baseOp(%3 : !VPURegMapped.Index<0:0:0>) offset(0) <R_VPU_32_RTM> %27 128
      ELF.Reloc baseOp(%6 : !VPURegMapped.Index<0:0:1>) offsetOf(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) <R_VPU_64> %24 0
      ELF.RelocImmOffset baseOp(%6 : !VPURegMapped.Index<0:0:1>) offset(0) <R_VPU_32_RTM> %27 128
      ELF.Reloc baseOp(%8 : !VPURegMapped.Index<0:0:2>) offsetOf(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) <R_VPU_64> %24 0
    }
    %36 = ELF.CreateRelocationSection secName(".rlt.DMA_NetInput1") sourceSymbolTableSection(%22) targetSection(%12) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_64> %19 0
      ELF.RelocImmOffset baseOp(%7 : !VPURegMapped.Index<0:0:1>) offset(16) <R_VPU_64> %19 0
    }
    %37 = ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput1") sourceSymbolTableSection(%23) targetSection(%12) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%9 : !VPURegMapped.Index<0:0:2>) offset(24) <R_VPU_64> %21 0
    }
    %38 = ELF.CreateRelocationSection secName(".rlt.text.dmaTasks1") sourceSymbolTableSection(%31) targetSection(%12) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offsetOf(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) <R_VPU_64> %24 2097152
      ELF.RelocImmOffset baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(0) <R_VPU_32_RTM> %28 128
      ELF.Reloc baseOp(%7 : !VPURegMapped.Index<0:0:1>) offsetOf(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) <R_VPU_64> %24 2097152
      ELF.RelocImmOffset baseOp(%7 : !VPURegMapped.Index<0:0:1>) offset(0) <R_VPU_32_RTM> %28 128
      ELF.Reloc baseOp(%9 : !VPURegMapped.Index<0:0:2>) offsetOf(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) <R_VPU_64> %24 2097152
    }
    %39 = ELF.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%32) targetSection(%14) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offsetOf(%3 : !VPURegMapped.Index<0:0:0>) <R_VPU_64> %16 0
      ELF.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offsetOf(%4 : !VPURegMapped.Index<0:0:0>) <R_VPU_64> %17 0
      ELF.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offsetOf(%2 : !VPURegMapped.Index<0:0:0>) <R_VPU_64> %18 0
    }
    return %arg1, %arg2 : memref<1x16x16x16xf16, #NHWC, @DDR>, memref<1x16x16x16xf16, #NHWC, @DDR>
  }
}
