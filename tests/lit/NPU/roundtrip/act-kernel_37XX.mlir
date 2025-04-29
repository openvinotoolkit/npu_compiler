//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --export-ELF %s -o %t.elf
// RUN: vpux-translate --vpu-arch=%arch% --import-ELF %t.elf | FileCheck %s
// RUN: rm %t.elf
// REQUIRES: arch-NPU37XX
//

module @Test attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.ExecutorResource 1 of @DMA_NN
  IE.TileResource 1 of @NCE {
    IE.MemoryResource 2097152 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @SHAVE_ACT
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
    %5 = VPUMI37XX.DeclareKernelText kernel_path("activation_hswish_3720xx") -> !VPURegMapped.Index<0:0:0>
    %6 = VPUMI37XX.DeclareKernelArgs kernel_path("activation_hswish_3720xx") -> !VPURegMapped.Index<0:0:0>
    %7 = VPUMI37XX.DeclareKernelEntry kernel_path("activation_hswish_3720xx") -> !VPURegMapped.Index<0:0:0>
    %8 = VPUMI37XX.ActKernelRange kernel_text_index(%5 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%7 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.ActKernelInvocation range_index(%8 : <0:0:0>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%3 : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI37XX.KernelParams inputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16>) waits(%3 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %4 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) nextDMAIdx(%11 : !VPURegMapped.Index<0:0:1>) updates(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %24 = VPUMI37XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
    %25 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.actKernelRtConfigSec"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %24 : !VPURegMapped.Index<0:0:0>
    }
    %26 = ELFNPU37XX.Symbol %25 name("sym_actKernelRtConfigsSec") : !ELFNPU37XX.Section
    %27 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.actKernelRtConfig") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %26 : !ELFNPU37XX.Symbol
    }
    %28 = VPUMI37XX.MappedInference dmas(%4 : !VPURegMapped.Index<0:0:0>) actKernelRanges(%8 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%9 : !VPURegMapped.Index<0:0:0>) barriers(%2 : !VPURegMapped.Index<0:0:0>) actShaveRt(%24 : !VPURegMapped.Index<0:0:0>) dmaCount([2]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(2) -> !VPURegMapped.Index<0:0:0>
    %29 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %4 : !VPURegMapped.Index<0:0:0>
      ELFNPU37XX.PutOpInSection %11 : !VPURegMapped.Index<0:0:1>
    }
    %30 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %2 : !VPURegMapped.Index<0:0:0>
      ELFNPU37XX.PutOpInSection %3 : !VPURegMapped.Index<0:0:1>
    }
    %31 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.KernelText"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %5 : !VPURegMapped.Index<0:0:0>
    }
    %32 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.KernelData"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %6 : !VPURegMapped.Index<0:0:0>
    }
    %33 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.KernelParams"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %10 : !VPURegMapped.Index<0:0:0>
    }
    %34 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelRanges"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %8 : !VPURegMapped.Index<0:0:0>
    }
    %35 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelInvocations"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %9 : !VPURegMapped.Index<0:0:0>
    }
    %36 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %28 : !VPURegMapped.Index<0:0:0>
    }
    %37 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.DPUInvariants"} -> !ELFNPU37XX.Section {
    }
    %38 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.DPUVariants"} -> !ELFNPU37XX.Section {
    }
    %39 = ELFNPU37XX.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 8 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELFNPU37XX.Section {
      %78 = VPUMI37XX.NetworkMetadata -> !VPURegMapped.Index<0:0:0>
    }
    %40 = ELFNPU37XX.Symbol %29 name("sym_dmaSection0") : !ELFNPU37XX.Section
    %41 = ELFNPU37XX.Symbol %30 name("sym_barrierSection") : !ELFNPU37XX.Section
    %42 = ELFNPU37XX.Symbol %34 name("sym_actKernelRangeSection") : !ELFNPU37XX.Section
    %43 = ELFNPU37XX.Symbol %35 name("sym_actKernelInvo") : !ELFNPU37XX.Section
    %44 = ELFNPU37XX.Symbol %31 name("sym_kernelTextSection") : !ELFNPU37XX.Section
    %45 = ELFNPU37XX.Symbol %32 name("sym_kernelDataSection") : !ELFNPU37XX.Section
    %46 = ELFNPU37XX.Symbol %33 name("sym_kernelParamsSection") : !ELFNPU37XX.Section
    %47 = ELFNPU37XX.Symbol %37 name("sym_inVariantsSection") : !ELFNPU37XX.Section
    %48 = ELFNPU37XX.Symbol %38 name("sym_variantsSection") : !ELFNPU37XX.Section
    %49 = ELFNPU37XX.Symbol %arg0 name("input") size(2000) : memref<1x1x1x1000xf16>
    %50 = ELFNPU37XX.Symbol %arg1 name("hswish") size(2000) : memref<1x1x1x1000xf16>
    %51 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %49 : !ELFNPU37XX.Symbol
    }
    %52 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %50 : !ELFNPU37XX.Symbol
    }
    %53 = ELFNPU37XX.CreateLogicalSection secType(SHT_NOBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.BuffersIO"} -> !ELFNPU37XX.Section {
    }
    %54 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.ConstIO"} -> !ELFNPU37XX.Section {
    }
    %55 = ELFNPU37XX.Symbol %54 name("sym_constSection") : !ELFNPU37XX.Section
    %56 = ELFNPU37XX.Symbol %53 name("sym_bufferSection") : !ELFNPU37XX.Section
    %57 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.buffers") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %56 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %55 : !ELFNPU37XX.Symbol
    }
    %c0_i8 = arith.constant 0 : i8
    %58 = ELFNPU37XX.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
    %c1_i8 = arith.constant 1 : i8
    %59 = ELFNPU37XX.Symbol %c1_i8 name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
    %c2_i8 = arith.constant 2 : i8
    %60 = ELFNPU37XX.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
    %c3_i8 = arith.constant 3 : i8
    %61 = ELFNPU37XX.Symbol %c3_i8 name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
    %c4_i8 = arith.constant 4 : i8
    %62 = ELFNPU37XX.Symbol %c4_i8 name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
    %c5_i8 = arith.constant 5 : i8
    %63 = ELFNPU37XX.Symbol %c5_i8 name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
    %c6_i8 = arith.constant 6 : i8
    %64 = ELFNPU37XX.Symbol %c6_i8 name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
    %65 = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %58 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %59 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %60 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %61 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %62 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %63 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %64 : !ELFNPU37XX.Symbol
    }
    %66 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %40 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %41 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %44 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %45 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %46 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %42 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %43 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %47 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %48 : !ELFNPU37XX.Symbol
      %77 = ELFNPU37XX.Symbol %28 name("MappedInference_entry") type(<VPU_STT_ENTRY>) : !VPURegMapped.Index<0:0:0>
    }
    %67 = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetInput") sourceSymbolTableSection(%51) targetSection(%29) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_64> %49 0
    }
    %68 = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput") sourceSymbolTableSection(%52) targetSection(%29) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%11 : !VPURegMapped.Index<0:0:1>) offset(24) <R_VPU_64> %50 0
    }
    %69 = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.dmaTasks") sourceSymbolTableSection(%65) targetSection(%29) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc offset(24) <R_VPU_64> %58 0
      ELFNPU37XX.Reloc offset(0) <R_VPU_32_RTM> %61 128
      ELFNPU37XX.Reloc offset(144) <R_VPU_64> %58 2000
    }
    %70 = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%66) targetSection(%33) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(12) <R_VPU_32> %46 72
      ELFNPU37XX.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_32> %46 88
      ELFNPU37XX.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(48) <R_VPU_32> %46 120
      ELFNPU37XX.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offset(52) <R_VPU_32> %46 136
    }
    %71 = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%65) targetSection(%33) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc offset(0) <R_VPU_32> %58 0
      ELFNPU37XX.Reloc offset(36) <R_VPU_32> %58 2000
    }
    %72 = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.ActKernelRanges") sourceSymbolTableSection(%66) targetSection(%34) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc offset(8) <R_VPU_32> %44 0
    }
    %73 = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%65) targetSection(%35) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc offset(0) <R_VPU_32_RTM> %60 24
    }
    %74 = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%66) targetSection(%35) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%9 : !VPURegMapped.Index<0:0:0>) offset(8) <R_VPU_32> %45 0
      ELFNPU37XX.Reloc baseOp(%9 : !VPURegMapped.Index<0:0:0>) offset(4) <R_VPU_32> %46 0
    }
    %75 = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%27) targetSection(%36) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc baseOp(%28 : !VPURegMapped.Index<0:0:0>) offset(340) <R_VPU_32> %26 0
    }
    %76 = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%66) targetSection(%36) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      ELFNPU37XX.Reloc offset(72) <R_VPU_64> %40 0
      ELFNPU37XX.Reloc offset(312) <R_VPU_64> %41 0
      ELFNPU37XX.Reloc offset(237) <R_VPU_64> %42 0
      ELFNPU37XX.Reloc offset(272) <R_VPU_64> %43 0
    }
    return %arg1 : memref<1x1x1x1000xf16>
  }
}

//CHECK-DAG: %[[VAL0:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, 4294967295> -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: %[[VAL1:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, 4294967295> -> !VPURegMapped.Index<0:0:1>
//CHECK: %[[VAL2:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.BarrierConfigs"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL0]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL1]] : !VPURegMapped.Index<0:0:1>

//CHECK-DAG: %[[VAL3:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2000xui8, [@CMX_NN, 0]>
//CHECK-DAG: %[[VAL5:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<2000xui8, [@CMX_NN, 0]>
//CHECK-DAG: %[[VAL6:.*]] = VPUMI37XX.NNDMA  {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 2000 : i64, srcWidth = 2000 : i64, srcStride = 2000 : i64, srcPlaneStride = 0 : i64, dstWidth = 2000 : i64, dstStride = 2000 : i64, dstPlaneStride = 0 : i64>, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VAL5]] : memref<2000xui8, [@CMX_NN, 0]>) outputs(%[[VALOUTPUT:.*]] : {{.*}}) waits(%[[VAL1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
//CHECK-DAG: %[[VAL4:.*]] = VPUMI37XX.NNDMA  {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 2000 : i64, srcWidth = 2000 : i64, srcStride = 2000 : i64, srcPlaneStride = 0 : i64, dstWidth = 2000 : i64, dstStride = 2000 : i64, dstPlaneStride = 0 : i64>, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VALINPUT:.*]] : {{.*}}) outputs(%[[VAL3]] : memref<2000xui8, [@CMX_NN, 0]>) nextDMAIdx(%[[VAL6]] : !VPURegMapped.Index<0:0:1>) updates(%[[VAL0]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
//CHECK: %[[VAL7:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.dmaTasks"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL4]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL6]] : !VPURegMapped.Index<0:0:1>

//CHECK-DAG: %[[VAL8:.*]] = VPUMI37XX.DeclareKernelText
//CHECK-DAG: %[[VAL9:.*]] = VPUMI37XX.DeclareKernelArgs
//CHECK-DAG: %[[VAL10:.*]] = VPUMI37XX.DeclareKernelEntry
//CHECK-DAG: %[[VAL11:.*]] = VPUMI37XX.ActKernelRange kernel_text_index(%[[VAL8]] : !VPURegMapped.Index<0:0:0>) kernel_args_index(%[[VAL9]] : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: %[[VAL12:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.ActKernelRanges"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL11]] : !VPURegMapped.Index<0:0:0>
//CHECK-DAG: %[[VAL13:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.KernelText"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL8]] : !VPURegMapped.Index<0:0:0>
//CHECK-DAG: %[[VAL14:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.KernelData"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL9]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[VAL15:.*]] = VPUMI37XX.ActKernelInvocation  range_index(%[[VAL11]] : <0:0:0>) waits(%[[VAL0]] : !VPURegMapped.Index<0:0:0>) updates(%[[VAL1]] : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: %[[VAL16:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.ActKernelInvocations"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL15]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[VAL17:.*]] = VPUMI37XX.KernelParams {isCompiled = false} inputs(%[[VAL3]] : memref<2000xui8, [@CMX_NN, 0]>) outputs(%[[VAL5]] : memref<2000xui8, [@CMX_NN, 0]>) dynamicInputShapes(() : ()) dynamicOutputShapes(() : ()) kernel_type("Softmax") kernel_params({{.*}}) -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: %[[VAL18:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.KernelParams"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL17]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[VAL27:.*]] = VPUMI37XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: %[[VAL28:.*]] = ELFNPU37XX.CreateSection {{.*}} {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.actKernelRtConfigSec"} -> !ELFNPU37XX.Section {
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL27]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[VAL29:.*]] = VPUMI37XX.MappedInference
//CHECK-SAME: dmas(%[[VAL4]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: actKernelRanges(%[[VAL11]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: actKernelInvocations(%[[VAL15]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: barriers(%[[VAL0]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: actShaveRt(%[[VAL27]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: dmaCount([2]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(2)
//CHECK-SAME: -> !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[VAL30:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.MappedInference"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL29]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[VAL31:.*]] = ELFNPU37XX.CreateMetadataSection {{.*}} secName = ".metadata"
//CHECK-NEXT: VPUMI37XX.NetworkMetadata

//CHECK-DAG: %[[VAL32:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.DPUInvariants"

//CHECK-DAG: %[[VAL33:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.DPUVariants"

//CHECK-DAG: %[[VAL34:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".data.ConstIO"

//CHECK-DAG: %[[VAL35:.*]] = ELFNPU37XX.CreateLogicalSection {{.*}} secName = ".data.BuffersIO"

//CHECK-DAG: %[[VAL36:.*]] = ELFNPU37XX.Symbol %[[VAL28]] name("sym_actKernelRtConfigsSec") {{.*}} : !ELFNPU37XX.Section

//CHECK-DAG: %[[VAL41:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.actKernelRtConfig")
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL36]] : !ELFNPU37XX.Symbol

//CHECK-DAG: %[[VAL42:.*]] = ELFNPU37XX.Symbol %[[VALINPUT]] name("input") type(<STT_NOTYPE>) size(2000) {value = 0 : ui64} : {{.*}}
//CHECK-DAG: %[[VAL43:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELFNPU37XX.Section
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL42]] : !ELFNPU37XX.Symbol

//CHECK-DAG: %[[VAL44:.*]] = ELFNPU37XX.Symbol %[[VALOUTPUT]] name("hswish") type(<STT_NOTYPE>) size(2000) {value = 0 : ui64} : {{.*}}
//CHECK-DAG: %[[VAL45:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL44]] : !ELFNPU37XX.Symbol

//CHECK-DAG: %[[VAL46:.*]] = ELFNPU37XX.Symbol %[[VAL35]] name("sym_bufferSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG: %[[VAL47:.*]] = ELFNPU37XX.Symbol %[[VAL34]] name("sym_constSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG: %[[VAL48:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.buffers") secFlags("SHF_NONE") -> !ELFNPU37XX.Section
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL46]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL47]] : !ELFNPU37XX.Symbol

//CHECK-DAG: %[[VAL49:.*]] = ELFNPU37XX.Symbol %[[VAL7]] name("sym_dmaSection0") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG: %[[VAL50:.*]] = ELFNPU37XX.Symbol %[[VAL2]] name("sym_barrierSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG: %[[VAL51:.*]] = ELFNPU37XX.Symbol %[[VAL13]] name("sym_kernelTextSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG: %[[VAL52:.*]] = ELFNPU37XX.Symbol %[[VAL14]] name("sym_kernelDataSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG: %[[VAL53:.*]] = ELFNPU37XX.Symbol %[[VAL18]] name("sym_kernelParamsSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG: %[[VAL54:.*]] = ELFNPU37XX.Symbol %[[VAL12]] name("sym_actKernelRangeSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG: %[[VAL55:.*]] = ELFNPU37XX.Symbol %[[VAL16]] name("sym_actKernelInvo") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG: %[[VAL56:.*]] = ELFNPU37XX.Symbol %[[VAL32]] name("sym_inVariantsSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG: %[[VAL57:.*]] = ELFNPU37XX.Symbol %[[VAL33]] name("sym_variantsSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG: %[[VAL58:.*]] = ELFNPU37XX.Symbol %[[VAL30]] name("MappedInference_entry") type(<VPU_STT_ENTRY>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG: %[[VAL59:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELFNPU37XX.Section
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL49]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL50]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL51]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL52]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL53]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL54]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL55]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL56]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL57]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL58]] : !ELFNPU37XX.Symbol

//CHECK: %[[VALC0:.*]] = arith.constant 0 : i8
//CHECK: %[[VAL60:.*]] = ELFNPU37XX.Symbol %[[VALC0]] name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
//CHECK: %[[VALC1:.*]] = arith.constant 1 : i8
//CHECK: %[[VAL61:.*]] = ELFNPU37XX.Symbol %[[VALC1]] name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
//CHECK: %[[VALC2:.*]] = arith.constant 2 : i8
//CHECK: %[[VAL62:.*]] = ELFNPU37XX.Symbol %[[VALC2]] name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
//CHECK: %[[VALC3:.*]] = arith.constant 3 : i8
//CHECK: %[[VAL63:.*]] = ELFNPU37XX.Symbol %[[VALC3]] name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
//CHECK: %[[VALC4:.*]] = arith.constant 4 : i8
//CHECK: %[[VAL64:.*]] = ELFNPU37XX.Symbol %[[VALC4]] name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
//CHECK: %[[VALC5:.*]] = arith.constant 5 : i8
//CHECK: %[[VAL65:.*]] = ELFNPU37XX.Symbol %[[VALC5]] name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
//CHECK: %[[VALC6:.*]] = arith.constant 6 : i8
//CHECK: %[[VAL66:.*]] = ELFNPU37XX.Symbol %[[VALC6]] name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
//CHECK: %[[VAL67:.*]] = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELFNPU37XX.Section {
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL60]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL61]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL62]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL63]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL64]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL65]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL66]] : !ELFNPU37XX.Symbol

//CHECK: %[[VAL68:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetInput") sourceSymbolTableSection(%[[VAL43]]) targetSection(%[[VAL7]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELFNPU37XX.Section
//CHECK-DAG: ELFNPU37XX.Reloc offset(16) <R_VPU_64> %[[VAL42]] 0
//CHECK: %[[VAL69:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput") sourceSymbolTableSection(%[[VAL45]]) targetSection(%[[VAL7]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELFNPU37XX.Section
//CHECK-DAG: ELFNPU37XX.Reloc offset(152) <R_VPU_64> %[[VAL44]] 0

//CHECK: %[[VAL70:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.dmaTasks") sourceSymbolTableSection(%[[VAL67]]) targetSection(%[[VAL7]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section
//CHECK-DAG: ELFNPU37XX.Reloc offset(24) <R_VPU_64> %[[VAL60]] 0
//CHECK-DAG: ELFNPU37XX.Reloc offset(0) <R_VPU_32_RTM> %[[VAL63]] 128
//CHECK-DAG: ELFNPU37XX.Reloc offset(144) <R_VPU_64> %[[VAL60]] 2000


//CHECK: %[[VAL71:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[VAL59]]) targetSection(%[[VAL18]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section
//CHECK-DAG: ELFNPU37XX.Reloc offset(12) <R_VPU_32> %[[VAL53]] 72
//CHECK-DAG: ELFNPU37XX.Reloc offset(16) <R_VPU_32> %[[VAL53]] 88
//CHECK-DAG: ELFNPU37XX.Reloc offset(48) <R_VPU_32> %[[VAL53]] 120
//CHECK-DAG: ELFNPU37XX.Reloc offset(52) <R_VPU_32> %[[VAL53]] 136
//CHECK: %[[VAL72:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[VAL67]]) targetSection(%[[VAL18]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section
//CHECK-DAG: ELFNPU37XX.Reloc offset(0) <R_VPU_32> %[[VAL60]] 0
//CHECK-DAG: ELFNPU37XX.Reloc offset(36) <R_VPU_32> %[[VAL60]] 2000

//CHECK: %[[VAL73:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.ActKernelRanges") sourceSymbolTableSection(%[[VAL59]]) targetSection(%[[VAL12]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section
//CHECK-DAG: ELFNPU37XX.Reloc offset(8) <R_VPU_32> %[[VAL51]] 0
//CHECK: %[[VAL74:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[VAL67]]) targetSection(%[[VAL16]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section
//CHECK-DAG: ELFNPU37XX.Reloc offset(0) <R_VPU_32_RTM> %[[VAL62]] 24
//CHECK: %[[VAL75:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[VAL59]]) targetSection(%[[VAL16]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section
//CHECK-DAG: ELFNPU37XX.Reloc offset(8) <R_VPU_32> %[[VAL52]] 0
//CHECK-DAG: ELFNPU37XX.Reloc offset(4) <R_VPU_32> %[[VAL53]] 0

//CHECK: %[[VAL76:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%[[VAL41]]) targetSection(%[[VAL30]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section
//CHECK-DAG: ELFNPU37XX.Reloc offset(340) <R_VPU_32> %[[VAL36]] 0
//CHECK: %[[VAL77:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%[[VAL59]]) targetSection(%[[VAL30]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section
//CHECK-DAG: ELFNPU37XX.Reloc offset(72) <R_VPU_64> %[[VAL49]] 0
//CHECK-DAG: ELFNPU37XX.Reloc offset(312) <R_VPU_64> %[[VAL50]] 0
//CHECK-DAG: ELFNPU37XX.Reloc offset(232) <R_VPU_64> %[[VAL54]] 0
//CHECK-DAG: ELFNPU37XX.Reloc offset(272) <R_VPU_64> %[[VAL55]] 0
