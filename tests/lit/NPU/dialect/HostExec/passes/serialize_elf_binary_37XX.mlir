//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --serialize-elf-to-binary %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK-LABEL: @OneInputOneOutput
module @OneInputOneOutput attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode = #config.compilation_mode<HostCompile>, config.revisionID = #config.revision_id<REVISION_NONE>} {
  config.PipelineOptions @Options {
    config.Option @VPU.FP16CompressedConv : false
    config.Option @VPU.ReduceSupported : false
    config.Option @VPU.AutoPaddingODU : false
    config.Option @VPU.AutoPaddingIDU : false
    config.Option @VPU.SprLUTEnabled : false
    config.Option @VPU.BarrierMaxVariantSum : 256
    config.Option @VPU.BarrierMaxVariantCount : 256
    config.Option @VPU.MaxKernelSize : 11
  }
  config.Resources 2 of @NCE at 1.300000e+03 MHz {
    config.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @SHAVE_NN
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 2 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 8 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x60x60xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x60x60xf16>
  }
  module @module0 attributes {config.arch = #config.arch_kind<NPU37XX>} {
    net.NetworkInfo {inferenceTiming = 2282 : i64} entryPoint : @dma_copy inputsInfo : {
      DataInfo "input" : tensor<1x3x60x60xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<1x3x60x60xf16>
    }
    func.func nested @dma_copy(%arg0: memref<1x3x60x60xf16, @DDR>, %arg1: memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR> {
      %0 = VPUMI37XX.ConfigureBarrier {consumer_count = 0 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
      %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x60x60xf16, @DDR>
      %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x60x60xf16, @DDR>
      %3 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%1 : memref<1x3x60x60xf16, @DDR>) outputs(%2 : memref<1x3x60x60xf16, @DDR>) updates(%0 : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
      %4 = VPUMI37XX.MappedInference dmas(%3 : !VPURegMapped.Index<0:0:0>) barriers(%0 : !VPURegMapped.Index<0:0:0>) dmaCount([1, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(1) -> !VPURegMapped.Index<0:0:0>
      %5 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELFNPU37XX.Section {
        ELFNPU37XX.PutOpInSection %3 : !VPURegMapped.Index<0:0:0>
      }
      %6 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELFNPU37XX.Section {
        ELFNPU37XX.PutOpInSection %0 : !VPURegMapped.Index<0:0:0>
      }
      %7 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELFNPU37XX.Section {
        ELFNPU37XX.PutOpInSection %4 : !VPURegMapped.Index<0:0:0>
      }
      %8 = ELFNPU37XX.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 8 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELFNPU37XX.Section {
        %35 = VPUMI37XX.NetworkMetadata -> !VPURegMapped.Index<0:0:0>
      }
      %9 = ELFNPU37XX.CreateSection secType(VPU_SHT_PERF_METRICS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".perf.metrics"} -> !ELFNPU37XX.Section {
        %35 = VPUMI37XX.PerformanceMetrics -> !VPURegMapped.Index<0:0:0>
      }
      %10 = ELFNPU37XX.CreateSection secType(SHT_NOTE) secFlags("SHF_NONE") {secAddrAlign = 4 : i64, secInfo = 0 : i64, secName = ".note.LoaderABIVersion"} -> !ELFNPU37XX.Section {
        ELFNPU37XX.ABIVersion(1 _ 2 _ 0)
      }
      %11 = ELFNPU37XX.CreateSection secType(SHT_NOTE) secFlags("SHF_NONE") {secAddrAlign = 4 : i64, secInfo = 0 : i64, secName = ".note.MappedInferenceVersion"} -> !ELFNPU37XX.Section {
        VPUMI37XX.MappedInferenceVersion(7 _ 0 _ 4)
      }
      %12 = ELFNPU37XX.CreateSection secType(VPU_SHT_PLATFORM_INFO) secFlags("SHF_NONE") {secAddrAlign = 8 : i64, secInfo = 0 : i64, secName = ".meta.PlatformInfo"} -> !ELFNPU37XX.Section {
        VPUMI37XX.PlatformInfo {archKind = #config.arch_kind<NPU37XX>}
      }
      %13 = ELFNPU37XX.Symbol %5 name("sym_dmaSection0") : !ELFNPU37XX.Section
      %14 = ELFNPU37XX.Symbol %6 name("sym_barrierSection") : !ELFNPU37XX.Section
      %15 = ELFNPU37XX.Symbol %arg0 name("input") size(21600) : memref<1x3x60x60xf16, @DDR>
      %16 = ELFNPU37XX.Symbol %arg1 name("output") size(21600) : memref<1x3x60x60xf16, @DDR>
      %17 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELFNPU37XX.Section {
        ELFNPU37XX.PutOpInSection %15 : !ELFNPU37XX.Symbol
      }
      %18 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section {
        ELFNPU37XX.PutOpInSection %16 : !ELFNPU37XX.Symbol
      }
      %c0_i8 = arith.constant 0 : i8
      %19 = ELFNPU37XX.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
      %c1_i8 = arith.constant 1 : i8
      %20 = ELFNPU37XX.Symbol %c1_i8 name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
      %c2_i8 = arith.constant 2 : i8
      %21 = ELFNPU37XX.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
      %c3_i8 = arith.constant 3 : i8
      %22 = ELFNPU37XX.Symbol %c3_i8 name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
      %c4_i8 = arith.constant 4 : i8
      %23 = ELFNPU37XX.Symbol %c4_i8 name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
      %c5_i8 = arith.constant 5 : i8
      %24 = ELFNPU37XX.Symbol %c5_i8 name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
      %c6_i8 = arith.constant 6 : i8
      %25 = ELFNPU37XX.Symbol %c6_i8 name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
      %c7_i8 = arith.constant 7 : i8
      %26 = ELFNPU37XX.Symbol %c7_i8 name("VPU_NNRD_SYM_HW_REGISTER") {isBuiltin} : i8
      %27 = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELFNPU37XX.Section {
        ELFNPU37XX.PutOpInSection %19 : !ELFNPU37XX.Symbol
        ELFNPU37XX.PutOpInSection %20 : !ELFNPU37XX.Symbol
        ELFNPU37XX.PutOpInSection %21 : !ELFNPU37XX.Symbol
        ELFNPU37XX.PutOpInSection %22 : !ELFNPU37XX.Symbol
        ELFNPU37XX.PutOpInSection %23 : !ELFNPU37XX.Symbol
        ELFNPU37XX.PutOpInSection %24 : !ELFNPU37XX.Symbol
        ELFNPU37XX.PutOpInSection %25 : !ELFNPU37XX.Symbol
        ELFNPU37XX.PutOpInSection %26 : !ELFNPU37XX.Symbol
      }
      %28 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
        ELFNPU37XX.PutOpInSection %13 : !ELFNPU37XX.Symbol
        ELFNPU37XX.PutOpInSection %14 : !ELFNPU37XX.Symbol
        %35 = ELFNPU37XX.Symbol %4 name("MappedInference_entry") type(<VPU_STT_ENTRY>) : !VPURegMapped.Index<0:0:0>
      }
      %29 = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetInput0") sourceSymbolTableSection(%17) targetSection(%5) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELFNPU37XX.Section {
        ELFNPU37XX.Reloc baseOp(%3 : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_64> %15 0 {description = "VPUMI37XX.NNDMA input from network input dma reloc"}
      }
      %30 = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput0") sourceSymbolTableSection(%18) targetSection(%5) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELFNPU37XX.Section {
        ELFNPU37XX.Reloc baseOp(%3 : !VPURegMapped.Index<0:0:0>) offset(24) <R_VPU_64> %16 0 {description = "VPUMI37XX.NNDMA output from network output dma reloc"}
      }
      %31 = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection(%27) targetSection(%5) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
        ELFNPU37XX.Reloc baseOp(%3 : !VPURegMapped.Index<0:0:0>) offset(64) <R_VPU_64_LSHIFT> %25 0 {description = ""}
        ELFNPU37XX.Reloc baseOp(%3 : !VPURegMapped.Index<0:0:0>) offset(72) <R_VPU_64_LSHIFT> %25 0 {description = ""}
      }
      %32 = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%28) targetSection(%7) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
        ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(72) <R_VPU_64> %13 0 {description = ""}
        ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(112) <R_VPU_64> %13 0 {description = ""}
        ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(312) <R_VPU_64> %14 0 {description = "barrierTasks in mapped inference reloc"}
      }
      %33 = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.BarrierConfigs") sourceSymbolTableSection(%27) targetSection(%6) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
        ELFNPU37XX.Reloc baseOp(%0 : !VPURegMapped.Index<0:0:0>) offset(8) <R_VPU_16_SUM> %25 0 {description = ""}
      }
      %34 = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%27) targetSection(%7) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
        ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(72) <R_VPU_64_MULT_SUB> %24 1 {description = ""}
        ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(80) <R_VPU_64_MULT_SUB> %24 1 {description = ""}
        ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(112) <R_VPU_64_MULT> %24 0 {description = ""}
        ELFNPU37XX.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(120) <R_VPU_64_MULT> %24 0 {description = ""}
      }
      return %arg1 : memref<1x3x60x60xf16, @DDR>
    }
    config.Resources {activity_factor = 0.000000e+00 : f64} 2 of @NCE at 1.300000e+03 MHz {
      config.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
      config.ExecutorResource 2 of @SHAVE_ACT
      config.ExecutorResource 1 of @SHAVE_NN
      config.ExecutorResource 1 of @DPU
    }
    config.PipelineOptions @Options {
      config.Option @VPU.FP16CompressedConv : false
      config.Option @VPU.ReduceSupported : false
      config.Option @VPU.AutoPaddingODU : false
      config.Option @VPU.AutoPaddingIDU : false
      config.Option @VPU.SprLUTEnabled : false
      config.Option @VPU.BarrierMaxVariantSum : 256
      config.Option @VPU.BarrierMaxVariantCount : 256
      config.Option @VPU.MaxKernelSize : 11
    }
    config.ExecutorResource 2 of @DMA_NN
    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 8 : i64, config.derateFactor = 6.000000e-01 : f64}
  }
  func.func @main(%arg0: memref<1x3x60x60xf16, @DDR>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
    %alloc = memref.alloc() : memref<1x3x60x60xf16, @DDR>
    Core.NestedCall @module0::@dma_copy(%arg0, %alloc) : (memref<1x3x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR>
    memref.copy %alloc, %arg1 : memref<1x3x60x60xf16, @DDR> to memref<1x3x60x60xf16>
    return %arg1 : memref<1x3x60x60xf16>
  }

  // CHECK:   HostExec.Binary @module0 {
  // CHECK:   HostExec.BinaryData @serialized_dma_copy
  // CHECK-SAME:   <object = "\7FELF\02\01\00\00\00\{{.*}}">
  // CHECK:   func.func private @dma_copy(memref<1x3x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR>
  // CHECK:   }
  // CHECK:   func.func @main(%arg0: memref<1x3x60x60xf16, @DDR>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
  // CHECK:   %alloc = memref.alloc() : memref<1x3x60x60xf16, @DDR>
  // CHECK:   Core.NestedCall @module0::@dma_copy(%arg0, %alloc) : (memref<1x3x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR>
}
