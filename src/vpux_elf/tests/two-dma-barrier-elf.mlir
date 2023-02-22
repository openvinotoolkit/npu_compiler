//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
module @Test  attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"}  {
  IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.MemoryResource 2097152 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  IE.ExecutorResource 1 of @DMA_NN
  IE.ExecutorResource 1 of @SHAVE_UPA
  IE.ExecutorResource 1 of @SHAVE_ACT
  IE.ExecutorResource 1 of @NCE  {
    IE.ExecutorResource 1 of @DPU
  }

IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x1x2x1000xf16>
} outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x1x2x1000xf16>
}

func @main(%arg0: memref<1x1x2x1000xf16>, %arg1: memref<1x1x2x1000xf16>) -> memref<1x1x2x1000xf16> {

    %buffer = VPURT.DeclareBuffer "DDR" <0> -> memref<1x1x2x1000xf16, @DDR>
    %barrier0 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <0,-1> -> !VPUIPRegMapped.Index<0>

    %dma0 = VPUIPRegMapped.NNDMA inputs(%arg0 : memref<1x1x2x1000xf16>)
                                    outputs(%buffer : memref<1x1x2x1000xf16, @DDR>)
                                    updates(%barrier0 : !VPUIPRegMapped.Index<0>)
                                    start_after(0) -> !VPUIPRegMapped.Index<0>
    %dma1 = VPUIPRegMapped.NNDMA inputs(%buffer : memref<1x1x2x1000xf16, @DDR>)
                                    outputs(%arg1 : memref<1x1x2x1000xf16>)
                                    previousDMA(%dma0 : !VPUIPRegMapped.Index<0>)
                                    waits(%barrier0 :!VPUIPRegMapped.Index<0>)
                                    start_after(0) -> !VPUIPRegMapped.Index<1>

    %mappedInference = VPUIPRegMapped.MappedInference
                            dmas(%dma0 : !VPUIPRegMapped.Index<0>) // we can verify with the MappedInference Verifier: if we have dmaCount(>1) we need to have dmaTasks() (reverse applies also), else error; the same for the other ...
                            barriers(%barrier0 : !VPUIPRegMapped.Index<0>)
                            dmaCount(2)
                            invariantCount(0)
                            variantCount(0)
                            actKernelRangesCount(0)
                            actKernelInvocationsCount(0)
                            barrierCount(1)
                            -> !VPUIPRegMapped.Index<0>

    %dmaSection = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secName=".text.dmaTasks", secInfo = 0, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %dma0 : !VPUIPRegMapped.Index<0>
        ELF.PutOpInSection %dma1 : !VPUIPRegMapped.Index<1>
    }

    %barriersSection = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secName = ".text.barriers", secInfo = 0, secAddrAlign = 64} -> !ELF.Section
    {
        ELF.PutOpInSection %barrier0 : !VPUIPRegMapped.Index<0>
    }

    %mappedInfSec = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secName=".text.mappedInference", secInfo = 0, secAddrAlign = 64} -> !ELF.Section
    {
        ELF.PutOpInSection %mappedInference : !VPUIPRegMapped.Index<0>
    }

    %metadataSec = ELF.CreateMetadataSection secFlags(SHF_EXECINSTR) {secName=".text.networkMetadata", secInfo = 0, secAddrAlign = 64} -> !ELF.Section
    {
        %metadata = VPUIPRegMapped.NetworkMetadata -> !VPUIPRegMapped.Index<0>
    }

    %scratchSection = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(SHF_NONE) {secName=".bss.ddrScratch", secInfo = 0, secAddrAlign = 64} -> !ELF.Section
    {
        ELF.PutOpInSection %buffer : memref<1x1x2x1000xf16, @DDR>
    }

    %sym_for_dmaSection = ELF.Symbol %dmaSection name("symDmaSection") : !ELF.Section
    %sym_for_barrierSection = ELF.Symbol %barriersSection name("symBarriersSection") : !ELF.Section
    %sym_for_mappedInfSec = ELF.Symbol %mappedInfSec name("symMappedInfSec") : !ELF.Section
    %sym_for_scratchSection = ELF.Symbol %scratchSection name("symScratchSection") : !ELF.Section

    %symArg0 = ELF.Symbol %arg0 name("inputCNN") size(4000) : memref<1x1x2x1000xf16>
    %symArg1 = ELF.Symbol %arg1 name("outputCNN") size(4000) : memref<1x1x2x1000xf16>

    %const_0 = arith.constant 0 : i8
    %const_1 = arith.constant 1 : i8
    %const_2 = arith.constant 2 : i8
    %const_3 = arith.constant 3 : i8
    %const_4 = arith.constant 4 : i8
    %const_5 = arith.constant 5 : i8
    %const_6 = arith.constant 6 : i8

    %sym_0 = ELF.Symbol %const_0 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") { isBuiltin } : i8
    %sym_1 = ELF.Symbol %const_1 name("VPU_NNRD_SYM_RTM_IVAR") { isBuiltin } : i8
    %sym_2 = ELF.Symbol %const_2 name("VPU_NNRD_SYM_RTM_ACT") { isBuiltin } : i8
    %sym_3 = ELF.Symbol %const_3 name("VPU_NNRD_SYM_RTM_DMA0") { isBuiltin } : i8
    %sym_4 = ELF.Symbol %const_4 name("VPU_NNRD_SYM_RTM_DMA1") { isBuiltin } : i8
    %sym_5 = ELF.Symbol %const_5 name("VPU_NNRD_SYM_FIFO_BASE") { isBuiltin } : i8
    %sym_6 = ELF.Symbol %const_6 name("VPU_NNRD_SYM_BARRIERS_START") { isBuiltin } : i8


    %vpu_symtab = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags(SHF_NONE) { isBuiltin } -> !ELF.Section
    {
    ELF.PutOpInSection %sym_0 : !ELF.Symbol
    ELF.PutOpInSection %sym_1 : !ELF.Symbol
    ELF.PutOpInSection %sym_2 : !ELF.Symbol
    ELF.PutOpInSection %sym_3 : !ELF.Symbol
    ELF.PutOpInSection %sym_4 : !ELF.Symbol
    ELF.PutOpInSection %sym_5 : !ELF.Symbol
    ELF.PutOpInSection %sym_6 : !ELF.Symbol
    }

    %genericSymSection = ELF.CreateSymbolTableSection secName(".symtab") secFlags(SHF_NONE) -> !ELF.Section
    {
        ELF.PutOpInSection %sym_for_dmaSection : !ELF.Symbol
        ELF.PutOpInSection %sym_for_barrierSection : !ELF.Symbol
        ELF.PutOpInSection %sym_for_mappedInfSec : !ELF.Symbol
        ELF.PutOpInSection %sym_for_scratchSection : !ELF.Symbol

        ELF.Symbol %mappedInference name("MappedInference") type("VPU_STT_ENTRY") : !VPUIPRegMapped.Index<0>
    }

    %inputSymSection = ELF.CreateSymbolTableSection secName(".symtab.inputs") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section
    {
        ELF.PutOpInSection %symArg0 : !ELF.Symbol
    }
    %outputSymSection = ELF.CreateSymbolTableSection secName(".symtab.outputs") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section
    {
        ELF.PutOpInSection %symArg1 : !ELF.Symbol
    }

    %mappedInferenceRelocs = ELF.CreateRelocationSection secName(".rlt.mappedInference") sourceSymbolTableSection(%genericSymSection) targetSection(%mappedInfSec) secFlags(SHF_INFO_LINK) -> !ELF.Section
    {
        ELF.Reloc baseOp(%mappedInference : !VPUIPRegMapped.Index<0>) offsetOf(%dma0 : !VPUIPRegMapped.Index<0>) "R_VPU_64" %sym_for_dmaSection 0
        ELF.Reloc baseOp(%mappedInference : !VPUIPRegMapped.Index<0>) offsetOf(%barrier0 : !VPUIPRegMapped.Index<0>) "R_VPU_64" %sym_for_barrierSection 0
    }

    %dmaRelocs = ELF.CreateRelocationSection secName(".rlt.dmaRelocations") sourceSymbolTableSection(%genericSymSection) targetSection(%dmaSection) secFlags(SHF_INFO_LINK) -> !ELF.Section
    {
        ELF.Reloc baseOp(%dma0 : !VPUIPRegMapped.Index<0>) offsetOf(%buffer : memref<1x1x2x1000xf16, @DDR>) "R_VPU_64" %sym_for_scratchSection 0
        ELF.Reloc baseOp(%dma1 : !VPUIPRegMapped.Index<1>) offsetOf(%buffer : memref<1x1x2x1000xf16, @DDR>) "R_VPU_64" %sym_for_scratchSection 0
    }

    %dmaSpecialRelocs = ELF.CreateRelocationSection secName(".rlt.dmaSpecialRelocations") sourceSymbolTableSection(%vpu_symtab) targetSection(%dmaSection) secFlags(SHF_INFO_LINK) -> !ELF.Section
    {
        ELF.RelocImmOffset baseOp(%dma0 : !VPUIPRegMapped.Index<0>) offset(0) "R_VPU_32_RTM" %sym_3 128
    }

    %inputRelocs = ELF.CreateRelocationSection secName(".rlt.inputs") sourceSymbolTableSection(%inputSymSection) targetSection(%dmaSection) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section
    {
        ELF.Reloc baseOp(%dma0 : !VPUIPRegMapped.Index<0>) offsetOf(%arg0 : memref<1x1x2x1000xf16>) "R_VPU_64" %symArg0 0
    }

    %outputRelocs = ELF.CreateRelocationSection secName(".rlt.outputs") sourceSymbolTableSection(%outputSymSection) targetSection(%dmaSection) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section
    {
        ELF.Reloc baseOp(%dma1 : !VPUIPRegMapped.Index<1>) offsetOf(%arg1 : memref<1x1x2x1000xf16>) "R_VPU_64" %symArg1 0
    }

    return %arg1 : memref<1x1x2x1000xf16>
}
}
