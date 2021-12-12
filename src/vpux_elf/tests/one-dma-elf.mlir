module @Test {

IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x1x1x1000xf16>
} outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x1x1x1000xf16>
}

func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {

    %dma0 = VPUIPRegMapped.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>) start_after(0) -> memref<1x1x1x1000xf16>
    %mappedInference = VPUIPRegMapped.MappedInference
                            dmas(%dma0 : memref<1x1x1x1000xf16>)
                            dmaCount(1)
                            invariantCount(0)
                            variantCount(0)
                            actInvocationsCount(0)

    %dmaSection = ELF.CreateSection secFlags(SHF_EXECINSTR) {secName=".text.dmaTasks", secType="SHT_PROGBITS", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutAnyOpInSection %dma0 : memref<1x1x1x1000xf16>
    }
    %mappedInfSec = ELF.CreateSection secFlags(SHF_EXECINSTR) {secName=".text.mappedInference", secType="SHT_PROGBITS", secInfo = 1, secAddrAlign = 64} -> !ELF.Section
    {
        ELF.PutAnyOpInSection %mappedInference : index
    }

    %sym_for_dmaSection = ELF.Symbol %dmaSection : !ELF.Section
    %sym_for_mappedInfSec = ELF.Symbol %mappedInfSec : !ELF.Section

    %symArg0 = ELF.Symbol %arg0 size(2000 ) : memref<1x1x1x1000xf16>
    %symArg1 = ELF.Symbol %arg1 size(2000 ) : memref<1x1x1x1000xf16>

    %genericSymSection = ELF.CreateSymbolTableSection secName(".symTab") secFlags(SHF_NONE) -> !ELF.Section
    {
        ELF.PutAnyOpInSection %sym_for_dmaSection : !ELF.Symbol
        ELF.PutAnyOpInSection %sym_for_mappedInfSec : !ELF.Symbol
        ELF.Symbol %mappedInference name("MappedInference") type("VPU_STT_ENTRY") : index
    }

    %inputSymSection = ELF.CreateSymbolTableSection secName(".symTab.inputs") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section
    {
        ELF.PutAnyOpInSection %symArg0 : !ELF.Symbol
    }
    %outputSymSection = ELF.CreateSymbolTableSection secName(".symTab.outputs") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section
    {
        ELF.PutAnyOpInSection %symArg1 : !ELF.Symbol
    }

    %mappedInferenceRelocs = ELF.CreateRelocationSection secName(".RelA.mappedInference") sourceSymbolTableSection(%genericSymSection) targetSection(%mappedInfSec) secFlags(SHF_NONE) -> !ELF.Section
    {
        ELF.Reloc 0 "R_VPU_64" %sym_for_dmaSection 0
    }

    %inputRelocs = ELF.CreateRelocationSection secName(".RelA.inputs") sourceSymbolTableSection(%inputSymSection) targetSection(%dmaSection) secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section
    {
        ELF.Reloc 16 "R_VPU_64" %symArg0 0
    }

    %outputRelocs = ELF.CreateRelocationSection secName(".RelA.outputs") sourceSymbolTableSection(%outputSymSection) targetSection(%dmaSection) secFlags("VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section
    {
        ELF.Reloc 24 "R_VPU_64" %symArg1 0
    }

    return %arg1 : memref<1x1x1x1000xf16>
}
}
