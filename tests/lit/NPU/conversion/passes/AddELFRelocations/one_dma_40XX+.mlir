//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --convert-VPUASM-to-NPUReg40XX --create-elf-relocations %s | FileCheck %s
// REQUIRES: arch-NPU40XX
module @OneDMAWithoutAttributes attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.TileResource 6 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @SHAVE_ACT
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
  IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @io.NetworkInput0 aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USERINPUT) {
        VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @io.NetworkOutput0 aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT) {
        VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @program.DMA.cmx.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DMA> {elfMemOffsetAttrKey = 0 : ui64}
      }
      ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.DMA.cmx.0.0::@DeclareTaskBuffer_DMA_0_0_0) input(@io.NetworkInput0::@DeclareBuffer0) outputs([@io.NetworkOutput0::@DeclareBuffer1]) waits([]) updates([]) start_after(0) clean_after(0) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>) {elfMemOffsetAttrKey = 0 : ui64}
      }

      ELF.CreateSection @note.MappedInferenceVersion aligned(4) secType(SHT_NOTE) secFlags("SHF_NONE") {
        VPUASM.MappedInferenceVersion @MappedInferenceVersion_0_0(11 _ 4 _ 10)
      }

      ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.MappedInference {elfMemOffsetAttrKey = 0 : ui64} @MappedInference : dmas([[@task.dma.0.0::@NNDMA_0_0_0]]) dmaCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(0) mappedInferenceVersion(@note.MappedInferenceVersion::@MappedInferenceVersion_0_0)
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.DMA.cmx.0.0 of(@program.DMA.cmx.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.task.dma.0.0 of(@task.dma.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.mapped_inference of(@program.mapped_inference) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @entry of(@program.mapped_inference::@MappedInference) type(<VPU_STT_ENTRY>) size(0) value(0)
      }
      ELF.CreateSymbolTableSection @symtab.io.NetworkInput secFlags("VPU_SHF_USERINPUT|VPU_SHF_JIT") {
        ELF.Symbol @elfsym.io.NetworkInput0 of(@io.NetworkInput0) type(<STT_SECTION>) size(48) value(0)
      }
      ELF.CreateSymbolTableSection @symtab.io.NetworkOutput secFlags("VPU_SHF_USEROUTPUT|VPU_SHF_JIT") {
        ELF.Symbol @elfsym.io.NetworkOutput0 of(@io.NetworkOutput0) type(<STT_SECTION>) size(48) value(0)
      }
      ELF.CreateMetadataSection @MetadataSection aligned(8) secFlags("SHF_NONE") {
        VPUASM.NetworkMetadata @NetworkMetadata
      }
      // CHECK:       ELF.CreateRelocationSection @rela.task.dma.0.0.symtab.io.NetworkInput
      // CHECK-SAME:    target(@task.dma.0.0)
      // CHECK-SAME:    symtab(@symtab.io.NetworkInput)
      // CHECK-SAME:    secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT")
      // CHECK:       ELF.Reloc
      // CHECK-SAME:    offset({{[0-9]+}})
      // CHECK-SAME:    sourceSym(@symtab.io.NetworkInput::@elfsym.io.NetworkInput0)
      // CHECK-SAME:    relocType(<R_VPU_64>)
      // CHECK-SAME:    addend({{[0-9]+}})

      // CHECK:       ELF.CreateRelocationSection @rela.task.dma.0.0.symtab.io.NetworkOutput
      // CHECK-SAME:    target(@task.dma.0.0)
      // CHECK-SAME:    symtab(@symtab.io.NetworkOutput)
      // CHECK-SAME:    secFlags("VPU_SHF_JIT|VPU_SHF_USEROUTPUT")
      // CHECK:       ELF.Reloc
      // CHECK-SAME:    offset({{[0-9]+}})
      // CHECK-SAME:    sourceSym(@symtab.io.NetworkOutput::@elfsym.io.NetworkOutput0)
      // CHECK-SAME:    relocType(<R_VPU_64>)
      // CHECK-SAME:    addend({{[0-9]+}})

      // CHECK:       ELF.CreateRelocationSection @rela.program.mapped_inference.symtab
      // CHECK-SAME:    target(@program.mapped_inference)
      // CHECK-SAME:    symtab(@symtab)
      // CHECK-SAME:    secFlags("SHF_NONE")
      // CHECK:       ELF.Reloc
      // CHECK-SAME:    offset({{[0-9]+}})
      // CHECK-SAME:    sourceSym(@symtab::@elfsym.task.dma.0.0)
      // CHECK-SAME:    relocType(<R_VPU_64>)
      // CHECK-SAME:    addend({{[0-9]+}})

    }
    return
  }
}
