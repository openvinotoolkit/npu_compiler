//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% allow-custom-values=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --split-input-file --convert-VPUASM-to-NPUReg60XX --create-elf-relocations %s | FileCheck %s
// REQUIRES: platform-NPU6010

module @OneDMAWithoutAttributes {
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DMA_NN
  config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  config.MemoryResource 524288000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  VPUASM.InputBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  }
  VPUASM.OutputBindings outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  }
  VPUASM.ProfilingBindings profilingDeclarations : {
  }
  func.func @main() {
    ELF.Main {
      ELF.CreateLogicalSection @io.NetworkInput0 aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USERINPUT) secLocation(<NetworkInput>) {
        VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <801360> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @io.NetworkOutput0 aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT) secLocation(<NetworkOutput>) {
        VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @program.DMA.cmx.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DMA> {elfMemOffsetAttrKey = 0 : ui64}
      }
      ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_0_0) input(@io.NetworkInput0::@DeclareBuffer0) outputs([]) waits([]) updates([]) start_after(1) clean_after(0) dma_descriptor(<numPlanes = 0 : i32, len = 801360 : i32, srcWidth = 801360 : i32, srcStride = 801360 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)
      }
      ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DpuPVP @DpuPVP_0_0
      }
      ELF.CreateSection @note.MappedInferenceVersion aligned(4) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
        VPUASM.MappedInferenceVersion @MappedInferenceVersion_0_0(11 _ 4 _ 10)
      }
      ELF.CreateSection @program.nnrt_config aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
        VPUASM.nnrtConfig @MappedInference_nnrtConfigManaged :
      }
      ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC") secLocation(<DDR>) {
        VPUASM.ManagedMappedInference @MappedInference_managed : dmas ([[@task.dma.0.0::@NNDMA_0_0_0]]) workItems() barrierTasks() bootstrapBarriers()
         nnrtConfig(@program.nnrt_config::@MappedInference_nnrtConfigManaged)
         mappedInferenceVersion(@note.MappedInferenceVersion::@MappedInferenceVersion_0_0_0) dpuPvpTask(@dpu.pvp::@DpuPVP_0_0)
         {actshv_used = 0 : ui8, barrierConfigurationStride = 0 : i64, barrierConfigurationTasksCount = 0 : i64, barrierCount = 0 : i64,
          barriersReprogrammingCount = 0 : i64, bootstrapWorkItemsCount = 0 : i64, bootstrapBarriersCount = 0 : i64, disableDmaSwFifo = true,
          dmaCount = [[0, 0], [0, 0]], dma_from_cmx_used = 0 : ui8, dma_from_ddr_used = 1 : ui8, dpu_used = 0 : ui8, final_barrier_id = 0 : i64,
          workItemsCount = 1 : i64,
          workloadManagementBarrierProgrammingMode = #VPURegMapped.workload_management_barrier_programming_mode<ALL_BARRIER_DMAS_SCHEDULED>}
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.DMA.cmx.0.0 of(@program.DMA.cmx.0.0) type(<STT_SECTION>)
        ELF.Symbol @elfsym.task.dma.0.0 of(@task.dma.0.0) type(<STT_SECTION>)
        ELF.Symbol @elfsym.dpu.pvp of(@dpu.pvp) type(<STT_SECTION>)
        ELF.Symbol @elfsym.program.nnrt_config of(@program.nnrt_config) type(<STT_SECTION>)
        ELF.Symbol @elfsym.program.mapped_inference of(@program.mapped_inference) type(<STT_SECTION>)
        ELF.Symbol @entry of(@program.mapped_inference::@MappedInference_managed) type(<VPU_STT_ENTRY>)
      }
      ELF.CreateSymbolTableSection @symtab.io.NetworkInput secFlags("VPU_SHF_USERINPUT|VPU_SHF_JIT") {
        ELF.Symbol @elfsym.io.NetworkInput0 of(@io.NetworkInput0) type(<STT_SECTION>) size(48)
      }
      ELF.CreateSymbolTableSection @symtab.io.NetworkOutput secFlags("VPU_SHF_USEROUTPUT|VPU_SHF_JIT") {
        ELF.Symbol @elfsym.io.NetworkOutput0 of(@io.NetworkOutput0) type(<STT_SECTION>) size(48)
      }
      ELF.CreateMetadataSection @MetadataSection aligned(8) secFlags("SHF_NONE") {
        VPUASM.NetworkMetadata @NetworkMetadata
      }
      // CHECK:        ELF.CreateRelocationSection @rela.task.dma.0.0.symtab.io.NetworkInput
      // CHECK-SAME:    target(@task.dma.0.0)
      // CHECK-SAME:    symtab(@symtab.io.NetworkInput)
      // CHECK-SAME:    secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT")
      // CHECK:       ELF.Reloc
      // CHECK-SAME:    offset({{[0-9]+}})
      // CHECK-SAME:    sourceSym(@symtab.io.NetworkInput::@elfsym.io.NetworkInput0)
      // CHECK-SAME:    relocType(<R_VPU_64>)
      // CHECK-SAME:    addend({{[0-9]+}})

      // CHECK-NOT:       ELF.CreateRelocationSection @rela.task.dma.0.0.symtab.io.NetworkOutput

      // CHECK:       ELF.CreateRelocationSection @rela.program.mapped_inference.symtab
      // CHECK-SAME:    target(@program.mapped_inference)
      // CHECK-SAME:    symtab(@symtab)
      // CHECK-SAME:    secFlags("SHF_NONE")
      // CHECK:       ELF.Reloc
      // CHECK-SAME:    offset({{[0-9]+}})
      // CHECK-SAME:    sourceSym(@symtab::@elfsym.program.nnrt_config)
      // CHECK-SAME:    relocType(<R_VPU_64>)
      // CHECK-SAME:    addend({{[0-9]+}})
    }
    return
  }
}
